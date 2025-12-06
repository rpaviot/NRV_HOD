"""
Null test for fast galaxy-galaxy lensing pre-computation.

This test performs the pre-computation step for ΔΣ using the precompute_deltasigma
module. It demonstrates the workflow for creating a lensing grid that can be used
for fast interpolation during HOD parameter fitting.

Workflow:
1. Load halo and particle catalogs
2. Downsample particle catalog by factor 20
3. Pre-select particles within 3×Rvir of ANY halo center (reduces data volume)
4. For each selected particle, compute ΔΣ using ξ_gm from ALL particles
5. Save pre-computed ΔΣ grid to disk

Key parameters:
- 3×Rvir: Criterion for pre-selecting particles near halos (saves disk space)
- search_radius ~100 Mpc/h: Maximum scale for ξ_gm computation (BAO scale)

Note: The search_radius for ξ_gm computation is independent of the 3×Rvir
particle selection criterion. We compute ξ_gm up to BAO scales even though
we only save ΔΣ for particles near halos.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_spherical,
    save_precomputed_lensing
)
from test_utils import print_header, print_timing


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("Fast ΔΣ Pre-computation Test Configuration")

    # Paths (same as test_performance.py)
    halo_path = '/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet'
    particle_path = '/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet'

    # Cosmology (Flamingo L1000N1800)
    cosmo_params = {
        'H0': 67.74,
        'Om0': 0.3089,
        'Ob0': 0.0486,
        'sigma8': 0.8159,
        'ns': 0.9667
    }

    # Box parameters
    Lbox = 681  # Mpc/h

    # Mean matter density [Msun/h / (Mpc/h)^3]
    # RHO_M = Omega_m * rho_crit
    # rho_crit = 2.77536627e11 h^2 Msun / Mpc^3
    rho_crit = 2.77536627e11  # h^2 Msun / Mpc^3
    RHO_M = cosmo_params['Om0'] * rho_crit  # Msun/h / (Mpc/h)^3

    # Radial bins for ΔΣ computation
    # Covering typical scales from 0.1 to ~30 Mpc/h
    rp_bins = np.logspace(-1, 1.5, 16)  # 0.1 to ~31.6 Mpc/h

    # Particle selection criterion: only keep particles within 3×Rvir of any halo
    # This reduces the amount of data we need to save to disk
    r_factor_selection = 3.0  # Select particles within 3×Rvir

    # Search radius for spherical profile computation
    # Should cover maximum rp_bin with some margin
    search_radius_spherical = 100  # Mpc/h (~47 Mpc/h)

    # Downsampling factor for particles (reduced for better statistics)
    downsample_factor = 40  # Was 40, but spherical method needs more particles

    # Computation method
    method = 'spherical'  # 'spherical' (recommended) or 'cylindrical' (legacy)

    # Output path for pre-computed lensing grid
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'precomputed_lensing_grid.h5'

    print(f"\n  Halo catalog: {Path(halo_path).name}")
    print(f"  Particle catalog: {Path(particle_path).name}")
    print(f"  Particle downsampling: {downsample_factor}x")
    print(f"  Box size: {Lbox} Mpc/h")
    print(f"  RHO_M: {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"\n  ΔΣ computation parameters:")
    print(f"    Method: {method}")
    print(f"    rp bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"    Particle selection: within {r_factor_selection:.1f}×Rvir of halos")
    print(f"    Search radius: {search_radius_spherical:.1f} Mpc/h")
    print(f"\n  Output path: {output_path}")

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print_header("Step 1: Load Halo and Particle Catalogs")

    start = time.time()
    df_halos = pd.read_parquet(halo_path)
    elapsed_halos = time.time() - start
    print_timing("Load halo catalog", elapsed_halos)
    print(f"    → Loaded {len(df_halos):,} halos")

    start = time.time()
    df_particles = pd.read_parquet(particle_path)
    elapsed_particles = time.time() - start
    print_timing("Load particle catalog", elapsed_particles)
    print(f"    → Loaded {len(df_particles):,} particles (already downsampled)")

    # Additional downsampling by factor 20
    n_particles_original = len(df_particles)
    df_particles = df_particles.iloc[::downsample_factor].reset_index(drop=True)
    n_particles_downsampled = len(df_particles)

    print(f"    → Further downsampled to {n_particles_downsampled:,} particles")
    print(f"    → Total downsampling: {n_particles_original/n_particles_downsampled:.1f}x")

    # ========================================================================
    # Step 2: Extract Positions
    # ========================================================================
    print_header("Step 2: Extract Positions from Catalogs")

    # Extract halo positions and virial radii
    halo_positions = df_halos[['x', 'y', 'z']].values
    halo_rvir = df_halos['rvir'].values/1000

    # Extract particle positions
    particle_positions = df_particles[['x', 'y', 'z']].values

    print(f"  Halo positions: {halo_positions.shape}")
    print(f"  Halo Rvir range: [{halo_rvir.min():.3f}, {halo_rvir.max():.3f}] Mpc/h")
    print(f"  Particle positions: {particle_positions.shape}")

    # ========================================================================
    # Step 3: Pre-select Particles Near Halos (within 3×Rvir)
    # ========================================================================
    print_header("Step 3: Pre-select Particles Near Halos")

    print(f"\n  Building KD-tree for particle positions...")
    start = time.time()
    kdtree_particles = build_particle_kdtree(particle_positions, Lbox)
    elapsed_kdtree = time.time() - start
    print_timing("Build particle KD-tree", elapsed_kdtree)

    print(f"\n  Selecting particles within {r_factor_selection:.1f}×Rvir of any halo...")
    start = time.time()

    # Find all particles within 3×Rvir of any halo center
    selected_particle_indices = set()

    for i, (halo_pos, rvir) in enumerate(zip(halo_positions, halo_rvir)):
        if (i + 1) % max(1, len(halo_positions) // 10) == 0:
            print(f"    Progress: {i+1}/{len(halo_positions)} ({100*(i+1)/len(halo_positions):.1f}%)")

        # Query particles within r_factor_selection × Rvir
        search_radius = r_factor_selection * rvir
        nearby_indices = kdtree_particles.query_ball_point(halo_pos, r=search_radius)
        selected_particle_indices.update(nearby_indices)

    # Convert to sorted array
    selected_particle_indices = np.array(sorted(selected_particle_indices))
    selected_particle_positions = particle_positions[selected_particle_indices]

    elapsed_selection = time.time() - start
    print_timing("\nParticle selection time", elapsed_selection)

    print(f"\n  Selection results:")
    print(f"    Total particles (downsampled): {len(particle_positions):,}")
    print(f"    Selected particles: {len(selected_particle_indices):,}")
    print(f"    Selection fraction: {len(selected_particle_indices)/len(particle_positions)*100:.1f}%")
    print(f"    Memory saved: {(1 - len(selected_particle_indices)/len(particle_positions))*100:.1f}%")

    # ========================================================================
    # Step 4: Pre-compute ΔΣ at Selected Particle Positions
    # ========================================================================
    print_header("Step 4: Pre-compute ΔΣ at Selected Particle Positions")

    print(f"\n  Computing ΔΣ for {len(selected_particle_indices):,} selected particles...")
    print(f"  Using {method} method with search radius {search_radius_spherical:.1f} Mpc/h")
    print(f"  This may take a while...\n")

    start = time.time()

    all_deltasigma = []
    n_success = 0
    n_failed = 0

    for i, particle_pos in enumerate(selected_particle_positions):
        if (i + 1) % max(1, len(selected_particle_positions) // 10) == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(selected_particle_positions) - i - 1) / rate
            print(f"    Progress: {i+1}/{len(selected_particle_positions)} "
                  f"({100*(i+1)/len(selected_particle_positions):.1f}%) | "
                  f"Rate: {rate:.1f} part/s | ETA: {remaining/60:.1f} min")

        # Query particles within search radius
        nearby_indices = kdtree_particles.query_ball_point(particle_pos, r=search_radius_spherical)

        if len(nearby_indices) < 50:  # Need more particles for spherical method
            n_failed += 1
            continue

        nearby_particles = particle_positions[nearby_indices]

        try:
            delta_sigma = compute_deltasigma_spherical(
                position=particle_pos,
                nearby_particles=nearby_particles,
                RHO_M=RHO_M,
                rp_bins=rp_bins,
                n_particles_total=len(particle_positions),
                Lbox=Lbox
            )

            all_deltasigma.append(delta_sigma)
            n_success += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:  # Only print first few errors
                print(f"    Warning: Failed at position {particle_pos}: {e}")

    elapsed_precompute = time.time() - start
    print_timing("\nTotal pre-computation time", elapsed_precompute)

    # Convert to arrays
    positions = selected_particle_positions[:n_success]
    deltasigma_values = np.array(all_deltasigma)

    print(f"\n  Computation results:")
    print(f"    Successful: {n_success:,}")
    print(f"    Failed: {n_failed:,}")
    print(f"    Success rate: {n_success/(n_success+n_failed)*100:.1f}%")
    print(f"    ΔΣ shape: {deltasigma_values.shape}")
    print(f"    ΔΣ range: [{deltasigma_values.min():.2e}, {deltasigma_values.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 5: Save Pre-computed Lensing Grid
    # ========================================================================
    print_header("Step 5: Save Pre-computed Lensing Grid")

    metadata = {
        'cosmology': 'Flamingo',
        'H0': cosmo_params['H0'],
        'Om0': cosmo_params['Om0'],
        'Ob0': cosmo_params['Ob0'],
        'sigma8': cosmo_params['sigma8'],
        'ns': cosmo_params['ns'],
        'redshift': 1.0,  # Assuming z=1
        'RHO_M': RHO_M,
        'Lbox': Lbox,
        'n_halos': len(df_halos),
        'n_particles_total': len(particle_positions),
        'n_particles_selected': len(selected_particle_indices),
        'downsample_factor': downsample_factor,
        'r_factor_selection': r_factor_selection,
        'search_radius_spherical': search_radius_spherical,
        'method': method,
        'n_success': n_success,
        'n_failed': n_failed,
    }

    save_precomputed_lensing(
        output_path=str(output_path),
        positions=positions,
        deltasigma_values=deltasigma_values,
        rp_bins=rp_bins,
        metadata=metadata
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Performance Summary")

    total_time = elapsed_halos + elapsed_particles + elapsed_kdtree + elapsed_selection + elapsed_precompute

    print(f"\n  {'Step':<45s} {'Time':>10s}  {'%':>6s}")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'1. Load halo catalog':<45s} {elapsed_halos:>9.3f}s  {elapsed_halos/total_time*100:>5.1f}%")
    print(f"  {'2. Load & downsample particle catalog':<45s} {elapsed_particles:>9.3f}s  {elapsed_particles/total_time*100:>5.1f}%")
    print(f"  {'3. Build particle KD-tree':<45s} {elapsed_kdtree:>9.3f}s  {elapsed_kdtree/total_time*100:>5.1f}%")
    print(f"  {'4. Select particles near halos':<45s} {elapsed_selection:>9.3f}s  {elapsed_selection/total_time*100:>5.1f}%")
    print(f"  {'5. Pre-compute ΔΣ grid':<45s} {elapsed_precompute:>9.3f}s  {elapsed_precompute/total_time*100:>5.1f}%")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'TOTAL':<45s} {total_time:>9.3f}s  100.0%")

    print(f"\n  Key metrics:")
    print(f"    Number of halos: {len(df_halos):,}")
    print(f"    Number of particles (downsampled): {len(df_particles):,}")
    print(f"    Number of selected particles: {len(selected_particle_indices):,}")
    print(f"    Selection efficiency: {len(selected_particle_indices)/len(df_particles)*100:.1f}%")
    print(f"    Number of successful ΔΣ computations: {n_success:,}")
    print(f"    Time per ΔΣ computation: {elapsed_precompute/n_success*1e3:.2f} ms")
    print(f"    Computations per second: {n_success/elapsed_precompute:,.1f}")
    print(f"    Output file size: {output_path.stat().st_size / 1e6:.1f} MB")

    print("\n" + "="*70)
    print("  Fast ΔΣ pre-computation test completed successfully!")
    print("="*70 + "\n")

    print(f"\nNext steps:")
    print(f"  1. Load the pre-computed grid with load_precomputed_lensing()")
    print(f"  2. Use interpolation to get ΔΣ at galaxy positions during HOD sampling")
    print(f"  3. This avoids recomputing correlation functions for every HOD trial")


def compare_lensing_pipelines():
    """
    Compare standard vs. fast lensing pipelines using realistic HOD galaxy sample.

    Uses the same HOD parameters from test_performance.py to populate a realistic
    galaxy catalog, then compares lensing computation using:
    1. Fast pipeline: Pre-computed grid + interpolation
    2. Standard pipeline: Direct correlation function computation
    """
    from HOD_NRV.HOD_numerical.HOD.HOD_catalogue import HaloOccupation
    from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal
    from HOD_NRV.HOD_numerical.twopoint_calculator.fast_two_point import FastDeltaSigmaCalculator

    print_header("Lensing Pipeline Comparison (Realistic HOD)")

    # ========================================================================
    # Configuration (same as test_performance.py)
    # ========================================================================
    halo_path = '/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet'
    particle_path = '/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet'

    cosmo_params = {
        'H0': 67.74,
        'Om0': 0.3089,
        'Ob0': 0.0486,
        'sigma8': 0.8159,
        'ns': 0.9667
    }

    column_mapping = {
        "x": "x", "y": "y", "z": "z",
        "vx": "vx", "vy": "vy", "vz": "vz",
        "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms"
    }

    Lbox = 681  # Mpc/h
    rho_crit = 2.77536627e11
    RHO_M = cosmo_params['Om0'] * rho_crit

    # HOD parameters (same as test_performance.py)
    hod_params_base = {
        'Mmin': 12.0,
        'sig_M': 0.4,
        'As': 0.2,
        'M1': 13.0,
        'alpha': 0.8,
        'kappa': 0.8
    }

    target_ngal = 2e-4  # (h/Mpc)^3

    # Lensing bins (same as test_performance.py)
    rp_lens_bins = np.logspace(-1, 1.5, 16)
    bins_comp = np.geomspace(5e-2, 100, 31)
    downsample_factor = 10

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    precomputed_grid_path = output_dir / 'precomputed_lensing_grid.h5'

    print(f"\n  Configuration:")
    print(f"    Box size: {Lbox} Mpc/h")
    print(f"    Target ngal: {target_ngal:.2e} (h/Mpc)^3")
    print(f"    HOD parameters: {hod_params_base}")

    # ========================================================================
    # Step 1: Check Pre-computed Grid
    # ========================================================================
    print_header("Step 1: Load Pre-computed Lensing Grid")

    if not precomputed_grid_path.exists():
        print(f"\n  ⚠️  Pre-computed grid not found!")
        print(f"  Running pre-computation first...\n")
        main()

    print(f"\n  Loading pre-computed grid from: {precomputed_grid_path.name}")

    # Initialize both mean and median calculators
    start = time.time()
    fast_calc_mean = FastDeltaSigmaCalculator(
        precomputed_file=str(precomputed_grid_path),
        interpolation_method='idw',
        k_neighbors=8,
        Lbox=Lbox,
        use_median=False
    )
    elapsed_load_mean = time.time() - start
    print_timing("Load pre-computed grid (mean)", elapsed_load_mean)

    start = time.time()
    fast_calc_median = FastDeltaSigmaCalculator(
        precomputed_file=str(precomputed_grid_path),
        interpolation_method='idw',
        k_neighbors=8,
        Lbox=Lbox,
        use_median=True
    )
    elapsed_load_median = time.time() - start
    print_timing("Load pre-computed grid (median)", elapsed_load_median)

    # ========================================================================
    # Step 2: Load Data and Initialize HOD
    # ========================================================================
    print_header("Step 2: Load Data and Initialize HOD")

    print(f"\n  Loading catalogs...")
    start = time.time()
    df_halos = pd.read_parquet(halo_path)
    df_particles = pd.read_parquet(particle_path)
    elapsed_load_data = time.time() - start
    print_timing("Load halo + particle catalogs", elapsed_load_data)
    print(f"    → Halos: {len(df_halos):,}")
    print(f"    → Particles: {len(df_particles):,}")

    print(f"\n  Initializing HaloOccupation...")
    start = time.time()
    halo = HaloOccupation(
        cosmology=cosmo_params,
        zeff=1.0,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition="vir",
        DataFrame=df_halos,
        DataFrame_part=df_particles,
        assembly_bias=False,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False
    )
    elapsed_init = time.time() - start
    print_timing("Initialize HaloOccupation", elapsed_init)

    # ========================================================================
    # Step 3: Populate Galaxies with HOD
    # ========================================================================
    print_header("Step 3: Populate Galaxies with HOD")

    halo.set_halo_model('ELG_GHOD')

    print(f"\n  Finding Ac for target ngal = {target_ngal:.2e}...")
    start = time.time()
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, hod_params_base, target_ngal, Ac_fiducial=1.0
    )
    elapsed_rescale = time.time() - start
    print_timing("Rescale Ac/As", elapsed_rescale)
    print(f"    → Ac = {float(Ac_rescaled):.6f}")
    print(f"    → As = {float(As_rescaled):.6f}")

    hod_params = hod_params_base.copy()
    hod_params['Ac'] = Ac_rescaled
    hod_params['As'] = As_rescaled

    print(f"\n  Populating halos...")
    start = time.time()
    halo.populate_haloes(hod_params, random_seed=42)
    elapsed_populate = time.time() - start

    n_galaxies = len(halo.positions_gal)
    ngal_actual = n_galaxies / Lbox**3

    print_timing("Populate galaxies", elapsed_populate)
    print(f"    → Total galaxies: {n_galaxies:,}")
    print(f"    → Achieved ngal: {ngal_actual:.6e} (h/Mpc)^3")
    print(f"    → Satellite fraction: {halo.satellite_fraction:.4f}")

    # ========================================================================
    # Step 4a: Fast Pipeline (Interpolation with Mean)
    # ========================================================================
    print_header("Step 4a: Fast Pipeline (Mean Interpolation)")

    print(f"\n  Computing ΔΣ using fast interpolation (MEAN)...")
    start = time.time()
    rp_fast_mean, ds_fast_mean = fast_calc_mean.compute_deltasigma_for_galaxies(
        halo.positions_gal, rp_bins=rp_lens_bins, verbose=False
    )
    elapsed_fast_mean = time.time() - start

    print_timing("Fast pipeline (mean) computation", elapsed_fast_mean)
    print(f"    → {n_galaxies/elapsed_fast_mean:.1f} galaxies/second")
    print(f"    → ΔΣ range: [{ds_fast_mean.min():.2e}, {ds_fast_mean.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 4b: Fast Pipeline (Interpolation with Median)
    # ========================================================================
    print_header("Step 4b: Fast Pipeline (Median Interpolation)")

    print(f"\n  Computing ΔΣ using fast interpolation (MEDIAN)...")
    start = time.time()
    rp_fast_median, ds_fast_median = fast_calc_median.compute_deltasigma_for_galaxies(
        halo.positions_gal, rp_bins=rp_lens_bins, verbose=False
    )
    elapsed_fast_median = time.time() - start

    print_timing("Fast pipeline (median) computation", elapsed_fast_median)
    print(f"    → {n_galaxies/elapsed_fast_median:.1f} galaxies/second")
    print(f"    → ΔΣ range: [{ds_fast_median.min():.2e}, {ds_fast_median.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 5: Standard Pipeline (Direct Computation)
    # ========================================================================
    print_header("Step 5: Standard Pipeline (Direct Computation)")

    print(f"\n  Applying same downsampling to particles (factor {downsample_factor})...")
    df_particles_downsampled = df_particles.iloc[::downsample_factor].reset_index(drop=True)
    particle_positions = df_particles_downsampled[['x', 'y', 'z']].values
    print(f"    → Using {len(particle_positions):,} particles")

    print(f"\n  Computing ΔΣ using standard method...")
    print(f"  (This will be slower - computing cross-correlation function...)")
    start = time.time()
    rp_standard, ds_standard = halo.compute_galaxy_lensing(rp_lens_bins, bins_comp=bins_comp)
    elapsed_standard = time.time() - start

    print_timing("Standard pipeline computation", elapsed_standard)
    print(f"    → {n_galaxies/elapsed_standard:.2f} galaxies/second")
    print(f"    → ΔΣ range: [{ds_standard.min():.2e}, {ds_standard.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 6: Comparison and Analysis
    # ========================================================================
    print_header("Step 6: Comparison Results")

    # Compute speedups
    speedup_mean = elapsed_standard / elapsed_fast_mean
    speedup_median = elapsed_standard / elapsed_fast_median

    print(f"\n  ⏱  Timing Comparison:")
    print(f"    Fast pipeline (mean):   {elapsed_fast_mean:10.3f} s  ({speedup_mean:6.1f}× speedup)")
    print(f"    Fast pipeline (median): {elapsed_fast_median:10.3f} s  ({speedup_median:6.1f}× speedup)")
    print(f"    Standard pipeline:      {elapsed_standard:10.3f} s")

    # Compute relative differences
    rel_diff_mean = (ds_fast_mean - ds_standard) / np.abs(ds_standard) * 100
    abs_rel_diff_mean = np.abs(rel_diff_mean)

    rel_diff_median = (ds_fast_median - ds_standard) / np.abs(ds_standard) * 100
    abs_rel_diff_median = np.abs(rel_diff_median)

    print(f"\n  📊 Accuracy Comparison:")
    print(f"    {'rp [Mpc/h]':<12s} {'ΔΣ_mean':<15s} {'ΔΣ_median':<15s} {'ΔΣ_standard':<15s} {'Mean Err[%]':<12s} {'Median Err[%]':<12s}")
    print(f"    {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")

    for i, rp_val in enumerate(rp_fast_mean):
        print(f"    {rp_val:11.4f}  {ds_fast_mean[i]:14.4e}  {ds_fast_median[i]:14.4e}  "
              f"{ds_standard[i]:14.4e}  {rel_diff_mean[i]:+11.2f}  {rel_diff_median[i]:+11.2f}")

    print(f"\n  📈 Statistical Summary (Mean Interpolation):")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_mean):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_mean):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_mean):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_mean**2)):6.2f}%")

    print(f"\n  📈 Statistical Summary (Median Interpolation):")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_median):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_median):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_median):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_median**2)):6.2f}%")

    # Save comparison results
    comparison_file = output_dir / 'lensing_comparison.npz'

    np.savez(
        comparison_file,
        rp=rp_fast_mean,
        ds_fast_mean=ds_fast_mean,
        ds_fast_median=ds_fast_median,
        ds_standard=ds_standard,
        elapsed_fast_mean=elapsed_fast_mean,
        elapsed_fast_median=elapsed_fast_median,
        elapsed_standard=elapsed_standard,
        speedup_mean=speedup_mean,
        speedup_median=speedup_median,
        n_galaxies=n_galaxies,
        ngal_actual=ngal_actual,
        hod_params=hod_params,
        target_ngal=target_ngal,
        galaxy_positions=halo.positions_gal  # Save for interpolation testing
    )

    print(f"\n  💾 Results saved to: {comparison_file.name}")

    print("\n" + "="*70)
    print(f"  ✅ Comparison completed!")
    print(f"  Fast pipeline (mean):   {speedup_mean:.1f}× faster with ~{np.mean(abs_rel_diff_mean):.1f}% accuracy")
    print(f"  Fast pipeline (median): {speedup_median:.1f}× faster with ~{np.mean(abs_rel_diff_median):.1f}% accuracy")
    print("="*70 + "\n")

    # Determine which method is better
    if np.mean(abs_rel_diff_median) < np.mean(abs_rel_diff_mean):
        print(f"  🎯 MEDIAN interpolation is {np.mean(abs_rel_diff_mean)/np.mean(abs_rel_diff_median):.2f}× more accurate!")
    else:
        print(f"  🎯 MEAN interpolation is {np.mean(abs_rel_diff_median)/np.mean(abs_rel_diff_mean):.2f}× more accurate!")
    print()

    return rp_fast_mean, ds_fast_mean, ds_fast_median, ds_standard, speedup_mean, speedup_median


if __name__ == "__main__":
    # Uncomment to run pre-computation only:
    # main()

    # Run comparison with realistic HOD population:
    compare_lensing_pipelines()
