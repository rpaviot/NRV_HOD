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

from HOD_NRV.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_at_position,
    save_precomputed_lensing
)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_timing(step_name, elapsed_time, units="s"):
    """Print timing result."""
    if units == "ms":
        print(f"  ⏱  {step_name:40s} {elapsed_time*1000:8.2f} ms")
    else:
        print(f"  ⏱  {step_name:40s} {elapsed_time:8.3f} s")


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

    # Search radius for ξ_gm computation (independent of particle selection)
    # This should cover BAO scales (~100-150 Mpc/h)
    search_radius_xigm = 100.0  # Mpc/h - maximum scale for ξ_gm computation

    # Line-of-sight integration limit for Σ(rp)
    chi_max = 150.0  # Mpc/h

    # Downsampling factor for particles
    downsample_factor = 20

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
    print(f"    rp bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"    Particle selection: within {r_factor_selection:.1f}×Rvir of halos")
    print(f"    ξ_gm search radius: {search_radius_xigm:.1f} Mpc/h (BAO scale)")
    print(f"    χ_max (line-of-sight): {chi_max:.1f} Mpc/h")
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
    print(f"  Each particle computes ξ_gm using ALL {len(particle_positions):,} particles")
    print(f"  Search radius for ξ_gm: {search_radius_xigm:.1f} Mpc/h")
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

        # Query ALL particles within search_radius_xigm for ξ_gm computation
        nearby_indices = kdtree_particles.query_ball_point(particle_pos, r=search_radius_xigm)

        if len(nearby_indices) < 10:
            n_failed += 1
            continue

        nearby_particles = particle_positions[nearby_indices]

        try:
            delta_sigma = compute_deltasigma_at_position(
                position=particle_pos,
                nearby_particles=nearby_particles,
                RHO_M=RHO_M,
                rp_bins=rp_bins,
                chi_max=chi_max
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
        'search_radius_xigm': search_radius_xigm,
        'chi_max': chi_max,
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


if __name__ == "__main__":
    main()
