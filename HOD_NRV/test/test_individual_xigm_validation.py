"""
Individual xi_gm validation test: Compute xi_gm at each central galaxy position.

This test validates the galaxy-matter correlation function by:
1. Populating CENTRALS ONLY (no satellites → simpler interpretation)
2. Computing xi_gm individually at each central position (NO grid interpolation)
3. Stacking individual xi_gm profiles to get ensemble average
4. Comparing to standard HOD pipeline

Key advantages:
- Zero grid interpolation artifacts (pure direct computation)
- Clean statistical analysis (mean, median, scatter)
- Can test on subset for speed
- Diagnostic: If this fails, issue is in core xi_gm computation

Workflow:
1. Load data and initialize HOD
2. Populate centrals only (As=0 → no satellites)
3. For each central: compute xi_gm(r) by correlating with nearby particles
4. Stack profiles: xi_gm_mean, xi_gm_median, xi_gm_std
5. Compare to standard pipeline

This follows the same methodology as test_direct_central_validation.py but for
xi_gm instead of ΔΣ.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from scipy.spatial import cKDTree

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD.HOD_catalogue import HaloOccupation
from HOD_NRV.utils.emulator_utils import rescale_Ac_to_target_ngal
from HOD_NRV.twopoint_calculator.precompute_deltasigma import (
    periodic_distance,
    build_particle_kdtree
)
from test_utils import print_header, print_timing


def compute_xigm_at_position(
    galaxy_pos: np.ndarray,
    nearby_particles: np.ndarray,
    r_bins: np.ndarray,
    boxsize: float,
    n_particles_total: int,
    volume_total: float
) -> np.ndarray:
    """
    Compute galaxy-matter correlation function xi_gm(r) at a single galaxy position.

    This directly computes the correlation function by counting particles in
    spherical shells around the galaxy and comparing to expected density.

    Parameters
    ----------
    galaxy_pos : np.ndarray, shape (3,)
        Galaxy position [Mpc/h]
    nearby_particles : np.ndarray, shape (N_nearby, 3)
        Positions of nearby particles [Mpc/h]
    r_bins : np.ndarray
        Radial bin edges [Mpc/h]
    boxsize : float
        Simulation box size for periodic boundaries [Mpc/h]
    n_particles_total : int
        Total number of particles in full catalog
    volume_total : float
        Total volume of simulation [Mpc/h]^3

    Returns
    -------
    xi_gm : np.ndarray, shape (len(r_bins)-1,)
        Galaxy-matter correlation function

    Notes
    -----
    The correlation function is computed as:
    xi_gm(r) = n(r) / n̄ - 1

    where:
    - n(r) is the observed particle density in shell at radius r
    - n̄ is the mean particle density in the box

    This is equivalent to:
    1 + xi_gm(r) = [N_observed(r) / V_shell(r)] / [N_total / V_box]

    References
    ----------
    .. [1] Peebles (1980), "The Large-Scale Structure of the Universe"
    .. [2] Davis & Peebles (1983), ApJ 267, 465
    """
    # Compute 3D distances with periodic boundaries
    r = periodic_distance(nearby_particles, galaxy_pos, boxsize)

    # Count particles in spherical shells
    counts, _ = np.histogram(r, bins=r_bins)

    # Shell volumes
    shell_volumes = (4.0/3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)

    # Observed number density in each shell
    n_observed = counts / shell_volumes

    # Mean number density in box
    n_mean = n_particles_total / volume_total

    # Correlation function: xi = n/n̄ - 1
    xi_gm = (n_observed / n_mean) - 1.0

    return xi_gm


def main(n_centrals_max=None, downsample_factor=20):
    """
    Individual xi_gm validation test: compute xi_gm at each central position.

    Parameters
    ----------
    n_centrals_max : int, optional
        Maximum number of centrals to process (for testing). None = all centrals.
    downsample_factor : int, default=20
        Particle downsampling factor (higher = faster but noisier)
    """
    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("Individual xi_gm Validation Test Configuration")

    # Paths
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

    column_mapping = {
        "x": "x", "y": "y", "z": "z",
        "vx": "vx", "vy": "vy", "vz": "vz",
        "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms"
    }

    # Box parameters
    Lbox = 681  # Mpc/h
    rho_crit = 2.77536627e11
    RHO_M = cosmo_params['Om0'] * rho_crit

    # HOD parameters (ELG with NO SATELLITES)
    hod_params_base = {
        'Mmin': 12.0,
        'sig_M': 0.4,
        'As': 0.0,  # NO SATELLITES! (centrals only)
        'M1': 13.0,
        'alpha': 0.8,
        'kappa': 0.8
    }

    target_ngal = 2e-3  # (h/Mpc)^3

    # Radial bins for xi_gm
    # Match the bins_comp from standard pipeline for direct comparison
    r_bins = np.geomspace(5e-2, 100, 31)  # 0.05 to 100 Mpc/h

    # Search radius for xi_gm computation
    # Should be >= max(r_bins) to capture full profile
    search_radius = 100.0  # Mpc/h

    # Output
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print(f"\n  Configuration:")
    print(f"    Halo catalog: {Path(halo_path).name}")
    print(f"    Particle catalog: {Path(particle_path).name}")
    print(f"    Box size: {Lbox} Mpc/h")
    print(f"    RHO_M: {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"\n  HOD parameters (CENTRALS ONLY):")
    print(f"    Mmin: {hod_params_base['Mmin']}")
    print(f"    sig_M: {hod_params_base['sig_M']}")
    print(f"    As: {hod_params_base['As']} (NO SATELLITES!)")
    print(f"    Target ngal: {target_ngal:.2e} (h/Mpc)^3")
    print(f"\n  xi_gm computation:")
    print(f"    r bins: {len(r_bins)-1} from {r_bins[0]:.2f} to {r_bins[-1]:.2f} Mpc/h")
    print(f"    Search radius: {search_radius:.1f} Mpc/h")
    print(f"    Particle downsampling: {downsample_factor}×")

    if n_centrals_max:
        print(f"\n    Max centrals to process: {n_centrals_max:,} (SUBSET TEST)")
    else:
        print(f"\n    Max centrals to process: ALL")

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print_header("Step 1: Load Halo and Particle Catalogs")

    start = time.time()
    df_halos = pd.read_parquet(halo_path)
    elapsed_load_halos = time.time() - start
    print_timing("Load halo catalog", elapsed_load_halos)
    print(f"    → Loaded {len(df_halos):,} halos")

    start = time.time()
    df_particles = pd.read_parquet(particle_path)
    elapsed_load_particles = time.time() - start
    print_timing("Load particle catalog", elapsed_load_particles)
    print(f"    → Loaded {len(df_particles):,} particles (pre-downsampled)")

    # Additional downsampling
    n_particles_original = len(df_particles)
    df_particles = df_particles.iloc[::downsample_factor].reset_index(drop=True)
    n_particles_downsampled = len(df_particles)
    particle_positions = df_particles[['x', 'y', 'z']].values

    print(f"    → Further downsampled to {n_particles_downsampled:,} particles")
    print(f"    → Total downsampling: {n_particles_original/n_particles_downsampled:.1f}×")

    # ========================================================================
    # Step 2: Initialize HOD and Populate Centrals Only
    # ========================================================================
    print_header("Step 2: Initialize HOD and Populate Centrals")

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

    halo.set_halo_model('ELG_GHOD')

    print(f"\n  Finding Ac for target ngal = {target_ngal:.2e}...")
    start = time.time()
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, hod_params_base, target_ngal, Ac_fiducial=1.0
    )
    elapsed_rescale = time.time() - start
    print_timing("Rescale Ac/As", elapsed_rescale)
    print(f"    → Ac = {float(Ac_rescaled):.6f}")
    print(f"    → As = {float(As_rescaled):.6f} (will be set to 0)")

    # Force As=0 to get centrals only
    hod_params = hod_params_base.copy()
    hod_params['Ac'] = Ac_rescaled
    hod_params['As'] = 0.0  # CENTRALS ONLY!

    print(f"\n  Populating halos (CENTRALS ONLY)...")
    start = time.time()
    halo.populate_haloes(hod_params, random_seed=42)
    elapsed_populate = time.time() - start

    n_galaxies = len(halo.positions_gal)
    ngal_actual = n_galaxies / Lbox**3

    print_timing("Populate galaxies", elapsed_populate)
    print(f"    → Total galaxies: {n_galaxies:,}")
    print(f"    → Achieved ngal: {ngal_actual:.6e} (h/Mpc)^3")
    print(f"    → Satellite fraction: {halo.satellite_fraction:.4f} (should be 0.0)")

    if halo.satellite_fraction > 0.01:
        print(f"\n  ⚠️  WARNING: Non-zero satellite fraction detected!")
        print(f"      This should be a centrals-only test.")

    # Get central positions
    central_positions = halo.positions_gal.copy()

    # Limit to subset if requested
    if n_centrals_max and n_centrals_max < len(central_positions):
        print(f"\n  Limiting to first {n_centrals_max:,} centrals for testing...")
        central_positions = central_positions[:n_centrals_max]
        n_centrals = n_centrals_max
    else:
        n_centrals = len(central_positions)

    print(f"\n  Will compute individual xi_gm for {n_centrals:,} centrals")

    # ========================================================================
    # Step 3: Build KDTree for Particle Queries
    # ========================================================================
    print_header("Step 3: Build KDTree for Particle Queries")

    print(f"\n  Building KD-tree for {len(particle_positions):,} particles...")
    start = time.time()
    kdtree = build_particle_kdtree(particle_positions, Lbox)
    elapsed_kdtree = time.time() - start
    print_timing("Build KDTree", elapsed_kdtree)

    # ========================================================================
    # Step 4: Compute Individual xi_gm at Each Central Position
    # ========================================================================
    print_header("Step 4: Compute Individual xi_gm at Each Central")

    print(f"\n  Computing xi_gm individually for {n_centrals:,} centrals...")
    print(f"  Method: Direct particle counting in spherical shells")
    print(f"  Search radius: {search_radius:.1f} Mpc/h")
    print(f"  This may take a while...\n")

    start = time.time()

    individual_xigm = []
    valid_positions = []
    n_success = 0
    n_failed = 0

    volume_total = Lbox**3

    for i, central_pos in enumerate(central_positions):
        if (i + 1) % max(1, n_centrals // 10) == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n_centrals - i - 1) / rate
            print(f"    Progress: {i+1}/{n_centrals} ({100*(i+1)/n_centrals:.1f}%) | "
                  f"Rate: {rate:.1f} gal/s | ETA: {remaining:.1f} s")

        # Query particles within search radius
        nearby_indices = kdtree.query_ball_point(central_pos, r=search_radius)

        if len(nearby_indices) < 50:
            n_failed += 1
            continue

        nearby_particles = particle_positions[nearby_indices]

        try:
            # Compute xi_gm at this central position
            xi_gm = compute_xigm_at_position(
                galaxy_pos=central_pos,
                nearby_particles=nearby_particles,
                r_bins=r_bins,
                boxsize=Lbox,
                n_particles_total=len(particle_positions),
                volume_total=volume_total
            )

            individual_xigm.append(xi_gm)
            valid_positions.append(central_pos)
            n_success += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"    Warning: Failed at position {central_pos}: {e}")

    elapsed_individual = time.time() - start
    print_timing("\nTotal individual computation time", elapsed_individual)

    print(f"\n  Individual computation results:")
    print(f"    Successful: {n_success:,}")
    print(f"    Failed: {n_failed:,}")
    print(f"    Success rate: {n_success/(n_success+n_failed)*100:.1f}%")
    print(f"    Time per galaxy: {elapsed_individual/n_success*1e3:.2f} ms")

    # Convert to array for stacking
    individual_xigm = np.array(individual_xigm)  # shape (N_centrals, N_r_bins)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])

    print(f"    Shape: {individual_xigm.shape}")

    # ========================================================================
    # Step 5: Stack Individual Profiles
    # ========================================================================
    print_header("Step 5: Stack Individual xi_gm Profiles")

    print(f"\n  Computing ensemble statistics from {n_success:,} individual profiles...")

    # Compute statistics
    xigm_direct_mean = np.mean(individual_xigm, axis=0)
    xigm_direct_median = np.median(individual_xigm, axis=0)
    xigm_direct_std = np.std(individual_xigm, axis=0)
    xigm_direct_sem = xigm_direct_std / np.sqrt(n_success)  # Standard error of mean

    print(f"\n  Direct computation statistics:")
    print(f"    Mean xi_gm range: [{xigm_direct_mean.min():.4e}, {xigm_direct_mean.max():.4e}]")
    print(f"    Median xi_gm range: [{xigm_direct_median.min():.4e}, {xigm_direct_median.max():.4e}]")
    print(f"    Std xi_gm range: [{xigm_direct_std.min():.4e}, {xigm_direct_std.max():.4e}]")
    print(f"    SEM xi_gm range: [{xigm_direct_sem.min():.4e}, {xigm_direct_sem.max():.4e}]")

    # ========================================================================
    # Step 6: Standard HOD Pipeline
    # ========================================================================
    print_header("Step 6: Standard HOD Pipeline (for comparison)")

    print(f"\n  Computing xi_gm using standard HOD method...")
    print(f"  (Cross-correlation function approach)")

    start = time.time()
    # Use same bins as individual computation for direct comparison
    r_standard, xigm_standard = halo.compute_galaxy_clustering(
        bins1=r_bins,
        mode='s',
        catalog2=halo.positions_part
    )
    elapsed_standard = time.time() - start

    print_timing("Standard pipeline computation", elapsed_standard)
    print(f"    → xi_gm range: [{xigm_standard.min():.4e}, {xigm_standard.max():.4e}]")

    # ========================================================================
    # Step 7: Comparison and Analysis
    # ========================================================================
    print_header("Step 7: Direct vs Standard Comparison")

    # Relative differences
    # For xi_gm, be careful with near-zero values
    # Use absolute difference / (1 + |xi_standard|) for robustness
    rel_diff_mean = (xigm_direct_mean - xigm_standard) / (1 + np.abs(xigm_standard)) * 100
    rel_diff_median = (xigm_direct_median - xigm_standard) / (1 + np.abs(xigm_standard)) * 100

    print(f"\n  📊 Detailed Comparison:")
    print(f"    {'r [Mpc/h]':<12s} {'Direct Mean':<14s} {'Direct Median':<14s} "
          f"{'Standard':<14s} {'Mean Err[%]':<12s} {'Median Err[%]':<12s} {'SEM':<12s}")
    print(f"    {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    for i, r in enumerate(r_centers):
        print(f"    {r:11.4f}  {xigm_direct_mean[i]:+13.4e}  {xigm_direct_median[i]:+13.4e}  "
              f"{xigm_standard[i]:+13.4e}  {rel_diff_mean[i]:+11.2f}  "
              f"{rel_diff_median[i]:+11.2f}  {xigm_direct_sem[i]:11.2e}")

    # Statistical summary
    abs_rel_diff_mean = np.abs(rel_diff_mean)
    abs_rel_diff_median = np.abs(rel_diff_median)

    print(f"\n  📈 Statistical Summary (Mean):")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_mean):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_mean):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_mean):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_mean**2)):6.2f}%")

    print(f"\n  📈 Statistical Summary (Median):")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_median):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_median):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_median):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_median**2)):6.2f}%")

    # Scatter analysis
    # Mask out bins where mean is near zero to avoid division issues
    mask_nonzero = np.abs(xigm_direct_mean) > 1e-3
    if np.any(mask_nonzero):
        fractional_scatter = xigm_direct_std[mask_nonzero] / np.abs(xigm_direct_mean[mask_nonzero])
        print(f"\n  📊 Scatter Analysis (non-zero bins):")
        print(f"    Mean fractional scatter (σ/|μ|): {np.mean(fractional_scatter)*100:.2f}%")
        print(f"    Max fractional scatter (σ/|μ|):  {np.max(fractional_scatter)*100:.2f}%")

    # ========================================================================
    # Step 8: Save Results
    # ========================================================================
    print_header("Step 8: Save Results")

    output_file = output_dir / 'individual_xigm_validation.npz'

    np.savez(
        output_file,
        # Radial bins
        r_bins=r_bins,
        r_centers=r_centers,
        # Individual profiles
        individual_xigm=individual_xigm,
        valid_positions=np.array(valid_positions),
        # Stacked profiles
        xigm_direct_mean=xigm_direct_mean,
        xigm_direct_median=xigm_direct_median,
        xigm_direct_std=xigm_direct_std,
        xigm_direct_sem=xigm_direct_sem,
        # Standard pipeline
        xigm_standard=xigm_standard,
        r_standard=r_standard,
        # Comparison
        rel_diff_mean=rel_diff_mean,
        rel_diff_median=rel_diff_median,
        # Metadata
        n_centrals=n_centrals,
        n_success=n_success,
        n_failed=n_failed,
        hod_params=hod_params,
        downsample_factor=downsample_factor,
        search_radius=search_radius
    )

    print(f"\n  💾 Results saved to: {output_file.name}")
    print(f"    File size: {output_file.stat().st_size / 1e6:.2f} MB")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Performance Summary")

    total_time = (elapsed_load_halos + elapsed_load_particles + elapsed_init +
                  elapsed_rescale + elapsed_populate + elapsed_kdtree +
                  elapsed_individual + elapsed_standard)

    print(f"\n  {'Step':<45s} {'Time':>10s}  {'%':>6s}")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'1. Load halos':<45s} {elapsed_load_halos:>9.3f}s  {elapsed_load_halos/total_time*100:>5.1f}%")
    print(f"  {'2. Load particles':<45s} {elapsed_load_particles:>9.3f}s  {elapsed_load_particles/total_time*100:>5.1f}%")
    print(f"  {'3. Initialize HOD':<45s} {elapsed_init:>9.3f}s  {elapsed_init/total_time*100:>5.1f}%")
    print(f"  {'4. Rescale Ac':<45s} {elapsed_rescale:>9.3f}s  {elapsed_rescale/total_time*100:>5.1f}%")
    print(f"  {'5. Populate centrals':<45s} {elapsed_populate:>9.3f}s  {elapsed_populate/total_time*100:>5.1f}%")
    print(f"  {'6. Build KDTree':<45s} {elapsed_kdtree:>9.3f}s  {elapsed_kdtree/total_time*100:>5.1f}%")
    print(f"  {'7. Individual xi_gm computation':<45s} {elapsed_individual:>9.3f}s  {elapsed_individual/total_time*100:>5.1f}%")
    print(f"  {'8. Standard pipeline':<45s} {elapsed_standard:>9.3f}s  {elapsed_standard/total_time*100:>5.1f}%")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'TOTAL':<45s} {total_time:>9.3f}s  100.0%")

    # Final verdict
    print("\n" + "="*70)
    print("  ✅ Individual xi_gm Validation Test Complete!")
    print("="*70)

    # Interpretation
    mean_error = np.mean(abs_rel_diff_mean)
    median_error = np.mean(abs_rel_diff_median)

    if mean_error < 5.0:
        verdict = "✓ EXCELLENT agreement"
    elif mean_error < 10.0:
        verdict = "✓ GOOD agreement"
    elif mean_error < 20.0:
        verdict = "⚠ MODERATE agreement"
    else:
        verdict = "✗ POOR agreement"

    print(f"\n  Verdict: {verdict}")
    print(f"    Direct (mean) vs Standard: {mean_error:.2f}% mean error")
    print(f"    Direct (median) vs Standard: {median_error:.2f}% mean error")
    print(f"\n  This test has ZERO grid interpolation - any differences are from:")
    print(f"    - Particle downsampling ({downsample_factor}×)")
    print(f"    - Statistical noise ({n_success:,} centrals)")
    print(f"    - Method differences (direct counting vs pycorr cross-correlation)")
    print(f"\n  Scale consistency check:")
    print(f"    Direct method search_radius: {search_radius:.1f} Mpc/h")
    print(f"    Radial bins: {r_bins[0]:.3f} - {r_bins[-1]:.1f} Mpc/h")
    if search_radius >= r_bins[-1]:
        print(f"    ✓ Search radius covers all bins!")
    else:
        print(f"    ⚠ Search radius may truncate large-scale signal")
    print("\n" + "="*70 + "\n")

    return (r_centers, xigm_direct_mean, xigm_direct_median, xigm_standard,
            individual_xigm, rel_diff_mean, rel_diff_median)


if __name__ == "__main__":
    # Quick test on subset (fast)
    # main(n_centrals_max=100, downsample_factor=20)

    # Full test on all centrals (slower but better statistics)
    main(n_centrals_max=None, downsample_factor=20)
