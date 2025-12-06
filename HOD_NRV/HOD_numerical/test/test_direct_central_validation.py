"""
Direct validation test: Individual ΔΣ computation at central galaxy positions.

This test provides the cleanest validation by:
1. Populating CENTRALS ONLY (no satellites → simpler interpretation)
2. Computing ΔΣ individually at each central position (NO interpolation)
3. Stacking individual profiles to get ensemble average
4. Comparing to standard HOD pipeline

Key advantages:
- Zero interpolation artifacts (pure direct computation)
- Clean statistical analysis (mean, median, scatter)
- Can test on subset for speed
- Diagnostic: If this fails, issue is in core ΔΣ computation

Workflow:
1. Load data and initialize HOD
2. Populate centrals only (As=0 → no satellites)
3. For each central: compute ΔΣ_i(rp) using spherical method
4. Stack profiles: ΔΣ_mean, ΔΣ_median, ΔΣ_std
5. Compare to standard pipeline

CRITICAL: Scale Consistency
----------------------------
The direct spherical method and standard pipeline must use matching integration
ranges to give comparable results:

- Standard pipeline (standard_two_point_calculator.py):
  * bins_comp: 0.005 to 100 Mpc/h for ξ_gm(r) computation
  * chi_max: 150 Mpc/h for Σ(rp) line-of-sight integration

- Direct spherical method (compute_deltasigma_spherical):
  * search_radius: Must be ≥150 Mpc/h to match standard pipeline
  * Uses Abel transform instead of ξ_gm, but needs same particle range

If search_radius < 150 Mpc/h:
  → Direct method will UNDERESTIMATE ΔΣ by missing large-scale contributions
  → Causes systematic ~120% error if search_radius ≈ 47 Mpc/h
  → Fix: Set search_radius = 150.0 Mpc/h
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.HOD.HOD_catalogue import HaloOccupation
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal
from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_spherical, compute_deltasigma_at_position
)
from test_utils import print_header, print_timing


def main(n_centrals_max=None, downsample_factor=20):
    """
    Direct validation test: compute ΔΣ individually at each central position.

    Parameters
    ----------
    n_centrals_max : int, optional
        Maximum number of centrals to process (for testing). None = all centrals.
    downsample_factor : int, default=20
        Particle downsampling factor (higher = faster but less accurate)
    """
    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("Direct Central Validation Test Configuration")

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

    # HOD parameters (ELG with NO SATELLITES)
    hod_params_base = {
        'Mmin': 12.0,
        'sig_M': 0.4,
        'As': 0.0,  # NO SATELLITES! (centrals only)
        'M1': 13.0,
        'alpha': 0.8,
        'kappa': 0.8
    }

    target_ngal = 2e-4  # (h/Mpc)^3

    # Lensing bins
    rp_bins = np.logspace(-1, 1.5, 16)  # 0.1 to ~31.6 Mpc/h
    bins_comp = np.geomspace(5e-2, 100, 31)

    # Search radius for ΔΣ computation
    # CRITICAL: Must match standard pipeline's integration range!
    # Standard pipeline uses chi_max=150 Mpc/h (see standard_two_point_calculator.py:219)
    # and bins_comp up to 100 Mpc/h for ξ_gm computation
    search_radius = 100.0  # Mpc/h - matches standard pipeline chi_max

    # Output
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print(f"\n  Configuration:")
    print(f"    Halo catalog: {Path(halo_path).name}")
    print(f"    Particle catalog: {Path(particle_path).name}")
    print(f"    Box size: {Lbox} Mpc/h")
    print(f"\n  HOD parameters (CENTRALS ONLY):")
    print(f"    Mmin: {hod_params_base['Mmin']}")
    print(f"    sig_M: {hod_params_base['sig_M']}")
    print(f"    As: {hod_params_base['As']} (NO SATELLITES!)")
    print(f"    Target ngal: {target_ngal:.2e} (h/Mpc)^3")
    print(f"\n  ΔΣ computation:")
    print(f"    rp bins: {len(rp_bins)-1} from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"    Search radius: {search_radius:.1f} Mpc/h")
    print(f"    bins_comp range: {bins_comp[0]:.3f} to {bins_comp[-1]:.1f} Mpc/h")
    print(f"    Particle downsampling: {downsample_factor}×")

    # Diagnostic: Check scale consistency
    if search_radius < bins_comp[-1]:
        print(f"\n    ⚠️  WARNING: search_radius ({search_radius:.1f}) < bins_comp max ({bins_comp[-1]:.1f})")
        print(f"        This will cause ΔΣ underestimation!")
    if search_radius < rp_bins[-1] * 2:
        print(f"\n    ⚠️  WARNING: search_radius ({search_radius:.1f}) < 2×rp_max ({rp_bins[-1]*2:.1f})")
        print(f"        Recommended: search_radius > 2×rp_max for accurate Abel transform")

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

    # Get RHO_M from the halo class (ensures consistency with standard pipeline)
    RHO_M = halo.RHO_M
    print(f"    → RHO_M from HOD class: {RHO_M:.3e} Msun/h / (Mpc/h)^3")

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

    print(f"\n  Will compute individual ΔΣ for {n_centrals:,} centrals")

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
    # Step 4: Compute Individual ΔΣ at Each Central Position
    # ========================================================================
    print_header("Step 4: Compute Individual ΔΣ at Each Central")

    print(f"\n  Computing ΔΣ individually for {n_centrals:,} centrals...")
    print(f"  Method: spherical (Abel transform)")
    print(f"  Search radius: {search_radius:.1f} Mpc/h")
    print(f"  This may take a while...\n")

    start = time.time()

    individual_deltasigma = []
    valid_positions = []
    n_success = 0
    n_failed = 0

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
            # Compute ΔΣ at this central position
            delta_sigma = compute_deltasigma_spherical(
                position=central_pos,
                nearby_particles=nearby_particles,
                RHO_M=RHO_M,
                rp_bins=rp_bins,
                n_particles_total=len(particle_positions),
                Lbox=Lbox
            )

            individual_deltasigma.append(delta_sigma)
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
    individual_deltasigma = np.array(individual_deltasigma)  # shape (N_centrals, N_rp_bins)
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

    print(f"    Shape: {individual_deltasigma.shape}")

    # ========================================================================
    # Step 5: Stack Individual Profiles
    # ========================================================================
    print_header("Step 5: Stack Individual ΔΣ Profiles")

    print(f"\n  Computing ensemble statistics from {n_success:,} individual profiles...")

    # Compute statistics
    ds_direct_mean = np.mean(individual_deltasigma, axis=0)
    ds_direct_median = np.median(individual_deltasigma, axis=0)
    ds_direct_std = np.std(individual_deltasigma, axis=0)
    ds_direct_sem = ds_direct_std / np.sqrt(n_success)  # Standard error of mean

    print(f"\n  Direct computation statistics:")
    print(f"    Mean ΔΣ range: [{ds_direct_mean.min():.2e}, {ds_direct_mean.max():.2e}] Msun h/pc²")
    print(f"    Median ΔΣ range: [{ds_direct_median.min():.2e}, {ds_direct_median.max():.2e}] Msun h/pc²")
    print(f"    Std ΔΣ range: [{ds_direct_std.min():.2e}, {ds_direct_std.max():.2e}] Msun h/pc²")
    print(f"    SEM ΔΣ range: [{ds_direct_sem.min():.2e}, {ds_direct_sem.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 6: Standard HOD Pipeline
    # ========================================================================
    print_header("Step 6: Standard HOD Pipeline (for comparison)")

    print(f"\n  Computing ΔΣ using standard HOD method...")
    print(f"  (Cross-correlation function approach)")

    start = time.time()
    rp_standard, ds_standard = halo.compute_galaxy_lensing(rp_bins, bins_comp=bins_comp)
    elapsed_standard = time.time() - start

    print_timing("Standard pipeline computation", elapsed_standard)
    print(f"    → ΔΣ range: [{ds_standard.min():.2e}, {ds_standard.max():.2e}] Msun h/pc²")

    # ========================================================================
    # Step 7: Comparison and Analysis
    # ========================================================================
    print_header("Step 7: Direct vs Standard Comparison")

    # Relative differences
    rel_diff_mean = (ds_direct_mean - ds_standard) / np.abs(ds_standard) * 100
    rel_diff_median = (ds_direct_median - ds_standard) / np.abs(ds_standard) * 100

    print(f"\n  📊 Detailed Comparison:")
    print(f"    {'rp [Mpc/h]':<12s} {'Direct Mean':<14s} {'Direct Median':<14s} "
          f"{'Standard':<14s} {'Mean Err[%]':<12s} {'Median Err[%]':<12s} {'SEM':<12s}")
    print(f"    {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    for i, rp in enumerate(rp_centers):
        print(f"    {rp:11.4f}  {ds_direct_mean[i]:13.4e}  {ds_direct_median[i]:13.4e}  "
              f"{ds_standard[i]:13.4e}  {rel_diff_mean[i]:+11.2f}  "
              f"{rel_diff_median[i]:+11.2f}  {ds_direct_sem[i]:11.2e}")

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
    print(f"\n  📊 Scatter Analysis:")
    print(f"    Mean fractional scatter (σ/μ): {np.mean(ds_direct_std/ds_direct_mean)*100:.2f}%")
    print(f"    Max fractional scatter (σ/μ):  {np.max(ds_direct_std/ds_direct_mean)*100:.2f}%")

    # ========================================================================
    # Step 8: Save Results
    # ========================================================================
    print_header("Step 8: Save Results")

    output_file = output_dir / 'direct_central_validation.npz'

    np.savez(
        output_file,
        # Radial bins
        rp_bins=rp_bins,
        rp_centers=rp_centers,
        # Individual profiles
        individual_deltasigma=individual_deltasigma,
        valid_positions=np.array(valid_positions),
        # Stacked profiles
        ds_direct_mean=ds_direct_mean,
        ds_direct_median=ds_direct_median,
        ds_direct_std=ds_direct_std,
        ds_direct_sem=ds_direct_sem,
        # Standard pipeline
        ds_standard=ds_standard,
        rp_standard=rp_standard,
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
    print(f"  {'7. Individual ΔΣ computation':<45s} {elapsed_individual:>9.3f}s  {elapsed_individual/total_time*100:>5.1f}%")
    print(f"  {'8. Standard pipeline':<45s} {elapsed_standard:>9.3f}s  {elapsed_standard/total_time*100:>5.1f}%")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'TOTAL':<45s} {total_time:>9.3f}s  100.0%")

    # Final verdict
    print("\n" + "="*70)
    print("  ✅ Direct Central Validation Test Complete!")
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
    print(f"\n  This test has ZERO interpolation - any differences are from:")
    print(f"    - Particle downsampling ({downsample_factor}×)")
    print(f"    - Statistical noise ({n_success:,} centrals)")
    print(f"    - Method differences (spherical Abel transform vs ξ_gm integration)")
    print(f"\n  Scale consistency check:")
    print(f"    Direct method search_radius: {search_radius:.1f} Mpc/h")
    print(f"    Standard pipeline bins_comp: {bins_comp[0]:.3f} - {bins_comp[-1]:.1f} Mpc/h")
    print(f"    Standard pipeline chi_max: 150 Mpc/h (hardcoded)")
    if search_radius >= 150.0 and search_radius >= bins_comp[-1]:
        print(f"    ✓ Scales are consistent!")
    else:
        print(f"    ⚠ Scale mismatch may affect comparison")
    print("\n" + "="*70 + "\n")

    return (rp_centers, ds_direct_mean, ds_direct_median, ds_standard,
            individual_deltasigma, rel_diff_mean, rel_diff_median)


if __name__ == "__main__":
    # Quick test on subset (fast)
    # main(n_centrals_max=100, downsample_factor=20)

    # Full test on all centrals (slower but better statistics)
    main(n_centrals_max=None, downsample_factor=10)
