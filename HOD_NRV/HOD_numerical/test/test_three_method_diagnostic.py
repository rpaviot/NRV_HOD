"""
Three-Method Diagnostic: Compare ΔΣ computation methods on massive halos.

This diagnostic test compares three different approaches to computing galaxy-galaxy
lensing signal ΔΣ(rp) around the 10 most massive halos:

Method 1: STANDARD (ξ_gm → integration)
  - Compute ξ_gm(r) via pycorr cross-correlation
  - Integrate using DeltaSigmaCalculator
  - This is the "reference" method used in standard HOD pipeline

Method 2: SPHERICAL (ρ(r) → cylindrical projection)
  - Compute 3D density profile ρ(r) in spherical shells
  - Project to Σ(rp) using cylindrical line-of-sight integration
  - Uses compute_deltasigma_spherical()

Method 3: CYLINDRICAL 2D (direct projected counting)
  - Directly count particles in projected cylindrical annuli
  - No intermediate ρ(r) profile
  - Uses compute_deltasigma_at_position()

Purpose:
--------
If Method 1 and either Method 2 or 3 agree, the issue in test_direct_central_validation.py
is NOT in the computation method itself, but in:
  - Averaging over many galaxies with poor statistics
  - Particle downsampling effects
  - Stacking/ensemble averaging methodology

If all methods disagree, there's a fundamental issue in one or more implementations.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_spherical,
    compute_deltasigma_at_position,
    periodic_distance,
    compute_xigm_at_position
)
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import DeltaSigmaCalculator
from test_utils import print_header, print_timing


def main(n_halos=10, downsample_factor=2, search_radius=150.0, chi_max=150.0):
    """
    Three-method diagnostic test on massive halos.

    Parameters
    ----------
    n_halos : int, default=10
        Number of most massive halos to test
    downsample_factor : int, default=2
        Particle downsampling factor
    search_radius : float, default=150.0
        Search radius for finding nearby particles [Mpc/h]
    chi_max : float, default=150.0
        Line-of-sight integration limit [Mpc/h]
    """
    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("Three-Method Diagnostic Configuration")

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

    # Box parameters
    Lbox = 681  # Mpc/h
    rho_crit = 2.77536627e11
    RHO_M = cosmo_params['Om0'] * rho_crit

    # Binning
    rp_bins = np.logspace(-1, 1.5, 16)  # 0.1 to ~31.6 Mpc/h
    r_bins = np.geomspace(5e-2, 100, 31)  # For ξ_gm computation
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])    # Output
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print(f"\n  Configuration:")
    print(f"    Halo catalog: {Path(halo_path).name}")
    print(f"    Particle catalog: {Path(particle_path).name}")
    print(f"    Box size: {Lbox} Mpc/h")
    print(f"    RHO_M: {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"\n  Test parameters:")
    print(f"    Number of halos: {n_halos} (most massive)")
    print(f"    Search radius: {search_radius:.1f} Mpc/h")
    print(f"    chi_max: {chi_max:.1f} Mpc/h")
    print(f"    Particle downsampling: {downsample_factor}×")
    print(f"\n  Binning:")
    print(f"    rp bins: {len(rp_bins)-1} from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"    r bins (ξ_gm): {len(r_bins)-1} from {r_bins[0]:.3f} to {r_bins[-1]:.1f} Mpc/h")

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

    # Compute particle mass (CRITICAL: use original count before downsampling)
    # But wait - n_particles_original is ALREADY downsampled from the full simulation!
    # We need to account for the TOTAL downsampling to get correct normalization
    V_box = Lbox**3
    m_particle = RHO_M * V_box / n_particles_downsampled
    print(f"    → Particle mass: {m_particle:.3e} Msun/h")

    # ========================================================================
    # Step 2: Select Most Massive Halos
    # ========================================================================
    print_header("Step 2: Select Most Massive Halos")

    # Sort by mass and select top N
    df_halos_sorted = df_halos.sort_values('mass', ascending=False)
    df_massive = df_halos_sorted.head(n_halos).reset_index(drop=True)

    print(f"\n  Selected {n_halos} most massive halos:")
    print(f"    Mass range: {df_massive['mass'].min():.3e} to {df_massive['mass'].max():.3e} Msun/h")
    print(f"    Mean mass: {df_massive['mass'].mean():.3e} Msun/h")
    print(f"    Mean R_vir: {df_massive['rvir'].mean():.2f} Mpc/h")

    halo_positions = df_massive[['x', 'y', 'z']].values
    halo_masses = df_massive['mass'].values
    halo_rvir = df_massive['rvir'].values

    # ========================================================================
    # Step 3: Build KDTree
    # ========================================================================
    print_header("Step 3: Build KDTree for Particle Queries")

    print(f"\n  Building KD-tree for {len(particle_positions):,} particles...")
    start = time.time()
    kdtree = build_particle_kdtree(particle_positions, Lbox)
    elapsed_kdtree = time.time() - start
    print_timing("Build KDTree", elapsed_kdtree)

    # ========================================================================
    # Step 4: Compute ΔΣ for Each Halo with Three Methods
    # ========================================================================
    print_header("Step 4: Compute ΔΣ for Each Halo (Three Methods)")

    # Storage for results
    deltasigma_method1 = []  # Standard (ξ_gm → integration)
    deltasigma_method2 = []  # Spherical (ρ(r) → cylindrical projection)
    deltasigma_method3 = []  # Cylindrical 2D (direct projected counting)
    xigm_profiles = []  # Store ξ_gm for reference

    print(f"\n  Processing {n_halos} halos...")
    print(f"  Methods:")
    print(f"    1. Standard (ξ_gm → integration via DeltaSigmaCalculator)")
    print(f"    2. Spherical (ρ(r) → cylindrical projection)")
    print(f"    3. Cylindrical 2D (direct projected counting)")
    print()

    start_total = time.time()

    for i, (halo_pos, mass, rvir) in enumerate(zip(halo_positions, halo_masses, halo_rvir)):
        print(f"  Halo {i+1}/{n_halos}: M = {mass:.3e} Msun/h, R_vir = {rvir:.2f} Mpc/h")

        # Find nearby particles
        start = time.time()
        nearby_indices = kdtree.query_ball_point(halo_pos, r=search_radius)
        n_nearby = len(nearby_indices)
        print(f"    → Found {n_nearby:,} particles within {search_radius:.1f} Mpc/h")

        if n_nearby < 100:
            print(f"    ⚠️  WARNING: Too few particles, skipping this halo")
            continue

        nearby_particles = particle_positions[nearby_indices]
        elapsed_query = time.time() - start

        # ----------------------------------------------------------------------
        # Method 1: Standard (ξ_gm → integration)
        # ----------------------------------------------------------------------
        start = time.time()

        # Compute ξ_gm at this halo position
        xi_gm = compute_xigm_at_position(
            galaxy_pos=halo_pos,
            nearby_particles=nearby_particles,
            r_bins=r_bins,
            boxsize=Lbox,
            n_particles_total=n_particles_downsampled,
            volume_total=V_box
        )
        xigm_profiles.append(xi_gm)

        # Integrate ξ_gm to get ΔΣ using DeltaSigmaCalculator
        r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
        calc = DeltaSigmaCalculator(r_centers, xi_gm, RHO_M)
        calc.compute_sigma(r_centers, chi_max=chi_max)
        ds_method1 = calc.compute_deltasigma_averaged(rp_bins)
        deltasigma_method1.append(ds_method1)

        elapsed_method1 = time.time() - start
        print(f"    Method 1: {elapsed_method1:.3f} s")

        # ----------------------------------------------------------------------
        # Method 2: Spherical (ρ(r) → cylindrical projection)
        # ----------------------------------------------------------------------
        start = time.time()

        ds_method2 = compute_deltasigma_spherical(
            position=halo_pos,
            nearby_particles=nearby_particles,
            RHO_M=RHO_M,
            rp_bins=rp_bins,
            particle_mass=m_particle,
            Lbox=Lbox,
            chi_max=chi_max
        )
        deltasigma_method2.append(ds_method2)

        elapsed_method2 = time.time() - start
        print(f"    Method 2: {elapsed_method2:.3f} s")

        # ----------------------------------------------------------------------
        # Method 3: Cylindrical 2D (direct projected counting)
        # ----------------------------------------------------------------------
        start = time.time()

        ds_method3 = compute_deltasigma_at_position(
            position=halo_pos,
            nearby_particles=nearby_particles,
            RHO_M=RHO_M,
            rp_bins=rp_bins,
            chi_max=chi_max,
            particle_mass=m_particle,
            Lbox=Lbox,
            los_axis='z'
        )
        deltasigma_method3.append(ds_method3)

        elapsed_method3 = time.time() - start
        print(f"    Method 3: {elapsed_method3:.3f} s")
        print()

    elapsed_total = time.time() - start_total
    print_timing("Total computation time", elapsed_total)

    n_success = len(deltasigma_method1)
    print(f"\n  Successfully processed {n_success}/{n_halos} halos")

    if n_success == 0:
        print("\n  ❌ No halos processed successfully!")
        return

    # Convert to arrays
    deltasigma_method1 = np.array(deltasigma_method1)  # shape (n_success, n_rp_bins)
    deltasigma_method2 = np.array(deltasigma_method2)
    deltasigma_method3 = np.array(deltasigma_method3)
    xigm_profiles = np.array(xigm_profiles)  # shape (n_success, n_r_bins)



    # ========================================================================
    # Step 5: Aggregate Results (Mean and Scatter)
    # ========================================================================
    print_header("Step 5: Aggregate Results Across Halos")

    # Compute statistics
    ds1_mean = np.mean(deltasigma_method1, axis=0)
    ds1_std = np.std(deltasigma_method1, axis=0)
    ds1_sem = ds1_std / np.sqrt(n_success)

    ds2_mean = np.mean(deltasigma_method2, axis=0)
    ds2_std = np.std(deltasigma_method2, axis=0)
    ds2_sem = ds2_std / np.sqrt(n_success)

    ds3_mean = np.mean(deltasigma_method3, axis=0)
    ds3_std = np.std(deltasigma_method3, axis=0)
    ds3_sem = ds3_std / np.sqrt(n_success)

    print(f"\n  Aggregated over {n_success} halos:")
    print(f"\n  Method 1 (Standard):")
    print(f"    Mean ΔΣ range: [{ds1_mean.min():.2e}, {ds1_mean.max():.2e}] Msun h/pc²")
    print(f"    Mean scatter: {np.mean(ds1_std):.2e} Msun h/pc²")

    print(f"\n  Method 2 (Spherical):")
    print(f"    Mean ΔΣ range: [{ds2_mean.min():.2e}, {ds2_mean.max():.2e}] Msun h/pc²")
    print(f"    Mean scatter: {np.mean(ds2_std):.2e} Msun h/pc²")

    print(f"\n  Method 3 (Cylindrical 2D):")
    print(f"    Mean ΔΣ range: [{ds3_mean.min():.2e}, {ds3_mean.max():.2e}] Msun h/pc²")
    print(f"    Mean scatter: {np.mean(ds3_std):.2e} Msun h/pc²")

    # ========================================================================
    # Step 6: Compare Methods
    # ========================================================================
    print_header("Step 6: Method Comparison")

    # Relative differences (vs Method 1 as reference)
    rel_diff_2vs1 = (ds2_mean - ds1_mean) / np.abs(ds1_mean) * 100
    rel_diff_3vs1 = (ds3_mean - ds1_mean) / np.abs(ds1_mean) * 100

    print(f"\n  📊 Detailed Comparison (Method 1 = Reference):")
    print(f"    {'rp [Mpc/h]':<12s} {'Method 1':<14s} {'Method 2':<14s} {'Method 3':<14s} "
          f"{'M2 err[%]':<12s} {'M3 err[%]':<12s}")
    print(f"    {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")

    for i, rp in enumerate(rp_centers):
        print(f"    {rp:11.4f}  {ds1_mean[i]:13.4e}  {ds2_mean[i]:13.4e}  {ds3_mean[i]:13.4e}  "
              f"{rel_diff_2vs1[i]:+11.2f}  {rel_diff_3vs1[i]:+11.2f}")

    # Statistical summary
    abs_rel_diff_2vs1 = np.abs(rel_diff_2vs1)
    abs_rel_diff_3vs1 = np.abs(rel_diff_3vs1)

    print(f"\n  📈 Statistical Summary:")
    print(f"\n  Method 2 vs Method 1:")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_2vs1):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_2vs1):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_2vs1):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_2vs1**2)):6.2f}%")

    print(f"\n  Method 3 vs Method 1:")
    print(f"    Mean absolute relative difference:   {np.mean(abs_rel_diff_3vs1):6.2f}%")
    print(f"    Median absolute relative difference: {np.median(abs_rel_diff_3vs1):6.2f}%")
    print(f"    Max absolute relative difference:    {np.max(abs_rel_diff_3vs1):6.2f}%")
    print(f"    RMS relative difference:             {np.sqrt(np.mean(rel_diff_3vs1**2)):6.2f}%")

    # ========================================================================
    # Step 7: Save Results
    # ========================================================================
    print_header("Step 7: Save Results")

    output_file = output_dir / 'three_method_diagnostic.npz'

    np.savez(
        output_file,
        # Radial bins
        rp_bins=rp_bins,
        rp_centers=rp_centers,
        r_bins=r_bins,
        r_centers=r_centers,
        # Individual halo profiles
        deltasigma_method1=deltasigma_method1,
        deltasigma_method2=deltasigma_method2,
        deltasigma_method3=deltasigma_method3,
        xigm_profiles=xigm_profiles,
        # Aggregated profiles
        ds1_mean=ds1_mean,
        ds1_std=ds1_std,
        ds1_sem=ds1_sem,
        ds2_mean=ds2_mean,
        ds2_std=ds2_std,
        ds2_sem=ds2_sem,
        ds3_mean=ds3_mean,
        ds3_std=ds3_std,
        ds3_sem=ds3_sem,
        # Comparisons
        rel_diff_2vs1=rel_diff_2vs1,
        rel_diff_3vs1=rel_diff_3vs1,
        # Metadata
        n_halos=n_success,
        halo_masses=halo_masses[:n_success],
        halo_rvir=halo_rvir[:n_success],
        search_radius=search_radius,
        chi_max=chi_max,
        downsample_factor=downsample_factor
    )

    print(f"\n  💾 Results saved to: {output_file.name}")
    print(f"    File size: {output_file.stat().st_size / 1e6:.2f} MB")

    # ========================================================================
    # Step 8: Create Diagnostic Plots
    # ========================================================================
    print_header("Step 8: Create Diagnostic Plots")

    print(f"\n  Creating visualization plots...")

    # Plot 1: Individual Halo Profiles (5 most massive)
    # ------------------------------------------------------------------
    n_plot_halos = min(5, n_success)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_plot_halos):
        ax = axes[i]

        # Plot all three methods
        ax.loglog(rp_centers, deltasigma_method1[i], 'o-', label='Method 1 (Standard)',
                 color='blue', markersize=4, linewidth=2)
        ax.loglog(rp_centers, deltasigma_method2[i], 's--', label='Method 2 (Spherical)',
                 color='red', markersize=4, linewidth=1.5, alpha=0.8)
        ax.loglog(rp_centers, deltasigma_method3[i], '^:', label='Method 3 (Cylindrical)',
                 color='green', markersize=4, linewidth=1.5, alpha=0.8)

        ax.set_xlabel(r'$r_p$ [Mpc/h]', fontsize=10)
        ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]', fontsize=10)
        ax.set_title(f'Halo {i+1}: M = {halo_masses[i]:.2e} M$_\\odot$/h\n'
                    f'R_vir = {halo_rvir[i]:.2f} Mpc/h', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='best')

    # Remove extra subplot if we have fewer than 6 halos
    if n_plot_halos < 6:
        for i in range(n_plot_halos, 6):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plot1_file = output_dir / 'diagnostic_individual_halos.png'
    plt.savefig(plot1_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Individual halo profiles: {plot1_file.name}")

    # Plot 2: Stacked Mean Profile Comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.errorbar(rp_centers, ds1_mean, yerr=ds1_sem, fmt='o-',
               label='Method 1 (Standard)', color='blue',
               markersize=6, linewidth=2, capsize=3, capthick=1.5)
    ax.errorbar(rp_centers, ds2_mean, yerr=ds2_sem, fmt='s--',
               label='Method 2 (Spherical)', color='red',
               markersize=6, linewidth=1.5, capsize=3, capthick=1.5, alpha=0.8)
    ax.errorbar(rp_centers, ds3_mean, yerr=ds3_sem, fmt='^:',
               label='Method 3 (Cylindrical)', color='green',
               markersize=6, linewidth=1.5, capsize=3, capthick=1.5, alpha=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r_p$ [Mpc/h]', fontsize=14)
    ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]', fontsize=14)
    ax.set_title(f'Stacked Mean $\Delta\Sigma$ Profile\n'
                f'Averaged over {n_success} most massive halos', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    plot2_file = output_dir / 'diagnostic_stacked_mean.png'
    plt.savefig(plot2_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Stacked mean profile: {plot2_file.name}")

    # Plot 3: Relative Error Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rp_centers, rel_diff_2vs1, 'o-', label='Method 2 vs Method 1',
           color='red', markersize=6, linewidth=2)
    ax.plot(rp_centers, rel_diff_3vs1, 's--', label='Method 3 vs Method 1',
           color='green', markersize=6, linewidth=2)

    # Reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhspan(-5, 5, color='gray', alpha=0.2, label='±5% range')

    ax.set_xscale('log')
    ax.set_xlabel(r'$r_p$ [Mpc/h]', fontsize=14)
    ax.set_ylabel('Relative Difference [%]', fontsize=14)
    ax.set_title('Method Comparison: Relative Differences vs Standard Method', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    plot3_file = output_dir / 'diagnostic_relative_errors.png'
    plt.savefig(plot3_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Relative error plot: {plot3_file.name}")

    # Plot 4: ξ_gm Profiles (for reference)
    # ------------------------------------------------------------------
    xigm_mean = np.mean(xigm_profiles, axis=0)
    xigm_std = np.std(xigm_profiles, axis=0)
    xigm_sem = xigm_std / np.sqrt(n_success)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.errorbar(r_centers, xigm_mean, yerr=xigm_sem, fmt='o-',
               color='purple', markersize=6, linewidth=2,
               capsize=3, capthick=1.5, label='Mean $\\xi_{gm}(r)$')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r$ [Mpc/h]', fontsize=14)
    ax.set_ylabel(r'$\xi_{gm}(r)$', fontsize=14)
    ax.set_title(f'Galaxy-Matter Correlation Function\n'
                f'Averaged over {n_success} most massive halos', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    plot4_file = output_dir / 'diagnostic_xigm_profile.png'
    plt.savefig(plot4_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ ξ_gm profile: {plot4_file.name}")

    print(f"\n  📊 All plots saved to: {output_dir}/")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Summary")

    total_time = (elapsed_load_halos + elapsed_load_particles + elapsed_kdtree + elapsed_total)

    print(f"\n  Total runtime: {total_time:.1f} s")
    print(f"  Time per halo: {elapsed_total/n_success:.2f} s")

    # Verdict
    mean_error_2vs1 = np.mean(abs_rel_diff_2vs1)
    mean_error_3vs1 = np.mean(abs_rel_diff_3vs1)

    def get_verdict(error):
        if error < 1.0:
            return "✓ EXCELLENT agreement"
        elif error < 5.0:
            return "✓ VERY GOOD agreement"
        elif error < 10.0:
            return "✓ GOOD agreement"
        elif error < 20.0:
            return "⚠ MODERATE agreement"
        else:
            return "✗ POOR agreement"

    print(f"\n  Verdict:")
    print(f"    Method 2 vs Method 1: {get_verdict(mean_error_2vs1)}")
    print(f"      Mean error: {mean_error_2vs1:.2f}%")
    print(f"\n    Method 3 vs Method 1: {get_verdict(mean_error_3vs1)}")
    print(f"      Mean error: {mean_error_3vs1:.2f}%")

    print(f"\n  Interpretation:")
    if mean_error_2vs1 < 5.0 and mean_error_3vs1 < 5.0:
        print(f"    ✓ All three methods agree well on massive halos!")
        print(f"    → Issue in test_direct_central_validation.py is likely:")
        print(f"      - Poor statistics from low target_ngal")
        print(f"      - Stacking/averaging artifacts")
        print(f"      - Cosmic variance in galaxy selection")
    elif mean_error_2vs1 > 20.0 or mean_error_3vs1 > 20.0:
        print(f"    ⚠ Methods disagree significantly!")
        print(f"    → Check implementation of methods with large errors")
    else:
        print(f"    ~ Methods show moderate differences")
        print(f"    → May be due to finite particle sampling or method differences")

    print("\n" + "="*70 + "\n")

    return (rp_centers, ds1_mean, ds2_mean, ds3_mean,
            deltasigma_method1, deltasigma_method2, deltasigma_method3)


if __name__ == "__main__":
    # Run diagnostic on 10 most massive halos
    main(n_halos=10, downsample_factor=2, search_radius=150.0, chi_max=150.0)
