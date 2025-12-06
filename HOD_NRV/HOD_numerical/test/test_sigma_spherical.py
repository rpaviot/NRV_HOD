"""
Simple diagnostic: Compute Sigma via spherical profile + projection.

Method:
1. Compute 3D density profile ρ(r) in spherical shells around halo center
2. Project to get Σ(rp) using Abel transform: Σ(rp) = 2 ∫_{rp}^{∞} ρ(r) r/√(r²-rp²) dr
3. Compute ΔΣ(rp) = Σ̄(<rp) - Σ(rp)

This should give positive ΔΣ around halos if the method is correct.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.interpolate import interp1d

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    periodic_distance,
    compute_rho_profile_spherical,
    project_rho_to_sigma,
    compute_delta_sigma_from_sigma
)
from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration


def main():
    print("="*70)
    print("  Spherical Shell Method: ρ(r) → Σ(rp) → ΔΣ(rp)")
    print("="*70)

    # ========================================================================
    # Configuration
    # ========================================================================
    halo_path = '/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet'
    particle_path = '/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet'

    Lbox = 681  # Mpc/h
    Om0 = 0.3089
    rho_crit = 2.77536627e11  # h^2 Msun / Mpc^3
    RHO_M = Om0 * rho_crit

    # Radial bins for 3D profile
    r_bins_3d = np.logspace(-2, 1.5, 50)  # 0.01 to ~31.6 Mpc/h (fine sampling)

    # Projected bins for Σ(rp)
    rp_bins = np.logspace(-1, 1.5, 20)  # 0.1 to ~31.6 Mpc/h

    # Downsampling (reduce for better statistics)
    downsample_factor = 10  # Was 40, but need more particles to reduce shot noise

    print(f"\nConfiguration:")
    print(f"  Box size: {Lbox} Mpc/h")
    print(f"  RHO_M: {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"  3D bins: {len(r_bins_3d)-1} bins from {r_bins_3d[0]:.2f} to {r_bins_3d[-1]:.2f} Mpc/h")
    print(f"  Projected bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"  Particle downsampling: {downsample_factor}×")

    # ========================================================================
    # Load data
    # ========================================================================
    print(f"\nLoading catalogs...")
    df_halos = pd.read_parquet(halo_path)
    df_particles = pd.read_parquet(particle_path)

    # Downsample
    df_particles = df_particles.iloc[::downsample_factor].reset_index(drop=True)
    n_particles = len(df_particles)

    print(f"  Halos: {len(df_halos):,}")
    print(f"  Particles: {n_particles:,}")

    # Extract positions
    halo_positions = df_halos[['x', 'y', 'z']].values
    halo_rvir = df_halos['rvir'].values / 1000.0  # kpc/h → Mpc/h
    halo_mass = df_halos['mass'].values
    particle_positions = df_particles[['x', 'y', 'z']].values

    # Particle mass (use ORIGINAL particle count, not downsampled)
    n_particles_original = len(df_particles) * downsample_factor
    V_box = Lbox**3
    m_particle_true = RHO_M * V_box / n_particles_original

    print(f"\nParticle mass calculation:")
    print(f"  Original N_particles: {n_particles_original:,}")
    print(f"  Downsampled N_particles: {n_particles:,}")
    print(f"  True m_particle: {m_particle_true:.3e} Msun/h")
    print(f"  Downsampled weight: {m_particle_true * downsample_factor:.3e} Msun/h (used in calculation)")

    # Use downsampled particle mass for calculation (each particle represents 40 particles)
    m_particle = m_particle_true * downsample_factor

    # ========================================================================
    # Build KDTree
    # ========================================================================
    print(f"\nBuilding KD-tree...")
    kdtree = build_particle_kdtree(particle_positions, Lbox)

    # ========================================================================
    # Select test halos
    # ========================================================================
    n_test_halos = 5
    np.random.seed(42)

    log_mass = np.log10(halo_mass)
    mass_percentiles = [20, 40, 60, 80, 95]

    test_indices = []
    for percentile in mass_percentiles:
        mass_threshold = np.percentile(log_mass, percentile)
        candidates = np.where(np.abs(log_mass - mass_threshold) < 0.1)[0]
        if len(candidates) > 0:
            test_indices.append(np.random.choice(candidates))

    print(f"\nSelected {len(test_indices)} test halos:")
    for i, idx in enumerate(test_indices):
        pos = halo_positions[idx]
        rvir = halo_rvir[idx]
        mass = halo_mass[idx]
        print(f"  Halo {i+1}: M={mass:.2e} Msun/h, Rvir={rvir:.3f} Mpc/h")

    # ========================================================================
    # Compute profiles
    # ========================================================================
    print(f"\nComputing ρ(r) → Σ(rp) → ΔΣ(rp) for each halo...\n")

    fig, axes = plt.subplots(3, len(test_indices), figsize=(4*len(test_indices), 12))

    results = []

    for i, idx in enumerate(test_indices):
        halo_pos = halo_positions[idx]
        rvir = halo_rvir[idx]
        mass = halo_mass[idx]

        # Query particles within maximum bin radius
        search_radius = r_bins_3d[-1]
        nearby_indices = kdtree.query_ball_point(halo_pos, r=search_radius)
        nearby_particles = particle_positions[nearby_indices]

        print(f"  Halo {i+1} (M={mass:.2e} Msun/h, Rvir={rvir:.3f} Mpc/h):")
        print(f"    Nearby particles: {len(nearby_indices):,}")

        # Step 1: Compute 3D density profile ρ(r)
        r_centers, rho, counts = compute_rho_profile_spherical(
            halo_pos, nearby_particles, r_bins_3d, m_particle, Lbox
        )

        print(f"    ρ(r) range: [{rho.min():.3e}, {rho.max():.3e}] Msun/h / (Mpc/h)^3")
        print(f"    ρ(r) / RHO_M range: [{rho.min()/RHO_M:.2f}, {rho.max()/RHO_M:.2f}]")

        # Step 2: Project to Σ(rp)
        rp_centers, sigma = project_rho_to_sigma(r_centers, rho, rp_bins)

        print(f"    Σ(rp) range: [{sigma.min():.3e}, {sigma.max():.3e}] Msun/h / (Mpc/h)^2")

        # Step 3: Compute ΔΣ(rp)
        delta_sigma, sigma_mean = compute_delta_sigma_from_sigma(rp_centers, sigma, rp_bins)

        print(f"    ΔΣ(rp) range: [{delta_sigma.min():.3e}, {delta_sigma.max():.3e}] Msun/h / (Mpc/h)^2")
        print(f"    ΔΣ positive fraction: {(delta_sigma > 0).sum()} / {len(delta_sigma)}")

        results.append((r_centers, rho, rp_centers, sigma, delta_sigma, sigma_mean, mass, rvir))

        # Plot ρ(r) - Row 1
        ax = axes[0, i]
        ax.loglog(r_centers, rho, 'o-', linewidth=2, label='ρ(r)')
        ax.axhline(RHO_M, color='red', linestyle='--', linewidth=2, label=r'$\bar{\rho}_m$')
        ax.axvline(rvir, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Rvir')
        ax.set_xlabel('r [Mpc/h]', fontsize=11)
        ax.set_ylabel(r'$\rho(r)$ [M$_\odot$ h / (Mpc/h)$^3$]', fontsize=11)
        ax.set_title(f'Halo {i+1}: ρ(r)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot Σ(rp) - Row 2
        ax = axes[1, i]
        ax.loglog(rp_centers, sigma, 's-', linewidth=2, label='Σ(rp)')
        ax.axvline(rvir, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_xlabel('rp [Mpc/h]', fontsize=11)
        ax.set_ylabel(r'$\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
        ax.set_title(f'Halo {i+1}: Σ(rp)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot ΔΣ(rp) - Row 3
        ax = axes[2, i]
        ax.plot(rp_centers, delta_sigma, 'o-', linewidth=2, label='ΔΣ(rp)', color='blue')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(rvir, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_xlabel('rp [Mpc/h]', fontsize=11)
        ax.set_ylabel(r'$\Delta\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
        ax.set_title(f'Halo {i+1}: ΔΣ(rp)', fontsize=10)
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent
    output_path = output_dir / 'sigma_spherical_method.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")

    print("\n" + "="*70)
    print("  Test completed!")
    print("="*70)


if __name__ == "__main__":
    main()
