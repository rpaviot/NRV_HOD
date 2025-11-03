"""
Diagnostic test: Compute Σ(r) and ΔΣ(r) around actual halo centers.

This test computes surface density profiles around real halo centers
(within 3×Rvir) using the same number of particles as the original test,
but focused on halo-centric regions where the signal is strongest.

If profiles look good around halos → interpolation is the problem
If profiles look bad around halos → pre-computation method is the problem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_at_position
)


def compute_sigma_at_position(
    position: np.ndarray,
    nearby_particles: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    chi_max: float = 150.0,
    particle_mass: float = None,
    n_particles_total: int = None,
    Lbox: float = None,
    los_axis: str = 'z'
) -> tuple:
    """
    Compute Sigma(rp) and DeltaSigma(rp) at a single position.

    Returns
    -------
    rp_centers : np.ndarray
        Center of radial bins [Mpc/h]
    sigma : np.ndarray
        Surface density [Msun/h / (Mpc/h)²]
    delta_sigma : np.ndarray
        Excess surface density [Msun/h / (Mpc/h)²]
    counts : np.ndarray
        Number of particles in each bin
    m_particle : float
        Particle mass used [Msun/h]
    """
    # Determine axis indices
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    los_idx = axis_map[los_axis.lower()]
    perp_idx = [i for i in range(3) if i != los_idx]

    # Compute displacement vectors
    displacement = nearby_particles - position

    # Filter particles within chi_max along line-of-sight
    chi = np.abs(displacement[:, los_idx])
    mask_los = chi <= chi_max

    # Compute projected separation (perpendicular to line-of-sight)
    rp = np.sqrt(displacement[:, perp_idx[0]]**2 + displacement[:, perp_idx[1]]**2)

    # Apply line-of-sight filter
    rp_filtered = rp[mask_los]

    # Histogram particles in projected radial bins
    counts, _ = np.histogram(rp_filtered, bins=rp_bins)

    # Determine particle mass
    if particle_mass is not None:
        m_particle = particle_mass
    elif n_particles_total is not None and Lbox is not None:
        V_box = Lbox**3
        m_particle = RHO_M * V_box / n_particles_total
    else:
        m_particle = 1.0

    # Compute annulus areas [Mpc²/h²]
    area_annulus = np.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2)

    # Compute surface density Σ(rp) [Msun/h / (Mpc/h)²]
    # counts = number of particles in cylindrical annulus with |chi| <= chi_max
    # Surface density Σ = (counts × m_particle) / area_annulus
    sigma = (counts * m_particle) / area_annulus

    # Compute mean surface density within each radius (Σ̄(<rp))
    cumulative_counts = np.cumsum(counts)
    cumulative_area = np.pi * rp_bins[1:]**2
    sigma_bar = (cumulative_counts * m_particle) / cumulative_area

    # Compute excess surface density ΔΣ(rp) = Σ̄(<rp) - Σ(rp)
    delta_sigma = sigma_bar - sigma

    # Radial bin centers
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

    return rp_centers, sigma, delta_sigma, counts, m_particle


def main():
    print("="*70)
    print("  Diagnostic Test: Sigma(r) and DeltaSigma(r) Around Halo Centers")
    print("="*70)

    # ========================================================================
    # Configuration
    # ========================================================================
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
    rho_crit = 2.77536627e11  # h^2 Msun / Mpc^3
    RHO_M = cosmo_params['Om0'] * rho_crit

    # Radial bins for Σ(rp) computation
    rp_bins = np.logspace(-1, 1.5, 20)  # 0.1 to ~31.6 Mpc/h

    # Line-of-sight integration limit
    chi_max = 150.0  # Mpc/h

    # Halo selection: use particles within r_factor × Rvir
    r_factor = 3.0  # Search within 3×Rvir of halo centers

    # Downsampling (same as test_fast_dsigma.py)
    downsample_factor = 40

    print(f"\nConfiguration:")
    print(f"  Box size: {Lbox} Mpc/h")
    print(f"  RHO_M: {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"  rho_crit: {rho_crit:.3e} h^2 Msun / Mpc^3")
    print(f"  Omega_m: {cosmo_params['Om0']:.4f}")
    print(f"  Search radius: {r_factor}×Rvir around halo centers")
    print(f"  χ_max: {chi_max} Mpc/h")
    print(f"  rp bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"  Particle downsampling: {downsample_factor}×")

    # ========================================================================
    # Load catalogs
    # ========================================================================
    print(f"\nLoading halo catalog...")
    df_halos = pd.read_parquet(halo_path)
    print(f"  Loaded {len(df_halos):,} halos")

    print(f"\nLoading particle catalog...")
    df_particles = pd.read_parquet(particle_path)
    print(f"  Loaded {len(df_particles):,} particles (already downsampled)")

    # Additional downsampling (same as test_fast_dsigma.py)
    df_particles = df_particles.iloc[::downsample_factor].reset_index(drop=True)
    n_particles = len(df_particles)
    print(f"  Further downsampled to {n_particles:,} particles")

    # Extract positions
    halo_positions = df_halos[['x', 'y', 'z']].values
    halo_rvir = df_halos['rvir'].values / 1000.0  # Convert kpc/h → Mpc/h
    halo_mass = df_halos['mass'].values
    particle_positions = df_particles[['x', 'y', 'z']].values

    print(f"\n  Data shapes:")
    print(f"    Halo positions: {halo_positions.shape}")
    print(f"    Halo Rvir range: [{halo_rvir.min():.3f}, {halo_rvir.max():.3f}] Mpc/h (converted from kpc/h)")
    print(f"    Halo mass range: [{halo_mass.min():.2e}, {halo_mass.max():.2e}] Msun/h")
    print(f"    Particle positions: {particle_positions.shape}")

    # Compute particle mass
    V_box = Lbox**3
    m_particle = RHO_M * V_box / n_particles
    print(f"\nParticle mass calculation:")
    print(f"  V_box = {V_box:.3e} (Mpc/h)^3")
    print(f"  Total mass = RHO_M × V_box = {RHO_M * V_box:.3e} Msun/h")
    print(f"  m_particle = Total_mass / N_particles = {m_particle:.3e} Msun/h")

    # ========================================================================
    # Build KD-tree
    # ========================================================================
    print(f"\nBuilding KD-tree...")
    kdtree = build_particle_kdtree(particle_positions, Lbox)

    # ========================================================================
    # Select test halos (sample from different mass bins)
    # ========================================================================
    n_test_halos = 5
    np.random.seed(42)

    # Select halos from different mass ranges
    log_mass = np.log10(halo_mass)
    mass_percentiles = [20, 40, 60, 80, 95]  # Sample from different mass ranges

    test_indices = []
    for percentile in mass_percentiles:
        mass_threshold = np.percentile(log_mass, percentile)
        # Find halos near this mass
        candidates = np.where(np.abs(log_mass - mass_threshold) < 0.1)[0]
        if len(candidates) > 0:
            test_indices.append(np.random.choice(candidates))

    n_test_halos = len(test_indices)

    print(f"\nSelected {n_test_halos} test halos from different mass bins:")
    for i, idx in enumerate(test_indices):
        pos = halo_positions[idx]
        rvir = halo_rvir[idx]
        mass = halo_mass[idx]
        print(f"  Halo {i+1}: idx={idx}, M={mass:.2e} Msun/h, Rvir={rvir:.3f} Mpc/h, "
              f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # ========================================================================
    # Compute Sigma(r) and DeltaSigma(r) at halo centers
    # ========================================================================
    print(f"\nComputing Sigma(rp) and DeltaSigma(rp) at halo centers...")
    print(f"  (Using particles within {r_factor}×Rvir for each halo)\n")

    # Create figure with 4 rows and 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()

    # Store results for later plotting
    results = []

    for i, idx in enumerate(test_indices):
        halo_pos = halo_positions[idx]
        rvir = halo_rvir[idx]
        mass = halo_mass[idx]

        # Query nearby particles - use max rp bin to avoid bias
        # (searching too far without binning causes underestimated Sigma_bar)
        search_radius = rp_bins[-1] + chi_max  # Max projected + LOS distance
        nearby_indices = kdtree.query_ball_point(halo_pos, r=search_radius)
        nearby_particles = particle_positions[nearby_indices]

        print(f"  Halo {i+1} (M={mass:.2e} Msun/h, Rvir={rvir:.3f} Mpc/h):")
        print(f"    Position: ({halo_pos[0]:.2f}, {halo_pos[1]:.2f}, {halo_pos[2]:.2f}) Mpc/h")
        print(f"    Search radius: {r_factor}×{rvir:.3f} = {search_radius:.3f} Mpc/h")
        print(f"    Nearby particles: {len(nearby_indices):,}")

        # Compute Sigma(rp) and DeltaSigma(rp) at halo center
        rp_centers, sigma, delta_sigma, counts, m_part = compute_sigma_at_position(
            position=halo_pos,
            nearby_particles=nearby_particles,
            RHO_M=RHO_M,
            rp_bins=rp_bins,
            chi_max=chi_max,
            particle_mass=m_particle,
            n_particles_total=n_particles,
            Lbox=Lbox
        )

        # Store results with halo info
        results.append((rp_centers, sigma, delta_sigma, counts, m_part, mass, rvir))

        # Convert to Msun h/pc²
        sigma_pc2 = sigma / 1e12
        delta_sigma_pc2 = delta_sigma / 1e12

        print(f"    Sigma range: [{sigma.min():.3e}, {sigma.max():.3e}] Msun/h / (Mpc/h)²")
        print(f"    Sigma range: [{sigma_pc2.min():.3e}, {sigma_pc2.max():.3e}] Msun h/pc²")
        print(f"    DeltaSigma range: [{delta_sigma.min():.3e}, {delta_sigma.max():.3e}] Msun/h / (Mpc/h)²")
        print(f"    DeltaSigma range: [{delta_sigma_pc2.min():.3e}, {delta_sigma_pc2.max():.3e}] Msun h/pc²")
        print(f"    Particle counts in bins: min={counts.min()}, max={counts.max()}, total={counts.sum()}")
        print(f"    Particle mass used: {m_part:.3e} Msun/h")

        # Plot Sigma(r) - First row
        ax = axes[i]
        ax.loglog(rp_centers, sigma, 'o-', label=f'Halo {i+1} - Σ(rp)', linewidth=2)
        ax.axhline(RHO_M, color='red', linestyle='--', linewidth=2, label=r'$\bar{\rho}_m$')
        ax.axvline(rvir, color='green', linestyle=':', linewidth=2, label=f'Rvir={rvir:.2f}')
        ax.set_xlabel('rp [Mpc/h]', fontsize=11)
        ax.set_ylabel(r'$\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
        ax.set_title(f'Halo {i+1}: M={mass:.2e} Msun/h', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot DeltaSigma for each test halo - Second row
    for i, (idx, result) in enumerate(zip(test_indices, results)):
        rp_centers, sigma, delta_sigma, counts, _, mass, rvir = result
        # Plot DeltaSigma(r)
        ax = axes[i + 6]
        # Plot actual values (not absolute) - should be positive around halo centers
        ax.plot(rp_centers, delta_sigma, 'o-', color='blue', linewidth=2, label='ΔΣ(rp)')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(rvir, color='green', linestyle=':', linewidth=2, alpha=0.5, label=f'Rvir={rvir:.2f}')
        ax.set_xlabel('rp [Mpc/h]', fontsize=11)
        ax.set_ylabel(r'$\Delta\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
        ax.set_title(f'Halo {i+1}: ΔΣ(rp)', fontsize=10)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot all Sigma together - Row 3, left
    ax = axes[9]
    for i, result in enumerate(results):
        rp_centers, sigma, delta_sigma, counts, _, mass, rvir = result
        ax.loglog(rp_centers, sigma, 'o-', alpha=0.7, label=f'Halo {i+1} (M={mass:.1e})')

    ax.axhline(RHO_M, color='red', linestyle='--', linewidth=2, label=r'$\bar{\rho}_m$ (mean)')
    ax.set_xlabel('rp [Mpc/h]', fontsize=11)
    ax.set_ylabel(r'$\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
    ax.set_title('All Test Halos - Σ(rp)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot all DeltaSigma together - Row 3, middle
    ax = axes[10]
    for i, result in enumerate(results):
        rp_centers, sigma, delta_sigma, counts, _, mass, rvir = result
        ax.plot(rp_centers, delta_sigma, 'o-', alpha=0.7, linewidth=2, label=f'Halo {i+1}')

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('rp [Mpc/h]', fontsize=11)
    ax.set_ylabel(r'$\Delta\Sigma(r_p)$ [M$_\odot$ h / (Mpc/h)$^2$]', fontsize=11)
    ax.set_title('All Test Halos - ΔΣ(rp)', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in [5, 11]:
        axes[i].axis('off')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    output_path = output_dir / 'sigma_diagnostic_halo_centers.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plot to: {output_path}")

    # plt.show()  # Comment out for batch runs

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "="*70)
    print("  Summary Statistics")
    print("="*70)

    print(f"\n  Cosmological parameters:")
    print(f"    RHO_M = {RHO_M:.3e} Msun/h / (Mpc/h)^3")
    print(f"    Expected Sigma at large scales ~ RHO_M × 2 × chi_max")
    print(f"                                  = {RHO_M * 2 * chi_max:.3e} Msun/h / (Mpc/h)²")
    print(f"                                  = {RHO_M * 2 * chi_max / 1e12:.3e} Msun h/pc²")

    print(f"\n  Particle mass:")
    print(f"    m_particle = {m_particle:.3e} Msun/h")
    print(f"    For reference, M_halo ~ 10^12 Msun/h would contain:")
    print(f"      N_particles ~ {1e12 / m_particle:.1f} particles")

    print("\n" + "="*70)
    print("  Diagnostic test completed!")
    print("="*70)


if __name__ == "__main__":
    main()
