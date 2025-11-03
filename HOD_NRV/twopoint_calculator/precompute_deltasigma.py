"""
Pre-computation utilities for fast galaxy-galaxy lensing calculations.

This module handles the expensive one-time computation of ΔΣ at particle
positions near halos, which can then be used for fast interpolation during
HOD parameter sampling.

References
----------
.. [1] Yuan et al. (2021), MNRAS, arXiv:2110.11412 (AbacusHOD paper)
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional
from HOD_NRV.utils.utils_functions import gauss_legendre_integration


def build_particle_kdtree(positions_part: np.ndarray,
                         Lbox: float) -> cKDTree:
    """
    Build KD-tree from particle catalog for efficient neighbor queries.

    Parameters
    ----------
    positions_part : np.ndarray, shape (N_particles, 3)
        Particle positions [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h] for periodic boundary conditions

    Returns
    -------
    kdtree : scipy.spatial.cKDTree
        KD-tree with periodic boundary conditions

    Examples
    --------
    >>> positions = np.random.rand(10000, 3) * 1000  # 1000 Mpc/h box
    >>> kdtree = build_particle_kdtree(positions, Lbox=1000.0)
    >>> # Query neighbors within 5 Mpc/h of a point
    >>> indices = kdtree.query_ball_point([500, 500, 500], r=5.0)
    """
    # Build KD-tree with periodic boundary conditions
    kdtree = cKDTree(positions_part, boxsize=Lbox)
    return kdtree


def periodic_distance(pos1: np.ndarray, pos2: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Compute distance with periodic boundary conditions.

    Parameters
    ----------
    pos1 : np.ndarray, shape (N, 3) or (3,)
        First set of positions [Mpc/h]
    pos2 : np.ndarray, shape (3,)
        Second position (typically a center) [Mpc/h]
    boxsize : float
        Simulation box size [Mpc/h]

    Returns
    -------
    distances : np.ndarray
        Distances accounting for periodic boundaries [Mpc/h]

    Examples
    --------
    >>> pos1 = np.array([[1, 1, 1], [99, 99, 99]])
    >>> pos2 = np.array([2, 2, 2])
    >>> d = periodic_distance(pos1, pos2, boxsize=100.0)
    >>> # Distance to [99,99,99] wraps around: ~4.24 instead of ~169
    """
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > boxsize/2, boxsize - delta, delta)
    return np.sqrt(np.sum(delta**2, axis=-1))


def compute_rho_profile_spherical(
    center_pos: np.ndarray,
    nearby_particles: np.ndarray,
    r_bins: np.ndarray,
    particle_mass: float,
    boxsize: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D density profile ρ(r) in spherical shells around a center.

    Parameters
    ----------
    center_pos : np.ndarray, shape (3,)
        Center position [Mpc/h]
    nearby_particles : np.ndarray, shape (N, 3)
        Particle positions [Mpc/h]
    r_bins : np.ndarray
        Radial bin edges [Mpc/h]
    particle_mass : float
        Mass per particle [Msun/h]
    boxsize : float
        Box size for periodic boundaries [Mpc/h]

    Returns
    -------
    r_centers : np.ndarray
        Bin centers [Mpc/h]
    rho : np.ndarray
        Density in each shell [Msun/h / (Mpc/h)^3]
    counts : np.ndarray
        Number of particles in each shell

    Examples
    --------
    >>> r_bins = np.logspace(-2, 1.5, 50)
    >>> r_cen, rho, counts = compute_rho_profile_spherical(
    ...     center_pos, particles, r_bins, m_particle, Lbox
    ... )
    """
    # Compute 3D distances with periodic boundaries
    r = periodic_distance(nearby_particles, center_pos, boxsize)

    # Histogram in spherical shells
    counts, _ = np.histogram(r, bins=r_bins)

    # Shell volumes
    shell_volumes = (4.0/3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)

    # Density = mass / volume
    rho = (counts * particle_mass) / shell_volumes
    # Bin centers
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])

    return r_centers, rho, counts


def project_rho_to_sigma_abel(
    r_profile: np.ndarray,
    rho_profile: np.ndarray,
    rp_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D density ρ(r) to 2D surface density Σ(rp) using Abel transform.

    .. deprecated::
        This Abel transform method assumes perfect spherical symmetry and can
        produce systematic errors. Use :func:`project_rho_to_sigma_cylindrical`
        instead, which matches the standard ξ_gm pipeline methodology.

    The projection is: Σ(rp) = 2 ∫_{rp}^{r_max} ρ(r) r / √(r² - rp²) dr

    Parameters
    ----------
    r_profile : np.ndarray
        3D radial coordinates [Mpc/h]
    rho_profile : np.ndarray
        3D density profile [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected radial bin edges [Mpc/h]

    Returns
    -------
    rp_centers : np.ndarray
        Projected bin centers [Mpc/h]
    sigma : np.ndarray
        Surface density [Msun/h / (Mpc/h)^2]

    Notes
    -----
    This uses the Abel transform to project a spherically symmetric 3D
    density profile into a 2D projected surface density.

    Uses Gauss-Legendre quadrature for fast and accurate integration.

    **Warning**: This method assumes perfect spherical symmetry, which does not
    exist in real particle distributions. It can lead to systematic biases
    at small (underestimation) and large (overestimation) projected radii.

    References
    ----------
    .. [1] Bracewell, R. N. (2000), "The Fourier Transform and Its Applications"
    """
    # Create interpolator for ρ(r)
    rho_interp = interp1d(r_profile, rho_profile, kind='cubic',
                          bounds_error=False, fill_value=0.0)

    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    sigma = np.zeros(len(rp_centers))

    r_max = r_profile[-1]

    for i, rp in enumerate(rp_centers):
        if rp >= r_max:
            sigma[i] = 0.0
            continue

        # Abel transform integrand: ρ(r) * r / √(r² - rp²)
        def integrand(r):
            return rho_interp(r) * r / np.sqrt(r**2 - rp**2)

        # Integrate from rp to r_max using Gauss-Legendre quadrature
        result = gauss_legendre_integration(integrand, rp, r_max)
        sigma[i] = 2.0 * result

    return rp_centers, sigma


def project_rho_to_sigma_cylindrical(
    r_profile: np.ndarray,
    rho_profile: np.ndarray,
    rp_bins: np.ndarray,
    chi_max: float = 100.0,
    rho_mean: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D density ρ(r) to 2D surface density Σ(rp) using cylindrical integration.

    This method matches the standard ξ_gm (galaxy-matter cross-correlation) methodology
    used in the two-point correlation pipeline:

    Σ(rp) = 2 ∫₀^{χ_max} ρ(√(rp² + χ²)) dχ

    Unlike Abel transform, this integrates along the line-of-sight direction at fixed
    projected radius, which is how lensing measurements are actually performed.

    Parameters
    ----------
    r_profile : np.ndarray
        3D radial coordinates [Mpc/h]
    rho_profile : np.ndarray
        3D density profile [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected radial bin edges [Mpc/h]
    chi_max : float, default=100.0
        Maximum line-of-sight distance for integration [Mpc/h]
    rho_mean : float, optional
        Cosmic mean density [Msun/h / (Mpc/h)^3]. If provided, extrapolates
        to rho_mean at large radii (physically correct). If None, extrapolates to 0.

    Returns
    -------
    rp_centers : np.ndarray
        Projected bin centers [Mpc/h]
    sigma : np.ndarray
        Surface density [Msun/h / (Mpc/h)^2]

    Notes
    -----
    This cylindrical projection method:
    1. Does NOT assume spherical symmetry in the projection step
    2. Matches exactly how ξ_gm is computed in observations and standard pipelines
    3. Integrates along line-of-sight at fixed projected separation
    4. More robust to particle discreteness and asymmetric distributions

    The integration uses Gauss-Legendre quadrature for accuracy.

    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715 - Weak lensing methodology
    .. [2] Cacciato et al. (2009), MNRAS 394, 929 - Galaxy-matter cross-correlation

    Examples
    --------
    >>> r = np.logspace(-2, 2, 100)
    >>> rho = compute_rho_profile(...)
    >>> rp_bins = np.logspace(-1, 1.5, 15)
    >>> rp_cen, sigma = project_rho_to_sigma_cylindrical(r, rho, rp_bins, chi_max=100.0)
    """
    # Create interpolator for ρ(r)
    # At large r, extrapolate to cosmic mean density (physically correct)
    fill_value_high = rho_mean if rho_mean is not None else 0.0
    rho_interp = interp1d(r_profile, rho_profile, kind='cubic',
                          bounds_error=False, fill_value=(rho_profile[0], fill_value_high))

    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    sigma = np.zeros(len(rp_centers))

    for i, rp in enumerate(rp_centers):
        # Integrand: ρ(r) where r = √(rp² + χ²)
        # This is the density at 3D radius r, projected onto the plane at separation rp
        def integrand(chi):
            r = np.sqrt(rp**2 + chi**2)
            return rho_interp(r)

        # Integrate from 0 to chi_max along line-of-sight
        # Factor of 2 accounts for symmetric integration from -chi_max to +chi_max
        sigma[i] = 2.0 * gauss_legendre_integration(integrand, 0, chi_max)

    return rp_centers, sigma


def compute_delta_sigma_from_sigma(
    rp_centers: np.ndarray,
    sigma: np.ndarray,
    rp_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ΔΣ(rp) = Σ̄(<rp) - Σ(rp) from binned surface density profile.

    Uses Gauss-Legendre quadrature for accurate integration, following the
    same methodology as in standard_two_point_calculator.
    """
    # Create spline interpolation of sigma
    spline_sigma = interp1d(rp_centers, sigma, bounds_error=False,
                           kind='cubic', fill_value=(sigma[0], 0))

    # Integrand for mean surface density: sigma(r) * r
    def sigma_mean_integrand(r):
        return spline_sigma(r) * r

    # Compute sigma_mean using Gauss-Legendre integration
    # Σ̄(<rp) = (2/rp²) ∫₀^rp Σ(rp') rp' drp'
    sigma_mean = 2 * gauss_legendre_integration(
        sigma_mean_integrand, 0, rp_centers) / rp_centers**2

    # Excess surface density
    delta_sigma = sigma_mean - sigma

    return delta_sigma, sigma_mean



def compute_deltasigma_spherical(
    position: np.ndarray,
    nearby_particles: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    particle_mass: Optional[float] = None,
    n_particles_total: Optional[int] = None,
    Lbox: Optional[float] = None,
    chi_max: float = 100.0
) -> np.ndarray:
    """
    Compute ΔΣ(rp) using spherical shell method with cylindrical projection.

    Method:
    1. Compute 3D density profile ρ(r) in spherical shells
    2. Project to Σ(rp) using cylindrical line-of-sight integration (matching ξ_gm)
    3. Compute ΔΣ(rp) = Σ̄(<rp) - Σ(rp)

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Position at which to compute ΔΣ [Mpc/h]
    nearby_particles : np.ndarray, shape (N_nearby, 3)
        Positions of nearby particles [Mpc/h]
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    particle_mass : float, optional
        Mass of individual particles [Msun/h]. If not provided, computes
        from n_particles_total and Lbox.
    n_particles_total : int, optional
        Total number of particles in full catalog
    Lbox : float, optional
        Simulation box size [Mpc/h]
    chi_max : float, default=100.0
        Maximum line-of-sight distance for cylindrical integration [Mpc/h]

    Returns
    -------
    delta_sigma : np.ndarray, shape (len(rp_bins)-1,)
        Surface mass density contrast [Msun h/pc²]

    Notes
    -----
    This method combines the best of both approaches:
    1. Computes ρ(r) from 3D spherical shells (captures all particles efficiently)
    2. Projects using cylindrical integration matching standard ξ_gm methodology
    3. Does NOT assume spherical symmetry in projection step
    4. Consistent with how lensing is measured in observations

    The cylindrical projection Σ(rp) = 2 ∫₀^{χ_max} ρ(√(rp² + χ²)) dχ ensures
    consistency with the two-point correlation function pipeline.

    Examples
    --------
    >>> ds = compute_deltasigma_spherical(
    ...     pos, particles, RHO_M, rp_bins,
    ...     particle_mass=m_p, Lbox=1000.0, chi_max=100.0
    ... )
    """
    # Determine particle mass
    if particle_mass is not None:
        m_particle = particle_mass
    elif n_particles_total is not None and Lbox is not None:
        V_box = Lbox**3
        m_particle = RHO_M * V_box / n_particles_total
    else:
        raise ValueError("Must provide either particle_mass or (n_particles_total, Lbox)")

    # Create fine radial bins for 3D profile
    # Match the binning used in xi_gm computation for direct comparison
    r_bins = np.geomspace(5e-2, 100, 31)  # Same as test_three_method_diagnostic.py

    # Step 1: Compute 3D density profile ρ(r)
    r_centers, rho, counts = compute_rho_profile_spherical(
        position, nearby_particles, r_bins, m_particle, Lbox
    )

    # Step 2: Project to Σ(rp) at FINE radial grid (matching xi_gm method)
    # CRITICAL: Use r_bins (fine grid) not rp_bins (coarse output bins)
    # This avoids the zero-first-bin issue from constant extrapolation
    # Pass RHO_M for physically correct large-r extrapolation
    rp_centers_fine, sigma_fine = project_rho_to_sigma_cylindrical(
        r_centers, rho, r_bins, chi_max=chi_max, rho_mean=RHO_M
    )

    # Step 3: Compute ΔΣ(rp) at the fine grid
    delta_sigma_fine, sigma_mean_fine = compute_delta_sigma_from_sigma(
        rp_centers_fine, sigma_fine, r_bins
    )

    # Step 4: Average ΔΣ over requested rp_bins (matching DeltaSigmaCalculator.compute_deltasigma_averaged)
    # Create spline interpolator for ΔΣ at fine points
    spline_deltasigma = interp1d(rp_centers_fine, delta_sigma_fine,
                                  bounds_error=False, kind='cubic',
                                  fill_value=(delta_sigma_fine[0], delta_sigma_fine[-1]))

    # Integrand for averaging: ΔΣ(r) * r
    def deltasigma_integrand(r):
        return spline_deltasigma(r) * r

    # Average ΔΣ over each output bin: <ΔΣ> = (2/Δr²) ∫ ΔΣ(r) r dr
    diff_sq = np.diff(rp_bins**2)
    delta_sigma_averaged = 2 * gauss_legendre_integration(
        deltasigma_integrand, rp_bins[:-1], rp_bins[1:]
    ) / diff_sq

    # Convert to [Msun h/pc²]: 1 Mpc² = 10^12 pc²
    delta_sigma_averaged = delta_sigma_averaged / 1e12

    return delta_sigma_averaged


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


def compute_deltasigma_at_position(
    position: np.ndarray,
    nearby_particles: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    chi_max: float = 150.0,
    particle_mass: Optional[float] = None,
    n_particles_total: Optional[int] = None,
    Lbox: Optional[float] = None,
    los_axis: str = 'z'
) -> np.ndarray:
    """
    Compute ΔΣ(rp) at a single position using direct projected particle counting.

    For dark matter particles with uniform mass, this directly computes the
    projected surface density by counting particles in projected radial bins,
    avoiding the need to compute correlation functions.

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Position at which to compute ΔΣ [Mpc/h]
    nearby_particles : np.ndarray, shape (N_nearby, 3)
        Positions of nearby particles [Mpc/h]
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    chi_max : float, default=150.0
        Maximum line-of-sight distance for particle selection [Mpc/h]
    particle_mass : float, optional
        Mass of individual particles [Msun/h]. If provided, uses this directly.
        If not provided, computes from n_particles_total and Lbox.
    n_particles_total : int, optional
        Total number of particles in full catalog (for particle mass calculation)
        Only used if particle_mass is not provided.
    Lbox : float, optional
        Simulation box size [Mpc/h] (for particle mass calculation)
        Only used if particle_mass is not provided.
    los_axis : str, default='z'
        Line-of-sight axis for projection ('x', 'y', or 'z')

    Returns
    -------
    delta_sigma : np.ndarray, shape (len(rp_bins)-1,)
        Surface mass density contrast [Msun h/pc²]

    Notes
    -----
    The computation follows:
    1. Project particles onto plane perpendicular to line-of-sight
    2. Filter particles within chi_max along line-of-sight
    3. Compute Σ(rp) = (N_particles × m_particle) / A_annulus
    4. Compute ΔΣ = Σ̄(<rp) - Σ(rp)

    For uniform-mass particles: m_particle = RHO_M × V_box / N_total

    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715
    .. [2] Cacciato et al. (2009), MNRAS 394, 929
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

    # Determine particle mass (all DM particles have same mass)
    if particle_mass is not None:
        # Use provided particle mass directly
        m_particle = particle_mass
    elif n_particles_total is not None and Lbox is not None:
        # Compute from total particles and box volume
        V_box = Lbox**3
        m_particle = RHO_M * V_box / n_particles_total  # [Msun/h]
    else:
        # Fallback: assume particles represent local density
        m_particle = 1.0  # Will normalize later

    # Compute annulus areas [Mpc²/h²]
    area_annulus = np.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2)

    # Compute surface density Σ(rp) [Msun/h / (Mpc/h)²]
    # counts = number of particles in cylindrical annulus with |chi| <= chi_max
    # Volume of annulus = area_annulus × 2×chi_max
    # Density ρ = (counts × m_particle) / (area_annulus × 2×chi_max)
    # Surface density Σ = ∫ ρ dχ from -chi_max to +chi_max = 2×chi_max × ρ
    # Therefore: Σ = (counts × m_particle) / area_annulus
    sigma = (counts * m_particle) / area_annulus

    # Compute mean surface density inside each radius using spline interpolation
    # Σ̄(<rp) = (2/rp²) ∫₀^rp Σ(r') r' dr'
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

    # Create spline interpolation of sigma
    spline_sigma = interp1d(rp_centers, sigma, bounds_error=False,
                           kind='cubic', fill_value=(sigma[0], 0))

    # Integrand for mean surface density: sigma(r) * r
    def sigma_mean_integrand(r):
        return spline_sigma(r) * r

    # Compute sigma_mean using Gauss-Legendre integration
    sigma_mean = 2 * gauss_legendre_integration(
        sigma_mean_integrand, 0, rp_centers) / rp_centers**2

    # Compute ΔΣ = Σ̄(<rp) - Σ(rp)
    delta_sigma = sigma_mean - sigma

    # Convert to [Msun h/pc²]: 1 Mpc² = 10^12 pc²
    delta_sigma = delta_sigma / 1e12

    return delta_sigma


def precompute_lensing_grid(
    halo_positions: np.ndarray,
    halo_rvir: np.ndarray,
    particle_positions: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    Lbox: float,
    r_factor: float = 3.0,
    los_axis: str = 'z',
    particle_mass: Optional[float] = None,
    method: str = 'spherical',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute ΔΣ at particle positions within r_factor×R_vir of halos.

    This is the main pre-computation function that processes all halos
    and creates a database of ΔΣ values at particle positions.

    Parameters
    ----------
    halo_positions : np.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    halo_rvir : np.ndarray, shape (N_halos,)
        Virial radius of each halo [Mpc/h]
    particle_positions : np.ndarray, shape (N_particles, 3)
        Particle positions [Mpc/h]
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    r_factor : float, default=3.0
        Factor by which to extend search radius beyond R_vir
    los_axis : str, default='z'
        Line-of-sight axis for projection ('x', 'y', or 'z')
        Only used if method='cylindrical'
    particle_mass : float, optional
        Mass of individual particles [Msun/h]. If not provided, computes
        automatically from RHO_M × V_box / N_particles (Option 1: upweighting).
    method : str, default='spherical'
        Method to compute ΔΣ: 'spherical' (Abel transform, recommended) or
        'cylindrical' (direct projection, legacy)
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    positions : np.ndarray, shape (N_total, 3)
        Positions where ΔΣ was computed [Mpc/h]
    deltasigma_values : np.ndarray, shape (N_total, len(rp_bins)-1)
        Pre-computed ΔΣ(rp) at each position [Msun h/pc²]

    Examples
    --------
    >>> # Pre-compute lensing grid for a simulation (auto-upweighting)
    >>> positions, deltasigma = precompute_lensing_grid(
    ...     halo_pos, halo_rvir, particle_pos,
    ...     RHO_M=8.6e10, rp_bins=np.logspace(-1, 1.5, 15),
    ...     Lbox=1000.0
    ... )
    >>> # Particle mass is automatically: RHO_M × V_box / N_particles
    >>> # Save to disk
    >>> save_precomputed_lensing('lensing_grid.h5', positions, deltasigma, rp_bins)

    Notes
    -----
    This function can be computationally expensive (hours for large simulations).
    It should be run once per simulation snapshot and the results saved to disk.
    """
    if verbose:
        print(f"Building KD-tree for {len(particle_positions)} particles...")

    kdtree = build_particle_kdtree(particle_positions, Lbox)

    # Lists to store results
    all_positions = []
    all_deltasigma = []

    n_halos = len(halo_positions)

    if verbose:
        print(f"Processing {n_halos} halos...")

    for i, (halo_pos, rvir) in enumerate(zip(halo_positions, halo_rvir)):
        if verbose and (i + 1) % max(1, n_halos // 10) == 0:
            print(f"  Progress: {i+1}/{n_halos} ({100*(i+1)/n_halos:.1f}%)")

        # Find particles within r_factor * R_vir
        search_radius = 100
        nearby_indices = kdtree.query_ball_point(halo_pos, r=search_radius,workers=-1)

        if len(nearby_indices) < 10:
            # Skip halos with too few particles
            continue

        nearby_particles = particle_positions[nearby_indices]

        # Compute ΔΣ at each particle position
        for particle_pos in nearby_particles:
            # Use particles within search_radius of THIS particle position
            local_indices = kdtree.query_ball_point(particle_pos, r=search_radius,workers=-1)
            local_particles = particle_positions[local_indices]

            if len(local_particles) < 10:
                continue

            try:
                if method == 'spherical':
                    delta_sigma = compute_deltasigma_spherical(
                        particle_pos, local_particles, RHO_M, rp_bins,
                        particle_mass=particle_mass,
                        n_particles_total=len(particle_positions),
                        Lbox=Lbox,
                        chi_max=100.0  # Line-of-sight integration limit
                    )
                elif method == 'cylindrical':
                    delta_sigma = compute_deltasigma_at_position(
                        particle_pos, local_particles, RHO_M, rp_bins,
                        particle_mass=particle_mass,
                        n_particles_total=len(particle_positions),
                        Lbox=Lbox,
                        los_axis=los_axis
                    )
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'spherical' or 'cylindrical'.")

                all_positions.append(particle_pos)
                all_deltasigma.append(delta_sigma)

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to compute ΔΣ at position {particle_pos}: {e}")
                continue

    if verbose:
        print(f"Done! Computed ΔΣ at {len(all_positions)} positions")

    positions = np.array(all_positions)
    deltasigma_values = np.array(all_deltasigma)

    return positions, deltasigma_values


def save_precomputed_lensing(
    output_path: str,
    positions: np.ndarray,
    deltasigma_values: np.ndarray,
    rp_bins: np.ndarray,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save pre-computed lensing data to HDF5 file.

    Parameters
    ----------
    output_path : str
        Path to output HDF5 file
    positions : np.ndarray, shape (N, 3)
        Positions where ΔΣ was computed [Mpc/h]
    deltasigma_values : np.ndarray, shape (N, M)
        Pre-computed ΔΣ(rp) values [Msun h/pc²]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    metadata : dict, optional
        Additional metadata to store (cosmology, snapshot info, etc.)

    Examples
    --------
    >>> metadata = {
    ...     'cosmology': 'Planck2018',
    ...     'redshift': 0.5,
    ...     'RHO_M': 8.6e10,
    ...     'Lbox': 1000.0
    ... }
    >>> save_precomputed_lensing('lensing.h5', pos, ds, rp_bins, metadata)
    """
    with h5py.File(output_path, 'w') as f:
        # Save main data
        f.create_dataset('positions', data=positions, compression='gzip')
        f.create_dataset('deltasigma', data=deltasigma_values, compression='gzip')
        f.create_dataset('rp_bins', data=rp_bins)

        # Save metadata
        if metadata is not None:
            meta_grp = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    meta_grp.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    meta_grp.create_dataset(key, data=value)

    print(f"Saved pre-computed lensing data to {output_path}")
    print(f"  {len(positions)} positions")
    print(f"  {len(rp_bins)-1} radial bins")


def load_precomputed_lensing(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load pre-computed lensing data from HDF5 file.

    Parameters
    ----------
    input_path : str
        Path to input HDF5 file

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Positions where ΔΣ was computed [Mpc/h]
    deltasigma_values : np.ndarray, shape (N, M)
        Pre-computed ΔΣ(rp) values [Msun h/pc²]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    metadata : dict
        Metadata dictionary

    Examples
    --------
    >>> pos, ds, rp_bins, meta = load_precomputed_lensing('lensing.h5')
    >>> print(f"Loaded {len(pos)} pre-computed positions")
    >>> print(f"Cosmology: {meta.get('cosmology', 'unknown')}")
    """
    with h5py.File(input_path, 'r') as f:
        positions = f['positions'][:]
        deltasigma_values = f['deltasigma'][:]
        rp_bins = f['rp_bins'][:]

        metadata = {}
        if 'metadata' in f:
            meta_grp = f['metadata']
            # Load attributes
            for key in meta_grp.attrs:
                metadata[key] = meta_grp.attrs[key]
            # Load datasets
            for key in meta_grp.keys():
                metadata[key] = meta_grp[key][:]

    print(f"Loaded pre-computed lensing data from {input_path}")
    print(f"  {len(positions)} positions")
    print(f"  {len(rp_bins)-1} radial bins")

    return positions, deltasigma_values, rp_bins, metadata
