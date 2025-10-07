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
from ..utils import gauss_legendre_integration


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


def compute_deltasigma_at_position(
    position: np.ndarray,
    nearby_particles: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    chi_max: float = 150.0
) -> np.ndarray:
    """
    Compute ΔΣ(rp) at a single position using nearby matter particles.

    This follows the same methodology as the DeltaSigmaCalculator class
    in two_point.py, but operates on a single position.

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
        Maximum line-of-sight distance for integration [Mpc/h]

    Returns
    -------
    delta_sigma : np.ndarray, shape (len(rp_bins)-1,)
        Surface mass density contrast [Msun h/pc²]

    Notes
    -----
    The computation follows:
    1. Compute galaxy-matter correlation ξ_gm(r) from particle distribution
    2. Integrate to get surface density Σ(rp)
    3. Compute ΔΣ = Σ̄(<rp) - Σ(rp)

    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715
    .. [2] Cacciato et al. (2009), MNRAS 394, 929
    """
    # Compute distances from position to particles
    distances = np.linalg.norm(nearby_particles - position, axis=1)

    # Create fine radial bins for correlation function
    r_bins_fine = np.geomspace(5e-3, 100, 81)
    r_centers = np.sqrt(r_bins_fine[:-1] * r_bins_fine[1:])

    # Compute correlation function ξ_gm(r) using particle distribution
    # This is a simplified version - ideally would use pycorr or similar
    hist, _ = np.histogram(distances, bins=r_bins_fine)

    # Compute expected number from uniform distribution
    volume_shells = 4/3 * np.pi * (r_bins_fine[1:]**3 - r_bins_fine[:-1]**3)
    n_expected = len(nearby_particles) * volume_shells / (4/3 * np.pi * np.max(distances)**3)

    # Correlation function
    xi_gm = (hist / n_expected) - 1.0
    xi_gm = np.maximum(xi_gm, -0.99)  # Avoid negative divergences

    # Create spline interpolator
    spline_xigm = interp1d(r_centers, xi_gm, bounds_error=False,
                           kind='cubic', fill_value=(xi_gm[0], 0))

    # Compute surface density Σ(rp)
    def sigma_integrand(chi, r_proj):
        """Integrand for surface density"""
        return spline_xigm(np.sqrt(r_proj**2 + chi**2))

    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    SIGMA = 2 * gauss_legendre_integration(sigma_integrand, 0, chi_max, r_proj=rp_centers)

    # Create spline for SIGMA
    spline_SIGMA = interp1d(rp_centers, SIGMA, bounds_error=False,
                           fill_value=(SIGMA[0], 0))

    # Compute mean surface density inside rp
    def sigma_mean_integrand(r):
        return spline_SIGMA(r) * r

    SIGMA_MEAN = 2 * gauss_legendre_integration(
        sigma_mean_integrand, 0, rp_centers) / rp_centers**2

    # Compute ΔΣ
    Delta_Sigma = SIGMA_MEAN - SIGMA
    Delta_Sigma = Delta_Sigma * RHO_M / 1e12  # Convert to Msun h/pc²

    # Average within bins
    spline_DeltaSigma = interp1d(rp_centers, Delta_Sigma, bounds_error=False,
                                kind='cubic', fill_value=(Delta_Sigma[0], Delta_Sigma[-1]))

    diff_sq = np.diff(rp_bins**2)
    Delta_Sigma_averaged = 2 * gauss_legendre_integration(
        spline_DeltaSigma, rp_bins[:-1], rp_bins[1:]) / diff_sq

    return Delta_Sigma_averaged


def precompute_lensing_grid(
    halo_positions: np.ndarray,
    halo_rvir: np.ndarray,
    particle_positions: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    Lbox: float,
    r_factor: float = 3.0,
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
    >>> # Pre-compute lensing grid for a simulation
    >>> positions, deltasigma = precompute_lensing_grid(
    ...     halo_pos, halo_rvir, particle_pos,
    ...     RHO_M=8.6e10, rp_bins=np.logspace(-1, 1.5, 15),
    ...     Lbox=1000.0
    ... )
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
        search_radius = r_factor * rvir
        nearby_indices = kdtree.query_ball_point(halo_pos, r=search_radius)

        if len(nearby_indices) < 10:
            # Skip halos with too few particles
            continue

        nearby_particles = particle_positions[nearby_indices]

        # Compute ΔΣ at each particle position
        for particle_pos in nearby_particles:
            # Use particles within search_radius of THIS particle position
            local_indices = kdtree.query_ball_point(particle_pos, r=search_radius)
            local_particles = particle_positions[local_indices]

            if len(local_particles) < 10:
                continue

            try:
                delta_sigma = compute_deltasigma_at_position(
                    particle_pos, local_particles, RHO_M, rp_bins
                )

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
