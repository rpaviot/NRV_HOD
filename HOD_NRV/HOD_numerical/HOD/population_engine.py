"""
Galaxy Population Engine for NRVpy HOD Code

This module contains the core algorithms for populating dark matter halos with 
galaxies using Halo Occupation Distribution (HOD) models. It includes functions
for central and satellite galaxy population, NFW profile positioning, and 
redshift space distortion applications.

Author: NRVpy Development Team
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple, Optional

from HOD_NRV.HOD_numerical.satellites.NFW_jax import position_satellites, dispersion_velocities_satellites
from HOD_NRV.utilsf.utils_functions import random_uniform_jax, random_poisson_jax


def filter_halo_data(pre_cond: jnp.ndarray, has_sat: jnp.ndarray, **arrays) -> dict:
    """
    Apply double filtering (pre_cond then has_sat) to multiple arrays.
    
    This utility function eliminates repetitive array filtering code by applying
    two sequential boolean masks to multiple arrays at once.
    
    Parameters
    ----------
    pre_cond : jnp.ndarray
        First filter condition (e.g., probS > 0)
    has_sat : jnp.ndarray  
        Second filter condition applied after pre_cond (e.g., N_s > 0)
    **arrays : keyword arguments
        Named arrays to filter. None values are preserved as None.
        
    Returns
    -------
    dict : Dictionary with filtered arrays using the same keys
    
    Examples
    --------
    >>> # Instead of multiple filtering lines:
    >>> filtered = filter_halo_data(
    ...     pre_cond, has_sat,
    ...     positions=halo_positions,
    ...     velocities=halo_velocities,
    ...     radius=halo_radius,
    ...     concentration=halo_conc,
    ...     vrms=halo_vrms,
    ...     shapes=halo_shapes if triaxial else None
    ... )
    >>> # Clean access:
    >>> filtered_positions = filtered['positions']
    
    Notes
    -----
    This function handles the common pattern of filtering arrays twice:
    first by a pre-condition, then by a secondary condition on the 
    already-filtered data.
    """
    filtered = {}
    for name, array in arrays.items():
        if array is not None:
            filtered[name] = array[pre_cond][has_sat]
        else:
            filtered[name] = None
    return filtered



def populate_centrals(key: jrandom.PRNGKey,
                     positions: jnp.ndarray,
                     velocities: jnp.ndarray, 
                     probC: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Populate central galaxies using probabilistic sampling from HOD model.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key for reproducible sampling
    positions : jnp.ndarray, shape (N_halos, 3)
        Halo positions in Cartesian coordinates [Mpc/h]
    velocities : jnp.ndarray, shape (N_halos, 3)
        Halo velocities [km/s]
    probC : jnp.ndarray, shape (N_halos,)
        Central occupation probabilities from HOD model [0, 1]
        
    Returns
    -------
    cent_positions : jnp.ndarray, shape (N_centrals, 3)
        Positions of realized central galaxies [Mpc/h]
    cent_velocities : jnp.ndarray, shape (N_centrals, 3) 
        Velocities of realized central galaxies [km/s]
    is_cent : jnp.ndarray, shape (N_halos,), dtype=bool
        Boolean array indicating which halos host central galaxies
        
    Examples
    --------
    >>> key = jax.random.key(42)
    >>> pos_c, vel_c, has_c = populate_centrals(key, halo_pos, halo_vel, prob_central)
    
    Notes
    -----
    Central galaxies are positioned exactly at halo centers and inherit halo 
    velocities. The probabilistic sampling uses uniform random numbers compared
    to the HOD central occupation probability.
    
    The `is_cent` return value is crucial for conformity models where satellite
    occupation depends on actual central realization rather than just probability.
    """
    n_halos = len(positions)
    rand_uniform = random_uniform_jax(key, n_halos)
    is_cent = probC > rand_uniform
    
    # Extract positions and velocities of halos with centrals
    cent_positions = positions[is_cent]
    cent_velocities = velocities[is_cent]
    
    return cent_positions, cent_velocities, is_cent


def populate_satellites(key_s: jrandom.PRNGKey,
                       key_s_pos: jrandom.PRNGKey,
                       key_s_vel: jrandom.PRNGKey,
                       positions: jnp.ndarray,
                       velocities: jnp.ndarray,
                       probS: jnp.ndarray,
                       radius: jnp.ndarray,
                       concentration: jnp.ndarray,
                       vrms: jnp.ndarray,
                       SpherePoints: jnp.ndarray,
                       triaxial_NFW: bool = False,
                       shapes: Optional[jnp.ndarray] = None,
                       ratios: Optional[jnp.ndarray] = None,
                       f_exp: float = 0.0,
                       tau: float = 6.0,
                       lambda_NFW: float = 1.0,
                       use_fast: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Populate satellite galaxies using NFW profiles and Poisson sampling.
    
    Parameters
    ----------
    key_s : jax.random.PRNGKey
        Random key for Poisson sampling of satellite numbers
    key_s_pos : jax.random.PRNGKey  
        Random key for satellite position sampling
    key_s_vel : jax.random.PRNGKey
        Random key for satellite velocity sampling
    positions : jnp.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    velocities : jnp.ndarray, shape (N_halos, 3)
        Halo center velocities [km/s] 
    probS : jnp.ndarray, shape (N_halos,)
        Mean satellite occupation numbers from HOD model
    radius : jnp.ndarray, shape (N_halos,)
        Halo virial radii [Mpc/h]
    concentration : jnp.ndarray, shape (N_halos,)
        Halo concentration parameters (dimensionless)
    vrms : jnp.ndarray, shape (N_halos,)
        Halo velocity dispersions [km/s]
    SpherePoints : jnp.ndarray
        Pre-generated points on unit sphere for directional sampling
    triaxial_NFW : bool, default=False
        Whether to use triaxial (elliptical) NFW profiles
    shapes : jnp.ndarray, optional, shape (N_halos, 3, 3)
        Halo orientation matrices for triaxial profiles
    ratios : jnp.ndarray, optional, shape (N_halos, 2)  
        Axis ratios [b/a, c/a] for triaxial profiles
    f_exp : float, default=0.0
        Fraction of satellites following exponential profile [0, 1]
    tau : float, default=6.0
        Exponential decay scale in units of NFW scale radius Rs
    lambda_NFW : float, default=1.0
        NFW profile rescaling factor
    use_fast : bool, default=True
        If True, use optimized satellite positioning (~10-100x faster).
        If False, use original functions (for testing/comparison).

    Returns
    -------
    sat_positions : jnp.ndarray, shape (N_satellites, 3)
        Satellite galaxy positions [Mpc/h] 
    sat_velocities : jnp.ndarray, shape (N_satellites, 3)
        Satellite galaxy velocities [km/s]
        
    Examples
    --------
    >>> keys = jax.random.split(jax.random.key(42), 3)
    >>> sat_pos, sat_vel = populate_satellites(
    ...     keys[0], keys[1], keys[2], halo_pos, halo_vel, prob_sat,
    ...     halo_R, halo_c, halo_vrms, sphere_points
    ... )
    
    Notes
    -----
    Satellite positions are drawn from NFW density profiles using inverse
    transform sampling. Extended NFW profiles with exponential components
    are used when f_exp > 0 or lambda_NFW ≠ 1.
    
    Satellite velocities follow a Gaussian distribution with dispersion
    set by the halo velocity dispersion parameter vrms.
    
    Only halos with probS > 0 are considered for satellite population to
    improve computational efficiency.
    
    References
    ----------
    .. [1] Navarro, Frenk & White (1997), ApJ 490, 493
    .. [2] Zheng et al. (2007), ApJ 659, 1  
    """
    # Pre-filter halos with non-zero satellite probability
    pre_cond = probS > 0
    probS_filtered = probS[pre_cond]
    
    # Sample satellite numbers using Poisson distribution
    N_s = random_poisson_jax(key_s, probS_filtered)
    has_sat = N_s > 0
    
    # Extract data for halos that actually have satellites using clean filtering
    N_s_filtered = N_s[has_sat]
    N_s_tot = int(jnp.sum(N_s_filtered))
    
    # Handle case of no satellites
    if N_s_tot == 0:
        empty_array = jnp.array([]).reshape(0, 3)
        return empty_array, empty_array
    
    # Apply double filtering to all halo data arrays
    filtered = filter_halo_data(
        pre_cond, has_sat,
        positions=positions,
        velocities=velocities,
        radius=radius,
        concentration=concentration,
        vrms=vrms,
        shapes=shapes if triaxial_NFW else None,
        ratios=ratios if triaxial_NFW else None
    )
    
    # Sample satellite positions using unified NFW interface
    sat_positions = position_satellites(
        key=key_s_pos,
        SpherePoints=SpherePoints,
        halo_centers=filtered['positions'],
        Rvir=filtered['radius'],
        c=filtered['concentration'],
        N_s=N_s_filtered,
        N_s_tot=N_s_tot,
        triaxial_NFW=triaxial_NFW,
        shapes=filtered['shapes'],
        ratios=filtered['ratios'],
        f_exp=f_exp,
        tau=tau,
        lambda_NFW=lambda_NFW,
        use_fast=use_fast
    )
    
    # Sample satellite velocities from dispersion model
    sat_velocities = dispersion_velocities_satellites(
        key_s_vel, filtered['velocities'], filtered['vrms'], N_s_filtered, N_s_tot
    )
    
    return sat_positions, sat_velocities


def combine_galaxy_populations(cent_positions: jnp.ndarray,
                              cent_velocities: jnp.ndarray,
                              sat_positions: jnp.ndarray, 
                              sat_velocities: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Combine central and satellite galaxy populations into single arrays.
    
    Parameters
    ----------
    cent_positions : jnp.ndarray, shape (N_centrals, 3)
        Central galaxy positions [Mpc/h]
    cent_velocities : jnp.ndarray, shape (N_centrals, 3)
        Central galaxy velocities [km/s]
    sat_positions : jnp.ndarray, shape (N_satellites, 3)  
        Satellite galaxy positions [Mpc/h]
    sat_velocities : jnp.ndarray, shape (N_satellites, 3)
        Satellite galaxy velocities [km/s]
        
    Returns
    -------
    positions_gal : jnp.ndarray, shape (N_total, 3)
        Combined galaxy positions [Mpc/h]
    velocities_gal : jnp.ndarray, shape (N_total, 3)
        Combined galaxy velocities [km/s]
        
    Examples
    --------
    >>> pos_gal, vel_gal = combine_galaxy_populations(
    ...     cent_pos, cent_vel, sat_pos, sat_vel
    ... )
    
    Notes
    -----
    Central galaxies are placed first in the combined arrays, followed by
    satellite galaxies. This ordering is preserved throughout the pipeline.
    """
    positions_gal = jnp.vstack([cent_positions, sat_positions])
    velocities_gal = jnp.vstack([cent_velocities, sat_velocities])
    return positions_gal, velocities_gal


def apply_rsd_to_galaxies(positions: jnp.ndarray,
                         velocities: jnp.ndarray,
                         rsd_factor: float,
                         rsd_axis_index: int,
                         Lbox: float) -> jnp.ndarray:
    """
    Apply redshift space distortions to galaxy positions.
    
    Parameters
    ----------
    positions : jnp.ndarray, shape (N, 3)
        Real-space galaxy positions [Mpc/h]
    velocities : jnp.ndarray, shape (N, 3)
        Galaxy velocities [km/s]
    rsd_factor : float
        RSD conversion factor: 1/(H(z)*a) [Mpc h^-1 / (km s^-1)]
    rsd_axis_index : int  
        Axis along which to apply RSD (0=x, 1=y, 2=z)
    Lbox : float
        Simulation box size [Mpc/h]
        
    Returns
    -------
    rsd_positions : jnp.ndarray, shape (N, 3)
        Galaxy positions in redshift space with periodic wrapping
        
    Examples
    --------
    >>> rsd_pos = apply_rsd_to_galaxies(pos, vel, rsd_fac, 2, 1000.0)  # RSD along z
    
    Notes
    -----
    Redshift space positions are computed as:
    s_parallel = r_parallel + v_parallel / (H(z)*a)
    
    where the parallel direction is specified by rsd_axis_index.
    Periodic boundary conditions are applied: positions ∈ [0, Lbox).
    
    References
    ----------
    .. [1] Kaiser (1987), MNRAS 227, 1
    .. [2] Hamilton (1998), in "The Evolving Universe", p. 185
    """
    # Apply RSD shift along specified axis
    shift = velocities[:, rsd_axis_index] * rsd_factor
    rsd_positions = positions.at[:, rsd_axis_index].add(shift)
    
    # Apply periodic boundary conditions
    rsd_positions = (rsd_positions + Lbox) % Lbox
    
    return rsd_positions


def populate_haloes_full(positions: jnp.ndarray,
                        velocities: jnp.ndarray,
                        mass: jnp.ndarray,
                        radius: jnp.ndarray,
                        concentration: jnp.ndarray,
                        vrms: jnp.ndarray,
                        logM: jnp.ndarray,
                        hod_model,  # HOD_models.Occupation instance
                        dict_params: dict,
                        SpherePoints: jnp.ndarray,
                        triaxial_NFW: bool = False,
                        shapes: Optional[jnp.ndarray] = None,
                        ratios: Optional[jnp.ndarray] = None,
                        f_exp: float = 0.0,
                        tau: float = 6.0,
                        lambda_NFW: float = 1.0,
                        apply_rsd: bool = True,
                        rsd_factor: float = 1.0,
                        rsd_axis_index: int = 2,
                        Lbox: float = 1000.0,
                        random_seed: Optional[int] = None,
                        use_fast: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
    """
    Complete halo population pipeline: centrals + satellites + RSD.
    
    Parameters
    ----------
    positions : jnp.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    velocities : jnp.ndarray, shape (N_halos, 3)  
        Halo velocities [km/s]
    mass : jnp.ndarray, shape (N_halos,)
        Halo masses [Msun/h] 
    radius : jnp.ndarray, shape (N_halos,)
        Halo virial radii [Mpc/h]
    concentration : jnp.ndarray, shape (N_halos,)
        Halo concentration parameters (dimensionless)
    vrms : jnp.ndarray, shape (N_halos,) 
        Halo velocity dispersions [km/s]
    logM : jnp.ndarray, shape (N_halos,)
        Log10 of halo masses
    hod_model : HOD_models.Occupation
        HOD model instance for computing occupation probabilities
    dict_params : dict
        HOD parameter dictionary
    SpherePoints : jnp.ndarray
        Pre-generated unit sphere points for satellite positioning
    triaxial_NFW : bool, default=False
        Whether to use triaxial NFW profiles
    shapes : jnp.ndarray, optional, shape (N_halos, 3, 3)
        Halo shape tensors for triaxial profiles
    ratios : jnp.ndarray, optional, shape (N_halos, 2)
        Halo axis ratios for triaxial profiles  
    f_exp : float, default=0.0
        Extended NFW exponential fraction
    tau : float, default=6.0
        Extended NFW exponential scale
    lambda_NFW : float, default=1.0
        Extended NFW rescaling factor
    apply_rsd : bool, default=True
        Whether to apply redshift space distortions
    rsd_factor : float, default=1.0
        RSD conversion factor
    rsd_axis_index : int, default=2
        RSD axis (0=x, 1=y, 2=z)  
    Lbox : float, default=1000.0
        Simulation box size [Mpc/h]
    random_seed : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    positions_gal : jnp.ndarray, shape (N_galaxies, 3)
        Final galaxy positions (with RSD if requested) [Mpc/h]
    velocities_gal : jnp.ndarray, shape (N_galaxies, 3)
        Final galaxy velocities [km/s]
    satellite_fraction : float
        Fraction of galaxies that are satellites (N_satellites / N_total)
    cent_halo_indices : jnp.ndarray, shape (N_centrals,)
        Indices of halos that host central galaxies. Used for optimized
        lensing calculations with precomputed halo-center profiles.
        
    Examples
    --------
    >>> pos_gal, vel_gal, sat_frac = populate_haloes_full(
    ...     halo_pos, halo_vel, halo_mass, halo_R, halo_c, halo_vrms,
    ...     halo_logM, hod_model, hod_params, sphere_points
    ... )
    
    Notes
    -----
    This function orchestrates the complete halo population workflow:
    1. Compute HOD occupation probabilities
    2. Populate central galaxies  
    3. Populate satellite galaxies using NFW profiles
    4. Combine central and satellite populations
    5. Apply redshift space distortions (optional)
    
    For conformity models, satellite occupation depends on actual central
    realization rather than just central probability.
    """
    # Set up random number generation
    if random_seed is None:
        random_seed = np.random.randint(low=1, high=int(1e18))
    
    key = jrandom.key(random_seed)
    key_c, key_s, key_s_pos, key_s_vel = jrandom.split(key, num=4)
    
    # Compute central occupation probability and populate centrals
    probC = hod_model.compute_central_occupation(logM, dict_params)
    cent_positions, cent_velocities, is_cent = populate_centrals(
        key_c, positions, velocities, probC
    )

    # Track which halos have centrals (for optimized lensing)
    cent_halo_indices = jnp.where(is_cent)[0]
    
    # Compute satellite occupation probability
    if hod_model.conformity:
        # For conformity: satellite occupation depends on actual central realization
        probS = hod_model.compute_satellite_occupation(
            logM, dict_params, has_central=is_cent
        )
    else:
        # Standard case: satellite occupation independent of central realization
        probS = hod_model.compute_satellite_occupation(logM, dict_params)
    
    # Populate satellites
    sat_positions, sat_velocities = populate_satellites(
        key_s, key_s_pos, key_s_vel, positions, velocities, probS,
        radius, concentration, vrms, SpherePoints,
        triaxial_NFW=triaxial_NFW, shapes=shapes, ratios=ratios,
        f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW, use_fast=use_fast
    )
    
    # Combine populations
    positions_gal, velocities_gal = combine_galaxy_populations(
        cent_positions, cent_velocities, sat_positions, sat_velocities
    )
    
    # Apply redshift space distortions if requested
    if apply_rsd:
        positions_gal = apply_rsd_to_galaxies(
            positions_gal, velocities_gal, rsd_factor, rsd_axis_index, Lbox
        )
    else:
        # Just apply periodic boundary conditions
        positions_gal = (positions_gal + Lbox) % Lbox
    
    # Calculate satellite fraction
    n_centrals = len(cent_positions)
    n_satellites = len(sat_positions)
    n_total = n_centrals + n_satellites
    satellite_fraction = n_satellites / n_total if n_total > 0 else 0.0

    return positions_gal, velocities_gal, satellite_fraction, cent_halo_indices