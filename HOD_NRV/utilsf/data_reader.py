"""
Data Reading and Initialization Module for NRVpy HOD Code

This module provides functions for reading halo catalogs, setting up cosmological 
parameters, and preparing data arrays for the Halo Occupation Distribution modeling.
All functions are designed to be called during the initialization of HaloOccupation.

Author: NRVpy Development Team
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from typing import Dict, Optional, Tuple, Union, Any

from .utils_functions import set_mass_function


def read_halo_catalog(halo_path: Optional[str] = None, 
                     DataFrame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Read halo catalog from parquet file or validate provided DataFrame.
    
    Parameters
    ----------
    halo_path : str, optional
        Path to parquet file containing halo catalog data
    DataFrame : pd.DataFrame, optional
        Pre-loaded DataFrame with halo catalog data
        
    Returns
    -------
    pd.DataFrame
        Halo catalog DataFrame
        
    Raises
    ------
    ValueError
        If neither halo_path nor DataFrame is provided
        
    Examples
    --------
    >>> # Load from file
    >>> df = read_halo_catalog(halo_path="/path/to/halos.parquet")
    >>> 
    >>> # Use existing DataFrame
    >>> df = read_halo_catalog(DataFrame=existing_df)
    
    Notes
    -----
    The DataFrame must contain columns that match the column_mapping dictionary
    used elsewhere in the code. Required columns include positions, velocities,
    mass, radius, concentration, and velocity dispersion.
    """
    if halo_path:
        return pd.read_parquet(halo_path)
    elif DataFrame is not None:
        return DataFrame
    else:
        raise ValueError("You must provide either a DataFrame or a halo_path.")


def setup_cosmology(dict_cosmology: Dict[str, float], 
                   zeff: float, 
                   mass_definition: str) -> Tuple[FlatLambdaCDM, float, float, np.ndarray, np.ndarray]:
    """
    Initialize cosmological parameters and compute derived quantities.
    
    Parameters
    ----------
    dict_cosmology : dict
        Dictionary containing cosmological parameters with keys:
        - 'Om0': Matter density parameter at z=0
        - 'Ob0': Baryon density parameter at z=0  
        - 'H0' or other Hubble parameter (H0 fixed at 100 in implementation)
    zeff : float
        Effective redshift for calculations
    mass_definition : str
        Halo mass definition (e.g., "M200c", "Mvir")
        
    Returns
    -------
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object
    RHO_M : float
        Matter density in units of Msun/h/(Mpc/h)**3
    rsd_factor : float
        Redshift space distortion factor: 1/(H(z)*a)
    logM_bins : np.ndarray
        Logarithmic mass bins for mass function computation
    mass_function : np.ndarray
        Halo mass function dn/dlogM in units of h^3 Mpc^-3
        
    Examples
    --------
    >>> cosmo_dict = {'Om0': 0.3, 'Ob0': 0.049}
    >>> cosmo, rho_m, rsd_fac, logM, mf = setup_cosmology(cosmo_dict, 1.0, "Mvir")
    
    Notes
    -----
    The Hubble parameter H0 is fixed at 100 km/s/Mpc in the implementation.
    Mass function is computed using the Tinker08 prescription via colossus.
    """
    # Set up astropy cosmology (H0 fixed at 100)
    cosmology = FlatLambdaCDM(
        Om0=dict_cosmology['Om0'], 
        Ob0=dict_cosmology['Ob0'], 
        H0=100
    )
    
    # Compute matter density in simulation units
    RHO_M = (
        (cosmology.critical_density0 * cosmology.Om0)
        .to(u.Msun/u.Mpc**3) / cosmology.h**2
    ).value
    
    # Compute RSD factor: 1/(H(z)*a)
    Hz = cosmology.H(zeff)
    a = 1. / (1 + zeff)
    rsd_factor = 1. / (Hz * a).value
    
    # Set up mass function bins and compute mass function
    logM_bins = np.log10(np.logspace(11, 15, 1024))
    mass_function = set_mass_function(
        dict_cosmology, logM_bins, z=zeff, mass_definition=mass_definition
    )
    
    return cosmology, RHO_M, rsd_factor, logM_bins, mass_function


def setup_data_arrays(df: pd.DataFrame,
                     column_mapping: Dict[str, str],
                     backend: str = "jax") -> tuple:
    """
    Convert pandas DataFrame columns to arrays for core halo properties.

    Parameters
    ----------
    df : pd.DataFrame
        Halo catalog DataFrame
    column_mapping : dict
        Dictionary mapping standard names to DataFrame column names:
        - 'x', 'y', 'z': Position coordinates
        - 'vx', 'vy', 'vz': Velocity components
        - 'mass': Halo mass
        - 'radius': Halo virial radius
        - 'c': Concentration parameter
        - 'vrms': Velocity dispersion
    backend : str, default "jax"
        Array backend: "jax" returns jnp arrays; "numba" returns float64 numpy arrays.

    Returns
    -------
    positions : array, shape (N, 3)
        Halo positions in Cartesian coordinates [Mpc/h]
    velocities : array, shape (N, 3)
        Halo velocities [km/s]
    mass : array, shape (N,)
        Halo masses [Msun/h]
    radius : array, shape (N,)
        Halo virial radii [Mpc/h]
    concentration : array, shape (N,)
        Halo concentration parameters (dimensionless)
    vrms : array, shape (N,)
        Halo velocity dispersions [km/s]
    logM : array, shape (N,)
        Logarithm (base 10) of halo masses

    Examples
    --------
    >>> cm = {'x': 'pos_x', 'y': 'pos_y', 'z': 'pos_z', 'vx': 'vel_x', ...}
    >>> pos, vel, mass, R, c, vrms, logM = setup_data_arrays(df, cm)
    """
    cm = column_mapping  # shorthand

    if backend == "numba":
        positions = np.stack([
            np.asarray(df[cm['x']], dtype=np.float64),
            np.asarray(df[cm['y']], dtype=np.float64),
            np.asarray(df[cm['z']], dtype=np.float64),
        ], axis=-1)
        velocities = np.stack([
            np.asarray(df[cm['vx']], dtype=np.float64),
            np.asarray(df[cm['vy']], dtype=np.float64),
            np.asarray(df[cm['vz']], dtype=np.float64),
        ], axis=-1)
        mass          = np.asarray(df[cm['mass']],   dtype=np.float64)
        radius        = np.asarray(df[cm['radius']], dtype=np.float64)
        concentration = np.asarray(df[cm['c']],      dtype=np.float64)
        vrms          = np.asarray(df[cm['vrms']],   dtype=np.float64)
        logM          = np.log10(mass)
    else:  # "jax"
        positions = jnp.stack([
            jnp.array(df[cm['x']]),
            jnp.array(df[cm['y']]),
            jnp.array(df[cm['z']])
        ], axis=-1)
        velocities = jnp.stack([
            jnp.array(df[cm['vx']]),
            jnp.array(df[cm['vy']]),
            jnp.array(df[cm['vz']])
        ], axis=-1)
        mass          = jnp.array(df[cm['mass']])
        radius        = jnp.array(df[cm['radius']])
        concentration = jnp.array(df[cm['c']])
        vrms          = jnp.array(df[cm['vrms']])
        logM          = jnp.log10(mass)

    return positions, velocities, mass, radius, concentration, vrms, logM


def setup_particle_data_arrays(df_part: pd.DataFrame,
                              column_mapping: Dict[str, str]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert particle DataFrame columns to JAX arrays.
    
    Parameters
    ----------
    df_part : pd.DataFrame
        Particle catalog DataFrame  
    column_mapping : dict
        Dictionary mapping standard names to DataFrame column names
        
    Returns
    -------
    positions_part : jnp.ndarray, shape (N_part, 3)
        Particle positions [Mpc/h]
    velocities_part : jnp.ndarray, shape (N_part, 3)
        Particle velocities [km/s]
        
    Examples
    --------
    >>> pos_part, vel_part = setup_particle_data_arrays(df_particles, column_mapping)
    """
    cm = column_mapping
    
    positions_part = jnp.stack([
        jnp.array(df_part[cm['x']]),
        jnp.array(df_part[cm['y']]),
        jnp.array(df_part[cm['z']])
    ], axis=-1)
    
    velocities_part = jnp.stack([
        jnp.array(df_part[cm['vx']]),
        jnp.array(df_part[cm['vy']]),
        jnp.array(df_part[cm['vz']])
    ], axis=-1)
    
    return positions_part, velocities_part


def setup_triaxial_shapes(df: pd.DataFrame,
                         column_mapping: Dict[str, str],
                         backend: str = "jax") -> tuple:
    """
    Set up triaxial halo shape data for elliptical NFW profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Halo catalog DataFrame
    column_mapping : dict
        Dictionary with triaxial shape column mappings:
        - 'a_x', 'a_y', 'a_z': Major axis components
        - 'b_x', 'b_y', 'b_z': Intermediate axis components
        - 'b_to_a': Axis ratio b/a
        - 'c_to_a': Axis ratio c/a
    backend : str, default "jax"
        Array backend: "jax" returns jnp arrays; "numba" returns float64 numpy arrays.

    Returns
    -------
    shapes : array, shape (N, 3, 3)
        Orthonormal basis vectors for each halo: [a_hat, b_hat, c_hat]
        where c_hat = a_hat × b_hat
    ratios : array, shape (N, 2)
        Axis ratios [b/a, c/a] for each halo

    Examples
    --------
    >>> cm = {'a_x': 'axis_A_x', 'a_y': 'axis_A_y', ...}
    >>> shapes, ratios = setup_triaxial_shapes(df, cm)

    Notes
    -----
    The major (a) and intermediate (b) axes are normalized to unit vectors.
    The minor axis (c) is computed as the cross product c = a × b to ensure
    a right-handed orthonormal coordinate system.
    """
    cm = column_mapping

    if backend == "numba":
        a = np.stack([
            np.asarray(df[cm['a_x']], dtype=np.float64),
            np.asarray(df[cm['a_y']], dtype=np.float64),
            np.asarray(df[cm['a_z']], dtype=np.float64),
        ], axis=-1)
        a = a / np.linalg.norm(a, axis=-1)[:, None]

        b = np.stack([
            np.asarray(df[cm['b_x']], dtype=np.float64),
            np.asarray(df[cm['b_y']], dtype=np.float64),
            np.asarray(df[cm['b_z']], dtype=np.float64),
        ], axis=-1)
        b = b / np.linalg.norm(b, axis=-1)[:, None]

        c = np.cross(a, b)
        shapes = np.stack([a, b, c], axis=-1)

        ratios = np.stack([
            np.asarray(df[cm['b_to_a']], dtype=np.float64),
            np.asarray(df[cm['c_to_a']], dtype=np.float64),
        ], axis=-1)
    else:  # "jax"
        a = jnp.stack([
            jnp.array(df[cm['a_x']]),
            jnp.array(df[cm['a_y']]),
            jnp.array(df[cm['a_z']])
        ], axis=-1)
        a = a / jnp.linalg.norm(a, axis=-1)[:, None]

        b = jnp.stack([
            jnp.array(df[cm['b_x']]),
            jnp.array(df[cm['b_y']]),
            jnp.array(df[cm['b_z']])
        ], axis=-1)
        b = b / jnp.linalg.norm(b, axis=-1)[:, None]

        c = jnp.cross(a, b)
        shapes = jnp.stack([a, b, c], axis=-1)

        ratios = jnp.stack([
            jnp.array(df[cm['b_to_a']]),
            jnp.array(df[cm['c_to_a']])
        ], axis=-1)

    return shapes, ratios


def setup_assembly_bias_data(df: pd.DataFrame,
                            column_mapping: Dict[str, str]) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Set up assembly bias data from intrinsic and external halo properties.
    
    Parameters
    ----------
    df : pd.DataFrame
        Halo catalog DataFrame
    column_mapping : dict
        Dictionary with assembly bias column mappings:
        - 'fI': Intrinsic property (e.g., concentration, age)
        - 'fE': External property (e.g., density, tidal field)
        
    Returns
    -------
    fI: jnp.ndarray or None, shape (N,)
        Intrinsic property values for assembly bias
    fE : jnp.ndarray or None, shape (N,)  
        External property values for assembly bias
        
    Raises
    ------
    AttributeError
        If assembly bias is requested but neither 'fI' nor 'fE' columns are provided
        
    Examples
    --------
    >>> cm = {'fI': 'concentration', 'fE': 'local_density'}
    >>> fI_vals, fE_vals = setup_assembly_bias_data(df, cm)
    
    Notes
    -----
    Assembly bias allows halo occupation to depend on secondary halo properties
    beyond mass. Both intrinsic (formation history) and external (environment)
    properties can be used.
    """
    cm = column_mapping
    
    if 'fI' not in cm and 'fE' not in cm:
        raise AttributeError("Assembly bias is enabled but 'fI' AND 'fE' are missing in column mapping.")
    
    fI = jnp.array(df[cm['fI']]) if 'fI' in cm else None
    fE = jnp.array(df[cm['fE']]) if 'fE' in cm else None
    
    return fI,fE


def apply_rsd_preprocessing(positions: jnp.ndarray,
                          velocities: jnp.ndarray, 
                          Lbox: float,
                          rsd_factor: float,
                          rsd_axis_index: int) -> jnp.ndarray:
    """
    Apply redshift space distortions to position data along specified axis.
    
    Parameters
    ----------
    positions : jnp.ndarray, shape (N, 3)
        Real-space positions [Mpc/h]
    velocities : jnp.ndarray, shape (N, 3)  
        Velocities [km/s]
    Lbox : float
        Simulation box size [Mpc/h]
    rsd_factor : float
        RSD conversion factor: 1/(H(z)*a) [Mpc h^-1 / (km s^-1)]
    rsd_axis_index : int
        Axis along which to apply RSD (0=x, 1=y, 2=z)
        
    Returns
    -------
    rsd_positions : jnp.ndarray, shape (N, 3)
        Positions with RSD applied and periodic boundary conditions
        
    Examples
    --------
    >>> rsd_pos = apply_rsd_preprocessing(pos, vel, 1000.0, rsd_fac, 2)  # RSD along z
    
    Notes
    -----
    Redshift space distortions are applied as:
    s_parallel = r_parallel + v_parallel / (H(z)*a)
    
    The result is wrapped to the periodic box: [0, Lbox).
    """
    # Apply RSD shift along specified axis
    shift = velocities[:, rsd_axis_index] * rsd_factor
    rsd_positions = positions.at[:, rsd_axis_index].add(shift)
    
    # Apply periodic boundary conditions
    rsd_positions = (rsd_positions + Lbox) % Lbox
    
    return rsd_positions


def validate_rsd_axis(rsd_axis: str) -> int:
    """
    Validate and convert RSD axis string to index.
    
    Parameters
    ----------
    rsd_axis : str
        RSD axis designation: 'x', 'y', or 'z'
        
    Returns
    -------
    int
        Axis index (0=x, 1=y, 2=z)
        
    Raises
    ------
    ValueError
        If rsd_axis is not one of 'x', 'y', 'z'
        
    Examples
    --------
    >>> axis_idx = validate_rsd_axis('z')  # returns 2
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if rsd_axis not in axis_map:
        raise ValueError("rsd_axis must be one of 'x', 'y', or 'z'")
    return axis_map[rsd_axis]