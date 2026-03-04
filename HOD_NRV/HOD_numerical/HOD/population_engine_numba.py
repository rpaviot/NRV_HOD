"""
NumPy/Numba Galaxy Population Engine for NRVpy HOD Code

Mirrors population_engine.py but uses NumPy/Numba instead of JAX.

Key differences vs JAX version
--------------------------------
- Randomness: np.random.seed(seed) at call site — no PRNG key management
- N_s_tot: exact Poisson draw — no bucket padding workaround
- N_s > 0 filter: applied freely — no JAX shape-recompilation concern
- Array ops: in-place NumPy — no .at[].add() immutable syntax
- Strategy dispatch: simple if/elif — no ABC/dataclass overhead
"""

import numpy as np
from typing import Tuple, Optional

from HOD_NRV.HOD_numerical.satellites.NFW import (
    position_satellites_numba,
    dispersion_velocities_numba,
)


def populate_centrals_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    probC: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Populate central galaxies (NumPy backend).

    Parameters
    ----------
    positions : (N_halos, 3) halo positions [Mpc/h]
    velocities : (N_halos, 3) halo velocities [km/s]
    probC : (N_halos,) central occupation probabilities

    Returns
    -------
    cent_positions : (N_centrals, 3)
    cent_velocities : (N_centrals, 3)
    is_cent : (N_halos,) boolean mask
    """
    probC = np.asarray(probC, dtype=np.float64)
    u = np.random.uniform(0.0, 1.0, len(probC))
    is_cent = probC > u
    return positions[is_cent], velocities[is_cent], is_cent


def populate_satellites_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    probS: np.ndarray,
    radius: np.ndarray,
    concentration: np.ndarray,
    vrms: np.ndarray,
    triaxial_NFW: bool = False,
    shapes: Optional[np.ndarray] = None,
    ratios: Optional[np.ndarray] = None,
    f_exp: float = 0.0,
    tau: float = 6.0,
    lambda_NFW: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Populate satellite galaxies using NFW profiles (NumPy/Numba backend).

    Parameters
    ----------
    positions : (N_halos, 3) halo positions [Mpc/h]
    velocities : (N_halos, 3) halo velocities [km/s]
    probS : (N_halos,) mean satellite occupation numbers
    radius : (N_halos,) virial radii [kpc/h]
    concentration : (N_halos,) concentrations
    vrms : (N_halos,) velocity dispersions [km/s]
    triaxial_NFW : use elliptical profiles
    shapes : (N_halos, 3, 3) rotation matrices
    ratios : (N_halos, 2) axis ratios [b/a, c/a]
    f_exp, tau, lambda_NFW : extended NFW parameters

    Returns
    -------
    sat_positions : (N_satellites, 3)
    sat_velocities : (N_satellites, 3)
    """
    probS = np.asarray(probS, dtype=np.float64)
    pre_cond = probS > 0
    if not np.any(pre_cond):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    # Poisson-sample satellite counts (only for halos with probS > 0)
    probS_filtered = probS[pre_cond]
    N_s = np.random.poisson(probS_filtered).astype(np.int64)
    N_s_tot = int(N_s.sum())

    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    # Apply N_s > 0 filter freely (no JAX shape-recompilation concern)
    has_sat = N_s > 0
    pre_cond_indices = np.where(pre_cond)[0]
    final_indices = pre_cond_indices[has_sat]
    N_s_filt = N_s[has_sat]

    filt_positions = positions[final_indices]
    filt_velocities = velocities[final_indices]
    filt_radius = radius[final_indices]
    filt_concentration = concentration[final_indices]
    filt_vrms = vrms[final_indices]
    filt_shapes = shapes[final_indices] if (triaxial_NFW and shapes is not None) else None
    filt_ratios = ratios[final_indices] if (triaxial_NFW and ratios is not None) else None

    sat_positions = position_satellites_numba(
        filt_positions, filt_radius, filt_concentration, N_s_filt,
        triaxial_NFW=triaxial_NFW, shapes=filt_shapes, ratios=filt_ratios,
        f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW,
    )
    sat_velocities = dispersion_velocities_numba(filt_velocities, filt_vrms, N_s_filt)

    return sat_positions, sat_velocities


def combine_galaxy_populations_numba(
    cent_pos: np.ndarray,
    cent_vel: np.ndarray,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine central and satellite populations into single arrays.

    Parameters
    ----------
    cent_pos : (N_centrals, 3) central positions [Mpc/h]
    cent_vel : (N_centrals, 3) central velocities [km/s]
    sat_pos : (N_satellites, 3) satellite positions [Mpc/h]
    sat_vel : (N_satellites, 3) satellite velocities [km/s]

    Returns
    -------
    positions_gal : (N_total, 3)
    velocities_gal : (N_total, 3)
    """
    positions_gal = np.vstack([cent_pos, sat_pos])
    velocities_gal = np.vstack([cent_vel, sat_vel])
    return positions_gal, velocities_gal


def apply_rsd_to_galaxies_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    rsd_factor: float,
    rsd_axis_index: int,
    Lbox: float,
) -> np.ndarray:
    """Apply redshift space distortions to galaxy positions.

    Parameters
    ----------
    positions : (N, 3) real-space positions [Mpc/h]
    velocities : (N, 3) velocities [km/s]
    rsd_factor : 1/(H(z)*a) conversion factor [Mpc h^-1 / (km s^-1)]
    rsd_axis_index : LOS axis (0=x, 1=y, 2=z)
    Lbox : box size [Mpc/h]

    Returns
    -------
    rsd_positions : (N, 3) redshift-space positions with periodic wrapping
    """
    positions = positions.copy()
    positions[:, rsd_axis_index] += velocities[:, rsd_axis_index] * rsd_factor
    return (positions + Lbox) % Lbox


def populate_haloes_full_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    concentration: np.ndarray,
    vrms: np.ndarray,
    logM: np.ndarray,
    hod_model,
    dict_params: dict,
    triaxial_NFW: bool = False,
    shapes: Optional[np.ndarray] = None,
    ratios: Optional[np.ndarray] = None,
    f_exp: float = 0.0,
    tau: float = 6.0,
    lambda_NFW: float = 1.0,
    apply_rsd: bool = True,
    rsd_factor: float = 1.0,
    rsd_axis_index: int = 2,
    Lbox: float = 1000.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Complete halo population pipeline: centrals + satellites + RSD (NumPy/Numba).

    Drop-in replacement for populate_haloes_full() with same return signature.

    Parameters
    ----------
    positions : (N_halos, 3) halo positions [Mpc/h]
    velocities : (N_halos, 3) halo velocities [km/s]
    mass : (N_halos,) halo masses [Msun/h]
    radius : (N_halos,) virial radii [kpc/h]
    concentration : (N_halos,) concentrations
    vrms : (N_halos,) velocity dispersions [km/s]
    logM : (N_halos,) log10 halo masses
    hod_model : HOD_models.Occupation instance
    dict_params : HOD parameter dictionary
    triaxial_NFW : use elliptical NFW profiles
    shapes : (N_halos, 3, 3) rotation matrices
    ratios : (N_halos, 2) axis ratios [b/a, c/a]
    f_exp : extended NFW exponential fraction
    tau : extended NFW exponential scale
    lambda_NFW : extended NFW rescaling factor
    apply_rsd : apply redshift space distortions
    rsd_factor : RSD conversion factor
    rsd_axis_index : RSD axis (0=x, 1=y, 2=z)
    Lbox : box size [Mpc/h]
    random_seed : integer seed for np.random reproducibility

    Returns
    -------
    positions_gal_real : (N_galaxies, 3) real-space positions [Mpc/h]. Use for lensing.
    positions_gal_rsd : (N_galaxies, 3) redshift-space positions [Mpc/h]. Use for clustering.
    velocities_gal : (N_galaxies, 3) galaxy velocities [km/s]
    satellite_fraction : float
    cent_halo_indices : (N_centrals,) indices of halos hosting centrals
    """
    if random_seed is None:
        random_seed = np.random.randint(1, int(2**32)-1)
    np.random.seed(random_seed)

    # Compute occupation probabilities (JAX-JIT; handles numpy->jax conversion)
    probC = np.asarray(hod_model.compute_central_occupation(logM, dict_params))

    cent_positions, cent_velocities, is_cent = populate_centrals_numba(
        positions, velocities, probC
    )
    cent_halo_indices = np.where(is_cent)[0]

    if hod_model.conformity:
        probS = np.asarray(hod_model.compute_satellite_occupation(
            logM, dict_params, has_central=is_cent
        ))
    else:
        probS = np.asarray(hod_model.compute_satellite_occupation(logM, dict_params))

    sat_positions, sat_velocities = populate_satellites_numba(
        positions, velocities, probS, radius, concentration, vrms,
        triaxial_NFW=triaxial_NFW, shapes=shapes, ratios=ratios,
        f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW,
    )

    positions_gal, velocities_gal = combine_galaxy_populations_numba(
        cent_positions, cent_velocities, sat_positions, sat_velocities
    )

    # Real-space positions with PBC applied (used for lensing)
    positions_gal_real = (positions_gal + Lbox) % Lbox

    # RSD positions (used for clustering)
    if apply_rsd:
        positions_gal_rsd = apply_rsd_to_galaxies_numba(
            positions_gal, velocities_gal, rsd_factor, rsd_axis_index, Lbox
        )
    else:
        positions_gal_rsd = positions_gal_real

    n_centrals = len(cent_positions)
    n_satellites = len(sat_positions)
    n_total = n_centrals + n_satellites
    satellite_fraction = n_satellites / n_total if n_total > 0 else 0.0

    return positions_gal_real, positions_gal_rsd, velocities_gal, satellite_fraction, cent_halo_indices
