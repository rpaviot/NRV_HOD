import math
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration,random_uniform_jax,random_poisson_numba
from numba import vectorize, njit, prange
from functools import partial


def NFW_PDF(x):
    """ 
    Universal NFW PDF profile as a function of x = r/rs
    """
    return x /(1 + x)**2

def generate_NFW_profile(key_r,N, x_max=50):
    """
    Parameters
    ----------
    key_r : key
    N : number of simulated points 
    x_max, optional: The upper bond of the profile. Default to 50.

    Returns 
    ----------
    Generates a random distribution following an universal NFW profile.
    """
    
    
    x = jnp.logspace(-2, jnp.log10(x_max), 10_000_000)  
    pdf = NFW_PDF(x)
    cdf = jnp.cumsum(pdf) / jnp.sum(pdf)  
    uniform_samples = random_uniform_jax(key_r,N)
    result = jnp.interp(uniform_samples, cdf, pdf)
    return result



@njit(parallel=True, fastmath=True)
def get_satellites(NFW_profile, unit_vectors,halo_c,halo_rvir, halo_centers, 
                      halo_velocities,vrms,satellite_counts, N_tot,s=1.):
    """
    
    Parameters
    
    ----------
    NFW_profile : distribution of point that follow a NFW profile.
    unit_vectors : distribution of point on the unit sphere.
    halo_c : Array of shape (N,) of the concentration of the dark matter haloes.
    halo_rvir : Array of shape (N,) of the virial radius of the dark matter haloes.
    halo_centers : Array of shape (N,3) of halo' centers coordinates.
    halo_velocities : Array of shape (N,3) tha yield halo' velocities along each dimension.
    vrms : Array of shape (N,) that give the velocity dispersion of the halo 
    satellite_counts: Array of shape (N,) that give the number of satellites per halo.
    N_tot : Integer, total number of satellites 
    s : Float, optional. Deviation of the NFW profile 
    
    Returns 
    ----------
    sat_pos: the position of the dark mat
    """


    # Create the indices for expanding halo parameters
    indices = np.repeat(np.arange(len(halo_c)), satellite_counts)

    halo_centers = halo_centers[indices]
    halo_velocities = halo_velocities[indices]
    c = halo_c[indices]
    halo_rvir = halo_rvir[indices]
    vrms = vrms[indices]

    sat_pos = np.empty_like(halo_centers)
    vx = np.empty_like(c)
    vy = np.empty_like(c)
    vz = np.empty_like(c)

    for i in prange(N_tot):
        sig = vrms[i] * 0.577
        while True:
            rand_idx = np.random.randint(0, len(NFW_profile))
            if NFW_profile[rand_idx] <= (c[i]*s):
                break 

        etaVir = NFW_profile[rand_idx] / (c[i]*s)  # r/rvir
        p = etaVir * halo_rvir[i] / 1000
        sat_pos[i] = halo_centers[i] + unit_vectors[rand_idx] * p
        vx[i] = np.random.normal(loc=halo_velocities[i,0], scale=sig)
        vy[i] = np.random.normal(loc=halo_velocities[i,1], scale=sig)
        vz[i] = np.random.normal(loc=halo_velocities[i,2], scale=sig)


    return sat_pos,np.stack((vx,vy,vz),axis=-1)


# =============================================================================
# NumPy/Numba NFW backend (no JAX dependency, CPU-optimized)
# =============================================================================

# Normalized radial grid for NumPy/Numba CDF inversion
_X_NORM_GRID = np.geomspace(1e-4, 1.0, 1000)


@njit(parallel=True, fastmath=True)
def _sample_radii_kernel(u, sat_Rvir, sat_Rs, sat_c, x_norm_grid):
    """Inverse-CDF radii sampling for NFW profiles (Numba parallel kernel).

    Parameters
    ----------
    u : (N,) uniform [0, 1] variates
    sat_Rvir : (N,) per-satellite virial radii [kpc/h]
    sat_Rs : (N,) per-satellite scale radii [kpc/h]
    sat_c : (N,) per-satellite concentrations
    x_norm_grid : (M,) normalized radial grid in [0, 1]

    Returns
    -------
    radii : (N,) sampled radii [kpc/h]
    """
    N = len(u)
    n_grid = len(x_norm_grid)
    radii = np.empty(N, dtype=np.float64)
    for i in prange(N):
        Rvir_i = sat_Rvir[i]
        Rs_i = sat_Rs[i]
        c_i = sat_c[i]
        norm_denom = math.log(1.0 + c_i) - c_i / (1.0 + c_i)
        rbins = np.empty(n_grid, dtype=np.float64)
        cdf_vals = np.empty(n_grid, dtype=np.float64)
        for j in range(n_grid):
            r_j = x_norm_grid[j] * Rvir_i
            x_j = r_j / Rs_i
            cdf_vals[j] = (math.log(1.0 + x_j) - x_j / (1.0 + x_j)) / norm_denom
            rbins[j] = r_j
        radii[i] = np.interp(u[i], cdf_vals, rbins)
    return radii


@njit(parallel=True, fastmath=True)
def _sample_sphere_kernel(u_theta, u_phi):
    """Uniform sphere sampling from [0,1] variates (Numba parallel kernel).

    Parameters
    ----------
    u_theta : (N,) uniform [0, 1] variates for polar angle
    u_phi : (N,) uniform [0, 1] variates for azimuthal angle

    Returns
    -------
    directions : (N, 3) unit vectors uniformly distributed on S^2
    """
    N = len(u_theta)
    two_pi = 2.0 * math.pi
    directions = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        cos_t = 2.0 * u_theta[i] - 1.0
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = two_pi * u_phi[i]
        directions[i, 0] = sin_t * math.cos(phi)
        directions[i, 1] = sin_t * math.sin(phi)
        directions[i, 2] = cos_t
    return directions


@njit(parallel=True, fastmath=True)
def _dispersion_kernel(halo_velocities_exp, vrms_exp):
    """Gaussian velocity dispersion sampling (Numba parallel kernel).

    Parameters
    ----------
    halo_velocities_exp : (N, 3) per-satellite halo velocities [km/s]
    vrms_exp : (N,) per-satellite velocity dispersions [km/s]

    Returns
    -------
    velocities : (N, 3) satellite velocities [km/s]
    """
    N = len(vrms_exp)
    velocities = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        sig = vrms_exp[i] * 0.577
        velocities[i, 0] = np.random.normal(halo_velocities_exp[i, 0], sig)
        velocities[i, 1] = np.random.normal(halo_velocities_exp[i, 1], sig)
        velocities[i, 2] = np.random.normal(halo_velocities_exp[i, 2], sig)
    return velocities


def spherical_NFW_positions_numba(halo_centers, Rvir, c, N_s):
    """Spherical NFW satellite positioning (NumPy/Numba backend).

    Parameters
    ----------
    halo_centers : (N_halos, 3) halo positions [Mpc/h]
    Rvir : (N_halos,) virial radii [kpc/h]
    c : (N_halos,) concentrations
    N_s : (N_halos,) integer satellite counts

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions [Mpc/h]
    """
    N_s = np.asarray(N_s, dtype=np.int64)
    N_s_tot = int(N_s.sum())
    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64)

    Rs = Rvir / c
    halo_indices = np.repeat(np.arange(len(Rvir)), N_s)

    u = np.random.uniform(0.0, 1.0, N_s_tot)
    u_theta = np.random.uniform(0.0, 1.0, N_s_tot)
    u_phi = np.random.uniform(0.0, 1.0, N_s_tot)

    radii = _sample_radii_kernel(u, Rvir[halo_indices], Rs[halo_indices],
                                  c[halo_indices], _X_NORM_GRID)
    directions = _sample_sphere_kernel(u_theta, u_phi)

    return directions * (radii / 1000.0)[:, None] + halo_centers[halo_indices]


def elliptical_NFW_positions_numba(halo_centers, Rvir, c, shapes, axis_ratios, N_s):
    """Elliptical (triaxial) NFW satellite positioning (NumPy/Numba backend).

    Parameters
    ----------
    halo_centers : (N_halos, 3) halo positions [Mpc/h]
    Rvir : (N_halos,) virial radii [kpc/h]
    c : (N_halos,) concentrations
    shapes : (N_halos, 3, 3) rotation matrices
    axis_ratios : (N_halos, 2) axis ratios [b/a, c/a]
    N_s : (N_halos,) integer satellite counts

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions [Mpc/h]
    """
    N_s = np.asarray(N_s, dtype=np.int64)
    N_s_tot = int(N_s.sum())
    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64)

    Rs = Rvir / c
    halo_indices = np.repeat(np.arange(len(Rvir)), N_s)

    u = np.random.uniform(0.0, 1.0, N_s_tot)
    u_theta = np.random.uniform(0.0, 1.0, N_s_tot)
    u_phi = np.random.uniform(0.0, 1.0, N_s_tot)

    radii = _sample_radii_kernel(u, Rvir[halo_indices], Rs[halo_indices],
                                  c[halo_indices], _X_NORM_GRID)
    directions = _sample_sphere_kernel(u_theta, u_phi)

    R_per_sat = shapes[halo_indices]          # (N_s_tot, 3, 3)
    ar_per_sat = axis_ratios[halo_indices]    # (N_s_tot, 2)
    scale_vec = np.stack(
        [np.ones(N_s_tot), ar_per_sat[:, 0], ar_per_sat[:, 1]], axis=-1
    )
    rotated = np.einsum('ijk,ik->ij', R_per_sat, scale_vec * directions)

    return rotated * (radii / 1000.0)[:, None] + halo_centers[halo_indices]


def extended_NFW_positions_numba(halo_centers, Rvir, c, N_s,
                                  f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """Extended NFW satellite positioning (inner NFW + outer exponential).

    Parameters
    ----------
    halo_centers : (N_halos, 3) halo positions [Mpc/h]
    Rvir : (N_halos,) virial radii [kpc/h]
    c : (N_halos,) concentrations
    N_s : (N_halos,) integer satellite counts
    f_exp : fraction of satellites following exponential profile [0, 1]
    tau : exponential decay scale in units of Rs
    lambda_NFW : NFW concentration rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions [Mpc/h]
    """
    N_s = np.asarray(N_s, dtype=np.int64)
    N_s_tot = int(N_s.sum())
    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64)

    Rs = Rvir / c
    halo_indices = np.repeat(np.arange(len(Rvir)), N_s)
    sat_Rvir = Rvir[halo_indices]
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]

    u = np.random.uniform(0.0, 1.0, N_s_tot)
    w = np.random.uniform(0.0, 1.0, N_s_tot)
    u_theta = np.random.uniform(0.0, 1.0, N_s_tot)
    u_phi = np.random.uniform(0.0, 1.0, N_s_tot)

    # Exponential component: r in [0, Rmax], decays as exp(-r/(tau*Rs))
    Rmax_exp = sat_Rvir * 3.0
    u_max = 1.0 - np.exp(-Rmax_exp / (tau * sat_Rs))
    u_scaled = u * u_max
    radii_exp = -tau * sat_Rs * np.log(1.0 - u_scaled)

    # NFW component with rescaled concentration
    radii_nfw = _sample_radii_kernel(u, sat_Rvir, sat_Rs / lambda_NFW,
                                      sat_c * lambda_NFW, _X_NORM_GRID)

    radii = np.where(w < f_exp, radii_exp, radii_nfw)
    directions = _sample_sphere_kernel(u_theta, u_phi)

    return directions * (radii / 1000.0)[:, None] + halo_centers[halo_indices]


def extended_elliptical_NFW_positions_numba(halo_centers, Rvir, c, shapes, axis_ratios,
                                             N_s, f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """Extended elliptical NFW: elliptical inside Rvir, isotropic outside.

    Parameters
    ----------
    halo_centers : (N_halos, 3) halo positions [Mpc/h]
    Rvir : (N_halos,) virial radii [kpc/h]
    c : (N_halos,) concentrations
    shapes : (N_halos, 3, 3) rotation matrices
    axis_ratios : (N_halos, 2) axis ratios [b/a, c/a]
    N_s : (N_halos,) integer satellite counts
    f_exp : fraction of satellites following exponential profile [0, 1]
    tau : exponential decay scale in units of Rs
    lambda_NFW : NFW concentration rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions [Mpc/h]
    """
    N_s = np.asarray(N_s, dtype=np.int64)
    N_s_tot = int(N_s.sum())
    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64)

    Rs = Rvir / c
    halo_indices = np.repeat(np.arange(len(Rvir)), N_s)
    sat_Rvir = Rvir[halo_indices]
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]

    u = np.random.uniform(0.0, 1.0, N_s_tot)
    w = np.random.uniform(0.0, 1.0, N_s_tot)
    u_theta = np.random.uniform(0.0, 1.0, N_s_tot)
    u_phi = np.random.uniform(0.0, 1.0, N_s_tot)

    # Exponential component: r in [0, Rmax], decays as exp(-r/(tau*Rs))
    Rmax_exp = sat_Rvir * 3.0
    u_max = 1.0 - np.exp(-Rmax_exp / (tau * sat_Rs))
    u_scaled = u * u_max
    radii_exp = -tau * sat_Rs * np.log(1.0 - u_scaled)

    # NFW component
    radii_nfw = _sample_radii_kernel(u, sat_Rvir, sat_Rs / lambda_NFW,
                                      sat_c * lambda_NFW, _X_NORM_GRID)

    radii = np.where(w < f_exp, radii_exp, radii_nfw)
    directions = _sample_sphere_kernel(u_theta, u_phi)

    # Elliptical transform applied only inside Rvir
    is_inside = radii <= sat_Rvir
    R_per_sat = shapes[halo_indices]
    ar_per_sat = axis_ratios[halo_indices]
    scale_vec = np.stack(
        [np.ones(N_s_tot), ar_per_sat[:, 0], ar_per_sat[:, 1]], axis=-1
    )
    rotated = np.einsum('ijk,ik->ij', R_per_sat, scale_vec * directions)
    final_directions = np.where(is_inside[:, None], rotated, directions)

    return final_directions * (radii / 1000.0)[:, None] + halo_centers[halo_indices]


def dispersion_velocities_numba(halo_velocities, vrms, N_s):
    """Gaussian velocity dispersion for satellites (NumPy backend).

    Uses numpy for reproducibility with np.random.seed.

    Parameters
    ----------
    halo_velocities : (N_halos, 3) halo velocities [km/s]
    vrms : (N_halos,) velocity dispersions [km/s]
    N_s : (N_halos,) integer satellite counts

    Returns
    -------
    velocities : (N_s_tot, 3) satellite velocities [km/s]
    """
    N_s = np.asarray(N_s, dtype=np.int64)
    halo_indices = np.repeat(np.arange(len(halo_velocities)), N_s)
    N_s_tot = len(halo_indices)
    if N_s_tot == 0:
        return np.empty((0, 3), dtype=np.float64)
    sig = vrms[halo_indices] * 0.577
    noise = np.random.normal(0.0, 1.0, (N_s_tot, 3))
    return halo_velocities[halo_indices] + noise * sig[:, None]


def position_satellites_numba(halo_centers, Rvir, c, N_s,
                               triaxial_NFW=False, shapes=None, ratios=None,
                               f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """Unified dispatch for Numba/NumPy satellite positioning.

    Mirrors position_satellites() from NFW_jax.py but without JAX.
    Strategy is selected identically: extended when f_exp>0 or lambda_NFW!=1,
    elliptical when triaxial_NFW=True.

    Parameters
    ----------
    halo_centers : (N_halos, 3) halo positions [Mpc/h]
    Rvir : (N_halos,) virial radii [kpc/h]
    c : (N_halos,) concentrations
    N_s : (N_halos,) integer satellite counts
    triaxial_NFW : bool
    shapes : (N_halos, 3, 3) rotation matrices (required if triaxial_NFW)
    ratios : (N_halos, 2) axis ratios (required if triaxial_NFW)
    f_exp : exponential fraction
    tau : exponential decay scale
    lambda_NFW : NFW rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions [Mpc/h]
    """
    use_extended = (f_exp > 0.0 or lambda_NFW != 1.0)
    if use_extended and triaxial_NFW:
        return extended_elliptical_NFW_positions_numba(
            halo_centers, Rvir, c, shapes, ratios, N_s,
            f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW,
        )
    elif use_extended:
        return extended_NFW_positions_numba(
            halo_centers, Rvir, c, N_s,
            f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW,
        )
    elif triaxial_NFW:
        return elliptical_NFW_positions_numba(halo_centers, Rvir, c, shapes, ratios, N_s)
    else:
        return spherical_NFW_positions_numba(halo_centers, Rvir, c, N_s)
