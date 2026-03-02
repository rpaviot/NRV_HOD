import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, vmap
from HOD_NRV.utilsf.utils_functions import *
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@partial(jit, static_argnames=['N'])
def sample_unit_sphere(key, N):
    """
    Uniform sampling on unit sphere using analytical method.

    Parameters
    ----------
    key : PRNG key
    N : number of points to sample

    Returns
    -------
    directions : (N, 3) array of unit vectors uniformly distributed on sphere
    """
    key1, key2 = jrandom.split(key)
    cos_theta = 2.0 * jrandom.uniform(key1, (N,)) - 1.0
    sin_theta = jnp.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * jnp.pi * jrandom.uniform(key2, (N,))

    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = cos_theta
    return jnp.stack((x, y, z), axis=-1)


@jit
def NFW_CDF(r, Rs, c):
    """
    Parameters
    ----------
    r : radius
    Rs : scale radius of the NFW profile
    c : concentration of the NFW profile

    Returns
    ----------
    Return the CDF of the NFW profile.

    """
    CDF = (jnp.log(1+r/Rs) - (r/Rs)/(1+r/Rs))/(jnp.log(1+c) - c/(1+c))
    return CDF


# Normalized radial grid for inverse CDF interpolation
_X_NORM_GRID = jnp.geomspace(1e-4, 1.0, 1000)


@jit
def single_inverse_CDF(u, Rvir, Rs, c):
    """
    Inverse CDF using pre-computed normalized radial grid.

    Parameters
    ----------
    u : uniform random [0,1]
    Rvir : virial radius
    Rs : scale radius
    c : concentration

    Returns
    -------
    r : sampled radius following NFW distribution
    """
    rbins = _X_NORM_GRID * Rvir
    cdf = NFW_CDF(rbins, Rs, c)
    return jnp.interp(u, cdf, rbins)

inverse_CDF_batch = vmap(single_inverse_CDF, in_axes=(0, 0, 0, 0))


@partial(jit, static_argnames=['N_s_tot'])
def spherical_NFW_satellites_positions(key, halo_centers, Rvir, c, N_s, N_s_tot):
    """
    Spherical NFW satellite positioning.

    Parameters
    ----------
    key : PRNG key
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    N_s : (num_halos,) satellites per halo
    N_s_tot : int, total satellites

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    key, key_r, key_dir = jrandom.split(key, 3)
    u_samples = random_uniform_jax(key_r, (N_s_tot,))

    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    radii = inverse_CDF_batch(u_samples, sat_Rvir, sat_Rs, sat_c)
    directions = sample_unit_sphere(key_dir, N_s_tot)

    sat_positions = directions * (radii / 1000)[:, None] + halo_centers[halo_indices]

    return sat_positions


@partial(jit, static_argnames=['N_s_tot'])
def elliptical_NFW_satellites_positions(
    key, halo_centers, Rvir, c, shapes, axis_ratios, N_s, N_s_tot
):
    """
    Elliptical NFW satellite positioning.

    Parameters
    ----------
    key : PRNG key
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    shapes : (num_halos, 3, 3) rotation matrices
    axis_ratios : (num_halos, 2) [b/a, c/a]
    N_s : (num_halos,) satellites per halo
    N_s_tot : int, total satellites

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    key_m, key_dir = jrandom.split(key)
    u_samples = random_uniform_jax(key_m, (N_s_tot,))

    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    radii_m = inverse_CDF_batch(u_samples, sat_Rvir, sat_Rs, sat_c)
    directions = sample_unit_sphere(key_dir, N_s_tot)

    # Elliptical transformation
    R_per_sat = shapes[halo_indices]
    ar_per_sat = axis_ratios[halo_indices]
    b_over_a = ar_per_sat[:, 0]
    c_over_a = ar_per_sat[:, 1]

    scale_vectors = jnp.stack([jnp.ones_like(b_over_a), b_over_a, c_over_a], axis=-1)
    D_times_u = scale_vectors * directions
    D_times_u_exp = D_times_u[:, :, None]
    rotated = jnp.matmul(R_per_sat, D_times_u_exp)[:, :, 0]

    sat_positions = rotated * radii_m[:, None] / 1000.0 + halo_centers[halo_indices]

    return sat_positions


@partial(jit, static_argnames=['N_s_tot'])
def dispersion_velocities_satellites(key, halo_velocities, vrms_h, N_s, N_s_tot):
    num_halos = len(halo_velocities)

    indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sig = vrms_h[indices] * 0.577
    key_vx, key_vy, key_vz = jrandom.split(key, 3)
    vx_sat = jrandom.normal(key_vx, shape=(len(indices),)) * sig + halo_velocities[indices, 0]
    vy_sat = jrandom.normal(key_vy, shape=(len(indices),)) * sig + halo_velocities[indices, 1]
    vz_sat = jrandom.normal(key_vz, shape=(len(indices),)) * sig + halo_velocities[indices, 2]
    return jnp.stack([vx_sat, vy_sat, vz_sat], axis=-1)


@jit
def exponential_profile_CDF_continuous(r, tau, Rs):
    """
    CDF for exponential profile dN/dr = exp(-r / (tau * Rs)) for r in [0, Rmax].

    Parameters
    ----------
    r : radius
    tau : exponential decay scale in units of Rs
    Rs : scale radius

    Returns
    -------
    CDF value
    """
    return 1.0 - jnp.exp(-r / (tau * Rs))


@jit
def single_exponential_inverse_CDF_continuous(u, tau, Rs, Rmax):
    """
    Inverse CDF for exponential profile dN/dr = exp(-r / (tau * Rs)) for r in [0, Rmax].

    Parameters
    ----------
    u : uniform random variable [0,1]
    tau : exponential decay scale in units of Rs
    Rs : scale radius
    Rmax : maximum radius cutoff

    Returns
    -------
    radius sample in [0, Rmax]
    """
    u_max = 1.0 - jnp.exp(-Rmax / (tau * Rs))
    u_scaled = u * u_max

    return -tau * Rs * jnp.log(1.0 - u_scaled)


@partial(jit, static_argnames=['N_s_tot'])
def extended_NFW_satellites_positions(key, halo_centers, Rvir, c, N_s, N_s_tot,
                                     f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """
    Extended NFW satellite positioning.

    Parameters
    ----------
    key : PRNG key
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    N_s : (num_halos,) satellites per halo
    N_s_tot : int, total satellites
    f_exp : float, fraction using exponential profile
    tau : float, exponential decay scale
    lambda_NFW : float, NFW rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    key_comp, key_exp, key_dir = jrandom.split(key, 3)

    component_choice = random_uniform_jax(key_comp, (N_s_tot,))
    use_exponential = component_choice < f_exp

    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    u_samples = random_uniform_jax(key_exp, (N_s_tot,))

    # Exponential component
    Rmax_exp = sat_Rvir * 3.0
    radii_exp = vmap(single_exponential_inverse_CDF_continuous, in_axes=(0, None, 0, 0))(
        u_samples, tau, sat_Rs, Rmax_exp
    )

    # NFW component with pre-computed grid
    Rs_scaled = sat_Rs / lambda_NFW
    c_scaled = sat_c * lambda_NFW
    radii_nfw = inverse_CDF_batch(u_samples, sat_Rvir, Rs_scaled, c_scaled)

    radii = jnp.where(use_exponential, radii_exp, radii_nfw)

    directions = sample_unit_sphere(key_dir, N_s_tot)
    sat_positions = directions * (radii / 1000)[:, None] + halo_centers[halo_indices]

    return sat_positions


@partial(jit, static_argnames=['N_s_tot'])
def extended_elliptical_NFW_satellites_positions(key, halo_centers, Rvir, c,
                                                 shapes, axis_ratios, N_s, N_s_tot,
                                                 f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """
    Extended elliptical NFW satellite positioning.

    Parameters
    ----------
    key : PRNG key
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    shapes : (num_halos, 3, 3) rotation matrices
    axis_ratios : (num_halos, 2) [b/a, c/a]
    N_s : (num_halos,) satellites per halo
    N_s_tot : int, total satellites
    f_exp : float, fraction using exponential profile
    tau : float, exponential decay scale
    lambda_NFW : float, NFW rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    key_comp, key_exp, key_dir = jrandom.split(key, 3)
    component_choice = random_uniform_jax(key_comp, (N_s_tot,))
    use_exponential = component_choice < f_exp

    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    u_samples = random_uniform_jax(key_exp, (N_s_tot,))

    # Exponential component
    Rmax_exp = sat_Rvir * 3.0
    radii_exp = vmap(single_exponential_inverse_CDF_continuous, in_axes=(0, None, 0, 0))(
        u_samples, tau, sat_Rs, Rmax_exp
    )

    # NFW component with pre-computed grid
    Rs_scaled = sat_Rs / lambda_NFW
    c_scaled = sat_c * lambda_NFW
    radii_nfw = inverse_CDF_batch(u_samples, sat_Rvir, Rs_scaled, c_scaled)

    radii_m = jnp.where(use_exponential, radii_exp, radii_nfw)

    directions = sample_unit_sphere(key_dir, N_s_tot)

    is_inside_rvir = radii_m <= sat_Rvir

    # Elliptical transformation
    R_per_sat = shapes[halo_indices]
    ar_per_sat = axis_ratios[halo_indices]
    b_over_a = ar_per_sat[:, 0]
    c_over_a = ar_per_sat[:, 1]

    scale_vectors = jnp.stack([jnp.ones_like(b_over_a), b_over_a, c_over_a], axis=-1)
    D_times_u = scale_vectors * directions
    D_times_u_exp = D_times_u[:, :, None]
    rotated_elliptical = jnp.matmul(R_per_sat, D_times_u_exp)[:, :, 0]

    final_directions = jnp.where(
        is_inside_rvir[:, None],
        rotated_elliptical,
        directions
    )

    sat_positions = final_directions * radii_m[:, None] / 1000.0 + halo_centers[halo_indices]

    return sat_positions


# =============================================================================
# Strategy Pattern for Satellite Positioning
# =============================================================================


@dataclass
class SatellitePositioningStrategy(ABC):
    """Abstract base class for satellite positioning strategies."""

    @abstractmethod
    def position_satellites(self,
                          key: jrandom.PRNGKey,
                          halo_centers: jnp.ndarray,
                          Rvir: jnp.ndarray,
                          c: jnp.ndarray,
                          N_s: jnp.ndarray,
                          N_s_tot: int,
                          **kwargs) -> jnp.ndarray:
        """Position satellites using this strategy's algorithm."""
        pass


@dataclass
class SphericalNFWStrategy(SatellitePositioningStrategy):
    """Strategy for spherical NFW satellite positioning."""

    def position_satellites(self, key, halo_centers, Rvir, c, N_s, N_s_tot, **kwargs):
        return spherical_NFW_satellites_positions(key, halo_centers, Rvir, c, N_s, N_s_tot)


@dataclass
class EllipticalNFWStrategy(SatellitePositioningStrategy):
    """Strategy for elliptical NFW satellite positioning."""

    def position_satellites(self, key, halo_centers, Rvir, c, N_s, N_s_tot, shapes, ratios, **kwargs):
        return elliptical_NFW_satellites_positions(key, halo_centers, Rvir, c, shapes, ratios, N_s, N_s_tot)


@dataclass
class ExtendedNFWStrategy(SatellitePositioningStrategy):
    """Strategy for extended NFW satellite positioning."""

    def position_satellites(self, key, halo_centers, Rvir, c, N_s, N_s_tot, f_exp, tau, lambda_NFW, **kwargs):
        return extended_NFW_satellites_positions(key, halo_centers, Rvir, c, N_s, N_s_tot,
                                                f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW)


@dataclass
class ExtendedEllipticalNFWStrategy(SatellitePositioningStrategy):
    """Strategy for extended elliptical NFW satellite positioning."""

    def position_satellites(self, key, halo_centers, Rvir, c, N_s, N_s_tot,
                          shapes, ratios, f_exp, tau, lambda_NFW, **kwargs):
        return extended_elliptical_NFW_satellites_positions(key, halo_centers, Rvir, c,
                                                           shapes, ratios, N_s, N_s_tot,
                                                           f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW)


def position_satellites(key: jrandom.PRNGKey,
                       halo_centers: jnp.ndarray,
                       Rvir: jnp.ndarray,
                       c: jnp.ndarray,
                       N_s: jnp.ndarray,
                       N_s_tot: int,
                       triaxial_NFW: bool = False,
                       shapes: Optional[jnp.ndarray] = None,
                       ratios: Optional[jnp.ndarray] = None,
                       f_exp: float = 0.0,
                       tau: float = 6.0,
                       lambda_NFW: float = 1.0) -> jnp.ndarray:
    """
    Unified interface for satellite positioning using appropriate NFW strategy.

    Automatically selects and applies the correct positioning strategy based on
    the provided parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for satellite position sampling
    halo_centers : jnp.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    Rvir : jnp.ndarray, shape (N_halos,)
        Halo virial radii [Mpc/h]
    c : jnp.ndarray, shape (N_halos,)
        Halo concentration parameters
    N_s : jnp.ndarray, shape (N_halos,)
        Number of satellites per halo
    N_s_tot : int
        Total number of satellites across all halos
    triaxial_NFW : bool, default=False
        Whether to use triaxial (elliptical) NFW profiles
    shapes : jnp.ndarray, optional, shape (N_halos, 3, 3)
        Halo orientation matrices for triaxial profiles
    ratios : jnp.ndarray, optional, shape (N_halos, 2)
        Axis ratios [b/a, c/a] for triaxial profiles
    f_exp : float, default=0.0
        Exponential component fraction [0, 1] for extended profiles
    tau : float, default=6.0
        Exponential decay scale in units of Rs for extended profiles
    lambda_NFW : float, default=1.0
        NFW profile rescaling factor for extended profiles

    Returns
    -------
    sat_positions : jnp.ndarray, shape (N_s_tot, 3)
        Satellite galaxy positions [Mpc/h]

    Notes
    -----
    Strategy selection is automatic based on parameters:
    - Extended profiles used when f_exp > 0 or lambda_NFW != 1
    - Elliptical profiles used when triaxial_NFW = True
    - This gives 4 possible strategies: spherical, elliptical, extended, extended+elliptical
    """
    use_extended = (f_exp > 0.0 or lambda_NFW != 1.0)

    if use_extended and triaxial_NFW:
        strategy = ExtendedEllipticalNFWStrategy()
    elif use_extended:
        strategy = ExtendedNFWStrategy()
    elif triaxial_NFW:
        strategy = EllipticalNFWStrategy()
    else:
        strategy = SphericalNFWStrategy()

    return strategy.position_satellites(
        key=key,
        halo_centers=halo_centers,
        Rvir=Rvir,
        c=c,
        N_s=N_s,
        N_s_tot=N_s_tot,
        shapes=shapes,
        ratios=ratios,
        f_exp=f_exp,
        tau=tau,
        lambda_NFW=lambda_NFW
    )
