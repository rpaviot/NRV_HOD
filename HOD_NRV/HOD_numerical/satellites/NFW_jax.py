import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit,vmap
from HOD_NRV.utilsf.utils_functions import *
from numba import vectorize, njit, prange
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

def sample_unit_sphere_jax(key_theta,key_phi, N):
    """
    Parameters
    ----------
    key_theta : key
    key_phi : key
    N : number of simulated points 
    
    Returns 
    ----------
    Generates a random distribution on the unit sphere.
    """
    
    theta = jnp.arccos(1 - 2 * random_uniform_jax(key_theta,N))
    phi = 2 * jnp.pi * random_uniform_jax(key_phi,N)
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack((x, y, z), axis=-1)

def create_point_on_unit_sphere(key):
    key, key_theta, key_phi = jrandom.split(key,num=3)
    SpherePoints = sample_unit_sphere_jax(key_theta,key_phi,N=10_000_000)
    return SpherePoints


@jit
def NFW_CDF(r,Rs,c):
    """
    Parameters
    ----------
    r : radius 
    r_s : scale radius of the NFW profile
    c : Concentration of the NFW profile
    
    Returns 
    ----------
    Return the CDF of the NFW profile.

    """
    CDF = (jnp.log(1+r/Rs) - (r/Rs)/(1+r/Rs))/(jnp.log(1+c) - c/(1+c))
    return CDF

@jit
def single_inverse_CDF(u,Rvir,Rs,c):
    """
    Parameters
    ----------
    r : radius 
    r_s : scale radius of the NFW profile
    c : Concentration of the NFW profile
    
    Returns 
    ----------
    Return the inverse CDF of the NFW profile.

    """
    rbins = jnp.geomspace(0.1, Rvir, 1000)
    cdf = NFW_CDF(rbins,Rs,c)
    return jnp.interp(u,cdf,rbins)


@partial(jit,static_argnames=['N_s_tot'])
def spherical_NFW_satellites_positions(key,SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot):
    """
    Generate NFW satellite positions with improved performance
    
    Parameters
    ----------
    key : PRNG key
    halo_centers : Array of shape (num_halos, 3) with halo center coordinates
    Rvir : Array of shape (num_halos,) with virial radii
    c : Array of shape (num_halos,) with concentration parameters
    N_s : Array of shape (num_halos,) with number of satellites per halo
    
    Returns
    -------
    sat_positions : Array with satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c
    
    key, key_r, key_theta = jrandom.split(key, 3)
    u_samples = random_uniform_jax(key_r, (N_s_tot,))
    
    # Create arrays that map each satellite to its halo properties
    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s,total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]
    
    radii = vmap(single_inverse_CDF)(u_samples, sat_Rvir, sat_Rs, sat_c)
    directions = jrandom.permutation(key_theta,SpherePoints)[:N_s_tot]    
    sat_positions = directions * (radii / 1000)[:, None] + halo_centers[halo_indices]
    
    return sat_positions


@partial(jit, static_argnames=['N_s_tot'])
def elliptical_NFW_satellites_positions(
    key, SpherePoints, halo_centers, Rvir, c, shapes, axis_ratios, N_s, N_s_tot
):
    """
    Sample satellites so that the *ellipsoidal radius* m
    (defined by D^{-1} R^{-1} x) follows the NFW CDF exactly.

    Parameters same as your previous elliptical function.
    axis_ratios: (num_halos, 2) where columns are [b/a, c/a]
    shapes: (num_halos, 3, 3) rotation matrices (R)
    """

    num_halos = len(Rvir)
    Rs = Rvir / c

    # PRNG splits
    key_m, key_dir = jrandom.split(key)

    # sample uniform u for inverse CDF
    u_samples = random_uniform_jax(key_m, (N_s_tot,))

    # Map satellites to halo properties
    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)

    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    # invert CDF to get ellipsoidal radii m (in same units as Rvir)
    radii_m = vmap(single_inverse_CDF)(u_samples, sat_Rvir, sat_Rs, sat_c)  # (N_s_tot,)

    # Sample *uniform* unit directions (you can provide SpherePoints or sample anew)
    # We'll use permutation of precomputed SpherePoints to get directions
    directions = jrandom.permutation(key_dir, SpherePoints)[:N_s_tot]  # (N_s_tot, 3)

    # per-satellite rotation and axis ratios
    R_per_sat = shapes[halo_indices]       # (N_s_tot, 3, 3)
    ar_per_sat = axis_ratios[halo_indices] # (N_s_tot, 2): [b/a, c/a]
    b_over_a = ar_per_sat[:, 0]
    c_over_a = ar_per_sat[:, 1]

    # build diagonal scale D = diag(a, b, c) with a = 1
    scale_vectors = jnp.stack([jnp.ones_like(b_over_a), b_over_a, c_over_a], axis=-1)  # (N_s_tot, 3)
    D_times_u = scale_vectors * directions  # elementwise multiply: (N_s_tot, 3)

    # rotate: R @ (D * u)
    D_times_u_exp = D_times_u[:, :, None]  # (N_s_tot, 3, 1)
    rotated = jnp.matmul(R_per_sat, D_times_u_exp)[:, :, 0]  # (N_s_tot, 3)

    # multiply by ellipsoidal radius m and convert units (you used /1000 earlier)
    sat_positions = rotated * radii_m[:, None] / 1000.0 + halo_centers[halo_indices]

    return sat_positions


@partial(jit,static_argnames=['N_s_tot'])
def dispersion_velocities_satellites(key, halo_velocities, vrms_h, N_s, N_s_tot):
    num_halos = len(halo_velocities)

    indices = jnp.repeat(jnp.arange(num_halos), N_s,total_repeat_length=N_s_tot)
    sig = vrms_h[indices] * 0.577
    key_vx, key_vy, key_vz = jrandom.split(key, 3)
    vx_sat = jrandom.normal(key_vx, shape=(len(indices),)) * sig + halo_velocities[indices, 0]
    vy_sat = jrandom.normal(key_vy, shape=(len(indices),)) * sig + halo_velocities[indices, 1]
    vz_sat = jrandom.normal(key_vz, shape=(len(indices),)) * sig + halo_velocities[indices, 2]
    return jnp.stack([vx_sat, vy_sat, vz_sat],axis=-1)


@jit
def exponential_profile_CDF_continuous(r, tau, Rs, Rvir):
    """
    CDF for exponential profile with continuity enforcement at Rvir.

    The exponential component is normalized so that:
    - rho_exp(Rvir) = rho_NFW(Rvir) (continuity of density)
    - Profile extends from Rvir to infinity

    Following Rocher et al., this ensures continuity in dN/dr_p.

    Parameters
    ----------
    r : radius (must be >= Rvir)
    tau : exponential decay scale in units of Rs
    Rs : scale radius
    Rvir : virial radius

    Returns
    -------
    CDF value for r >= Rvir
    """
    # For r >= Rvir, the exponential profile is:
    # rho_exp(r) = rho_NFW(Rvir) * exp(-(r - Rvir)/(tau*Rs))
    # CDF starting from Rvir:
    # F(r) = integral from Rvir to r of rho_exp(r') dr'
    # Normalized by integral from Rvir to infinity

    # We sample from normalized CDF: (1 - exp(-(r-Rvir)/(tau*Rs)))
    delta_r = r - Rvir
    return 1.0 - jnp.exp(-delta_r / (tau * Rs))


@jit
def single_exponential_inverse_CDF_continuous(u, tau, Rs, Rvir, Rmax):
    """
    Inverse CDF for exponential profile starting at Rvir with continuity.

    Samples radii r >= Rvir from exponential decay that's continuous
    with NFW profile at r = Rvir.

    Parameters
    ----------
    u : uniform random variable [0,1]
    tau : exponential decay scale in units of Rs
    Rs : scale radius
    Rvir : virial radius (starting point of exponential)
    Rmax : maximum radius cutoff

    Returns
    -------
    radius sample (r >= Rvir)
    """
    # Exponential profile: rho(r) = A * exp(-(r - Rvir)/(tau*Rs))
    # CDF: F(r) = 1 - exp(-(r - Rvir)/(tau*Rs))
    # Inverse: r = Rvir - tau*Rs * ln(1 - u)

    # Account for truncation at Rmax
    delta_max = Rmax - Rvir
    u_max = 1.0 - jnp.exp(-delta_max / (tau * Rs))
    u_scaled = u * u_max  # Scale u to [0, u_max]

    return Rvir - tau * Rs * jnp.log(1.0 - u_scaled)


@partial(jit, static_argnames=['N_s_tot'])
def extended_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot,
                                     f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """
    Generate satellite positions using extended NFW profile with continuity at Rvir.

    Key features:
    - Inner region (r < Rvir): Rescaled NFW profile (with lambda_NFW)
    - Outer region (r > Rvir): Exponential profile continuous with NFW at Rvir
    - Fraction f_exp determines proportion using outer exponential component

    This follows Rocher et al. methodology ensuring dN/dr continuity at r = Rvir.

    Parameters
    ----------
    key : PRNG key
    SpherePoints : precomputed unit sphere points
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    N_s : (num_halos,) number of satellites per halo
    N_s_tot : int, total satellites
    f_exp : float, fraction using exponential profile (beyond Rvir)
    tau : float, exponential decay scale in units of Rs
    lambda_NFW : float, NFW rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    # Component selection: exponential (outer) vs NFW (inner)
    key_comp, key_exp, key_theta = jrandom.split(key, 3)

    # Determine which satellites use which profile
    component_choice = random_uniform_jax(key_comp, (N_s_tot,))
    use_exponential = component_choice < f_exp

    # Map satellites to halo properties
    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    # Generate radii for each component
    u_samples = random_uniform_jax(key_exp, (N_s_tot,))

    # Exponential component radii: Start at Rvir with continuity
    # Use in_axes to broadcast tau scalar across all satellites
    Rmax_exp = sat_Rvir * 3.0  # Allow extension to 3*Rvir
    radii_exp = vmap(single_exponential_inverse_CDF_continuous, in_axes=(0, None, 0, 0, 0))(
        u_samples, tau, sat_Rs, sat_Rvir, Rmax_exp
    )

    # NFW component radii (rescaled by lambda_NFW)
    Rs_scaled = sat_Rs / lambda_NFW
    c_scaled = sat_c * lambda_NFW
    radii_nfw = vmap(single_inverse_CDF)(u_samples, sat_Rvir, Rs_scaled, c_scaled)

    # Select appropriate radii based on component choice
    radii = jnp.where(use_exponential, radii_exp, radii_nfw)

    # Generate directions and positions (isotropic for spherical profile)
    directions = jrandom.permutation(key_theta, SpherePoints)[:N_s_tot]
    sat_positions = directions * (radii / 1000)[:, None] + halo_centers[halo_indices]

    return sat_positions


@partial(jit, static_argnames=['N_s_tot'])
def extended_elliptical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c,
                                                shapes, axis_ratios, N_s, N_s_tot,
                                                f_exp=0.0, tau=6.0, lambda_NFW=1.0):
    """
    Extended elliptical NFW with exponential component and NFW rescaling.

    Key features:
    - Inner region (r < Rvir): Elliptical NFW profile
    - Outer region (r > Rvir): Isotropic exponential profile
    - Continuity: Exponential profile continuous with NFW at r = Rvir

    This follows Rocher et al. methodology where dN/dr_p is continuous at Rvir.

    Parameters
    ----------
    key : PRNG key
    SpherePoints : precomputed unit sphere points
    halo_centers : (num_halos, 3) halo positions
    Rvir : (num_halos,) virial radii
    c : (num_halos,) concentrations
    shapes : (num_halos, 3, 3) rotation matrices for elliptical transformation
    axis_ratios : (num_halos, 2) [b/a, c/a] axis ratios
    N_s : (num_halos,) number of satellites per halo
    N_s_tot : int, total satellites
    f_exp : float, fraction using exponential profile (beyond Rvir)
    tau : float, exponential decay scale in units of Rs
    lambda_NFW : float, NFW rescaling factor

    Returns
    -------
    sat_positions : (N_s_tot, 3) satellite positions
    """
    num_halos = len(Rvir)
    Rs = Rvir / c

    # Component selection: exponential (outer) vs NFW (inner)
    key_comp, key_exp, key_dir = jrandom.split(key, 3)
    component_choice = random_uniform_jax(key_comp, (N_s_tot,))
    use_exponential = component_choice < f_exp

    # Satellite-to-halo mapping
    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]

    # Generate radii
    u_samples = random_uniform_jax(key_exp, (N_s_tot,))

    # Exponential radii: Start at Rvir, extend outward with continuity
    # Use in_axes to broadcast tau scalar across all satellites
    Rmax_exp = sat_Rvir * 3.0
    radii_exp = vmap(single_exponential_inverse_CDF_continuous, in_axes=(0, None, 0, 0, 0))(
        u_samples, tau, sat_Rs, sat_Rvir, Rmax_exp
    )

    # Rescaled NFW radii (always within Rvir)
    Rs_scaled = sat_Rs / lambda_NFW
    c_scaled = sat_c * lambda_NFW
    radii_nfw = vmap(single_inverse_CDF)(u_samples, sat_Rvir, Rs_scaled, c_scaled)

    # Select radii based on component
    radii_m = jnp.where(use_exponential, radii_exp, radii_nfw)

    # Generate isotropic directions
    directions = jrandom.permutation(key_dir, SpherePoints)[:N_s_tot]

    # Apply elliptical transformation ONLY for satellites inside Rvir
    # Satellites with r > Rvir remain isotropic
    is_inside_rvir = radii_m <= sat_Rvir

    # Elliptical transformation for inner satellites
    R_per_sat = shapes[halo_indices]
    ar_per_sat = axis_ratios[halo_indices]
    b_over_a = ar_per_sat[:, 0]
    c_over_a = ar_per_sat[:, 1]

    scale_vectors = jnp.stack([jnp.ones_like(b_over_a), b_over_a, c_over_a], axis=-1)
    D_times_u = scale_vectors * directions

    D_times_u_exp = D_times_u[:, :, None]
    rotated_elliptical = jnp.matmul(R_per_sat, D_times_u_exp)[:, :, 0]

    # For outer satellites: use isotropic directions directly
    # For inner satellites: use elliptical transformation
    final_directions = jnp.where(
        is_inside_rvir[:, None],  # (N_s_tot, 1) for broadcasting
        rotated_elliptical,       # Elliptical for r < Rvir
        directions                # Isotropic for r > Rvir
    )

    sat_positions = final_directions * radii_m[:, None] / 1000.0 + halo_centers[halo_indices]

    return sat_positions


# =============================================================================
# Strategy Pattern for Satellite Positioning
# =============================================================================


@dataclass
class SatellitePositioningStrategy(ABC):
    """
    Abstract base class for satellite positioning strategies.
    
    This class defines the interface for different NFW positioning algorithms,
    allowing for clean separation of positioning logic and easy extension.
    """
    
    @abstractmethod
    def position_satellites(self, 
                          key: jrandom.PRNGKey,
                          SpherePoints: jnp.ndarray,
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
    """Strategy for standard spherical NFW satellite positioning."""
    
    def position_satellites(self, key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot, **kwargs):
        return spherical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot)


@dataclass  
class EllipticalNFWStrategy(SatellitePositioningStrategy):
    """Strategy for triaxial/elliptical NFW satellite positioning."""
    
    def position_satellites(self, key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot, shapes, ratios, **kwargs):
        return elliptical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, shapes, ratios, N_s, N_s_tot)


@dataclass
class ExtendedNFWStrategy(SatellitePositioningStrategy):
    """Strategy for extended NFW satellite positioning with exponential component."""
    
    def position_satellites(self, key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot, f_exp, tau, lambda_NFW, **kwargs):
        return extended_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot, 
                                               f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW)


@dataclass
class ExtendedEllipticalNFWStrategy(SatellitePositioningStrategy):
    """Strategy for extended elliptical NFW satellite positioning."""
    
    def position_satellites(self, key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot, 
                          shapes, ratios, f_exp, tau, lambda_NFW, **kwargs):
        return extended_elliptical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, 
                                                          shapes, ratios, N_s, N_s_tot, 
                                                          f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW)


def position_satellites(key: jrandom.PRNGKey,
                       SpherePoints: jnp.ndarray,
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
    the provided parameters. This function eliminates the need for complex
    conditional logic in calling code.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for satellite position sampling
    SpherePoints : jnp.ndarray
        Pre-generated points on unit sphere for directional sampling
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
        
    Examples
    --------
    >>> # Standard spherical NFW
    >>> positions = position_satellites(key, sphere_points, centers, rvir, conc, n_sats, n_tot)
    >>> 
    >>> # Extended elliptical NFW
    >>> positions = position_satellites(key, sphere_points, centers, rvir, conc, n_sats, n_tot,
    ...                               triaxial_NFW=True, shapes=shapes, ratios=ratios,
    ...                               f_exp=0.2, lambda_NFW=1.5)
    
    Notes
    -----
    Strategy selection is automatic based on parameters:
    - Extended profiles used when f_exp > 0 or lambda_NFW ≠ 1
    - Elliptical profiles used when triaxial_NFW = True
    - This gives 4 possible strategies: spherical, elliptical, extended, extended+elliptical
    
    All individual NFW positioning functions remain available for direct use
    if needed for specific applications.
    """
    # Strategy selection logic
    use_extended = (f_exp > 0.0 or lambda_NFW != 1.0)
    
    if use_extended and triaxial_NFW:
        strategy = ExtendedEllipticalNFWStrategy()
    elif use_extended:
        strategy = ExtendedNFWStrategy()
    elif triaxial_NFW:
        strategy = EllipticalNFWStrategy()
    else:
        strategy = SphericalNFWStrategy()
    
    # Execute strategy with all parameters
    return strategy.position_satellites(
        key=key,
        SpherePoints=SpherePoints,
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