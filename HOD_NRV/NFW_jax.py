import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit,vmap
from .utils import * 
from numba import vectorize, njit, prange
from functools import partial


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

# @partial(jit, static_argnames=['N_s_tot'])
# def elliptical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, shapes, axis_ratios, N_s, N_s_tot):
#     """
#     Generate elliptical NFW satellite positions with halo-specific shapes and axis ratios.
    
#     Parameters
#     ----------
#     key : JAX PRNG key
#     SpherePoints : Array of unit vectors (M, 3)
#     halo_centers : (num_halos, 3)
#     Rvir : (num_halos,)
#     c : (num_halos,)
#     shapes : (num_halos, 3, 3), rotation matrices per halo
#     axis_ratios : (num_halos, 2), [b/a, c/a] per halo
#     N_s : (num_halos,), satellites per halo
#     N_s_tot : int, total number of satellites

#     Returns
#     -------
#     sat_positions : (N_s_tot, 3)
#     """

#     num_halos = len(Rvir)
#     Rs = Rvir / c

#     # Split keys
#     key_r, key_theta = jrandom.split(key)

#     # Sample radii from inverse CDF
#     u_samples = random_uniform_jax(key_r, (N_s_tot,))

#     # Map satellites to their halo properties
#     halo_indices = jnp.repeat(jnp.arange(num_halos), N_s, total_repeat_length=N_s_tot)

#     sat_Rs = Rs[halo_indices]
#     sat_c = c[halo_indices]
#     sat_Rvir = Rvir[halo_indices]

#     radii = vmap(single_inverse_CDF)(u_samples, sat_Rvir, sat_Rs, sat_c)

#     # Sample unit vectors on the sphere
#     directions = jrandom.permutation(key_theta, SpherePoints)[:N_s_tot]  # shape (N_s_tot, 3)

#     # Get per-satellite rotation matrix and axis ratios
#     R_per_sat = shapes[halo_indices]               # (N_s_tot, 3, 3)
#     ar_per_sat = axis_ratios[halo_indices]         # (N_s_tot, 2), [b/a, c/a]

#     # Build scale matrix per satellite
#     b_over_a = ar_per_sat[:, 0]
#     c_over_a = ar_per_sat[:, 1]

#     scale_matrices = jnp.stack([
#         jnp.ones_like(b_over_a),  # a/a = 1
#         b_over_a,
#         c_over_a
#     ], axis=-1)  # shape (N_s_tot, 3)

#     # Convert to diagonal matrices (N_s_tot, 3, 3)
#     D_matrices = scale_matrices[:, :, None] * jnp.eye(3)[None, :, :]

#     # Transform unit vectors: x' = R @ D @ x
#     directions_exp = directions[:, :, None]                          # (N_s_tot, 3, 1)
#     scaled = jnp.matmul(D_matrices, directions_exp)                 # scale: (N_s_tot, 3, 1)
#     rotated = jnp.matmul(R_per_sat, scaled)[:, :, 0]                # rotate and squeeze

#     # Scale by radii (convert to kpc/h if needed)
#     sat_positions = rotated * (radii / 1000.0)[:, None] + halo_centers[halo_indices]
#     return sat_positions


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