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
    phi = 2 * jnp.pi * 2 * random_uniform_jax(key_phi,N)
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
    cdf = NFW_CDF(rbins)
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
    coordinates = jrandom.permutation(key_theta,SpherePoints)[:N_s_tot]    
    sat_positions = coordinates * (radii / 1000)[:, None] + halo_centers[halo_indices]
    
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
    return jnp.column_stack([vx_sat, vy_sat, vz_sat])