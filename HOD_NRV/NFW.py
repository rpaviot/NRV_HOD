import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from .utils import * 
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


def sample_unit_sphere_jax(key_theta,key_phi, N):
    """
    Parameters
    ----------
    key_theta : key
    key_phi : key
    N : number of simulated points 
    x_max, optional: The upper bond of the profile. Default to 50.
    
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


