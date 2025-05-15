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

@jit
def NFW_radius(u_samples,Rvir,c,mask):

    """
    Parameters
    ----------
    u_samples : Array of shape (Ns_max, N_halo) with Ns_max = max(N_s) with N_s the number of satellites per halo,
    and N_halo the total number of halo. random uniform distribution of points in [0,1]  
    Rvir : Virial radius of th halos
    c : Concentration of the halos
    mask : boolean array of shape u_samples 
    
    Returns 
    ---------- 
    Return the randomly draw NFW radii.

    """
    Rs=Rvir/c
    # vmap across satellites within each halo, then halos
    batched_eval = vmap(
        vmap(single_inverse_CDF,in_axes=(0, None, None, None)),  # over satellites
        in_axes=(0, 0, 0, 0)  # over halos
    )
    
    r_samples = batched_eval(u_samples,Rvir,Rs,c)

    # Optionally mask padded outputs
    r_samples = jnp.where(mask, r_samples, jnp.nan)
    return r_samples



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
    N_s_tot = jnp.sum(N_s)
    Rs = Rvir / c
    
    # Split keys for different random operations
    key, key_radii, key_angles = jrandom.split(key, 3)
    u_samples = random_uniform_jax(key_radii, (N_s_tot,))
    
    # Create arrays that map each satellite to its halo properties
    halo_indices = jnp.repeat(jnp.arange(num_halos), N_s,total_repeat_length=N_s_tot)
    sat_Rs = Rs[halo_indices]
    sat_c = c[halo_indices]
    sat_Rvir = Rvir[halo_indices]
    
    radii = vmap(single_inverse_CDF)(u_samples, sat_Rvir, sat_Rs, sat_c)
    coordinates = jrandom.permutation(key_angles,SpherePoints)[:N_s_tot]    
    sat_positions = coordinates * (radii / 1000)[:, None] + halo_centers[halo_indices]
    
    return sat_positions


# def spherical_NFW_satellites_positions(key, SpherePoints, halo_centers, halo_Rvir, halo_concentration, halo_N_s, N_halo):


#     max_N_s, tot_N_s = jnp.max(halo_N_s),jnp.sum(halo_N_s)

#     key, subkey_r, subkey_theta = jrandom.split(key)
#     u_samples = random_uniform_jax(subkey_r,(N_halo, max_N_s))

#     # Create the mask: True where index < N_s[i]
#     indices = jnp.arange(max_N_s)
#     mask = indices[None, :] < halo_N_s[:, None]

#     # Zero out invalid entries (optional, only if needed; could also just use the mask)
#     # u_samples_padded = jnp.where(mask, u_samples, 0.0)
#     radius = NFW_radius(u_samples,halo_Rvir,halo_concentration,mask)
#     coordinates = jrandom.shuffle(subkey_theta,SpherePoints)[:tot_N_s]

#     sat_positions = halo_centers + radius*coordinates
#     return sat_positions