import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from numba import vectorize, njit, prange
from jax import jit
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.special import roots_legendre
from functools import partial

#Precision for DeltaSigma estimation.
n_legendre = 200
x_legendre, w_legendre = roots_legendre(n_legendre)


def gauss_legendre_integration(f, a, b, **kwargs):
    ##Performs gauss_legendre_integration. This is super fast bro
    x_scaled =  0.5*((b - a) * x_legendre[:,None] + (a + b))
    f_values = f(x_scaled, **kwargs) 
    integral = 0.5 * (b - a) * np.dot(w_legendre, f_values)
    return integral


def set_massfunction(dict_cosmology,logM, z):
    ##Colossus mass function. Will be replace by pyccl mass function in the future.
    cosmo = cosmology.setCosmology('myCosmo', dict_cosmology)
    dndlogM = mass_function.massFunction(10**logM, z , mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')
    return dndlogM*np.log(10)


@njit(parallel=True, fastmath=True)
def random_uniform_numba(n,a=0,b=1):
    result = np.zeros(n)
    for i in prange(n):
        result[i] = np.random.uniform(a, b)
    return result

@njit(fastmath=True, parallel=True)
def random_poisson_numba(prob):
    n = len(prob)
    result = np.zeros(n)
    for i in prange(n):
        result[i] = np.random.poisson(prob[i])
    return result

def random_uniform_jax(key,size,a=0,b=1):
    result = jrandom.uniform(key,size,minval=a,maxval=b)
    return result

def random_poisson_jax(key,prob):
    result = jrandom.poisson(key,prob, (prob.shape[0],))
    return result
