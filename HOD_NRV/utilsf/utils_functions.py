import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from numba import vectorize, njit, prange
from jax import jit
from scipy.special import roots_legendre
from functools import partial
import pandas as pd

#Precision for integration.
n_legendre = 200
x_legendre, w_legendre = roots_legendre(n_legendre)


def gauss_legendre_integration(f, a, b, **kwargs):
    ##Performs gauss_legendre_integration. This is super fast bro
    x_scaled =  0.5*((b - a) * x_legendre[:,None] + (a + b))
    f_values = f(x_scaled, **kwargs) 
    integral = 0.5 * (b - a) * np.dot(w_legendre, f_values)
    return integral



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

