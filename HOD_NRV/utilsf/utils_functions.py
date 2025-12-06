import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from numba import vectorize, njit, prange
from jax import jit
from colossus.cosmology import cosmology
from colossus.lss import mass_function
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


def set_mass_function(dict_cosmology,logM, z,mass_definition):
    ##Colossus mass function. Will be replace by pyccl mass function in the future.
    cosmo = cosmology.setCosmology('myCosmo', dict_cosmology)
    dndlogM = mass_function.massFunction(10**logM, z , mdef = mass_definition, model = 'tinker08', q_out = 'dndlnM')
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


class JaxDataSet:
    def __init__(self, df: pd.DataFrame, columns=None):
        object.__setattr__(self, '_columns', [])

        # Use specified columns or all if None
        if columns is None:
            columns = df.columns
        else:
            columns = list(columns)

        for col in columns:
            setattr(self, col, jnp.array(df[col].values))

    def __setattr__(self, name, value):
        if isinstance(value, jnp.ndarray):
            if name not in self._columns:
                self._columns.append(name)
        object.__setattr__(self, name, value)

    def __getitem__(self, mask):
        out = JaxDataSet.__new__(JaxDataSet)
        object.__setattr__(out, '_columns', self._columns)  # or .copy() if you modify it elsewhere
        for col in self._columns:
            object.__setattr__(out, col, getattr(self, col)[mask])
        return out

    # def __getitem__(self, mask):
    #     out = JaxDataSet.__new__(JaxDataSet)
    #     object.__setattr__(out, '_columns', self._columns.copy())
    #     for col in self._columns:
    #         setattr(out, col, getattr(self, col)[mask])
    #     return out

    @property
    def columns(self):
        return self._columns