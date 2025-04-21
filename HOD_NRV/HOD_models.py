import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jax import jit
from .utils import * 
from scipy.interpolate import CubicSpline as CS


sqrtpi = jnp.sqrt(2 * jnp.pi)

@jit
def func_erf(x):
    return jsp.erf(x)

@jit
def LRG_Zheng07(logM, Ac, Mmin, sig_M):
    Ncen = Ac / 2. * (1 + func_erf((logM - Mmin)/sig_M))
    return Ncen

@jit
def ELG_GHOD(logM, Ac, Mmin ,sig_M):
    Ncen = Ac / (sqrtpi*sig_M) * jnp.exp(-((logM - Mmin) ** 2) / (2 * sig_M ** 2))
    return Ncen

@jit
def ELG_SFR(logM, Ac, Mmin, sig_M, gamma):
    exp_part =ELG_GHOD(logM, Ac, Mmin ,sig_M)
    power_part = Ac/(sqrtpi*sig_M) *jnp.power(logM/Mmin, gamma)
    Ncen = jnp.where(logM < Mmin, exp_part, power_part)
    return Ncen

@jit
def HOD_satellite(logM, As, Mmin, M1, alpha, kappa):
    Nsat = As * jnp.power((10**logM - kappa*10**Mmin) / (10**M1),alpha)
    Nsat = jnp.where(logM > Mmin + jnp.log10(kappa),Nsat,0)
    return Nsat


def compute_ngal_(logM,mass_function,probC,probS):
    integrand = mass_function*(probC + probS)
    func_intg = CS(logM,integrand)
    ngal = gauss_legendre_integration(func_intg,logM.min(),logM.max())
    return ngal
    
def compute_fsat_(logM,mass_function,probC,probS):
    ngal = compute_ngal_(logM,mass_function,probC,probS)
    integrand = mass_function*probS
    func_intg = CS(logM,integrand)
    nsat = gauss_legendre_integration(func_intg,logM.min(),logM.max())
    return nsat/ngal

    
class Occupation:
    central_funcs = {
        "LRG": (LRG_Zheng07, ["Ac", "Mmin", "sig_M"]),
        "ELG_GHOD": (ELG_GHOD, ["Ac", "Mmin", "sig_M"]),
        "ELG_SFR": (ELG_SFR, ["Ac", "Mmin", "sig_M", "gamma"]),
    }

    satellite_params = ["As", "Mmin", "M1", "alpha", "kappa"]

    def __init__(self, hod_type,cosmo_params,z_snap):
        if hod_type not in self.central_funcs:
            raise ValueError(f"Unknown HOD type: {hod_type}")
        self.hod_type = hod_type
        self.HOD_central, self.central_params = self.central_funcs[hod_type]
        self.HOD_satellite = HOD_satellite
        
        params = {}
        #self.params = {key: params.get(key, None) for key in self.central_params + self.satellite_params}
        self.key = {key for key in self.central_params + self.satellite_params}
        self.logM_bins = jnp.geomspace(10.6,15,10000)

        self.mass_function = set_massfunction(cosmo_params,self.logM_bins,z=z_snap)


    def set_params(self, dict_params):
        try:
            [dict_params[key] for key in self.key]
        except KeyError as e:
            raise ValueError(f"Missing an argument: {e.args[0]}")
        self.central_args, self.satellite_args = (
            [dict_params[key] for key in self.central_params],
            [dict_params[key] for key in self.satellite_params])
        
        self.params = dict_params
        
        
    def compute_HOD_occupation(self,logM,dict_params):
        self.set_params(dict_params)
        probC = self.HOD_central(logM, *self.central_args)
        probS = self.HOD_satellite(logM, *self.satellite_args)
    
        return probC,probS
    
    def compute_ngal(self,dict_params):
        probC,probS = self.compute_HOD_occupation(self.logM_bins,dict_params)
        ngal = compute_ngal_(self.logM_bins,self.mass_function,probC,probS)
        return ngal

    def compute_fsat(self,dict_params):
        probC,probS = self.compute_HOD_occupation(self.logM_bins,dict_params)
        fsat = compute_fsat_(self.logM_bins,self.mass_function,probC,probS)
        return fsat
    
