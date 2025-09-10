import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jax import jit
from scipy.interpolate import CubicSpline as CS
from .utils import gauss_legendre_integration

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

@jit
def HOD_satellite_conformity(logM, As, Mmin, M1, alpha, kappa, M1_EE, has_central):
    """
    AbacusHOD-style conformity: satellite occupation depends on central presence
    
    Parameters
    ----------
    logM : halo mass
    As : satellite amplitude
    Mmin : minimum mass for centrals
    M1 : standard satellite mass scale  
    alpha : satellite power law index
    kappa : satellite cutoff parameter
    M1_EE : satellite mass scale when ELG central present
    has_central : boolean array indicating central presence
    
    Returns
    -------
    Satellite occupation probability
    """
    # Use different M1 based on central presence
    M1_eff = jnp.where(has_central, M1_EE, M1)
    
    Nsat = As * jnp.power((10**logM - kappa*10**Mmin) / (10**M1_eff), alpha)
    Nsat = jnp.where(logM > Mmin + jnp.log10(kappa), Nsat, 0)
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

def assembly_bias_mass(logM,A,B,IntrinsicProperty,ExternalProperty):
    return logM + A*IntrinsicProperty + B*ExternalProperty


    
class Occupation:
    central_funcs = {
        "LRG": (LRG_Zheng07, ["Ac", "Mmin", "sig_M"]),
        "ELG_GHOD": (ELG_GHOD, ["Ac", "Mmin", "sig_M"]),
        "ELG_SFR": (ELG_SFR, ["Ac", "Mmin", "sig_M", "gamma"]),
    }

    satellite_params=["As", "Mmin", "M1", "alpha", "kappa"]
    satellite_conformity_params=["As", "Mmin", "M1", "alpha", "kappa", "M1_EE"]
    assembly_bias_params=['A_cent','B_cent','A_sat','B_sat']

    def __init__(self, hod_type,logM_bins,mass_function,assembly_bias=False,conformity=False,fI=None,fE=None):

        if hod_type not in self.central_funcs:
            raise AttributeError(f"Unknown HOD type: {hod_type}")
        
        self.logM_bins = logM_bins
        self.mass_function=mass_function
        self.hod_type = hod_type
        self.HOD_central, self.central_params = self.central_funcs[hod_type]
        self.assembly_bias = assembly_bias
        self.conformity = conformity
        
        # Set satellite function and parameters based on conformity
        if self.conformity:
            self.HOD_satellite = HOD_satellite_conformity
            sat_params = self.satellite_conformity_params
        else:
            self.HOD_satellite = HOD_satellite
            sat_params = self.satellite_params

        if self.assembly_bias:
            self.key = set(self.central_params + sat_params + self.assembly_bias_params)
        else:
            self.key = set(self.central_params + sat_params)

        self.fI=fI
        self.fE=fE
        
    def set_params(self, dict_params):
        try:
            [dict_params[key] for key in self.key]
        except KeyError as e:
            raise KeyError(f"Missing an argument: {e.args[0]}")

        # Set satellite parameter list based on conformity
        sat_params = self.satellite_conformity_params if self.conformity else self.satellite_params
        
        self.central_args, self.satellite_args = (
            [dict_params[key] for key in self.central_params],
            [dict_params[key] for key in sat_params])
        
        self.params = dict_params

        if self.assembly_bias:
            id_cent = self.central_params.index("Mmin"),self.satellite_params.index("M1")
            Mmin_mod = assembly_bias_mass(dict_params["Mmin"],dict_params["Acent"],dict_params["Bcent"],self.fI,self.fE)
            self.central_args[id_cent] = Mmin_mod
            if self.satellite_bias:
                id_sat = self.satellite_params.index("M1")
                M1_mod = assembly_bias_mass(dict_params["M1"],dict_params["Asat"],dict_params["Bsat"],self.fI,self.fE)
                self.satellite_args[id_sat] = M1_mod
    

    def compute_HOD_occupation(self,logM,dict_params,has_central=None):
        self.set_params(dict_params)
        probC = self.HOD_central(logM, *self.central_args)
        
        if self.conformity and has_central is not None:
            # For conformity, use the actual central realization
            probS = self.HOD_satellite(logM, *self.satellite_args, has_central)
        else:
            probS = self.HOD_satellite(logM, *self.satellite_args)
            
        return probC,probS
    
    def compute_central_occupation(self,logM,dict_params):
        self.set_params(dict_params)
        probC = self.HOD_central(logM, *self.central_args)
        return probC
    
    def compute_satellite_occupation(self,logM,dict_params,has_central=None):
        self.set_params(dict_params)
        
        if self.conformity and has_central is not None:
            # For conformity, use the actual central realization
            probS = self.HOD_satellite(logM, *self.satellite_args, has_central)
        else:
            probS = self.HOD_satellite(logM, *self.satellite_args)
            
        return probS
    
    def compute_ngal(self,dict_params):
        probC,probS = self.compute_HOD_occupation(self.logM_bins,dict_params)
        ngal = compute_ngal_(self.logM_bins,self.mass_function,probC,probS)
        return ngal

    def compute_fsat(self,dict_params):
        probC,probS = self.compute_HOD_occupation(self.logM_bins,dict_params)
        fsat = compute_fsat_(self.logM_bins,self.mass_function,probC,probS)
        return fsat
    
