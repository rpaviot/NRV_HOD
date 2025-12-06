"""
Optimized Halo Model Power Spectrum Calculator with Full Vmap
=============================================================

This module provides a highly optimized JAX-based halo model calculator
for computing galaxy-galaxy (P_gg) and galaxy-matter (P_gm) power spectra
across multiple redshifts.

Key Features:
- Full vmap over ALL dimensions including redshift
- Support for Standard HOD and CSMF (Conditional Stellar Mass Function)
- Correct unit conversion between natural units and h-units
- Pre-computed quantities for efficient MCMC sampling

Unit Conversion Rules (CCL natural -> h-units):
    k [h/Mpc]                = k [1/Mpc] / h
    P(k) [(Mpc/h)³]          = P(k) [Mpc³] × h³
    M_halo [Msun/h]          = M [Msun] × h
    M_stellar [Msun/h²]      = M* [Msun] × h²
    dn/dlogM [(Mpc/h)⁻³]     = dn/dlogM [Mpc⁻³] / h²  (NOT h³, because of h in logM!)
    ρ_m [(Msun/h)/(Mpc/h)³]  = ρ_m [Msun/Mpc³] / h²
    n_gal [(Mpc/h)⁻³]        = n_gal [Mpc⁻³] / h³

Author: Optimized implementation for MCMC chains
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf as erf_jax
from functools import partial
import pyccl as ccl
from scipy.integrate import simpson
from scipy.special import erf
from scipy.interpolate import interp1d
import time
from enum import Enum
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Import Hankel transform utilities
from HOD_NRV.utilsf.hankel_transforms import (
    Pk_to_wgg_direct,
    Pk_to_DeltaSigma_direct,
    Pk_gm_to_DeltaSigma_traditional
)
from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


# ============================================================================
# HOD Types Enumeration
# ============================================================================

class HODType(Enum):
    """Supported HOD model types"""
    STANDARD = "standard"
    CSMF = "csmf"


# ============================================================================
# HOD Parameter Definitions
# ============================================================================

# Parameter names expected for each HOD type
STANDARD_HOD_PARAMS = ['log10Mmin', 'siglnM', 'log10M0', 'log10M1', 'alpha']
CSMF_HOD_PARAMS = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']


def validate_hod_params(hod_type: HODType, params: Dict) -> bool:
    """Validate that required HOD parameters are present"""
    if hod_type == HODType.STANDARD:
        required = STANDARD_HOD_PARAMS
    elif hod_type == HODType.CSMF:
        required = CSMF_HOD_PARAMS
    else:
        raise ValueError(f"Unknown HOD type: {hod_type}")
    
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing HOD parameters for {hod_type.value}: {missing}")
    return True


# ============================================================================
# Simpson Integration Weights
# ============================================================================

def get_simpson_weights(n: int) -> jnp.ndarray:
    """
    Compute Simpson's rule weights for integration.
    
    Parameters
    ----------
    n : int
        Number of points (must be odd for proper Simpson's rule)
    
    Returns
    -------
    weights : jnp.ndarray
        Integration weights
    """
    weights = np.ones(n)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return jnp.array(weights / 3.0)


# ============================================================================
# Standard HOD Model
# ============================================================================

class StandardHOD:
    """
    Standard Zheng et al. HOD model.
    
    The model is defined by:
        N_cen(M) = 0.5 * [1 + erf((ln(M) - ln(Mmin)) / sigma_lnM)]
        N_sat(M) = [(M - M0) / M1]^alpha  for M > M0, else 0
    
    IMPORTANT: To match pyccl's HaloProfileHOD behavior, HOD mass parameters
    are stored in h-units (Msun/h) and compared directly against halo masses
    in natural units (Msun). This is pyccl's convention.
    
    Parameters
    ----------
    params : dict
        HOD parameters:
        - log10Mmin : log10 of minimum halo mass [Msun/h if units_per_h else Msun]
        - siglnM : scatter in ln(M)
        - log10M0 : log10 of satellite cutoff mass [Msun/h if units_per_h else Msun]
        - log10M1 : log10 of satellite normalization mass [Msun/h if units_per_h else Msun]
        - alpha : satellite power law slope
    units_per_h : bool
        If True, mass parameters are assumed to be in Msun/h (standard convention).
        These are then used directly against natural-unit mass arrays to match pyccl.
    h : float
        Hubble parameter h = H0/100 (required if units_per_h=True)
    """
    
    def __init__(self, params: Dict, units_per_h: bool = False, h: Optional[float] = None):
        validate_hod_params(HODType.STANDARD, params)
        
        self.units_per_h = units_per_h
        self.h = h
        self.hod_type = HODType.STANDARD
        
        if units_per_h and h is None:
            raise ValueError("Must provide h when units_per_h=True")
        
        # Store input parameters
        self.params = params.copy()
        self.sigma_lnM = params['siglnM']
        self.alpha = params['alpha']
        
        # pyccl works entirely in natural units (Msun, Mpc).
        # If units_per_h=True, input masses are in Msun/h and must be
        # converted to Msun by multiplying by h.
        # For log10 quantities: log10(M [Msun]) = log10(M [Msun/h]) + log10(h)
        
        if units_per_h:
            log10h = np.log10(h)
            self.Mmin = 10**(params['log10Mmin'] + log10h)  # Msun/h -> Msun
            self.M0 = 10**(params['log10M0'] + log10h)
            self.M1 = 10**(params['log10M1'] + log10h)
        else:
            self.Mmin = 10**params['log10Mmin']  # Already in Msun
            self.M0 = 10**params['log10M0']
            self.M1 = 10**params['log10M1']
        
        # Store as JAX arrays for JIT compilation
        self._Mmin_jax = jnp.array(self.Mmin)
        self._M0_jax = jnp.array(self.M0)
        self._M1_jax = jnp.array(self.M1)
        self._sigma_lnM_jax = jnp.array(self.sigma_lnM)
        self._alpha_jax = jnp.array(self.alpha)
    
    def N_central(self, M: jnp.ndarray) -> jnp.ndarray:
        """
        Central occupation number.
        
        Parameters
        ----------
        M : array
            Halo mass in NATURAL units (Msun)
        
        Returns
        -------
        N_cen : array
            Mean number of central galaxies
        """
        M = jnp.atleast_1d(M)
        x = jnp.log(M / self._Mmin_jax) / self._sigma_lnM_jax
        return 0.5 * (1.0 + erf_jax(x))
    
    def N_satellite(self, M: jnp.ndarray) -> jnp.ndarray:
        """
        Satellite occupation number.
        
        Parameters
        ----------
        M : array
            Halo mass in NATURAL units (Msun)
        
        Returns
        -------
        N_sat : array
            Mean number of satellite galaxies
        """
        M = jnp.atleast_1d(M)
        # Use jnp.where for JAX compatibility
        result = jnp.where(
            M > self._M0_jax,
            ((M - self._M0_jax) / self._M1_jax) ** self._alpha_jax,
            0.0
        )
        return result


# ============================================================================
# CSMF HOD Model (Conditional Stellar Mass Function)
# ============================================================================

class CSMF_HOD:
    """
    Conditional Stellar Mass Function HOD (Dvornik+2022).
    
    This model describes the galaxy population as a function of both
    halo mass and stellar mass.
    
    IMPORTANT: Similar to StandardHOD, mass parameters are used directly
    without unit conversion to maintain consistency with how pyccl handles
    HOD parameters.
    
    Parameters
    ----------
    params : dict
        CSMF parameters:
        - M0 : characteristic stellar mass normalization (log10 if masses_are_log10=True)
        - M1 : characteristic halo mass for SHMR (log10 if masses_are_log10=True)
        - gamma1 : low-mass slope of SHMR
        - gamma2 : high-mass slope of SHMR
        - sigma_c : scatter in central stellar mass
        - alpha_s : satellite Schechter function slope
        - b0 : satellite normalization intercept
        - b1 : satellite normalization slope
    masses_are_log10 : bool
        If True, M0 and M1 are provided as log10 values
    units_per_h : bool
        If True, mass parameters are in h-units (stored but not converted)
    h : float
        Hubble parameter (stored for reference)
    """
    
    def __init__(self, params: Dict, masses_are_log10: bool = True,
                 units_per_h: bool = False, h: Optional[float] = None):
        validate_hod_params(HODType.CSMF, params)
        
        self.units_per_h = units_per_h
        self.h = h
        self.hod_type = HODType.CSMF
        self.params = params.copy()
        
        if units_per_h and h is None:
            raise ValueError("Must provide h when units_per_h=True")
        
        # pyccl works entirely in natural units (Msun, Mpc).
        # If units_per_h=True:
        #   - Halo masses (M1) are in Msun/h -> multiply by h
        #   - Stellar masses (M0) are in Msun/h² -> multiply by h²
        # For log10 quantities: add log10(h) or 2*log10(h)
        
        if masses_are_log10:
            if units_per_h:
                log10h = np.log10(h)
                self.M0 = 10**(params['M0'] + 2*log10h)  # Stellar: Msun/h² -> Msun
                self.M1 = 10**(params['M1'] + log10h)    # Halo: Msun/h -> Msun
            else:
                self.M0 = 10**params['M0']
                self.M1 = 10**params['M1']
        else:
            if units_per_h:
                self.M0 = params['M0'] * h**2  # Stellar: Msun/h² -> Msun
                self.M1 = params['M1'] * h     # Halo: Msun/h -> Msun
            else:
                self.M0 = params['M0']
                self.M1 = params['M1']
        
        # Other parameters (dimensionless)
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.sigma_c = params['sigma_c']
        self.alpha_s = params['alpha_s']
        self.b0 = params['b0']
        self.b1 = params['b1']
        
        # Store as JAX arrays
        self._params_jax = {
            'M0': jnp.array(self.M0),
            'M1': jnp.array(self.M1),
            'gamma1': jnp.array(self.gamma1),
            'gamma2': jnp.array(self.gamma2),
            'sigma_c': jnp.array(self.sigma_c),
            'alpha_s': jnp.array(self.alpha_s),
            'b0': jnp.array(self.b0),
            'b1': jnp.array(self.b1)
        }
    
    def Mstar_central(self, Mh: jnp.ndarray) -> jnp.ndarray:
        """Stellar-to-halo mass relation for centrals"""
        Mh = jnp.atleast_1d(Mh)
        x = Mh / self._params_jax['M1']
        return self._params_jax['M0'] * (x**self._params_jax['gamma1']) / \
               ((1 + x)**(self._params_jax['gamma1'] - self._params_jax['gamma2']))
    
    def Mstar_satellite(self, Mh: jnp.ndarray) -> jnp.ndarray:
        """Characteristic stellar mass for satellites"""
        return 0.56 * self.Mstar_central(Mh)
    
    def phi_star_satellite(self, Mh: jnp.ndarray) -> jnp.ndarray:
        """Normalization of satellite Schechter function"""
        Mh = jnp.atleast_1d(Mh)
        m13 = Mh / 1e13
        log_phi = self._params_jax['b0'] + self._params_jax['b1'] * jnp.log10(m13)
        return 10**log_phi
    
    def N_central(self, Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float) -> jnp.ndarray:
        """
        Mean number of centrals in stellar mass bin [Mstar_min, Mstar_max].
        
        Parameters
        ----------
        Mh : array
            Halo mass in NATURAL units (Msun)
        Mstar_min, Mstar_max : float
            Stellar mass bin edges in NATURAL units (Msun)
        
        Returns
        -------
        N_cen : array
            Mean number of central galaxies
        """
        Mh = jnp.atleast_1d(Mh)
        Mstar_c = self.Mstar_central(Mh)
        
        log_Mstar_c = jnp.log10(Mstar_c)
        log_min = jnp.log10(Mstar_min)
        log_max = jnp.log10(Mstar_max)
        
        sqrt2_sigma = jnp.sqrt(2) * self._params_jax['sigma_c']
        
        erf_max = erf_jax((log_max - log_Mstar_c) / sqrt2_sigma)
        erf_min = erf_jax((log_min - log_Mstar_c) / sqrt2_sigma)
        
        return jnp.maximum(0.5 * (erf_max - erf_min), 0.0)
    
    def N_satellite(self, Mh: np.ndarray, Mstar_min: float, Mstar_max: float,
                    n_points: int = 100) -> np.ndarray:
        """
        Mean number of satellites in stellar mass bin.
        
        Note: Uses scipy for integration, returns numpy array.
        
        Parameters
        ----------
        Mh : array
            Halo mass in NATURAL units (Msun)
        Mstar_min, Mstar_max : float
            Stellar mass bin edges in NATURAL units (Msun)
        n_points : int
            Number of integration points
        
        Returns
        -------
        N_sat : array
            Mean number of satellite galaxies
        """
        Mh = np.atleast_1d(Mh)
        
        log_Mstar = np.linspace(np.log10(Mstar_min), np.log10(Mstar_max), n_points)
        Mstar = 10**log_Mstar
        
        Mstar_jax = jnp.array(Mstar)
        Mh_jax = jnp.array(Mh)
        
        Phi_s = self._Phi_satellite_grid(Mstar_jax, Mh_jax)
        
        Phi_s_np = np.array(Phi_s)
        integrand = Phi_s_np * Mstar[:, np.newaxis] * np.log(10)
        
        N_s = simpson(integrand, x=log_Mstar, axis=0)
        return np.maximum(N_s, 0)
    
    def _Phi_satellite_grid(self, Mstar: jnp.ndarray, Mh: jnp.ndarray) -> jnp.ndarray:
        """Satellite CSMF on grid"""
        Mstar_s = self.Mstar_satellite(Mh)
        phi_s = self.phi_star_satellite(Mh)
        
        if Mstar.shape != Mh.shape:
            Mstar = Mstar[:, jnp.newaxis] if Mstar.ndim == 1 else Mstar
            Mstar_s = Mstar_s[jnp.newaxis, :] if Mstar_s.ndim == 1 else Mstar_s
            phi_s = phi_s[jnp.newaxis, :] if phi_s.ndim == 1 else phi_s
        
        x = Mstar / Mstar_s
        return (phi_s / Mstar_s) * (x**self._params_jax['alpha_s']) * jnp.exp(-x**2)


# ============================================================================
# HOD Factory Function
# ============================================================================

def create_hod(hod_type: Union[HODType, str], params: Dict,
               units_per_h: bool = False, h: Optional[float] = None,
               **kwargs) -> Union[StandardHOD, CSMF_HOD]:
    """
    Factory function to create HOD models.
    
    Parameters
    ----------
    hod_type : HODType or str
        Type of HOD model ('standard' or 'csmf')
    params : dict
        HOD parameters
    units_per_h : bool
        If True, mass parameters are in h-units
    h : float
        Hubble parameter (required if units_per_h=True)
    **kwargs : dict
        Additional keyword arguments passed to HOD constructor
    
    Returns
    -------
    hod : StandardHOD or CSMF_HOD
        Initialized HOD model
    """
    if isinstance(hod_type, str):
        hod_type = HODType(hod_type.lower())
    
    if hod_type == HODType.STANDARD:
        return StandardHOD(params, units_per_h=units_per_h, h=h)
    elif hod_type == HODType.CSMF:
        masses_are_log10 = kwargs.get('masses_are_log10', True)
        return CSMF_HOD(params, masses_are_log10=masses_are_log10,
                        units_per_h=units_per_h, h=h)
    else:
        raise ValueError(f"Unknown HOD type: {hod_type}")


# ============================================================================
# Multi-Redshift Halo Model with Full Vmap Optimization
# ============================================================================

class MultiRedshiftHaloModel:
    """
    Maximum optimization halo model: vmap over ALL dimensions including redshift.
    
    This class pre-computes everything and uses a single vmap call
    to compute P(k,z) for all k and z simultaneously.
    
    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object
    hod_type : HODType or str
        Type of HOD model ('standard' or 'csmf')
    hod_params : dict
        HOD parameters (masses in h-units if units_per_h=True)
    z_array : array-like
        Redshifts to compute
    k_array : array-like, optional
        k values in NATURAL units (1/Mpc). Default: logspace(1e-3, 100, 256)
    M_min, M_max : float, optional
        Mass integration range in NATURAL units (Msun)
    n_M_points : int, optional
        Number of mass integration points
    units_per_h : bool
        If True, input HOD masses are in h-units and output will be in h-units
    masses_are_log10 : bool
        For CSMF: whether M0, M1 are log10 values (default True)
    verbose : bool
        Print timing information during initialization
    
    Attributes
    ----------
    k_array : ndarray
        k values in output units (h/Mpc if units_per_h else 1/Mpc)
    z_array : ndarray
        Redshift array
    """
    
    def __init__(self, cosmo, hod_type: Union[HODType, str], hod_params: Dict,
                 z_array, k_array: Optional[np.ndarray] = None,
                 M_min: float = 1e9, M_max: float = 1e16, n_M_points: int = 256,
                 units_per_h: bool = False, masses_are_log10: bool = True,
                 verbose: bool = True):
        
        self.cosmo = cosmo
        self.h = cosmo['h']
        self.units_per_h = units_per_h
        self.verbose = verbose
        
        # Parse HOD type
        if isinstance(hod_type, str):
            hod_type = HODType(hod_type.lower())
        self.hod_type = hod_type
        self.hod_params = hod_params.copy()
        
        # Create HOD model (handles unit conversion internally)
        self.hod = create_hod(
            hod_type, hod_params,
            units_per_h=units_per_h, h=self.h,
            masses_are_log10=masses_are_log10
        )
        
        # Store redshifts
        self.z_array = np.atleast_1d(z_array)
        self.a_array = 1.0 / (1.0 + self.z_array)
        self.n_z = len(self.z_array)
        
        # k array in NATURAL units (1/Mpc) for internal computation
        if k_array is None:
            self.k_array_natural = np.geomspace(1e-3, 100, 256)
        else:
            self.k_array_natural = np.atleast_1d(k_array)
        self.n_k = len(self.k_array_natural)
        
        # Mass array in NATURAL units (Msun)
        self.M_arr_natural = np.geomspace(M_min, M_max, n_M_points)
        self.log10M = np.log10(self.M_arr_natural)
        self.dlog10M = self.log10M[1] - self.log10M[0]
        self.n_M = n_M_points
        
        # JAX arrays
        self.M_arr_jax = jnp.array(self.M_arr_natural)
        self.log10M_jax = jnp.array(self.log10M)
        self.simpson_weights_M = get_simpson_weights(n_M_points)
        
        # Setup CCL components
        self.mass_def = ccl.halos.MassDef200c
        self.concentration = ccl.halos.ConcentrationDuffy08(mass_def=self.mass_def)
        self.mass_func = ccl.halos.MassFuncTinker08(mass_def=self.mass_def)
        self.halo_bias = ccl.halos.HaloBiasTinker10(mass_def=self.mass_def)
        self.nfw_profile = ccl.halos.HaloProfileNFW(
            mass_def=self.mass_def,
            concentration=self.concentration,
            fourier_analytic=True
        )

        # Comoving matter density (redshift-independent due to mass conservation)
        # Compute once at z=0 (a=1.0) in NATURAL units (Msun/Mpc³)
        self.RHO_M_comoving = ccl.rho_x(self.cosmo, 1.0, 'matter', is_comoving=True)

        # Pre-compute and build vmap functions
        if verbose:
            print("\nPre-computing all quantities...")
            t_total = time.time()
        
        t0 = time.time()
        self._precompute_redshift_quantities()
        if verbose:
            print(f"  Redshift quantities: {time.time()-t0:.3f}s")
        
        t0 = time.time()
        self._precompute_fourier_profiles()
        if verbose:
            print(f"  Fourier profiles: {time.time()-t0:.3f}s")
        
        t0 = time.time()
        self._build_vmap_functions()
        if verbose:
            print(f"  JIT/vmap compilation: {time.time()-t0:.3f}s")
            print(f"Total initialization: {time.time()-t_total:.3f}s\n")
        
        # Cache for occupation numbers (for CSMF with stellar mass bins)
        self._Ncen_cache = {}
        self._Nsat_cache = {}
    
    def _precompute_redshift_quantities(self):
        """Pre-compute n(M,z), b(M,z), P_lin(k,z) in NATURAL units"""
        self.n_M_z = np.zeros((self.n_z, self.n_M))
        self.b_h_z = np.zeros((self.n_z, self.n_M))
        self.Pk_lin_z = np.zeros((self.n_z, self.n_k))

        for i, a in enumerate(self.a_array):
            # All quantities in NATURAL units from CCL
            self.n_M_z[i] = self.mass_func(self.cosmo, self.M_arr_natural, a)  # Mpc⁻³
            self.b_h_z[i] = self.halo_bias(self.cosmo, self.M_arr_natural, a)  # dimensionless
            self.Pk_lin_z[i] = ccl.linear_power(self.cosmo, self.k_array_natural, a)  # Mpc³

        # Convert to JAX arrays
        self.n_M_z_jax = jnp.array(self.n_M_z)
        self.b_h_z_jax = jnp.array(self.b_h_z)
        self.Pk_lin_z_jax = jnp.array(self.Pk_lin_z)
    
    def _precompute_fourier_profiles(self):
        """Pre-compute NFW Fourier profiles in NATURAL units"""
        # Shape: (n_z, n_k, n_M)
        self.u_fourier = np.zeros((self.n_z, self.n_k, self.n_M))
        
        for iz, a in enumerate(self.a_array):
            for ik, k in enumerate(self.k_array_natural):
                self.u_fourier[iz, ik, :] = np.array([
                    self.nfw_profile._fourier(self.cosmo, k, M, a) 
                    for M in self.M_arr_natural
                ])
        
        # Derived profiles
        self.u_sat = self.u_fourier / self.M_arr_natural[np.newaxis, np.newaxis, :]
        self.u_matter = self.u_fourier / self.RHO_M_comoving  # Constant comoving density

        # Convert to JAX
        self.u_sat_jax = jnp.array(self.u_sat)
        self.u_matter_jax = jnp.array(self.u_matter)
    
    def _build_vmap_functions(self):
        """Build fully vmapped functions for P_gg and P_gm"""
        dlog10M = self.dlog10M
        weights = self.simpson_weights_M
        M_arr = self.M_arr_jax
        RHO_M = self.RHO_M_comoving  # Capture constant comoving density
        
        # P_gg kernel for single (k, z)
        @jit
        def _Pgg_single_k_z(N_c, N_s, u_sat, b_h, n_M, Pk_lin, n_gal):
            """P_gg for single k, single z"""
            # 1-halo: central-satellite
            integrand_cs = 2 * N_c * N_s * u_sat * n_M
            I_1h_cs = jnp.sum(integrand_cs * weights) * dlog10M
            
            # 1-halo: satellite-satellite
            integrand_ss = N_s**2 * u_sat**2 * n_M
            I_1h_ss = jnp.sum(integrand_ss * weights) * dlog10M
            
            P_1h = (I_1h_cs + I_1h_ss) / n_gal**2
            
            # 2-halo
            integrand_2h = (N_c + N_s * u_sat) * b_h * n_M
            I_2h = jnp.sum(integrand_2h * weights) * dlog10M
            P_2h = Pk_lin * (I_2h / n_gal)**2
            
            return P_1h + P_2h
        
        # vmap over k
        _Pgg_all_k_single_z = vmap(
            _Pgg_single_k_z, 
            in_axes=(None, None, 0, None, None, 0, None)
        )
        
        # vmap over z
        @jit
        def _Pgg_all_k_all_z(N_c, N_s, u_sat_all, b_h_all, n_M_all, Pk_lin_all, n_gal_all):
            """P_gg for all k, all z"""
            def single_z(u_sat_z, b_h_z, n_M_z, Pk_lin_z, n_gal_z):
                return _Pgg_all_k_single_z(N_c, N_s, u_sat_z, b_h_z, n_M_z, Pk_lin_z, n_gal_z)
            
            return vmap(single_z)(u_sat_all, b_h_all, n_M_all, Pk_lin_all, n_gal_all)
        
        self._Pgg_all = _Pgg_all_k_all_z
        
        # P_gm kernel for single (k, z)
        @jit
        def _Pgm_single_k_z(N_c, N_s, u_sat, u_matter, b_h, n_M, Pk_lin, n_gal):
            """P_gm for single k, single z"""
            # Matter normalization correction (using constant comoving density)
            integrand_M1 = (M_arr / RHO_M) * b_h * n_M
            I_M1 = 1.0 - jnp.sum(integrand_M1 * weights) * dlog10M

            # 1-halo
            integrand_1h = (N_c + N_s * u_sat) * u_matter * n_M
            P_1h = jnp.sum(integrand_1h * weights) * dlog10M / n_gal

            # 2-halo galaxy term
            integrand_g = (N_c + N_s * u_sat) * b_h * n_M
            I_2h_g = jnp.sum(integrand_g * weights) * dlog10M / n_gal

            # 2-halo matter term
            integrand_m = u_matter * b_h * n_M
            I_M2 = jnp.sum(integrand_m * weights) * dlog10M
            I_2h_m = I_M1 + I_M2

            P_2h = Pk_lin * I_2h_g * I_2h_m

            return P_1h + P_2h

        # vmap over k
        _Pgm_all_k_single_z = vmap(
            _Pgm_single_k_z,
            in_axes=(None, None, 0, 0, None, None, 0, None)
        )

        # vmap over z
        @jit
        def _Pgm_all_k_all_z(N_c, N_s, u_sat_all, u_matter_all, b_h_all, n_M_all,
                            Pk_lin_all, n_gal_all):
            def single_z(u_sat_z, u_matter_z, b_h_z, n_M_z, Pk_lin_z, n_gal_z):
                return _Pgm_all_k_single_z(
                    N_c, N_s, u_sat_z, u_matter_z, b_h_z, n_M_z, Pk_lin_z, n_gal_z
                )

            return vmap(single_z)(
                u_sat_all, u_matter_all, b_h_all, n_M_all, Pk_lin_all, n_gal_all
            )
        
        self._Pgm_all = _Pgm_all_k_all_z
        
        # Warmup JIT compilation
        dummy_Nc = jnp.ones(self.n_M)
        dummy_Ns = jnp.ones(self.n_M)
        dummy_ngal = jnp.ones(self.n_z) * 1e-3

        _ = self._Pgg_all(
            dummy_Nc, dummy_Ns,
            self.u_sat_jax, self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, dummy_ngal
        )

        _ = self._Pgm_all(
            dummy_Nc, dummy_Ns,
            self.u_sat_jax, self.u_matter_jax,
            self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, dummy_ngal
        )
    
    def _get_occupation_numbers(self, Mstar_min: Optional[float] = None,
                                 Mstar_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get occupation numbers with caching.
        
        For Standard HOD: Mstar_min/max are ignored
        For CSMF: Mstar_min/max define the stellar mass bin
        
        Parameters
        ----------
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
            If units_per_h=True, these should be in Msun/h²
        
        Returns
        -------
        N_c, N_s : arrays
            Central and satellite occupation numbers
        """
        if self.hod_type == HODType.CSMF:
            if Mstar_min is None or Mstar_max is None:
                raise ValueError("CSMF requires Mstar_min and Mstar_max")
            
            # Convert stellar masses to natural units if needed
            if self.units_per_h:
                Mstar_min_natural = Mstar_min * self.h**2  # Msun/h² -> Msun
                Mstar_max_natural = Mstar_max * self.h**2
            else:
                Mstar_min_natural = Mstar_min
                Mstar_max_natural = Mstar_max
            
            cache_key = (Mstar_min_natural, Mstar_max_natural)
        else:
            cache_key = (None, None)
        
        if cache_key in self._Ncen_cache:
            return self._Ncen_cache[cache_key], self._Nsat_cache[cache_key]
        
        if self.hod_type == HODType.CSMF:
            N_c = np.array(self.hod.N_central(self.M_arr_jax, Mstar_min_natural, Mstar_max_natural))
            N_s = self.hod.N_satellite(self.M_arr_natural, Mstar_min_natural, Mstar_max_natural)
        else:
            N_c = np.array(self.hod.N_central(self.M_arr_jax))
            N_s = np.array(self.hod.N_satellite(self.M_arr_jax))
        
        self._Ncen_cache[cache_key] = N_c
        self._Nsat_cache[cache_key] = N_s
        
        return N_c, N_s
    
    def compute_ngal_all_z(self, Mstar_min: Optional[float] = None,
                           Mstar_max: Optional[float] = None) -> np.ndarray:
        """
        Compute galaxy number density at all redshifts.
        
        Parameters
        ----------
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        
        Returns
        -------
        n_gal_z : array
            Galaxy number density at each redshift
            Units: (h/Mpc)³ if units_per_h else Mpc⁻³
        """
        N_c, N_s = self._get_occupation_numbers(Mstar_min, Mstar_max)
        N_tot = N_c + N_s
        
        # Compute in natural units
        n_gal_z_natural = np.sum(
            N_tot[np.newaxis, :] * self.n_M_z * np.array(self.simpson_weights_M)[np.newaxis, :],
            axis=1
        ) * self.dlog10M
        
        # Convert to h-units if requested
        if self.units_per_h:
            return n_gal_z_natural / self.h**3  # Mpc⁻³ -> (h/Mpc)³
        return n_gal_z_natural
    
    def compute_Pgg_all_z(self, Mstar_min: Optional[float] = None,
                          Mstar_max: Optional[float] = None,
                          verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P_gg(k,z) using full vmap over all k and z.
        
        Parameters
        ----------
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        verbose : bool, optional
            Print galaxy number densities (default: self.verbose)
        
        Returns
        -------
        Pk_gg_z : array, shape (n_z, n_k)
            Galaxy-galaxy power spectrum
            Units: (Mpc/h)³ if units_per_h else Mpc³
        n_gal_z : array, shape (n_z,)
            Galaxy number density at each redshift
        """
        if verbose is None:
            verbose = self.verbose
        
        N_c, N_s = self._get_occupation_numbers(Mstar_min, Mstar_max)
        N_c_jax = jnp.array(N_c)
        N_s_jax = jnp.array(N_s)
        
        # Compute n_gal in natural units for internal calculation
        N_tot = N_c + N_s
        n_gal_z_natural = np.sum(
            N_tot[np.newaxis, :] * self.n_M_z * np.array(self.simpson_weights_M)[np.newaxis, :],
            axis=1
        ) * self.dlog10M
        n_gal_z_jax = jnp.array(n_gal_z_natural)
        
        if verbose:
            n_unit = "(h/Mpc)³" if self.units_per_h else "Mpc⁻³"
            print("Galaxy number densities:")
            n_gal_display = n_gal_z_natural / self.h**3 if self.units_per_h else n_gal_z_natural
            for iz, (z, ng) in enumerate(zip(self.z_array, n_gal_display)):
                print(f"  z={z:.3f}: n_gal = {ng:.6e} {n_unit}")
        
        # Compute P_gg in natural units (Mpc³)
        Pk_gg_z_natural = np.array(self._Pgg_all(
            N_c_jax, N_s_jax,
            self.u_sat_jax, self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, n_gal_z_jax
        ))
        
        # Convert output if requested
        if self.units_per_h:
            Pk_gg_z = Pk_gg_z_natural * self.h**3  # Mpc³ -> (Mpc/h)³
            n_gal_z = n_gal_z_natural / self.h**3
        else:
            Pk_gg_z = Pk_gg_z_natural
            n_gal_z = n_gal_z_natural
        
        return Pk_gg_z, n_gal_z
    
    def compute_Pgm_all_z(self, Mstar_min: Optional[float] = None,
                          Mstar_max: Optional[float] = None,
                          verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P_gm(k,z) using full vmap over all k and z.
        
        Parameters
        ----------
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        verbose : bool, optional
            Print galaxy number densities (default: self.verbose)
        
        Returns
        -------
        Pk_gm_z : array, shape (n_z, n_k)
            Galaxy-matter power spectrum
            Units: (Mpc/h)³ if units_per_h else Mpc³
        n_gal_z : array, shape (n_z,)
            Galaxy number density at each redshift
        """
        if verbose is None:
            verbose = self.verbose
        
        N_c, N_s = self._get_occupation_numbers(Mstar_min, Mstar_max)
        N_c_jax = jnp.array(N_c)
        N_s_jax = jnp.array(N_s)
        
        # Compute n_gal in natural units
        N_tot = N_c + N_s
        n_gal_z_natural = np.sum(
            N_tot[np.newaxis, :] * self.n_M_z * np.array(self.simpson_weights_M)[np.newaxis, :],
            axis=1
        ) * self.dlog10M
        n_gal_z_jax = jnp.array(n_gal_z_natural)
        
        if verbose:
            n_unit = "(h/Mpc)³" if self.units_per_h else "Mpc⁻³"
            print("Galaxy number densities:")
            n_gal_display = n_gal_z_natural / self.h**3 if self.units_per_h else n_gal_z_natural
            for iz, (z, ng) in enumerate(zip(self.z_array, n_gal_display)):
                print(f"  z={z:.3f}: n_gal = {ng:.6e} {n_unit}")
        
        # Compute P_gm in natural units
        Pk_gm_z_natural = np.array(self._Pgm_all(
            N_c_jax, N_s_jax,
            self.u_sat_jax, self.u_matter_jax,
            self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, n_gal_z_jax
        ))
        
        # Convert output if requested
        if self.units_per_h:
            Pk_gm_z = Pk_gm_z_natural * self.h**3
            n_gal_z = n_gal_z_natural / self.h**3
        else:
            Pk_gm_z = Pk_gm_z_natural
            n_gal_z = n_gal_z_natural
        
        return Pk_gm_z, n_gal_z
    
    def compute_both_all_z(self, Mstar_min: Optional[float] = None,
                           Mstar_max: Optional[float] = None,
                           verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute both P_gg and P_gm efficiently at all redshifts.
        
        Parameters
        ----------
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        verbose : bool, optional
            Print galaxy number densities (default: self.verbose)
        
        Returns
        -------
        Pk_gg_z : array, shape (n_z, n_k)
            Galaxy-galaxy power spectrum
        Pk_gm_z : array, shape (n_z, n_k)
            Galaxy-matter power spectrum
        n_gal_z : array, shape (n_z,)
            Galaxy number density at each redshift
        """
        if verbose is None:
            verbose = self.verbose
        
        N_c, N_s = self._get_occupation_numbers(Mstar_min, Mstar_max)
        N_c_jax = jnp.array(N_c)
        N_s_jax = jnp.array(N_s)
        
        # Compute n_gal in natural units
        N_tot = N_c + N_s
        n_gal_z_natural = np.sum(
            N_tot[np.newaxis, :] * self.n_M_z * np.array(self.simpson_weights_M)[np.newaxis, :],
            axis=1
        ) * self.dlog10M
        n_gal_z_jax = jnp.array(n_gal_z_natural)
        
        if verbose:
            n_unit = "(h/Mpc)³" if self.units_per_h else "Mpc⁻³"
            print("Galaxy number densities:")
            n_gal_display = n_gal_z_natural / self.h**3 if self.units_per_h else n_gal_z_natural
            for iz, (z, ng) in enumerate(zip(self.z_array, n_gal_display)):
                print(f"  z={z:.3f}: n_gal = {ng:.6e} {n_unit}")
        
        # Compute both in natural units
        Pk_gg_z_natural = np.array(self._Pgg_all(
            N_c_jax, N_s_jax,
            self.u_sat_jax, self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, n_gal_z_jax
        ))
        
        Pk_gm_z_natural = np.array(self._Pgm_all(
            N_c_jax, N_s_jax,
            self.u_sat_jax, self.u_matter_jax,
            self.b_h_z_jax, self.n_M_z_jax,
            self.Pk_lin_z_jax, n_gal_z_jax
        ))
        
        # Convert output if requested
        if self.units_per_h:
            Pk_gg_z = Pk_gg_z_natural * self.h**3
            Pk_gm_z = Pk_gm_z_natural * self.h**3
            n_gal_z = n_gal_z_natural / self.h**3
        else:
            Pk_gg_z = Pk_gg_z_natural
            Pk_gm_z = Pk_gm_z_natural
            n_gal_z = n_gal_z_natural
        
        return Pk_gg_z, Pk_gm_z, n_gal_z

    def compute_wgg_all_z(self, rp_bins: np.ndarray,
                          bin_avg: bool = False,
                          Mstar_min: Optional[float] = None,
                          Mstar_max: Optional[float] = None,
                          verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute projected galaxy clustering w_gg(r_p) at all redshifts.

        Parameters
        ----------
        rp_bins : array
            Projected separation bins [Mpc/h or Mpc]
        bin_avg : bool, default=False
            If True, average w_gg within each bin (like DeltaSigmaCalculator.compute_deltasigma_averaged)
            If False, evaluate at bin centers
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        verbose : bool, optional
            Print information

        Returns
        -------
        rp : array, shape (n_bins,) or (n_bins-1,)
            Projected radii. If bin_avg=False: bin centers. If bin_avg=True: n_bins-1
        wgg_z : array, shape (n_z, n_bins) or (n_z, n_bins-1)
            Projected correlation function at each redshift

        Notes
        -----
        Uses FAST-PT's direct Hankel transform: P_gg(k) -> w_gg(r_p)
        """
        if verbose is None:
            verbose = self.verbose

        # Get P_gg at all redshifts in appropriate units
        Pk_gg_z, n_gal_z = self.compute_Pgg_all_z(
            Mstar_min=Mstar_min, Mstar_max=Mstar_max, verbose=verbose
        )

        # Get k array in appropriate units
        k = self.get_k_array()

        # Compute w_gg for each redshift
        wgg_z_list = []

        for iz in range(self.n_z):
            # Transform P_gg(k) -> w_gg(r) using FAST-PT
            r_wgg, wgg_full = Pk_to_wgg_direct(k, Pk_gg_z[iz])

            if bin_avg:
                # Average w_gg within each bin
                # Create spline interpolation
                spline_wgg = interp1d(r_wgg, wgg_full, bounds_error=False,
                                     kind='cubic', fill_value=(wgg_full[0], wgg_full[-1]))

                # Integrand for averaging: w_gg(r) * r
                def wgg_integrand(r):
                    return spline_wgg(r) * r

                # Average over each bin: <w_gg> = (2/Δr²) ∫ w_gg(r) r dr
                diff_sq = np.diff(rp_bins**2)
                wgg_averaged = 2 * gauss_legendre_integration(
                    wgg_integrand, rp_bins[:-1], rp_bins[1:]
                ) / diff_sq

                wgg_z_list.append(wgg_averaged)
            else:
                # Evaluate at bin centers
                rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
                wgg_interp = np.interp(rp_centers, r_wgg, wgg_full)
                wgg_z_list.append(wgg_interp)

        wgg_z = np.array(wgg_z_list)

        # Determine output radii
        if bin_avg:
            rp = rp_bins[:-1]  # Return lower bin edges (n_bins-1)
        else:
            rp = np.sqrt(rp_bins[:-1] * rp_bins[1:])  # Bin centers

        return rp, wgg_z

    def compute_DeltaSigma_all_z(self, rp_bins: np.ndarray,
                                 method: str = 'direct',
                                 bin_avg: bool = False,
                                 chi_max: float = 100.0,
                                 Mstar_min: Optional[float] = None,
                                 Mstar_max: Optional[float] = None,
                                 verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute galaxy-galaxy lensing ΔΣ(R) at all redshifts.

        Parameters
        ----------
        rp_bins : array
            Projected separation bins [Mpc/h or Mpc]
        method : str, default='direct'
            'direct' - Direct Hankel transform (Method 2, faster)
            'traditional' - Via xi_gm integration (Method 1, matches numerical code)
        bin_avg : bool, default=False
            If True, average ΔΣ within each bin (like DeltaSigmaCalculator.compute_deltasigma_averaged)
            If False, evaluate at bin centers
        chi_max : float, default=100.0
            Maximum line-of-sight distance for traditional method [Mpc/h or Mpc]
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (required for CSMF)
        verbose : bool, optional
            Print information

        Returns
        -------
        rp : array, shape (n_bins,) or (n_bins-1,)
            Projected radii. If bin_avg=False: bin centers. If bin_avg=True: n_bins-1
        DeltaSigma_z : array, shape (n_z, n_bins) or (n_z, n_bins-1)
            Surface mass density contrast [Msun h/pc²] at each redshift

        Notes
        -----
        Method 'direct': Uses FAST-PT direct transform P_gm(k) -> ΔΣ(R)
        Method 'traditional': P_gm(k) -> xi_gm(r) -> ΔΣ(R) (matches numerical code)
        """
        if verbose is None:
            verbose = self.verbose

        if method not in ['direct', 'traditional']:
            raise ValueError(f"Unknown method: {method}. Use 'direct' or 'traditional'")

        # Get P_gm at all redshifts in appropriate units
        Pk_gm_z, n_gal_z = self.compute_Pgm_all_z(
            Mstar_min=Mstar_min, Mstar_max=Mstar_max, verbose=verbose
        )

        # Get k array and RHO_M in appropriate units
        k = self.get_k_array()
        rho_m = self.get_RHO_M()

        # Compute ΔΣ for each redshift
        ds_z_list = []

        for iz in range(self.n_z):
            if method == 'direct':
                # Method 2: Direct Hankel transform
                r_ds, ds_full = Pk_to_DeltaSigma_direct(k, Pk_gm_z[iz], rho_m)
                # Output is in [(Msun/h) / (Mpc/h)²] or [Msun/Mpc²]
                # Convert to [Msun h/pc²]: divide by 1e12 (Mpc² -> pc²)
                ds_full = ds_full / 1e12

            else:  # method == 'traditional'
                # Method 1: Via xi_gm integration
                # Determine output radii for interpolation
                if bin_avg:
                    # Need to compute at fine grid for later averaging
                    r_output = rp_bins
                else:
                    # Compute at bin centers
                    r_output = np.sqrt(rp_bins[:-1] * rp_bins[1:])

                # This already returns in [Msun h/pc²]
                ds_full = Pk_gm_to_DeltaSigma_traditional(
                    k, Pk_gm_z[iz], r_output, rho_m, chi_max=chi_max
                )
                r_ds = r_output

            if bin_avg:
                # Average ΔΣ within each bin
                # Create spline interpolation
                spline_ds = interp1d(r_ds, ds_full, bounds_error=False,
                                    kind='cubic', fill_value=(ds_full[0], ds_full[-1]))

                # Integrand for averaging: ΔΣ(r) * r
                def ds_integrand(r):
                    return spline_ds(r) * r

                # Average over each bin: <ΔΣ> = (2/Δr²) ∫ ΔΣ(r) r dr
                diff_sq = np.diff(rp_bins**2)
                ds_averaged = 2 * gauss_legendre_integration(
                    ds_integrand, rp_bins[:-1], rp_bins[1:]
                ) / diff_sq

                ds_z_list.append(ds_averaged)
            else:
                # Evaluate at bin centers (or already at bin centers for traditional method)
                if method == 'direct':
                    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
                    ds_interp = np.interp(rp_centers, r_ds, ds_full)
                    ds_z_list.append(ds_interp)
                else:
                    # Already at bin centers for traditional method
                    ds_z_list.append(ds_full)

        ds_z = np.array(ds_z_list)

        # Determine output radii
        if bin_avg:
            rp = rp_bins[:-1]  # Return lower bin edges (n_bins-1)
        else:
            rp = np.sqrt(rp_bins[:-1] * rp_bins[1:])  # Bin centers

        return rp, ds_z

    def get_k_array(self) -> np.ndarray:
        """
        Return k array in appropriate units.
        
        Returns
        -------
        k : array
            k values in h/Mpc if units_per_h else 1/Mpc
        """
        if self.units_per_h:
            return self.k_array_natural / self.h  # 1/Mpc -> h/Mpc
        return self.k_array_natural
    
    def get_M_array(self) -> np.ndarray:
        """
        Return mass array in appropriate units.

        Returns
        -------
        M : array
            Masses in Msun/h if units_per_h else Msun
        """
        if self.units_per_h:
            return self.M_arr_natural * self.h  # Msun -> Msun/h
        return self.M_arr_natural

    def get_RHO_M(self) -> float:
        """
        Return comoving matter density in appropriate units.

        Returns
        -------
        RHO_M : float
            Comoving matter density
            [(Msun/h) / (Mpc/h)³] if units_per_h else [Msun/Mpc³]

        Notes
        -----
        The comoving matter density is constant across redshift due to
        mass conservation in comoving coordinates.
        """
        if self.units_per_h:
            # [(Msun/h) / (Mpc/h)³] = [Msun/Mpc³] / h²
            return self.RHO_M_comoving / self.h**2
        return self.RHO_M_comoving

    def clear_cache(self):
        """Clear the occupation number cache"""
        self._Ncen_cache = {}
        self._Nsat_cache = {}
    
    def update_hod_params(self, new_params: Dict, Mstar_min: Optional[float] = None,
                          Mstar_max: Optional[float] = None):
        """
        Update HOD parameters without reinitializing CCL quantities.
        
        This is efficient for MCMC sampling where only HOD parameters change.
        
        Parameters
        ----------
        new_params : dict
            New HOD parameters
        Mstar_min, Mstar_max : float, optional
            Stellar mass bin edges (for CSMF)
        """
        self.hod_params = new_params.copy()
        
        # Recreate HOD with new parameters
        self.hod = create_hod(
            self.hod_type, new_params,
            units_per_h=self.units_per_h, h=self.h,
            masses_are_log10=True
        )
        
        # Clear cache to force recomputation
        self.clear_cache()


# ============================================================================
# Convenience Function for MCMC
# ============================================================================

def compute_power_spectra(cosmo, hod_type: str, hod_params: Dict,
                          z_array, k_array: Optional[np.ndarray] = None,
                          Mstar_min: Optional[float] = None,
                          Mstar_max: Optional[float] = None,
                          units_per_h: bool = False,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to compute power spectra.
    
    This creates a new model each time, so it's less efficient than
    reusing a MultiRedshiftHaloModel instance for MCMC.
    
    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology
    hod_type : str
        'standard' or 'csmf'
    hod_params : dict
        HOD parameters
    z_array : array
        Redshifts
    k_array : array, optional
        k values (1/Mpc)
    Mstar_min, Mstar_max : float, optional
        Stellar mass bin (for CSMF)
    units_per_h : bool
        If True, use h-units
    **kwargs : dict
        Additional arguments for MultiRedshiftHaloModel
    
    Returns
    -------
    k : array
        k values
    Pk_gg : array
        Galaxy-galaxy power spectrum
    Pk_gm : array
        Galaxy-matter power spectrum
    n_gal : array
        Galaxy number density
    """
    model = MultiRedshiftHaloModel(
        cosmo, hod_type, hod_params, z_array,
        k_array=k_array, units_per_h=units_per_h,
        verbose=False, **kwargs
    )
    
    Pk_gg, Pk_gm, n_gal = model.compute_both_all_z(
        Mstar_min=Mstar_min, Mstar_max=Mstar_max, verbose=False
    )
    
    return model.get_k_array(), Pk_gg, Pk_gm, n_gal


# ============================================================================
# Unit Conversion Summary
# ============================================================================

UNIT_CONVERSION_INFO = """
Unit Conversion Summary
=======================

pyccl works entirely in NATURAL units (Msun, Mpc, 1/Mpc).

Converting from h-units to natural (multiply by h):
===================================================
    M [Msun]      = M [Msun/h] * h
    M* [Msun]     = M* [Msun/h²] * h²
    k [1/Mpc]     = k [h/Mpc] * h

For log10 quantities:
    log10(M [Msun]) = log10(M [Msun/h]) + log10(h)
    log10(M* [Msun]) = log10(M* [Msun/h²]) + 2*log10(h)

Converting from natural to h-units (divide by h):
=================================================
    k [h/Mpc]           = k [1/Mpc] / h
    P(k) [(Mpc/h)³]     = P(k) [Mpc³] * h³
    n_gal [(h/Mpc)³]    = n_gal [Mpc⁻³] / h³

When units_per_h=True, the model:
1. CONVERTS INPUT: HOD masses from Msun/h to Msun (multiply by h)
2. CONVERTS OUTPUT: k, P(k), n_gal from natural to h-units
"""


if __name__ == "__main__":
    print(UNIT_CONVERSION_INFO)