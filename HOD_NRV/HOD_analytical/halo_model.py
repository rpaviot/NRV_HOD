"""
Optimized Halo Model Power Spectrum Calculator with Full JAX/Vmap
=================================================================

Key Design Principles:
- z_array defines the exact redshifts for output (no interpolation)
- Gauss-Legendre integration everywhere (no trapezoid)
- Pure JAX functions for JIT/vmap compatibility
- numpy only for pyccl calls, JAX for everything else
- Single z → squeezed output shapes
- Multi z → Mstar arrays must match z_array length

Unit Conversion Strategy:
    pyccl computes in natural units (Msun, Mpc, 1/Mpc)
    When units_per_h=True: outputs converted to h-units
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf as erf_jax
from functools import partial
import pyccl as ccl
from scipy.special import roots_legendre
from enum import Enum
from typing import Dict, Optional, Union, Tuple, Callable

# Import Hankel transform utilities
from HOD_NRV.utilsf.hankel_transforms import (
    Pk_to_wgg_direct,
    Pk_to_DeltaSigma_direct,
    Pk_gm_to_DeltaSigma_traditional,
)

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Gauss-Legendre Integration Utilities
# ============================================================================

N_GL = 200
_x_gl, _w_gl = roots_legendre(N_GL)
GL_X = jnp.array(_x_gl)  # Nodes in [-1, 1]
GL_W = jnp.array(_w_gl)  # Weights


@jit
def gl_nodes_scaled(a: float, b: float) -> jnp.ndarray:
    """
    Scale GL nodes from [-1, 1] to [a, b].
    
    Parameters
    ----------
    a, b : float
        Integration bounds
    
    Returns
    -------
    nodes : array, shape (N_GL,)
        Scaled nodes in [a, b]
    
    Example
    -------
    >>> log10M_nodes = gl_nodes_scaled(9.0, 16.0)  # Mass range 1e9 to 1e16
    >>> M_nodes = 10.0 ** log10M_nodes
    """
    return 0.5 * ((b - a) * GL_X + (a + b))


@jit
def gl_integrate_precomputed(integrand: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """
    Gauss-Legendre integration when integrand is already evaluated at GL nodes.
    
    This is the fast path: just a dot product with the weights.
    
    Parameters
    ----------
    integrand : array, shape (N_GL,) or (N_GL, ...)
        Integrand values evaluated at GL nodes (from gl_nodes_scaled)
    a, b : float
        Integration bounds
    
    Returns
    -------
    integral : float or array
        Result of ∫_a^b f(x) dx
    
    Example
    -------
    >>> # Precompute at GL nodes
    >>> log10M_nodes = gl_nodes_scaled(9.0, 16.0)
    >>> n_M = mass_function(10**log10M_nodes)  # shape (N_GL,)
    >>> N_c = occupation(10**log10M_nodes)      # shape (N_GL,)
    >>> 
    >>> # Integrate: just a dot product!
    >>> n_gal = gl_integrate_precomputed(N_c * n_M, 9.0, 16.0)
    """
    return 0.5 * (b - a) * jnp.dot(GL_W, integrand)


def make_gl_integrator(f: Callable) -> Callable:
    """
    Create a JIT-compiled Gauss-Legendre integrator for function f.
    
    Use this when you need to integrate a function that is NOT precomputed
    at GL nodes (e.g., integrating over stellar mass in CSMF).
    
    Parameters
    ----------
    f : callable
        Function to integrate. Must be JAX-compatible (jittable).
        Signature: f(x, *args) where x has shape (N_GL,)
    
    Returns
    -------
    integrate : callable
        JIT-compiled function with signature: integrate(a, b, *args) -> float
    
    Example
    -------
    >>> @jit
    ... def gaussian(x, mu, sigma):
    ...     return jnp.exp(-0.5 * ((x - mu) / sigma)**2)
    >>> 
    >>> integrate_gaussian = make_gl_integrator(gaussian)
    >>> 
    >>> # Integrate N(0,1) from -3 to 3
    >>> result = integrate_gaussian(-3.0, 3.0, 0.0, 1.0)
    >>> print(f"Result: {result:.6f}")  # ~2.507 (unnormalized)
    """
    @jit
    def integrate(a: float, b: float, *args):
        x_scaled = gl_nodes_scaled(a, b)
        f_values = f(x_scaled, *args)
        return 0.5 * (b - a) * jnp.dot(GL_W, f_values)
    
    return integrate


# ============================================================================
# HOD Types and Validation
# ============================================================================

class HODType(Enum):
    STANDARD = "standard"
    CSMF = "csmf"


STANDARD_HOD_PARAMS = ['log10Mmin', 'siglnM', 'log10M0', 'log10M1', 'alpha']
CSMF_HOD_PARAMS = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']


def validate_hod_params(hod_type: HODType, params: Dict) -> bool:
    """Validate that required HOD parameters are present."""
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
# Pure JIT Functions for HOD (no self, maximum efficiency)
# ============================================================================

# --- Standard HOD ---

@jit
def standard_N_central(M: jnp.ndarray, Mmin: float, sigma_lnM: float) -> jnp.ndarray:
    """
    Central occupation for Standard HOD.
    
    N_cen(M) = 0.5 * [1 + erf((ln(M) - ln(Mmin)) / sigma_lnM)]
    """
    x = jnp.log(M / Mmin) / sigma_lnM
    return 0.5 * (1.0 + erf_jax(x))


@jit
def standard_N_satellite(M: jnp.ndarray, M0: float, M1: float, alpha: float) -> jnp.ndarray:
    """
    Satellite occupation for Standard HOD.
    
    N_sat(M) = [(M - M0) / M1]^alpha  for M > M0, else 0
    """
    return jnp.where(M > M0, ((M - M0) / M1) ** alpha, 0.0)


# --- CSMF HOD ---

@jit
def csmf_Mstar_central(Mh: jnp.ndarray, M0: float, M1: float,
                        gamma1: float, gamma2: float) -> jnp.ndarray:
    """Stellar-to-halo mass relation for centrals."""
    x = Mh / M1
    return M0 * (x ** gamma1) / ((1 + x) ** (gamma1 - gamma2))


@jit
def csmf_N_central(Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float,
                   M0: float, M1: float, gamma1: float, gamma2: float,
                   sigma_c: float) -> jnp.ndarray:
    """
    Central occupation for CSMF in stellar mass bin.
    
    Integrates lognormal distribution over [Mstar_min, Mstar_max].
    """
    Mstar_c = csmf_Mstar_central(Mh, M0, M1, gamma1, gamma2)
    log_Mstar_c = jnp.log10(Mstar_c)
    sqrt2_sigma = jnp.sqrt(2.0) * sigma_c
    
    erf_max = erf_jax((jnp.log10(Mstar_max) - log_Mstar_c) / sqrt2_sigma)
    erf_min = erf_jax((jnp.log10(Mstar_min) - log_Mstar_c) / sqrt2_sigma)
    
    return jnp.maximum(0.5 * (erf_max - erf_min), 0.0)


@jit
def _csmf_satellite_integrand(log10_Mstar: jnp.ndarray, Mh: float,
                               M0: float, M1: float, gamma1: float, gamma2: float,
                               alpha_s: float, b0: float, b1: float) -> jnp.ndarray:
    """
    Integrand for CSMF satellite occupation: Φ(M*|Mh) * M* * ln(10).
    
    Parameters
    ----------
    log10_Mstar : array, shape (N_GL,)
        Log10 stellar mass at GL nodes
    Mh : float
        Single halo mass
    ... : HOD parameters
    
    Returns
    -------
    integrand : array, shape (N_GL,)
    """
    Mstar = 10.0 ** log10_Mstar
    
    # Characteristic stellar mass
    Mstar_c = csmf_Mstar_central(Mh, M0, M1, gamma1, gamma2)
    Mstar_s = 0.56 * Mstar_c
    
    # Normalization
    log_phi = b0 + b1 * jnp.log10(Mh / 1e13)
    phi_s = 10.0 ** log_phi
    
    # Schechter-like function
    x = Mstar / Mstar_s
    Phi = (phi_s / Mstar_s) * (x ** alpha_s) * jnp.exp(-x**2)
    
    # Jacobian for d(log10 M*) integration
    return Phi * Mstar * jnp.log(10.0)


# Create the integrator for satellite occupation
_integrate_csmf_satellite = make_gl_integrator(_csmf_satellite_integrand)


@jit
def csmf_N_satellite_single(Mh: float, Mstar_min: float, Mstar_max: float,
                            M0: float, M1: float, gamma1: float, gamma2: float,
                            alpha_s: float, b0: float, b1: float) -> float:
    """
    Satellite occupation for CSMF at single halo mass.
    
    Integrates Schechter function over [Mstar_min, Mstar_max].
    """
    log10_Mstar_min = jnp.log10(Mstar_min)
    log10_Mstar_max = jnp.log10(Mstar_max)
    
    N_s = _integrate_csmf_satellite(
        log10_Mstar_min, log10_Mstar_max,
        Mh, M0, M1, gamma1, gamma2, alpha_s, b0, b1
    )
    return jnp.maximum(N_s, 0.0)


# vmap over halo mass array
@jit
def csmf_N_satellite(Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float,
                     M0: float, M1: float, gamma1: float, gamma2: float,
                     alpha_s: float, b0: float, b1: float) -> jnp.ndarray:
    """
    Satellite occupation for CSMF over halo mass array.
    
    Parameters
    ----------
    Mh : array, shape (n_M,)
        Halo masses
    Mstar_min, Mstar_max : float
        Stellar mass bin edges
    ... : HOD parameters
    
    Returns
    -------
    N_sat : array, shape (n_M,)
    """
    return vmap(
        lambda m: csmf_N_satellite_single(
            m, Mstar_min, Mstar_max, M0, M1, gamma1, gamma2, alpha_s, b0, b1
        )
    )(Mh)


# ============================================================================
# HOD Wrapper Classes
# ============================================================================

class StandardHOD:
    """Standard Zheng et al. HOD model wrapper."""
    
    def __init__(self, params: Dict):
        validate_hod_params(HODType.STANDARD, params)
        self.hod_type = HODType.STANDARD
        self.params = params.copy()
        
        # Store as JAX scalars
        self.Mmin = jnp.array(10.0 ** params['log10Mmin'])
        self.sigma_lnM = jnp.array(params['siglnM'])
        self.M0 = jnp.array(10.0 ** params['log10M0'])
        self.M1 = jnp.array(10.0 ** params['log10M1'])
        self.alpha = jnp.array(params['alpha'])
    
    def N_central(self, M: jnp.ndarray) -> jnp.ndarray:
        return standard_N_central(M, self.Mmin, self.sigma_lnM)
    
    def N_satellite(self, M: jnp.ndarray) -> jnp.ndarray:
        return standard_N_satellite(M, self.M0, self.M1, self.alpha)


class CSMF_HOD:
    """CSMF HOD model wrapper (Dvornik+2022)."""
    
    def __init__(self, params: Dict, masses_are_log10: bool = True):
        validate_hod_params(HODType.CSMF, params)
        self.hod_type = HODType.CSMF
        self.params = params.copy()
        
        # Convert masses
        if masses_are_log10:
            self.M0 = jnp.array(10.0 ** params['M0'])
            self.M1 = jnp.array(10.0 ** params['M1'])
        else:
            self.M0 = jnp.array(params['M0'])
            self.M1 = jnp.array(params['M1'])
        
        self.gamma1 = jnp.array(params['gamma1'])
        self.gamma2 = jnp.array(params['gamma2'])
        self.sigma_c = jnp.array(params['sigma_c'])
        self.alpha_s = jnp.array(params['alpha_s'])
        self.b0 = jnp.array(params['b0'])
        self.b1 = jnp.array(params['b1'])
    
    def N_central(self, Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float) -> jnp.ndarray:
        return csmf_N_central(
            Mh, Mstar_min, Mstar_max,
            self.M0, self.M1, self.gamma1, self.gamma2, self.sigma_c
        )
    
    def N_satellite(self, Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float) -> jnp.ndarray:
        return csmf_N_satellite(
            Mh, Mstar_min, Mstar_max,
            self.M0, self.M1, self.gamma1, self.gamma2,
            self.alpha_s, self.b0, self.b1
        )


def create_hod(hod_type: Union[HODType, str], params: Dict,
               **kwargs) -> Union[StandardHOD, CSMF_HOD]:
    """Factory function for HOD models."""
    if isinstance(hod_type, str):
        hod_type = HODType(hod_type.lower())
    
    if hod_type == HODType.STANDARD:
        return StandardHOD(params)
    elif hod_type == HODType.CSMF:
        return CSMF_HOD(params, masses_are_log10=kwargs.get('masses_are_log10', True))
    else:
        raise ValueError(f"Unknown HOD type: {hod_type}")


# ============================================================================
# Multi-Redshift Halo Model
# ============================================================================

class MultiRedshiftHaloModel:
    """
    Optimized halo model with Gauss-Legendre integration.
    
    Design
    ------
    - z_array defines exact output redshifts (no interpolation)
    - Single z: output shapes are squeezed (n_k,) instead of (1, n_k)
    - Multi z with CSMF: Mstar arrays must have length == len(z_array)
    - All mass integrals use GL quadrature (precomputed at GL nodes)
    
    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object
    hod_type : HODType or str
        'standard' or 'csmf'
    hod_params : dict
        HOD parameters
    z_array : float or array
        Redshift(s) for output
    k_array : array, optional
        k values in natural units (1/Mpc)
    M_min, M_max : float
        Mass integration range in natural units (Msun)
    units_per_h : bool
        If True, output in h-units
    masses_are_log10 : bool
        For CSMF: whether M0, M1 are log10 values
    verbose : bool
        Print timing info
    median_Mstar : float or array, optional
        Median stellar mass for stellar point mass GGL contribution.
        Only available for CSMF models.
        - If scalar: same stellar mass for all redshifts
        - If array: must have length == len(z_array)
        - Units: [Msun/h²] (consistent with CSMF M0, M1)
        - If None: no stellar component added (default)
    """
    
    def __init__(self, cosmo, hod_type: Union[HODType, str], hod_params: Dict,
                 z_array, k_array: Optional[np.ndarray] = None,
                 M_min: float = 1e9, M_max: float = 1e16,
                 units_per_h: bool = False, masses_are_log10: bool = True,
                 verbose: bool = True, median_Mstar: Optional[Union[float, np.ndarray]] = None):
        
        self.cosmo = cosmo
        self.h = cosmo['h']
        self.units_per_h = units_per_h
        self.verbose = verbose
        
        # Parse HOD
        if isinstance(hod_type, str):
            hod_type = HODType(hod_type.lower())
        self.hod_type = hod_type
        self.hod_params = hod_params.copy()
        self.masses_are_log10 = masses_are_log10
        self.hod = create_hod(hod_type, hod_params, masses_are_log10=masses_are_log10)

        # Redshift setup
        self.z_array = np.atleast_1d(z_array)
        self.is_single_z = (len(self.z_array) == 1)
        self.n_z = len(self.z_array)
        self.a_array = 1.0 / (1.0 + self.z_array)

        # Validate median_Mstar parameter (must come after z_array setup)
        if median_Mstar is not None:
            if self.hod_type != HODType.CSMF:
                raise ValueError(
                    "median_Mstar parameter is only available for CSMF models. "
                    "Standard HOD does not include stellar mass modeling."
                )

            # Convert to array and validate shape
            median_Mstar_arr = np.atleast_1d(median_Mstar)
            if len(median_Mstar_arr) == 1:
                # Scalar: broadcast to all z
                self.median_Mstar = np.full(self.n_z, float(median_Mstar_arr[0]))
            elif len(median_Mstar_arr) == self.n_z:
                # Array: use as-is
                self.median_Mstar = np.array(median_Mstar_arr)
            else:
                raise ValueError(
                    f"median_Mstar array length ({len(median_Mstar_arr)}) must be "
                    f"1 (scalar) or match z_array length ({self.n_z})"
                )
        else:
            self.median_Mstar = None

        # Initialize cache for stellar component DeltaSigma
        self._DeltaSigma_stellar_cache = {}
        
        # k array (natural units)
        if k_array is None:
            self.k_array_natural = np.geomspace(1e-5, 100, 2048)
        else:
            self.k_array_natural = np.atleast_1d(k_array)
        self.n_k = len(self.k_array_natural)
        
        # Mass array at GL nodes
        self.log10M_min = np.log10(M_min)
        self.log10M_max = np.log10(M_max)
        
        # Get GL nodes in log10(M) space
        log10M_gl = np.array(gl_nodes_scaled(self.log10M_min, self.log10M_max))
        self.M_arr_natural = 10.0 ** log10M_gl  # (N_GL,)
        self.n_M = N_GL
        
        # JAX mass array (with unit conversion)
        #if self.units_per_h:
        #    self.M_arr_jax = jnp.array(self.M_arr_natural * self.h)
        #else:
        #    self.M_arr_jax = jnp.array(self.M_arr_natural)
        
        self.M_arr_jax = jnp.array(self.M_arr_natural)

        # Comoving matter density
        self.RHO_M_comoving = ccl.rho_x(self.cosmo, 1.0, 'matter', is_comoving=True)
        
        # CCL setup
        self.mass_def = ccl.halos.MassDef200c
        self.concentration = ccl.halos.ConcentrationDuffy08(mass_def=self.mass_def)
        self.mass_func = ccl.halos.MassFuncTinker08(mass_def=self.mass_def)
        self.halo_bias = ccl.halos.HaloBiasTinker10(mass_def=self.mass_def)
        self.nfw_profile = ccl.halos.HaloProfileNFW(
            mass_def=self.mass_def,
            concentration=self.concentration,
            fourier_analytic=True
        )
        
        # Precompute and build kernels
        if verbose:
            print("Initializing MultiRedshiftHaloModel...")
        
        self._precompute_ccl_quantities()
        self._build_jax_kernels()
        
        if verbose:
            print(f"  Ready: {self.n_z} redshift(s), {self.n_k} k points, {self.n_M} mass points (GL)")
    
    def _precompute_ccl_quantities(self):
        """Compute CCL quantities at GL mass nodes, convert to JAX."""
        if self.verbose:
            print("  Precomputing CCL quantities at GL nodes...")
        
        # Temporary numpy arrays
        n_M_np = np.zeros((self.n_z, self.n_M))
        b_h_np = np.zeros((self.n_z, self.n_M))
        Pk_lin_np = np.zeros((self.n_z, self.n_k))
        u_fourier_np = np.zeros((self.n_z, self.n_k, self.n_M))
        
        for iz, a in enumerate(self.a_array):
            n_M_np[iz] = self.mass_func(self.cosmo, self.M_arr_natural, a)
            b_h_np[iz] = self.halo_bias(self.cosmo, self.M_arr_natural, a)
            Pk_lin_np[iz] = ccl.linear_power(self.cosmo, self.k_array_natural, a)
            
            for ik, k in enumerate(self.k_array_natural):
                u_fourier_np[iz, ik, :] = np.array([
                    self.nfw_profile._fourier(self.cosmo, k, M, a)
                    for M in self.M_arr_natural
                ])
        
        # Unit conversions
        if self.units_per_h:
            n_M_np /= self.h**3
            Pk_lin_np *= self.h**3
        
        # Derived quantities
        RHO_M = self.get_RHO_M()
        u_sat_np = u_fourier_np / self.M_arr_natural[np.newaxis, np.newaxis, :]
        u_matter_np = u_fourier_np / RHO_M
        if self.units_per_h:
            u_matter_np *= self.h
        
        # Convert to JAX
        self.n_M_z = jnp.array(n_M_np)       # (n_z, N_GL)
        self.b_h_z = jnp.array(b_h_np)       # (n_z, N_GL)
        self.Pk_lin_z = jnp.array(Pk_lin_np) # (n_z, n_k)
        self.u_sat = jnp.array(u_sat_np)     # (n_z, n_k, N_GL)
        self.u_matter = jnp.array(u_matter_np)  # (n_z, n_k, N_GL)

        # Print stellar mass info if enabled
        if self.verbose and self.median_Mstar is not None:
            print("  Stellar point mass component enabled:")
            if self.is_single_z:
                print(f"    z={self.z_array[0]:.3f}: M_star = {self.median_Mstar[0]:.2e} Msun/h²")
            else:
                for z, mstar in zip(self.z_array, self.median_Mstar):
                    print(f"    z={z:.3f}: M_star = {mstar:.2e} Msun/h²")

    def _build_jax_kernels(self):
        """Build JAX kernels using GL integration."""
        if self.verbose:
            print("  Building JAX kernels...")
        
        log10M_min = self.log10M_min
        log10M_max = self.log10M_max
        RHO_M = self.get_RHO_M()
        M_arr = self.M_arr_jax
        
        # --- P_gg kernel (single k, single z) ---
        @jit
        def _Pgg_kernel(N_c, N_s, u_sat_kz, b_h_z, n_M_z, Pk_lin_kz, n_gal_z):
            # 1-halo: central-satellite
            integrand_cs = 2 * N_c * N_s * u_sat_kz * n_M_z
            I_1h_cs = gl_integrate_precomputed(integrand_cs, log10M_min, log10M_max)
            
            # 1-halo: satellite-satellite
            integrand_ss = N_s**2 * u_sat_kz**2 * n_M_z
            I_1h_ss = gl_integrate_precomputed(integrand_ss, log10M_min, log10M_max)
            
            P_1h = (I_1h_cs + I_1h_ss) / n_gal_z**2
            
            # 2-halo
            integrand_2h = (N_c + N_s * u_sat_kz) * b_h_z * n_M_z
            I_2h = gl_integrate_precomputed(integrand_2h, log10M_min, log10M_max)
            P_2h = Pk_lin_kz * (I_2h / n_gal_z)**2
            
            return P_1h + P_2h
        
        # --- P_gm kernel (single k, single z) ---
        @jit
        def _Pgm_kernel(N_c, N_s, u_sat_kz, u_matter_kz, b_h_z, n_M_z, Pk_lin_kz, n_gal_z):
            # Matter normalization
            integrand_M1 = (M_arr / RHO_M) * b_h_z * n_M_z
            I_M1 = 1.0 - gl_integrate_precomputed(integrand_M1, log10M_min, log10M_max)
            
            # 1-halo
            integrand_1h = (N_c + N_s * u_sat_kz) * u_matter_kz * n_M_z
            P_1h = gl_integrate_precomputed(integrand_1h, log10M_min, log10M_max) / n_gal_z
            
            # 2-halo galaxy term
            integrand_g = (N_c + N_s * u_sat_kz) * b_h_z * n_M_z
            I_2h_g = gl_integrate_precomputed(integrand_g, log10M_min, log10M_max) / n_gal_z
            
            # 2-halo matter term
            integrand_m = u_matter_kz * b_h_z * n_M_z
            I_M2 = gl_integrate_precomputed(integrand_m, log10M_min, log10M_max)
            I_2h_m = I_M1 + I_M2
            
            P_2h = Pk_lin_kz * I_2h_g * I_2h_m
            
            return P_1h + P_2h
        
        # --- n_gal kernel (single z) ---
        @jit
        def _ngal_kernel(N_c, N_s, n_M_z):
            integrand = (N_c + N_s) * n_M_z
            return gl_integrate_precomputed(integrand, log10M_min, log10M_max)
        
        # --- vmap over k ---
        _Pgg_all_k = vmap(
            _Pgg_kernel,
            in_axes=(None, None, 0, None, None, 0, None)
        )
        
        _Pgm_all_k = vmap(
            _Pgm_kernel,
            in_axes=(None, None, 0, 0, None, None, 0, None)
        )
        
        # --- Full computation for single z ---
        @jit
        def compute_single_z(N_c, N_s, u_sat_z, u_matter_z, b_h_z, n_M_z, Pk_lin_z):
            n_gal = _ngal_kernel(N_c, N_s, n_M_z)
            P_gg = _Pgg_all_k(N_c, N_s, u_sat_z, b_h_z, n_M_z, Pk_lin_z, n_gal)
            P_gm = _Pgm_all_k(N_c, N_s, u_sat_z, u_matter_z, b_h_z, n_M_z, Pk_lin_z, n_gal)
            return P_gg, P_gm, n_gal
        
        # --- vmap over z ---
        @jit
        def compute_all_z(N_c_all, N_s_all, u_sat_all, u_matter_all, b_h_all, n_M_all, Pk_lin_all):
            return vmap(compute_single_z)(
                N_c_all, N_s_all, u_sat_all, u_matter_all, b_h_all, n_M_all, Pk_lin_all
            )
        
        # Store
        self._compute_single_z = compute_single_z
        self._compute_all_z = compute_all_z
        self._ngal_kernel = _ngal_kernel

    def _compute_stellar_point_mass_DeltaSigma(self, rp: np.ndarray) -> np.ndarray:
        """
        Compute stellar point mass contribution to Delta Sigma.

        Formula: ΔΣ_stellar(R) = M_star / (π * R²)

        Parameters
        ----------
        rp : array
            Projected radii [Mpc/h] or [Mpc] depending on units_per_h

        Returns
        -------
        DeltaSigma_stellar : array
            Shape (n_z, n_rp) or (n_rp,) if single z
            Units: [Msun h/pc²] or [Msun/pc²]
        """
        if self.median_Mstar is None:
            # No stellar component, return zeros
            if self.is_single_z:
                return np.zeros(len(rp))
            else:
                return np.zeros((self.n_z, len(rp)))

        # Unit conversion for stellar mass
        # Input: median_Mstar in [Msun/h²] (CSMF convention)
        # Need: [Msun] (natural units)


        # Compute in natural units: [Msun/Mpc²]
        # Shape: (n_z, 1) * (n_rp,) -> (n_z, n_rp)
        Mstar_2d = self.median_Mstar[:, np.newaxis]  # (n_z, 1)
        rp_2d = rp[np.newaxis, :]  # (1, n_rp)
        DeltaSigma_Mpc = Mstar_2d / (np.pi * rp_2d**2)  # [Msun/Mpc²]

        # Convert to pc²: [Msun/pc²]
        DeltaSigma_pc = DeltaSigma_Mpc / 1e12  # 1 Mpc² = 10^12 pc²

        if self.units_per_h:
            DeltaSigma_pc = DeltaSigma_pc*self.h

        # Return proper shape
        if self.is_single_z:
            return DeltaSigma_pc[0]  # (n_rp,)
        else:
            return DeltaSigma_pc  # (n_z, n_rp)

    def _get_occupation_numbers(self,
                                 Mstar_min: Optional[Union[float, np.ndarray]] = None,
                                 Mstar_max: Optional[Union[float, np.ndarray]] = None):
        """
        Get N_c, N_s at each redshift.
        
        Returns shape (n_z, N_GL).
        
        For CSMF:
        - Scalar Mstar_min/max: same bin for all z
        - Array Mstar_min/max: must have length == n_z
        """
        if self.hod_type == HODType.CSMF:
            if Mstar_min is None or Mstar_max is None:
                raise ValueError("CSMF requires Mstar_min and Mstar_max")
            
            is_array = isinstance(Mstar_min, (list, np.ndarray, jnp.ndarray))
            
            if not is_array:
                # Scalar: same bin for all z
                N_c_single = self.hod.N_central(self.M_arr_jax, Mstar_min, Mstar_max)
                N_s_single = self.hod.N_satellite(self.M_arr_jax, Mstar_min, Mstar_max)
                N_c = jnp.broadcast_to(N_c_single, (self.n_z, self.n_M))
                N_s = jnp.broadcast_to(N_s_single, (self.n_z, self.n_M))
            else:
                # Array: must match n_z
                Mstar_min = jnp.asarray(Mstar_min)
                Mstar_max = jnp.asarray(Mstar_max)
                
                if len(Mstar_min) != self.n_z:
                    raise ValueError(
                        f"Mstar arrays length ({len(Mstar_min)}) must match "
                        f"z_array length ({self.n_z})"
                    )
                
                # vmap over stellar mass bins
                N_c_bound = partial(self.hod.N_central, self.M_arr_jax)
                N_s_bound = partial(self.hod.N_satellite, self.M_arr_jax)
                
                N_c = vmap(N_c_bound)(Mstar_min, Mstar_max)
                N_s = vmap(N_s_bound)(Mstar_min, Mstar_max)
            
            return N_c, N_s
        
        else:
            # Standard HOD
            N_c_single = self.hod.N_central(self.M_arr_jax)
            N_s_single = self.hod.N_satellite(self.M_arr_jax)
            N_c = jnp.broadcast_to(N_c_single, (self.n_z, self.n_M))
            N_s = jnp.broadcast_to(N_s_single, (self.n_z, self.n_M))
            return N_c, N_s
    
    # ========================================================================
    # Public API: Power Spectra
    # ========================================================================
    
    def compute_power_spectra(self,
                              compute_gg: bool = True,
                              compute_gm: bool = True,
                              Mstar_min: Optional[Union[float, np.ndarray]] = None,
                              Mstar_max: Optional[Union[float, np.ndarray]] = None,
                              verbose: Optional[bool] = None) -> Tuple[Optional[np.ndarray],
                                                                        Optional[np.ndarray],
                                                                        np.ndarray]:
        """
        Compute power spectra at all redshifts.
        
        Parameters
        ----------
        compute_gg, compute_gm : bool
            Which spectra to compute
        Mstar_min, Mstar_max : float or array, optional
            Stellar mass bin(s) for CSMF.
            If array, must have length == len(z_array)
        verbose : bool, optional
            Print info
        
        Returns
        -------
        P_gg : array or None
            Shape (n_k,) if single z, (n_z, n_k) if multi z
        P_gm : array or None
            Shape (n_k,) if single z, (n_z, n_k) if multi z
        n_gal : array
            Shape () if single z, (n_z,) if multi z
        """
        if verbose is None:
            verbose = self.verbose
        
        # Get occupation numbers
        N_c, N_s = self._get_occupation_numbers(Mstar_min, Mstar_max)
        
        # Compute
        P_gg_all, P_gm_all, n_gal_all = self._compute_all_z(
            N_c, N_s, self.u_sat, self.u_matter, self.b_h_z, self.n_M_z, self.Pk_lin_z
        )
        
        # Squeeze if single z
        if self.is_single_z:
            P_gg_all = P_gg_all[0]
            P_gm_all = P_gm_all[0]
            n_gal_all = n_gal_all[0]
        
        if verbose:
            self._print_ngal(n_gal_all)
        
        P_gg = np.array(P_gg_all) if compute_gg else None
        P_gm = np.array(P_gm_all) if compute_gm else None
        n_gal = np.array(n_gal_all)
        
        return P_gg, P_gm, n_gal
    
    def compute_Pgg_all_z(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute P_gg at all redshifts."""
        P_gg, _, n_gal = self.compute_power_spectra(compute_gg=True, compute_gm=False, **kwargs)
        return P_gg, n_gal
    
    def compute_Pgm_all_z(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute P_gm at all redshifts."""
        _, P_gm, n_gal = self.compute_power_spectra(compute_gg=False, compute_gm=True, **kwargs)
        return P_gm, n_gal
    
    def compute_both_all_z(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute both P_gg and P_gm at all redshifts."""
        return self.compute_power_spectra(compute_gg=True, compute_gm=True, **kwargs)
    
    def compute_ngal_all_z(self, **kwargs) -> np.ndarray:
        """Compute galaxy number density at all redshifts."""
        _, _, n_gal = self.compute_power_spectra(compute_gg=False, compute_gm=False, **kwargs)
        return n_gal
    
    # ========================================================================
    # Public API: Real-Space Observables
    # ========================================================================
    
    def compute_wgg_all_z(self, rp: np.ndarray,
                          bin_avg: bool = False,
                          rp_bins: Optional[np.ndarray] = None,
                          Mstar_min: Optional[Union[float, np.ndarray]] = None,
                          Mstar_max: Optional[Union[float, np.ndarray]] = None,
                          verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute projected clustering w_gg(rp) at all redshifts.
        
        Parameters
        ----------
        rp : array
            Projected separations
        bin_avg : bool
            If True, average within bins defined by rp_bins
        rp_bins : array, optional
            Bin edges (required if bin_avg=True)
        Mstar_min, Mstar_max : float or array, optional
            Stellar mass bin(s) for CSMF
        verbose : bool, optional
            Print info
        
        Returns
        -------
        rp_out : array
            Output radii
        wgg : array
            Shape (n_rp,) if single z, (n_z, n_rp) if multi z
        """
        if verbose is None:
            verbose = self.verbose
        
        Pk_gg, _ = self.compute_Pgg_all_z(
            Mstar_min=Mstar_min, Mstar_max=Mstar_max, verbose=verbose
        )
        
        k = self.get_k_array()
        
        # Ensure 2D for uniform processing
        if self.is_single_z:
            Pk_gg = Pk_gg[np.newaxis, :]
        
        # Hankel transform each z
        wgg_list = []
        for iz in range(self.n_z):
            if bin_avg:
                rp_out, wgg = Pk_to_wgg_direct(k, Pk_gg[iz], rp, rp_bins=rp_bins)
            else:
                rp_out, wgg = Pk_to_wgg_direct(k, Pk_gg[iz], rp)
            wgg_list.append(wgg)
        
        wgg = np.array(wgg_list)
        
        # Squeeze if single z
        if self.is_single_z:
            wgg = wgg[0]
        
        return rp_out, wgg
    
    def compute_DeltaSigma_all_z(self, rp: np.ndarray,
                                  bin_avg: bool = False,
                                  rp_bins: Optional[np.ndarray] = None,
                                  method: str = 'direct',
                                  Mstar_min: Optional[Union[float, np.ndarray]] = None,
                                  Mstar_max: Optional[Union[float, np.ndarray]] = None,
                                  verbose: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute galaxy-galaxy lensing ΔΣ(rp) at all redshifts.
        
        Parameters
        ----------
        rp : array
            Projected separations
        bin_avg : bool
            If True, average within bins defined by rp_bins
        rp_bins : array, optional
            Bin edges (required if bin_avg=True)
        method : str
            'direct' or 'traditional'
        Mstar_min, Mstar_max : float or array, optional
            Stellar mass bin(s) for CSMF
        verbose : bool, optional
            Print info
        
        Returns
        -------
        rp_out : array
            Output radii
        DeltaSigma : array
            Shape (n_rp,) if single z, (n_z, n_rp) if multi z
            Includes both halo model and stellar point mass contributions
            (if median_Mstar was provided during initialization)
        """
        if verbose is None:
            verbose = self.verbose
        
        if method not in ['direct', 'traditional']:
            raise ValueError(f"Unknown method: {method}")
        
        Pk_gm, _ = self.compute_Pgm_all_z(
            Mstar_min=Mstar_min, Mstar_max=Mstar_max, verbose=verbose
        )
        
        k = self.get_k_array()
        rho_m = self.get_RHO_M()
        
        # Ensure 2D for uniform processing
        if self.is_single_z:
            Pk_gm = Pk_gm[np.newaxis, :]
        
        # Transform each z
        ds_list = []
        for iz in range(self.n_z):
            if method == 'direct':
                if bin_avg:
                    rp_out, ds = Pk_to_DeltaSigma_direct(k, Pk_gm[iz], rho_m, rp, rp_bins=rp_bins)
                else:
                    rp_out, ds = Pk_to_DeltaSigma_direct(k, Pk_gm[iz], rho_m, rp)
            else:
                if bin_avg:
                    rp_out, ds = Pk_gm_to_DeltaSigma_traditional(k, Pk_gm[iz], rho_m, rp, rp_bins=rp_bins)
                else:
                    rp_out, ds = Pk_gm_to_DeltaSigma_traditional(k, Pk_gm[iz], rho_m, rp)
            ds_list.append(ds)
        
        DeltaSigma = np.array(ds_list)

        # Add stellar component if available
        if self.median_Mstar is not None:
            # Check cache
            rp_key = tuple(rp_out)
            if rp_key not in self._DeltaSigma_stellar_cache:
                # Compute and cache
                DeltaSigma_stellar = self._compute_stellar_point_mass_DeltaSigma(rp_out)
                self._DeltaSigma_stellar_cache[rp_key] = DeltaSigma_stellar
            else:
                DeltaSigma_stellar = self._DeltaSigma_stellar_cache[rp_key]

            # Add to total (handle both single z and multi z cases)
            if self.is_single_z and DeltaSigma_stellar.ndim == 1:
                # Single z case: both are (n_rp,)
                DeltaSigma += DeltaSigma_stellar
            elif not self.is_single_z and DeltaSigma_stellar.ndim == 2:
                # Multi z case: both are (n_z, n_rp)
                DeltaSigma += DeltaSigma_stellar
            else:
                # Shape mismatch - this shouldn't happen
                raise ValueError(
                    f"Shape mismatch in stellar component addition: "
                    f"DeltaSigma.shape={DeltaSigma.shape}, "
                    f"DeltaSigma_stellar.shape={DeltaSigma_stellar.shape}"
                )

        # Squeeze if single z
        if self.is_single_z:
            DeltaSigma = DeltaSigma[0]

        return rp_out, DeltaSigma
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_k_array(self) -> np.ndarray:
        """Return k array in appropriate units."""
        if self.units_per_h:
            return self.k_array_natural / self.h
        return self.k_array_natural
    
    def get_M_array(self) -> np.ndarray:
        """Return mass array (at GL nodes) in appropriate units."""
        if self.units_per_h:
            return self.M_arr_natural * self.h
        return self.M_arr_natural
    
    def get_RHO_M(self) -> float:
        """Return comoving matter density in appropriate units."""
        if self.units_per_h:
            return self.RHO_M_comoving / self.h**2
        return self.RHO_M_comoving
    
    def update_hod_params(self, new_params: Dict):
        """Update HOD parameters without reinitializing CCL quantities."""
        self.hod_params = new_params.copy()
        self.hod = create_hod(
            self.hod_type, new_params, masses_are_log10=self.masses_are_log10
        )
    
    def _print_ngal(self, n_gal):
        """Print galaxy number densities."""
        n_unit = "(h/Mpc)³" if self.units_per_h else "Mpc⁻³"
        if self.is_single_z:
            print(f"  z={self.z_array[0]:.3f}: n_gal = {float(n_gal):.6e} {n_unit}")
        else:
            print("Galaxy number densities:")
            for z, ng in zip(self.z_array, n_gal):
                print(f"  z={z:.3f}: n_gal = {float(ng):.6e} {n_unit}")


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # GL integration utilities
    'N_GL',
    'GL_X',
    'GL_W',
    'gl_nodes_scaled',
    'gl_integrate_precomputed',
    'make_gl_integrator',
    # HOD types
    'HODType',
    'STANDARD_HOD_PARAMS',
    'CSMF_HOD_PARAMS',
    'validate_hod_params',
    # Pure HOD functions
    'standard_N_central',
    'standard_N_satellite',
    'csmf_Mstar_central',
    'csmf_N_central',
    'csmf_N_satellite',
    # HOD classes
    'StandardHOD',
    'CSMF_HOD',
    'create_hod',
    # Main model
    'MultiRedshiftHaloModel',
]
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