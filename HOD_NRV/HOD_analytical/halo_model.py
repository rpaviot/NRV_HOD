"""
Halo Model Power Spectrum Calculator - INTERPAX VERSION
=========================================================

Updated to work with the interpax-based Cosmology class.

Key features:
1. Uses interpax for β^NL interpolation (via Cosmology.beta_nl_interp)
2. Corrected β^NL implementation with I^11, I^12, I^21, I^22 terms
3. Clean JAX-based implementation with Gauss-Legendre integration
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf as erf_jax
from jax.scipy.special import sici
from scipy.special import roots_legendre
from typing import Dict, Optional, Union, List
from HOD_NRV.HOD_analytical.pycosmo import Cosmology,HAS_INTERPAX
import pyccl as ccl

# Import the interpax-based Cosmology

try:
    from HOD_NRV.utilsf.hankel_transforms import (
        Pk_to_wgg_direct,
        Pk_to_DeltaSigma_direct,
        Pk_gm_to_DeltaSigma_traditional,
    )
    HAS_HANKEL = True
except ImportError:
    HAS_HANKEL = False

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Gauss-Legendre Integration
# ============================================================================

N_GL = 200
_x_gl, _w_gl = roots_legendre(N_GL)
GL_X = jnp.array(_x_gl)
GL_W = jnp.array(_w_gl)


@jit
def gl_nodes_scaled(a: float, b: float) -> jnp.ndarray:
    """Scale GL nodes from [-1,1] to [a,b]."""
    return 0.5 * ((b - a) * GL_X + (a + b))


@jit
def gl_integrate(integrand: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Integrate using precomputed GL weights."""
    return 0.5 * (b - a) * jnp.dot(GL_W, integrand)


# ============================================================================
# NFW Fourier Transform
# ============================================================================

@jit
def nfw_fourier_u(k: jnp.ndarray, R_s: jnp.ndarray, c: jnp.ndarray, 
                  f_scale: float = 1.0) -> jnp.ndarray:
    """
    NFW Fourier transform. Returns shape (n_k, n_M).
    """
    x = k[:, None] * R_s[None, :]
    norm = 1.0 / (jnp.log(1.0 + c) - c / (1.0 + c))
    si_x, ci_x = sici(x)
    si_cx, ci_cx = sici((1.0 + c) * x)
    
    term1 = jnp.sin(x) * (si_cx - si_x)
    term2 = jnp.cos(x) * (ci_cx - ci_x)
    term3 = jnp.sin(c[None, :] * x) / ((1.0 + c[None, :]) * x)
    
    u = norm[None, :] * (term1 + term2 - term3)
    u = jnp.where(x < 1e-8, 1.0, u)
    return u * f_scale


@jit
def nfw_fourier_u_single(k: jnp.ndarray, R_s: float, c: float) -> jnp.ndarray:
    """NFW Fourier transform for single halo. Returns shape (n_k,)."""
    x = k * R_s
    norm = 1.0 / (jnp.log(1.0 + c) - c / (1.0 + c))
    si_x, ci_x = sici(x)
    si_cx, ci_cx = sici((1.0 + c) * x)
    
    term1 = jnp.sin(x) * (si_cx - si_x)
    term2 = jnp.cos(x) * (ci_cx - ci_x)
    term3 = jnp.sin(c * x) / ((1.0 + c) * x)
    
    u = norm * (term1 + term2 - term3)
    return jnp.where(x < 1e-8, 1.0, u)


# ============================================================================
# HOD Functions - Analytical Framework
# ============================================================================
# All mass parameters are in log10 units for consistency with HOD_models.py
# No assembly bias or conformity support (use numerical framework for those)

SQRT_2PI = jnp.sqrt(2 * jnp.pi)

# Parameter definitions for each HOD type
HOD_PARAM_DEFINITIONS = {
    'LRG': {
        'central_params': ['Ac', 'log10Mmin', 'sig_M'],
        'satellite_params': ['As', 'log10Mmin', 'log10M1', 'alpha', 'kappa'],
    },
    'ELG_GHOD': {
        'central_params': ['Ac', 'log10Mmin', 'sig_M'],
        'satellite_params': ['As', 'log10Mmin', 'log10M1', 'alpha', 'kappa'],
    },
    'ELG_SFR': {
        'central_params': ['Ac', 'log10Mmin', 'sig_M', 'gamma'],
        'satellite_params': ['As', 'log10Mmin', 'log10M1', 'alpha', 'kappa'],
    },
}

# Legacy parameter names (for CSMF)
CSMF_HOD_PARAMS = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']


def get_required_params(hod_type: str) -> List[str]:
    """Get unique required parameters for a given HOD type."""
    hod_type = hod_type.upper()
    if hod_type not in HOD_PARAM_DEFINITIONS:
        raise ValueError(f"Unknown HOD type: {hod_type}. Supported: {list(HOD_PARAM_DEFINITIONS.keys())}")

    defn = HOD_PARAM_DEFINITIONS[hod_type]
    # Combine and deduplicate (log10Mmin appears in both)
    all_params = set(defn['central_params']) | set(defn['satellite_params'])
    return list(all_params)


def validate_hod_params(hod_type: str, params: Dict) -> bool:
    """Validate that all required parameters are present."""
    hod_type = hod_type.upper()
    if hod_type == 'CSMF':
        required = CSMF_HOD_PARAMS
    else:
        required = get_required_params(hod_type)

    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing HOD parameters for {hod_type}: {missing}")
    return True


# ============================================================================
# Central Occupation Functions (all use log10 masses)
# ============================================================================

@jit
def lrg_N_central(logM: jnp.ndarray, Ac: float, log10Mmin: float, sig_M: float) -> jnp.ndarray:
    """
    LRG central occupation (Zheng+07 style).

    Parameters
    ----------
    logM : array
        log10 of halo mass
    Ac : float
        Central amplitude (typically 1.0)
    log10Mmin : float
        log10 of minimum mass threshold
    sig_M : float
        Scatter in log10(M)

    Returns
    -------
    N_cen : array
        Mean central occupation
    """
    return Ac / 2.0 * (1.0 + erf_jax((logM - log10Mmin) / sig_M))


@jit
def elg_ghod_N_central(logM: jnp.ndarray, Ac: float, log10Mmin: float, sig_M: float) -> jnp.ndarray:
    """
    ELG Gaussian HOD central occupation.

    Parameters
    ----------
    logM : array
        log10 of halo mass
    Ac : float
        Central amplitude
    log10Mmin : float
        log10 of peak mass
    sig_M : float
        Gaussian width in dex

    Returns
    -------
    N_cen : array
        Mean central occupation
    """
    return Ac / (SQRT_2PI * sig_M) * jnp.exp(-((logM - log10Mmin) ** 2) / (2.0 * sig_M ** 2))


@jit
def elg_sfr_N_central(logM: jnp.ndarray, Ac: float, log10Mmin: float,
                       sig_M: float, gamma: float) -> jnp.ndarray:
    """
    ELG SFR-based central occupation.

    Gaussian at low mass, power-law tail at high mass.

    Parameters
    ----------
    logM : array
        log10 of halo mass
    Ac : float
        Central amplitude
    log10Mmin : float
        log10 of transition mass
    sig_M : float
        Gaussian width in dex
    gamma : float
        Power-law slope at high mass

    Returns
    -------
    N_cen : array
        Mean central occupation
    """
    exp_part = elg_ghod_N_central(logM, Ac, log10Mmin, sig_M)
    power_part = Ac / (SQRT_2PI * sig_M) * jnp.power(logM / log10Mmin, gamma)
    return jnp.where(logM < log10Mmin, exp_part, power_part)


# ============================================================================
# Satellite Occupation Function (unified for all HOD types)
# ============================================================================

@jit
def unified_N_satellite(logM: jnp.ndarray, As: float, log10Mmin: float,
                        log10M1: float, alpha: float, kappa: float) -> jnp.ndarray:
    """
    Unified satellite occupation for all HOD types.

    Matches the parametrization in HOD_models.py.

    Parameters
    ----------
    logM : array
        log10 of halo mass
    As : float
        Satellite amplitude
    log10Mmin : float
        log10 of central minimum mass
    log10M1 : float
        log10 of satellite mass scale
    alpha : float
        Power-law slope
    kappa : float
        Satellite cutoff factor (satellites require M > kappa * Mmin)

    Returns
    -------
    N_sat : array
        Mean satellite occupation
    """
    Mmin = 10.0 ** log10Mmin
    M1 = 10.0 ** log10M1
    M = 10.0 ** logM

    Nsat = As * jnp.power((M - kappa * Mmin) / M1, alpha)
    return jnp.where(M > kappa * Mmin, Nsat, 0.0)


# ============================================================================
# Legacy functions (for backward compatibility with old StandardHOD)
# ============================================================================

@jit
def standard_N_central(M: jnp.ndarray, Mmin: float, sigma_lnM: float) -> jnp.ndarray:
    """Legacy: uses linear mass and sigma_lnM."""
    x = jnp.log(M / Mmin) / sigma_lnM
    return 0.5 * (1.0 + erf_jax(x))


@jit
def standard_N_satellite(M: jnp.ndarray, M0: float, M1: float, alpha: float) -> jnp.ndarray:
    """Legacy: uses linear mass and M0 cutoff."""
    return jnp.where(M > M0, ((M - M0) / M1) ** alpha, 0.0)


@jit
def csmf_Mstar_central(Mh: jnp.ndarray, M0: float, M1: float,
                        gamma1: float, gamma2: float) -> jnp.ndarray:
    x = Mh / M1
    return M0 * (x ** gamma1) / ((1 + x) ** (gamma1 - gamma2))


@jit
def csmf_N_central(Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float,
                   M0: float, M1: float, gamma1: float, gamma2: float,
                   sigma_c: float) -> jnp.ndarray:
    Mstar_c = csmf_Mstar_central(Mh, M0, M1, gamma1, gamma2)
    log_Mstar_c = jnp.log10(Mstar_c)
    sqrt2_sigma = jnp.sqrt(2.0) * sigma_c
    erf_max = erf_jax((jnp.log10(Mstar_max) - log_Mstar_c) / sqrt2_sigma)
    erf_min = erf_jax((jnp.log10(Mstar_min) - log_Mstar_c) / sqrt2_sigma)
    return jnp.maximum(0.5 * (erf_max - erf_min), 0.0)


def _csmf_satellite_integrand(log10_Mstar, Mh, M0, M1, gamma1, gamma2,
                               alpha_s, b0, b1):
    Mstar = 10.0 ** log10_Mstar
    Mstar_c = csmf_Mstar_central(Mh, M0, M1, gamma1, gamma2)
    Mstar_s = 0.56 * Mstar_c
    log_phi = b0 + b1 * jnp.log10(Mh / 1e13)
    phi_s = 10.0 ** log_phi
    x = Mstar / Mstar_s
    Phi = (phi_s / Mstar_s) * (x ** alpha_s) * jnp.exp(-x**2)
    return Phi * Mstar * jnp.log(10.0)


@jit
def csmf_N_satellite_single(Mh: float, Mstar_min: float, Mstar_max: float,
                            M0: float, M1: float, gamma1: float, gamma2: float,
                            alpha_s: float, b0: float, b1: float) -> float:
    log10_min, log10_max = jnp.log10(Mstar_min), jnp.log10(Mstar_max)
    x_scaled = gl_nodes_scaled(log10_min, log10_max)
    integrand = _csmf_satellite_integrand(x_scaled, Mh, M0, M1, gamma1, gamma2,
                                           alpha_s, b0, b1)
    return jnp.maximum(gl_integrate(integrand, log10_min, log10_max), 0.0)


@jit
def csmf_N_satellite(Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float,
                     M0: float, M1: float, gamma1: float, gamma2: float,
                     alpha_s: float, b0: float, b1: float) -> jnp.ndarray:
    return vmap(lambda m: csmf_N_satellite_single(
        m, Mstar_min, Mstar_max, M0, M1, gamma1, gamma2, alpha_s, b0, b1
    ))(Mh)


# ============================================================================
# HOD Wrapper Classes
# ============================================================================

class AnalyticalHOD:
    """
    Simplified HOD for analytical halo model calculations.

    Supports: 'LRG', 'ELG_GHOD', 'ELG_SFR'
    All mass parameters expected in log10 units.
    No assembly bias or conformity (use numerical framework for those).

    Parameters
    ----------
    hod_type : str
        Type of HOD model: 'LRG', 'ELG_GHOD', or 'ELG_SFR'

    Example
    -------
    >>> hod = AnalyticalHOD('LRG')
    >>> params = {
    ...     'Ac': 1.0, 'log10Mmin': 12.5, 'sig_M': 0.3,
    ...     'As': 1.0, 'log10M1': 13.5, 'alpha': 1.0, 'kappa': 1.0
    ... }
    >>> hod.set_params(params)
    >>> logM = jnp.linspace(10, 15, 100)
    >>> N_c = hod.N_central(logM)
    >>> N_s = hod.N_satellite(logM)
    """

    SUPPORTED_TYPES = ['LRG', 'ELG_GHOD', 'ELG_SFR']

    # Central function dispatch
    _central_funcs = {
        'LRG': lrg_N_central,
        'ELG_GHOD': elg_ghod_N_central,
        'ELG_SFR': elg_sfr_N_central,
    }

    def __init__(self, hod_type: str = 'LRG'):
        self.hod_type = hod_type.upper()
        if self.hod_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown HOD type: {hod_type}. "
                f"Supported: {self.SUPPORTED_TYPES}"
            )
        self.params = None
        self._central_func = self._central_funcs[self.hod_type]
        self._param_def = HOD_PARAM_DEFINITIONS[self.hod_type]

    def set_params(self, params: Dict):
        """
        Set HOD parameters.

        Parameters
        ----------
        params : dict
            HOD parameters. Required keys depend on hod_type:
            - LRG: Ac, log10Mmin, sig_M, As, log10M1, alpha, kappa
            - ELG_GHOD: Ac, log10Mmin, sig_M, As, log10M1, alpha, kappa
            - ELG_SFR: Ac, log10Mmin, sig_M, gamma, As, log10M1, alpha, kappa
        """
        validate_hod_params(self.hod_type, params)
        self.params = params.copy()

        # Store central params
        self.Ac = jnp.array(params['Ac'])
        self.log10Mmin = jnp.array(params['log10Mmin'])
        self.sig_M = jnp.array(params['sig_M'])
        if 'gamma' in self._param_def['central_params']:
            self.gamma = jnp.array(params['gamma'])

        # Store satellite params
        self.As = jnp.array(params['As'])
        self.log10M1 = jnp.array(params['log10M1'])
        self.alpha = jnp.array(params['alpha'])
        self.kappa = jnp.array(params['kappa'])

    def N_central(self, logM: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean central occupation.

        Parameters
        ----------
        logM : array
            log10 of halo mass

        Returns
        -------
        N_c : array
            Mean central occupation
        """
        if self.params is None:
            raise ValueError("HOD parameters not set. Call set_params() first.")

        if self.hod_type == 'LRG':
            return lrg_N_central(logM, self.Ac, self.log10Mmin, self.sig_M)
        elif self.hod_type == 'ELG_GHOD':
            return elg_ghod_N_central(logM, self.Ac, self.log10Mmin, self.sig_M)
        elif self.hod_type == 'ELG_SFR':
            return elg_sfr_N_central(logM, self.Ac, self.log10Mmin, self.sig_M, self.gamma)

    def N_satellite(self, logM: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean satellite occupation.

        Parameters
        ----------
        logM : array
            log10 of halo mass

        Returns
        -------
        N_s : array
            Mean satellite occupation
        """
        if self.params is None:
            raise ValueError("HOD parameters not set. Call set_params() first.")

        return unified_N_satellite(
            logM, self.As, self.log10Mmin, self.log10M1, self.alpha, self.kappa
        )


# Legacy class for backward compatibility
class StandardHOD:
    """
    DEPRECATED: Use AnalyticalHOD instead.

    Legacy HOD class using linear masses and different parametrization.
    """
    STANDARD_HOD_PARAMS = ['log10Mmin', 'siglnM', 'log10M0', 'log10M1', 'alpha']

    def __init__(self, params: Optional[Dict] = None):
        import warnings
        warnings.warn(
            "StandardHOD is deprecated. Use AnalyticalHOD instead.",
            DeprecationWarning
        )
        self.hod_type = 'standard'
        self.params = None
        if params is not None:
            self.set_params(params)

    def set_params(self, params: Dict):
        missing = [p for p in self.STANDARD_HOD_PARAMS if p not in params]
        if missing:
            raise ValueError(f"Missing HOD parameters: {missing}")
        self.params = params.copy()
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
    def __init__(self, params: Optional[Dict] = None, masses_are_log10: bool = True):
        self.hod_type = 'csmf'
        self.masses_are_log10 = masses_are_log10
        self.params = None
        if params is not None:
            self.set_params(params)
    
    def set_params(self, params: Dict):
        validate_hod_params('csmf', params)
        self.params = params.copy()
        self.M0 = jnp.array(10.0 ** params['M0'] if self.masses_are_log10 else params['M0'])
        self.M1 = jnp.array(10.0 ** params['M1'] if self.masses_are_log10 else params['M1'])
        self.gamma1 = jnp.array(params['gamma1'])
        self.gamma2 = jnp.array(params['gamma2'])
        self.sigma_c = jnp.array(params['sigma_c'])
        self.alpha_s = jnp.array(params['alpha_s'])
        self.b0 = jnp.array(params['b0'])
        self.b1 = jnp.array(params['b1'])
    
    def N_central(self, Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float) -> jnp.ndarray:
        return csmf_N_central(Mh, Mstar_min, Mstar_max,
                              self.M0, self.M1, self.gamma1, self.gamma2, self.sigma_c)
    
    def N_satellite(self, Mh: jnp.ndarray, Mstar_min: float, Mstar_max: float) -> jnp.ndarray:
        return csmf_N_satellite(Mh, Mstar_min, Mstar_max,
                                self.M0, self.M1, self.gamma1, self.gamma2,
                                self.alpha_s, self.b0, self.b1)


def create_hod(hod_type: str, **kwargs) -> Union[AnalyticalHOD, CSMF_HOD]:
    """
    Factory function to create HOD objects.

    Parameters
    ----------
    hod_type : str
        Type of HOD model:
        - 'LRG': LRG-style step function central (Zheng+07)
        - 'ELG_GHOD': ELG Gaussian central
        - 'ELG_SFR': ELG SFR-based with power-law tail
        - 'CSMF': Conditional stellar mass function
        - 'standard': DEPRECATED, use 'LRG' instead
    **kwargs
        Additional arguments passed to the HOD constructor

    Returns
    -------
    hod : AnalyticalHOD or CSMF_HOD
        The HOD object

    Example
    -------
    >>> hod = create_hod('LRG')
    >>> hod.set_params({'Ac': 1.0, 'log10Mmin': 12.5, 'sig_M': 0.3,
    ...                 'As': 1.0, 'log10M1': 13.5, 'alpha': 1.0, 'kappa': 1.0})
    """
    hod_type_upper = hod_type.upper()

    # Handle new HOD types
    if hod_type_upper in AnalyticalHOD.SUPPORTED_TYPES:
        return AnalyticalHOD(hod_type_upper)

    # Handle CSMF
    if hod_type_upper == 'CSMF':
        return CSMF_HOD(masses_are_log10=kwargs.get('masses_are_log10', True))

    # Legacy support
    if hod_type.lower() == 'standard':
        import warnings
        warnings.warn(
            "hod_type='standard' is deprecated. Use 'LRG' instead.",
            DeprecationWarning
        )
        return StandardHOD()

    raise ValueError(
        f"Unknown HOD type: {hod_type}. "
        f"Supported: {AnalyticalHOD.SUPPORTED_TYPES + ['CSMF']}"
    )


# ============================================================================
# Power Spectrum Functions (Standard - no β^NL)
# ============================================================================

@jit
def _compute_ngal(N_c, N_s, n_M, log10M_min, log10M_max):
    return gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)


@jit
def _compute_Pgg(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                 log10M_min, log10M_max, f_c, f_s):
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    
    # 1-halo
    integrand_cs = 2.0 * N_c * N_s * u_s * n_M
    I_cs = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_cs)
    integrand_ss = (N_s ** 2) * (u_s ** 2) * n_M
    I_ss = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_ss)
    P_1h = (I_cs + I_ss) / n_gal ** 2
    
    # 2-halo
    integrand_2h = (N_c * u_c + N_s * u_s) * b_h * n_M
    I_2h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_2h)
    P_2h = Pk_lin * (I_2h / n_gal) ** 2
    
    return P_1h + P_2h


@jit
def _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                 log10M_min, log10M_max, f_c, f_s):
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, 1.0)
    
    # 1-halo
    integrand_1h = (N_c + N_s * u_s) * u_m * (M / rho_m) * n_M
    P_1h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_1h) / n_gal
    
    # 2-halo
    integrand_g = (N_c * u_c + N_s * u_s) * b_h * n_M
    I_g = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_g) / n_gal
    
    I_m1 = 1.0 - gl_integrate((M / rho_m) * b_h * n_M, log10M_min, log10M_max)
    integrand_m2 = u_m * (M / rho_m) * b_h * n_M
    I_m2 = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_m2)
    
    return P_1h + Pk_lin * I_g * (I_m1 + I_m2)


# ============================================================================
# β^NL Correction Functions
# ============================================================================

@jit
def _compute_I_NL_22(beta_nl_gl, H1, H2, b_h, n_M, log10M_min, log10M_max):
    """
    I^22 - Double integral over [M_min, ∞] × [M_min, ∞].
    H1, H2 are (n_k, n_M) arrays including profiles.
    """
    scale = (0.5 * (log10M_max - log10M_min)) ** 2
    bn_w = b_h * n_M * GL_W
    Hb1 = H1 * bn_w[None, :]
    Hb2 = H2 * bn_w[None, :]
    return scale * jnp.einsum('kij,ki,kj->k', beta_nl_gl, Hb1, Hb2)


@jit
def _compute_I_NL_12(beta_nl_Mmin_row, H2, b_h, n_M, log10M_min, log10M_max):
    """I^12 - First mass at M_min, integrate over second."""
    scale = 0.5 * (log10M_max - log10M_min)
    bn_w = b_h * n_M * GL_W
    integrand = beta_nl_Mmin_row * H2 * bn_w[None, :]
    return scale * jnp.sum(integrand, axis=1)


@jit
def _compute_I_NL_21(beta_nl_Mmin_col, H1, b_h, n_M, log10M_min, log10M_max):
    """I^21 - Second mass at M_min, integrate over first."""
    scale = 0.5 * (log10M_max - log10M_min)
    bn_w = b_h * n_M * GL_W
    integrand = beta_nl_Mmin_col * H1 * bn_w[None, :]
    return scale * jnp.sum(integrand, axis=1)


@jit
def _compute_I_NL_gg(beta_nl_gl, H_g, b_h, n_M, log10M_min, log10M_max):
    """I^NL for galaxy-galaxy (only I^22)."""
    return _compute_I_NL_22(beta_nl_gl, H_g, H_g, b_h, n_M, log10M_min, log10M_max)


@jit
def _compute_I_NL_gm(beta_nl_gl, beta_nl_Mmin_col, H_g, H_m, b_h, n_M,
                      u_m_Mmin, A_Mmin, log10M_min, log10M_max):
    """I^NL for galaxy-matter (I^21 + I^22)."""
    I_22 = _compute_I_NL_22(beta_nl_gl, H_g, H_m, b_h, n_M, log10M_min, log10M_max)
    integral_21 = _compute_I_NL_21(beta_nl_Mmin_col, H_g, b_h, n_M, log10M_min, log10M_max)
    I_21 = A_Mmin * u_m_Mmin * integral_21
    return I_21 + I_22



# ============================================================================
# Power Spectrum with β^NL
# ============================================================================

@jit
def _compute_Pgg_with_beta_nl(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                               log10M_min, log10M_max, f_c, f_s, beta_nl_gl):
    P_gg = _compute_Pgg(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                        log10M_min, log10M_max, f_c, f_s)
    
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal
    
    I_NL = _compute_I_NL_gg(beta_nl_gl, H_g, b_h, n_M, log10M_min, log10M_max)
    return P_gg + Pk_lin * I_NL


@jit
def _compute_Pgm_with_beta_nl(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                               log10M_min, log10M_max, f_c, f_s,
                               beta_nl_gl, beta_nl_Mmin_col, R_s_Mmin, c_Mmin):
    P_gm = _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                        log10M_min, log10M_max, f_c, f_s)
    
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, 1.0)
    u_m_Mmin = nfw_fourier_u_single(k, R_s_Mmin, c_Mmin)
    
    H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal
    W_m = M / rho_m
    H_m = u_m * W_m[None, :]
    A_Mmin = 1.0 - gl_integrate(W_m * b_h * n_M, log10M_min, log10M_max)
    
    I_NL = _compute_I_NL_gm(beta_nl_gl, beta_nl_Mmin_col, H_g, H_m, b_h, n_M,
                            u_m_Mmin, A_Mmin, log10M_min, log10M_max)
    return P_gm + Pk_lin * I_NL



# ============================================================================
# HaloModel Class
# ============================================================================

class HaloModel(Cosmology):
    """
    Halo Model with interpax-based β^NL support.
    
    Inherits from the simplified Cosmology class that uses interpax
    for β^NL interpolation.
    """
    
    def __init__(
        self,
        cosmo_params: Dict[str, float],
        z: Union[float, np.ndarray, List[float]],
        hod_type: str,
        f_c: float = 1.0,
        f_s: float = 1.0,
        M_min: float = 1e9,
        M_max: float = 1e16,
        masses_are_log10: bool = True,
        units_per_h: bool = True,
        k_array: Optional[Union[float, np.ndarray]] = None,
        Mstar_min: Optional[Union[float, np.ndarray]] = None,
        Mstar_max: Optional[Union[float, np.ndarray]] = None,
        median_Mstar: Optional[Union[float, np.ndarray]] = None,
        include_beta_nl: bool = False,
        beta_nl_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        **cosmo_kwargs
    ):
        # Initialize Cosmology base class
        super().__init__(cosmo_params, beta_nl_kwargs=beta_nl_kwargs,
                         verbose=verbose,k_array=k_array, units_per_h=units_per_h, **cosmo_kwargs)
        
        self.verbose = verbose
        
        # Redshift setup
        self.z_array = np.atleast_1d(z)
        self.n_z = len(self.z_array)
        self.is_single_z = (self.n_z == 1)
        self.a_array = 1.0 / (1.0 + self.z_array)
        
        # HOD setup
        self.hod_type = hod_type.lower()
        self.masses_are_log10 = masses_are_log10
        self.hod = create_hod(self.hod_type, masses_are_log10=masses_are_log10)
        
        if self.hod_type.upper() == 'CSMF' and (Mstar_min is None or Mstar_max is None):
            raise ValueError("CSMF HOD requires Mstar_min and Mstar_max")
        
        self._Mstar_min_jax, self._Mstar_max_jax = self._prepare_mstar_arrays(Mstar_min, Mstar_max)
        self.RHO_M = self.get_rho_m()
        self.f_c = f_c
        self.f_s = f_s
        
        # Mass array at GL nodes
        self.log10M_min = np.log10(M_min)
        self.log10M_max = np.log10(M_max)
        log10M_gl = np.array(gl_nodes_scaled(self.log10M_min, self.log10M_max))
        self.M_array = 10.0 ** log10M_gl
        self._log10M_gl_jax = jnp.array(log10M_gl)
        
        # Stellar mass for point mass contribution
        self.median_Mstar = self._prepare_median_Mstar(median_Mstar)
        
        # β^NL cache
        self.include_beta_nl = include_beta_nl
        self._beta_nl_gl_cache = {}
        self._beta_nl_Mmin_row_cache = {}
        self._beta_nl_Mmin_col_cache = {}
        self._beta_nl_Mmin_Mmin_cache = {}
        
        # Precompute CCL quantities
        self._precompute_ccl()
        
        # Compute β^NL
        if include_beta_nl:
            self._compute_beta_nl()
        
        if self.verbose:
            print(f"HaloModel: {self.n_z} z, {self.n_k} k, {N_GL} M points")
            print(f"  HOD: {self.hod_type}, f_c={f_c}, f_s={f_s}")
            print(f"  Mass: [10^{self.log10M_min:.1f}, 10^{self.log10M_max:.1f}]")
            print(f"  β^NL: {'enabled' if include_beta_nl else 'disabled'}")
    
    def _prepare_mstar_arrays(self, Mstar_min, Mstar_max):
        # Only CSMF needs Mstar arrays
        hod_type_upper = self.hod_type.upper()
        if hod_type_upper != 'CSMF':
            return jnp.zeros(self.n_z), jnp.zeros(self.n_z)

        Mstar_min = jnp.atleast_1d(Mstar_min)
        Mstar_max = jnp.atleast_1d(Mstar_max)

        if len(Mstar_min) == 1:
            Mstar_min = jnp.full(self.n_z, Mstar_min[0])
        if len(Mstar_max) == 1:
            Mstar_max = jnp.full(self.n_z, Mstar_max[0])

        return jnp.asarray(Mstar_min), jnp.asarray(Mstar_max)
    
    def _prepare_median_Mstar(self, median_Mstar):
        if median_Mstar is None:
            return None
        if self.hod_type.upper() != 'CSMF':
            raise ValueError("median_Mstar only for CSMF")
        
        arr = np.atleast_1d(median_Mstar)
        if len(arr) == 1:
            return np.full(self.n_z, float(arr[0]))
        elif len(arr) == self.n_z:
            return np.array(arr)
        raise ValueError(f"median_Mstar length must be 1 or {self.n_z}")
    
    def _precompute_ccl(self):
        """Precompute CCL quantities at GL nodes and M_min.

        When units_per_h=True, user provides HOD masses in Msun/h.
        CCL requires natural units (Msun), so we convert: M_ccl = M_h / h.
        Output quantities are rescaled for h-unit consistency:
        - n_M: multiply by h^3
        - R_s: multiply by h
        """
        M_min_val = 10.0 ** self.log10M_min

        # When units_per_h=True, M_array is in Msun/h (matches user's HOD params)
        # CCL needs natural units (Msun), so convert for CCL calls
        if self.units_per_h:
            M_ccl = self.M_array / self.h  # Msun/h -> Msun
            M_min_ccl = M_min_val / self.h
        else:
            M_ccl = self.M_array
            M_min_ccl = M_min_val

        self._n_M = np.zeros((self.n_z, N_GL))
        self._b_h = np.zeros((self.n_z, N_GL))
        self._R_s = np.zeros((self.n_z, N_GL))
        self._c = np.zeros((self.n_z, N_GL))
        self._Pk_lin = np.zeros((self.n_z, self.n_k))
        self._R_s_Mmin = np.zeros(self.n_z)
        self._c_Mmin = np.zeros(self.n_z)

        for iz, a in enumerate(self.a_array):
            # CCL calls with natural unit masses
            self._n_M[iz] = self.mass_func(self.ccl_cosmo, M_ccl, a)
            self._b_h[iz] = self.halo_bias_model(self.ccl_cosmo, M_ccl, a)

            # Use parent class linear_power() - already handles h-units
            self._Pk_lin[iz] = self.linear_power(z=self.z_array[iz])

            # R_s and concentration from CCL (natural units)
            R_vir = self.mass_def.get_radius(self.ccl_cosmo, M_ccl, a) / a
            self._c[iz] = self.concentration_model(self.ccl_cosmo, M_ccl, a)
            self._R_s[iz] = R_vir / self._c[iz]

            R_vir_Mmin = self.mass_def.get_radius(self.ccl_cosmo, M_min_ccl, a) / a
            self._c_Mmin[iz] = self.concentration_model(self.ccl_cosmo, M_min_ccl, a)
            self._R_s_Mmin[iz] = R_vir_Mmin / self._c_Mmin[iz]

            # Apply h-unit conversions for internal consistency
            if self.units_per_h:
                self._n_M[iz] /= self.h**3   # dn/dlogM factor (includes mass dependence)
                self._R_s[iz] *= self.h       # [Mpc] -> [Mpc/h]
                self._R_s_Mmin[iz] *= self.h

        # Convert to JAX arrays
        self._M_jax = jnp.array(self.M_array)  # Stays in user's units (h-units if units_per_h)
        self._k_jax = jnp.array(self.get_k())  # Uses get_k() which handles h-units
        self._n_M_jax = jnp.array(self._n_M)
        self._b_h_jax = jnp.array(self._b_h)
        self._R_s_jax = jnp.array(self._R_s)
        self._c_jax = jnp.array(self._c)
        self._Pk_lin_jax = jnp.array(self._Pk_lin)
        self._R_s_Mmin_jax = jnp.array(self._R_s_Mmin)
        self._c_Mmin_jax = jnp.array(self._c_Mmin)
    
    def _compute_beta_nl(self):
        """Compute β^NL using the interpax-based interpolator."""
        if not HAS_INTERPAX:
            if self.verbose:
                print("Warning: interpax not available, skipping β^NL")
            return
        
        if self.emu is None:
            if self.verbose:
                print("Warning: DarkEmulator not available, skipping β^NL")
            return
        
        # Default β^NL parameters
        beta_nl_opts = {
            'n_k': 100,
            'n_mass': 20,
            'k_min': 1e-2,
            'k_max': 10.0,
            'log_M_min': 12.0,
            'log_M_max': 15.0,
            'method': 'linear',
            'verbose': self.verbose,
        }
        beta_nl_opts.update(self._beta_nl_kwargs)
        
        # Build interpolator for all redshifts
        self.compute_beta_nl(self.z_array, **beta_nl_opts)
        
        if self.beta_nl_interp is None:
            return
        
        if self.verbose:
            print("  Interpolating β^NL to GL nodes...")
        
        # Convert masses to h-units for emulator
        if self.units_per_h:
            log10M_gl_h=self._log10M_gl_jax
            log10M_min_h = self.log10M_min 
        else:
            log10M_gl_h = self._log10M_gl_jax - jnp.log10(self.h)
            log10M_min_h = self.log10M_min - np.log10(self.h)

        k_h = self.get_k()
        M_min_h = 10.0 ** log10M_min_h
        
        for iz, z in enumerate(self.z_array):
            # Get β^NL on full (k, M, M) grid
            beta_nl_gl = self.beta_nl_interp.interpolate_to_mass_grid(
                log10M_gl_h, float(z), k_target=jnp.array(k_h)
            )
            self._beta_nl_gl_cache[iz] = beta_nl_gl
            
            # β^NL at M_min row/column
            beta_nl_Mmin_row = np.zeros((self.n_k, N_GL))
            for j in range(N_GL):
                M2_h = 10.0 ** float(log10M_gl_h[j])
                beta_nl_Mmin_row[:, j] = np.asarray(
                    self.beta_nl_interp(k_h, M_min_h, M2_h, float(z))
                )
            
            self._beta_nl_Mmin_row_cache[iz] = jnp.array(beta_nl_Mmin_row)
            self._beta_nl_Mmin_col_cache[iz] = self._beta_nl_Mmin_row_cache[iz]  # Symmetric
            
            # β^NL at (M_min, M_min)
            self._beta_nl_Mmin_Mmin_cache[iz] = jnp.array(
                self.beta_nl_interp(k_h, M_min_h, M_min_h, float(z))
            )
        
        if self.verbose:
            print(f"  β^NL cached for {self.n_z} redshifts")
    
    def _get_occupation(self, iz: int):
        """Get N_c, N_s for redshift index iz."""
        # AnalyticalHOD uses log10(M), legacy StandardHOD uses linear M
        if isinstance(self.hod, AnalyticalHOD):
            return self.hod.N_central(self._log10M_gl_jax), self.hod.N_satellite(self._log10M_gl_jax)
        elif isinstance(self.hod, StandardHOD):
            # Legacy StandardHOD uses linear mass
            return self.hod.N_central(self._M_jax), self.hod.N_satellite(self._M_jax)
        elif isinstance(self.hod, CSMF_HOD):
            # CSMF uses linear mass and Mstar bounds
            return (
                self.hod.N_central(self._M_jax, self._Mstar_min_jax[iz], self._Mstar_max_jax[iz]),
                self.hod.N_satellite(self._M_jax, self._Mstar_min_jax[iz], self._Mstar_max_jax[iz])
            )
        else:
            raise ValueError(f"Unknown HOD type: {type(self.hod)}")
    
    def _squeeze(self, arr):
        return arr[0] if self.is_single_z else arr
    
    def set_hod_params(self, hod_params: Dict):
        self.hod.set_params(hod_params)
    
    def update_f(self, f_c=None, f_s=None):
        if f_c is not None:
            self.f_c = f_c
        if f_s is not None:
            self.f_s = f_s
    
    def ngal(self, hod_params=None):
        """Compute galaxy number density."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")
        
        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            n = _compute_ngal(N_c, N_s, self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            result.append(n)
        
        ngal = jnp.array(result)
        # if self.units_per_h:
        #     ngal *= self.h ** 3
        return self._squeeze(np.asarray(ngal))
    
    def Pgg(self, hod_params=None):
        """Compute galaxy-galaxy power spectrum."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")
        
        use_beta_nl = self.include_beta_nl and len(self._beta_nl_gl_cache) > 0
        
        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            
            if use_beta_nl:
                P = _compute_Pgg_with_beta_nl(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                    self._beta_nl_gl_cache[iz],
                )
            else:
                P = _compute_Pgg(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                )
            result.append(P)
        
        Pgg = jnp.stack(result)
        # if self.units_per_h:
        #     Pgg *= self.h ** 3
        return self._squeeze(np.asarray(Pgg))
    
    def Pgm(self, hod_params=None):
        """Compute galaxy-matter power spectrum."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")
        
        use_beta_nl = self.include_beta_nl and len(self._beta_nl_gl_cache) > 0
        
        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            
            if use_beta_nl:
                P = _compute_Pgm_with_beta_nl(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self._M_jax, self.RHO_M,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                    self._beta_nl_gl_cache[iz],
                    self._beta_nl_Mmin_col_cache[iz],
                    self._R_s_Mmin_jax[iz], self._c_Mmin_jax[iz],
                )
            else:
                P = _compute_Pgm(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self._M_jax, self.RHO_M,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                )
            result.append(P)
        
        Pgm = jnp.stack(result)
        # if self.units_per_h:
        #     Pgm *= self.h ** 3
        return self._squeeze(np.asarray(Pgm))
    
    
    def get_A_Mmin(self, iz: int = 0) -> float:
        """Get A(M_min) = 1 - ∫(M/ρ_m)*b*n dM."""
        W_m = self._M_jax / self.RHO_M
        return float(1.0 - gl_integrate(
            W_m * self._b_h_jax[iz] * self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        ))
    
    def effective_halo_mass(self, hod_params=None):
        """Compute effective halo mass."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")
        
        result = []
        for iz in range(self.n_z):
            N_c, _ = self._get_occupation(iz)
            n_c = gl_integrate(N_c * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            M_eff = gl_integrate(N_c * self._n_M_jax[iz] * self._M_jax,
                                 self.log10M_min, self.log10M_max) / n_c
            result.append(M_eff)
        
        M_eff = jnp.array(result)
        if self.units_per_h:
            M_eff *= self.h
        return self._squeeze(np.asarray(M_eff))
    
    def satellite_fraction(self, hod_params=None):
        """Compute satellite fraction."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")
        
        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            n_c = gl_integrate(N_c * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            n_s = gl_integrate(N_s * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            result.append(n_s / (n_c + n_s))
        
        return self._squeeze(np.asarray(result))
    
    def diagnose_beta_nl_terms(self, iz: int = 0, hod_params=None):
        """Diagnostic method to show contribution of each I^NL term."""
        if not self.include_beta_nl:
            print("β^NL not enabled")
            return
        
        if hod_params is not None:
            self.set_hod_params(hod_params)
        
        N_c, N_s = self._get_occupation(iz)
        
        # Compute profiles
        u_c = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], self.f_c)
        u_s = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], self.f_s)
        u_m = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], 1.0)
        u_m_Mmin = nfw_fourier_u_single(self._k_jax, self._R_s_Mmin_jax[iz], self._c_Mmin_jax[iz])
        
        n_gal = gl_integrate((N_c + N_s) * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
        
        # H_g and H_m with profiles
        H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal
        W_m = self._M_jax / self.RHO_M
        H_m = u_m * W_m[None, :]
        
        A_Mmin = self.get_A_Mmin(iz)
        
        # Compute individual terms for P_gm
        I_22_gm = _compute_I_NL_22(
            self._beta_nl_gl_cache[iz], H_g, H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        
        integral_21 = _compute_I_NL_21(
            self._beta_nl_Mmin_col_cache[iz], H_g,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_21_gm = A_Mmin * u_m_Mmin * integral_21
        
        # Compute individual terms for P_gg
        I_22_gg = _compute_I_NL_gg(
            self._beta_nl_gl_cache[iz], H_g,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        
        # Compute individual terms for P_mm
        I_22_mm = _compute_I_NL_22(
            self._beta_nl_gl_cache[iz], H_m, H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        
        I_11_mm = (A_Mmin ** 2) * (u_m_Mmin ** 2) * self._beta_nl_Mmin_Mmin_cache[iz]
        
        integral_12_mm = _compute_I_NL_12(
            self._beta_nl_Mmin_row_cache[iz], H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_12_mm = A_Mmin * u_m_Mmin * integral_12_mm
        
        integral_21_mm = _compute_I_NL_21(
            self._beta_nl_Mmin_col_cache[iz], H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_21_mm = A_Mmin * u_m_Mmin * integral_21_mm
        
        print(f"=== β^NL Diagnostic for z={self.z_array[iz]:.2f} ===")
        print(f"A(M_min) = {A_Mmin:.4f}")
        print(f"  (Mass fraction below M_min = 10^{self.log10M_min:.1f})")
        print()
        print("P_gg:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_gg))):.4e}")
        print()
        print("P_gm:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_gm))):.4e}")
        print(f"  I^21:  mean|I| = {float(jnp.mean(jnp.abs(I_21_gm))):.4e}")
        print(f"  I^21/I^22:     = {float(jnp.mean(jnp.abs(I_21_gm / (I_22_gm + 1e-30)))):.2%}")
        print()
        print("P_mm:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_mm))):.4e}")
        print(f"  I^21:  mean|I| = {float(jnp.mean(jnp.abs(I_21_mm))):.4e}")
        print(f"  I^12:  mean|I| = {float(jnp.mean(jnp.abs(I_12_mm))):.4e}")
        print(f"  I^11:  mean|I| = {float(jnp.mean(jnp.abs(I_11_mm))):.4e}")
        
        return {
            'A_Mmin': A_Mmin,
            'I_22_gg': np.asarray(I_22_gg),
            'I_22_gm': np.asarray(I_22_gm),
            'I_21_gm': np.asarray(I_21_gm),
            'I_22_mm': np.asarray(I_22_mm),
            'I_21_mm': np.asarray(I_21_mm),
            'I_12_mm': np.asarray(I_12_mm),
            'I_11_mm': np.asarray(I_11_mm),
        }
    
    def wgg(self, rp, rp_bins=None, hod_params=None):
        """Compute projected correlation function."""
        if not HAS_HANKEL:
            raise ImportError("Hankel transform utilities not available")
        
        Pgg = self.Pgg(hod_params)
        k = self.get_k()
        
        if self.is_single_z:
            Pgg = Pgg[np.newaxis, :]
        
        result = []
        for iz in range(self.n_z):
            rp_out, wgg_iz = Pk_to_wgg_direct(k, Pgg[iz], rp, rp_bins=rp_bins)
            result.append(wgg_iz)
        
        return rp_out, self._squeeze(np.array(result))
    
    def DeltaSigma(self, rp, rp_bins=None, method='direct', hod_params=None,
                   include_stellar=True):
        """Compute excess surface density."""
        if not HAS_HANKEL:
            raise ImportError("Hankel transform utilities not available")
        
        Pgm = self.Pgm(hod_params)
        k = self.get_k()
        
        if self.is_single_z:
            Pgm = Pgm[np.newaxis, :]
        
        result = []
        for iz in range(self.n_z):
            if method == 'direct':
                rp_out, ds_iz = Pk_to_DeltaSigma_direct(k, Pgm[iz], self.RHO_M, rp, rp_bins=rp_bins)
            elif method == 'traditional':
                rp_out, ds_iz = Pk_gm_to_DeltaSigma_traditional(k, Pgm[iz], self.RHO_M, rp, rp_bins=rp_bins)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if include_stellar and self.median_Mstar is not None:
                ds_stellar = self.median_Mstar[iz] / (np.pi * rp_out**2) / 1e12
                if self.units_per_h:
                    ds_stellar *=self.h

                ds_iz = ds_iz + ds_stellar
            
            result.append(ds_iz)
        
        return rp_out, self._squeeze(np.array(result))


__all__ = [
    # Gauss-Legendre integration
    'N_GL', 'GL_X', 'GL_W',
    'gl_nodes_scaled', 'gl_integrate',
    # NFW Fourier transforms
    'nfw_fourier_u', 'nfw_fourier_u_single',
    # HOD parameter definitions
    'HOD_PARAM_DEFINITIONS', 'CSMF_HOD_PARAMS',
    # Central occupation functions
    'lrg_N_central', 'elg_ghod_N_central', 'elg_sfr_N_central',
    # Satellite occupation functions
    'unified_N_satellite',
    # HOD classes
    'AnalyticalHOD', 'StandardHOD', 'CSMF_HOD', 'create_hod',
    # Main class
    'HaloModel',
    # Hankel transforms
    'HAS_HANKEL',
]