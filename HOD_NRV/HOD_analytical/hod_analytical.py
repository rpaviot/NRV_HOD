"""
HOD Occupation Functions and Wrapper Classes
=============================================

Analytical HOD occupation functions (central + satellite) and wrapper classes
for use in halo model power spectrum calculations.

Contents:
- Gauss-Legendre integration utilities
- HOD parameter definitions and validation
- Central occupation functions (LRG, ELG_GHOD, ELG_SFR)
- Satellite occupation (unified)
- Legacy functions (standard, CSMF)
- Wrapper classes: AnalyticalHOD, StandardHOD, CSMF_HOD
- Factory function: create_hod
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf as erf_jax
from scipy.special import roots_legendre
from typing import Dict, Optional, Union, List

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


__all__ = [
    # Gauss-Legendre integration
    'N_GL', 'GL_X', 'GL_W',
    'gl_nodes_scaled', 'gl_integrate',
    # HOD parameter definitions
    'HOD_PARAM_DEFINITIONS', 'CSMF_HOD_PARAMS',
    'get_required_params', 'validate_hod_params',
    # Central occupation functions
    'lrg_N_central', 'elg_ghod_N_central', 'elg_sfr_N_central',
    # Satellite occupation functions
    'unified_N_satellite',
    # Legacy functions
    'standard_N_central', 'standard_N_satellite',
    'csmf_Mstar_central', 'csmf_N_central', 'csmf_N_satellite',
    # HOD classes
    'AnalyticalHOD', 'StandardHOD', 'CSMF_HOD', 'create_hod',
]
