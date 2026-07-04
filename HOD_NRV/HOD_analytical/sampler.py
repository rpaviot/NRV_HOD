"""
CSMF HOD Fitter v2 - Updated for New HaloModel API
====================================================

This module provides a class for fitting the Conditional Stellar Mass Function
(CSMF) HOD model to galaxy-galaxy lensing measurements (Delta Sigma) and
optionally galaxy clustering (WGG) using:
- Nautilus nested sampling algorithm (for full posterior)
- iminuit minimization (for fast best-fit finding)
- scipy differential evolution (for robust global optimization)

Key Updates in v2:
------------------
1. Compatible with the new HaloModel class API (interpax-based)
2. Optional β^NL (non-linear bias) correction support
3. Support for multiple galaxy samples: BGS (SFR, GMM) and LRG
4. Lens magnification correction for LRG samples
5. Flexible sample selection and joint fitting

Features:
---------
- Support for fitting multiple stellar mass bins simultaneously
- Each mass bin has its own stellar mass range and effective redshift
- Uses the multi-redshift CSMF halo model functionality
- Flexible prior specification (uniform, Gaussian, fixed)
- Choice of observables to fit (DeltaSigma, WGG, or both)
- Support for different rp bins for WGG and DeltaSigma
- Efficient model evaluation using pre-computed halo model quantities
- Fast best-fit finding with iminuit (MIGRAD, HESSE, MINOS)

Author: CSMF Fitting Pipeline
Version: 2.0
Modified: January 2025 - Updated for new HaloModel class API with β^NL support
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from scipy.stats import norm
from enum import Enum
import warnings
import os

from HOD_NRV.HOD_analytical.halo_model import HaloModel


# Try to import pyccl
try:
    import pyccl as ccl
    HAS_CCL = True
except ImportError:
    HAS_CCL = False
    warnings.warn("pyccl not installed. Some features may not work.")

# Try to import nautilus
try:
    from nautilus import Prior, Sampler
    HAS_NAUTILUS = True
except ImportError:
    HAS_NAUTILUS = False
    warnings.warn("nautilus-sampler not installed. Install with: pip install nautilus-sampler")

# Try to import iminuit
try:
    from iminuit import Minuit
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False
    warnings.warn("iminuit not installed. Install with: pip install iminuit")

# scipy is always available
from scipy.optimize import differential_evolution, minimize as scipy_minimize


# ============================================================================
# Enums and Constants
# ============================================================================

class SampleType(Enum):
    """Galaxy sample types."""
    BGS_SFR = "BGS_SFR"
    BGS_GMM = "BGS_GMM"
    LRG = "LRG"


# LRG magnification alpha coefficients (alpha - 1 factors)
# alpha values: 2.14, 2.25, 2.7, 3.2 for mass bins 0,1,2,3
# Factor (alpha - 1) because the factor 2 is already in mag_contribution
LRG_ALPHA_MINUS_ONE = {
    0: 2.14 - 1,  # 1.14
    1: 2.25 - 1,  # 1.25
    2: 2.70 - 1,  # 1.70
    3: 3.20 - 1,  # 2.20
}


# ============================================================================
# Default Cosmology
# ============================================================================

DEFAULT_COSMO_PARAMS = {
    'h': 0.6766,
    'Omc': 0.11933 / (0.6766)**2,
    'Omb': 0.02242 / (0.6766)**2,
    's8': 0.8102,
    'A_s': 2.105209331337507e-09,
    'n_s': 0.9665,
    'Omnu': 0.0014034
}


# ============================================================================
# Prior Configuration
# ============================================================================

@dataclass
class ParameterPrior:
    """
    Configuration for a single parameter prior.
    
    Attributes
    ----------
    name : str
        Parameter name
    prior_type : str
        Type of prior: 'uniform', 'gaussian', or 'fixed'
    bounds : tuple, optional
        (min, max) for uniform priors
    mean : float, optional
        Mean for Gaussian priors
    std : float, optional
        Standard deviation for Gaussian priors
    fixed_value : float, optional
        Value for fixed parameters
    """
    name: str
    prior_type: str = 'uniform'
    bounds: Optional[Tuple[float, float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    fixed_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate the prior configuration"""
        if self.prior_type == 'uniform':
            if self.bounds is None:
                raise ValueError(f"Parameter {self.name}: uniform prior requires bounds")
        elif self.prior_type == 'gaussian':
            if self.mean is None or self.std is None:
                raise ValueError(f"Parameter {self.name}: Gaussian prior requires mean and std")
        elif self.prior_type == 'fixed':
            if self.fixed_value is None:
                raise ValueError(f"Parameter {self.name}: fixed prior requires fixed_value")
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")


DEFAULT_CSMF_PRIORS = {
    'M0': ParameterPrior(name='M0', prior_type='uniform', bounds=(8.0, 13.0)),
    'M1': ParameterPrior(name='M1', prior_type='uniform', bounds=(9.0, 14.0)),
    'gamma1': ParameterPrior(name='gamma1', prior_type='uniform', bounds=(2.5, 15.0)),
    #'gamma1': ParameterPrior(name='gamma1', prior_type='gaussian', mean=7,std=2),
    'gamma2': ParameterPrior(name='gamma2', prior_type='uniform', bounds=(0.0, 5.0)),
    'sigma_c': ParameterPrior(name='sigma_c', prior_type='uniform', bounds=(0.01, 1.2)),
    'alpha_s': ParameterPrior(name='alpha_s', prior_type='uniform', bounds=(-5, 5)),
    'b0': ParameterPrior(name='b0', prior_type='uniform', bounds=(-5, 5)),
    'b1': ParameterPrior(name='b1', prior_type='uniform', bounds=(-5, 5)),
    'f_c': ParameterPrior(name='f_c', prior_type='uniform', bounds=(0.2, 1.0)),
    'f_s': ParameterPrior(name='f_s', prior_type='uniform', bounds=(0.2, 1.0)),
}

FIDUCIAL_CSMF_PARAMS = {
    'M0': 10.5,
    'M1': 12.0,
    'gamma1': 7.0,
    'gamma2': 0.2,
    'sigma_c': 0.2,
    'alpha_s': -1.0,
    'b0': -2.0,
    'b1': 1.0,
    'f_c': 0.7,
    'f_s': 0.7
}


# ============================================================================
# Data Container Classes
# ============================================================================

@dataclass
class MassBinData:
    """
    Container for a single stellar mass bin's data.
    
    Supports BGS (SFR/GMM) and LRG samples with optional magnification correction.
    """
    massbin_id: int
    z_eff: float
    logmstar_min: float
    logmstar_max: float
    sample_type: SampleType = SampleType.BGS_SFR
    logmstar_median: Optional[float] = None
    
    # Delta Sigma data
    rp: np.ndarray = field(default_factory=lambda: np.array([]))
    rp_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_sigma_err: np.ndarray = field(default_factory=lambda: np.array([]))
    cov_delta_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # WGG data
    rp_wgg: Optional[np.ndarray] = None
    rp_bins_wgg: Optional[np.ndarray] = None
    wp: Optional[np.ndarray] = None
    wp_err: Optional[np.ndarray] = None
    cov_wgg: Optional[np.ndarray] = None
    
    # Galaxy number density (abundance anchor, optional)
    n_gal: Optional[float] = None          # measured comoving n_gal [h^3/Mpc^3]
    n_gal_err: Optional[float] = None      # its 1-sigma error (same units)

    # Magnification contribution (for LRG)
    mag_contribution: Optional[np.ndarray] = None
    alpha_minus_one: Optional[float] = None  # (alpha - 1) factor for LRG
    
    # Other
    mean_sigma_crit: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.rp_wgg is None:
            self.rp_wgg = self.rp.copy() if len(self.rp) > 0 else np.array([])
        if self.rp_bins_wgg is None:
            self.rp_bins_wgg = self.rp_bins.copy() if len(self.rp_bins) > 0 else np.array([])
        
        # Set alpha_minus_one for LRG if not provided
        if self.sample_type == SampleType.LRG and self.alpha_minus_one is None:
            self.alpha_minus_one = LRG_ALPHA_MINUS_ONE.get(self.massbin_id, 1.0)
    
    def get_corrected_delta_sigma(self) -> np.ndarray:
        """
        Get Delta Sigma with magnification correction applied (for LRG).
        
        For LRG samples: DS_corrected = DS_measured - (alpha - 1) * mag_contribution
        For BGS samples: returns the original delta_sigma unchanged.
        """
        if self.sample_type == SampleType.LRG and self.mag_contribution is not None:
            correction = self.alpha_minus_one * self.mag_contribution
            return self.delta_sigma - correction
        else:
            return self.delta_sigma
    
    @classmethod
    def from_npz(
        cls,
        filepath: str,
        massbin_id: int,
        sample_type: SampleType,
        h: float = 0.6766
    ) -> 'MassBinData':
        """
        Load mass bin data from npz file.
        
        Parameters
        ----------
        filepath : str
            Path to the npz file
        massbin_id : int
            Mass bin identifier
        sample_type : SampleType
            Type of galaxy sample (BGS_SFR, BGS_GMM, or LRG)
        h : float
            Hubble parameter (for unit conversions if needed)
        """
        data = np.load(filepath)
        
        # Handle different key names for rp
        if 'rp_delta_sigma' in data:
            rp = data['rp_delta_sigma']
        elif 'rp' in data:
            rp = data['rp']
        else:
            raise KeyError("Could not find 'rp' or 'rp_delta_sigma' in data file")
        
        rp_wgg = data.get('rp_wgg', rp)
        rp_bins_wgg = data.get('rp_bins_wgg', data['rp_bins'])
        
        # Handle different key names for covariance
        if 'cov_delta_sigma' in data:
            cov_delta_sigma = data['cov_delta_sigma']
        elif 'covariance_matrix' in data:
            cov_delta_sigma = data['covariance_matrix']
        else:
            raise KeyError("Could not find covariance matrix in data file")
        
        # Load galaxy number density (abundance anchor) if present
        n_gal = data.get('n_gal', None)
        n_gal = float(n_gal) if n_gal is not None else None
        n_gal_err = data.get('n_gal_err', None)
        n_gal_err = float(n_gal_err) if n_gal_err is not None else None

        # Load magnification contribution if available (for LRG)
        mag_contribution = data.get('mag_contribution', None)
        
        # Get alpha_minus_one for LRG
        alpha_minus_one = None
        if sample_type == SampleType.LRG:
            alpha_minus_one = LRG_ALPHA_MINUS_ONE.get(massbin_id, 1.0)
        
        return cls(
            massbin_id=massbin_id,
            z_eff=float(data['z_eff']),
            logmstar_min=float(data['logmstar_min']),
            logmstar_max=float(data['logmstar_max']),
            logmstar_median=float(data.get('logmstar_median', 
                                           0.5 * (data['logmstar_min'] + data['logmstar_max']))),
            sample_type=sample_type,
            rp=rp,
            rp_bins=data['rp_bins'],
            delta_sigma=data['delta_sigma'],
            delta_sigma_err=data['delta_sigma_err'],
            cov_delta_sigma=cov_delta_sigma,
            rp_wgg=rp_wgg,
            rp_bins_wgg=rp_bins_wgg,
            wp=data.get('wp', None),
            wp_err=data.get('wp_err', None),
            cov_wgg=data.get('cov_wgg', None),
            n_gal=n_gal,
            n_gal_err=n_gal_err,
            mag_contribution=mag_contribution,
            alpha_minus_one=alpha_minus_one,
            mean_sigma_crit=data.get('mean_sigma_crit', None),
        )


# ============================================================================
# Result Container Classes
# ============================================================================

@dataclass
class MinuitResult:
    """Container for iminuit minimization results."""
    best_fit: Dict[str, float]
    errors: Dict[str, float]
    minos_errors: Optional[Dict[str, Tuple[float, float]]]
    covariance: np.ndarray
    correlation: np.ndarray
    chi2: float
    ndof: int
    reduced_chi2: float
    fval: float
    is_valid: bool
    has_accurate_covar: bool
    param_names: List[str]
    minuit: Any
    prior_penalty: float = 0.0  # NEW: track Gaussian prior penalty
    
    def print_summary(self):
        """Print a summary of the fit results."""
        print("\n" + "="*70)
        print("MINUIT FIT RESULTS")
        print("="*70)
        
        print(f"\nFit status: {'CONVERGED' if self.is_valid else 'FAILED'}")
        print(f"Covariance: {'ACCURATE' if self.has_accurate_covar else 'APPROXIMATE'}")
        print(f"\nχ² = {self.chi2:.2f}")
        print(f"ndof = {self.ndof}")
        print(f"χ²/ndof = {self.reduced_chi2:.3f}")
        
        print(f"\nBest-fit parameters:")
        print("-"*50)
        
        for name in self.param_names:
            val = self.best_fit[name]
            err = self.errors.get(name, 0)
            
            if self.minos_errors and name in self.minos_errors:
                err_lo, err_hi = self.minos_errors[name]
                print(f"  {name:12s} = {val:12.5f}  +{err_hi:.5f}  {err_lo:.5f}")
            else:
                print(f"  {name:12s} = {val:12.5f} ± {err:.5f}")
        
        fixed_params = {k: v for k, v in self.best_fit.items() if k not in self.param_names}
        if fixed_params:
            print(f"\nFixed parameters:")
            print("-"*50)
            for name, val in fixed_params.items():
                print(f"  {name:12s} = {val:12.5f} (FIXED)")
        
        print("="*70)


@dataclass
class DifferentialEvolutionResult:
    """Container for scipy differential evolution results."""
    best_fit: Dict[str, float]
    chi2: float
    ndof: int
    reduced_chi2: float
    success: bool
    message: str
    n_iterations: int
    n_function_evals: int
    param_names: List[str]
    scipy_result: Any
    errors: Optional[Dict[str, float]] = None
    covariance: Optional[np.ndarray] = None
    
    def print_summary(self):
        """Print a summary of the fit results."""
        print("\n" + "="*70)
        print("DIFFERENTIAL EVOLUTION FIT RESULTS")
        print("="*70)
        
        print(f"\nFit status: {'CONVERGED' if self.success else 'FAILED'}")
        print(f"Message: {self.message}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Function evaluations: {self.n_function_evals}")
        print(f"\nχ² = {self.chi2:.2f}")
        print(f"ndof = {self.ndof}")
        print(f"χ²/ndof = {self.reduced_chi2:.3f}")
        
        print(f"\nBest-fit parameters:")
        print("-"*50)
        
        for name in self.param_names:
            val = self.best_fit[name]
            if self.errors and name in self.errors:
                err = self.errors[name]
                print(f"  {name:12s} = {val:12.5f} ± {err:.5f}")
            else:
                print(f"  {name:12s} = {val:12.5f}")
        
        fixed_params = {k: v for k, v in self.best_fit.items() if k not in self.param_names}
        if fixed_params:
            print(f"\nFixed parameters:")
            print("-"*50)
            for name, val in fixed_params.items():
                print(f"  {name:12s} = {val:12.5f} (FIXED)")
        
        print("="*70)


# ============================================================================
# Main CSMF Fitter Class
# ============================================================================

class CSMFFitter:
    """
    CSMF HOD Fitter v2 - Updated for new HaloModel API.
    
    Supports:
    - BGS samples (SFR and GMM selection)
    - LRG samples with magnification correction
    - Optional β^NL (non-linear bias) correction
    - Multiple fitting methods: differential evolution, iminuit, nautilus
    
    Parameters
    ----------
    cosmo_params : dict, optional
        Cosmology parameters. Default uses the fiducial cosmology.
    observables : list of str
        Which observables to fit. Options: ['DeltaSigma', 'WGG']
    rp_min : float, optional
        Minimum r_p to include in fit [Mpc/h]
    rp_max : float, optional
        Maximum r_p to include in fit [Mpc/h]
    rp_min_wgg : float, optional
        Minimum r_p for WGG fit [Mpc/h]
    rp_max_wgg : float, optional
        Maximum r_p for WGG fit [Mpc/h]
    include_beta_nl : bool
        Whether to include β^NL (non-linear bias) correction. Default: False
    beta_nl_kwargs : dict, optional
        Additional kwargs for β^NL computation
    units_per_h : bool
        Whether to use h-units. Default: True
    verbose : bool
        Print progress information. Default: True
    
    Example
    -------
    >>> fitter = CSMFFitter(
    ...     observables=['DeltaSigma'],
    ...     rp_min=0.1,
    ...     rp_max=30.0,
    ...     include_beta_nl=False
    ... )
    >>> fitter.load_bgs_data('/path/to/data/', mass_bins=[0, 1, 2, 3])
    >>> fitter.load_lrg_data('/path/to/data/', mass_bins=[0, 1, 2, 3])
    >>> fitter.set_priors(fixed_params={'f_c': 1.0, 'f_s': 1.0})
    >>> result = fitter.minimize_de(maxiter=1000, workers=-1)
    """
    
    def __init__(
        self,
        cosmo_params: Optional[Dict] = None,
        observables: List[str] = None,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        rp_min_wgg: Optional[float] = None,
        rp_max_wgg: Optional[float] = None,
        include_beta_nl: bool = False,
        beta_nl_kwargs: Optional[Dict] = None,
        beta_nl_source: str = 'emulator',
        k_array: Optional = None,
        units_per_h: bool = True,
        verbose: bool = True
    ):
        self.observables = observables or ['DeltaSigma']
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.rp_min_wgg = rp_min_wgg if rp_min_wgg is not None else rp_min
        self.rp_max_wgg = rp_max_wgg if rp_max_wgg is not None else rp_max
        self.include_beta_nl = include_beta_nl
        self.beta_nl_kwargs = beta_nl_kwargs or {}
        self.beta_nl_source = beta_nl_source
        self.units_per_h = units_per_h
        self.verbose = verbose
        self.k_array = k_array
        
        if cosmo_params is None:
            cosmo_params = DEFAULT_COSMO_PARAMS
        self.cosmo_params = cosmo_params
        
        # Data storage - separate lists for different sample types
        self.bgs_mass_bins: List[MassBinData] = []
        self.lrg_mass_bins: List[MassBinData] = []
        
        # Combined list for fitting
        self.mass_bins: List[MassBinData] = []
        
        self.priors: Dict[str, ParameterPrior] = {}
        self.fixed_params: Dict[str, float] = {}
        
        # Halo model cache
        self._halo_model: Optional[Any] = None
        self._z_eff_array: Optional[np.ndarray] = None
        self._mstar_min_array: Optional[np.ndarray] = None
        self._mstar_max_array: Optional[np.ndarray] = None
        self._massbin_id_to_index: Optional[Dict[int, int]] = None
        self._median_mstar_array: Optional[np.ndarray] = None
        
        # Results storage
        self.sampler: Optional[Any] = None
        self.results: Optional[Dict] = None
        self.minuit_result: Optional[MinuitResult] = None
        self.de_result: Optional[DifferentialEvolutionResult] = None
        
        # Cache for number of data points
        self._n_data_points: Optional[int] = None
        
        if self.verbose:
            print(f"CSMFFitter v2 initialized")
            print(f"  Observables: {self.observables}")
            print(f"  β^NL correction: {'enabled' if include_beta_nl else 'disabled'}")
            print(f"  rp range (DS): [{rp_min}, {rp_max}]")
            print(f"  rp range (WGG): [{self.rp_min_wgg}, {self.rp_max_wgg}]")
    #dsigma_wgg_pip_lowscales_LRG_
    def _get_file_pattern(self, sample_type: SampleType) -> str:
        """Get the file pattern for a given sample type."""
        if sample_type == SampleType.BGS_SFR:
            return 'dsigma_wgg_PIP_lowscales_BGS_massbin{}.npz'
        elif sample_type == SampleType.BGS_GMM:
            return 'dsigma_wgg_PIP_lowscales_BGS_GMM_massbin{}.npz'
        elif sample_type == SampleType.LRG:
            return 'dsigma_wgg_CP_lowscales_LRG_massbin{}.npz'
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
    
    def load_bgs_data(
        self,
        data_dir: str,
        mass_bins: Optional[List[int]] = None,
        selection: str = 'SFR',
        file_pattern: Optional[str] = None
    ):
        """
        Load BGS (Bright Galaxy Survey) data.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the data files
        mass_bins : list of int, optional
            Mass bin indices to load. Default: [0, 1, 2, 3, 4, 5, 6, 7]
        selection : str
            Selection type: 'SFR' or 'GMM'. Default: 'SFR'
        file_pattern : str, optional
            Custom file pattern. If None, uses default for selection type.
        """
        if mass_bins is None:
            mass_bins = list(range(8))
        
        sample_type = SampleType.BGS_SFR if selection.upper() == 'SFR' else SampleType.BGS_GMM
        
        if file_pattern is None:
            file_pattern = self._get_file_pattern(sample_type)
        
        self.bgs_mass_bins = []
        
        for mb in mass_bins:
            filepath = os.path.join(data_dir, file_pattern.format(mb))
            
            if not os.path.exists(filepath):
                warnings.warn(f"File not found: {filepath}")
                continue
            
            data = MassBinData.from_npz(
                filepath, mb, sample_type, self.cosmo_params['h']
            )
            self.bgs_mass_bins.append(data)
            
            if self.verbose:
                print(f"Loaded BGS {selection} mass bin {mb}: z_eff={data.z_eff:.3f}, "
                      f"log(M*)=[{data.logmstar_min:.2f}, {data.logmstar_max:.2f}]")
        
        self.bgs_mass_bins.sort(key=lambda x: x.z_eff)
        self._update_combined_mass_bins()
        
        if self.verbose:
            print(f"Total BGS bins loaded: {len(self.bgs_mass_bins)}")
    
    def load_lrg_data(
        self,
        data_dir: str,
        mass_bins: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
        alpha_values: Optional[Dict[int, float]] = None
    ):
        """
        Load LRG (Luminous Red Galaxy) data with magnification correction.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the data files
        mass_bins : list of int, optional
            Mass bin indices to load. Default: [0, 1, 2, 3]
        file_pattern : str, optional
            Custom file pattern. If None, uses default.
        alpha_values : dict, optional
            Custom alpha values for each mass bin. If None, uses defaults:
            {0: 2.14, 1: 2.25, 2: 2.70, 3: 3.20}
        """
        if mass_bins is None:
            mass_bins = list(range(4))
        
        if file_pattern is None:
            file_pattern = self._get_file_pattern(SampleType.LRG)
        
        # Update alpha values if provided
        if alpha_values is not None:
            for mb, alpha in alpha_values.items():
                LRG_ALPHA_MINUS_ONE[mb] = alpha - 1
        
        self.lrg_mass_bins = []
        
        for mb in mass_bins:
            filepath = os.path.join(data_dir, file_pattern.format(mb))
            
            if not os.path.exists(filepath):
                warnings.warn(f"File not found: {filepath}")
                continue
            
            data = MassBinData.from_npz(
                filepath, mb, SampleType.LRG, self.cosmo_params['h']
            )
            self.lrg_mass_bins.append(data)
            
            if self.verbose:
                has_mag = data.mag_contribution is not None
                mag_status = f"(α-1={data.alpha_minus_one:.2f})" if has_mag else "(no mag)"
                print(f"Loaded LRG mass bin {mb}: z_eff={data.z_eff:.3f}, "
                      f"log(M*)=[{data.logmstar_min:.2f}, {data.logmstar_max:.2f}] {mag_status}")
        
        self.lrg_mass_bins.sort(key=lambda x: x.z_eff)
        self._update_combined_mass_bins()
        
        if self.verbose:
            print(f"Total LRG bins loaded: {len(self.lrg_mass_bins)}")
    
    def _update_combined_mass_bins(self):
        """Update the combined mass bins list from BGS and LRG."""
        self.mass_bins = self.bgs_mass_bins + self.lrg_mass_bins
        self.mass_bins.sort(key=lambda x: (x.sample_type.value, x.z_eff))
        
        if self.verbose and self.mass_bins:
            print(f"\nTotal mass bins for fitting: {len(self.mass_bins)}")
            print(f"  BGS: {len(self.bgs_mass_bins)}")
            print(f"  LRG: {len(self.lrg_mass_bins)}")
    
    def load_data(
        self,
        data_dir: str,
        mass_bins: Optional[List[int]] = None,
        file_pattern: str = 'dsigma_wgg_BGS_massbin{}.npz'
    ):
        """
        Legacy method for backward compatibility.
        Loads data assuming BGS SFR selection.
        """
        self.load_bgs_data(data_dir, mass_bins, selection='SFR', file_pattern=file_pattern)
    
    def set_priors(
        self,
        priors: Optional[Dict[str, ParameterPrior]] = None,
        fixed_params: Optional[Dict[str, float]] = None
    ):
        """Set parameter priors."""
        self.priors = DEFAULT_CSMF_PRIORS.copy()
        
        if priors is not None:
            for name, prior in priors.items():
                self.priors[name] = prior
        
        self.fixed_params = {}
        if fixed_params is not None:
            for name, value in fixed_params.items():
                self.fixed_params[name] = value
                self.priors[name] = ParameterPrior(
                    name=name,
                    prior_type='fixed',
                    fixed_value=value
                )
        
        if self.verbose:
            print("\nParameter configuration:")
            for name, prior in self.priors.items():
                if prior.prior_type == 'fixed':
                    print(f"  {name}: FIXED at {prior.fixed_value}")
                elif prior.prior_type == 'uniform':
                    print(f"  {name}: Uniform{prior.bounds}")
                elif prior.prior_type == 'gaussian':
                    print(f"  {name}: N({prior.mean}, {prior.std})")
    
    def _get_free_param_names(self) -> List[str]:
        """Get list of free (non-fixed) parameter names in consistent order."""
        canonical_order = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 
                          'alpha_s', 'b0', 'b1', 'f_c', 'f_s']
        return [
            name for name in canonical_order
            if name in self.priors and self.priors[name].prior_type != 'fixed'
        ]
    
    def _build_full_params(self, free_params: Dict[str, float]) -> Dict[str, float]:
        """Build full parameter dictionary including fixed params."""
        full_params = {}
        
        for name, prior in self.priors.items():
            if prior.prior_type == 'fixed':
                full_params[name] = prior.fixed_value
        
        full_params.update(free_params)
        return full_params
    
    def _initialize_halo_model(self):
        """Initialize a SINGLE unified halo model for all mass bins."""
        # Import HaloModel here to avoid circular imports
 #       from HOD_NRV.HOD_analytical.halo_model import HaloModel
        
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_bgs_data() or load_lrg_data() first.")
        
        self._z_eff_array = np.array([mb.z_eff for mb in self.mass_bins])
        
        if self.verbose:
            print(f"\nInitializing unified halo model...")
            print(f"  Number of mass bins: {len(self.mass_bins)}")
            print(f"  β^NL correction: {'enabled' if self.include_beta_nl else 'disabled'}")
        
        self._mstar_min_array = np.array([10**mb.logmstar_min for mb in self.mass_bins])
        self._mstar_max_array = np.array([10**mb.logmstar_max for mb in self.mass_bins])
        
        median_mstar_list = []
        for mb in self.mass_bins:
            if mb.logmstar_median is not None:
                median_mstar_list.append(10**mb.logmstar_median)
            else:
                median_mstar_list.append(None)
        
        if all(m is None for m in median_mstar_list):
            self._median_mstar_array = None
        elif any(m is None for m in median_mstar_list):
            raise ValueError(
                "Mixed stellar mass bins: all bins must either have or not have median stellar masses."
            )
        else:
            self._median_mstar_array = np.array(median_mstar_list)
        
        # Create unique identifier for each mass bin
        self._massbin_uid_to_index = {}
        for i, mb in enumerate(self.mass_bins):
            uid = (mb.sample_type.value, mb.massbin_id)
            self._massbin_uid_to_index[uid] = i
        
        # Build beta_nl_kwargs
        beta_nl_kwargs = self.beta_nl_kwargs.copy() if self.beta_nl_kwargs else {}
        if not self.include_beta_nl:
            beta_nl_kwargs = None
        
        # Initialize HaloModel with the NEW API
        self._halo_model = HaloModel(
            cosmo_params=self.cosmo_params,
            z=self._z_eff_array,
            hod_type='CSMF',
            Mstar_min=self._mstar_min_array,
            Mstar_max=self._mstar_max_array,
            f_c=1.0,  # Will be updated in _compute_model_observables
            f_s=1.0,  # Will be updated in _compute_model_observables
            units_per_h=self.units_per_h,
            masses_are_log10=True,
            median_Mstar=self._median_mstar_array,
            include_beta_nl=self.include_beta_nl,
            beta_nl_kwargs=beta_nl_kwargs,
            beta_nl_source=self.beta_nl_source,
            verbose=self.verbose,
            k_array=self.k_array,mass_function='Tinker08'
        )
        
        # Set initial HOD parameters
        init_params = {
            'M0': FIDUCIAL_CSMF_PARAMS['M0'],
            'M1': FIDUCIAL_CSMF_PARAMS['M1'],
            'gamma1': FIDUCIAL_CSMF_PARAMS['gamma1'],
            'gamma2': FIDUCIAL_CSMF_PARAMS['gamma2'],
            'sigma_c': FIDUCIAL_CSMF_PARAMS['sigma_c'],
            'alpha_s': FIDUCIAL_CSMF_PARAMS['alpha_s'],
            'b0': FIDUCIAL_CSMF_PARAMS['b0'],
            'b1': FIDUCIAL_CSMF_PARAMS['b1'],
        }
        self._halo_model.set_hod_params(init_params)
        
        # Count total data points for chi2/ndof calculation
        self._count_data_points()
        
        if self.verbose:
            print(f"\nUnified halo model initialized successfully!")
            print(f"Total data points: {self._n_data_points}")
    
    def _count_data_points(self):
        """Count total number of data points after scale cuts."""
        n_total = 0
        
        for mass_bin in self.mass_bins:
            if 'DeltaSigma' in self.observables:
                _, _, _, mask = self._apply_scale_cuts(mass_bin)
                n_total += np.sum(mask)
            
            if 'WGG' in self.observables and mass_bin.wp is not None:
                _, _, _, mask_wgg = self._apply_scale_cuts_wgg(mass_bin)
                n_total += np.sum(mask_wgg)
        
        self._n_data_points = n_total
    
    def _compute_model_observables(
        self,
        params: Dict[str, float],
    ) -> Tuple[Dict[Tuple, np.ndarray], Dict[Tuple, Optional[np.ndarray]]]:
        """
        Compute model observables for ALL mass bins simultaneously.
        
        Returns dictionaries keyed by (sample_type, massbin_id) tuples.
        """
        # Extract CSMF HOD parameters
        csmf_params = {
            'M0': params['M0'],
            'M1': params['M1'],
            'gamma1': params['gamma1'],
            'gamma2': params['gamma2'],
            'sigma_c': params['sigma_c'],
            'alpha_s': params['alpha_s'],
            'b0': params['b0'],
            'b1': params['b1'],
        }
        
        # Update HOD parameters
        self._halo_model.set_hod_params(csmf_params)
        
        # Update profile rescaling factors
        self._halo_model.update_f(
            f_c=params.get('f_c', 1.0),
            f_s=params.get('f_s', 1.0)
        )
        
        # Get rp arrays from first mass bin (assuming all have same rp)
        rp_ds = self.mass_bins[0].rp
        rp_bins_ds = self.mass_bins[0].rp_bins
        
        # Compute Delta Sigma for all redshifts
        _, ds_all = self._halo_model.DeltaSigma(
            rp=rp_ds,
            rp_bins=rp_bins_ds)#include_stripping=False)
        
        # Build output dictionary
        ds_dict = {}
        for i, mass_bin in enumerate(self.mass_bins):
            uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
            if self._halo_model.is_single_z:
                ds_dict[uid] = np.asarray(ds_all)
            else:
                ds_dict[uid] = np.asarray(ds_all[i])
        
        # Compute WGG if needed
        wgg_dict = {}
        if 'WGG' in self.observables:
            has_wgg = any(mb.wp is not None for mb in self.mass_bins)
            
            if has_wgg:
                rp_wgg = self.mass_bins[0].rp_wgg
                rp_bins_wgg = self.mass_bins[0].rp_bins_wgg
                
                _, wgg_all = self._halo_model.wgg(
                    rp=rp_wgg,
                    rp_bins=rp_bins_wgg,
                )
                
                for i, mass_bin in enumerate(self.mass_bins):
                    uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
                    if mass_bin.wp is not None:
                        if self._halo_model.is_single_z:
                            wgg_dict[uid] = np.asarray(wgg_all)
                        else:
                            wgg_dict[uid] = np.asarray(wgg_all[i])
                    else:
                        wgg_dict[uid] = None
            else:
                for mass_bin in self.mass_bins:
                    uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
                    wgg_dict[uid] = None
        else:
            for mass_bin in self.mass_bins:
                uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
                wgg_dict[uid] = None
        
        return ds_dict, wgg_dict
    
    def _apply_scale_cuts(
        self, 
        mass_bin: MassBinData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply r_p scale cuts to Delta Sigma data.
        
        For LRG samples, returns magnification-corrected Delta Sigma.
        """
        rp = mass_bin.rp
        # Use corrected delta sigma for LRG
        ds = mass_bin.get_corrected_delta_sigma()
        cov = mass_bin.cov_delta_sigma
        
        mask = np.ones(len(rp), dtype=bool)
        
        if self.rp_min is not None:
            mask &= (rp >= self.rp_min)
        if self.rp_max is not None:
            mask &= (rp <= self.rp_max)
        
        rp_cut = rp[mask]
        ds_cut = ds[mask]
        cov_cut = cov[np.ix_(mask, mask)]
        
        return rp_cut, ds_cut, cov_cut, mask
    
    def _apply_scale_cuts_wgg(
        self, 
        mass_bin: MassBinData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply r_p scale cuts to WGG data."""
        rp = mass_bin.rp_wgg
        wp = mass_bin.wp
        cov = mass_bin.cov_wgg
        
        mask = np.ones(len(rp), dtype=bool)
        
        if self.rp_min_wgg is not None:
            mask &= (rp >= self.rp_min_wgg)
        if self.rp_max_wgg is not None:
            mask &= (rp <= self.rp_max_wgg)
        
        rp_cut = rp[mask]
        wp_cut = wp[mask]
        cov_cut = cov[np.ix_(mask, mask)]
        
        return rp_cut, wp_cut, cov_cut, mask
    
    def log_likelihood(self, param_dict: Dict[str, float]) -> float:
        """Compute log-likelihood for the CSMF model."""
        params = self._build_full_params(param_dict)
        
        try:
            ds_dict, wgg_dict = self._compute_model_observables(params)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Model evaluation failed: {e}")
            return -1e100
        
        total_log_L = 0.0
        
        for mass_bin in self.mass_bins:
            uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
            
            if 'DeltaSigma' in self.observables:
                rp_cut, ds_data, cov, mask = self._apply_scale_cuts(mass_bin)
                
                if len(rp_cut) == 0:
                    continue
                
                try:
                    ds_model_full = ds_dict[uid]
                    ds_model = ds_model_full[mask]
                    
                    if not np.all(np.isfinite(ds_model)):
                        return -1e100
                    
                    residual = ds_data - ds_model
                    
                    try:
                        cov_inv = np.linalg.inv(cov)
                        chi2 = residual @ cov_inv @ residual
                    except np.linalg.LinAlgError:
                        err = mass_bin.delta_sigma_err[mask]
                        chi2 = np.sum((residual / err)**2)
                    
                    log_L_bin = -0.5 * chi2
                    total_log_L += log_L_bin
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: DeltaSigma likelihood failed: {e}")
                    return -1e100
            
            if 'WGG' in self.observables and mass_bin.wp is not None:
                rp_wgg_cut, wp_data, cov_wgg, mask_wgg = self._apply_scale_cuts_wgg(mass_bin)
                
                if len(rp_wgg_cut) == 0:
                    continue
                
                try:
                    wgg_model_full = wgg_dict[uid]
                    
                    if wgg_model_full is None:
                        continue
                    
                    wgg_model = wgg_model_full[mask_wgg]
                    
                    if not np.all(np.isfinite(wgg_model)):
                        return -1e100
                    
                    residual = wp_data - wgg_model
                    
                    try:
                        cov_inv = np.linalg.inv(cov_wgg)
                        chi2 = residual @ cov_inv @ residual
                    except np.linalg.LinAlgError:
                        err = mass_bin.wp_err[mask_wgg]
                        chi2 = np.sum((residual / err)**2)
                    
                    log_L_bin = -0.5 * chi2
                    total_log_L += log_L_bin
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: WGG likelihood failed: {e}")
                    return -1e100

        # Galaxy number-density (abundance) anchor: Gaussian chi^2 per mass bin on
        # the model n_gal vs the measured n_gal (h^3/Mpc^3; units_per_h=True). HOD
        # params were already set on the halo model by _compute_model_observables.
        if 'ngal' in self.observables:
            try:
                ng_model = np.asarray(self._halo_model.ngal()).ravel()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: n_gal evaluation failed: {e}")
                return -1e100
            for i, mass_bin in enumerate(self.mass_bins):
                if mass_bin.n_gal is None or mass_bin.n_gal_err is None:
                    continue
                if not np.isfinite(ng_model[i]):
                    return -1e100
                r = (ng_model[i] - mass_bin.n_gal) / mass_bin.n_gal_err
                total_log_L += -0.5 * r * r

        return total_log_L

    def build_batched_loglike(self, jit: bool = True):
        """Build a ``jax.vmap``-batched ΔΣ log-likelihood for ``vectorized=True``.

        Returns ``(loglike_fn, free_names)``. ``loglike_fn`` maps a
        ``(n_batch, n_free)`` array of free-parameter points — columns ordered as
        ``free_names`` — to a ``(n_batch,)`` numpy array of log-likelihoods. It is
        the batched twin of :meth:`log_likelihood`: pure data χ² summed over mass
        bins, with the prior (including any Gaussian γ1) left to nautilus's prior
        transform exactly as in the serial path.

        Mechanics: the de-stated pure forward pass
        ``HaloModel.make_deltasigma_jax`` (theta[10] -> ΔΣ[n_z, nbin]) is wrapped in
        ``jax.vmap`` over the batch; data, scale-cut mask and inverse covariance are
        pre-computed once per bin. Assumes a shared rp grid / scale cut across bins
        (the model already evaluates all bins on ``mass_bins[0].rp``) and
        DeltaSigma-only observables.
        """
        import jax
        import jax.numpy as jnp
        from HOD_NRV.utilsf.hankel_jax import build_direct_deltasigma

        if 'WGG' in self.observables:
            raise NotImplementedError(
                "build_batched_loglike supports DeltaSigma-only fits.")

        if self._halo_model is None:
            self._initialize_halo_model()
        hm = self._halo_model

        # Pure forward pass theta[10] -> ΔΣ[n_z, nbin] on the model's k-grid.
        k = np.asarray(hm.get_k())
        rp_bins = np.asarray(self.mass_bins[0].rp_bins)
        rp_centers = np.asarray(self.mass_bins[0].rp)
        ds_builder = build_direct_deltasigma(k, rp_bins)
        predict = hm.make_deltasigma_jax(ds_builder, rp_centers)

        # Per-bin data / mask / inverse covariance (bins in z order = vmap order).
        masks, data_rows, covinv_rows = [], [], []
        for mb in self.mass_bins:
            _, ds_data, cov, mask = self._apply_scale_cuts(mb)
            masks.append(mask)
            data_rows.append(ds_data)
            covinv_rows.append(np.linalg.inv(cov))
        mask0 = masks[0]
        if not all(np.array_equal(m, mask0) for m in masks):
            raise ValueError(
                "build_batched_loglike assumes a shared rp scale cut across bins.")
        mask_idx = jnp.asarray(np.where(mask0)[0])         # kept-bin indices
        ds_data_all = jnp.asarray(np.stack(data_rows))     # (n_z, ncut)
        covinv_all = jnp.asarray(np.stack(covinv_rows))    # (n_z, ncut, ncut)

        # Optional galaxy number-density (abundance) anchor: a Gaussian n_gal term
        # per mass bin, the batched twin of the serial log_likelihood's. Bins with
        # no measured n_gal get zero inverse-variance (no contribution).
        use_ngal = 'ngal' in self.observables
        ng_predict = vng = ng_data = ng_invvar = None
        if use_ngal:
            ng_predict = hm.make_ngal_jax()                # theta[10] -> (n_z,)
            vng = jax.vmap(ng_predict)                      # (n,10) -> (n, n_z)
            ngd, ngiv = [], []
            for mb in self.mass_bins:
                if mb.n_gal is None or mb.n_gal_err is None:
                    ngd.append(0.0); ngiv.append(0.0)
                else:
                    ngd.append(float(mb.n_gal))
                    ngiv.append(1.0 / float(mb.n_gal_err) ** 2)
            ng_data = jnp.asarray(ngd)                      # (n_z,)
            ng_invvar = jnp.asarray(ngiv)                   # (n_z,)

        # Static map: nautilus free-param columns -> 10-slot theta vector.
        names = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s',
                 'b0', 'b1', 'f_c', 'f_s']
        free_names = [n for n in names
                      if n in self.priors and self.priors[n].prior_type != 'fixed']
        free_idx = {n: i for i, n in enumerate(free_names)}
        const_val = {}
        for n in names:
            if n in free_idx:
                continue
            if n in self.priors and self.priors[n].prior_type == 'fixed':
                const_val[n] = float(self.priors[n].fixed_value)
            else:
                const_val[n] = 1.0     # f_c / f_s absent entirely -> profile default
        take_col = [free_idx.get(n, -1) for n in names]    # -1 == fixed constant
        consts = [const_val.get(n, 0.0) for n in names]

        vpredict = jax.vmap(predict)                       # (n,10) -> (n, n_z, nbin)

        def _loglike(points):
            points = jnp.asarray(points)
            npts = points.shape[0]
            cols = [points[:, take_col[s]] if take_col[s] >= 0
                    else jnp.full((npts,), consts[s]) for s in range(len(names))]
            theta = jnp.stack(cols, axis=1)                # (n, 10)
            ds = vpredict(theta)                           # (n, n_z, nbin)
            resid = ds_data_all[None] - ds[:, :, mask_idx]  # (n, n_z, ncut)
            chi2 = jnp.einsum('nzi,zij,nzj->n', resid, covinv_all, resid)
            good = jnp.isfinite(ds).all(axis=(1, 2))
            if use_ngal:
                ng = vng(theta)                            # (n, n_z)
                rng = ng - ng_data[None]                   # (n, n_z)
                chi2 = chi2 + jnp.sum(rng * rng * ng_invvar[None], axis=1)
                good = good & jnp.isfinite(ng).all(axis=1)
            logL = -0.5 * chi2
            good = good & jnp.isfinite(logL)
            return jnp.where(good, logL, -1e100)

        compiled = jax.jit(_loglike) if jit else _loglike

        def loglike_fn(points):
            return np.asarray(compiled(points))

        return loglike_fn, free_names

    # ========================================================================
    # Minimization Methods
    # ========================================================================
    
    def _get_initial_values(
        self, 
        start_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Get initial parameter values for minimization."""
        param_names = self._get_free_param_names()
        init_vals = {}
        
        for name in param_names:
            if start_params and name in start_params:
                init_vals[name] = start_params[name]
            elif name in FIDUCIAL_CSMF_PARAMS:
                init_vals[name] = FIDUCIAL_CSMF_PARAMS[name]
            else:
                prior = self.priors[name]
                if prior.prior_type == 'uniform':
                    init_vals[name] = 0.5 * (prior.bounds[0] + prior.bounds[1])
                elif prior.prior_type == 'gaussian':
                    init_vals[name] = prior.mean
        
        return init_vals
    
    def _get_parameter_limits(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Get parameter limits for iminuit."""
        param_names = self._get_free_param_names()
        limits = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                limits[name] = prior.bounds
            elif prior.prior_type == 'gaussian':
                limits[name] = (prior.mean - 5*prior.std, prior.mean + 5*prior.std)
            else:
                limits[name] = (None, None)
        
        return limits
    
    def _get_parameter_errors(self) -> Dict[str, float]:
        """Get initial step sizes for iminuit."""
        param_names = self._get_free_param_names()
        errors = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                errors[name] = 0.01 * (prior.bounds[1] - prior.bounds[0])
            elif prior.prior_type == 'gaussian':
                errors[name] = 0.1 * prior.std
            else:
                errors[name] = 0.1
        
        return errors
    
    def _compute_prior_penalty(self, param_dict: Dict[str, float]) -> float:
            """
            Compute the negative log-prior penalty for Gaussian priors.
            
            For uniform priors, returns 0 (no penalty within bounds).
            For Gaussian priors, returns 0.5 * ((x - mean) / std)^2
            
            Parameters
            ----------
            param_dict : dict
                Dictionary of parameter names and values
            
            Returns
            -------
            penalty : float
                The negative log-prior contribution (to be added to -log_likelihood)
            """
            penalty = 0.0
            
            for name, value in param_dict.items():
                if name not in self.priors:
                    continue
                
                prior = self.priors[name]
                
                if prior.prior_type == 'gaussian':
                    # Gaussian prior: -log(P) = 0.5 * ((x-μ)/σ)²
                    z = (value - prior.mean) / prior.std
                    penalty += 0.5 * z**2
                
                elif prior.prior_type == 'uniform':
                    # Check bounds (should already be enforced by limits)
                    if prior.bounds is not None:
                        if value < prior.bounds[0] or value > prior.bounds[1]:
                            return 1e100
            
            return penalty
    
    def _get_de_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for differential evolution."""
        param_names = self._get_free_param_names()
        bounds = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                bounds[name] = prior.bounds
            elif prior.prior_type == 'gaussian':
                bounds[name] = (prior.mean - 5*prior.std, prior.mean + 5*prior.std)
            else:
                bounds[name] = (-100, 100)
        
        return bounds
    
    def minimize(
        self,
        start_params: Optional[Dict[str, float]] = None,
        run_hesse: bool = True,
        run_minos: bool = False,
        minos_params: Optional[List[str]] = None,
        print_level: int = 1,
        **minuit_kwargs
    ) -> MinuitResult:
        """
        Find best-fit parameters using iminuit minimization.
        
        Parameters
        ----------
        start_params : dict, optional
            Starting parameter values.
        run_hesse : bool
            Run HESSE after MIGRAD for error estimation. Default: True
        run_minos : bool
            Run MINOS for asymmetric errors. Default: False
        minos_params : list, optional
            Parameters for which to run MINOS.
        print_level : int
            Minuit print level. Default: 1
        **minuit_kwargs
            Additional arguments passed to Minuit
        
        Returns
        -------
        result : MinuitResult
            Container with best-fit values, errors, covariance, etc.
        """
        if not HAS_IMINUIT:
            raise ImportError("iminuit is required. Install with: pip install iminuit")
        
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_bgs_data() or load_lrg_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        if self._halo_model is None:
            self._initialize_halo_model()
        
        param_names = self._get_free_param_names()
        init_vals = self._get_initial_values(start_params)
        limits = self._get_parameter_limits()
        init_errors = self._get_parameter_errors()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("MINUIT MINIMIZATION")
            print(f"{'='*70}")
            print(f"Free parameters ({len(param_names)}): {param_names}")
            print(f"Fixed parameters: {list(self.fixed_params.keys())}")
            print(f"\nInitial values:")
            for name in param_names:
                print(f"  {name}: {init_vals[name]:.4f}")
        
# Cost function WITH Gaussian prior support
        def cost_func(*args):
            param_dict = {name: val for name, val in zip(param_names, args)}
            
            # Negative log-likelihood
            neg_log_L = -self.log_likelihood(param_dict)
            
            # Add Gaussian prior penalty terms
            prior_penalty = self._compute_prior_penalty(param_dict)
            
            return neg_log_L + prior_penalty
        
        m = Minuit(cost_func, *[init_vals[name] for name in param_names], name=param_names)
        
        for name, (lo, hi) in limits.items():
            m.limits[name] = (lo, hi)
        
        for name, err in init_errors.items():
            m.errors[name] = err
        
        m.print_level = print_level
        
        if self.verbose:
            print("\nRunning MIGRAD...")
        
        m.migrad()
        
        if not m.valid:
            warnings.warn("MIGRAD did not converge!")
        
        if run_hesse:
            if self.verbose:
                print("Running HESSE...")
            m.hesse()
        
        minos_errors = None
        if run_minos:
            if self.verbose:
                print("Running MINOS...")
            
            if minos_params is None:
                m.minos()
            else:
                for param in minos_params:
                    if param in param_names:
                        m.minos(param)
            
            minos_errors = {}
            for name in param_names:
                if name in m.merrors:
                    merr = m.merrors[name]
                    minos_errors[name] = (merr.lower, merr.upper)
        
        best_fit = {name: m.values[name] for name in param_names}
        best_fit.update(self.fixed_params)
        
        errors = {name: m.errors[name] for name in param_names}
        
        if m.covariance is not None:
            covariance = np.array(m.covariance)
            correlation = np.array(m.covariance.correlation())
        else:
            n_free = len(param_names)
            covariance = np.zeros((n_free, n_free))
            correlation = np.eye(n_free)
        
        chi2 = 2 * m.fval
        n_free_params = len(param_names)
        ndof = self._n_data_points - n_free_params
        reduced_chi2 = chi2 / ndof if ndof > 0 else np.inf
        
# Calculate prior penalty at best-fit for reporting
        best_fit_free = {name: m.values[name] for name in param_names}
        prior_penalty = self._compute_prior_penalty(best_fit_free)
        
        self.minuit_result = MinuitResult(
            best_fit=best_fit,
            errors=errors,
            minos_errors=minos_errors,
            covariance=covariance,
            correlation=correlation,
            chi2=chi2,
            ndof=ndof,
            reduced_chi2=reduced_chi2,
            fval=m.fval,
            is_valid=m.valid,
            has_accurate_covar=m.accurate,
            param_names=param_names,
            minuit=m,
            prior_penalty=prior_penalty  # NEW
        )
        
        if self.verbose:
            self.minuit_result.print_summary()
        
        return self.minuit_result
    
    def minimize_de(
        self,
        strategy: str = 'best1bin',
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: Optional[int] = None,
        workers: int = 1,
        polish: bool = True,
        compute_errors: bool = True,
        disp: bool = False,
        callback: Optional[Callable] = None,
        **de_kwargs
    ) -> DifferentialEvolutionResult:
        """
        Find best-fit parameters using scipy differential evolution.
        
        Parameters
        ----------
        strategy : str
            Differential evolution strategy. Default: 'best1bin'
        maxiter : int
            Maximum number of generations. Default: 1000
        popsize : int
            Population size multiplier. Default: 15
        tol : float
            Relative tolerance for convergence. Default: 0.01
        mutation : tuple
            Mutation constant. Default: (0.5, 1.0)
        recombination : float
            Recombination constant. Default: 0.7
        seed : int, optional
            Random seed for reproducibility
        workers : int
            Number of parallel workers. Default: 1 (use -1 for all cores)
        polish : bool
            Use L-BFGS-B to polish the best result. Default: True
        compute_errors : bool
            Estimate parameter errors. Default: True
        disp : bool
            Print convergence messages. Default: False
        callback : callable, optional
            Function called after each iteration
        **de_kwargs
            Additional arguments passed to differential_evolution
        
        Returns
        -------
        result : DifferentialEvolutionResult
            Container with best-fit values, chi2, errors, etc.
        """
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_bgs_data() or load_lrg_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        if self._halo_model is None:
            self._initialize_halo_model()
        
        param_names = self._get_free_param_names()
        bounds = self._get_de_bounds()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("DIFFERENTIAL EVOLUTION MINIMIZATION")
            print(f"{'='*70}")
            print(f"Free parameters ({len(param_names)}): {param_names}")
            print(f"Fixed parameters: {list(self.fixed_params.keys())}")
            print(f"\nBounds:")
            for name in param_names:
                lo, hi = bounds[name]
                print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
            print(f"\nSettings:")
            print(f"  strategy: {strategy}")
            print(f"  maxiter: {maxiter}")
            print(f"  popsize: {popsize} (total pop: {popsize * len(param_names)})")
            print(f"  tol: {tol}")
            print(f"  workers: {workers}")
            print(f"  polish: {polish}")
        
        def cost_func(x):
            param_dict = {name: val for name, val in zip(param_names, x)}
            log_L = self.log_likelihood(param_dict)
            
            if not np.isfinite(log_L):
                return 1e100
            
            return -2.0 * log_L
        
        bounds_list = [bounds[name] for name in param_names]
        
        if self.verbose:
            print("\nRunning differential evolution...")
        
        result = differential_evolution(
            cost_func,
            bounds=bounds_list,
            strategy=strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            workers=workers,
            polish=polish,
            disp=disp,
            callback=callback,
            **de_kwargs
        )
        
        best_fit = {name: val for name, val in zip(param_names, result.x)}
        best_fit.update(self.fixed_params)
        
        chi2 = result.fun
        n_free_params = len(param_names)
        ndof = self._n_data_points - n_free_params
        reduced_chi2 = chi2 / ndof if ndof > 0 else np.inf
        
        errors = None
        covariance = None
        
        if compute_errors and result.success:
            if self.verbose:
                print("Computing parameter errors via Hessian...")
            
            try:
                errors, covariance = self._compute_errors_finite_diff(
                    cost_func, result.x, param_names
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not compute errors: {e}")
        
        self.de_result = DifferentialEvolutionResult(
            best_fit=best_fit,
            chi2=chi2,
            ndof=ndof,
            reduced_chi2=reduced_chi2,
            success=result.success,
            message=result.message,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
            param_names=param_names,
            scipy_result=result,
            errors=errors,
            covariance=covariance
        )
        
        if self.verbose:
            self.de_result.print_summary()
        
        return self.de_result
    
    def _compute_errors_finite_diff(
        self,
        cost_func: Callable,
        x_best: np.ndarray,
        param_names: List[str],
        eps: float = 1e-4
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute parameter errors via finite difference Hessian."""
        n_params = len(x_best)
        hessian = np.zeros((n_params, n_params))
        
        for i in range(n_params):
            for j in range(i, n_params):
                x_pp = x_best.copy()
                x_pm = x_best.copy()
                x_mp = x_best.copy()
                x_mm = x_best.copy()
                
                hi = eps * max(abs(x_best[i]), 1.0)
                hj = eps * max(abs(x_best[j]), 1.0)
                
                x_pp[i] += hi
                x_pp[j] += hj
                x_pm[i] += hi
                x_pm[j] -= hj
                x_mp[i] -= hi
                x_mp[j] += hj
                x_mm[i] -= hi
                x_mm[j] -= hj
                
                f_pp = cost_func(x_pp)
                f_pm = cost_func(x_pm)
                f_mp = cost_func(x_mp)
                f_mm = cost_func(x_mm)
                
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj)
                hessian[j, i] = hessian[i, j]
        
        try:
            covariance = 2.0 * np.linalg.inv(hessian)
            
            diag = np.diag(covariance)
            if np.any(diag < 0):
                warnings.warn("Some covariance diagonal elements are negative. "
                              "Errors may be unreliable.")
                diag = np.abs(diag)
            
            errors = {name: np.sqrt(diag[i]) for i, name in enumerate(param_names)}
            
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert Hessian. Returning no errors.")
            errors = None
            covariance = None
        
        return errors, covariance
    
    # ========================================================================
    # Nautilus Sampling Methods
    # ========================================================================
    
    def _build_nautilus_prior(self) -> 'Prior':
        """Build nautilus Prior object from parameter configuration."""
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler is required")
        
        prior = Prior()
        csmf_param_names = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']
        optional_params = ['f_c', 'f_s']
        
        for name in csmf_param_names + optional_params:
            if name not in self.priors:
                continue
            
            param_prior = self.priors[name]
            
            if param_prior.prior_type == 'fixed':
                continue
            elif param_prior.prior_type == 'uniform':
                prior.add_parameter(name, dist=param_prior.bounds)
            elif param_prior.prior_type == 'gaussian':
                prior.add_parameter(
                    name,
                    dist=norm(loc=param_prior.mean, scale=param_prior.std)
                )
        
        return prior
    
    def run(
        self,
        n_live: int = 2000,
        n_eff: int = 10000,
        discard_exploration: bool = True,
        filepath: Optional[str] = None,
        seed: Optional[int] = None,
        pool: Optional[Any] = None,
        vectorized: bool = False,
        **nautilus_kwargs
    ) -> Dict:
        """
        Run the Nautilus nested sampler.
        
        Parameters
        ----------
        n_live : int
            Number of live points. Default: 2000
        n_eff : int
            Target effective sample size. Default: 10000
        discard_exploration : bool
            Discard exploration phase samples. Default: True
        filepath : str, optional
            Path to save sampler state
        seed : int, optional
            Random seed
        pool : optional
            Multiprocessing pool
        **nautilus_kwargs
            Additional arguments passed to Sampler
        
        Returns
        -------
        results : dict
            Dictionary containing posterior samples and evidence
        """
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler is required")
        
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_bgs_data() or load_lrg_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        if self._halo_model is None:
            self._initialize_halo_model()
        
        nautilus_prior = self._build_nautilus_prior()

        if self.verbose:
            print(f"\nFree parameters: {list(nautilus_prior.keys)}")
            print(f"Fixed parameters: {self.fixed_params}")

        if vectorized:
            likelihood, free_names = self.build_batched_loglike()
            # nautilus passes points as (n_batch, n_dim) in prior-key order with
            # pass_dict=False; our batched columns must follow the same order.
            if list(nautilus_prior.keys) != free_names:
                raise RuntimeError(
                    "Free-parameter ordering mismatch between nautilus prior "
                    f"{list(nautilus_prior.keys)} and batched likelihood {free_names}")
            vec_kwargs = dict(vectorized=True, pass_dict=False)
            if self.verbose:
                print("Using vmap-batched vectorized likelihood (pass_dict=False).")
        else:
            likelihood = self.log_likelihood
            vec_kwargs = {}

        self.sampler = Sampler(
            nautilus_prior,
            likelihood,
            n_live=n_live,
            filepath=filepath,
            seed=seed,
            pool=pool,
            **vec_kwargs,
            **nautilus_kwargs
        )
        
        if self.verbose:
            print("\nStarting Nautilus sampler...")
        
        self.sampler.run(
            n_eff=n_eff,
            discard_exploration=discard_exploration,
            verbose=self.verbose
        )
        
        points, log_w, log_l = self.sampler.posterior()
        
        self.results = {
            'points': points,
            'log_w': log_w,
            'log_l': log_l,
            'log_z': self.sampler.log_z,
            'param_names': list(nautilus_prior.keys),
            'fixed_params': self.fixed_params.copy(),
        }
        
        if self.verbose:
            print(f"\nSampling complete!")
            print(f"log(Z) = {self.sampler.log_z:.2f}")
        
        return self.results
    
    # ========================================================================
    # Results and Diagnostics
    # ========================================================================
    
    def get_best_fit(self) -> Dict[str, float]:
        """Get best-fit parameters (from sampling or minimization)."""
        if self.de_result is not None:
            return self.de_result.best_fit.copy()
        
        if self.minuit_result is not None:
            return self.minuit_result.best_fit.copy()
        
        if self.results is None:
            raise ValueError("No results available. Run minimize_de(), minimize(), or run() first.")
        
        idx_best = np.argmax(self.results['log_l'])
        best_point = self.results['points'][idx_best]
        best_fit = dict(zip(self.results['param_names'], best_point))
        best_fit.update(self.results['fixed_params'])
        
        return best_fit
    
    def get_posterior_summary(self) -> Dict[str, Dict[str, float]]:
        """Get posterior summary statistics from sampling."""
        if self.results is None:
            raise ValueError("No sampling results available. Run run() first.")
        
        points = self.results['points']
        weights = np.exp(self.results['log_w'])
        weights /= weights.sum()
        
        summary = {}
        
        for i, name in enumerate(self.results['param_names']):
            values = points[:, i]
            
            mean = np.average(values, weights=weights)
            var = np.average((values - mean)**2, weights=weights)
            std = np.sqrt(var)
            
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            
            median = sorted_values[np.searchsorted(cumsum, 0.5)]
            p16 = sorted_values[np.searchsorted(cumsum, 0.16)]
            p84 = sorted_values[np.searchsorted(cumsum, 0.84)]
            
            summary[name] = {
                'median': median,
                'mean': mean,
                'std': std,
                'p16': p16,
                'p84': p84,
                'err_low': median - p16,
                'err_high': p84 - median,
            }
        
        return summary
    
    def compute_model_prediction(
        self,
        params: Optional[Dict[str, float]] = None,
        return_components: bool = False
    ) -> Dict[Tuple, Dict[str, np.ndarray]]:
        """
        Compute model predictions for all mass bins.
        
        Parameters
        ----------
        params : dict, optional
            Parameters to use. If None, uses best-fit.
        return_components : bool
            If True, return separate 1-halo and 2-halo terms (not implemented)
        
        Returns
        -------
        predictions : dict
            Dictionary keyed by (sample_type, massbin_id) with model predictions
        """
        if params is None:
            params = self.get_best_fit()
        
        ds_dict, wgg_dict = self._compute_model_observables(params)
        
        predictions = {}
        for mass_bin in self.mass_bins:
            uid = (mass_bin.sample_type.value, mass_bin.massbin_id)
            predictions[uid] = {
                'rp': mass_bin.rp,
                'delta_sigma_model': ds_dict[uid],
                'delta_sigma_data': mass_bin.get_corrected_delta_sigma(),
                'delta_sigma_err': mass_bin.delta_sigma_err,
                'z_eff': mass_bin.z_eff,
                'sample_type': mass_bin.sample_type.value,
            }
            
            if wgg_dict[uid] is not None:
                predictions[uid]['rp_wgg'] = mass_bin.rp_wgg
                predictions[uid]['wp_model'] = wgg_dict[uid]
                predictions[uid]['wp_data'] = mass_bin.wp
                predictions[uid]['wp_err'] = mass_bin.wp_err
        
        return predictions
    
    def save_results(self, filepath: str):
        """Save results to npz file."""
        save_dict = {}
        
        # Save mass bin info
        save_dict['n_bgs_bins'] = len(self.bgs_mass_bins)
        save_dict['n_lrg_bins'] = len(self.lrg_mass_bins)
        save_dict['include_beta_nl'] = self.include_beta_nl
        
        if self.results is not None:
            save_dict.update({
                'points': self.results['points'],
                'log_w': self.results['log_w'],
                'log_l': self.results['log_l'],
                'log_z': self.results['log_z'],
                'param_names': np.array(self.results['param_names'], dtype=str),
            })
            for name, value in self.results['fixed_params'].items():
                save_dict[f'fixed_{name}'] = value
        
        if self.minuit_result is not None:
            save_dict['minuit_best_fit'] = np.array([
                self.minuit_result.best_fit[name] 
                for name in self.minuit_result.param_names
            ])
            save_dict['minuit_errors'] = np.array([
                self.minuit_result.errors[name] 
                for name in self.minuit_result.param_names
            ])
            save_dict['minuit_covariance'] = self.minuit_result.covariance
            save_dict['minuit_chi2'] = self.minuit_result.chi2
            save_dict['minuit_ndof'] = self.minuit_result.ndof
            save_dict['minuit_param_names'] = np.array(self.minuit_result.param_names, dtype=str)
        
        if self.de_result is not None:
            save_dict['de_best_fit'] = np.array([
                self.de_result.best_fit[name] 
                for name in self.de_result.param_names
            ])
            save_dict['de_chi2'] = self.de_result.chi2
            save_dict['de_ndof'] = self.de_result.ndof
            save_dict['de_param_names'] = np.array(self.de_result.param_names, dtype=str)
            save_dict['de_success'] = self.de_result.success
            save_dict['de_n_iterations'] = self.de_result.n_iterations
            save_dict['de_n_function_evals'] = self.de_result.n_function_evals
            
            if self.de_result.errors is not None:
                save_dict['de_errors'] = np.array([
                    self.de_result.errors[name] 
                    for name in self.de_result.param_names
                ])
            if self.de_result.covariance is not None:
                save_dict['de_covariance'] = self.de_result.covariance
        
        np.savez(filepath, **save_dict)
        
        if self.verbose:
            print(f"Results saved to: {filepath}")
    
    def print_data_summary(self):
        """Print a summary of loaded data."""
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        
        if self.bgs_mass_bins:
            print(f"\nBGS Mass Bins ({len(self.bgs_mass_bins)}):")
            print("-"*50)
            for mb in self.bgs_mass_bins:
                print(f"  Bin {mb.massbin_id}: z={mb.z_eff:.3f}, "
                      f"log(M*)=[{mb.logmstar_min:.2f}, {mb.logmstar_max:.2f}], "
                      f"type={mb.sample_type.value}")
        
        if self.lrg_mass_bins:
            print(f"\nLRG Mass Bins ({len(self.lrg_mass_bins)}):")
            print("-"*50)
            for mb in self.lrg_mass_bins:
                has_mag = mb.mag_contribution is not None
                mag_str = f"α-1={mb.alpha_minus_one:.2f}" if has_mag else "no mag"
                print(f"  Bin {mb.massbin_id}: z={mb.z_eff:.3f}, "
                      f"log(M*)=[{mb.logmstar_min:.2f}, {mb.logmstar_max:.2f}], "
                      f"{mag_str}")
        
        print(f"\nTotal mass bins for fitting: {len(self.mass_bins)}")
        print("="*70)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_fitter_from_config(
    config_dict: Dict,
    data_dir: str,
    bgs_mass_bins: Optional[List[int]] = None,
    lrg_mass_bins: Optional[List[int]] = None,
    bgs_selection: str = 'SFR'
) -> CSMFFitter:
    """
    Create a CSMFFitter from a configuration dictionary.
    

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with keys:
        - cosmo_params: cosmology parameters
        - observables: list of observables to fit
        - rp_min, rp_max: scale cuts
        - include_beta_nl: whether to include β^NL correction
        - priors: parameter prior configurations
        - fixed_params: fixed parameter values
    data_dir : str
        Directory containing data files
    bgs_mass_bins : list of int, optional
        BGS mass bin indices to load
    lrg_mass_bins : list of int, optional
        LRG mass bin indices to load
    bgs_selection : str
        BGS selection type: 'SFR' or 'GMM'
    
    Returns
    -------
    fitter : CSMFFitter
        Configured CSMFFitter instance
    """
    fitter = CSMFFitter(
        cosmo_params=config_dict.get('cosmo_params'),
        observables=config_dict.get('observables', ['DeltaSigma']),
        rp_min=config_dict.get('rp_min'),
        rp_max=config_dict.get('rp_max'),
        rp_min_wgg=config_dict.get('rp_min_wgg'),
        rp_max_wgg=config_dict.get('rp_max_wgg'),
        include_beta_nl=config_dict.get('include_beta_nl', False),
        beta_nl_kwargs=config_dict.get('beta_nl_kwargs'),
        verbose=config_dict.get('verbose', True)
    )
    
    # Load BGS data
    if bgs_mass_bins is not None:
        fitter.load_bgs_data(data_dir, bgs_mass_bins, selection=bgs_selection)
    
    # Load LRG data
    if lrg_mass_bins is not None:
        fitter.load_lrg_data(data_dir, lrg_mass_bins)
    
    # Set priors
    priors = {}
    if 'priors' in config_dict:
        for name, prior_config in config_dict['priors'].items():
            priors[name] = ParameterPrior(name=name, **prior_config)
    
    fitter.set_priors(
        priors=priors if priors else None,
        fixed_params=config_dict.get('fixed_params')
    )
    
    return fitter


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums and Constants
    'SampleType',
    'LRG_ALPHA_MINUS_ONE',
    'DEFAULT_COSMO_PARAMS',
    'DEFAULT_CSMF_PRIORS',
    'FIDUCIAL_CSMF_PARAMS',
    
    # Data Classes
    'ParameterPrior',
    'MassBinData',
    'MinuitResult',
    'DifferentialEvolutionResult',
    
    # Main Fitter
    'CSMFFitter',
    
    # Convenience Functions
    'create_fitter_from_config',
    
    # Flags
    'HAS_CCL',
    'HAS_NAUTILUS',
    'HAS_IMINUIT',
]