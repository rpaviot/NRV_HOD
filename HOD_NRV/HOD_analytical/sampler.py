"""
CSMF HOD Fitter using Nautilus Sampler and iminuit Minimization
================================================================

This module provides a class for fitting the Conditional Stellar Mass Function
(CSMF) HOD model to galaxy-galaxy lensing measurements (Delta Sigma) and
optionally galaxy clustering (WGG) using:
- Nautilus nested sampling algorithm (for full posterior)
- iminuit minimization (for fast best-fit finding)

Features:
- Support for fitting multiple stellar mass bins simultaneously
- Each mass bin has its own stellar mass range and effective redshift
- Uses the multi-redshift CSMF halo model functionality
- Flexible prior specification (uniform, Gaussian, fixed)
- Choice of observables to fit (DeltaSigma, WGG, or both)
- Support for different rp bins for WGG and DeltaSigma
- Efficient model evaluation using pre-computed halo model quantities
- Fast best-fit finding with iminuit (MIGRAD, HESSE, MINOS)

Author: CSMF Fitting Pipeline
Modified: December 2025 - Added iminuit minimization support
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from scipy.stats import norm
import pyccl as ccl
import warnings
import os


from HOD_NRV.HOD_analytical.halo_model import (
    MultiRedshiftHaloModel, HODType, CSMF_HOD,
    create_hod, CSMF_HOD_PARAMS
)


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
    from iminuit.util import describe
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False
    warnings.warn("iminuit not installed. Install with: pip install iminuit")

# scipy is always available
from scipy.optimize import differential_evolution, minimize as scipy_minimize
from scipy.optimize import OptimizeResult


# ============================================================================
# Default Cosmology (from null_test.py)
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
    'M0': ParameterPrior(
        name='M0',
        prior_type='uniform',
        bounds=(8.0, 13.0)
    ),
    'M1': ParameterPrior(
        name='M1',
        prior_type='uniform',
        bounds=(9.0, 14.0)
    ),
    'gamma1': ParameterPrior(
        name='gamma1',
        prior_type='uniform',
        bounds=(1.0, 15.0)
    ),
    'gamma2': ParameterPrior(
        name='gamma2',
        prior_type='uniform',
        bounds=(0.0, 3.0)
    ),
    'sigma_c': ParameterPrior(
        name='sigma_c',
        prior_type='uniform',
        bounds=(0.1, 0.9)
    ),
    'alpha_s': ParameterPrior(
        name='alpha_s',
        prior_type='gaussian',
        mean=-1.1,
        std=1.5
    ),
    'b0': ParameterPrior(
        name='b0',
        prior_type='gaussian',
        mean=0.0,
        std=1.5
    ),
    'b1': ParameterPrior(
        name='b1',
        prior_type='gaussian',
        mean=1.5,
        std=2.0
    ),
    'f_c': ParameterPrior(
        name='f_c',
        prior_type='uniform',
        bounds=(0.4, 1.0)
    ),
    'f_s': ParameterPrior(
        name='f_s',
        prior_type='uniform',
        bounds=(0.4, 1.0)
    ),
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
    'f_c': 1.0,
    'f_s': 1.0
}


@dataclass
class MassBinData:
    """
    Container for a single stellar mass bin's data.
    """
    massbin_id: int
    z_eff: float
    logmstar_min: float
    logmstar_max: float
    logmstar_median: Optional[float] = None
    rp: np.ndarray = field(default_factory=lambda: np.array([]))
    rp_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_sigma_err: np.ndarray = field(default_factory=lambda: np.array([]))
    cov_delta_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    rp_wgg: Optional[np.ndarray] = None
    rp_bins_wgg: Optional[np.ndarray] = None
    wp: Optional[np.ndarray] = None
    wp_err: Optional[np.ndarray] = None
    cov_wgg: Optional[np.ndarray] = None
    mag_contribution: Optional[np.ndarray] = None
    mean_sigma_crit: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.rp_wgg is None:
            self.rp_wgg = self.rp.copy()
        if self.rp_bins_wgg is None:
            self.rp_bins_wgg = self.rp_bins.copy()
    
    @classmethod
    def from_npz(cls, filepath: str, massbin_id: int, h: float) -> 'MassBinData':
        data = np.load(filepath)

        if 'rp_delta_sigma' in data:
            rp = data['rp_delta_sigma']
        elif 'rp' in data:
            rp = data['rp']
        else:
            raise KeyError("Could not find 'rp' or 'rp_delta_sigma' in data file")

        rp_wgg = data.get('rp_wgg', rp)
        rp_bins_wgg = data.get('rp_bins_wgg', data['rp_bins'])

        if 'cov_delta_sigma' in data:
            cov_delta_sigma = data['cov_delta_sigma']
        elif 'covariance_matrix' in data:
            cov_delta_sigma = data['covariance_matrix']
        else:
            raise KeyError("Could not find covariance matrix in data file")

        return cls(
            massbin_id=massbin_id,
            z_eff=float(data['z_eff']),
            logmstar_min=float(data['logmstar_min']),# - 2*np.log10(DEFAULT_COSMO_PARAMS['h']),
            logmstar_max=float(data['logmstar_max']),# - 2*np.log10(DEFAULT_COSMO_PARAMS['h']),
            logmstar_median=float(data['logmstar_median']),#- 2*np.log10(DEFAULT_COSMO_PARAMS['h']),
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
            mag_contribution=data.get('mag_contribution', None),
            mean_sigma_crit=data.get('mean_sigma_crit', None),
        )


# ============================================================================
# Minuit Result Container
# ============================================================================

@dataclass
class MinuitResult:
    """
    Container for iminuit minimization results.
    
    Attributes
    ----------
    best_fit : dict
        Best-fit parameter values (including fixed params)
    errors : dict
        Parameter errors from HESSE (symmetric)
    minos_errors : dict, optional
        Asymmetric errors from MINOS (if computed)
    covariance : np.ndarray
        Covariance matrix for free parameters
    correlation : np.ndarray
        Correlation matrix for free parameters
    chi2 : float
        Chi-squared at best fit (-2 * log_likelihood)
    ndof : int
        Number of degrees of freedom
    reduced_chi2 : float
        Reduced chi-squared (chi2 / ndof)
    fval : float
        Function value at minimum (-log_likelihood or chi2/2)
    is_valid : bool
        Whether the fit converged
    has_accurate_covar : bool
        Whether covariance matrix is accurate
    param_names : list
        Names of free parameters (in order)
    minuit : Minuit
        The underlying Minuit object for further analysis
    """
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
    minuit: Any  # Minuit object
    
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
        
        # Print fixed parameters
        fixed_params = {k: v for k, v in self.best_fit.items() if k not in self.param_names}
        if fixed_params:
            print(f"\nFixed parameters:")
            print("-"*50)
            for name, val in fixed_params.items():
                print(f"  {name:12s} = {val:12.5f} (FIXED)")
        
        print("="*70)


@dataclass
class DifferentialEvolutionResult:
    """
    Container for scipy differential evolution results.
    
    Attributes
    ----------
    best_fit : dict
        Best-fit parameter values (including fixed params)
    chi2 : float
        Chi-squared at best fit (-2 * log_likelihood)
    ndof : int
        Number of degrees of freedom
    reduced_chi2 : float
        Reduced chi-squared (chi2 / ndof)
    success : bool
        Whether the optimization converged
    message : str
        Convergence message from scipy
    n_iterations : int
        Number of iterations
    n_function_evals : int
        Number of function evaluations
    param_names : list
        Names of free parameters (in order)
    scipy_result : OptimizeResult
        The underlying scipy result object
    errors : dict, optional
        Parameter errors (if computed via finite differences)
    covariance : np.ndarray, optional
        Covariance matrix (if computed)
    """
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
        
        # Print fixed parameters
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
    CSMF HOD Fitter using Nautilus nested sampling and iminuit minimization.
    
    This class handles fitting the CSMF HOD model to galaxy-galaxy lensing
    (Delta Sigma) and optionally galaxy clustering (w_p) measurements.
    
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
    units_per_h : bool
        Whether to use h-units. Default: True
    verbose : bool
        Print progress information. Default: True
    """
    
    def __init__(
        self,
        cosmo_params: Optional[Dict] = None,
        observables: List[str] = None,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        rp_min_wgg: Optional[float] = None,
        rp_max_wgg: Optional[float] = None,
        units_per_h: bool = True,
        verbose: bool = True
    ):
        self.observables = observables or ['DeltaSigma']
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.rp_min_wgg = rp_min_wgg if rp_min_wgg is not None else rp_min
        self.rp_max_wgg = rp_max_wgg if rp_max_wgg is not None else rp_max
        self.units_per_h = units_per_h
        self.verbose = verbose
        
        if cosmo_params is None:
            cosmo_params = DEFAULT_COSMO_PARAMS
        self.cosmo_params = cosmo_params
        self._setup_cosmology()
        
        self.mass_bins: List[MassBinData] = []
        self.priors: Dict[str, ParameterPrior] = {}
        self.fixed_params: Dict[str, float] = {}
        
        self._halo_model: Optional[MultiRedshiftHaloModel] = None
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
        
        # Cache for number of data points (for chi2/ndof calculation)
        self._n_data_points: Optional[int] = None
    
    def _setup_cosmology(self):
        """Initialize CCL cosmology object"""
        Omc = self.cosmo_params['Omc']
        Omb = self.cosmo_params['Omb']
        h = self.cosmo_params['h']
        As = self.cosmo_params['A_s']
        ns = self.cosmo_params['n_s']
        mnu = 0.06
        
        self.cosmo = ccl.Cosmology(
            Omega_c=Omc,
            Omega_b=Omb,
            m_nu=mnu,
            h=h,
            A_s=As,
            n_s=ns,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='camb',
            mass_split='normal',
            extra_parameters={"camb": {"halofit_version": "mead2020"}}
        )
        
        if self.verbose:
            print(f"Cosmology initialized: Ωc={Omc:.4f}, Ωb={Omb:.4f}, h={h}")
    
    def load_data(
        self,
        data_dir: str,
        mass_bins: Optional[List[int]] = None,
        file_pattern: str = 'dsigma_wgg_massbin{}_SFR.npz'
    ):
        """Load Delta Sigma measurements for specified mass bins."""
        if mass_bins is None:
            mass_bins = list(range(8))
        
        self.mass_bins = []

        for mb in mass_bins:
            filepath = os.path.join(data_dir, file_pattern.format(mb))

            if not os.path.exists(filepath):
                warnings.warn(f"File not found: {filepath}")
                continue

            data = MassBinData.from_npz(filepath, mb, self.cosmo_params['h'])
            self.mass_bins.append(data)
            if self.verbose:
                print(f"Loaded mass bin {mb}: z_eff={data.z_eff:.3f}, "
                      f"log(M*)=[{data.logmstar_min:.2f}, {data.logmstar_max:.2f}]")

        self.mass_bins.sort(key=lambda x: x.z_eff)

        if self.verbose:
            print(f"Total: {len(self.mass_bins)} mass bins loaded")
    
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
    
    def _get_free_param_names(self) -> List[str]:
        """Get list of free (non-fixed) parameter names in consistent order."""
        # Define the canonical order
        canonical_order = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1', 'f_c', 'f_s']
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
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self._z_eff_array = np.array([mb.z_eff for mb in self.mass_bins])
        
        if self.verbose:
            print(f"\nInitializing unified halo model...")
            print(f"  Number of mass bins: {len(self.mass_bins)}")
        
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

        self._massbin_id_to_index = {
            mb.massbin_id: i for i, mb in enumerate(self.mass_bins)
        }
        
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
        
        self._halo_model = MultiRedshiftHaloModel(
            self.cosmo,
            hod_type=HODType.CSMF,
            hod_params=init_params,
            z_array=self._z_eff_array,
            units_per_h=self.units_per_h,
            masses_are_log10=True,
            median_Mstar=self._median_mstar_array,
            verbose=self.verbose
        )
        
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
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Optional[np.ndarray]]]:
        """Compute model observables for ALL mass bins simultaneously."""
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

        self._halo_model.update_hod_params(csmf_params)

        rp_ds = self.mass_bins[0].rp
        rp_bins_ds = self.mass_bins[0].rp_bins
        
        _, ds_all = self._halo_model.compute_DeltaSigma_all_z(
            rp=rp_ds,
            bin_avg=True,
            rp_bins=rp_bins_ds,
            Mstar_min=self._mstar_min_array,
            Mstar_max=self._mstar_max_array,
            verbose=False
        )

        ds_dict = {}
        for i, mass_bin in enumerate(self.mass_bins):
            if self._halo_model.is_single_z:
                ds_dict[mass_bin.massbin_id] = ds_all
            else:
                ds_dict[mass_bin.massbin_id] = ds_all[i]

        wgg_dict = {}
        if 'WGG' in self.observables:
            has_wgg = any(mb.wp is not None for mb in self.mass_bins)
            
            if has_wgg:
                rp_wgg = self.mass_bins[0].rp_wgg
                rp_bins_wgg = self.mass_bins[0].rp_bins_wgg
                
                _, wgg_all = self._halo_model.compute_wgg_all_z(
                    rp=rp_wgg,
                    bin_avg=True,
                    rp_bins=rp_bins_wgg,
                    Mstar_min=self._mstar_min_array,
                    Mstar_max=self._mstar_max_array,
                    verbose=False
                )
                
                for i, mass_bin in enumerate(self.mass_bins):
                    if mass_bin.wp is not None:
                        if self._halo_model.is_single_z:
                            wgg_dict[mass_bin.massbin_id] = wgg_all
                        else:
                            wgg_dict[mass_bin.massbin_id] = wgg_all[i]
                    else:
                        wgg_dict[mass_bin.massbin_id] = None
            else:
                for mass_bin in self.mass_bins:
                    wgg_dict[mass_bin.massbin_id] = None
        else:
            for mass_bin in self.mass_bins:
                wgg_dict[mass_bin.massbin_id] = None

        return ds_dict, wgg_dict
    
    def _apply_scale_cuts(self, mass_bin: MassBinData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply r_p scale cuts to Delta Sigma data."""
        rp = mass_bin.rp
        ds = mass_bin.delta_sigma
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
    
    def _apply_scale_cuts_wgg(self, mass_bin: MassBinData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            if 'DeltaSigma' in self.observables:
                rp_cut, ds_data, cov, mask = self._apply_scale_cuts(mass_bin)

                if len(rp_cut) == 0:
                    continue

                try:
                    ds_model_full = ds_dict[mass_bin.massbin_id]
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
                    wgg_model_full = wgg_dict[mass_bin.massbin_id]

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

        return total_log_L
    
    # ========================================================================
    # iminuit Minimization Methods
    # ========================================================================
    
    def _create_minuit_cost_function(self) -> Tuple[Callable, List[str]]:
        """
        Create a cost function compatible with iminuit.
        
        iminuit expects a function where parameter names are the argument names.
        We create a wrapper that converts positional/keyword arguments to our
        param_dict format.
        
        Returns
        -------
        cost_func : callable
            Function with signature cost_func(M0, M1, gamma1, ...) -> float
            Returns -log_likelihood (to minimize)
        param_names : list
            List of parameter names in order
        """
        param_names = self._get_free_param_names()
        
        # Create a function with explicit parameter names
        # iminuit uses introspection, so we need to define parameters explicitly
        def cost_function(*args, **kwargs):
            # Build param dict from args and kwargs
            param_dict = {}
            
            # Handle positional arguments
            for i, val in enumerate(args):
                if i < len(param_names):
                    param_dict[param_names[i]] = val
            
            # Handle keyword arguments
            param_dict.update(kwargs)
            
            # Return negative log-likelihood (for minimization)
            log_L = self.log_likelihood(param_dict)
            return -log_L
        
        # Set the function signature for iminuit introspection
        # This is a workaround since we can't dynamically create function signatures
        cost_function._parameters = {name: None for name in param_names}
        
        return cost_function, param_names
    
    def _get_initial_values(self, start_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get initial parameter values for minimization.
        
        Parameters
        ----------
        start_params : dict, optional
            User-provided starting values
        
        Returns
        -------
        init_vals : dict
            Initial values for all free parameters
        """
        param_names = self._get_free_param_names()
        init_vals = {}
        
        for name in param_names:
            if start_params and name in start_params:
                init_vals[name] = start_params[name]
            elif name in FIDUCIAL_CSMF_PARAMS:
                init_vals[name] = FIDUCIAL_CSMF_PARAMS[name]
            else:
                # Use center of prior range for uniform, mean for Gaussian
                prior = self.priors[name]
                if prior.prior_type == 'uniform':
                    init_vals[name] = 0.5 * (prior.bounds[0] + prior.bounds[1])
                elif prior.prior_type == 'gaussian':
                    init_vals[name] = prior.mean
        
        return init_vals
    
    def _get_parameter_limits(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Get parameter limits for iminuit.
        
        Returns
        -------
        limits : dict
            Dictionary mapping param names to (min, max) tuples
        """
        param_names = self._get_free_param_names()
        limits = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                limits[name] = prior.bounds
            elif prior.prior_type == 'gaussian':
                # For Gaussian priors, use ±5 sigma as soft bounds
                limits[name] = (prior.mean - 5*prior.std, prior.mean + 5*prior.std)
            else:
                limits[name] = (None, None)
        
        return limits
    
    def _get_parameter_errors(self) -> Dict[str, float]:
        """
        Get initial step sizes for iminuit.
        
        Returns
        -------
        errors : dict
            Initial step sizes for each parameter
        """
        param_names = self._get_free_param_names()
        errors = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                # Use 1% of range as initial step
                errors[name] = 0.01 * (prior.bounds[1] - prior.bounds[0])
            elif prior.prior_type == 'gaussian':
                # Use 10% of sigma as initial step
                errors[name] = 0.1 * prior.std
            else:
                errors[name] = 0.1
        
        return errors
    
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
        
        This method uses MIGRAD for minimization, optionally followed by
        HESSE for error estimation and MINOS for asymmetric errors.
        
        Parameters
        ----------
        start_params : dict, optional
            Starting parameter values. If not provided, uses fiducial values
            or center of prior ranges.
        run_hesse : bool
            Run HESSE after MIGRAD for error estimation. Default: True
        run_minos : bool
            Run MINOS for asymmetric errors. Default: False (slower)
        minos_params : list, optional
            Parameters for which to run MINOS. If None, runs for all params.
        print_level : int
            Minuit print level (0=quiet, 1=normal, 2=verbose). Default: 1
        **minuit_kwargs
            Additional arguments passed to Minuit
        
        Returns
        -------
        result : MinuitResult
            Container with best-fit values, errors, covariance, etc.
        
        Examples
        --------
        >>> # Basic usage
        >>> result = fitter.minimize()
        >>> result.print_summary()
        
        >>> # With custom starting point
        >>> result = fitter.minimize(start_params={'M0': 10.0, 'M1': 12.5})
        
        >>> # With MINOS errors for specific parameters
        >>> result = fitter.minimize(run_minos=True, minos_params=['M0', 'M1'])
        """
        if not HAS_IMINUIT:
            raise ImportError("iminuit is required. Install with: pip install iminuit")
        
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        # Initialize halo model if needed
        if self._halo_model is None:
            self._initialize_halo_model()
        
        # Get parameter configuration
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
        
        # Create cost function that iminuit can use
        # iminuit passes positional arguments, so we need to map them to param names
        def cost_func(*args):
            # Map positional arguments to parameter dictionary
            param_dict = {name: val for name, val in zip(param_names, args)}
            return -self.log_likelihood(param_dict)
        
        # Create Minuit object with positional function
        m = Minuit(cost_func, *[init_vals[name] for name in param_names], name=param_names)
        
        # Set limits
        for name, (lo, hi) in limits.items():
            m.limits[name] = (lo, hi)
        
        # Set initial step sizes
        for name, err in init_errors.items():
            m.errors[name] = err
        
        # Set print level
        m.print_level = print_level
        
        # Run MIGRAD
        if self.verbose:
            print("\nRunning MIGRAD...")
        
        m.migrad()
        
        if not m.valid:
            warnings.warn("MIGRAD did not converge!")
        
        # Run HESSE for error estimation
        if run_hesse:
            if self.verbose:
                print("Running HESSE...")
            m.hesse()
        
        # Run MINOS for asymmetric errors
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
            
            # Extract MINOS errors
            minos_errors = {}
            for name in param_names:
                if name in m.merrors:
                    merr = m.merrors[name]
                    minos_errors[name] = (merr.lower, merr.upper)
        
        # Build results
        # m.values and m.errors are iminuit objects, need to extract properly
        best_fit = {name: m.values[name] for name in param_names}
        best_fit.update(self.fixed_params)
        
        errors = {name: m.errors[name] for name in param_names}
        
        # Get covariance and correlation matrices
        if m.covariance is not None:
            covariance = np.array(m.covariance)
            correlation = np.array(m.covariance.correlation())
        else:
            n_free = len(param_names)
            covariance = np.zeros((n_free, n_free))
            correlation = np.eye(n_free)
        
        # Calculate chi2 and ndof
        chi2 = 2 * m.fval  # fval is -log_likelihood, so chi2 = -2*log_L = 2*fval
        n_free_params = len(param_names)
        ndof = self._n_data_points - n_free_params
        reduced_chi2 = chi2 / ndof if ndof > 0 else np.inf
        
        # Create result object
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
            minuit=m
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
        
        This is a robust global optimizer that works well even when the
        likelihood surface has multiple local minima or returns -inf for
        some parameter combinations.
        
        Parameters
        ----------
        strategy : str
            Differential evolution strategy. Default: 'best1bin'
            Options: 'best1bin', 'best1exp', 'rand1exp', 'randtobest1exp',
                     'currenttobest1bin', 'currenttobest1exp', 'rand1bin',
                     'rand2bin', 'rand2exp', 'best2bin', 'best2exp'
        maxiter : int
            Maximum number of generations. Default: 1000
        popsize : int
            Population size multiplier. Total population is popsize * n_params.
            Default: 15
        tol : float
            Relative tolerance for convergence. Default: 0.01
        mutation : tuple
            Mutation constant (dithering range). Default: (0.5, 1.0)
        recombination : float
            Recombination constant (crossover probability). Default: 0.7
        seed : int, optional
            Random seed for reproducibility
        workers : int
            Number of parallel workers. -1 uses all CPUs. Default: 1
        polish : bool
            Use L-BFGS-B to polish the best result. Default: True
        compute_errors : bool
            Estimate parameter errors via finite differences at minimum.
            Default: True
        disp : bool
            Print convergence messages. Default: False
        callback : callable, optional
            Function called after each iteration: callback(xk, convergence)
        **de_kwargs
            Additional arguments passed to differential_evolution
        
        Returns
        -------
        result : DifferentialEvolutionResult
            Container with best-fit values, chi2, errors (if computed), etc.
        
        Examples
        --------
        >>> # Basic usage - robust global optimization
        >>> result = fitter.minimize_de()
        >>> result.print_summary()
        
        >>> # Faster but less thorough search
        >>> result = fitter.minimize_de(maxiter=500, popsize=10, polish=False)
        
        >>> # Parallel execution
        >>> result = fitter.minimize_de(workers=-1)  # Use all CPUs
        
        >>> # With custom callback to monitor progress
        >>> def my_callback(xk, convergence):
        ...     print(f"Convergence: {convergence:.4f}")
        >>> result = fitter.minimize_de(callback=my_callback)
        """
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        # Initialize halo model if needed
        if self._halo_model is None:
            self._initialize_halo_model()
        
        # Get parameter configuration
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
        
        # Create cost function that returns chi2 (handles -inf gracefully)
        def cost_func(x):
            param_dict = {name: val for name, val in zip(param_names, x)}
            log_L = self.log_likelihood(param_dict)
            
            # Handle -inf by returning a large but finite value
            if not np.isfinite(log_L):
                return 1e100
            
            # Return chi2 = -2 * log_L
            return -2.0 * log_L
        
        # Build bounds list in parameter order
        bounds_list = [bounds[name] for name in param_names]
        
        # Run differential evolution
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
        
        # Build best-fit dictionary
        best_fit = {name: val for name, val in zip(param_names, result.x)}
        best_fit.update(self.fixed_params)
        
        # Calculate chi2 and ndof
        chi2 = result.fun
        n_free_params = len(param_names)
        ndof = self._n_data_points - n_free_params
        reduced_chi2 = chi2 / ndof if ndof > 0 else np.inf
        
        # Compute errors via finite differences if requested
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
        
        # Create result object
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
    
    def _get_de_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for differential evolution.
        
        For Gaussian priors, uses ±5 sigma as bounds.
        
        Returns
        -------
        bounds : dict
            Dictionary mapping param names to (min, max) tuples
        """
        param_names = self._get_free_param_names()
        bounds = {}
        
        for name in param_names:
            prior = self.priors[name]
            if prior.prior_type == 'uniform':
                bounds[name] = prior.bounds
            elif prior.prior_type == 'gaussian':
                # Use ±5 sigma as bounds for Gaussian priors
                bounds[name] = (prior.mean - 5*prior.std, prior.mean + 5*prior.std)
            else:
                # Fallback to wide bounds
                bounds[name] = (-100, 100)
        
        return bounds
    
    def _compute_errors_finite_diff(
        self,
        cost_func: Callable,
        x_best: np.ndarray,
        param_names: List[str],
        eps: float = 1e-4
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Compute parameter errors via finite difference Hessian.
        
        Parameters
        ----------
        cost_func : callable
            Cost function (returns chi2)
        x_best : array
            Best-fit parameter values
        param_names : list
            Parameter names
        eps : float
            Step size for finite differences
        
        Returns
        -------
        errors : dict
            Parameter errors (sqrt of diagonal of covariance)
        covariance : ndarray
            Covariance matrix
        """
        n_params = len(x_best)
        hessian = np.zeros((n_params, n_params))
        
        f0 = cost_func(x_best)
        
        # Compute Hessian via finite differences
        for i in range(n_params):
            for j in range(i, n_params):
                x_pp = x_best.copy()
                x_pm = x_best.copy()
                x_mp = x_best.copy()
                x_mm = x_best.copy()
                
                # Scale step size by parameter value
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
                
                # Second derivative
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj)
                hessian[j, i] = hessian[i, j]
        
        # Covariance is inverse of Hessian (for chi2, Hessian = 2 * Fisher)
        # For chi2 minimization: cov = 2 * H^{-1}
        try:
            covariance = 2.0 * np.linalg.inv(hessian)
            
            # Ensure positive diagonal (errors)
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

    def profile_likelihood(
        self,
        param_name: str,
        param_range: Optional[Tuple[float, float]] = None,
        n_points: int = 50,
        subtract_minimum: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute profile likelihood for a single parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter to profile
        param_range : tuple, optional
            (min, max) range for the parameter. If None, uses ±3 sigma from best-fit.
        n_points : int
            Number of points in the profile. Default: 50
        subtract_minimum : bool
            If True, return Δχ² = χ² - χ²_min. Default: True
        
        Returns
        -------
        param_values : np.ndarray
            Parameter values
        delta_chi2 : np.ndarray
            Δχ² values (or χ² if subtract_minimum=False)
        """
        if not HAS_IMINUIT:
            raise ImportError("iminuit is required")
        
        if self.minuit_result is None:
            print("Running minimization first...")
            self.minimize()
        
        m = self.minuit_result.minuit
        
        # Determine range
        if param_range is None:
            best_val = self.minuit_result.best_fit[param_name]
            err = self.minuit_result.errors.get(param_name, 0.1 * abs(best_val))
            param_range = (best_val - 3*err, best_val + 3*err)
        
        # Compute profile
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        chi2_values = []
        
        for val in param_values:
            # Fix the parameter and minimize others
            m.values[param_name] = val
            m.fixed[param_name] = True
            m.migrad()
            chi2_values.append(2 * m.fval)
            m.fixed[param_name] = False
        
        # Restore best-fit
        for name, val in self.minuit_result.best_fit.items():
            if name in m.parameters:
                m.values[name] = val
        
        chi2_values = np.array(chi2_values)
        
        if subtract_minimum:
            chi2_values -= chi2_values.min()
        
        return param_values, chi2_values
    
    # ========================================================================
    # Nautilus Sampling Methods (unchanged)
    # ========================================================================
    
    def run(
        self,
        n_live: int = 2000,
        n_eff: int = 10000,
        discard_exploration: bool = True,
        filepath: Optional[str] = None,
        seed: Optional[int] = None,
        pool: Optional[Any] = None,
        **nautilus_kwargs
    ) -> Dict:
        """Run the Nautilus sampler."""
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler is required")
        
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        self._initialize_halo_model()
        
        nautilus_prior = self._build_nautilus_prior()
        
        if self.verbose:
            print(f"\nFree parameters: {list(nautilus_prior.keys)}")
            print(f"Fixed parameters: {self.fixed_params}")
        
        self.sampler = Sampler(
            nautilus_prior,
            self.log_likelihood,
            n_live=n_live,
            filepath=filepath,
            seed=seed,
            pool=pool,
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
    
    def get_best_fit(self) -> Dict[str, float]:
        """Get best-fit parameters (from sampling or minimization)."""
        # Prefer DE result if available (most robust)
        if self.de_result is not None:
            return self.de_result.best_fit.copy()
        
        # Then minuit result
        if self.minuit_result is not None:
            return self.minuit_result.best_fit.copy()
        
        # Finally sampling result
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
    
    def save_results(self, filepath: str):
        """Save results to npz file."""
        save_dict = {}
        
        # Save sampling results if available
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
        
        # Save minuit results if available
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
        
        # Save differential evolution results if available
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


# ============================================================================
# Convenience Functions
# ============================================================================

def create_fitter_from_config(
    config_dict: Dict,
    data_dir: str,
    mass_bins: Optional[List[int]] = None
) -> CSMFFitter:
    """Create a CSMFFitter from a configuration dictionary."""
    fitter = CSMFFitter(
        cosmo_params=config_dict.get('cosmo_params'),
        observables=config_dict.get('observables', ['DeltaSigma']),
        rp_min=config_dict.get('rp_min'),
        rp_max=config_dict.get('rp_max'),
        rp_min_wgg=config_dict.get('rp_min_wgg'),
        rp_max_wgg=config_dict.get('rp_max_wgg'),
        verbose=config_dict.get('verbose', True)
    )
    
    fitter.load_data(data_dir, mass_bins)
    
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
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CSMF Fitter Module with iminuit Minimization")
    print("=" * 70)
    print()
    print("This version supports both:")
    print("  - Nautilus nested sampling (full posterior)")
    print("  - iminuit minimization (fast best-fit)")
    print()
    print("Example usage:")
    print("""
    from csmf_fitter_with_minuit import CSMFFitter
    
    # Create fitter
    fitter = CSMFFitter(
        observables=['DeltaSigma'],
        rp_min=0.1,
        rp_max=30.0,
    )
    
    # Load data
    fitter.load_data('/path/to/data/', mass_bins=[0, 1, 2, 3])
    
    # Set priors
    fitter.set_priors(
        fixed_params={'f_c': 1.0, 'f_s': 1.0}
    )
    
    # Option 1: Robust global optimization with differential evolution
    # (Recommended - handles -inf likelihood gracefully)
    result = fitter.minimize_de(
        maxiter=1000,
        popsize=15,
        workers=-1,  # Use all CPUs
    )
    result.print_summary()
    
    # Access results
    print(f"Best-fit M0: {result.best_fit['M0']:.3f}")
    print(f"Chi2/ndof: {result.reduced_chi2:.2f}")
    
    # Option 2: Fast local minimization with iminuit
    # (Use after DE to refine, or if you have a good starting point)
    result_minuit = fitter.minimize(
        start_params=fitter.de_result.best_fit,  # Start from DE result
        run_hesse=True,
    )
    
    # Option 3: Full posterior with Nautilus
    results = fitter.run(n_live=1000, n_eff=5000)
    summary = fitter.get_posterior_summary()
    
    # Compare best-fits
    print("\\nDE best-fit:", fitter.de_result.best_fit)
    print("Minuit best-fit:", fitter.minuit_result.best_fit)
    print("Nautilus MAP:", fitter.get_best_fit())
    
    # Profile likelihood
    param_vals, delta_chi2 = fitter.profile_likelihood('M0')
    
    # Save everything
    fitter.save_results('csmf_results.npz')
    """)