"""
CSMF HOD Fitter using Nautilus Sampler
=======================================

This module provides a class for fitting the Conditional Stellar Mass Function
(CSMF) HOD model to galaxy-galaxy lensing measurements (Delta Sigma) and
optionally galaxy clustering (WGG) using the Nautilus nested sampling algorithm.

Features:
- Support for fitting multiple stellar mass bins simultaneously
- Each mass bin has its own stellar mass range and effective redshift
- Uses the multi-redshift CSMF halo model functionality
- Flexible prior specification (uniform, Gaussian, fixed)
- Choice of observables to fit (DeltaSigma, WGG, or both)
- Support for different rp bins for WGG and DeltaSigma
- Efficient model evaluation using pre-computed halo model quantities

Author: CSMF Fitting Pipeline
Modified: December 2025 - Updated for new halo model API
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
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
    # Characteristic stellar mass log10(M*/[Msun/h²])
    # Table shows log(L0/[h^-2 L_sun]) with prior [7, 13]
    # Converting to stellar mass: roughly [9.0, 14.0] in log10(Msun/h²)
    'M0': ParameterPrior(
        name='M0',
        prior_type='uniform',
        bounds=(8.0, 13.0)
    ),
    
    # Characteristic halo mass log10(Mh/[Msun/h])
    # Table shows [9.0, 14.0]
    'M1': ParameterPrior(
        name='M1',
        prior_type='uniform',
        bounds=(9.0, 14.0)
    ),
    
    # Low-mass slope of SHMR
    'gamma1': ParameterPrior(
        name='gamma1',
        prior_type='uniform',
        bounds=(2.5, 10.0)
    ),
    
    # High-mass slope of SHMR
    # Table shows [0, 2]
    'gamma2': ParameterPrior(
        name='gamma2',
        prior_type='uniform',
        bounds=(0.0, 3.0)
    ),
    
    # Scatter in central stellar mass
    # Table shows [0.1, 0.3]
    'sigma_c': ParameterPrior(
        name='sigma_c',
        prior_type='uniform',
        bounds=(0.1, 0.9)
    ),
    
    # Satellite Schechter function slope
    # Table shows N(-1.1, 0.9)
    'alpha_s': ParameterPrior(
        name='alpha_s',
        prior_type='gaussian',
        mean=-1.1,
        std=1.5
    ),
    
    # Satellite normalization intercept
    # Table shows N(0, 1.5)
    'b0': ParameterPrior(
        name='b0',
        prior_type='gaussian',
        mean=0.0,
        std=1.5
    ),
    
    # Satellite normalization slope
    # Table shows N(1.5, 2.0)
    'b1': ParameterPrior(
        name='b1',
        prior_type='gaussian',
        mean=1.5,
        std=2.0
    ),
    
    # Central concentration parameter (Fc in table)
    # Table shows [0.1, 1] - this is f_conc
    # Note: Not in standard CSMF, but can be added as extension
    'f_c': ParameterPrior(
        name='f_c',
        prior_type='uniform',
        bounds=(0.4, 1.0)
    ),
    
    # Satellite concentration parameter (Fs)
    # Can be fixed to 1 or varied
    'f_s': ParameterPrior(
        name='f_s',
        prior_type='fixed',
        fixed_value=1.0
    ),
}

# CSMF HOD parameters
FIDUCIAL_CSMF_PARAMS = {
    'M0': 10.5,         # log10(M*) [Msun/h²]
    'M1': 12.0,         # log10(Mh) [Msun/h]
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
    
    Attributes
    ----------
    massbin_id : int
        Mass bin identifier (0-7)
    z_eff : float
        Effective redshift
    logmstar_min : float
        Minimum log stellar mass
    logmstar_max : float
        Maximum log stellar mass
    logmstar_median : float, optional
        Median log stellar mass for stellar point mass GGL contribution
    rp : np.ndarray
        Projected radii for Delta Sigma [Mpc/h]
    rp_bins : np.ndarray
        Bin edges for Delta Sigma [Mpc/h]
    delta_sigma : np.ndarray
        Delta Sigma measurements [Msun h/pc²]
    delta_sigma_err : np.ndarray
        Delta Sigma errors
    cov_delta_sigma : np.ndarray
        Covariance matrix for Delta Sigma
    rp_wgg : np.ndarray, optional
        Projected radii for WGG [Mpc/h] (defaults to rp if not provided)
    rp_bins_wgg : np.ndarray, optional
        Bin edges for WGG [Mpc/h] (defaults to rp_bins if not provided)
    wp : np.ndarray, optional
        Projected clustering w_p(r_p)
    wp_err : np.ndarray, optional
        Errors on w_p
    cov_wgg : np.ndarray, optional
        Covariance matrix for w_p
    mag_contribution : np.ndarray, optional
        Magnification bias contribution
    mean_sigma_crit : np.ndarray, optional
        Mean critical surface density
    """
    massbin_id: int
    z_eff: float
    logmstar_min: float
    logmstar_max: float
    logmstar_median: Optional[float] = None
    rp: np.ndarray
    rp_bins: np.ndarray
    delta_sigma: np.ndarray
    delta_sigma_err: np.ndarray
    cov_delta_sigma: np.ndarray
    rp_wgg: Optional[np.ndarray] = None
    rp_bins_wgg: Optional[np.ndarray] = None
    wp: Optional[np.ndarray] = None
    wp_err: Optional[np.ndarray] = None
    cov_wgg: Optional[np.ndarray] = None
    mag_contribution: Optional[np.ndarray] = None
    mean_sigma_crit: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Set default values for WGG rp if not provided."""
        # By default, use same rp bins for WGG as for Delta Sigma
        if self.rp_wgg is None:
            self.rp_wgg = self.rp.copy()
        if self.rp_bins_wgg is None:
            self.rp_bins_wgg = self.rp_bins.copy()
    
    @classmethod
    def from_npz(cls, filepath: str, massbin_id: int, h: float) -> 'MassBinData':
        """
        Load mass bin data from an npz file.
        
        Parameters
        ----------
        filepath : str
            Path to npz file
        massbin_id : int
            Mass bin identifier
        h : float
            Hubble parameter (for unit conversions if needed)
        
        Returns
        -------
        MassBinData
            Loaded data container
        """
        data = np.load(filepath)

        # Handle different possible key names for rp
        if 'rp_delta_sigma' in data:
            rp = data['rp_delta_sigma']
        elif 'rp' in data:
            rp = data['rp']
        else:
            raise KeyError("Could not find 'rp' or 'rp_delta_sigma' in data file")

        # If rp_wgg is not provided, use rp_deltasigma
        rp_wgg = data.get('rp_wgg', rp)
        rp_bins_wgg = data.get('rp_bins_wgg', data['rp_bins'])

        # Handle covariance matrix key names
        if 'cov_delta_sigma' in data:
            cov_delta_sigma = data['cov_delta_sigma']
        elif 'covariance_matrix' in data:
            cov_delta_sigma = data['covariance_matrix']
        else:
            raise KeyError("Could not find covariance matrix in data file")

        return cls(
            massbin_id=massbin_id,
            z_eff=float(data['z_eff']),
            logmstar_min=float(data['logmstar_min']),
            logmstar_max=float(data['logmstar_max']),
            logmstar_median=float(data['logmstar_median']) if 'logmstar_median' in data else None,
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
# Main CSMF Fitter Class
# ============================================================================

class CSMFFitter:
    """
    CSMF HOD Fitter using Nautilus nested sampling.
    
    This class handles fitting the CSMF HOD model to galaxy-galaxy lensing
    (Delta Sigma) and optionally galaxy clustering (w_p) measurements.
    
    IMPORTANT: This version uses the multi-redshift CSMF functionality where
    a single MultiRedshiftHaloModel is created with all redshifts, and
    stellar mass bin arrays are passed for multi-bin computation.
    
    The rp bins for WGG and DeltaSigma can be the same or different.
    
    Parameters
    ----------
    cosmo_params : dict, optional
        Cosmology parameters. Default uses the fiducial cosmology.
    observables : list of str
        Which observables to fit. Options: ['DeltaSigma', 'WGG']
        Default: ['DeltaSigma']
    rp_min : float, optional
        Minimum r_p to include in fit [Mpc/h]. Default: None (use all)
    rp_max : float, optional
        Maximum r_p to include in fit [Mpc/h]. Default: None (use all)
    rp_min_wgg : float, optional
        Minimum r_p for WGG fit [Mpc/h]. Default: None (use rp_min)
    rp_max_wgg : float, optional
        Maximum r_p for WGG fit [Mpc/h]. Default: None (use rp_max)
    units_per_h : bool
        Whether to use h-units. Default: True
    verbose : bool
        Print progress information. Default: True
    
    Attributes
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object
    mass_bins : list of MassBinData
        Data for each mass bin
    priors : dict
        Prior configuration for each parameter
    fixed_params : dict
        Fixed parameter values
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
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler is required. Install with: pip install nautilus-sampler")
        
        # Store configuration
        self.observables = observables or ['DeltaSigma']
        self.rp_min = rp_min
        self.rp_max = rp_max
        # WGG scale cuts default to same as DeltaSigma if not specified
        self.rp_min_wgg = rp_min_wgg if rp_min_wgg is not None else rp_min
        self.rp_max_wgg = rp_max_wgg if rp_max_wgg is not None else rp_max
        self.units_per_h = units_per_h
        self.verbose = verbose
        
        # Set up cosmology
        if cosmo_params is None:
            cosmo_params = DEFAULT_COSMO_PARAMS
        self.cosmo_params = cosmo_params
        self._setup_cosmology()
        
        # Initialize data and prior storage
        self.mass_bins: List[MassBinData] = []
        self.priors: Dict[str, ParameterPrior] = {}
        self.fixed_params: Dict[str, float] = {}
        
        # Single unified halo model
        self._halo_model: Optional[MultiRedshiftHaloModel] = None
        
        # Store arrays for multi-bin computation
        self._z_eff_array: Optional[np.ndarray] = None
        self._mstar_min_array: Optional[np.ndarray] = None
        self._mstar_max_array: Optional[np.ndarray] = None
        self._massbin_id_to_index: Optional[Dict[int, int]] = None
        
        # Results
        self.sampler: Optional[Sampler] = None
        self.results: Optional[Dict] = None
    
    def _setup_cosmology(self):
        """Initialize CCL cosmology object"""
        Omc = self.cosmo_params['Omc']
        Omb = self.cosmo_params['Omb']
        h = self.cosmo_params['h']
        As = self.cosmo_params['A_s']
        ns = self.cosmo_params['n_s']
        mnu = 0.06  # Default neutrino mass
        
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
        file_pattern: str = 'dsigma_massbin{}.npz'
    ):
        """
        Load Delta Sigma measurements for specified mass bins.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the data files
        mass_bins : list of int, optional
            Which mass bins to load. Default: [0, 1, 2, 3, 4, 5, 6, 7]
        file_pattern : str
            Filename pattern with {} placeholder for mass bin number
        """
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
                # Report if WGG uses different rp bins
                if data.wp is not None:
                    if not np.allclose(data.rp, data.rp_wgg):
                        print(f"  Note: WGG uses different rp bins than DeltaSigma")

        # Sort by z_eff for consistency
        self.mass_bins.sort(key=lambda x: x.z_eff)

        if self.verbose:
            print(f"Total: {len(self.mass_bins)} mass bins loaded")
            if self.mass_bins:
                print(f"Redshift range: z=[{self.mass_bins[0].z_eff:.3f}, {self.mass_bins[-1].z_eff:.3f}]")
    
    def set_priors(
        self,
        priors: Optional[Dict[str, ParameterPrior]] = None,
        fixed_params: Optional[Dict[str, float]] = None
    ):
        """
        Set parameter priors.
        
        Parameters
        ----------
        priors : dict, optional
            Dictionary mapping parameter names to ParameterPrior objects.
            If None, uses default CSMF priors.
        fixed_params : dict, optional
            Dictionary of parameters to fix at specific values.
            These override any priors set for the same parameters.
        
        Examples
        --------
        >>> fitter.set_priors(
        ...     fixed_params={'f_c': 1.0, 'f_s': 1.0}  # Fix concentration params
        ... )
        
        >>> # Custom prior for gamma1
        >>> custom_priors = {
        ...     'gamma1': ParameterPrior('gamma1', 'gaussian', mean=3.5, std=1.0)
        ... }
        >>> fitter.set_priors(priors=custom_priors)
        """
        # Start with defaults
        self.priors = DEFAULT_CSMF_PRIORS.copy()
        
        # Update with user-provided priors
        if priors is not None:
            for name, prior in priors.items():
                self.priors[name] = prior
        
        # Handle fixed parameters
        self.fixed_params = {}
        if fixed_params is not None:
            for name, value in fixed_params.items():
                self.fixed_params[name] = value
                # Update the prior to be fixed
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
    
    def _build_nautilus_prior(self) -> Prior:
        """
        Build nautilus Prior object from parameter configuration.
        
        Returns
        -------
        prior : nautilus.Prior
            Configured prior object
        """
        prior = Prior()
        
        # Core CSMF parameters
        csmf_param_names = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']
        
        # Add optional parameters if not fixed
        optional_params = ['f_c', 'f_s']
        
        for name in csmf_param_names + optional_params:
            if name not in self.priors:
                continue
            
            param_prior = self.priors[name]
            
            if param_prior.prior_type == 'fixed':
                # Don't add fixed parameters to nautilus prior
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
        """Get list of free (non-fixed) parameter names"""
        return [
            name for name, prior in self.priors.items()
            if prior.prior_type != 'fixed'
        ]
    
    def _build_full_params(self, free_params: Dict[str, float]) -> Dict[str, float]:
        """
        Build full parameter dictionary including fixed params.
        
        Parameters
        ----------
        free_params : dict
            Dictionary of free parameter values from sampler
        
        Returns
        -------
        full_params : dict
            Complete parameter dictionary
        """
        full_params = {}
        
        # Add fixed parameters
        for name, prior in self.priors.items():
            if prior.prior_type == 'fixed':
                full_params[name] = prior.fixed_value
        
        # Add free parameters
        full_params.update(free_params)
        
        return full_params
    
    def _initialize_halo_model(self):
        """
        Initialize a SINGLE unified halo model for all mass bins.
        
        The model is created with z_array matching the z_eff of each mass bin.
        This allows passing Mstar_min/Mstar_max arrays where each element
        corresponds to a different stellar mass bin at its respective redshift.
        """
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Extract z_eff from all mass bins (in order)
        # This array will be used as z_array for the model
        self._z_eff_array = np.array([mb.z_eff for mb in self.mass_bins])
        
        if self.verbose:
            print(f"\nInitializing unified halo model...")
            print(f"  Number of mass bins: {len(self.mass_bins)}")
            print(f"  Redshift array: {self._z_eff_array}")
        
        # Create arrays for stellar mass bin edges
        self._mstar_min_array = np.array([10**mb.logmstar_min for mb in self.mass_bins])
        self._mstar_max_array = np.array([10**mb.logmstar_max for mb in self.mass_bins])

        # Extract median stellar masses for stellar point mass contribution
        # Store as array matching z_eff_array length
        median_mstar_list = []
        for mb in self.mass_bins:
            if mb.logmstar_median is not None:
                # Convert from log10 to linear [Msun/h²]
                median_mstar_list.append(10**mb.logmstar_median)
            else:
                # Use None to indicate no stellar component for this bin
                median_mstar_list.append(None)

        # Convert to array, handling None values
        # If ALL are None, set to None (no stellar component)
        # If SOME are None, raise error (not supported by model)
        if all(m is None for m in median_mstar_list):
            self._median_mstar_array = None
        elif any(m is None for m in median_mstar_list):
            raise ValueError(
                "Mixed stellar mass bins detected: some bins have logmstar_median, "
                "some don't. All bins must either have or not have median stellar masses."
            )
        else:
            self._median_mstar_array = np.array(median_mstar_list)

        # Create mapping from massbin_id to index in arrays
        self._massbin_id_to_index = {
            mb.massbin_id: i for i, mb in enumerate(self.mass_bins)
        }
        
        # Use fiducial parameters for initialization
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
        
        # Create the unified model with z_array = z_eff for each mass bin
        # This allows Mstar_min/Mstar_max arrays of same length
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
        
        if self.verbose:
            print(f"\nStellar mass bins configuration:")
            for i, mb in enumerate(self.mass_bins):
                median_str = ""
                if self._median_mstar_array is not None:
                    median_str = f", M*_median={self._median_mstar_array[i]:.2e} Msun/h²"
                print(f"  Bin {i} (ID={mb.massbin_id}): z_eff={mb.z_eff:.3f}, "
                      f"log(M*)=[{mb.logmstar_min:.2f}, {mb.logmstar_max:.2f}], "
                      f"M*=[{self._mstar_min_array[i]:.2e}, {self._mstar_max_array[i]:.2e}]"
                      f"{median_str}")
            if self._median_mstar_array is not None:
                print(f"\n  Stellar point mass component ENABLED for GGL")
            print(f"\nUnified halo model initialized successfully!")
    
    def _compute_model_observables(
        self,
        params: Dict[str, float],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Optional[np.ndarray]]]:
        """
        Compute model observables for ALL mass bins simultaneously.

        This uses the multi-bin functionality where Mstar_min/Mstar_max arrays
        are passed with length matching z_array. The model computes all bins
        in a single vectorized call.

        Parameters
        ----------
        params : dict
            HOD parameters

        Returns
        -------
        ds_dict : dict
            Dictionary mapping massbin_id to Delta Sigma predictions
        wgg_dict : dict
            Dictionary mapping massbin_id to w_p predictions (or None)
        """
        # Extract CSMF parameters
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

        # Update halo model with new parameters
        self._halo_model.update_hod_params(csmf_params)

        # Use the first mass bin's rp values as reference
        # (assumes all bins use the same rp grid - this is the common case)
        rp_ds = self.mass_bins[0].rp
        rp_bins_ds = self.mass_bins[0].rp_bins
        
        # Compute Delta Sigma for ALL redshifts/mass bins in one call
        # Pass Mstar arrays that match z_array length
        _, ds_all = self._halo_model.compute_DeltaSigma_all_z(
            rp=rp_ds,
            bin_avg=True,
            rp_bins=rp_bins_ds,
            Mstar_min=self._mstar_min_array,
            Mstar_max=self._mstar_max_array,
            verbose=False
        )

        # Build dictionary mapping massbin_id to predictions
        # ds_all has shape (n_z, n_rp) where n_z matches number of mass bins
        ds_dict = {}
        for i, mass_bin in enumerate(self.mass_bins):
            if self._halo_model.is_single_z:
                ds_dict[mass_bin.massbin_id] = ds_all
            else:
                ds_dict[mass_bin.massbin_id] = ds_all[i]

        # Compute WGG if requested
        wgg_dict = {}
        if 'WGG' in self.observables:
            # Check if any mass bin has WGG data
            has_wgg = any(mb.wp is not None for mb in self.mass_bins)
            
            if has_wgg:
                # Use the first mass bin's rp_wgg values as reference
                rp_wgg = self.mass_bins[0].rp_wgg
                rp_bins_wgg = self.mass_bins[0].rp_bins_wgg
                
                # Compute WGG for ALL redshifts/mass bins in one call
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
        """
        Apply r_p scale cuts to Delta Sigma data.
        
        Returns
        -------
        rp_cut : np.ndarray
            r_p values after cuts
        ds_cut : np.ndarray
            Delta Sigma after cuts
        cov_cut : np.ndarray
            Covariance matrix after cuts
        mask : np.ndarray
            Boolean mask
        """
        rp = mass_bin.rp
        ds = mass_bin.delta_sigma
        cov = mass_bin.cov_delta_sigma
        
        # Create mask
        mask = np.ones(len(rp), dtype=bool)
        
        if self.rp_min is not None:
            mask &= (rp >= self.rp_min)
        if self.rp_max is not None:
            mask &= (rp <= self.rp_max)
        
        # Apply mask
        rp_cut = rp[mask]
        ds_cut = ds[mask]
        cov_cut = cov[np.ix_(mask, mask)]
        
        return rp_cut, ds_cut, cov_cut, mask
    
    def _apply_scale_cuts_wgg(self, mass_bin: MassBinData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply r_p scale cuts to WGG data.
        
        Note: Uses separate rp_min_wgg and rp_max_wgg if specified.
        
        Returns
        -------
        rp_cut : np.ndarray
            r_p values after cuts
        wp_cut : np.ndarray
            w_p after cuts
        cov_cut : np.ndarray
            Covariance matrix after cuts
        mask : np.ndarray
            Boolean mask
        """
        rp = mass_bin.rp_wgg
        wp = mass_bin.wp
        cov = mass_bin.cov_wgg
        
        # Create mask using WGG-specific cuts
        mask = np.ones(len(rp), dtype=bool)
        
        if self.rp_min_wgg is not None:
            mask &= (rp >= self.rp_min_wgg)
        if self.rp_max_wgg is not None:
            mask &= (rp <= self.rp_max_wgg)
        
        # Apply mask
        rp_cut = rp[mask]
        wp_cut = wp[mask]
        cov_cut = cov[np.ix_(mask, mask)]
        
        return rp_cut, wp_cut, cov_cut, mask
    
    def log_likelihood(self, param_dict: Dict[str, float]) -> float:
        """
        Compute log-likelihood for the CSMF model.

        This version computes the model predictions for ALL mass bins
        directly at the data points (no interpolation needed).

        Parameters
        ----------
        param_dict : dict
            Dictionary of free parameter values

        Returns
        -------
        log_L : float
            Log-likelihood value
        """
        # Build full parameter set
        params = self._build_full_params(param_dict)

        try:
            # Compute model predictions for ALL bins
            ds_dict, wgg_dict = self._compute_model_observables(params)

        except Exception as e:
            # Return very low likelihood for failed evaluations
            if self.verbose:
                print(f"Warning: Model evaluation failed: {e}")
            return -1e100

        total_log_L = 0.0

        # Loop over mass bins to evaluate likelihood
        for mass_bin in self.mass_bins:
            # Delta Sigma likelihood
            if 'DeltaSigma' in self.observables:
                # Apply scale cuts
                rp_cut, ds_data, cov, mask = self._apply_scale_cuts(mass_bin)

                if len(rp_cut) == 0:
                    continue

                try:
                    # Get model prediction for this bin
                    ds_model_full = ds_dict[mass_bin.massbin_id]

                    # Apply the same scale cuts to model
                    ds_model = ds_model_full[mask]

                    if not np.all(np.isfinite(ds_model)):
                        if self.verbose:
                            print(f"[WARNING] NaN DeltaSigma in bin {mass_bin.massbin_id} → -inf")
                        return -1e100

                    # Compute chi-squared
                    residual = ds_data - ds_model

                    # Use covariance matrix
                    try:
                        cov_inv = np.linalg.inv(cov)
                        chi2 = residual @ cov_inv @ residual
                    except np.linalg.LinAlgError:
                        # Fall back to diagonal
                        err = mass_bin.delta_sigma_err[mask]
                        chi2 = np.sum((residual / err)**2)

                    # Log-likelihood
                    log_L_bin = -0.5 * chi2
                    total_log_L += log_L_bin

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: DeltaSigma likelihood failed for mass bin "
                              f"{mass_bin.massbin_id}: {e}")
                    return -1e100

            # WGG likelihood
            if 'WGG' in self.observables and mass_bin.wp is not None:
                # Apply scale cuts for WGG
                rp_wgg_cut, wp_data, cov_wgg, mask_wgg = self._apply_scale_cuts_wgg(mass_bin)

                if len(rp_wgg_cut) == 0:
                    continue

                try:
                    # Get model prediction for this bin
                    wgg_model_full = wgg_dict[mass_bin.massbin_id]

                    if wgg_model_full is None:
                        continue

                    # Apply the same scale cuts to model
                    wgg_model = wgg_model_full[mask_wgg]

                    if not np.all(np.isfinite(wgg_model)):
                        if self.verbose:
                            print(f"[WARNING] NaN w_p in bin {mass_bin.massbin_id} → -inf")
                        return -1e100

                    # Compute chi-squared
                    residual = wp_data - wgg_model

                    # Use covariance matrix
                    try:
                        cov_inv = np.linalg.inv(cov_wgg)
                        chi2 = residual @ cov_inv @ residual
                    except np.linalg.LinAlgError:
                        # Fall back to diagonal
                        err = mass_bin.wp_err[mask_wgg]
                        chi2 = np.sum((residual / err)**2)

                    # Log-likelihood
                    log_L_bin = -0.5 * chi2
                    total_log_L += log_L_bin

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: WGG likelihood failed for mass bin "
                              f"{mass_bin.massbin_id}: {e}")
                    return -1e100

        return total_log_L
    
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
        """
        Run the Nautilus sampler.
        
        Parameters
        ----------
        n_live : int
            Number of live points. Default: 2000
        n_eff : int
            Target effective sample size. Default: 10000
        discard_exploration : bool
            Whether to discard exploration phase. Default: True
        filepath : str, optional
            Path to save checkpoint file
        seed : int, optional
            Random seed for reproducibility
        pool : optional
            Pool for parallelization (int for multiprocessing, or pool object)
        **nautilus_kwargs
            Additional arguments passed to Sampler
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'points': posterior samples
            - 'log_w': log weights
            - 'log_l': log likelihoods
            - 'log_z': log evidence
            - 'param_names': parameter names
        """
        if not self.mass_bins:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.priors:
            print("No priors set. Using defaults.")
            self.set_priors()
        
        # Initialize unified halo model
        self._initialize_halo_model()
        
        # Build nautilus prior
        nautilus_prior = self._build_nautilus_prior()
        
        if self.verbose:
            print(f"\nFree parameters: {list(nautilus_prior.keys)}")
            print(f"Fixed parameters: {self.fixed_params}")
            print(f"Number of mass bins: {len(self.mass_bins)}")
            print(f"Observables: {self.observables}")
            print(f"\n{'='*70}")
            print("USING MULTI-REDSHIFT CSMF FUNCTIONALITY")
            print(f"Single unified model with {len(self._halo_model.z_array)} redshift(s)")
            print(f"Each redshift corresponds to a stellar mass bin")
            print(f"{'='*70}\n")
        
        # Create sampler
        self.sampler = Sampler(
            nautilus_prior,
            self.log_likelihood,
            n_live=n_live,
            filepath=filepath,
            seed=seed,
            pool=pool,
            **nautilus_kwargs
        )
        
        # Run sampler
        if self.verbose:
            print("\nStarting Nautilus sampler...")
        
        self.sampler.run(
            n_eff=n_eff,
            discard_exploration=discard_exploration,
            verbose=self.verbose
        )
        
        # Extract results
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
            print(f"Effective sample size: {len(points)}")
        
        return self.results
    
    def get_best_fit(self) -> Dict[str, float]:
        """
        Get best-fit (maximum likelihood) parameters.
        
        Returns
        -------
        best_fit : dict
            Best-fit parameter values
        """
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        # Find maximum likelihood point
        idx_best = np.argmax(self.results['log_l'])
        best_point = self.results['points'][idx_best]
        
        # Build dictionary
        best_fit = dict(zip(self.results['param_names'], best_point))
        
        # Add fixed parameters
        best_fit.update(self.results['fixed_params'])
        
        return best_fit
    
    def get_posterior_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get posterior summary statistics.
        
        Returns
        -------
        summary : dict
            For each parameter: median, mean, std, 16th, 84th percentiles
        """
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        points = self.results['points']
        weights = np.exp(self.results['log_w'])
        weights /= weights.sum()
        
        summary = {}
        
        for i, name in enumerate(self.results['param_names']):
            values = points[:, i]
            
            # Weighted statistics
            mean = np.average(values, weights=weights)
            var = np.average((values - mean)**2, weights=weights)
            std = np.sqrt(var)
            
            # Weighted percentiles
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
        """
        Save results to npz file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        # Convert param_names to array-friendly format
        save_dict = {
            'points': self.results['points'],
            'log_w': self.results['log_w'],
            'log_l': self.results['log_l'],
            'log_z': self.results['log_z'],
            'param_names': np.array(self.results['param_names'], dtype=str),
        }
        
        # Add fixed params
        for name, value in self.results['fixed_params'].items():
            save_dict[f'fixed_{name}'] = value
        
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
    """
    Create a CSMFFitter from a configuration dictionary.
    
    Parameters
    ----------
    config_dict : dict
        Configuration with keys:
        - 'cosmo_params': cosmology parameters (optional)
        - 'observables': list of observables to fit
        - 'rp_min', 'rp_max': scale cuts for DeltaSigma
        - 'rp_min_wgg', 'rp_max_wgg': scale cuts for WGG (optional)
        - 'fixed_params': dict of fixed parameters
        - 'priors': dict of prior configurations
    data_dir : str
        Directory containing data files
    mass_bins : list, optional
        Mass bins to load
    
    Returns
    -------
    fitter : CSMFFitter
        Configured fitter object
    """
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
    
    # Set up priors
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
    print("CSMF Fitter Module (Updated for new halo model API)")
    print("=" * 70)
    print()
    print("This version uses the multi-redshift CSMF halo model functionality!")
    print()
    print("Key features:")
    print("  - Support for different rp bins for WGG and DeltaSigma")
    print("  - Separate scale cuts for WGG (rp_min_wgg, rp_max_wgg)")
    print("  - Single unified halo model for all mass bins")
    print()
    print("Example usage:")
    print("""
    from sampler_updated import CSMFFitter, ParameterPrior
    
    # Create fitter with potentially different scale cuts for WGG
    fitter = CSMFFitter(
        observables=['DeltaSigma', 'WGG'],
        rp_min=0.1,      # Mpc/h for DeltaSigma
        rp_max=30.0,     # Mpc/h for DeltaSigma
        rp_min_wgg=0.5,  # Optional: different min for WGG
        rp_max_wgg=50.0, # Optional: different max for WGG
    )
    
    # Load data for all 8 mass bins
    fitter.load_data('/path/to/data/', mass_bins=[0, 1, 2, 3, 4, 5, 6, 7])
    
    # Set priors with some fixed parameters
    fitter.set_priors(
        fixed_params={
            'f_c': 1.0,  # Fix central concentration
            'f_s': 1.0,  # Fix satellite concentration
        }
    )
    
    # Run sampler
    results = fitter.run(
        n_live=2000,
        n_eff=10000,
        filepath='csmf_chain.hdf5',
        pool=4,  # Use 4 CPU cores
    )
    
    # Get results
    best_fit = fitter.get_best_fit()
    summary = fitter.get_posterior_summary()
    
    # Print summary
    for param, stats in summary.items():
        print(f"{param}: {stats['median']:.3f} +{stats['err_high']:.3f} -{stats['err_low']:.3f}")
    
    # Save results
    fitter.save_results('csmf_results.npz')
    """)