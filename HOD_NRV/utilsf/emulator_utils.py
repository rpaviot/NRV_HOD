"""
Utilities for HOD parameter emulation with fixed number density.

This module provides tools for generating HOD parameter grids using Latin Hypercube
Sampling (LHS) while maintaining a fixed galaxy number density by rescaling the
central amplitude parameter (Ac).

Key Concept
-----------
When central and satellite occupations are independent, the clustering/lensing
signals are governed by the ratio Ac/As. By fixing the number density, we can:
1. Assume a fiducial value for Ac
2. Compute ngal for given HOD parameters
3. Rescale Ac to achieve the target ngal: Ac_new = Ac_fid * (ngal_target / ngal_fid)

This approach reduces the dimensionality of the parameter space by one, making
Ac a derived parameter rather than a free parameter.

References
----------
.. [1] Rocher et al. (2023) - HOD emulation with fixed number density
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Set
from scipy.stats import qmc
import warnings


# Default parameter ranges for extensions
DEFAULT_PARAM_RANGES = {
    # Assembly bias parameters (typically -0.5 to 0.5)
    'A_cent': (-0.5, 0.5),
    'B_cent': (-0.5, 0.5),
    'A_sat': (-0.5, 0.5),
    'B_sat': (-0.5, 0.5),
    # Conformity parameter: M1_EE = kappa_EE * M1
    # kappa_EE > 1 means satellites prefer halos with centrals
    # kappa_EE < 1 means satellites avoid halos with centrals
    'kappa_EE': (0.5, 2.0),
    # Extended NFW profile parameters
    'f_exp': (0.0, 0.5),        # Exponential component fraction
    'tau': (2.0, 10.0),         # Decay scale in units of Rs
    'lambda_NFW': (0.5, 2.0),   # NFW rescaling factor
}


def compute_ngal_with_fiducial_Ac(
    hod_model,
    params: Dict[str, float],
    Ac_fiducial: float = 1.0
) -> float:
    """
    Compute galaxy number density with a fiducial Ac value.

    This function calculates the total galaxy number density (centrals + satellites)
    for a given set of HOD parameters with a specified fiducial central amplitude.

    Parameters
    ----------
    hod_model : HOD_models.Occupation
        HOD model instance with cosmology and mass function set
    params : dict
        HOD parameters (excluding Ac). Required keys depend on the model:
        - All models: 'Mmin', 'sig_M', 'As', 'M1', 'alpha', 'kappa'
        - ELG_SFR also needs: 'gamma'
    Ac_fiducial : float, default=1.0
        Fiducial value for the central amplitude parameter

    Returns
    -------
    ngal : float
        Galaxy number density [(Mpc/h)^-3] computed with the fiducial Ac

    Examples
    --------
    >>> from HOD_NRV.HOD_catalogue import HaloOccupation
    >>> from HOD_NRV.emulator_utils import compute_ngal_with_fiducial_Ac
    >>>
    >>> halo = HaloOccupation(cosmology=cosmo, zeff=1.0, Lbox=1000, ...)
    >>> halo.set_halo_model("ELG_GHOD")
    >>>
    >>> params = {'Mmin': 12.0, 'sig_M': 0.4, 'As': 0.5,
    ...           'M1': 13.5, 'alpha': 1.0, 'kappa': 1.0}
    >>> ngal = compute_ngal_with_fiducial_Ac(halo.HOD, params, Ac_fiducial=0.5)
    >>> print(f"Number density: {ngal:.2e} (Mpc/h)^-3")

    Notes
    -----
    The number density is computed by integrating the HOD over the halo mass function:

    .. math::
        n_{gal} = \\int [N_{cen}(M) + N_{sat}(M)] \\frac{dn}{dM} dM

    where the central occupation includes the fiducial Ac parameter.
    """
    # Create a copy of params and add Ac
    params_with_Ac = params.copy()
    params_with_Ac['Ac'] = Ac_fiducial

    # Compute number density using the HOD model's method
    ngal = hod_model.compute_ngal(params_with_Ac)

    return ngal


def rescale_Ac_to_target_ngal(
    hod_model,
    params: Dict[str, float],
    target_ngal: float,
    Ac_fiducial: float = 1.0
) -> Tuple[float, float]:
    """
    Rescale Ac AND As to achieve a target galaxy number density.

    This function computes the rescaling factor needed to match the target
    number density while preserving the ratio Ac/As (which governs clustering).

    Parameters
    ----------
    hod_model : HOD_models.Occupation
        HOD model instance with cosmology and mass function set
    params : dict
        HOD parameters including 'As' but excluding 'Ac'
    target_ngal : float
        Target galaxy number density [(Mpc/h)^-3]
    Ac_fiducial : float, default=1.0
        Initial fiducial value for Ac

    Returns
    -------
    Ac_rescaled : float
        Rescaled central amplitude
    As_rescaled : float
        Rescaled satellite amplitude (preserving Ac/As ratio)

    Raises
    ------
    ValueError
        If ngal cannot be computed (e.g., invalid parameters)

    Examples
    --------
    >>> Ac_new, As_new = rescale_Ac_to_target_ngal(
    ...     halo.HOD, params, target_ngal=1e-3, Ac_fiducial=1.0
    ... )
    >>> # Both Ac and As are rescaled by the same factor
    >>> print(f"Rescaled Ac: {Ac_new:.4f}, As: {As_new:.4f}")

    Notes
    -----
    The rescaling formula is:

    .. math::
        ratio = \\frac{n_{gal}^{target}}{n_{gal}^{fiducial}}

        A_c^{new} = A_c^{fid} \\times ratio

        A_s^{new} = A_s^{old} \\times ratio

    This preserves the ratio Ac/As which determines the clustering signal.
    """
    # Compute ngal with fiducial Ac
    ngal_fiducial = compute_ngal_with_fiducial_Ac(hod_model, params, Ac_fiducial)

    # Compute rescaling factor to maintain Ac/As ratio
    rescale_factor = target_ngal / ngal_fiducial

    # Rescale both Ac and As by the same factor to preserve Ac/As ratio
    Ac_rescaled = Ac_fiducial * rescale_factor
    As_rescaled = params['As'] * rescale_factor

    return Ac_rescaled, As_rescaled

def create_latin_hypercube(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate Latin Hypercube samples for HOD parameters.

    Latin Hypercube Sampling (LHS) provides better coverage of parameter space
    than random sampling, ensuring that each parameter range is evenly sampled.

    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to (min, max) tuples.
        Example: {'Mmin': (11.5, 12.5), 'sig_M': (0.2, 0.6), ...}
    n_samples : int
        Number of parameter combinations to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    samples : pd.DataFrame
        DataFrame with shape (n_samples, n_params) containing the LHS samples.
        Columns are parameter names, rows are different parameter combinations.

    Examples
    --------
    >>> param_ranges = {
    ...     'Mmin': (11.5, 12.5),
    ...     'sig_M': (0.2, 0.6),
    ...     'As': (0.1, 1.0),
    ...     'M1': (12.5, 14.0),
    ...     'alpha': (0.5, 1.5),
    ...     'kappa': (0.5, 1.5)
    ... }
    >>> samples = create_latin_hypercube(param_ranges, n_samples=100, random_seed=42)
    >>> print(samples.head())

    Notes
    -----
    This function uses scipy.stats.qmc.LatinHypercube for generating the samples.
    The samples are uniformly distributed within the specified ranges and provide
    better space-filling properties than random sampling.

    References
    ----------
    .. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (1979).
           Technometrics, 21(2), 239-245.
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=random_seed)

    # Generate samples in [0, 1]^n_params
    samples_unit = sampler.random(n=n_samples)

    # Scale to parameter ranges
    samples_scaled = np.zeros_like(samples_unit)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples_scaled[:, i] = min_val + (max_val - min_val) * samples_unit[:, i]

    # Create DataFrame
    df = pd.DataFrame(samples_scaled, columns=param_names)

    return df


def _get_required_parameters(
    halo,
    hod_type: str,
    conformity: bool,
    include_nfw_extensions: bool = False
) -> Set[str]:
    """
    Get the complete set of required HOD parameters based on model and extensions.

    Parameters
    ----------
    halo : HaloOccupation
        HaloOccupation instance to inspect for enabled extensions
    hod_type : str
        HOD model type ('LRG', 'ELG_GHOD', 'ELG_SFR')
    conformity : bool
        Whether conformity is enabled
    include_nfw_extensions : bool, default=False
        Whether to include NFW extension parameters

    Returns
    -------
    required_params : set
        Set of required parameter names (excluding Ac which is computed)
    """
    # Base parameters for all models
    required = {'Mmin', 'sig_M', 'As', 'M1', 'alpha', 'kappa'}

    # Add gamma for ELG_SFR
    if hod_type == 'ELG_SFR':
        required.add('gamma')

    # Add conformity parameter
    if conformity:
        required.add('kappa_EE')

    # Add assembly bias parameters if enabled in halo instance
    if hasattr(halo, 'assembly_bias') and halo.assembly_bias:
        required.update(['A_cent', 'B_cent', 'A_sat', 'B_sat'])

    # Add NFW extension parameters if requested
    if include_nfw_extensions:
        required.update(['f_exp', 'tau', 'lambda_NFW'])

    return required


def _validate_parameters(
    param_ranges: Dict[str, Tuple[float, float]],
    fixed_params: Optional[Dict[str, float]],
    required_params: Set[str],
    hod_type: str,
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that all required parameters are provided.

    Parameters
    ----------
    param_ranges : dict
        Parameter ranges for sampling
    fixed_params : dict or None
        Fixed parameter values
    required_params : set
        Required parameter names
    hod_type : str
        HOD model type for error messages
    verbose : bool
        Whether to print warnings

    Returns
    -------
    is_valid : bool
        True if all required parameters are covered
    missing_params : list
        List of missing parameter names
    """
    # Combine parameter sources
    provided_params = set(param_ranges.keys())
    if fixed_params is not None:
        provided_params.update(fixed_params.keys())

    # Find missing parameters
    missing_params = required_params - provided_params

    if missing_params and verbose:
        warnings.warn(
            f"\nMissing required parameters for {hod_type}: {missing_params}\n"
            f"These parameters should be in either 'param_ranges' or 'fixed_params'.\n"
            f"Suggested default ranges from DEFAULT_PARAM_RANGES:\n" +
            "\n".join([f"  '{p}': {DEFAULT_PARAM_RANGES.get(p, 'no default')}"
                      for p in missing_params])
        )

    return len(missing_params) == 0, list(missing_params)


def generate_hod_parameter_grid(
    halo,
    hod_type: str,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    target_ngal: float,
    Ac_fiducial: float = 1.0,
    fixed_params: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    conformity: bool = False,
    include_nfw_extensions: bool = False,
    auto_add_defaults: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate HOD parameter grid with fixed number density using Latin Hypercube Sampling.

    This is the main function that combines LHS parameter generation with Ac rescaling
    to create a grid of HOD parameters suitable for emulator training. Each parameter
    combination has Ac adjusted to match the target number density.

    Parameters
    ----------
    halo : HaloOccupation
        HaloOccupation instance with cosmology and halo catalog loaded
    hod_type : str
        HOD model type ('LRG', 'ELG_GHOD', 'ELG_SFR')
    param_ranges : dict
        Parameter ranges for free parameters (excluding Ac and any fixed params).
        Required parameters depend on model:
        - All models: 'Mmin', 'sig_M', 'As', 'M1', 'alpha', 'kappa'
        - ELG_SFR also requires: 'gamma'
        - Conformity models also require: 'kappa_EE' (M1_EE = kappa_EE * M1)
        Any parameter in fixed_params should NOT be in param_ranges.
    n_samples : int
        Number of parameter combinations to generate
    target_ngal : float
        Target galaxy number density [(Mpc/h)^-3]
    Ac_fiducial : float, default=1.0
        Fiducial value for Ac used in rescaling
    fixed_params : dict, optional
        Dictionary of parameters to hold fixed (not sampled in LHS).
        Example: {'M1': 13.5, 'kappa': 1.0} to fix M1 and kappa.
        These will be constant across all samples.
    random_seed : int, optional
        Random seed for reproducibility
    conformity : bool, default=False
        Whether to use conformity model (adds kappa_EE parameter where M1_EE = kappa_EE * M1)
    include_nfw_extensions : bool, default=False
        Whether to include extended NFW profile parameters (f_exp, tau, lambda_NFW).
        If True, these parameters must be in param_ranges or fixed_params.
    auto_add_defaults : bool, default=True
        If True, automatically add default ranges for missing extension parameters
        (assembly bias, conformity, NFW extensions) that are detected as required
        but not provided in param_ranges or fixed_params.
    save_path : str, optional
        If provided, save the parameter grid to this path (as parquet or csv)
    verbose : bool, default=True
        If True, print progress information and detected extensions

    Returns
    -------
    param_grid : pd.DataFrame
        DataFrame with columns for all HOD parameters including rescaled Ac.
        Shape: (n_samples, n_params+1) where 'Ac' is the rescaled central amplitude.

    Examples
    --------
    >>> from HOD_NRV.HOD_catalogue import HaloOccupation
    >>> from HOD_NRV.emulator_utils import generate_hod_parameter_grid
    >>>
    >>> # Setup halo catalog
    >>> halo = HaloOccupation(
    ...     cosmology=cosmo_params,
    ...     zeff=1.0,
    ...     Lbox=1000,
    ...     column_mapping=column_mapping,
    ...     mass_definition="vir",
    ...     DataFrame=df
    ... )
    >>>
    >>> # Define parameter ranges (excluding Ac and any fixed params)
    >>> param_ranges = {
    ...     'Mmin': (11.5, 12.5),
    ...     'sig_M': (0.2, 0.6),
    ...     'As': (0.1, 1.0),
    ...     'alpha': (0.5, 1.5),
    ...     'kappa': (0.5, 1.5)
    ... }
    >>>
    >>> # Fix M1 to a constant value (not sampled)
    >>> # Note: With conformity, M1_EE = kappa_EE * M1, so fixing M1 still allows
    >>> # conformity strength to vary via kappa_EE
    >>> fixed_params = {'M1': 13.5}
    >>>
    >>> # Generate parameter grid with fixed ngal and fixed M1
    >>> param_grid = generate_hod_parameter_grid(
    ...     halo=halo,
    ...     hod_type='ELG_GHOD',
    ...     param_ranges=param_ranges,
    ...     n_samples=1000,
    ...     target_ngal=1e-3,
    ...     Ac_fiducial=0.5,
    ...     fixed_params=fixed_params,
    ...     random_seed=42,
    ...     save_path='hod_param_grid.parquet'
    ... )
    >>>
    >>> # Use the parameters for population
    >>> for idx, row in param_grid.iterrows():
    ...     params = row.to_dict()
    ...     halo.populate_haloes(params)
    ...     # Compute clustering, lensing, etc.

    Notes
    -----
    This function performs the following steps:

    1. Generate LHS samples for free parameters (excluding Ac)
    2. For each parameter combination:
       - Compute ngal with fiducial Ac
       - Rescale Ac to match target_ngal
       - Store the rescaled Ac and achieved ngal
    3. Return DataFrame with all parameters including Ac

    The resulting parameter grid maintains fixed number density while sampling
    the ratio Ac/As, which governs the clustering and lensing signals.

    See Also
    --------
    create_latin_hypercube : Generate LHS samples
    rescale_Ac_to_target_ngal : Rescale Ac for target ngal
    """
    if verbose:
        print(f"Generating {n_samples} HOD parameter combinations for {hod_type}")
        print(f"Target number density: {target_ngal:.2e} (Mpc/h)^-3")

    # Detect required parameters based on extensions
    required_params = _get_required_parameters(
        halo, hod_type, conformity, include_nfw_extensions
    )

    if verbose:
        print(f"\nDetected extensions:")
        print(f"  Assembly bias: {halo.assembly_bias if hasattr(halo, 'assembly_bias') else False}")
        print(f"  Conformity: {conformity}")
        print(f"  NFW extensions: {include_nfw_extensions}")
        print(f"\nRequired parameters: {sorted(required_params)}")

    # Make a copy of param_ranges to potentially modify
    param_ranges_copy = param_ranges.copy()
    fixed_params_copy = fixed_params.copy() if fixed_params is not None else {}

    # Auto-add default ranges for missing parameters if requested
    if auto_add_defaults:
        provided_params = set(param_ranges_copy.keys()) | set(fixed_params_copy.keys())
        missing_params = required_params - provided_params

        if missing_params:
            if verbose:
                print(f"\nAuto-adding default ranges for missing parameters: {sorted(missing_params)}")

            for param in missing_params:
                if param in DEFAULT_PARAM_RANGES:
                    param_ranges_copy[param] = DEFAULT_PARAM_RANGES[param]
                    if verbose:
                        print(f"  {param}: {DEFAULT_PARAM_RANGES[param]}")
                else:
                    raise ValueError(
                        f"Parameter '{param}' is required but no default range is available. "
                        f"Please specify it in 'param_ranges' or 'fixed_params'."
                    )

    # Validate that all required parameters are provided
    is_valid, missing = _validate_parameters(
        param_ranges_copy, fixed_params_copy, required_params, hod_type, verbose
    )

    if not is_valid and not auto_add_defaults:
        raise ValueError(
            f"Missing required parameters: {missing}. "
            f"Either add them to param_ranges/fixed_params or set auto_add_defaults=True"
        )

    if verbose and fixed_params_copy:
        print(f"\nFixed parameters: {fixed_params_copy}")

    # Set HOD model
    halo.set_halo_model(hod_type, conformity=conformity)

    # Generate LHS samples for free parameters (those in param_ranges)
    if verbose:
        print("\nCreating Latin Hypercube samples...")
    lhs_samples = create_latin_hypercube(param_ranges_copy, n_samples, random_seed)

    # Add fixed parameters as constant columns
    if fixed_params_copy:
        for param_name, param_value in fixed_params_copy.items():
            lhs_samples[param_name] = param_value

    # Initialize array for Ac
    Ac_values = np.zeros(n_samples)

    # Rescale Ac for each parameter combination
    if verbose:
        print("Rescaling Ac to match target number density...")

    for i in range(n_samples):
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        # Get parameters for this sample
        params = lhs_samples.iloc[i].to_dict()

        # Rescale Ac and As to achieve target ngal (preserving Ac/As ratio)
        Ac_rescaled, _ = rescale_Ac_to_target_ngal(
            halo.HOD, params, target_ngal, Ac_fiducial
        )

        Ac_values[i] = Ac_rescaled
        # Note: As is NOT updated in the grid - we keep the sampled As values
        # Rescaling happens at prediction time by calling rescale_Ac_to_target_ngal

    # Combine into final DataFrame
    param_grid = lhs_samples.copy()
    param_grid.insert(0, 'Ac', Ac_values)  # Insert Ac as first column

    if verbose:
        print(f"\nDone! Parameter grid statistics:")
        print(f"  Target ngal: {target_ngal:.4e} (Mpc/h)^-3")
        print(f"  Ac range: [{Ac_values.min():.4f}, {Ac_values.max():.4f}]")

    # Save if path provided
    if save_path is not None:
        if save_path.endswith('.parquet'):
            param_grid.to_parquet(save_path, index=False)
        elif save_path.endswith('.csv'):
            param_grid.to_csv(save_path, index=False)
        else:
            # Default to parquet
            param_grid.to_parquet(save_path + '.parquet', index=False)

        if verbose:
            print(f"\nSaved parameter grid to: {save_path}")

    return param_grid


def load_parameter_grid(file_path: str) -> pd.DataFrame:
    """
    Load a previously saved parameter grid.

    Parameters
    ----------
    file_path : str
        Path to the saved parameter grid (parquet or csv)

    Returns
    -------
    param_grid : pd.DataFrame
        Loaded parameter grid

    Examples
    --------
    >>> param_grid = load_parameter_grid('hod_param_grid.parquet')
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        # Try both formats
        try:
            return pd.read_parquet(file_path)
        except:
            return pd.read_csv(file_path)
