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

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Set
from scipy.stats import qmc
import warnings

from HOD_NRV.HOD_numerical.HOD_models import (
    compute_ngal_with_fiducial_Ac,
    rescale_Ac_to_target_ngal,
)


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
    # ELG satellite cutoff parameters
    'Mcut': (11.0, 13.0),
    'Mmax': (13.5, 15.5),
    # Extended NFW profile parameters
    'f_exp': (0.0, 0.5),        # Exponential component fraction
    'tau': (2.0, 10.0),         # Decay scale in units of Rs
    'lambda_NFW': (0.5, 2.0),   # NFW rescaling factor
}


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
    include_nfw_extensions: bool = False,
    elg_satellite: bool = False,
) -> Set[str]:
    """
    Get the complete set of required HOD parameters based on model and extensions.

    Parameters
    ----------
    halo : HaloOccupation
        HaloOccupation instance to inspect for enabled extensions
    hod_type : str
        HOD model type ('LRG', 'ELG_GHOD', 'ELG_SFR', 'ELG_mHMQ')
    conformity : bool
        Whether conformity is enabled
    include_nfw_extensions : bool, default=False
        Whether to include NFW extension parameters
    elg_satellite : bool, default=False
        Whether to use ELG satellite cutoff (Mcut/Mmax) instead of standard kappa

    Returns
    -------
    required_params : set
        Set of required parameter names (excluding Ac which is computed)
    """
    # Base parameters for all models
    required = {'Mmin', 'sig_M', 'As', 'M1', 'alpha'}

    # Satellite cutoff: kappa (standard) or Mcut+Mmax (ELG)
    if elg_satellite:
        required.update({'Mcut', 'Mmax'})
    else:
        required.add('kappa')

    # Add gamma for ELG_SFR and ELG_mHMQ
    if hod_type in ('ELG_SFR', 'ELG_mHMQ'):
        required.add('gamma')

    # Add conformity parameter
    if conformity:
        required.add('kappa_EE')

    # Add assembly bias parameters if enabled in halo instance
    if hasattr(halo, 'assembly_bias') and halo.assembly_bias:
        required.update(['A_cent', 'B_cent', 'A_sat', 'B_sat'])

    # lambda_NFW is always free (all fit cases); f_exp/tau only for extended profile
    required.add('lambda_NFW')
    if include_nfw_extensions:
        required.update(['f_exp', 'tau'])

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
    fixed_params: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    conformity: bool = False,
    elg_satellite: bool = False,
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
        DataFrame with columns for all free LHS-sampled HOD parameters (no Ac).
        Shape: (n_samples, n_params). Ac is a derived parameter: it is computed
        by run_hod_grid() just before each forward-model call to hit target_ngal,
        but is never stored in the grid or checkpoint files.

    Examples
    --------
    >>> from HOD_NRV.HOD_numerical.HOD import HaloOccupation
    >>> from HOD_NRV.utilsf.emulator_utils import generate_hod_parameter_grid
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

    1. Generate LHS samples for free parameters (Ac excluded — it is derived)
    2. Add fixed_params as constant columns
    3. Return DataFrame with free params only (no Ac column)

    ``Ac`` is a fully derived parameter: given the sampled free params and
    ``target_ngal``, there is a unique ``Ac`` (and rescaled ``As``) that hits
    the number density constraint.  The derivation happens inside
    ``run_hod_grid`` just before each ``compute_avg_lensing`` call, but those
    derived values are NOT stored in ``params_array``.  The emulator is trained
    on the original free params (including ``As`` as sampled by LHS), and at
    inference time Nautilus samples those same free params directly.

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
        halo, hod_type, conformity, include_nfw_extensions, elg_satellite
    )

    if verbose:
        print(f"\nDetected extensions:")
        print(f"  Assembly bias: {halo.assembly_bias if hasattr(halo, 'assembly_bias') else False}")
        print(f"  Conformity: {conformity}")
        print(f"  ELG satellite: {elg_satellite}")
        print(f"  NFW extensions: lambda_NFW=True, extended profile (f_exp,tau)={include_nfw_extensions}")
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
    halo.set_halo_model(hod_type, conformity=conformity, elg_satellite=elg_satellite)

    # Generate LHS samples for free parameters (those in param_ranges)
    if verbose:
        print("\nCreating Latin Hypercube samples...")
    lhs_samples = create_latin_hypercube(param_ranges_copy, n_samples, random_seed)

    # Add fixed parameters as constant columns
    if fixed_params_copy:
        for param_name, param_value in fixed_params_copy.items():
            lhs_samples[param_name] = param_value

    # Ac is fully derived — it is NOT stored in the grid.
    # run_hod_grid() computes (Ac, As) on-the-fly to hit target_ngal, but does
    # not write those values back into params_array.
    param_grid = lhs_samples.copy()

    if verbose:
        print(f"\nDone! Parameter grid: {n_samples} points, {len(param_grid.columns)} "
              f"free params (Ac excluded — derived at evaluation time).")

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


def run_hod_grid(
    param_grid: pd.DataFrame,
    halo,
    rp_bins: np.ndarray,
    n_realizations: int = 5,
    galaxy_fraction: float = 0.10,
    precomputed_cache=None,
    save_path: Optional[str] = None,
    checkpoint_every: int = 200,
    base_seed: int = 42,
    mpi_rank: int = 0,
    target_ngal: Optional[float] = None,
    Ac_fiducial: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate DeltaSigma on a pre-generated HOD parameter grid.

    Iterates over rows of param_grid, calls halo.compute_avg_lensing() for each,
    and returns arrays suitable for emulator training. Supports fault-tolerant
    checkpointing and resuming from an existing checkpoint.

    Parameters
    ----------
    param_grid : pd.DataFrame
        Parameter grid, e.g. from generate_hod_parameter_grid(). Each row must
        contain the free HOD parameters (no Ac; Ac is derived when target_ngal
        is provided).
    halo : HaloOccupation
        HaloOccupation instance with halo and particle catalogs loaded and
        set_halo_model() already called.
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]. Shape: (n_bins+1,).
    n_realizations : int, default=5
        Number of HOD realizations averaged per grid point.
    galaxy_fraction : float, default=0.10
        Galaxy downsampling fraction per realization.
    precomputed_cache : HaloCenterLensingCache, optional
        If provided, use ``method="optimized"`` (precomputed halo-center profiles
        for centrals, pycorr only for satellites). This is **strongly recommended**
        for large grids: it gives a 3–9× speedup over the standard method depending
        on satellite fraction. Generate it once with
        ``common_test/precompute_halo_center_cache.py``, then load with
        ``HaloCenterLensingCache.load(path)``.
        If None (default), falls back to ``method="standard"`` (full pycorr pair
        counting per realization).
    save_path : str, optional
        If provided, save incremental .npz checkpoints to this path.
        Must end in .npz. Also used for resume detection.
    checkpoint_every : int, default=200
        Save a checkpoint every this many completed grid points.
    base_seed : int, default=42
        Base random seed; grid point i uses base_seed + i * n_realizations.
    mpi_rank : int, default=0
        MPI rank number, used only for log messages.
    target_ngal : float, optional
        If provided, call ``rescale_Ac_to_target_ngal`` just before each
        ``compute_avg_lensing`` call to derive ``(Ac, As_rescaled)`` that
        achieve ``target_ngal``. These derived values are used **only** for
        the HOD forward model call; they are NOT written back into
        ``params_array``. The emulator trains on the original free params
        (including ``As`` as LHS-sampled). ``param_grid`` is never modified.
    Ac_fiducial : float, default=0.01
        Fiducial Ac passed to ``rescale_Ac_to_target_ngal`` when ``target_ngal``
        is set. Only used when ``target_ngal`` is not None.

    Returns
    -------
    params_array : np.ndarray
        Shape (n_points, n_params). Contains the original free-param values
        from param_grid unchanged (no Ac column; rescaled Ac/As are never
        stored here).
    rp_centers : np.ndarray
        Shape (n_rp_bins,). Bin centers in Mpc/h.
    dsigma_array : np.ndarray
        Shape (n_points, n_rp_bins). Mean DeltaSigma per grid point.

    Notes
    -----
    MPI slicing (splitting param_grid across ranks) must be done by the caller
    before passing param_grid to this function. This function is single-process.

    **Particle subsampling** is controlled at `HaloOccupation` construction time
    via ``particle_fraction`` and ``particle_subsample_seed``, not here.
    The benchmark-optimal setting is ``particle_fraction=0.05``. Pass it when
    constructing the ``halo`` object before calling this function.

    If save_path already exists, the function resumes from the last checkpoint
    and skips already-completed rows.
    """
    n_points = len(param_grid)
    param_names = list(param_grid.columns)

    # Compute rp_centers from bin edges
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    n_rp = len(rp_centers)

    # Allocate output arrays
    params_array = param_grid.values.copy()
    dsigma_array = np.full((n_points, n_rp), np.nan)

    # Resume from existing checkpoint if available
    start_idx = 0
    if save_path is not None and os.path.exists(save_path):
        try:
            ckpt = np.load(save_path, allow_pickle=True)
            n_done = int(ckpt.get("n_completed", 0))
            if n_done > 0 and n_done <= n_points:
                dsigma_array[:n_done] = ckpt["dsigma_array"][:n_done]
                params_array[:n_done] = ckpt["params_array"][:n_done]
                start_idx = n_done
                print(f"[rank {mpi_rank}] Resuming from checkpoint: {n_done}/{n_points} done")
        except Exception as e:
            warnings.warn(f"[rank {mpi_rank}] Could not load checkpoint ({e}); starting fresh")

    method = "optimized" if precomputed_cache is not None else "standard"
    print(f"[rank {mpi_rank}] Evaluating {n_points - start_idx} grid points "
          f"({start_idx} already done) — method={method}")

    for i in range(start_idx, n_points):
        row_params = param_grid.iloc[i].to_dict()
        point_seed = base_seed + i * n_realizations

        # Derive (Ac, As_rescaled) to hit target_ngal.
        # These values are used ONLY for the forward model call — not stored in
        # params_array. The emulator trains on the original free params (row_params).
        if target_ngal is not None:
            Ac_r, As_r = rescale_Ac_to_target_ngal(
                halo.HOD, row_params, target_ngal, Ac_fiducial=Ac_fiducial
            )
            row_params_for_hod = {**row_params, 'Ac': Ac_r, 'As': As_r}
        else:
            row_params_for_hod = row_params  # caller must include 'Ac'

        try:
            _, ds_mean, _ = halo.compute_avg_lensing(
                row_params_for_hod,
                n_realizations=n_realizations,
                bins1=rp_bins,
                method=method,
                precomputed_cache=precomputed_cache,
                galaxy_fraction=galaxy_fraction,
                base_seed=point_seed,
            )
            dsigma_array[i] = ds_mean
        except Exception as e:
            warnings.warn(f"[rank {mpi_rank}] Grid point {i} failed: {e}")
            dsigma_array[i] = np.nan

        # Checkpoint
        if save_path is not None and ((i + 1) % checkpoint_every == 0 or i == n_points - 1):
            _save_grid_checkpoint(
                save_path, params_array, rp_centers, dsigma_array,
                param_names, n_completed=i + 1
            )
            print(f"[rank {mpi_rank}] Checkpoint saved: {i + 1}/{n_points}")

    print(f"[rank {mpi_rank}] Grid evaluation complete.")
    return params_array, rp_centers, dsigma_array


def _save_grid_checkpoint(
    path: str,
    params_array: np.ndarray,
    rp_centers: np.ndarray,
    dsigma_array: np.ndarray,
    param_names: List[str],
    n_completed: int,
):
    """Save grid evaluation checkpoint to .npz file."""
    import os
    np.savez_compressed(
        path,
        params_array=params_array,
        rp_centers=rp_centers,
        dsigma_array=dsigma_array,
        param_names=np.array(param_names, dtype=object),
        n_completed=n_completed,
    )


def merge_grid_chunks(
    directory: str,
    n_ranks: int,
    output: str,
    rank_prefix: str = "grid_rank",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge per-rank .npz grid files produced by run_hod_grid() into a single file.

    Parameters
    ----------
    directory : str
        Directory containing per-rank checkpoint files.
    n_ranks : int
        Number of MPI ranks (expects files rank0..rank{n_ranks-1}).
    output : str
        Path for the merged output .npz file.
    rank_prefix : str, default="grid_rank"
        Filename prefix; files are expected as {directory}/{rank_prefix}{i}.npz.

    Returns
    -------
    params_array : np.ndarray
        Merged parameter array, shape (N_total, n_params).
    rp_centers : np.ndarray
        Bin centers (same across all ranks).
    dsigma_array : np.ndarray
        Merged DeltaSigma array, shape (N_total, n_rp).
    """
    import os

    all_params = []
    all_dsigma = []
    rp_centers = None
    param_names = None

    for rank in range(n_ranks):
        fpath = os.path.join(directory, f"{rank_prefix}{rank}.npz")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected chunk file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True)
        n_done = int(data.get("n_completed", len(data["params_array"])))

        all_params.append(data["params_array"][:n_done])
        all_dsigma.append(data["dsigma_array"][:n_done])

        if rp_centers is None:
            rp_centers = data["rp_centers"]
        if param_names is None and "param_names" in data:
            param_names = list(data["param_names"])

    params_array = np.concatenate(all_params, axis=0)
    dsigma_array = np.concatenate(all_dsigma, axis=0)

    # Drop rows where DeltaSigma is all-NaN (failed evaluations)
    valid = ~np.all(np.isnan(dsigma_array), axis=1)
    params_array = params_array[valid]
    dsigma_array = dsigma_array[valid]

    print(f"Merged {params_array.shape[0]} valid grid points from {n_ranks} ranks")

    np.savez_compressed(
        output,
        params_array=params_array,
        rp_centers=rp_centers,
        dsigma_array=dsigma_array,
        param_names=np.array(param_names if param_names else [], dtype=object),
    )
    print(f"Merged grid saved to: {output}")

    return params_array, rp_centers, dsigma_array


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
