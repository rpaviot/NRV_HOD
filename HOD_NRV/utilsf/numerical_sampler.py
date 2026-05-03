"""
Nautilus nested sampler for fitting DeltaSigma with the numerical HOD pipeline.

Supports three progressively complex fit cases:
1. STANDARD_NFW — standard NFW satellite profiles with Ac/As rescaling
2. EXTENDED_PROFILE — adds exponential cutoff (f_exp, tau) to the satellite profile
3. CONFORMITY — adds AbacusHOD-style conformity (kappa_EE)

All cases fix the galaxy number density via rescale_Ac_to_target_ngal and
fix M1 = 13.0 by default.
"""

import warnings
import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple, Any

from interpax import Interpolator1D

from .emulator_utils import rescale_Ac_to_target_ngal
from .emulator_nn_flax import load_emulator, predict_dsigma, predict_wgg
from HOD_NRV.HOD_numerical.HOD import HaloOccupation


# Module-level state for fork-safe pool parallelism.
# Set by NumericalDeltaSigmaFitter.run() before the pool is created;
# inherited by worker processes via fork COW.
_fitter_instance = None


def _fitter_likelihood(theta):
    """Lightweight module-level wrapper used by Nautilus pool workers.

    Nautilus pickles this as a name reference (cheap), not as a closure.
    Workers inherit _fitter_instance via fork copy-on-write.
    """
    return _fitter_instance.log_likelihood(theta)


class FitCase(IntEnum):
    STANDARD_NFW = 1
    EXTENDED_PROFILE = 2
    CONFORMITY = 3


# Parameter definitions per case: (name, low, high)
_BASE_PARAMS = [
    ("As",          0.002, 0.05),
    ("Mmin",        11.5, 13.5),
    ("sig_M",       0.1, 2.0),
    ("gamma",       0.0, 10.0),
    ("alpha",       0.1, 2.0),
    ("kappa",       0.1, 2.0),
    ("lambda_NFW",  0.1, 2.0),
]

_EXTENDED_PARAMS = [
    ("f_exp", 0.1, 0.9),
    ("tau",   1.0, 10.0),
]

_CONFORMITY_PARAMS = [
    ("kappa_EE", 0.5, 2.0),
]

_ELG_SATELLITE_PARAMS = [
    ("Mcut", 11.0, 13.0),
    ("Mmax", 13.5, 15.5),
]

# All recognized HOD parameter names — used by set_priors() for validation.
_ALL_KNOWN_PARAMS = {
    p[0] for p in _BASE_PARAMS + _EXTENDED_PARAMS + _CONFORMITY_PARAMS + _ELG_SATELLITE_PARAMS
} | {"Ac", "As", "M1", "A_cent", "B_cent", "A_sat", "B_sat"}


def _get_free_params(fit_case: FitCase, elg_satellite: bool = False):
    """Return list of (name, low, high) for the given fit case."""
    if elg_satellite:
        params = [p for p in _BASE_PARAMS if p[0] != 'kappa'] + _ELG_SATELLITE_PARAMS
    else:
        params = list(_BASE_PARAMS)
    if fit_case >= FitCase.EXTENDED_PROFILE:
        params.extend(_EXTENDED_PARAMS)
    if fit_case >= FitCase.CONFORMITY:
        params.extend(_CONFORMITY_PARAMS)
    return [(n, lo, hi, "uniform") for (n, lo, hi) in params]


def _parse_param_config(param_config: dict):
    """Split a {name: spec | scalar} dict into (priors, fixed_params).

    Value formats
    -------------
    - ``(low, high)`` or ``[low, high]``         → uniform prior (free)
    - ``(mean, std, 'gaussian')`` (any order)    → Gaussian prior (free)
    - scalar                                     → fixed parameter

    Returns
    -------
    priors : list of (name, a, b, kind)
        kind is 'uniform' (a=low, b=high) or 'gaussian' (a=mean, b=std).
    fixed_params : list of (name, value)
    """
    priors = []
    fixed_params = []
    for name, value in param_config.items():
        if isinstance(value, (tuple, list)):
            strs = [v for v in value if isinstance(v, str)]
            nums = [v for v in value if not isinstance(v, str)]
            if strs:
                kind = strs[0].lower()
                if kind != "gaussian":
                    raise ValueError(
                        f"Unknown prior kind '{strs[0]}' for '{name}'. "
                        "Only 'gaussian' is supported."
                    )
                if len(nums) != 2:
                    raise ValueError(
                        f"Gaussian prior for '{name}' requires exactly "
                        "(mean, std, 'gaussian')."
                    )
                priors.append((name, float(nums[0]), float(nums[1]), "gaussian"))
            elif len(value) == 2:
                priors.append((name, float(value[0]), float(value[1]), "uniform"))
            else:
                raise ValueError(
                    f"Invalid prior spec for '{name}': {value!r}"
                )
        else:
            fixed_params.append((name, float(value)))
    return priors, fixed_params


class NumericalDeltaSigmaFitter:
    """
    Nautilus nested sampler for fitting galaxy-galaxy lensing (DeltaSigma)
    using the numerical HOD pipeline with the ELG_mHMQ central model.

    Parameters
    ----------
    cosmology : dict
        Cosmological parameters passed to HaloOccupation.
    zeff : float
        Effective redshift.
    Lbox : float
        Box side length [Mpc/h].
    column_mapping : dict
        Maps internal column names to catalog column names.
    mass_definition : str
        Halo mass definition (e.g. "200c").
    DataFrame : pd.DataFrame, optional
        Halo catalog DataFrame.
    halo_path : str, optional
        Path to halo catalog parquet file (used if DataFrame is None).
    DataFrame_part : pd.DataFrame, optional
        Particle catalog DataFrame.
    assembly_bias : bool
        Enable assembly bias weighting.
    NFW_scaled : bool
        Use mass-concentration scaled NFW profile.
    outerprofile : bool
        Include outer halo profile term.
    apply_rsd : bool
        Apply redshift-space distortions.
    triaxial_NFW : bool or object
        Triaxial NFW profile option.
    rsd_axis : str
        Axis along which RSD is applied ('x', 'y', or 'z').
    do_test : bool
        Run internal consistency checks at init.
    particle_fraction : float
        Fraction of particles to subsample (1.0 = all particles).
    particle_subsample_seed : int
        Random seed for particle subsampling.
    precomputed_cache : HaloCenterLensingCache
        Loaded from HDF5 via HaloCenterLensingCache.load().
    fit_case : FitCase
        Which model complexity to use.
    data_path : str
        Path to .npz file with observed DeltaSigma data.
    target_ngal : float
        Fixed galaxy number density [(Mpc/h)^-3].
    n_realizations : int
        Number of HOD realizations per likelihood evaluation.
    galaxy_fraction : float
        Galaxy downsampling fraction per realization.
    M1_fixed : float
        Fixed log10(M1) value.
    rp_bins : array-like, optional
        Projected separation bin edges [Mpc/h]. If None, loaded from data file.
    base_seed : int
        Base random seed for HOD realizations.
    rp_min : float, optional
        Minimum projected separation scale cut [Mpc/h]. Bins below this are excluded.
    rp_max : float, optional
        Maximum projected separation scale cut [Mpc/h]. Bins above this are excluded.
    """

    def __init__(
        self,
        # --- HaloOccupation constructor args ---
        cosmology: Dict[str, float],
        zeff: float,
        Lbox: float,
        column_mapping: Dict[str, str],
        mass_definition: str,
        DataFrame=None,
        halo_path: Optional[str] = None,
        DataFrame_part=None,
        assembly_bias: bool = False,
        NFW_scaled: bool = True,
        outerprofile: bool = True,
        apply_rsd: bool = True,
        triaxial_NFW=False,
        rsd_axis: str = 'z',
        do_test: bool = True,
        particle_fraction: float = 1.0,
        particle_subsample_seed: int = 42,
        # --- Precomputed cache (still passed externally) ---
        precomputed_cache=None,
        # --- Fitter-specific args ---
        fit_case: FitCase = FitCase.STANDARD_NFW,
        elg_satellite: bool = False,
        data_path: str = "",
        target_ngal: float = 2.3e-4,
        n_realizations: int = 5,
        galaxy_fraction: float = 0.1,
        M1_fixed: float = 13.0,
        rp_bins=None,
        base_seed: int = 1000,
        # --- Scale cuts ---
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        # --- Custom priors: {name: (low, high)} to sample, {name: scalar} to fix ---
        param_config: Optional[dict] = None,
    ):
        self.halo = HaloOccupation(
            cosmology=cosmology, zeff=zeff, Lbox=Lbox,
            column_mapping=column_mapping, mass_definition=mass_definition,
            DataFrame=DataFrame, halo_path=halo_path, DataFrame_part=DataFrame_part,
            assembly_bias=assembly_bias, NFW_scaled=NFW_scaled, outerprofile=outerprofile,
            apply_rsd=apply_rsd, triaxial_NFW=triaxial_NFW, rsd_axis=rsd_axis,
            do_test=do_test, particle_fraction=particle_fraction,
            particle_subsample_seed=particle_subsample_seed,
        )
        self.precomputed_cache = precomputed_cache
        self.fit_case = FitCase(fit_case)
        self.elg_satellite = elg_satellite
        self.target_ngal = target_ngal
        self.n_realizations = n_realizations
        self.galaxy_fraction = galaxy_fraction
        self.M1_fixed = M1_fixed
        self.fixed_params_dict = {"M1": M1_fixed}
        self.base_seed = base_seed
        self.rp_min = rp_min
        self.rp_max = rp_max

        # Configure HOD model
        use_conformity = (self.fit_case == FitCase.CONFORMITY)
        self.halo.set_halo_model("ELG_mHMQ", conformity=use_conformity, elg_satellite=elg_satellite)

        # Free parameters for this case
        self.free_params = _get_free_params(self.fit_case, elg_satellite=elg_satellite)
        self.param_names = [p[0] for p in self.free_params]
        self.n_params = len(self.free_params)

        # Load observed data (scale cuts applied inside)
        self._load_data(data_path)

        # Override rp_bins if provided
        if rp_bins is not None:
            self.rp_bins = np.asarray(rp_bins)

        if param_config is not None:
            priors, fixed = _parse_param_config(param_config)
            self.set_priors(priors, fixed)

    def _load_data(self, data_path: str):
        """Load observed DeltaSigma data from .npz file.

        Expected keys (flexible naming):
            rp or rp_centers — projected separations
            rp_bins or rp_edges — bin edges (optional, constructed from rp if absent)
            delta_sigma or dsigma — observed DeltaSigma signal
            covariance or cov — covariance matrix
        """
        data = np.load(data_path)

        # rp centers
        for key in ("rp", "rp_centers"):
            if key in data:
                self.rp = data[key]
                break
        else:
            raise KeyError("Data file must contain 'rp' or 'rp_centers'")

        # rp bin edges
        for key in ("rp_bins", "rp_edges"):
            if key in data:
                self.rp_bins = data[key]
                break
        else:
            # Construct log-spaced bin edges from centers
            log_rp = np.log10(self.rp)
            dlog = np.diff(log_rp)[0] / 2
            edges = np.concatenate([
                [log_rp[0] - dlog],
                (log_rp[:-1] + log_rp[1:]) / 2,
                [log_rp[-1] + dlog]
            ])
            self.rp_bins = 10**edges

        # DeltaSigma signal
        for key in ("delta_sigma", "dsigma"):
            if key in data:
                self.ds_obs = data[key]
                break
        else:
            raise KeyError("Data file must contain 'delta_sigma' or 'dsigma'")

        # Covariance
        for key in ("cov_delta_sigma", "cov_dsigma"):
            if key in data:
                self.cov = data[key]
                break
        else:
            raise KeyError("Data file must contain 'cov_delta_sigma' or 'cov_dsigma'")

        # Apply scale cuts
        mask = np.ones(len(self.rp), dtype=bool)
        if self.rp_min is not None:
            mask &= (self.rp >= self.rp_min)
        if self.rp_max is not None:
            mask &= (self.rp <= self.rp_max)

        if not mask.all():
            self.rp     = self.rp[mask]
            self.ds_obs = self.ds_obs[mask]
            self.cov    = self.cov[np.ix_(mask, mask)]
            idx = np.where(mask)[0]
            self.rp_bins = self.rp_bins[idx[0] : idx[-1] + 2]

        self.cov_inv = np.linalg.inv(self.cov)
        self.n_bins = len(self.ds_obs)

    def _derive_Ac_As(self, As_sampled: float, params_no_AcAs: dict) -> Tuple[float, float]:
        """Derive Ac and As by rescaling to target_ngal.

        Ac_fid = 0.01; As_sampled is passed directly. Both are rescaled
        proportionally by rescale_Ac_to_target_ngal to hit target_ngal.

        Returns (Ac_new, As_new).
        """
        Ac_fid = 0.01

        params_for_rescale = params_no_AcAs.copy()
        params_for_rescale["As"] = As_sampled

        Ac_new, As_new = rescale_Ac_to_target_ngal(
            self.halo.HOD, params_for_rescale, self.target_ngal, Ac_fiducial=Ac_fid
        )
        return Ac_new, As_new

    def _build_params(self, theta) -> Optional[Dict[str, float]]:
        """Convert flat parameter vector to full HOD parameter dict.

        Returns None if the derived parameters are unphysical (e.g. Ac <= 0).
        """
        # Nautilus passes theta as a dict {name: value}; fallback for array input.
        if isinstance(theta, dict):
            free_dict = dict(theta)
        else:
            free_dict = dict(zip(self.param_names, theta))

        As_sampled = free_dict.pop("As")

        # Merge free params with fixed params (fixed take precedence, Ac/As excluded)
        params_no_AcAs = dict(free_dict)
        params_no_AcAs.update(
            {k: v for k, v in self.fixed_params_dict.items() if k not in ("Ac", "As")}
        )

        try:
            Ac, As = self._derive_Ac_As(As_sampled, params_no_AcAs)
        except Exception:
            return None

        if Ac <= 0 or As <= 0 or not np.isfinite(Ac) or not np.isfinite(As):
            return None

        full_params = {"Ac": Ac, "As": As}
        full_params.update(params_no_AcAs)

        return full_params

    def set_priors(self, priors, fixed_params=()):
        """Override free-parameter bounds and fixed values after construction.

        Parameters
        ----------
        priors : list of (name, low, high)
            Free parameters. Unrecognized names emit a warning; all others
            are accepted.
        fixed_params : list of (name, value)
            Parameters held constant during sampling. Supersedes M1_fixed.
        """
        self.fixed_params_dict = dict(fixed_params)
        if "M1" not in self.fixed_params_dict:
            self.fixed_params_dict["M1"] = self.M1_fixed
        else:
            self.M1_fixed = self.fixed_params_dict["M1"]

        active = []
        for entry in priors:
            if len(entry) == 3:
                name, a, b = entry
                kind = "uniform"
            else:
                name, a, b, kind = entry
            if name not in _ALL_KNOWN_PARAMS:
                warnings.warn(
                    f"set_priors: '{name}' is not a recognized HOD parameter; ignoring."
                )
            else:
                active.append((name, float(a), float(b), kind))

        self.free_params = active
        self.param_names = [p[0] for p in active]
        self.n_params = len(active)

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Compute log-likelihood for a parameter vector.

        Parameters
        ----------
        theta : array of shape (n_params,)
            Free parameter values in the order of self.param_names.

        Returns
        -------
        float
            Log-likelihood value, or -1e100 on failure.
        """
        full_params = self._build_params(theta)
        if full_params is None:
            return -1e100

        try:
            rp, ds_mean, ds_std = self.halo.compute_avg_lensing(
                full_params,
                n_realizations=self.n_realizations,
                bins1=self.rp_bins,
                method="optimized",
                precomputed_cache=self.precomputed_cache,
                galaxy_fraction=self.galaxy_fraction,
                base_seed=self.base_seed,
            )
        except Exception:
            return -1e100

        if not np.all(np.isfinite(ds_mean)):
            return -1e100

        residual = ds_mean - self.ds_obs
        chi2 = residual @ self.cov_inv @ residual
        return -0.5 * chi2

    def run(
        self,
        n_live: int = 500,
        n_eff: int = 5000,
        filepath: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 1,
        **nautilus_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the Nautilus nested sampler.

        Parameters
        ----------
        n_live : int
            Number of live points.
        n_eff : int
            Target effective sample size.
        filepath : str, optional
            Checkpoint file path for resume capability.
        verbose : bool
            Whether to print progress.
        n_workers : int
            Number of parallel likelihood-evaluation workers. If > 1, a
            forkserver multiprocessing pool is created and passed to Nautilus.
            Each worker is initialised with a single-thread environment
            (OMP/XLA/OpenBLAS = 1) so N workers use N cores in total rather
            than N*ncpu. Set to the number of available CPUs for maximum
            throughput.
        **nautilus_kwargs
            Extra kwargs passed to nautilus.Sampler.run().

        Returns
        -------
        points : array of shape (n_eff, n_params)
            Posterior samples.
        weights : array of shape (n_eff,)
            Sample weights.
        log_l : array of shape (n_eff,)
            Log-likelihoods.
        log_z : float
            Log-evidence estimate.
        """
        import nautilus
        import os
        import HOD_NRV.utilsf.numerical_sampler as _self_mod

        from scipy.stats import norm as _norm
        prior = nautilus.Prior()
        for name, a, b, kind in self.free_params:
            if kind == "gaussian":
                prior.add_parameter(name, dist=_norm(loc=a, scale=b))
            else:
                prior.add_parameter(name, dist=(a, b))

        # Expose self to forked workers via module-level global (COW, no pickling).
        _self_mod._fitter_instance = self

        # Limit each worker to 1 thread so N workers use N cores total.
        # Save and restore so caller's environment is not permanently changed.
        _saved_env = {}
        if n_workers > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                _saved_env[var] = os.environ.get(var)
                os.environ[var] = "1"

        pool_arg = n_workers if n_workers > 1 else None

        try:
            sampler = nautilus.Sampler(
                prior,
                _fitter_likelihood,   # module-level function → cheap to pickle
                n_live=n_live,
                filepath=filepath,
                pool=pool_arg,        # integer → NautilusPool embeds likelihood in workers
                pass_dict=False,
            )
            sampler.run(n_eff=n_eff, verbose=verbose, **nautilus_kwargs)
        finally:
            _self_mod._fitter_instance = None
            for var, val in _saved_env.items():
                if val is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = val

        points, log_w, log_l = sampler.posterior()
        weights = np.exp(log_w - log_w.max())
        weights /= weights.sum()
        log_z = sampler.evidence()

        return points, weights, log_l, log_z

    def get_best_fit(self, points: np.ndarray, log_l: np.ndarray) -> Dict[str, float]:
        """Return the MAP (maximum a-posteriori) parameter estimate.

        Parameters
        ----------
        points : array of shape (n, n_params)
            Posterior samples from run().
        log_l : array of shape (n,)
            Corresponding log-likelihoods.

        Returns
        -------
        dict
            Full parameter dictionary including derived Ac and As.
        """
        idx_best = np.argmax(log_l)
        theta_best = points[idx_best]
        full_params = self._build_params(theta_best)

        # Add free parameter values for reference
        result = {}
        for i, name in enumerate(self.param_names):
            result[name] = theta_best[i]
        result.update(full_params)
        return result

    def get_posterior_summary(
        self, points: np.ndarray, weights: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute weighted posterior summary statistics.

        Parameters
        ----------
        points : array of shape (n, n_params)
            Posterior samples.
        weights : array of shape (n,)
            Normalized sample weights.

        Returns
        -------
        dict
            For each parameter: mean, std, median, q16, q84.
        """
        summary = {}
        for i, name in enumerate(self.param_names):
            vals = points[:, i]
            w = weights

            mean = np.average(vals, weights=w)
            std = np.sqrt(np.average((vals - mean) ** 2, weights=w))

            sorted_idx = np.argsort(vals)
            vals_sorted = vals[sorted_idx]
            w_sorted = w[sorted_idx]
            cumw = np.cumsum(w_sorted)
            cumw /= cumw[-1]

            median = vals_sorted[np.searchsorted(cumw, 0.5)]
            q16 = vals_sorted[np.searchsorted(cumw, 0.16)]
            q84 = vals_sorted[np.searchsorted(cumw, 0.84)]

            summary[name] = {
                "mean": mean,
                "std": std,
                "median": median,
                "q16": q16,
                "q84": q84,
            }
        return summary

    def save_results(
        self,
        path: str,
        points: np.ndarray,
        weights: np.ndarray,
        log_l: np.ndarray,
        log_z: float,
    ):
        """Save sampler results to .npz file.

        Parameters
        ----------
        path : str
            Output file path.
        points, weights, log_l : arrays
            Posterior samples, weights, log-likelihoods from run().
        log_z : float
            Log-evidence.
        """
        np.savez(
            path,
            points=points,
            weights=weights,
            log_l=log_l,
            log_z=log_z,
            param_names=self.param_names,
            fit_case=int(self.fit_case),
            target_ngal=self.target_ngal,
            M1_fixed=self.M1_fixed,
            rp=self.rp,
            rp_bins=self.rp_bins,
            ds_obs=self.ds_obs,
            cov=self.cov,
        )


# ---------------------------------------------------------------------------
# EmulatorFitter — Nautilus sampler backed by a pre-trained DeltaSigmaEmulator
# ---------------------------------------------------------------------------

# Module-level state for fork-safe pool parallelism (mirrors _fitter_instance).
_emulator_fitter_instance = None


def _emulator_likelihood(theta):
    """Module-level wrapper for EmulatorFitter, picklable by name."""
    return _emulator_fitter_instance.log_likelihood(theta)


class EmulatorFitter:
    """
    Nautilus nested sampler backed by a pre-trained DeltaSigmaEmulator.

    Identical interface to NumericalDeltaSigmaFitter but replaces the slow
    compute_avg_lensing() forward model (~3.2 s/call) with an emulator forward
    pass (~μs/call), making full Nautilus runs feasible in minutes.

    Supports optional joint fitting of DeltaSigma + w_gg by supplying a second
    wgg emulator path and corresponding data. The joint log-likelihood is:
        log L = -½(χ²_DS + χ²_wgg)
    with independent covariances assumed between the two probes.

    Parameters
    ----------
    emulator_path : str
        Path to the DeltaSigma emulator ``.npz`` file written by train_emulator().
    fit_case : FitCase
        Which model complexity to use (determines free parameters).
    data_path : str, optional
        Path to .npz file with observed DeltaSigma data. Flexible key names:
        ``rp`` / ``rp_centers``, ``rp_bins`` / ``rp_edges``,
        ``delta_sigma`` / ``dsigma``, ``cov_delta_sigma`` / ``cov_dsigma``.
        Provide either ``data_path`` or all three of (``ds_obs``, ``cov_inv``,
        ``rp_obs``).
    ds_obs : np.ndarray, shape (n_bins,), optional
        Observed DeltaSigma signal. Used when ``data_path`` is not given.
    cov_inv : np.ndarray, shape (n_bins, n_bins), optional
        Inverse covariance matrix. Used when ``data_path`` is not given.
    rp_obs : np.ndarray, shape (n_bins,), optional
        Projected separation bin centers [Mpc/h]. Used when ``data_path`` is
        not given.
    param_names_ordered : list of str, optional
        Ordered list of HOD parameter names as stored in the emulator grid.
        Must match the column order of the param_grid used during training.
        Defaults to the order read from the emulator metadata.
    M1_fixed : float, default=13.0
        Fixed log10(M1) value.
    rp_min : float, optional
        Minimum rp scale cut for DeltaSigma [Mpc/h].
    rp_max : float, optional
        Maximum rp scale cut for DeltaSigma [Mpc/h].
    emulator_wgg_path : str, optional
        Path to the w_gg emulator ``.npz`` file. When provided, enables joint
        DeltaSigma + w_gg fitting.
    data_path_wgg : str, optional
        Path to .npz file with observed w_gg data. Flexible key names:
        ``rp`` / ``rp_centers``, ``wgg`` / ``wp`` / ``w_gg``,
        ``cov_wgg`` / ``cov_wp`` / ``cov_w_gg``.
    wgg_obs : np.ndarray, shape (n_bins_wgg,), optional
        Observed w_p(rp) signal. Used when ``data_path_wgg`` is not given.
    cov_inv_wgg : np.ndarray, shape (n_bins_wgg, n_bins_wgg), optional
        Inverse covariance for w_gg. Used when ``data_path_wgg`` is not given.
    rp_obs_wgg : np.ndarray, shape (n_bins_wgg,), optional
        Projected separations for w_gg [Mpc/h]. Used when ``data_path_wgg``
        is not given.
    rp_min_wgg : float, optional
        Minimum rp scale cut for w_gg [Mpc/h].
    rp_max_wgg : float, optional
        Maximum rp scale cut for w_gg [Mpc/h].

    Examples
    --------
    DeltaSigma-only fit (existing interface):

    >>> fitter = EmulatorFitter(
    ...     emulator_path="emulator_STANDARD_NFW",
    ...     data_path="observed_ds.npz",
    ...     fit_case=FitCase.STANDARD_NFW,
    ... )

    Joint DeltaSigma + w_gg fit:

    >>> fitter = EmulatorFitter(
    ...     emulator_path="emulator_STANDARD_NFW",
    ...     data_path="observed_ds.npz",
    ...     fit_case=FitCase.STANDARD_NFW,
    ...     emulator_wgg_path="emulator_wgg_STANDARD_NFW",
    ...     data_path_wgg="observed_wgg.npz",
    ...     rp_min_wgg=1.0,
    ... )
    >>> points, weights, log_l, log_z = fitter.run(n_live=500, n_eff=5000)
    """

    def __init__(
        self,
        emulator_path: str,
        fit_case: FitCase = FitCase.STANDARD_NFW,
        # --- Data: provide EITHER data_path OR (ds_obs + cov_inv + rp_obs) ---
        data_path: str = "",
        ds_obs: Optional[np.ndarray] = None,
        cov_inv: Optional[np.ndarray] = None,
        rp_obs: Optional[np.ndarray] = None,
        # --- Remaining params ---
        param_names_ordered: Optional[list] = None,
        M1_fixed: float = 13.0,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        # --- Optional wgg emulator & data ---
        emulator_wgg_path: str = "",
        data_path_wgg: str = "",
        wgg_obs: Optional[np.ndarray] = None,
        cov_inv_wgg: Optional[np.ndarray] = None,
        rp_obs_wgg: Optional[np.ndarray] = None,
        rp_min_wgg: Optional[float] = None,
        rp_max_wgg: Optional[float] = None,
        # --- Custom priors: {name: (low, high)} to sample, {name: scalar} to fix ---
        param_config: Optional[dict] = None,
    ):
        self.fit_case = FitCase(fit_case)
        self.M1_fixed = M1_fixed
        self.fixed_params_dict = {"M1": M1_fixed}
        self.rp_min = rp_min
        self.rp_max = rp_max

        # Load emulator
        self.model, self.norm_stats = load_emulator(emulator_path)
        self.emulator_rp = self.norm_stats["rp_centers"]  # shape (n_rp_emulator,)

        # Free parameters for this case
        self.free_params = _get_free_params(self.fit_case)
        self.param_names = [p[0] for p in self.free_params]
        self.n_params = len(self.free_params)

        # Parameter order expected by the emulator (column order of training grid)
        if param_names_ordered is not None:
            self.emulator_param_order = param_names_ordered
        else:
            self.emulator_param_order = list(self.norm_stats["param_names"])

        # Load observed data
        if data_path:
            self._load_data(data_path)
        elif ds_obs is not None and cov_inv is not None and rp_obs is not None:
            self.rp_obs = np.asarray(rp_obs)
            self.ds_obs = np.asarray(ds_obs)
            self.cov_inv = np.asarray(cov_inv)

            mask = np.ones(len(self.rp_obs), dtype=bool)
            if rp_min is not None:
                mask &= (self.rp_obs >= rp_min)
            if rp_max is not None:
                mask &= (self.rp_obs <= rp_max)
            if not mask.all():
                self.rp_obs  = self.rp_obs[mask]
                self.ds_obs  = self.ds_obs[mask]
                self.cov_inv = self.cov_inv[np.ix_(mask, mask)]

            self.n_bins = len(self.ds_obs)
        else:
            raise ValueError(
                "Provide either data_path or all three of (ds_obs, cov_inv, rp_obs)."
            )

        # Pre-compute emulator → obs interpolation indices
        self._setup_interp()

        # --- Optional wgg emulator ---
        self.rp_min_wgg = rp_min_wgg
        self.rp_max_wgg = rp_max_wgg
        self._fit_wgg = False

        if emulator_wgg_path:
            self.model_wgg, self.norm_stats_wgg = load_emulator(emulator_wgg_path)
            self.emulator_rp_wgg = self.norm_stats_wgg["rp_centers"]

            if data_path_wgg:
                self._load_data_wgg(data_path_wgg)
            elif wgg_obs is not None and cov_inv_wgg is not None and rp_obs_wgg is not None:
                self.rp_obs_wgg = np.asarray(rp_obs_wgg)
                self.wgg_obs = np.asarray(wgg_obs)
                self.cov_inv_wgg = np.asarray(cov_inv_wgg)
                mask = np.ones(len(self.rp_obs_wgg), dtype=bool)
                if rp_min_wgg is not None:
                    mask &= (self.rp_obs_wgg >= rp_min_wgg)
                if rp_max_wgg is not None:
                    mask &= (self.rp_obs_wgg <= rp_max_wgg)
                if not mask.all():
                    self.rp_obs_wgg  = self.rp_obs_wgg[mask]
                    self.wgg_obs     = self.wgg_obs[mask]
                    self.cov_inv_wgg = self.cov_inv_wgg[np.ix_(mask, mask)]
            else:
                raise ValueError(
                    "emulator_wgg_path given but no wgg data provided. "
                    "Supply data_path_wgg or (wgg_obs, cov_inv_wgg, rp_obs_wgg)."
                )

            self._setup_interp_wgg()
            self._fit_wgg = True

        if param_config is not None:
            priors, fixed = _parse_param_config(param_config)
            self.set_priors(priors, fixed)

    def _load_data(self, data_path: str):
        """Load observed DeltaSigma data from .npz file.

        Expected keys (flexible naming):
            rp or rp_centers — projected separations
            rp_bins or rp_edges — bin edges (optional, constructed from rp if absent)
            delta_sigma or dsigma — observed DeltaSigma signal
            cov_delta_sigma or cov_dsigma — covariance matrix
        """
        data = np.load(data_path)

        # rp centers
        for key in ("rp", "rp_centers"):
            if key in data:
                self.rp_obs = data[key]
                break
        else:
            raise KeyError("Data file must contain 'rp' or 'rp_centers'")

        # rp bin edges
        for key in ("rp_bins", "rp_edges"):
            if key in data:
                self.rp_bins = data[key]
                break
        else:
            log_rp = np.log10(self.rp_obs)
            dlog = np.diff(log_rp)[0] / 2
            edges = np.concatenate([
                [log_rp[0] - dlog],
                (log_rp[:-1] + log_rp[1:]) / 2,
                [log_rp[-1] + dlog]
            ])
            self.rp_bins = 10**edges

        # DeltaSigma signal
        for key in ("delta_sigma", "dsigma"):
            if key in data:
                self.ds_obs = data[key]
                break
        else:
            raise KeyError("Data file must contain 'delta_sigma' or 'dsigma'")

        # Covariance
        for key in ("cov_delta_sigma", "cov_dsigma"):
            if key in data:
                self.cov = data[key]
                break
        else:
            raise KeyError("Data file must contain 'cov_delta_sigma' or 'cov_dsigma'")

        # Apply scale cuts
        mask = np.ones(len(self.rp_obs), dtype=bool)
        if self.rp_min is not None:
            mask &= (self.rp_obs >= self.rp_min)
        if self.rp_max is not None:
            mask &= (self.rp_obs <= self.rp_max)

        if not mask.all():
            self.rp_obs  = self.rp_obs[mask]
            self.ds_obs  = self.ds_obs[mask]
            self.cov     = self.cov[np.ix_(mask, mask)]
            idx = np.where(mask)[0]
            self.rp_bins = self.rp_bins[idx[0] : idx[-1] + 2]

        self.cov_inv = np.linalg.inv(self.cov)
        self.n_bins = len(self.ds_obs)

    def _setup_interp(self):
        """Pre-compute log rp grids for interpolation from emulator grid to obs grid."""
        self._log_rp_emu = np.log(self.emulator_rp)
        self._log_rp_obs = np.log(self.rp_obs)

    def _load_data_wgg(self, data_path_wgg: str):
        """Load observed w_gg data from .npz file.

        Flexible key names:
            rp or rp_centers        — projected separations
            wgg, wp, or w_gg        — observed w_p(rp) signal
            cov_wgg, cov_wp, or cov_w_gg — covariance matrix
        """
        data = np.load(data_path_wgg)

        for key in ("rp", "rp_centers"):
            if key in data:
                self.rp_obs_wgg = data[key]
                break
        else:
            raise KeyError("wgg data file must contain 'rp' or 'rp_centers'")

        for key in ("wgg", "wp", "w_gg"):
            if key in data:
                self.wgg_obs = data[key]
                break
        else:
            raise KeyError("wgg data file must contain 'wgg', 'wp', or 'w_gg'")

        for key in ("cov_wgg", "cov_wp", "cov_w_gg"):
            if key in data:
                self.cov_wgg = data[key]
                break
        else:
            raise KeyError("wgg data file must contain 'cov_wgg', 'cov_wp', or 'cov_w_gg'")

        # Apply scale cuts
        mask = np.ones(len(self.rp_obs_wgg), dtype=bool)
        if self.rp_min_wgg is not None:
            mask &= (self.rp_obs_wgg >= self.rp_min_wgg)
        if self.rp_max_wgg is not None:
            mask &= (self.rp_obs_wgg <= self.rp_max_wgg)
        if not mask.all():
            self.rp_obs_wgg = self.rp_obs_wgg[mask]
            self.wgg_obs    = self.wgg_obs[mask]
            self.cov_wgg    = self.cov_wgg[np.ix_(mask, mask)]

        self.cov_inv_wgg = np.linalg.inv(self.cov_wgg)

    def _setup_interp_wgg(self):
        """Pre-compute log rp grids for wgg interpolation from emulator grid to obs grid."""
        self._log_rp_emu_wgg = np.log(self.emulator_rp_wgg)
        self._log_rp_obs_wgg = np.log(self.rp_obs_wgg)

    def set_priors(self, priors, fixed_params=()):
        """Override free-parameter bounds and fixed values after construction.

        Parameters
        ----------
        priors : list of (name, low, high)
            Free parameters. Names absent from the emulator's parameter list
            are silently skipped; completely unrecognized names emit a warning.
        fixed_params : list of (name, value)
            Parameters held constant during sampling. Supersedes M1_fixed and
            any previously fixed value.
        """
        self.fixed_params_dict = dict(fixed_params)
        if "M1" not in self.fixed_params_dict:
            self.fixed_params_dict["M1"] = self.M1_fixed
        else:
            self.M1_fixed = self.fixed_params_dict["M1"]

        emulator_params = set(self.emulator_param_order)
        active = []
        for entry in priors:
            if len(entry) == 3:
                name, a, b = entry
                kind = "uniform"
            else:
                name, a, b, kind = entry
            if name not in _ALL_KNOWN_PARAMS:
                warnings.warn(
                    f"set_priors: '{name}' is not a recognized HOD parameter; ignoring."
                )
            elif name not in emulator_params:
                pass  # valid param, not in this emulator — silently skip
            else:
                active.append((name, float(a), float(b), kind))

        # Warn if any emulator param is uncovered
        covered = {p[0] for p in active} | set(self.fixed_params_dict)
        missing = [p for p in self.emulator_param_order if p not in covered]
        if missing:
            warnings.warn(
                f"set_priors: emulator params {missing} are neither free nor fixed. "
                "Likelihood evaluation will fail for those params."
            )

        self.free_params = active
        self.param_names = [p[0] for p in active]
        self.n_params = len(active)

    def log_likelihood(self, theta) -> float:
        """Compute log-likelihood using the emulator forward pass.

        Parameters
        ----------
        theta : dict or array
            Free parameter values from Nautilus (same params that appear in
            the training grid: As, Mmin, sig_M, gamma, M1, alpha, kappa,
            lambda_NFW, ...). No Ac — it is a derived parameter.

        Returns
        -------
        float
            Log-likelihood, or -1e100 on failure.
        """
        if isinstance(theta, dict):
            free_dict = dict(theta)
        else:
            free_dict = dict(zip(self.param_names, theta))

        try:
            theta_vec = np.array([
                self.fixed_params_dict[p] if p in self.fixed_params_dict else free_dict[p]
                for p in self.emulator_param_order
            ], dtype=np.float32)
            ds_pred = predict_dsigma(self.model, self.norm_stats, theta_vec)
        except Exception:
            return -1e100

        if not np.all(np.isfinite(ds_pred)):
            return -1e100

        # Interpolate emulator prediction onto obs rp grid (log-cubic)
        log_ds_emu = np.log(ds_pred)
        log_ds_at_obs = np.array(
            Interpolator1D(self._log_rp_emu, log_ds_emu, method='cubic')(self._log_rp_obs)
        )
        ds_at_obs = np.exp(log_ds_at_obs)

        residual = ds_at_obs - self.ds_obs
        chi2 = residual @ self.cov_inv @ residual

        # --- Optional wgg chi² ---
        if self._fit_wgg:
            try:
                theta_vec_wgg = np.array([
                    self.fixed_params_dict[p] if p in self.fixed_params_dict else free_dict[p]
                    for p in list(self.norm_stats_wgg["param_names"])
                ], dtype=np.float32)
                wgg_pred = predict_wgg(self.model_wgg, self.norm_stats_wgg, theta_vec_wgg)
            except Exception:
                return -1e100
            if not np.all(np.isfinite(wgg_pred)):
                return -1e100
            log_wgg_emu = np.log(wgg_pred)
            log_wgg_at_obs = np.array(
                Interpolator1D(self._log_rp_emu_wgg, log_wgg_emu, method='cubic')(self._log_rp_obs_wgg)
            )
            wgg_at_obs = np.exp(log_wgg_at_obs)
            residual_wgg = wgg_at_obs - self.wgg_obs
            chi2 += residual_wgg @ self.cov_inv_wgg @ residual_wgg

        return -0.5 * chi2

    def run(
        self,
        n_live: int = 500,
        n_eff: int = 5000,
        filepath: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 1,
        **nautilus_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the Nautilus nested sampler with the emulator likelihood.

        Parameters
        ----------
        n_live : int
            Number of live points.
        n_eff : int
            Target effective sample size.
        filepath : str, optional
            Checkpoint file path for resume capability.
        verbose : bool
            Whether to print progress.
        n_workers : int
            Number of parallel likelihood workers. Emulator calls are
            cheap (~μs), so n_workers=1 is usually sufficient.
        **nautilus_kwargs
            Extra kwargs forwarded to nautilus.Sampler.run().

        Returns
        -------
        points : np.ndarray, shape (n_eff, n_params)
        weights : np.ndarray, shape (n_eff,)
        log_l : np.ndarray, shape (n_eff,)
        log_z : float
        """
        import nautilus
        import HOD_NRV.utilsf.numerical_sampler as _self_mod

        from scipy.stats import norm as _norm
        prior = nautilus.Prior()
        for name, a, b, kind in self.free_params:
            if kind == "gaussian":
                prior.add_parameter(name, dist=_norm(loc=a, scale=b))
            else:
                prior.add_parameter(name, dist=(a, b))

        _self_mod._emulator_fitter_instance = self

        pool_arg = n_workers if n_workers > 1 else None

        try:
            sampler = nautilus.Sampler(
                prior,
                _emulator_likelihood,
                n_live=n_live,
                filepath=filepath,
                pool=pool_arg,
                pass_dict=False,
            )
            sampler.run(n_eff=n_eff, verbose=verbose, **nautilus_kwargs)
        finally:
            _self_mod._emulator_fitter_instance = None

        points, log_w, log_l = sampler.posterior()
        weights = np.exp(log_w - log_w.max())
        weights /= weights.sum()
        log_z = sampler.evidence()

        return points, weights, log_l, log_z

    def get_best_fit(self, points: np.ndarray, log_l: np.ndarray) -> Dict[str, float]:
        """Return the MAP (maximum a-posteriori) parameter estimate.

        Parameters
        ----------
        points : array of shape (n, n_params)
            Posterior samples from run().
        log_l : array of shape (n,)
            Corresponding log-likelihoods.

        Returns
        -------
        dict
            Free parameter values at the MAP, plus M1_fixed.
        """
        idx_best = np.argmax(log_l)
        theta_best = points[idx_best]
        result = dict(zip(self.param_names, theta_best))
        result["M1"] = self.M1_fixed
        return result

    def get_posterior_summary(
        self, points: np.ndarray, weights: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute weighted posterior summary statistics.

        Parameters
        ----------
        points : array of shape (n, n_params)
            Posterior samples.
        weights : array of shape (n,)
            Normalized sample weights.

        Returns
        -------
        dict
            For each parameter: mean, std, median, q16, q84.
        """
        summary = {}
        for i, name in enumerate(self.param_names):
            vals = points[:, i]
            w = weights

            mean = np.average(vals, weights=w)
            std = np.sqrt(np.average((vals - mean) ** 2, weights=w))

            sorted_idx = np.argsort(vals)
            vals_sorted = vals[sorted_idx]
            w_sorted = w[sorted_idx]
            cumw = np.cumsum(w_sorted)
            cumw /= cumw[-1]

            median = vals_sorted[np.searchsorted(cumw, 0.5)]
            q16 = vals_sorted[np.searchsorted(cumw, 0.16)]
            q84 = vals_sorted[np.searchsorted(cumw, 0.84)]

            summary[name] = {
                "mean": mean,
                "std": std,
                "median": median,
                "q16": q16,
                "q84": q84,
            }
        return summary

    def save_results(
        self,
        path: str,
        points: np.ndarray,
        weights: np.ndarray,
        log_l: np.ndarray,
        log_z: float,
    ):
        """Save sampler results to .npz file.

        Parameters
        ----------
        path : str
            Output file path.
        points, weights, log_l : arrays
            Posterior samples, weights, log-likelihoods from run().
        log_z : float
            Log-evidence.
        """
        arrays = dict(
            points=points,
            weights=weights,
            log_l=log_l,
            log_z=log_z,
            param_names=self.param_names,
            fit_case=int(self.fit_case),
            M1_fixed=self.M1_fixed,
            rp_obs=self.rp_obs,
            ds_obs=self.ds_obs,
        )
        # cov and rp_bins are only present when data was loaded from file
        if hasattr(self, "cov"):
            arrays["cov"] = self.cov
        if hasattr(self, "rp_bins"):
            arrays["rp_bins"] = self.rp_bins
        if self._fit_wgg:
            arrays["rp_obs_wgg"] = self.rp_obs_wgg
            arrays["wgg_obs"]    = self.wgg_obs
            if hasattr(self, "cov_wgg"):
                arrays["cov_wgg"] = self.cov_wgg
        np.savez(path, **arrays)
