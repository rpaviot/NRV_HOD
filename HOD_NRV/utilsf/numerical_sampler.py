"""
Nautilus nested sampler for fitting DeltaSigma with the numerical HOD pipeline.

Supports three progressively complex fit cases:
1. STANDARD_NFW — standard NFW satellite profiles with Ac/As rescaling
2. EXTENDED_PROFILE — adds exponential cutoff (f_exp, tau) to the satellite profile
3. CONFORMITY — adds AbacusHOD-style conformity (kappa_EE)

All cases fix the galaxy number density via rescale_Ac_to_target_ngal and
fix M1 = 13.0 by default.
"""

import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple, Any

from .emulator_utils import rescale_Ac_to_target_ngal
from .emulator_nn import load_emulator, predict_dsigma
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


def _get_free_params(fit_case: FitCase):
    """Return list of (name, low, high) for the given fit case."""
    params = list(_BASE_PARAMS)
    if fit_case >= FitCase.EXTENDED_PROFILE:
        params.extend(_EXTENDED_PARAMS)
    if fit_case >= FitCase.CONFORMITY:
        params.extend(_CONFORMITY_PARAMS)
    return params


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
        self.target_ngal = target_ngal
        self.n_realizations = n_realizations
        self.galaxy_fraction = galaxy_fraction
        self.M1_fixed = M1_fixed
        self.base_seed = base_seed
        self.rp_min = rp_min
        self.rp_max = rp_max

        # Configure HOD model
        use_conformity = (self.fit_case == FitCase.CONFORMITY)
        self.halo.set_halo_model("ELG_mHMQ", conformity=use_conformity)

        # Free parameters for this case
        self.free_params = _get_free_params(self.fit_case)
        self.param_names = [p[0] for p in self.free_params]
        self.n_params = len(self.free_params)

        # Load observed data (scale cuts applied inside)
        self._load_data(data_path)

        # Override rp_bins if provided
        if rp_bins is not None:
            self.rp_bins = np.asarray(rp_bins)

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

        # Build the non-Ac/As params needed for ngal rescaling
        params_no_AcAs = {
            "Mmin": free_dict["Mmin"],
            "sig_M": free_dict["sig_M"],
            "gamma": free_dict["gamma"],
            "M1": self.M1_fixed,
            "alpha": free_dict["alpha"],
            "kappa": free_dict["kappa"],
        }

        try:
            Ac, As = self._derive_Ac_As(As_sampled, params_no_AcAs)
        except Exception:
            return None

        if Ac <= 0 or As <= 0 or not np.isfinite(Ac) or not np.isfinite(As):
            return None

        # Full parameter dict
        full_params = {
            "Ac": Ac,
            "As": As,
            "Mmin": free_dict["Mmin"],
            "sig_M": free_dict["sig_M"],
            "gamma": free_dict["gamma"],
            "M1": self.M1_fixed,
            "alpha": free_dict["alpha"],
            "kappa": free_dict["kappa"],
            "lambda_NFW": free_dict["lambda_NFW"],
        }

        if self.fit_case >= FitCase.EXTENDED_PROFILE:
            full_params["f_exp"] = free_dict["f_exp"]
            full_params["tau"] = free_dict["tau"]

        if self.fit_case >= FitCase.CONFORMITY:
            full_params["kappa_EE"] = free_dict["kappa_EE"]

        return full_params

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

        prior = nautilus.Prior()
        for name, low, high in self.free_params:
            prior.add_parameter(name, dist=(low, high))

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
        for i, (name, _, _) in enumerate(self.free_params):
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
        for i, (name, _, _) in enumerate(self.free_params):
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

    Parameters
    ----------
    emulator_path : str
        Path to the saved .pt model file written by train_emulator().
        The companion <emulator_path>.norm.npz must also exist.
    ds_obs : np.ndarray, shape (n_bins,)
        Observed DeltaSigma signal.
    cov_inv : np.ndarray, shape (n_bins, n_bins)
        Inverse covariance matrix.
    rp_obs : np.ndarray, shape (n_bins,)
        Projected separation bin centers corresponding to ds_obs [Mpc/h].
    fit_case : FitCase
        Which model complexity to use (determines free parameters).
    target_ngal : float
        Fixed galaxy number density [(Mpc/h)^-3] used for Ac/As rescaling.
    param_names_ordered : list of str, optional
        Ordered list of HOD parameter names as stored in the emulator grid.
        Must match the column order of the param_grid used during training.
        Defaults to the order implied by fit_case via _get_free_params + M1.
    M1_fixed : float, default=13.0
        Fixed log10(M1) value.
    rp_min : float, optional
        Minimum rp scale cut [Mpc/h].
    rp_max : float, optional
        Maximum rp scale cut [Mpc/h].
    hod_model : HaloOccupation or object with .HOD, optional
        Needed only for the Ac/As rescaling step. If None, Ac rescaling
        is skipped and parameters are passed directly to the emulator — only
        valid if the emulator was trained with Ac already rescaled (as produced
        by generate_hod_parameter_grid).

    Examples
    --------
    >>> fitter = EmulatorFitter(
    ...     emulator_path="emulator_STANDARD_NFW.pt",
    ...     ds_obs=ds_obs, cov_inv=cov_inv, rp_obs=rp_centers,
    ...     fit_case=FitCase.STANDARD_NFW,
    ...     target_ngal=2e-4,
    ...     hod_model=halo,
    ... )
    >>> points, weights, log_l, log_z = fitter.run(n_live=500, n_eff=5000)
    """

    def __init__(
        self,
        emulator_path: str,
        ds_obs: np.ndarray,
        cov_inv: np.ndarray,
        rp_obs: np.ndarray,
        fit_case: FitCase = FitCase.STANDARD_NFW,
        target_ngal: float = 2.3e-4,
        param_names_ordered: Optional[list] = None,
        M1_fixed: float = 13.0,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        hod_model=None,
    ):
        self.fit_case = FitCase(fit_case)
        self.target_ngal = target_ngal
        self.M1_fixed = M1_fixed
        self.hod_model = hod_model
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
        # Default: full parameter dict order as built by _build_params_full()
        if param_names_ordered is not None:
            self.emulator_param_order = param_names_ordered
        else:
            self.emulator_param_order = list(self.norm_stats["param_names"])

        # Apply scale cuts to obs data
        self.rp_obs = np.asarray(rp_obs)
        self.ds_obs = np.asarray(ds_obs)
        self.cov_inv = np.asarray(cov_inv)

        mask = np.ones(len(self.rp_obs), dtype=bool)
        if rp_min is not None:
            mask &= (self.rp_obs >= rp_min)
        if rp_max is not None:
            mask &= (self.rp_obs <= rp_max)
        if not mask.all():
            self.rp_obs   = self.rp_obs[mask]
            self.ds_obs   = self.ds_obs[mask]
            self.cov_inv  = self.cov_inv[np.ix_(mask, mask)]

        self.n_bins = len(self.ds_obs)

        # Pre-compute emulator → obs interpolation indices
        # We interpolate emulator predictions onto rp_obs using log-linear interp
        self._setup_interp()

    def _default_emulator_param_order(self):
        """Default parameter order expected by the emulator.

        Matches the column order of the param_grid produced by
        generate_hod_parameter_grid(): free LHS-sampled params + fixed M1.
        Ac is NOT included — it is a derived parameter never stored in the grid.
        """
        order = ["As", "Mmin", "sig_M", "gamma", "M1", "alpha", "kappa", "lambda_NFW"]
        if self.fit_case >= FitCase.EXTENDED_PROFILE:
            order += ["f_exp", "tau"]
        if self.fit_case >= FitCase.CONFORMITY:
            order += ["kappa_EE"]
        return order

    def _setup_interp(self):
        """Pre-compute interpolation from emulator rp grid to obs rp grid."""
        # We will interpolate log(DS_emulator)(log rp_emulator) onto log rp_obs
        self._log_rp_emu = np.log(self.emulator_rp)
        self._log_rp_obs = np.log(self.rp_obs)

    def _build_params_full(self, theta) -> Optional[Dict[str, float]]:
        """Convert sampler parameter vector to full HOD parameter dict.

        Mirrors NumericalDeltaSigmaFitter._build_params but is used only
        to drive the Ac/As rescaling step. Returns None on failure.
        """
        if isinstance(theta, dict):
            free_dict = dict(theta)
        else:
            free_dict = dict(zip(self.param_names, theta))

        As_sampled = free_dict.pop("As")

        params_no_AcAs = {
            "Mmin": free_dict["Mmin"],
            "sig_M": free_dict["sig_M"],
            "gamma": free_dict["gamma"],
            "M1": self.M1_fixed,
            "alpha": free_dict["alpha"],
            "kappa": free_dict["kappa"],
        }

        if self.hod_model is not None:
            try:
                Ac_fid = 0.01
                params_for_rescale = params_no_AcAs.copy()
                params_for_rescale["As"] = As_sampled
                Ac, As = rescale_Ac_to_target_ngal(
                    self.hod_model.HOD, params_for_rescale,
                    self.target_ngal, Ac_fiducial=Ac_fid
                )
            except Exception:
                return None
            if Ac <= 0 or As <= 0 or not np.isfinite(Ac) or not np.isfinite(As):
                return None
        else:
            # No HOD model provided: pass As directly, Ac = 1 (placeholder)
            Ac = 1.0
            As = As_sampled

        full_params = {
            "Ac": Ac,
            "As": As,
            "Mmin": free_dict["Mmin"],
            "sig_M": free_dict["sig_M"],
            "gamma": free_dict["gamma"],
            "M1": self.M1_fixed,
            "alpha": free_dict["alpha"],
            "kappa": free_dict["kappa"],
            "lambda_NFW": free_dict["lambda_NFW"],
        }

        if self.fit_case >= FitCase.EXTENDED_PROFILE:
            full_params["f_exp"] = free_dict["f_exp"]
            full_params["tau"] = free_dict["tau"]

        if self.fit_case >= FitCase.CONFORMITY:
            full_params["kappa_EE"] = free_dict["kappa_EE"]

        return full_params

    def _params_to_emulator_vector(self, full_params: dict) -> np.ndarray:
        """Extract ordered parameter vector for emulator input."""
        return np.array(
            [full_params[k] for k in self.emulator_param_order],
            dtype=np.float32,
        )

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
                self.M1_fixed if p == 'M1' else free_dict[p]
                for p in self.emulator_param_order
            ], dtype=np.float32)
            ds_pred = predict_dsigma(self.model, self.norm_stats, theta_vec)
        except Exception:
            return -1e100

        if not np.all(np.isfinite(ds_pred)):
            return -1e100

        # Interpolate emulator prediction onto obs rp grid (log-linear)
        log_ds_emu = np.log(ds_pred)
        log_ds_at_obs = np.interp(
            self._log_rp_obs, self._log_rp_emu, log_ds_emu
        )
        ds_at_obs = np.exp(log_ds_at_obs)

        residual = ds_at_obs - self.ds_obs
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

        prior = nautilus.Prior()
        for name, low, high in self.free_params:
            prior.add_parameter(name, dist=(low, high))

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
