"""
Nautilus nested sampler for fitting galaxy-galaxy lensing (and optionally w_gg)
with the ELG_mHMQ HOD model, backed by a pre-trained DeltaSigma emulator.

Supports three progressively complex fit cases:
1. STANDARD_NFW — standard NFW satellite profiles with Ac/As rescaling
2. EXTENDED_PROFILE — adds exponential cutoff (f_exp, tau)
3. CONFORMITY — adds AbacusHOD-style conformity (kappa_EE)

All cases fix the galaxy number density (via the Ac/As rescaling baked into the
training grid) and fix M1 = 13.0 by default.
"""

import warnings
import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple

from interpax import Interpolator1D

from .emulator_nn_flax import load_emulator, predict_dsigma, predict_wgg


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

# All recognized HOD parameter names — used by set_priors() for validation.
_ALL_KNOWN_PARAMS = (
    {p[0] for p in _BASE_PARAMS + _EXTENDED_PARAMS + _CONFORMITY_PARAMS}
    | {"Ac", "As", "M1", "Mmax", "Mcut", "A_cent", "B_cent", "A_sat", "B_sat"}
)


def _get_free_params(fit_case: FitCase):
    """Return list of (name, low, high, 'uniform') for the given fit case."""
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
                raise ValueError(f"Invalid prior spec for '{name}': {value!r}")
        else:
            fixed_params.append((name, float(value)))
    return priors, fixed_params


# ---------------------------------------------------------------------------
# EmulatorFitter — Nautilus sampler backed by a pre-trained DeltaSigmaEmulator
# ---------------------------------------------------------------------------

# Module-level state for fork-safe pool parallelism. Set by EmulatorFitter.run()
# before the pool is created; inherited by worker processes via fork COW.
_emulator_fitter_instance = None


def _emulator_likelihood(theta):
    """Module-level wrapper for EmulatorFitter, picklable by name."""
    return _emulator_fitter_instance.log_likelihood(theta)


class EmulatorFitter:
    """
    Nautilus nested sampler backed by a pre-trained DeltaSigma emulator
    (and optionally a w_gg emulator for joint fits).

    Emulator calls are ~μs, so a full Nautilus run completes in minutes.

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
    ds_obs, cov_inv, rp_obs : np.ndarray, optional
        Direct-array alternative to ``data_path``.
    param_names_ordered : list of str, optional
        Ordered list of HOD parameter names as stored in the emulator grid.
        Must match the column order of the param_grid used during training.
        Defaults to the order read from the emulator metadata.
    M1_fixed : float, default=13.0
        Fixed log10(M1) value.
    rp_min, rp_max : float, optional
        Scale cuts for DeltaSigma [Mpc/h].
    emulator_wgg_path : str, optional
        Path to the w_gg emulator ``.npz`` file. When provided, enables joint
        DeltaSigma + w_gg fitting.
    data_path_wgg : str, optional
        Path to .npz file with observed w_gg data.
    wgg_obs, cov_inv_wgg, rp_obs_wgg : np.ndarray, optional
        Direct-array alternative to ``data_path_wgg``.
    rp_min_wgg, rp_max_wgg : float, optional
        Scale cuts for w_gg [Mpc/h].
    param_config : dict, optional
        ``{name: (low, high)}`` for free params, ``{name: scalar}`` for fixed.
    max_fsat : float, optional
        f_sat hard prior: matches the rejection rule applied in
        ``generate_hod_parameter_grid()`` at grid build time. Requires
        ``hod_occupation``.
    Ac_fiducial : float, default=0.01
        Same Ac fiducial used at grid build time (compute_fsat_batched is
        invariant under joint (Ac, As) rescaling).
    hod_occupation : object, optional
        Pass ``halo.HOD`` so ``compute_fsat_batched`` is available for the
        f_sat truncation prior.
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
        # --- Custom priors ---
        param_config: Optional[dict] = None,
        # --- f_sat truncation prior (mirrors grid-time rejection) ---
        max_fsat: Optional[float] = None,
        Ac_fiducial: float = 0.01,
        hod_occupation=None,
    ):
        self.fit_case = FitCase(fit_case)
        self.M1_fixed = M1_fixed
        self.fixed_params_dict = {"M1": M1_fixed}
        self.rp_min = rp_min
        self.rp_max = rp_max

        self.max_fsat = max_fsat
        self.Ac_fiducial = Ac_fiducial
        self.hod_occupation = hod_occupation
        if max_fsat is not None and hod_occupation is None:
            raise ValueError(
                "max_fsat requires hod_occupation (pass halo.HOD, an Occupation "
                "instance) so f_sat can be evaluated via compute_fsat_batched."
            )

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
        """Load observed DeltaSigma data from .npz file."""
        data = np.load(data_path)

        for key in ("rp", "rp_centers"):
            if key in data:
                self.rp_obs = data[key]
                break
        else:
            raise KeyError("Data file must contain 'rp' or 'rp_centers'")

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

        for key in ("delta_sigma", "dsigma"):
            if key in data:
                self.ds_obs = data[key]
                break
        else:
            raise KeyError("Data file must contain 'delta_sigma' or 'dsigma'")

        for key in ("cov_delta_sigma", "cov_dsigma"):
            if key in data:
                self.cov = data[key]
                break
        else:
            raise KeyError("Data file must contain 'cov_delta_sigma' or 'cov_dsigma'")

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
        self._log_rp_emu = np.log(self.emulator_rp)
        self._log_rp_obs = np.log(self.rp_obs)

    def _load_data_wgg(self, data_path_wgg: str):
        """Load observed w_gg data from .npz file."""
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
        self._log_rp_emu_wgg = np.log(self.emulator_rp_wgg)
        self._log_rp_obs_wgg = np.log(self.rp_obs_wgg)

    def set_priors(self, priors, fixed_params=()):
        """Override free-parameter bounds and fixed values after construction."""
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
        """Log-likelihood using the emulator forward pass."""
        if isinstance(theta, dict):
            free_dict = dict(theta)
        else:
            free_dict = dict(zip(self.param_names, theta))

        # f_sat truncation prior — identical rejection criterion to the one
        # used in generate_hod_parameter_grid() at grid build time.
        if self.max_fsat is not None:
            try:
                params_arrays = {
                    name: np.array([value])
                    for name, value in {**self.fixed_params_dict, **free_dict}.items()
                }
                fsat = float(self.hod_occupation.compute_fsat_batched(
                    params_arrays, self.Ac_fiducial
                )[0])
            except Exception:
                return -1e100
            if not np.isfinite(fsat) or fsat > self.max_fsat:
                return -1e100

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

        log_ds_emu = np.log(ds_pred)
        log_ds_at_obs = np.array(
            Interpolator1D(self._log_rp_emu, log_ds_emu, method='cubic')(self._log_rp_obs)
        )
        ds_at_obs = np.exp(log_ds_at_obs)

        residual = ds_at_obs - self.ds_obs
        chi2 = residual @ self.cov_inv @ residual

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
        vectorized: bool = False,
        **nautilus_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the Nautilus nested sampler with the emulator likelihood.

        With ``vectorized=True`` the sampler uses the jit/vmap-batched
        likelihood from :meth:`build_batched_loglike` in a single process
        (``vectorized=True, pool=None``) — no fork, no JAX-after-fork
        deadlock; parallelism comes from XLA's own threading.
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

        if vectorized:
            likelihood, _ = self.build_batched_loglike()
            pool_arg = None
            # n_batch=128 matches the loglike's pad_to multiple: no padding waste
            vec_kwargs = dict(vectorized=True, n_batch=128)
        else:
            likelihood = _emulator_likelihood
            pool_arg = n_workers if n_workers > 1 else None
            vec_kwargs = {}

        try:
            sampler = nautilus.Sampler(
                prior,
                likelihood,
                n_live=n_live,
                filepath=filepath,
                pool=pool_arg,
                pass_dict=False,
                **vec_kwargs,
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
        """Return the MAP (maximum a-posteriori) parameter estimate."""
        idx_best = np.argmax(log_l)
        theta_best = points[idx_best]
        result = dict(zip(self.param_names, theta_best))
        result["M1"] = self.M1_fixed
        return result

    def get_posterior_summary(
        self, points: np.ndarray, weights: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute weighted posterior summary statistics."""
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
        """Save sampler results to .npz file."""
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


# ---------------------------------------------------------------------------
# TabulatedFitter — Nautilus sampler backed by TabulatedDeltaSigma (no emulator)
# ---------------------------------------------------------------------------

class TabulatedFitter(EmulatorFitter):
    """
    Nautilus sampler calling TabulatedDeltaSigma.predict() directly.

    Replaces the LHS grid + NN emulator: each likelihood call rescales
    (Ac, As) to ``target_ngal`` (same convention as the grid pipeline),
    evaluates the tabulated DeltaSigma (exact expectation of the MC forward
    model, ~0.2-0.5 s/call), and log-log interpolates onto the observed rp
    grid. Uses pure NumPy/SciPy in the likelihood so fork-based Nautilus
    pools are safe (no JAX in workers) — except the per-halo occupation,
    which goes through the halo model's JAX functions (works in practice,
    see project notes).

    Parameters
    ----------
    tabulated_ds : TabulatedDeltaSigma
        Predictor built from a cache with xi_gm tabulation.
    occupation_rescale : Occupation
        NON-assembly-bias Occupation used only for the (Ac, As) -> ngal
        rescaling (mirrors _full_rescaled_params in run_emulator_chains).
    target_ngal : float
        Galaxy number density the (Ac, As) pair is rescaled to.
    Other arguments are as in EmulatorFitter (data_path / arrays, rp cuts,
    param_config, max_fsat, ...). ``emulator_path`` is not used.
    """

    def __init__(
        self,
        tabulated_ds,
        occupation_rescale,
        target_ngal: float,
        fit_case: FitCase = FitCase.STANDARD_NFW,
        data_path: str = "",
        ds_obs: Optional[np.ndarray] = None,
        cov_inv: Optional[np.ndarray] = None,
        rp_obs: Optional[np.ndarray] = None,
        M1_fixed: float = 13.0,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        param_config: Optional[dict] = None,
        max_fsat: Optional[float] = None,
        Ac_fiducial: float = 0.01,
        hod_occupation=None,
    ):
        self.fit_case = FitCase(fit_case)
        self.M1_fixed = M1_fixed
        self.fixed_params_dict = {"M1": M1_fixed}
        self.rp_min = rp_min
        self.rp_max = rp_max

        self.max_fsat = max_fsat
        self.Ac_fiducial = Ac_fiducial
        self.hod_occupation = hod_occupation
        if max_fsat is not None and hod_occupation is None:
            raise ValueError("max_fsat requires hod_occupation.")

        self.tab = tabulated_ds
        self.occupation_rescale = occupation_rescale
        self.target_ngal = target_ngal
        # Model rp grid = the cache binning the tabulated profiles live on
        self.emulator_rp = np.asarray(tabulated_ds.cache.rp_centers)
        self.emulator_param_order = []   # no emulator

        self.free_params = _get_free_params(self.fit_case)
        self.param_names = [p[0] for p in self.free_params]
        self.n_params = len(self.free_params)

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
                self.rp_obs = self.rp_obs[mask]
                self.ds_obs = self.ds_obs[mask]
                self.cov_inv = self.cov_inv[np.ix_(mask, mask)]
            self.n_bins = len(self.ds_obs)
        else:
            raise ValueError("Provide data_path or (ds_obs, cov_inv, rp_obs).")

        self._setup_interp()
        self._fit_wgg = False

        if param_config is not None:
            priors, fixed = _parse_param_config(param_config)
            self.set_priors(priors, fixed)

    def set_priors(self, priors, fixed_params=()):
        """As EmulatorFitter.set_priors but without emulator-name filtering."""
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
                warnings.warn(f"set_priors: unknown parameter '{name}'; ignoring.")
            else:
                active.append((name, float(a), float(b), kind))

        self.free_params = active
        self.param_names = [p[0] for p in active]
        self.n_params = len(active)

    def full_params(self, free_dict):
        """Merge free + fixed and rescale (Ac, As) to target_ngal."""
        from HOD_NRV.HOD_numerical.HOD_models import rescale_Ac_to_target_ngal
        merged = {**self.fixed_params_dict, **free_dict}
        merged.pop("Ac", None)
        Ac, As = rescale_Ac_to_target_ngal(
            self.occupation_rescale, merged, self.target_ngal,
            Ac_fiducial=self.Ac_fiducial)
        return {**merged, "Ac": float(np.asarray(Ac).ravel()[0]),
                "As": float(np.asarray(As).ravel()[0])}

    def predict_at_obs(self, free_dict, rp_eval=None):
        """Tabulated DeltaSigma log-log interpolated onto rp_eval (or rp_obs)."""
        from scipy.interpolate import CubicSpline
        full = self.full_params(free_dict)
        _, ds_pred, _ = self.tab.predict(full)
        if not np.all(np.isfinite(ds_pred)):
            raise ValueError("non-finite tabulated prediction")
        log_rp = np.log(rp_eval) if rp_eval is not None else self._log_rp_obs
        spl = CubicSpline(self._log_rp_emu, np.log(np.maximum(ds_pred, 1e-30)))
        return np.exp(spl(log_rp))

    def log_likelihood(self, theta) -> float:
        if isinstance(theta, dict):
            free_dict = dict(theta)
        else:
            free_dict = dict(zip(self.param_names, theta))

        if self.max_fsat is not None:
            try:
                params_arrays = {
                    name: np.array([value])
                    for name, value in {**self.fixed_params_dict, **free_dict}.items()
                }
                fsat = float(self.hod_occupation.compute_fsat_batched(
                    params_arrays, self.Ac_fiducial)[0])
            except Exception:
                return -1e100
            if not np.isfinite(fsat) or fsat > self.max_fsat:
                return -1e100

        try:
            ds_at_obs = self.predict_at_obs(free_dict)
        except Exception:
            return -1e100
        if not np.all(np.isfinite(ds_at_obs)):
            return -1e100

        residual = ds_at_obs - self.ds_obs
        return -0.5 * float(residual @ self.cov_inv @ residual)

    def build_batched_loglike(self, pad_to: int = 32, vmap_chunk: int = 8):
        """Build a jit-compiled batched log-likelihood for ``vectorized=True``.

        Returns ``(loglike_fn, free_names)``; ``loglike_fn`` maps an
        ``(n, n_free)`` array (columns in ``free_names`` = nautilus prior
        order) to an ``(n,)`` numpy array of log-likelihoods. It is the
        batched twin of :meth:`log_likelihood`: the (Ac, As) -> target_ngal
        rescale, the tabulated forward pass and the log-log cubic
        interpolation onto the observed rp grid are all reproduced in pure
        jnp. The scipy cubic-spline stages (ngal GL integral, rp
        interpolation) are linear in their inputs and precomputed as
        matrices by probing the exact NumPy code with basis vectors.
        Points run through ``jax.lax.map`` over ``vmap``-ed chunks of
        ``vmap_chunk`` points (caps the transient satellite-convolution
        tensor at ~vmap_chunk * 0.25 GB while giving XLA large ops to
        thread) and batches are padded to a multiple of ``pad_to`` to
        avoid shape-driven recompilation.
        """
        import jax
        import jax.numpy as jnp
        from scipy.interpolate import CubicSpline
        from HOD_NRV.HOD_numerical.HOD_models import (
            build_occupation_fn_jax, ELG_satellite_cutoff, HOD_satellite,
        )
        from HOD_NRV.utilsf.utils_functions import x_legendre, w_legendre

        predict_fn = self.tab.make_predict_jax()

        # ── ngal rescale: GL integral of a CubicSpline is linear in the
        # integrand values on logM_bins — probe it once with the identity ──
        occ_r = self.occupation_rescale
        logMb = np.asarray(occ_r.logM_bins)
        a, b = logMb.min(), logMb.max()
        gl_nodes = 0.5 * ((b - a) * np.asarray(x_legendre) + (a + b))
        basis_at_nodes = CubicSpline(logMb, np.eye(len(logMb)))(gl_nodes)
        W_ngal = 0.5 * (b - a) * (np.asarray(w_legendre) @ basis_at_nodes)
        occ_r_fn = build_occupation_fn_jax(occ_r)
        j_logMb = jnp.asarray(logMb)
        j_mf = jnp.asarray(np.asarray(occ_r.mass_function))
        j_W_ngal = jnp.asarray(W_ngal)

        # ── obs-grid interpolation: log-log CubicSpline is linear in
        # log(ds) on the cache rp grid ──
        W_obs = CubicSpline(
            self._log_rp_emu, np.eye(len(self._log_rp_emu)))(self._log_rp_obs)
        j_W_obs = jnp.asarray(W_obs)
        j_ds_obs = jnp.asarray(self.ds_obs)
        j_cov_inv = jnp.asarray(self.cov_inv)

        free_names = list(self.param_names)
        fixed = {k: float(v) for k, v in self.fixed_params_dict.items()}
        target_ngal = float(self.target_ngal)
        Ac_fid = float(self.Ac_fiducial)

        # ── optional max_fsat gate: twin of compute_fsat_batched (trapezoid,
        # no conformity/AB corrections, fiducial Ac, unrescaled As) ──
        max_fsat = self.max_fsat
        if max_fsat is not None:
            hocc = self.hod_occupation
            fs_cen_fn, fs_cen_names = hocc.HOD_central, list(hocc.central_params)
            if hocc.elg_satellite:
                fs_sat_fn = ELG_satellite_cutoff
                fs_sat_names = ["As", "M1", "alpha", "Mcut", "Mmax"]
            else:
                fs_sat_fn = HOD_satellite
                fs_sat_names = ["As", "Mmin", "M1", "alpha", "kappa"]
            j_logMb_fs = jnp.asarray(np.asarray(hocc.logM_bins))
            j_mf_fs = jnp.asarray(np.asarray(hocc.mass_function))

        def _loglike_one(theta):
            free = {name: theta[i] for i, name in enumerate(free_names)}
            merged = {**fixed, **free}
            merged.pop("Ac", None)

            if max_fsat is not None:
                fs_params = {**merged, "Ac": Ac_fid}
                pC = fs_cen_fn(j_logMb_fs, *[fs_params[n] for n in fs_cen_names])
                pS = fs_sat_fn(j_logMb_fs, *[fs_params[n] for n in fs_sat_names])
                fsat_gate = (jnp.trapezoid(j_mf_fs * pS, j_logMb_fs)
                             / jnp.trapezoid(j_mf_fs * (pC + pS), j_logMb_fs))

            # (Ac, As) -> target_ngal rescale (mass-function ngal, no AB)
            probC, probS = occ_r_fn(j_logMb, {**merged, "Ac": Ac_fid})
            ngal_fid = j_W_ngal @ (j_mf * (probC + probS))
            rf = target_ngal / ngal_fid
            full = {**merged, "Ac": Ac_fid * rf, "As": merged["As"] * rf}

            ds, ngal, fsat = predict_fn(full)
            log_ds = jnp.log(jnp.maximum(ds, 1e-30))
            ds_at_obs = jnp.exp(j_W_obs @ log_ds)
            resid = ds_at_obs - j_ds_obs
            logL = -0.5 * (resid @ j_cov_inv @ resid)

            good = jnp.isfinite(ds).all() & jnp.isfinite(logL)
            if max_fsat is not None:
                good = good & jnp.isfinite(fsat_gate) & (fsat_gate <= max_fsat)
            return jnp.where(good, logL, -1e100)

        if pad_to % vmap_chunk:
            raise ValueError("pad_to must be a multiple of vmap_chunk.")
        _loglike_chunk = jax.vmap(_loglike_one)
        compiled = jax.jit(lambda pts: jax.lax.map(
            _loglike_chunk,
            pts.reshape(-1, vmap_chunk, pts.shape[-1])).ravel())

        def loglike_fn(points):
            points = np.atleast_2d(np.asarray(points, dtype=np.float64))
            n = len(points)
            n_pad = (-n) % pad_to
            if n_pad:
                points = np.vstack([points, np.repeat(points[-1:], n_pad, 0)])
            return np.asarray(compiled(jnp.asarray(points)))[:n]

        return loglike_fn, free_names
