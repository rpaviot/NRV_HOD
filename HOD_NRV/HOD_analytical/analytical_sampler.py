"""
Analytical HOD fitter for ELG-style models (ELG_GHOD, ELG_SFR, LRG).

Fixes the galaxy number density by deriving (Ac, As) from the other HOD
parameters, mirroring the convention used in HOD_numerical / EmulatorFitter:
joint (Ac, As) rescaling preserves the Ac/As ratio and leaves DeltaSigma /
w_gg invariant, so the constraint is enforced by a single scalar rescale.

Free parameters (Ac is derived):
- ELG_GHOD: As, log10Mmin, sig_M, log10M1, alpha, kappa
- ELG_SFR : As, log10Mmin, sig_M, gamma, log10M1, alpha, kappa
- LRG     : As, log10Mmin, sig_M, log10M1, alpha, kappa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from .halo_model import HaloModel

try:
    from nautilus import Prior, Sampler
    HAS_NAUTILUS = True
except ImportError:
    HAS_NAUTILUS = False

try:
    from iminuit import Minuit
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False


# ---------------------------------------------------------------------------
# Ac / As rescaling helpers
# ---------------------------------------------------------------------------

def compute_ngal_with_fiducial_Ac(
    halo_model: HaloModel,
    params: Dict[str, float],
    Ac_fiducial: float = 1.0,
) -> float:
    """ngal evaluated at (Ac_fiducial, As) with all other params from `params`."""
    p = {**params, "Ac": Ac_fiducial}
    return float(np.asarray(halo_model.ngal(p)).item())


def rescale_Ac_to_target_ngal(
    halo_model: HaloModel,
    params: Dict[str, float],
    target_ngal: float,
    Ac_fiducial: float = 1.0,
) -> Tuple[float, float]:
    """Joint (Ac, As) rescale to hit `target_ngal`. Preserves Ac/As ratio."""
    n_fid = compute_ngal_with_fiducial_Ac(halo_model, params, Ac_fiducial)
    if not np.isfinite(n_fid) or n_fid <= 0.0:
        return np.nan, np.nan
    f = target_ngal / n_fid
    return Ac_fiducial * f, float(params["As"]) * f


# ---------------------------------------------------------------------------
# Parameter spec
# ---------------------------------------------------------------------------

# Default uniform priors for the free parameters (Ac excluded — derived).
_DEFAULT_PRIORS: Dict[str, Tuple[float, float]] = {
    "As":         (0.001, 0.1),
    "log10Mmin":  (11.0, 14.0),
    "sig_M":      (0.05, 2.0),
    "gamma":      (0.0, 10.0),
    "log10M1":    (12.0, 15.0),
    "alpha":      (0.3, 2.0),
    "kappa":      (0.1, 3.0),
}

_HOD_FREE_PARAMS: Dict[str, List[str]] = {
    "LRG":      ["As", "log10Mmin", "sig_M", "log10M1", "alpha", "kappa"],
    "ELG_GHOD": ["As", "log10Mmin", "sig_M", "log10M1", "alpha", "kappa"],
    "ELG_SFR":  ["As", "log10Mmin", "sig_M", "gamma", "log10M1", "alpha", "kappa"],
    "ELG_MHMQ": ["As", "log10Mmin", "sig_M", "gamma", "log10M1", "alpha", "kappa"],
}

# Concentration rescaling parameters (centrals / satellites). Fixed at 1.0 by
# default; declare them in ``param_config`` to make them free or fix elsewhere.
_PROFILE_PARAMS: List[str] = ["f_c", "f_s"]
_PROFILE_DEFAULT_FIXED: Dict[str, float] = {"f_c": 1.0, "f_s": 1.0}
_PROFILE_DEFAULT_PRIOR: Dict[str, Tuple[float, float]] = {
    "f_c": (0.4, 1.0),
    "f_s": (0.4, 1.0),
}


def _parse_param_config(
    hod_type: str,
    param_config: Optional[Dict[str, Any]],
) -> Tuple[List[Tuple[str, float, float, str]], Dict[str, float]]:
    """Split user dict into (priors, fixed_params).

    Value formats:
      (low, high) / [low, high]            -> uniform free
      (mean, std, 'gaussian')              -> Gaussian free
      scalar                               -> fixed
    """
    free_names = _HOD_FREE_PARAMS[hod_type]
    cfg: Dict[str, Any] = {n: _DEFAULT_PRIORS[n] for n in free_names}
    # f_c / f_s default to fixed at 1.0 (no rescaling); user can override.
    for n in _PROFILE_PARAMS:
        cfg[n] = _PROFILE_DEFAULT_FIXED[n]
    if param_config:
        for name, value in param_config.items():
            if name == "Ac":
                raise ValueError(
                    "Ac is derived from target_ngal; do not pass it in param_config."
                )
            cfg[name] = value

    priors: List[Tuple[str, float, float, str]] = []
    fixed: Dict[str, float] = {}
    for name, value in cfg.items():
        if isinstance(value, (tuple, list)):
            strs = [v for v in value if isinstance(v, str)]
            nums = [v for v in value if not isinstance(v, str)]
            if strs:
                if strs[0].lower() != "gaussian" or len(nums) != 2:
                    raise ValueError(f"Invalid Gaussian prior for {name}: {value!r}")
                priors.append((name, float(nums[0]), float(nums[1]), "gaussian"))
            elif len(value) == 2:
                priors.append((name, float(value[0]), float(value[1]), "uniform"))
            else:
                raise ValueError(f"Invalid prior spec for {name}: {value!r}")
        else:
            fixed[name] = float(value)
    return priors, fixed


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    best_fit: Dict[str, float]
    derived: Dict[str, float]      # Ac, As_rescaled, f_sat, etc.
    chi2: float
    ndof: int
    sampler: Optional[Any] = None  # nautilus sampler if used
    posterior: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

class AnalyticalHODFitter:
    """Fit DeltaSigma (and optionally w_gg) with the analytical HOD at fixed n_gal.

    Parameters
    ----------
    halo_model : HaloModel
        Pre-built single-redshift HaloModel. Its ``hod_type`` selects the HOD
        family (LRG / ELG_GHOD / ELG_SFR).
    target_ngal : float
        Number density the (Ac, As) rescale will hit at every likelihood call.
        Units must match ``halo_model.ngal()`` (h^3/Mpc^3 if units_per_h=True).
    rp_obs, ds_obs, cov_ds : array
        DeltaSigma data vector and full covariance (same units as model output).
    rp_obs_wgg, wgg_obs, cov_wgg : array, optional
        If given, w_gg is added to the joint likelihood.
    rp_min, rp_max, rp_min_wgg, rp_max_wgg : float, optional
        Scale cuts (inclusive).
    Ac_fiducial : float, default 1.0
        Fiducial Ac at which n_gal is evaluated for the rescale step.
    param_config : dict, optional
        Per-parameter overrides — see ``_parse_param_config`` docstring.
    ds_method : str, default 'direct'
        DeltaSigma method passed through to ``HaloModel.DeltaSigma``.
    """

    def __init__(
        self,
        halo_model: HaloModel,
        target_ngal: float,
        rp_obs: np.ndarray,
        ds_obs: np.ndarray,
        cov_ds: np.ndarray,
        rp_obs_wgg: Optional[np.ndarray] = None,
        wgg_obs: Optional[np.ndarray] = None,
        cov_wgg: Optional[np.ndarray] = None,
        rp_min: Optional[float] = None,
        rp_max: Optional[float] = None,
        rp_min_wgg: Optional[float] = None,
        rp_max_wgg: Optional[float] = None,
        Ac_fiducial: float = 1.0,
        param_config: Optional[Dict[str, Any]] = None,
        ds_method: str = "direct",
        verbose: bool = True,
    ):
        if not halo_model.is_single_z:
            raise ValueError(
                "AnalyticalHODFitter expects a single-redshift HaloModel."
            )

        self.halo_model = halo_model
        hod_type = halo_model.hod_type.upper()
        if hod_type not in _HOD_FREE_PARAMS:
            raise ValueError(f"Unsupported HOD type for this fitter: {hod_type}")
        self.hod_type = hod_type

        self.target_ngal = float(target_ngal)
        self.Ac_fiducial = float(Ac_fiducial)
        self.ds_method = ds_method
        self.verbose = verbose

        # DeltaSigma data + scale cut
        rp_obs = np.asarray(rp_obs)
        ds_obs = np.asarray(ds_obs)
        cov_ds = np.asarray(cov_ds)
        ds_mask = np.ones_like(rp_obs, dtype=bool)
        if rp_min is not None:
            ds_mask &= rp_obs >= rp_min
        if rp_max is not None:
            ds_mask &= rp_obs <= rp_max
        self.rp_ds = rp_obs[ds_mask]
        self.ds_obs = ds_obs[ds_mask]
        self.cov_ds_inv = np.linalg.inv(cov_ds[np.ix_(ds_mask, ds_mask)])
        self.rp_bins_ds = self._infer_bin_edges(self.rp_ds)

        # Optional w_gg
        self.fit_wgg = wgg_obs is not None
        if self.fit_wgg:
            rp_obs_wgg = np.asarray(rp_obs_wgg)
            wgg_obs = np.asarray(wgg_obs)
            cov_wgg = np.asarray(cov_wgg)
            mask = np.ones_like(rp_obs_wgg, dtype=bool)
            if rp_min_wgg is not None:
                mask &= rp_obs_wgg >= rp_min_wgg
            if rp_max_wgg is not None:
                mask &= rp_obs_wgg <= rp_max_wgg
            self.rp_wgg = rp_obs_wgg[mask]
            self.wgg_obs = wgg_obs[mask]
            self.cov_wgg_inv = np.linalg.inv(cov_wgg[np.ix_(mask, mask)])
            self.rp_bins_wgg = self._infer_bin_edges(self.rp_wgg)

        # Priors
        self.priors, self.fixed = _parse_param_config(self.hod_type, param_config)
        self.free_names = [p[0] for p in self.priors]

        self.n_data = len(self.ds_obs) + (len(self.wgg_obs) if self.fit_wgg else 0)

        if verbose:
            self._print_summary()

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _infer_bin_edges(rp: np.ndarray) -> np.ndarray:
        """Geometric mid-points for log-spaced rp centers."""
        log_rp = np.log10(rp)
        edges = np.empty(len(rp) + 1)
        edges[1:-1] = 10 ** (0.5 * (log_rp[:-1] + log_rp[1:]))
        edges[0] = rp[0] ** 2 / edges[1]
        edges[-1] = rp[-1] ** 2 / edges[-2]
        return edges

    def _print_summary(self):
        print(f"AnalyticalHODFitter ({self.hod_type})")
        print(f"  target n_gal = {self.target_ngal:.3e}")
        print(f"  Ac_fiducial  = {self.Ac_fiducial}")
        print(f"  free params  : {self.free_names}")
        if self.fixed:
            print(f"  fixed params : {self.fixed}")
        print(f"  n_data       = {self.n_data}  (DS={len(self.ds_obs)}"
              + (f", w_gg={len(self.wgg_obs)}" if self.fit_wgg else "") + ")")

    def _build_params(
        self, free_values: Dict[str, float]
    ) -> Optional[Tuple[Dict[str, float], float, float]]:
        """Apply fixed + free values, then derive (Ac, As) from target_ngal.

        Returns (hod_params, f_c, f_s), or None if the rescale fails.
        f_c / f_s are stripped from the HOD-param dict and applied via
        ``HaloModel.update_f`` before observables are computed (the NFW
        concentration rescaling is not a parameter of the occupation function).
        """
        merged = {**self.fixed, **free_values}
        f_c = float(merged.pop("f_c", 1.0))
        f_s = float(merged.pop("f_s", 1.0))
        # f_c / f_s rescale the NFW profile inside Pgg/Pgm; they leave the
        # occupation N_c/N_s — and therefore n_gal — invariant. The (Ac, As)
        # rescale below is independent of (f_c, f_s).
        for n in _HOD_FREE_PARAMS[self.hod_type]:
            if n not in merged:
                raise KeyError(f"Missing HOD parameter {n!r}")

        Ac, As = rescale_Ac_to_target_ngal(
            self.halo_model, merged, self.target_ngal, self.Ac_fiducial
        )
        if not (np.isfinite(Ac) and np.isfinite(As)) or Ac <= 0.0 or As <= 0.0:
            return None
        merged["Ac"] = Ac
        merged["As"] = As
        return merged, f_c, f_s

    def _model_observables(
        self, full_params: Dict[str, float], f_c: float, f_s: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.halo_model.update_f(f_c=f_c, f_s=f_s)
        self.halo_model.set_hod_params(full_params)
        _, ds = self.halo_model.DeltaSigma(
            self.rp_ds, rp_bins=self.rp_bins_ds, method=self.ds_method
        )
        wgg = None
        if self.fit_wgg:
            _, wgg = self.halo_model.wgg(self.rp_wgg, rp_bins=self.rp_bins_wgg)
        return np.asarray(ds), (None if wgg is None else np.asarray(wgg))

    # ----- public log-likelihood -------------------------------------------

    def log_likelihood(self, free_values: Dict[str, float]) -> float:
        built = self._build_params(free_values)
        if built is None:
            return -1e30
        full, f_c, f_s = built
        try:
            ds_m, wgg_m = self._model_observables(full, f_c, f_s)
        except Exception:
            return -1e30
        if not np.all(np.isfinite(ds_m)):
            return -1e30

        r = ds_m - self.ds_obs
        chi2 = float(r @ self.cov_ds_inv @ r)
        if self.fit_wgg:
            if not np.all(np.isfinite(wgg_m)):
                return -1e30
            r_w = wgg_m - self.wgg_obs
            chi2 += float(r_w @ self.cov_wgg_inv @ r_w)

        # Gaussian-prior contribution
        for name, a, b, kind in self.priors:
            if kind == "gaussian":
                chi2 += ((free_values[name] - a) / b) ** 2
        return -0.5 * chi2

    def chi2(self, free_values: Dict[str, float]) -> float:
        return -2.0 * self.log_likelihood(free_values)

    def derived_quantities(self, free_values: Dict[str, float]) -> Dict[str, float]:
        built = self._build_params(free_values)
        if built is None:
            return {"Ac": np.nan, "As": np.nan, "f_c": np.nan, "f_s": np.nan,
                    "f_sat": np.nan}
        full, f_c, f_s = built
        self.halo_model.update_f(f_c=f_c, f_s=f_s)
        self.halo_model.set_hod_params(full)
        return {
            "Ac": full["Ac"],
            "As": full["As"],
            "f_c": f_c,
            "f_s": f_s,
            "f_sat": float(np.asarray(self.halo_model.satellite_fraction()).item()),
        }

    # ----- iminuit ----------------------------------------------------------

    def minimize(
        self,
        start: Optional[Dict[str, float]] = None,
        errordef: float = 1.0,
        migrad_kwargs: Optional[Dict] = None,
    ) -> FitResult:
        if not HAS_IMINUIT:
            raise ImportError("iminuit not installed")

        start = start or {}
        init: Dict[str, float] = {}
        limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for name, a, b, kind in self.priors:
            if name in start:
                init[name] = float(start[name])
            elif kind == "uniform":
                init[name] = 0.5 * (a + b)
            else:  # gaussian
                init[name] = a
            limits[name] = (a, b) if kind == "uniform" else (None, None)

        names = self.free_names

        def cost(*values):
            return self.chi2(dict(zip(names, values)))

        cost.errordef = errordef
        m = Minuit(cost, name=names, **{n: init[n] for n in names})
        for n in names:
            m.limits[n] = limits[n]
        m.migrad(**(migrad_kwargs or {}))

        best = {n: float(m.values[n]) for n in names}
        return FitResult(
            best_fit=best,
            derived=self.derived_quantities(best),
            chi2=float(m.fval),
            ndof=self.n_data - len(names),
            sampler=m,
        )

    # ----- Nautilus ---------------------------------------------------------

    def _nautilus_prior(self) -> "Prior":
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler not installed")
        prior = Prior()
        for name, a, b, kind in self.priors:
            if kind == "uniform":
                prior.add_parameter(name, dist=(a, b))
            else:
                prior.add_parameter(name, dist=norm(loc=a, scale=b))
        return prior

    def run(
        self,
        n_live: int = 500,
        n_eff: int = 5000,
        n_workers: int = 1,
        verbose: bool = True,
        **sampler_kwargs,
    ) -> FitResult:
        if not HAS_NAUTILUS:
            raise ImportError("nautilus-sampler not installed")

        prior = self._nautilus_prior()
        names = self.free_names

        def log_l(param_dict):
            return self.log_likelihood({n: float(param_dict[n]) for n in names})

        sampler = Sampler(
            prior, log_l, n_live=n_live, n_dim=len(names),
            pool=n_workers, **sampler_kwargs,
        )
        sampler.run(verbose=verbose, n_eff=n_eff)

        points, log_w, log_l_vals = sampler.posterior()
        i_max = int(np.argmax(log_l_vals))
        best = {n: float(points[i_max, j]) for j, n in enumerate(names)}

        posterior = {n: points[:, j] for j, n in enumerate(names)}
        posterior["log_w"] = log_w
        posterior["log_l"] = log_l_vals

        return FitResult(
            best_fit=best,
            derived=self.derived_quantities(best),
            chi2=-2.0 * float(log_l_vals[i_max]),
            ndof=self.n_data - len(names),
            sampler=sampler,
            posterior=posterior,
        )


__all__ = [
    "AnalyticalHODFitter",
    "FitResult",
    "compute_ngal_with_fiducial_Ac",
    "rescale_Ac_to_target_ngal",
]
