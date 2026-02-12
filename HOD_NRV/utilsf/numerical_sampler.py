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

from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal


class FitCase(IntEnum):
    STANDARD_NFW = 1
    EXTENDED_PROFILE = 2
    CONFORMITY = 3


# Parameter definitions per case: (name, low, high)
_BASE_PARAMS = [
    ("ratio_As_Ac", 0.1, 3.0),
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
    halo : HaloOccupation
        Initialized HaloOccupation instance with particle data and precomputed cache.
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
    """

    def __init__(
        self,
        halo,
        precomputed_cache,
        fit_case: FitCase,
        data_path: str,
        target_ngal: float = 2.3e-4,
        n_realizations: int = 5,
        galaxy_fraction: float = 0.1,
        M1_fixed: float = 13.0,
        rp_bins=None,
        base_seed: int = 1000,
    ):
        self.halo = halo
        self.precomputed_cache = precomputed_cache
        self.fit_case = FitCase(fit_case)
        self.target_ngal = target_ngal
        self.n_realizations = n_realizations
        self.galaxy_fraction = galaxy_fraction
        self.M1_fixed = M1_fixed
        self.base_seed = base_seed

        # Configure HOD model
        use_conformity = (self.fit_case == FitCase.CONFORMITY)
        self.halo.set_halo_model("ELG_mHMQ", conformity=use_conformity)

        # Free parameters for this case
        self.free_params = _get_free_params(self.fit_case)
        self.param_names = [p[0] for p in self.free_params]
        self.n_params = len(self.free_params)

        # Load observed data
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
        for key in ("covariance", "cov"):
            if key in data:
                self.cov = data[key]
                break
        else:
            raise KeyError("Data file must contain 'covariance' or 'cov'")

        self.cov_inv = np.linalg.inv(self.cov)
        self.n_bins = len(self.ds_obs)

    def _derive_Ac_As(self, ratio_As_Ac: float, params_no_AcAs: dict) -> Tuple[float, float]:
        """Derive Ac and As from the ratio and target ngal.

        Steps:
        1. Set As_fid = ratio_As_Ac * Ac_fid (with Ac_fid = 1.0)
        2. Call rescale_Ac_to_target_ngal → Ac_new, As_new
        3. The ratio As/Ac is preserved by the rescaling.

        Returns (Ac_new, As_new).
        """
        Ac_fid = 1.0
        As_fid = ratio_As_Ac * Ac_fid

        params_for_rescale = params_no_AcAs.copy()
        params_for_rescale["As"] = As_fid

        Ac_new, As_new = rescale_Ac_to_target_ngal(
            self.halo.HOD, params_for_rescale, self.target_ngal, Ac_fiducial=Ac_fid
        )
        return Ac_new, As_new

    def _build_params(self, theta: np.ndarray) -> Optional[Dict[str, float]]:
        """Convert flat parameter vector to full HOD parameter dict.

        Returns None if the derived parameters are unphysical (e.g. Ac <= 0).
        """
        free_dict = dict(zip(self.param_names, theta))

        ratio_As_Ac = free_dict.pop("ratio_As_Ac")

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
            Ac, As = self._derive_Ac_As(ratio_As_Ac, params_no_AcAs)
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

        prior = nautilus.Prior()
        for name, low, high in self.free_params:
            prior.add_parameter(name, dist=(low, high))

        sampler = nautilus.Sampler(
            prior,
            self.log_likelihood,
            n_live=n_live,
            filepath=filepath,
        )

        sampler.run(n_eff=n_eff, verbose=verbose, **nautilus_kwargs)

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
