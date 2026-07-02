"""
TabCorr-style tabulated projected clustering wp_gg (Zheng & Guo 2016;
Lange et al. 2025 arXiv:2512.15962 Sect. 3.2).

wgg is quadratic in the halo occupation, so unlike DeltaSigma the
tabulation stores cross-correlations wp_ij(rp) between all pairs of
(logM [, fI]) halo bins, measured once with pycorr on redshift-space
halo-center positions (Kaiser RSD and halo exclusion are therefore
built in). A prediction is then:

    wp_gg = [ Sum_ij C_i C_j wp_ij                          (cen-cen)
            + 2 Sum_ij C_i S_j (wp_ij * K_mj)               (cen-sat)
            + Sum_ij S_i S_j (wp_ij * K_mi * K_mj) ] / N^2  (sat-sat)
            + wp_1h                                          (same halo)

where C_i/S_i are per-bin sums of the per-halo occupations <N_cen>/<N_sat>
(exact, assembly bias included), K_m is the analytic transverse
satellite-offset kernel of mass bin m (same quadrature as
TabulatedDeltaSigma — mirrors NFW_jax sampling, so arbitrary f_exp/tau/
lambda_NFW are exact, no profile-parameter interpolation), and wp_1h is
the analytic same-halo pair term with Poisson <Ns(Ns-1)> = <Ns>^2 and
conformity handled exactly.

LOS satellite offsets and Fingers-of-God drop out of wp up to pi_max
edge leakage (pairs pushed across |pi| = pi_max), which is negligible
for pi_max >~ 50 Mpc/h — validated in cross_check_tabulated.py --wgg.

Not supported: triaxial satellite profiles, subhalo placement.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .standard_two_point_calculator import compute_corr
from .halo_center_lensing import (
    build_tabulation_bins, satellite_offset_nodes, satellite_radial_nodes,
)


def refine_rp_edges(rp_bins: np.ndarray, n_sub: int = 3,
                    r_lo: float = 0.02, r_hi: float = 58.0) -> np.ndarray:
    """
    Build a tabulation rp grid nested inside the analysis binning rp_bins.

    Each analysis bin is split into n_sub geometric sub-bins, with padding
    bins below r_lo and above r_hi so the satellite offset convolution has
    support beyond the analysis range. Nesting makes the cen-cen term
    aggregate to exactly what pycorr measures in rp_bins (wp combines
    across fine rp bins with annulus-area Delta(rp^2) weights, since
    RR is separable in (rp, pi)).
    """
    parts = [np.geomspace(r_lo, rp_bins[0], 4)[:-1]]
    for k in range(len(rp_bins) - 1):
        parts.append(np.geomspace(rp_bins[k], rp_bins[k + 1], n_sub + 1)[:-1])
    parts.append(np.geomspace(rp_bins[-1], r_hi, 3))
    return np.concatenate(parts)


class WggTabulation:
    """Container for the wp_ij(rp) cross-correlation table (npz persistence)."""

    def __init__(self, wp_ij, rp_edges, pi_bins, bin_logM_edges,
                 bin_fI_edges=None, bin_counts=None, Lbox=None, rsd_axis='z'):
        self.wp_ij = np.asarray(wp_ij)                    # (n_bins, n_bins, n_rp)
        self.rp_edges = np.asarray(rp_edges)
        self.rp_centers = np.sqrt(rp_edges[:-1] * rp_edges[1:])
        self.pi_bins = np.asarray(pi_bins)
        self.bin_logM_edges = np.asarray(bin_logM_edges)
        self.bin_fI_edges = None if bin_fI_edges is None else np.asarray(bin_fI_edges)
        self.bin_counts = None if bin_counts is None else np.asarray(bin_counts)
        self.Lbox = Lbox
        self.rsd_axis = rsd_axis

    def save(self, path: str) -> None:
        data = dict(wp_ij=self.wp_ij, rp_edges=self.rp_edges, pi_bins=self.pi_bins,
                    bin_logM_edges=self.bin_logM_edges, Lbox=self.Lbox,
                    rsd_axis=self.rsd_axis)
        if self.bin_fI_edges is not None:
            data['bin_fI_edges'] = self.bin_fI_edges
        if self.bin_counts is not None:
            data['bin_counts'] = self.bin_counts
        np.savez_compressed(path, **data)
        print(f"Saved WggTabulation to {path} ({self.wp_ij.shape[0]} bins, "
              f"{self.wp_ij.shape[2]} rp bins)")

    @classmethod
    def load(cls, path: str) -> 'WggTabulation':
        d = np.load(path, allow_pickle=True)
        obj = cls(d['wp_ij'], d['rp_edges'], d['pi_bins'], d['bin_logM_edges'],
                  bin_fI_edges=d['bin_fI_edges'] if 'bin_fI_edges' in d else None,
                  bin_counts=d['bin_counts'] if 'bin_counts' in d else None,
                  Lbox=float(d['Lbox']), rsd_axis=str(d['rsd_axis']))
        print(f"Loaded WggTabulation from {path} ({obj.wp_ij.shape[0]} bins)")
        return obj


def precompute_wgg_tabulation(
    halo_positions_rsd: np.ndarray,
    halo_logM: np.ndarray,
    Lbox: float,
    rsd_axis: str,
    halo_fI: Optional[np.ndarray] = None,
    n_logM_bins: int = 16,
    n_fI_bins: int = 4,
    rp_edges: Optional[np.ndarray] = None,
    pi_bins: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> WggTabulation:
    """
    Tabulate wp_ij(rp) between all (logM [, fI]) halo-bin pairs with pycorr.

    halo_positions_rsd must already be in redshift space (halo centers
    shifted by v_los * rsd_factor, periodic-wrapped) so that Kaiser RSD
    is captured; centrals inherit exactly this, satellites additionally
    get intra-halo offsets/velocities whose LOS parts integrate out of wp.

    Cost: n_bins*(n_bins+1)/2 pycorr cross pair counts. The rp grid should
    extend beyond the analysis range on both sides (satellite offset
    convolution needs wp at rp +- ~3 Rvir).
    """
    if rp_edges is None:
        raise ValueError("Pass rp_edges=refine_rp_edges(<analysis rp_bins>) so "
                         "predictions aggregate exactly onto the analysis binning.")
    if pi_bins is None:
        pi_bins = np.linspace(0.0, 60.0, 61)

    bin_index, logM_edges, fI_edges = build_tabulation_bins(
        halo_logM, halo_fI, n_logM_bins, n_fI_bins)
    n_bins = n_logM_bins * (n_fI_bins if halo_fI is not None else 1)
    n_rp = len(rp_edges) - 1

    pos_by_bin = [np.ascontiguousarray(halo_positions_rsd[bin_index == b])
                  for b in range(n_bins)]
    counts = np.array([len(p) for p in pos_by_bin])
    wp_ij = np.zeros((n_bins, n_bins, n_rp))

    n_pairs = n_bins * (n_bins + 1) // 2
    if verbose:
        print(f"Tabulating wp_ij: {n_bins} bins -> {n_pairs} pair counts "
              f"(bin sizes {counts.min()}-{counts.max()})")
    t0 = time.time()
    done = 0
    for i in range(n_bins):
        if counts[i] < 2:
            done += n_bins - i
            continue
        for j in range(i, n_bins):
            if counts[j] < 2:
                done += 1
                continue
            _, wp = compute_corr(
                'rppi', pos_by_bin[i], rp_edges,
                catalog2=pos_by_bin[j] if j != i else None,
                bins2=pi_bins, boxsize=Lbox, los=rsd_axis, output='wp')
            wp = np.nan_to_num(wp, nan=0.0)
            wp_ij[i, j] = wp
            wp_ij[j, i] = wp
            done += 1
        if verbose and (i % max(1, n_bins // 8) == 0):
            print(f"  bin {i}/{n_bins} done ({done}/{n_pairs} pairs, "
                  f"{time.time()-t0:.0f}s)")
    if verbose:
        print(f"Tabulation complete ({time.time()-t0:.0f}s)")

    return WggTabulation(wp_ij, rp_edges, pi_bins, logM_edges,
                         bin_fI_edges=fI_edges, bin_counts=counts,
                         Lbox=Lbox, rsd_axis=rsd_axis)


class TabulatedWgg:
    """
    Tabulated wp_gg predictor. See module docstring for the method.

    Parameters
    ----------
    tab : WggTabulation
    halo : HaloOccupation (set_halo_model already called; same catalog
        the tabulation was built from)
    """

    def __init__(self, tab: WggTabulation, halo,
                 n_gl_u: int = 24, n_gl_mu: int = 12, n_gl_phi: int = 12,
                 n_gl_u_1h: int = 12, n_gl_mu_1h: int = 8, n_gl_psi_1h: int = 8):
        if not hasattr(halo, 'HOD'):
            raise ValueError("halo has no HOD model — call set_halo_model() first.")
        self.tab = tab
        self.halo = halo

        logM = np.asarray(halo.logM)
        self.n_m = len(tab.bin_logM_edges) - 1
        self.n_f = 1 if tab.bin_fI_edges is None else len(tab.bin_fI_edges) - 1

        i_m = np.clip(np.digitize(logM, tab.bin_logM_edges) - 1, 0, self.n_m - 1)
        if tab.bin_fI_edges is None:
            self.bin_index = i_m.astype(np.int64)
        else:
            ab_prop = halo.fI if getattr(halo, 'fI', None) is not None else halo.fE
            if ab_prop is None:
                raise ValueError("Tabulation has fI bins but halo has no fI/fE.")
            i_f = np.clip(np.digitize(np.asarray(ab_prop), tab.bin_fI_edges) - 1,
                          0, self.n_f - 1)
            self.bin_index = (i_m * self.n_f + i_f).astype(np.int64)
        self.i_m = i_m
        n_bins = self.n_m * self.n_f
        if tab.wp_ij.shape[0] != n_bins:
            raise ValueError("Tabulation bin count doesn't match bin edges.")

        counts_m = np.maximum(np.bincount(i_m, minlength=self.n_m), 1)
        Rv = np.asarray(halo.radius) / 1e3          # kpc/h -> Mpc/h
        conc = np.asarray(halo.concentration)
        self.Rvir_m = np.bincount(i_m, weights=Rv, minlength=self.n_m) / counts_m
        self.conc_m = np.bincount(i_m, weights=conc, minlength=self.n_m) / counts_m
        self.Rvir_m[self.Rvir_m == 0] = 1e-3
        self.conc_m[self.conc_m == 0] = 5.0
        self._Rv = Rv
        self._conc = conc

        # Fine (Rvir, c) groups for the 1-halo term: <Nsat>^2 weighting is
        # steep in mass, so per-mass-bin mean Rvir biases the pair-separation
        # tail. The 1-halo term needs no pair counts -> refine for free.
        nR, nC = 48, 4
        r_edges = np.geomspace(Rv.min() * 0.999, Rv.max() * 1.001, nR + 1)
        iR = np.clip(np.digitize(Rv, r_edges) - 1, 0, nR - 1)
        c_edges = np.quantile(conc, np.linspace(0, 1, nC + 1))
        c_edges[0] -= 1e-6
        c_edges[-1] += 1e-6
        iC = np.clip(np.digitize(conc, c_edges) - 1, 0, nC - 1)
        self.grp_1h = (iR * nC + iC).astype(np.int64)
        n_grp = nR * nC
        cnt_g = np.maximum(np.bincount(self.grp_1h, minlength=n_grp), 1)
        self.Rvir_g = np.bincount(self.grp_1h, weights=Rv, minlength=n_grp) / cnt_g
        self.conc_g = np.bincount(self.grp_1h, weights=conc, minlength=n_grp) / cnt_g
        self.Rvir_g[self.Rvir_g == 0] = 1e-3
        self.conc_g[self.conc_g == 0] = 5.0
        self.n_grp_1h = n_grp

        gl = lambda n: tuple(0.5 * (v + 1) if k == 0 else 0.5 * v
                             for k, v in enumerate(np.polynomial.legendre.leggauss(n)))
        self._u, self._u_w = gl(n_gl_u)
        self._mu, self._mu_w = gl(n_gl_mu)
        self._u1, self._u1_w = gl(n_gl_u_1h)
        self._mu1, self._mu1_w = gl(n_gl_mu_1h)
        t_phi, w_phi = np.polynomial.legendre.leggauss(n_gl_phi)
        self._cos_phi = np.cos(0.5 * (t_phi + 1) * np.pi)
        self._phi_w = 0.5 * w_phi
        t_psi, w_psi = np.polynomial.legendre.leggauss(n_gl_psi_1h)
        self._cos_psi = np.cos(0.5 * (t_psi + 1) * np.pi)
        self._psi_w = 0.5 * w_psi
        self._x_norm = np.geomspace(1e-4, 1.0, 1000)

    # ── internals ──────────────────────────────────────────────────────

    def _kernel(self, Rvir, conc, f_exp, tau, lam):
        return satellite_offset_nodes(Rvir, conc, f_exp, tau, lam,
                                      self._u, self._u_w, self._mu, self._mu_w,
                                      self._x_norm)

    def _convolve(self, out_rp, grid_rp, f_vals, rho, w_rho):
        """Transverse 2D convolution of radial profile f with offset pdf."""
        d2 = (out_rp[:, None, None] ** 2 + rho[None, :, None] ** 2
              + 2.0 * out_rp[:, None, None] * rho[None, :, None]
              * self._cos_phi[None, None, :])
        f = np.interp(np.sqrt(np.maximum(d2, 0.0)), grid_rp, f_vals,
                      left=f_vals[0], right=0.0)
        return (f @ self._phi_w) @ w_rho          # on out_rp

    def predict(
        self,
        dict_params: Dict,
        rp_bins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict wp_gg(rp) bin-averaged on rp_bins [Mpc/h].

        Returns (rp_centers, wgg, info) with info = ngal, fsat, wp_1h, wp_2h.
        """
        halo, tab = self.halo, self.tab
        f_exp = float(dict_params.get('f_exp', 0.0))
        tau = float(dict_params.get('tau', 6.0))
        lam = float(dict_params.get('lambda_NFW', 1.0))
        logM = np.asarray(halo.logM)

        probC, probS = halo.HOD.compute_HOD_occupation(logM, dict_params)
        # Bernoulli sampling in populate_centrals clips probC at 1 implicitly
        probC = np.minimum(np.asarray(probC, dtype=np.float64), 1.0)
        probS = np.asarray(probS, dtype=np.float64)
        N = probC.sum() + probS.sum()
        ngal = N / halo.Lbox ** 3
        fsat = probS.sum() / N

        # Same-halo pair expectations (Poisson satellites; conformity exact)
        if halo.HOD.conformity:
            ones = np.ones_like(logM, dtype=bool)
            lam1 = np.asarray(halo.HOD.compute_satellite_occupation(
                logM, dict_params, has_central=ones))
            lam0 = np.asarray(halo.HOD.compute_satellite_occupation(
                logM, dict_params, has_central=~ones))
            e_cs = probC * lam1
            e_ss = 0.5 * (probC * lam1 ** 2 + (1 - probC) * lam0 ** 2)
        else:
            e_cs = probC * probS
            e_ss = 0.5 * probS ** 2

        n_bins = self.n_m * self.n_f
        C = np.bincount(self.bin_index, weights=probC, minlength=n_bins)
        S = np.bincount(self.bin_index, weights=probS, minlength=n_bins)
        S_mf = S.reshape(self.n_m, self.n_f)

        # ── 2-halo terms on the (nested) tabulation grid ──
        rp_tab = tab.rp_centers
        wp = tab.wp_ij
        cc = np.einsum('i,ijr,j->r', C, wp, C)
        cs_j = np.einsum('i,ijr->jr', C, wp)                       # Sum_i C_i wp_ij
        wp_g = wp.reshape(self.n_m, self.n_f, n_bins, -1)
        ss_i = np.einsum('mf,mfjr->mjr', S_mf, wp_g)               # Sum_{i in m} S_i wp_ij

        total = cc.copy()

        # <Nsat>-weighted kernel Rvir/c per mass bin: satellites live in the
        # most massive halos of each bin, so plain bin means bias the
        # convolution kernels (visible in the 1h->2h transition region).
        wS_m = np.bincount(self.i_m, weights=probS, minlength=self.n_m)
        with np.errstate(invalid='ignore'):
            Rvir_w = np.where(wS_m > 0, np.bincount(
                self.i_m, weights=probS * self._Rv, minlength=self.n_m)
                / np.maximum(wS_m, 1e-300), self.Rvir_m)
            conc_w = np.where(wS_m > 0, np.bincount(
                self.i_m, weights=probS * self._conc, minlength=self.n_m)
                / np.maximum(wS_m, 1e-300), self.conc_m)

        kernels = [self._kernel(Rvir_w[m], conc_w[m], f_exp, tau, lam)
                   for m in range(self.n_m)]

        # cen-sat: 2 * Sum_m [Sum_i C_i Sum_{j in m} S_j wp_ij] * K_m
        for m in range(self.n_m):
            sel = slice(m * self.n_f, (m + 1) * self.n_f)
            prof = np.einsum('jr,j->r', cs_j[sel], S[sel])
            if not np.any(prof):
                continue
            total += 2.0 * self._convolve(rp_tab, rp_tab, prof, *kernels[m])

        # sat-sat: Sum_{m<=m'} (2 - delta) G_mm' * K_m * K_m'
        G = np.einsum('mjr,j->mjr', ss_i, S)   # weight j by S then group
        G = G.reshape(self.n_m, self.n_m, self.n_f, -1).sum(axis=2)  # (n_m,n_m,n_rp)
        for m in range(self.n_m):
            for mp in range(m, self.n_m):
                prof = G[m, mp]
                if not np.any(prof):
                    continue
                mult = 1.0 if m == mp else 2.0
                inner = self._convolve(rp_tab, rp_tab, prof, *kernels[m])
                total += mult * self._convolve(rp_tab, rp_tab, inner, *kernels[mp])

        # Exact aggregation of nested fine bins onto rp_bins: wp combines
        # with annulus-area weights (RR separable in rp, pi), matching the
        # pycorr estimator on rp_bins identically for the cen-cen term.
        fine_edges = tab.rp_edges
        idx = np.searchsorted(fine_edges, rp_bins)
        if not np.allclose(fine_edges[idx], rp_bins, rtol=1e-6):
            raise ValueError(
                "rp_bins are not nested in the tabulation rp grid — "
                "precompute with rp_edges=refine_rp_edges(rp_bins).")
        w_area = np.diff(fine_edges ** 2)
        wp_2h = np.array([
            np.sum(total[idx[k]:idx[k + 1]] * w_area[idx[k]:idx[k + 1]])
            / np.sum(w_area[idx[k]:idx[k + 1]])
            for k in range(len(rp_bins) - 1)
        ]) / N ** 2

        # ── 1-halo term, bin-averaged directly on rp_bins ──
        V = halo.Lbox ** 3
        P_cs = np.bincount(self.grp_1h, weights=e_cs, minlength=self.n_grp_1h)
        P_ss = np.bincount(self.grp_1h, weights=e_ss, minlength=self.n_grp_1h)
        wp_1h = np.zeros(len(rp_bins) - 1)
        area = np.pi * np.diff(rp_bins ** 2)
        for g in np.nonzero((P_cs > 0) | (P_ss > 0))[0]:
            if P_cs[g] > 0:
                # Analytic projection-angle CDF: Pr(rho <= R | r) =
                # 1 - sqrt(1 - (R/r)^2) for r > R, else 1 (mu uniform).
                r, w_r = satellite_radial_nodes(
                    self.Rvir_g[g], self.conc_g[g], f_exp, tau, lam,
                    self._u, self._u_w, self._x_norm)
                ratio2 = np.minimum((rp_bins[None, :] / r[:, None]) ** 2, 1.0)
                cdf = w_r @ (1.0 - np.sqrt(1.0 - ratio2))
                wp_1h += 2 * V * P_cs[g] * np.diff(cdf) / (N ** 2 * area)
            if P_ss[g] > 0:
                # rho1, rho2 nodes; relative-angle CDF is analytic:
                # Pr(|d rho| <= R) = arccos((rho1^2+rho2^2-R^2)/(2 rho1 rho2))/pi
                r, w_r = satellite_radial_nodes(
                    self.Rvir_g[g], self.conc_g[g], f_exp, tau, lam,
                    self._u1, self._u1_w, self._x_norm)
                sin_th = np.sqrt(1.0 - self._mu1 ** 2)
                rho = (r[:, None] * sin_th[None, :]).ravel()
                w_rho = (w_r[:, None] * self._mu1_w[None, :]).ravel()
                w_rho = w_rho / w_rho.sum()
                s2 = rho[:, None, None] ** 2 + rho[None, :, None] ** 2
                p2 = np.maximum(2.0 * rho[:, None, None] * rho[None, :, None], 1e-30)
                x = np.clip((s2 - rp_bins[None, None, :] ** 2) / p2, -1.0, 1.0)
                cdf = np.einsum('i,j,ijk->k', w_rho, w_rho, np.arccos(x)) / np.pi
                wp_1h += 2 * V * P_ss[g] * np.diff(cdf) / (N ** 2 * area)

        rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        wgg = wp_2h + wp_1h
        info = {'ngal': ngal, 'fsat': fsat, 'wp_1h': wp_1h, 'wp_2h': wp_2h}
        return rp_centers, wgg, info

    def __repr__(self) -> str:
        return (f"TabulatedWgg(n_logM_bins={self.n_m}, n_fI_bins={self.n_f}, "
                f"n_halos={len(self.bin_index)})")
