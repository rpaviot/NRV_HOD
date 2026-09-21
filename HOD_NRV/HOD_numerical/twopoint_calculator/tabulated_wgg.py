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
    build_occupation_cells, build_tabulation_bins, satellite_offset_nodes,
    satellite_radial_nodes,
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
                 n_gl_u_1h: int = 12, n_gl_mu_1h: int = 8, n_gl_psi_1h: int = 8,
                 sat_kernel_weighting: str = "occupation"):
        if not hasattr(halo, 'HOD'):
            raise ValueError("halo has no HOD model — call set_halo_model() first.")
        if sat_kernel_weighting not in ("occupation", "static"):
            raise ValueError("sat_kernel_weighting must be 'occupation' or "
                             f"'static', got {sat_kernel_weighting!r}")
        self.tab = tab
        self.halo = halo
        self.sat_kernel_weighting = sat_kernel_weighting
        self._K_key = None
        self._K = None
        self._oh_key = None
        self._oh = None

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
        # Empty bins/groups carry the placeholder (Rvir, c) above, which is not
        # a physical halo — never build a kernel there. Occupancy is a property
        # of the catalogue, not of the HOD, so these masks are cache-safe.
        self._grp_occupied = np.bincount(self.grp_1h, minlength=n_grp) > 0
        self._m_occupied = np.bincount(i_m, minlength=self.n_m) > 0

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

    def _sat_kernels(self, rp_tab, f_exp, tau, lam):
        """Per-mass-bin 2-halo kernels at the STATIC bin means, cached.

        Built at ``self.Rvir_m`` / ``self.conc_m`` — the same plain bin means
        :class:`TabulatedDeltaSigma` uses for its satellite convolution — so
        the operator is a function of the satellite profile alone, with no
        dependence on the occupation. That is what the tabulation is for: the
        HOD moves, the kernel does not. A chain that pins (f_exp, tau,
        lambda_NFW) therefore builds these once for the whole run.

        The trade against ``sat_kernel_weighting='occupation'`` is accuracy in
        the 1h->2h transition: satellites sit preferentially in the massive
        half of a mass bin, so the occupation-weighted Rvir is the more
        faithful kernel centre. Which one to use is an empirical question —
        validate any switch against ``cross_check_tabulated.py --wgg``.
        """
        key = (float(f_exp), float(tau), float(lam), int(len(rp_tab)))
        if self._K_key == key:
            return self._K
        K = np.zeros((self.n_m, len(rp_tab), len(rp_tab)))
        for m in np.nonzero(self._m_occupied)[0]:
            K[m] = self._convolve_operator(
                rp_tab, *self._kernel(self.Rvir_m[m], self.conc_m[m],
                                      f_exp, tau, lam))
        self._K_key, self._K = key, K
        return K

    def _one_halo_operators(self, rp_bins, f_exp, tau, lam):
        """Per-group 1-halo pair CDFs, cached on the profile parameters.

        The occupation enters the 1-halo term only through the two scalars
        P_cs[g], P_ss[g]: the pair-separation CDFs themselves are built from
        the STATIC group (Rvir, c) and the satellite profile. Precomputing
        them turns the per-call loop over groups into two matrix-vector
        products, exactly (no approximation) -- and a chain with a pinned
        profile builds them once.

        Returns (A_cs, A_ss), each (n_grp_1h, len(rp_bins) - 1), such that
        ``wp_1h = 2 V (P_cs @ A_cs + P_ss @ A_ss) / (N**2 area)``.
        """
        key = (float(f_exp), float(tau), float(lam), rp_bins.tobytes())
        if self._oh_key == key:
            return self._oh
        n_rp = len(rp_bins) - 1
        A_cs = np.zeros((self.n_grp_1h, n_rp))
        A_ss = np.zeros((self.n_grp_1h, n_rp))
        sin_th = np.sqrt(1.0 - self._mu1 ** 2)
        for g in np.nonzero(self._grp_occupied)[0]:
            # cen-sat: analytic projection-angle CDF, Pr(rho <= R | r) =
            # 1 - sqrt(1 - (R/r)^2) for r > R, else 1 (mu uniform).
            r, w_r = satellite_radial_nodes(
                self.Rvir_g[g], self.conc_g[g], f_exp, tau, lam,
                self._u, self._u_w, self._x_norm)
            ratio2 = np.minimum((rp_bins[None, :] / r[:, None]) ** 2, 1.0)
            A_cs[g] = np.diff(w_r @ (1.0 - np.sqrt(1.0 - ratio2)))

            # sat-sat: relative-angle CDF is analytic,
            # Pr(|d rho| <= R) = arccos((rho1^2+rho2^2-R^2)/(2 rho1 rho2))/pi.
            # Symmetric under rho1 <-> rho2, so only the upper triangle of the
            # node pairs is evaluated (off-diagonal pairs carry weight 2).
            r, w_r = satellite_radial_nodes(
                self.Rvir_g[g], self.conc_g[g], f_exp, tau, lam,
                self._u1, self._u1_w, self._x_norm)
            rho = (r[:, None] * sin_th[None, :]).ravel()
            w_rho = (w_r[:, None] * self._mu1_w[None, :]).ravel()
            w_rho = w_rho / w_rho.sum()
            iu, ju, mult_p = self._pair_triangle(len(rho))
            r1, r2 = rho[iu], rho[ju]
            s2 = (r1 ** 2 + r2 ** 2)[:, None]
            p2 = np.maximum(2.0 * r1 * r2, 1e-30)[:, None]
            x = np.clip((s2 - rp_bins[None, :] ** 2) / p2, -1.0, 1.0)
            A_ss[g] = np.diff((mult_p * w_rho[iu] * w_rho[ju])
                              @ np.arccos(x) / np.pi)
        self._oh_key, self._oh = key, (A_cs, A_ss)
        return self._oh

    def _convolve(self, out_rp, grid_rp, f_vals, rho, w_rho):
        """Transverse 2D convolution of radial profile f with offset pdf."""
        d2 = (out_rp[:, None, None] ** 2 + rho[None, :, None] ** 2
              + 2.0 * out_rp[:, None, None] * rho[None, :, None]
              * self._cos_phi[None, None, :])
        f = np.interp(np.sqrt(np.maximum(d2, 0.0)), grid_rp, f_vals,
                      left=f_vals[0], right=0.0)
        return (f @ self._phi_w) @ w_rho          # on out_rp

    def _convolve_operator(self, grid_rp, rho, w_rho):
        """Matrix form of :meth:`_convolve` for out_rp == grid_rp.

        ``_convolve`` is linear in ``f_vals`` — np.interp is piecewise linear
        in the tabulated values, followed by two fixed contractions — so the
        kernel is an (n_rp, n_rp) matrix M with ``M @ f == _convolve(f)``.
        Building M costs one convolution, after which the O(n_m^2) sat-sat
        pairs are matmuls rather than 2 * n_m^2 convolutions.
        """
        g = np.asarray(grid_rp)
        n = len(g)
        d = np.sqrt(np.maximum(
            g[:, None, None] ** 2 + rho[None, :, None] ** 2
            + 2.0 * g[:, None, None] * rho[None, :, None]
            * self._cos_phi[None, None, :], 0.0)).ravel()

        shape = (n, len(rho), len(self._cos_phi))
        rows = self._operator_rows(shape)

        # Linear-interpolation weights on the bracketing nodes. Clipping the
        # index and t to [0, 1] reproduces np.interp's left=f_vals[0] below
        # the grid; zeroing the weight above it reproduces right=0.0.
        idx = np.clip(np.searchsorted(g, d) - 1, 0, n - 2)
        t = np.clip((d - g[idx]) / (g[idx + 1] - g[idx]), 0.0, 1.0)
        w = np.where(d > g[-1], 0.0,
                     np.broadcast_to((w_rho[:, None] * self._phi_w[None, :])[None],
                                     shape).ravel())
        w_hi = w * t
        w_lo = w - w_hi

        M = (np.bincount(rows * n + idx, w_lo, n * n)
             + np.bincount(rows * n + idx + 1, w_hi, n * n))
        return M.reshape(n, n)

    def _pair_triangle(self, n):
        """Upper-triangle node-pair indices and their multiplicity, cached."""
        cache = getattr(self, '_tri_cache', None)
        if cache is None:
            cache = self._tri_cache = {}
        if n not in cache:
            iu, ju = np.triu_indices(n)
            cache[n] = (iu, ju, np.where(iu == ju, 1.0, 2.0))
        return cache[n]

    def _operator_rows(self, shape):
        """Row index of every (out_rp, rho, phi) entry, cached per shape."""
        cache = getattr(self, '_rows_cache', None)
        if cache is None:
            cache = self._rows_cache = {}
        if shape not in cache:
            cache[shape] = np.repeat(np.arange(shape[0]), shape[1] * shape[2])
        return cache[shape]

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

        if self.sat_kernel_weighting == "static":
            # Kernels at the plain bin means: independent of the occupation,
            # so they are built once per (f_exp, tau, lambda) and cached.
            K = self._sat_kernels(rp_tab, f_exp, tau, lam)
        else:
            # <Nsat>-weighted kernel Rvir/c per mass bin: satellites live in
            # the most massive halos of each bin, so plain bin means bias the
            # convolution kernels (visible in the 1h->2h transition region).
            wS_m = np.bincount(self.i_m, weights=probS, minlength=self.n_m)
            with np.errstate(invalid='ignore'):
                Rvir_w = np.where(wS_m > 0, np.bincount(
                    self.i_m, weights=probS * self._Rv, minlength=self.n_m)
                    / np.maximum(wS_m, 1e-300), self.Rvir_m)
                conc_w = np.where(wS_m > 0, np.bincount(
                    self.i_m, weights=probS * self._conc, minlength=self.n_m)
                    / np.maximum(wS_m, 1e-300), self.conc_m)

            # Kernels as (n_rp, n_rp) operators. Mass bins with no satellites
            # contribute nothing (both profiles below are weighted by S), so
            # their operator is left at zero rather than built.
            n_rp_tab = len(rp_tab)
            K = np.zeros((self.n_m, n_rp_tab, n_rp_tab))
            for m in np.nonzero(wS_m > 0)[0]:
                K[m] = self._convolve_operator(
                    rp_tab, *self._kernel(Rvir_w[m], conc_w[m],
                                          f_exp, tau, lam))

        # cen-sat: 2 * Sum_m [Sum_i C_i Sum_{j in m} S_j wp_ij] * K_m
        P_cs = np.einsum('mfr,mf->mr', cs_j.reshape(self.n_m, self.n_f, -1),
                         S.reshape(self.n_m, self.n_f))
        total += 2.0 * np.einsum('mij,mj->i', K, P_cs)

        # sat-sat: Sum_{m<=m'} (2 - delta) K_m' K_m G_mm'
        G = np.einsum('mjr,j->mjr', ss_i, S)   # weight j by S then group
        G = G.reshape(self.n_m, self.n_m, self.n_f, -1).sum(axis=2)  # (n_m,n_m,n_rp)
        inner = (K @ G.transpose(0, 2, 1)).transpose(0, 2, 1)   # [m,n,i]
        mult = np.triu(np.full((self.n_m, self.n_m), 2.0), 1) + np.eye(self.n_m)
        T = np.einsum('mn,mni->ni', mult, inner)
        total += np.einsum('nij,nj->i', K, T)

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
        # The per-group pair CDFs are a function of the satellite profile and
        # the static group (Rvir, c) alone, so they are precomputed once per
        # (f_exp, tau, lambda) and the occupation enters as two scalars.
        V = halo.Lbox ** 3
        P_cs = np.bincount(self.grp_1h, weights=e_cs, minlength=self.n_grp_1h)
        P_ss = np.bincount(self.grp_1h, weights=e_ss, minlength=self.n_grp_1h)
        area = np.pi * np.diff(rp_bins ** 2)
        A_cs, A_ss = self._one_halo_operators(rp_bins, f_exp, tau, lam)
        wp_1h = 2 * V * (P_cs @ A_cs + P_ss @ A_ss) / (N ** 2 * area)

        rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        wgg = wp_2h + wp_1h
        info = {'ngal': ngal, 'fsat': fsat, 'wp_1h': wp_1h, 'wp_2h': wp_2h}
        return rp_centers, wgg, info

    # ── jit twin ───────────────────────────────────────────────────────

    def _occupation_cells(self, n_sub_logM: int = 16, n_sub_fI: int = 32,
                          fI_sub: str = "width"):
        """Fine (logM, fI) occupation cells nested in the wgg tabulation bins
        (see :func:`build_occupation_cells`), cached per sub-division."""
        key = (n_sub_logM, n_sub_fI, fI_sub)
        cache = getattr(self, '_occ_cells_cache', None)
        if cache is None:
            cache = self._occ_cells_cache = {}
        if key not in cache:
            cache[key] = build_occupation_cells(
                self.halo, self.tab.bin_logM_edges, self.tab.bin_fI_edges,
                n_sub_logM, n_sub_fI, fI_sub)
        return cache[key]

    def make_predict_jax(self, rp_bins: np.ndarray, f_exp: float, tau: float,
                         lambda_NFW: float,
                         n_sub_logM: int = 16, n_sub_fI: int = 32,
                         fI_sub: str = "width"):
        """Build a pure-JAX twin of :meth:`predict` for a PINNED satellite profile.

        Returns ``predict_fn(params) -> (wgg, ngal, fsat)`` with ``wgg``
        bin-averaged on ``rp_bins``, for ``params`` a dict of (traced) scalars.
        It is the counterpart of :meth:`TabulatedDeltaSigma.make_predict_jax`
        and is built on the same fine occupation grid
        (:func:`build_occupation_cells`, ``n_m*n_sub_logM x n_f*n_sub_fI``
        cells): the occupation is evaluated at the per-cell mean (logM, fI)
        with the cell's halo count as weight instead of on every halo, which
        removes the 9.2M-halo NumPy work that dominates :meth:`predict`.
        Everything downstream is exact:

        - 2-halo: the per-bin sums C_i, S_i are reshape-sums over the cells
          (cells nest in the tabulation bins), contracted with ``wp_ij`` and
          the STATIC per-mass-bin satellite kernels of :meth:`_sat_kernels`.
        - 1-halo: the per-halo pair expectations e_cs, e_ss are functions of
          (logM, fI) only, so the per-group sums are a joint (cell, group)
          halo histogram applied to the cached :meth:`_one_halo_operators`;
          the histogram is folded into the operators once, so per call the
          same-halo term is two (n_cells,) @ (n_cells, n_rp) products.

        The profile is pinned because both operator sets are built by the
        NumPy code at construction and enter the jit as constants — the
        counterpart of what XLA does for DeltaSigma when (f_exp, tau,
        lambda_NFW) are Python floats. A free-profile twin would need the
        offset convolution in jnp; until then the sampler falls back to
        :meth:`predict` for free profiles. Requires
        ``sat_kernel_weighting='static'``: the occupation-weighted kernels of
        the default path depend on the HOD and cannot be constants.
        """
        import jax.numpy as jnp
        from ..HOD_models import build_occupation_fn_jax

        if self.sat_kernel_weighting != "static":
            raise ValueError(
                "make_predict_jax needs sat_kernel_weighting='static': the "
                "occupation-weighted kernels depend on the HOD and cannot be "
                "jit constants.")
        occ = self.halo.HOD
        tab = self.tab
        rp_bins = np.asarray(rp_bins, dtype=np.float64)
        f_exp, tau, lam = float(f_exp), float(tau), float(lambda_NFW)
        cells = self._occupation_cells(n_sub_logM, n_sub_fI, fI_sub)
        cen_coef, sat_coef = cells['cen_coef'], cells['sat_coef']
        n_mc, n_fc = cells['n_mc'], cells['n_fc']
        n_sub_fI = cells['n_sub_fI']
        n_m, n_f = self.n_m, self.n_f
        n_bins = n_m * n_f

        # ── constants: the same fine-bin aggregation as predict ──
        fine_edges = tab.rp_edges
        idx = np.searchsorted(fine_edges, rp_bins)
        if not np.allclose(fine_edges[idx], rp_bins, rtol=1e-6):
            raise ValueError(
                "rp_bins are not nested in the tabulation rp grid — "
                "precompute with rp_edges=refine_rp_edges(rp_bins).")
        w_area = np.diff(fine_edges ** 2)
        n_rp = len(rp_bins) - 1
        M_agg = np.zeros((n_rp, len(tab.rp_centers)))
        for k in range(n_rp):
            sl = slice(idx[k], idx[k + 1])
            M_agg[k, sl] = w_area[sl] / w_area[sl].sum()

        K = self._sat_kernels(tab.rp_centers, f_exp, tau, lam)
        mult = np.triu(np.full((n_m, n_m), 2.0), 1) + np.eye(n_m)

        # 1-halo: joint (cell, group) halo histogram folded into the operators
        A_cs, A_ss = self._one_halo_operators(rp_bins, f_exp, tau, lam)
        W = np.bincount(cells['cell'] * self.n_grp_1h + self.grp_1h,
                        minlength=cells['n_cells'] * self.n_grp_1h
                        ).reshape(cells['n_cells'], self.n_grp_1h).astype(np.float64)
        B_cs = W @ A_cs                                   # (n_cells, n_rp)
        B_ss = W @ A_ss
        area = np.pi * np.diff(rp_bins ** 2)
        V = self.halo.Lbox ** 3

        j_logM_cell = jnp.asarray(cells['logM_cell'].reshape(n_mc, n_fc))
        j_prop_cell = jnp.asarray(cells['prop_cell'].reshape(n_mc, n_fc))
        j_sign_cell = jnp.asarray(cells['sign_cell'].reshape(n_mc, n_fc))
        j_N_cell = jnp.asarray(cells['N_cell'].reshape(n_mc, n_fc))
        j_wp = jnp.asarray(tab.wp_ij)                     # (n_bins, n_bins, n_rp_tab)
        j_wp_g = j_wp.reshape(n_m, n_f, n_bins, -1)
        j_K = jnp.asarray(K)
        j_mult = jnp.asarray(mult)
        j_M_agg = jnp.asarray(M_agg)
        j_B_cs, j_B_ss = jnp.asarray(B_cs), jnp.asarray(B_ss)
        j_area = jnp.asarray(area)
        occ_fn = build_occupation_fn_jax(occ, split_conformity=True)
        Lbox3 = self.halo.Lbox ** 3
        has_ab, ab_method = occ.assembly_bias, occ.ab_method

        def predict_fn(params):
            cshift = sshift = 0.0
            if has_ab and ab_method == "mass":
                cshift = params.get(cen_coef, 0.0) * j_prop_cell
                sshift = params.get(sat_coef, 0.0) * j_prop_cell
            probC, probS, lam1, lam0 = occ_fn(j_logM_cell, params, cshift, sshift)
            if has_ab and ab_method == "direct":
                ab_c = params.get(cen_coef, 0.0) * j_sign_cell
                ab_s = params.get(sat_coef, 0.0) * j_sign_cell
                probC = probC + ab_c * jnp.minimum(probC, 1.0 - probC)
                probS = probS * (1.0 + ab_s)
                lam1 = lam1 * (1.0 + ab_s)
                lam0 = lam0 * (1.0 + ab_s)
            elif has_ab and ab_method == "variant":
                ab_c = params.get(cen_coef, 0.0) * j_prop_cell
                ab_s = params.get(sat_coef, 0.0) * j_prop_cell
                probC = jnp.minimum(probC, 1.0)
                probC = probC * (1.0 + ab_c * (1.0 - probC))
                probS = probS * (1.0 + ab_s)
                lam1 = lam1 * (1.0 + ab_s)
                lam0 = lam0 * (1.0 + ab_s)
            probC = jnp.minimum(probC, 1.0)

            wC = probC * j_N_cell
            wS = probS * j_N_cell
            N = jnp.sum(wC) + jnp.sum(wS)
            ngal = N / Lbox3
            fsat = jnp.sum(wS) / N

            # ── 2-halo on the tabulation grid (twin of predict) ──
            C = wC.reshape(n_m, n_sub_logM, n_f, n_sub_fI).sum(axis=(1, 3))
            S = wS.reshape(n_m, n_sub_logM, n_f, n_sub_fI).sum(axis=(1, 3))
            C_flat = C.reshape(n_bins)
            S_flat = S.reshape(n_bins)
            cc = jnp.einsum('i,ijr,j->r', C_flat, j_wp, C_flat)
            cs_j = jnp.einsum('i,ijr->jr', C_flat, j_wp)
            ss_i = jnp.einsum('mf,mfjr->mjr', S, j_wp_g)
            P_cs = jnp.einsum('mfr,mf->mr', cs_j.reshape(n_m, n_f, -1), S)
            total = cc + 2.0 * jnp.einsum('mij,mj->i', j_K, P_cs)
            G = jnp.einsum('mjr,j->mjr', ss_i, S_flat)
            G = G.reshape(n_m, n_m, n_f, -1).sum(axis=2)
            inner = (j_K @ G.transpose(0, 2, 1)).transpose(0, 2, 1)
            T = jnp.einsum('mn,mni->ni', j_mult, inner)
            total = total + jnp.einsum('nij,nj->i', j_K, T)
            wp_2h = (j_M_agg @ total) / N ** 2

            # ── 1-halo: per-cell pair expectations x folded operators ──
            e_cs = (probC * lam1).ravel()
            e_ss = (0.5 * (probC * lam1 ** 2 + (1.0 - probC) * lam0 ** 2)).ravel()
            wp_1h = 2.0 * V * (e_cs @ j_B_cs + e_ss @ j_B_ss) / (N ** 2 * j_area)

            return wp_2h + wp_1h, ngal, fsat

        return predict_fn

    def __repr__(self) -> str:
        return (f"TabulatedWgg(n_logM_bins={self.n_m}, n_fI_bins={self.n_f}, "
                f"n_halos={len(self.bin_index)})")
