"""
Measure beta^NL(r, M1, M2) from a halo catalog in real space.

Mirrors the convention used by Dark Emulator's
`get_xiauto_mass(xs, M1, M2, z)` (de_interface.py:362), which builds
xi_hh at delta-function masses M1, M2 from a 2D finite difference of the
threshold-emulator output:

    xi(M1, M2) = d^2 [n(>M1)*n(>M2)*xi(>M1, >M2)] / dM1 dM2
                 / [(dn/dM1)(dn/dM2)]

Discretized as a four-corner formula with M_p = (1+eps)*M, M_m = (1-eps)*M.
We apply exactly the same combination on the simulation side, but replacing
the emulator's threshold xi by xi measured with pycorr on threshold halo
subsamples. This yields a fixed-mass xi_hh(r, M, M') directly comparable
to `emu.get_xiauto_mass(xs, M, M', z)`.

beta^NL is then defined in real space as

    beta^NL(r, M1, M2) = xi_hh(r, M1, M2) / [b(M1) b(M2) xi_mm(r)] - 1

with b(M) anchored on a large-scale window (e.g. r in [30, 80] Mpc/h)
where xi_hh / xi_mm is approximately constant.

This convention extends below 1e12 M_sun/h, where Dark Emulator fails.

All masses in M_sun/h, lengths in Mpc/h.
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np

import hashlib
import os

from ..HOD_numerical.twopoint_calculator.standard_two_point_calculator import compute_corr
from .hankel_transforms import Pk_to_xi_gg


def _select_above(halo_positions: np.ndarray,
                  halo_log10M: np.ndarray,
                  log10M_threshold: float) -> np.ndarray:
    return halo_positions[halo_log10M >= log10M_threshold]


def measure_xi_hh_thresholds(
    halo_positions: np.ndarray,
    halo_log10M: np.ndarray,
    log10M_thresholds: Sequence[float],
    Lbox: float,
    r_bins: np.ndarray,
    los: str = "z",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Measure xi_hh(r) for every pair of mass-threshold halo subsamples.

    Parameters
    ----------
    halo_positions : (N, 3) array, Mpc/h, in [0, Lbox).
    halo_log10M    : (N,) array, log10(M) in M_sun/h.
    log10M_thresholds : sequence of T thresholds (sorted ascending).
        For δ-mass synthesis at K target masses with eps=0.01 the caller
        passes 2K thresholds (one above and one below each target).
    Lbox    : box size, Mpc/h.
    r_bins  : (n_r+1,) edges, Mpc/h.
    los     : line-of-sight axis, passed to pycorr.

    Returns
    -------
    dict with:
        r            (n_r,)             bin centers
        log10M_th    (T,)               thresholds (echoed)
        N_above      (T,)               halo counts above each threshold
        xi           (T, T, n_r)        xi_hh(>M_a, >M_b, r), symmetric in (a,b)
    """
    halo_positions = np.asarray(halo_positions, dtype=np.float64)
    halo_log10M = np.asarray(halo_log10M, dtype=np.float64)
    log10M_thresholds = np.asarray(log10M_thresholds, dtype=np.float64)
    T = len(log10M_thresholds)

    subsamples = []
    N_above = np.zeros(T, dtype=np.int64)
    for i, lt in enumerate(log10M_thresholds):
        sub = _select_above(halo_positions, halo_log10M, lt)
        subsamples.append(sub)
        N_above[i] = len(sub)
        if verbose:
            print(f"  threshold log10M >= {lt:.4f}: N = {N_above[i]}")

    n_r = len(r_bins) - 1
    xi = np.full((T, T, n_r), np.nan)
    r_centers = None

    # Auto and cross over all (i <= j)
    for i in range(T):
        if N_above[i] < 2:
            if verbose:
                print(f"  skipping i={i}, only {N_above[i]} halos")
            continue
        if verbose:
            print(f"  pycorr auto i={i} (N={N_above[i]})")
        r_c, xi_ii = compute_corr(
            mode="s", catalog1=subsamples[i], bins1=r_bins,
            boxsize=Lbox, los=los, output="auto",
        )
        r_centers = np.asarray(r_c)
        xi[i, i, :] = np.asarray(xi_ii)

        for j in range(i + 1, T):
            if N_above[j] < 2:
                continue
            if verbose:
                print(f"  pycorr cross i={i}, j={j} (N1={N_above[i]}, N2={N_above[j]})")
            r_c, xi_ij = compute_corr(
                mode="s", catalog1=subsamples[i], bins1=r_bins,
                catalog2=subsamples[j], boxsize=Lbox, los=los, output="auto",
            )
            xi[i, j, :] = np.asarray(xi_ij)
            xi[j, i, :] = xi[i, j, :]

    return {
        "r": r_centers,
        "log10M_th": log10M_thresholds,
        "N_above": N_above,
        "xi": xi,
    }


def _pair_indices_for_target(log10M_thresholds: np.ndarray,
                             log10M_target: float,
                             eps: float) -> tuple:
    """Find the (m, p) indices of (1-eps)*M and (1+eps)*M in the threshold list."""
    log10M_m = np.log10((1.0 - eps)) + log10M_target
    log10M_p = np.log10((1.0 + eps)) + log10M_target
    im = int(np.argmin(np.abs(log10M_thresholds - log10M_m)))
    ip = int(np.argmin(np.abs(log10M_thresholds - log10M_p)))
    return im, ip


def xi_hh_at_mass(
    threshold_result: Dict[str, np.ndarray],
    log10M_targets: Sequence[float],
    eps: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Build xi_hh(r, M_i, M_j) at delta-function target masses by combining
    threshold subsample measurements with the same four-corner formula
    Dark Emulator uses (de_interface.py:362-394).

    Requires `threshold_result` to contain thresholds at log10(M*(1±eps))
    for every M in `log10M_targets`.

    Parameters
    ----------
    threshold_result : output of `measure_xi_hh_thresholds`.
    log10M_targets   : (K,) target masses in log10(M_sun/h).
    eps              : finite-difference step (matches emulator's 0.01).

    Returns
    -------
    dict with:
        r        (n_r,)          bin centers
        M_targets (K,)           10**log10M_targets, M_sun/h
        xi_hh    (K, K, n_r)     xi at fixed (M_i, M_j)
    """
    log10M_th = threshold_result["log10M_th"]
    xi_thr = threshold_result["xi"]
    N_above = threshold_result["N_above"]
    r_centers = threshold_result["r"]

    K = len(log10M_targets)
    n_r = xi_thr.shape[-1]
    xi_out = np.full((K, K, n_r), np.nan)

    for a, lMa in enumerate(log10M_targets):
        ima, ipa = _pair_indices_for_target(log10M_th, lMa, eps)
        Nma = float(N_above[ima])
        Npa = float(N_above[ipa])
        # number of halos in the thin shell around M_a:
        dN_a = Nma - Npa
        for b, lMb in enumerate(log10M_targets):
            imb, ipb = _pair_indices_for_target(log10M_th, lMb, eps)
            Nmb = float(N_above[imb])
            Npb = float(N_above[ipb])
            dN_b = Nmb - Npb

            # Four-corner xi_hh combination — discrete analogue of
            # ∂²[N(>M_a) N(>M_b) xi(>M_a, >M_b)] / ∂M_a ∂M_b
            # with N (count) standing in for n (density) — the Lbox³ cancels.
            num = (Nma * Nmb * xi_thr[ima, imb]
                   - Nma * Npb * xi_thr[ima, ipb]
                   - Npa * Nmb * xi_thr[ipa, imb]
                   + Npa * Npb * xi_thr[ipa, ipb])
            denom = Nma * Nmb - Nma * Npb - Npa * Nmb + Npa * Npb
            if abs(denom) < 1e-30 or dN_a <= 0 or dN_b <= 0:
                continue
            xi_out[a, b, :] = num / denom

    return {
        "r": r_centers,
        "M_targets": 10.0 ** np.asarray(log10M_targets, dtype=np.float64),
        "xi_hh": xi_out,
    }


def dense_thresholds(log10M_min: float = 11.0,
                     log10M_max: float = 15.0,
                     step_dex: float = 0.05,
                     eps: float = 0.01) -> np.ndarray:
    """
    Build a fixed log10M ladder over [log10M_min, log10M_max] at step_dex
    spacing, with ±log10(1±eps) shells around each rung. Sorted/deduped.
    Once tabulated, any target mass on the ladder reuses the cache.
    """
    n_rungs = int(np.round((log10M_max - log10M_min) / step_dex)) + 1
    rungs = np.linspace(log10M_min, log10M_max, n_rungs)
    out = []
    for lM in rungs:
        out.append(lM + np.log10(1.0 - eps))
        out.append(lM + np.log10(1.0 + eps))
    arr = np.unique(np.round(np.asarray(out, dtype=np.float64), 8))
    return np.sort(arr)


def thresholds_for_targets(log10M_targets: Sequence[float],
                           eps: float = 0.01) -> np.ndarray:
    """
    Build the sorted, deduplicated threshold list needed to measure
    xi at every target mass via four-corner finite differences.
    """
    out = []
    for lM in log10M_targets:
        out.append(lM + np.log10(1.0 - eps))
        out.append(lM + np.log10(1.0 + eps))
    arr = np.unique(np.round(np.asarray(out, dtype=np.float64), 8))
    return np.sort(arr)


def _four_corner_fd(v_mm: np.ndarray, v_mp: np.ndarray,
                    v_pm: np.ndarray, v_pp: np.ndarray,
                    n_1p: float, n_1m: float,
                    n_2p: float, n_2m: float) -> np.ndarray:
    """
    Four-corner finite difference (Dark Emulator convention) converting
    threshold-evaluated quantities v(>M, >M') at the four shell corners
    (mm, mp, pm, pp) into a fixed-mass value at (M_1, M_2).

    numer = v_mm n_1m n_2m - v_mp n_1m n_2p - v_pm n_1p n_2m + v_pp n_1p n_2p
    denom = n_1m n_2m - n_1m n_2p - n_1p n_2m + n_1p n_2p

    Mirrors xi_hh_at_mass (counts replaced by densities; the conversion factor
    Lbox^3 cancels) and DE's hod_interface.py:345-349 / :391-393.
    """
    num = (v_mm * n_1m * n_2m
           - v_mp * n_1m * n_2p
           - v_pm * n_1p * n_2m
           + v_pp * n_1p * n_2p)
    denom = n_1m * n_2m - n_1m * n_2p - n_1p * n_2m + n_1p * n_2p
    return num / denom


def gamma_h_at_mass_shells(emu, k: np.ndarray, z: float,
                           log10M_targets: Sequence[float],
                           eps: float = 0.02) -> tuple:
    """
    Evaluate Dark Emulator's halo propagator Γ_h(k) at the upper/lower
    density shells corresponding to each target mass.

    Mirrors hod_interface.py:355-367 / 463-467 in dark_emulator_public.

    Parameters
    ----------
    emu : dark_emulator.darkemu.de_interface instance.
    k   : (n_k,) wavenumbers in h/Mpc.
    z   : redshift.
    log10M_targets : (K,) target masses in log10(M_sun/h).
    eps : mass-shell half-width (DE uses 0.02).

    Returns
    -------
    gp, gm : (K, n_k) arrays. gp = Γ_h at (1+eps)·M, gm at (1-eps)·M.

    Note: DE's g1 emulator is trained on log10(n) in [-8.5, -2.5]. For
    masses where the density falls outside this range the underlying
    spline extrapolates and the result should be treated as unreliable
    (typically below ~1e12 M_sun/h).
    """
    log10M_targets = np.atleast_1d(np.asarray(log10M_targets, dtype=np.float64))
    K = len(log10M_targets)
    n_k = len(k)
    gp = np.zeros((K, n_k))
    gm = np.zeros((K, n_k))
    for i, lM in enumerate(log10M_targets):
        M = 10.0 ** lM
        logdens_p = np.log10(emu.mass_to_dens((1.0 + eps) * M, z))
        logdens_m = np.log10(emu.mass_to_dens((1.0 - eps) * M, z))
        gp[i] = emu.g1.get(k, z, logdens_p).ravel()
        gm[i] = emu.g1.get(k, z, logdens_m).ravel()
    return gp, gm


def compute_xi_tree(
    emu,
    P_lin_func: Callable[[np.ndarray], np.ndarray],
    z: float,
    r_out: np.ndarray,
    log10M_targets: Sequence[float],
    eps: float = 0.02,
    k_grid: Optional[np.ndarray] = None,
    gamma_min_log10M: float = 12.0,
    fallback: str = "linear_bias",
    b_hh: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build the analytical large-scale ξ_tree(r, M_i, M_j) via the
    Dark Emulator prescription:

        ξ_tree = IFT[ Γ_h(k, M_1) · Γ_h(k, M_2) · P_lin(k) ]

    using a four-corner FD in (1±eps)·M shells to land at fixed mass.

    For target masses below `gamma_min_log10M` the DE propagator is
    unreliable; we fall back to a linear-bias model
        ξ_tree(r, M_i, M_j) ≈ b(M_i) b(M_j) ξ_lin(r)
    if `fallback == "linear_bias"` and `b_hh` is supplied; otherwise the
    target is left as NaN and a warning is printed.

    Parameters
    ----------
    emu             : Dark Emulator instance (cosmo.emu).
    P_lin_func      : callable, k (h/Mpc) -> P_lin(k, z) in (Mpc/h)^3.
    z               : redshift.
    r_out           : (n_r,) output r grid (Mpc/h).
    log10M_targets  : (K,) target masses.
    eps             : DE-style FD half-width (default 0.02).
    k_grid          : optional (n_k,) wavenumbers in h/Mpc.
                      Default: log-spaced 1e-4 to 5e2, 1024 points
                      (matches `xi_mm_from_pk`).
    gamma_min_log10M: below this log10M, use the fallback.
    fallback        : "linear_bias" or "none".
    b_hh            : (K,) large-scale biases — only used by the
                      linear-bias fallback. Must be supplied if the
                      fallback is to fire for any target.

    Returns
    -------
    dict with:
        r          (n_r,)            echoed r_out
        xi_tree    (K, K, n_r)       ξ_tree at fixed (M_i, M_j)
        used_de    (K,)              bool, True if Γ_h was used for that
                                     target (False if fallback fired).
    """
    log10M_targets = np.atleast_1d(np.asarray(log10M_targets, dtype=np.float64))
    K = len(log10M_targets)
    r_out = np.asarray(r_out, dtype=np.float64)
    n_r = len(r_out)

    if k_grid is None:
        k_grid = np.logspace(-4.0, np.log10(5e2), 1024)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    P_lin = np.asarray(P_lin_func(k_grid), dtype=np.float64)

    used_de = log10M_targets >= gamma_min_log10M

    xi_lin_at_r = None
    if (~used_de).any() and fallback == "linear_bias":
        r_lin, xi_lin = Pk_to_xi_gg(k_grid, P_lin)
        good = (r_lin > 0) & np.isfinite(xi_lin)
        xi_lin_at_r = np.interp(np.log(r_out),
                                 np.log(np.asarray(r_lin)[good]),
                                 np.asarray(xi_lin)[good])

    xi_tree = np.full((K, K, n_r), np.nan)

    de_idx = np.where(used_de)[0]
    if len(de_idx) > 0:
        gp, gm = gamma_h_at_mass_shells(
            emu, k_grid, z, log10M_targets[de_idx], eps=eps,
        )

        n_p = np.zeros(len(de_idx))
        n_m = np.zeros(len(de_idx))
        for ii, i in enumerate(de_idx):
            M = 10.0 ** log10M_targets[i]
            n_p[ii] = emu.mass_to_dens((1.0 + eps) * M, z)
            n_m[ii] = emu.mass_to_dens((1.0 - eps) * M, z)

        for ii, i in enumerate(de_idx):
            for jj, j in enumerate(de_idx):
                if j < i:
                    continue
                P_mm = gm[ii] * gm[jj] * P_lin
                P_mp = gm[ii] * gp[jj] * P_lin
                P_pm = gp[ii] * gm[jj] * P_lin
                P_pp = gp[ii] * gp[jj] * P_lin
                P_tree = _four_corner_fd(P_mm, P_mp, P_pm, P_pp,
                                          n_p[ii], n_m[ii],
                                          n_p[jj], n_m[jj])
                r_t, xi_t = Pk_to_xi_gg(k_grid, P_tree)
                r_t = np.asarray(r_t); xi_t = np.asarray(xi_t)
                good = (r_t > 0) & np.isfinite(xi_t)
                xi_interp = np.interp(np.log(r_out),
                                       np.log(r_t[good]), xi_t[good])
                xi_tree[i, j, :] = xi_interp
                if j != i:
                    xi_tree[j, i, :] = xi_interp

    if (~used_de).any():
        if fallback == "linear_bias":
            if b_hh is None:
                print("[compute_xi_tree] warning: gamma fallback requested but "
                      "b_hh not provided; leaving low-mass targets as NaN.")
            else:
                b_hh = np.asarray(b_hh, dtype=np.float64)
                for i in range(K):
                    for j in range(K):
                        if used_de[i] and used_de[j]:
                            continue
                        if not (np.isfinite(b_hh[i]) and np.isfinite(b_hh[j])):
                            continue
                        xi_tree[i, j, :] = b_hh[i] * b_hh[j] * xi_lin_at_r
        elif fallback == "none":
            for i in np.where(~used_de)[0]:
                print(f"[compute_xi_tree] log10M={log10M_targets[i]:.3f} below "
                      f"gamma_min_log10M={gamma_min_log10M}; skipping stitch.")
        else:
            raise ValueError(f"unknown fallback: {fallback!r}")

    return {"r": r_out, "xi_tree": xi_tree, "used_de": used_de}


def stitch_xi_hh(r: np.ndarray, xi_dir: np.ndarray, xi_tree: np.ndarray,
                 r_switch: float = 60.0) -> np.ndarray:
    """
    Smoothly switch between direct and analytical ξ_hh with a quartic
    exponential taper centered at r_switch.

        ξ_hh = ξ_dir · w(r) + ξ_tree · (1 − w(r)),    w(r) = exp[-(r/r_switch)^4]

    Mirrors hod_interface.py:419-423 in dark_emulator_public.

    Where ξ_tree is NaN the direct measurement is returned unchanged.
    Shapes: r is (n_r,), xi_dir / xi_tree broadcast over any leading axes.
    """
    r = np.asarray(r, dtype=np.float64)
    w = np.exp(-(r / float(r_switch)) ** 4)
    tree_nan = ~np.isfinite(xi_tree)
    xi_tree_safe = np.where(tree_nan, 0.0, xi_tree)
    w_eff = np.where(tree_nan, np.ones_like(xi_tree), w)
    return xi_dir * w_eff + xi_tree_safe * (1.0 - w_eff)


def xi_mm_from_pk(P_nl_func: Callable[[np.ndarray], np.ndarray],
                  k_min: float = 1e-4,
                  k_max: float = 5e2,
                  n_k: int = 1024) -> tuple:
    """
    Get xi_mm(r) by Hankel-transforming the nonlinear matter P(k).

    Parameters
    ----------
    P_nl_func : callable. P_nl_func(k) -> P_nl(k, z) at the snapshot
        redshift, in (Mpc/h)^3, with k in h/Mpc.
    k_min, k_max, n_k : log-spaced k grid for FAST-PT.

    Returns
    -------
    r, xi_mm : 1D arrays. r in Mpc/h.
    """
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    P = np.asarray(P_nl_func(k))
    r, xi = Pk_to_xi_gg(k, P)  # standard cosmology HT (mu=0.5, alpha=1.5)
    return np.asarray(r), np.asarray(xi)


def fit_bias_largescale(r: np.ndarray, xi_hh: np.ndarray,
                        xi_mm_func: Callable[[np.ndarray], np.ndarray],
                        r_min: float = 30.0,
                        r_max: float = 80.0) -> float:
    """
    Linear-bias estimate from the large-scale ratio xi_hh / xi_mm.

    b^2 = < xi_hh(r) / xi_mm(r) >_{r_min < r < r_max}
    """
    sel = (r >= r_min) & (r <= r_max) & np.isfinite(xi_hh) & (xi_hh > 0)
    if sel.sum() < 2:
        return np.nan
    xi_mm_at_r = np.asarray(xi_mm_func(r[sel]))
    good = (xi_mm_at_r > 0) & np.isfinite(xi_mm_at_r)
    if good.sum() < 2:
        return np.nan
    ratio = xi_hh[sel][good] / xi_mm_at_r[good]
    if np.any(ratio <= 0):
        return np.nan
    return float(np.sqrt(np.mean(ratio)))


def tabulate_xi_hh_thresholds(
    halo_positions: np.ndarray,
    halo_log10M: np.ndarray,
    log10M_thresholds: Sequence[float],
    Lbox: float,
    r_bins: np.ndarray,
    cache_path: Optional[str] = None,
    los: str = "z",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute (or load from disk) the cumulative-xi tabulation
    xi_hh(>M_a, >M_b, r) for every pair of mass thresholds.

    The pycorr stage is the dominant cost in the xi-space pipeline.
    Caching this output lets subsequent target-mass / eps / bias-window
    sweeps reuse it for free.

    Cache key: (Lbox, r_bins, log10M_thresholds, los, sha1 of halo
    positions+log10M). If the NPZ at `cache_path` already exists with
    matching metadata, it is loaded; else it is computed and saved.
    """
    log10M_thresholds = np.asarray(log10M_thresholds, dtype=np.float64)
    r_bins = np.asarray(r_bins, dtype=np.float64)

    h = hashlib.sha1()
    h.update(np.ascontiguousarray(halo_positions, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(halo_log10M, dtype=np.float64).tobytes())
    catalog_sha = h.hexdigest()

    if cache_path is not None and os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=False)
            ok = (
                float(d["Lbox"]) == float(Lbox)
                and str(d["los"]) == str(los)
                and d["log10M_th"].shape == log10M_thresholds.shape
                and np.allclose(d["log10M_th"], log10M_thresholds)
                and d["r_bins"].shape == r_bins.shape
                and np.allclose(d["r_bins"], r_bins)
                and str(d["catalog_sha"]) == catalog_sha
            )
            if ok:
                if verbose:
                    print(f"[tabulate_xi_hh_thresholds] cache hit: {cache_path}")
                return {
                    "r": d["r"],
                    "log10M_th": d["log10M_th"],
                    "N_above": d["N_above"],
                    "xi": d["xi"],
                }
            if verbose:
                print(f"[tabulate_xi_hh_thresholds] cache mismatch, recomputing: {cache_path}")
        except Exception as e:
            if verbose:
                print(f"[tabulate_xi_hh_thresholds] cache load failed ({e}), recomputing")

    res = measure_xi_hh_thresholds(
        halo_positions, halo_log10M, log10M_thresholds, Lbox, r_bins,
        los=los, verbose=verbose,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(
            cache_path,
            r=res["r"], log10M_th=res["log10M_th"],
            N_above=res["N_above"], xi=res["xi"],
            r_bins=r_bins, Lbox=float(Lbox), los=str(los),
            catalog_sha=catalog_sha,
        )
        if verbose:
            print(f"[tabulate_xi_hh_thresholds] saved cache: {cache_path}")
    return res


def beta_nl_xi_from_tabulation(
    threshold_result: Dict[str, np.ndarray],
    log10M_targets: Sequence[float],
    P_nl_func: Callable[[np.ndarray], np.ndarray],
    eps: float = 0.01,
    bias_window: tuple = (30.0, 80.0),
    stitch_large_scale: bool = False,
    emu=None,
    P_lin_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    z: Optional[float] = None,
    r_switch: float = 60.0,
    gamma_eps: float = 0.02,
    gamma_min_log10M: float = 12.0,
    gamma_fallback: str = "linear_bias",
    gamma_k_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Cheap post-processing: combine a cumulative-xi tabulation
    (output of `tabulate_xi_hh_thresholds`) with a target mass set,
    eps, and a P_nl callable to produce beta^NL(r, M_i, M_j).

    Returns the same dict shape as `compute_beta_nl_xi`.
    """
    log10M_targets = np.asarray(log10M_targets, dtype=np.float64)
    log10M_th = threshold_result["log10M_th"]

    # Sanity: target masses should lie close to the tabulated ladder.
    for lM in log10M_targets:
        nearest = log10M_th[np.argmin(np.abs(log10M_th - (lM + np.log10(1.0 - eps))))]
        if abs(nearest - (lM + np.log10(1.0 - eps))) > 0.5 * (
            np.diff(log10M_th).min() if len(log10M_th) > 1 else 0.0
        ) + 1e-6:
            print(f"[beta_nl_xi_from_tabulation] warning: target log10M={lM:.3f} "
                  f"not on tabulated ladder; snapping to nearest threshold.")

    fixed = xi_hh_at_mass(threshold_result, log10M_targets, eps=eps)
    r = fixed["r"]
    xi_hh_dir = fixed["xi_hh"]
    xi_hh = xi_hh_dir.copy()

    xi_tree = None
    used_de = None
    if stitch_large_scale:
        if emu is None or P_lin_func is None or z is None:
            raise ValueError(
                "stitch_large_scale=True requires emu, P_lin_func, and z."
            )
        # Need a provisional bias for any low-mass linear-bias fallback;
        # fit it from the un-stitched ξ_hh first, then refit after stitching.
        r_mm_tmp, xi_mm_tmp = xi_mm_from_pk(P_nl_func)
        good_tmp = (xi_mm_tmp > 0) & np.isfinite(xi_mm_tmp) & (r_mm_tmp > 0)
        xi_mm_func_tmp = lambda rr: np.exp(np.interp(np.log(rr),
                                                      np.log(r_mm_tmp[good_tmp]),
                                                      np.log(xi_mm_tmp[good_tmp])))
        K = len(log10M_targets)
        b_provisional = np.full(K, np.nan)
        for a in range(K):
            b_provisional[a] = fit_bias_largescale(
                r, xi_hh_dir[a, a, :], xi_mm_func_tmp,
                r_min=bias_window[0], r_max=bias_window[1],
            )

        tree_res = compute_xi_tree(
            emu=emu, P_lin_func=P_lin_func, z=float(z),
            r_out=r, log10M_targets=log10M_targets,
            eps=gamma_eps, k_grid=gamma_k_grid,
            gamma_min_log10M=gamma_min_log10M, fallback=gamma_fallback,
            b_hh=b_provisional,
        )
        xi_tree = tree_res["xi_tree"]
        used_de = tree_res["used_de"]
        xi_hh = stitch_xi_hh(r, xi_hh_dir, xi_tree, r_switch=r_switch)

    r_mm, xi_mm_grid = xi_mm_from_pk(P_nl_func)
    good = (xi_mm_grid > 0) & np.isfinite(xi_mm_grid) & (r_mm > 0)
    xi_mm = np.exp(np.interp(np.log(r), np.log(r_mm[good]),
                              np.log(xi_mm_grid[good])))
    xi_mm_func = lambda rr: np.exp(np.interp(np.log(rr),
                                              np.log(r_mm[good]),
                                              np.log(xi_mm_grid[good])))

    K = len(log10M_targets)
    b_hh = np.full(K, np.nan)
    for a in range(K):
        b_hh[a] = fit_bias_largescale(r, xi_hh[a, a, :], xi_mm_func,
                                       r_min=bias_window[0],
                                       r_max=bias_window[1])

    beta = np.full_like(xi_hh, np.nan)
    for a in range(K):
        if not np.isfinite(b_hh[a]):
            continue
        for b in range(K):
            if not np.isfinite(b_hh[b]):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                beta[a, b, :] = xi_hh[a, b, :] / (b_hh[a] * b_hh[b] * xi_mm) - 1.0

    out = {
        "r": r,
        "M_targets": fixed["M_targets"],
        "xi_hh": xi_hh,
        "xi_hh_dir": xi_hh_dir,
        "xi_mm": xi_mm,
        "b_hh": b_hh,
        "beta_nl": beta,
        "thresholds": np.asarray(log10M_th),
        "N_above": np.asarray(threshold_result["N_above"]),
        "bias_window": np.asarray(bias_window),
        "eps": eps,
        "stitched": bool(stitch_large_scale),
    }
    if stitch_large_scale:
        out["xi_tree"] = xi_tree
        out["used_de_for_gamma"] = used_de
        out["r_switch"] = float(r_switch)
    return out


def compute_beta_nl_xi(
    halo_positions: np.ndarray,
    halo_log10M: np.ndarray,
    log10M_targets: Sequence[float],
    Lbox: float,
    P_nl_func: Callable[[np.ndarray], np.ndarray],
    r_bins: Optional[np.ndarray] = None,
    eps: float = 0.01,
    bias_window: tuple = (30.0, 80.0),
    los: str = "z",
    cache_path: Optional[str] = None,
    log10M_thresholds: Optional[Sequence[float]] = None,
    verbose: bool = True,
    stitch_large_scale: bool = False,
    emu=None,
    P_lin_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    z: Optional[float] = None,
    r_switch: float = 60.0,
    gamma_eps: float = 0.02,
    gamma_min_log10M: float = 12.0,
    gamma_fallback: str = "linear_bias",
    gamma_k_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Top-level driver: measure beta^NL(r, M_i, M_j) from a halo catalog at
    delta-function target masses, mirroring Dark Emulator's convention.

    Returns
    -------
    dict with:
        r                 (n_r,)         bin centers, Mpc/h
        M_targets         (K,)           target masses, M_sun/h
        xi_hh             (K, K, n_r)
        xi_mm             (n_r,)         interpolated to r
        b_hh              (K,)           large-scale bias per target mass
        beta_nl           (K, K, n_r)    xi_hh / (b_i b_j xi_mm) - 1
        thresholds        (T,)           thresholds used
        N_above           (T,)
    """
    if r_bins is None:
        # Default: 0.5 -> 100 Mpc/h, 30 log bins. Avoid below ~halo size.
        r_bins = np.logspace(np.log10(0.5), np.log10(100.0), 31)

    log10M_targets = np.asarray(log10M_targets, dtype=np.float64)

    if log10M_thresholds is None:
        log10M_thresholds = thresholds_for_targets(log10M_targets, eps=eps)
    log10M_thresholds = np.asarray(log10M_thresholds, dtype=np.float64)

    if verbose:
        print(f"[beta_nl_xi] {len(log10M_thresholds)} thresholds "
              f"for {len(log10M_targets)} targets")

    thr_res = tabulate_xi_hh_thresholds(
        halo_positions, halo_log10M, log10M_thresholds, Lbox, r_bins,
        cache_path=cache_path, los=los, verbose=verbose,
    )

    return beta_nl_xi_from_tabulation(
        thr_res, log10M_targets, P_nl_func,
        eps=eps, bias_window=bias_window,
        stitch_large_scale=stitch_large_scale, emu=emu,
        P_lin_func=P_lin_func, z=z, r_switch=r_switch,
        gamma_eps=gamma_eps, gamma_min_log10M=gamma_min_log10M,
        gamma_fallback=gamma_fallback, gamma_k_grid=gamma_k_grid,
    )


__all__ = [
    "measure_xi_hh_thresholds",
    "xi_hh_at_mass",
    "thresholds_for_targets",
    "dense_thresholds",
    "tabulate_xi_hh_thresholds",
    "gamma_h_at_mass_shells",
    "compute_xi_tree",
    "stitch_xi_hh",
    "beta_nl_xi_from_tabulation",
    "xi_mm_from_pk",
    "fit_bias_largescale",
    "compute_beta_nl_xi",
]
