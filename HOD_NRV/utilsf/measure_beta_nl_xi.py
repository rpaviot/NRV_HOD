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
    max_per_sample: Optional[int] = None,
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
        N_orig = len(sub)
        if max_per_sample is not None and N_orig > int(max_per_sample):
            # Deterministic per-threshold seed so independent thresholds
            # get independent realisations but reruns are reproducible.
            seed = (0xC0FFEE
                    ^ (hash((float(lt), int(max_per_sample))) & 0xFFFFFFFF))
            rng = np.random.default_rng(seed)
            idx = rng.choice(N_orig, size=int(max_per_sample), replace=False)
            sub = sub[idx]
            if verbose:
                print(f"  downsampled threshold log10M >= {lt:.4f}: "
                      f"{N_orig} -> {len(sub)}")
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


def _interpolate_xi_tree_from_grid(
    grid_path: str,
    r_out: np.ndarray,
    log10M_targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Load a pre-computed ξ_tree grid and interpolate to target masses and r.

    The grid file (.npz) must contain: r, log10M_arr, xi_tree, used_de,
    z, eps, gamma_min_log10M (as saved by precompute_xi_tree_grid.py).
    """
    from scipy.interpolate import RegularGridInterpolator

    d = np.load(grid_path)
    r_grid = d["r"]
    log10M_grid = d["log10M_arr"]
    xi_grid = d["xi_tree"]  # (n_mass, n_mass, n_r)

    K = len(log10M_targets)
    n_r_out = len(r_out)

    log_r_grid = np.log(r_grid)
    log_r_out = np.log(r_out)

    xi_tree = np.full((K, K, n_r_out), np.nan)
    used_de = np.ones(K, dtype=bool)

    for i in range(K):
        for j in range(i, K):
            interp = RegularGridInterpolator(
                (log10M_grid, log10M_grid, log_r_grid),
                xi_grid,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            pts = np.column_stack([
                np.full(n_r_out, log10M_targets[i]),
                np.full(n_r_out, log10M_targets[j]),
                log_r_out,
            ])
            xi_tree[i, j, :] = interp(pts)
            if j != i:
                xi_tree[j, i, :] = xi_tree[i, j, :]

    return {"r": r_out, "xi_tree": xi_tree, "used_de": used_de}


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
    grid_path: Optional[str] = None,
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
    grid_path       : optional path to a pre-computed ξ_tree grid (.npz
                      from precompute_xi_tree_grid.py). If provided,
                      interpolates from the grid instead of computing
                      from scratch.

    Returns
    -------
    dict with:
        r          (n_r,)            echoed r_out
        xi_tree    (K, K, n_r)       ξ_tree at fixed (M_i, M_j)
        used_de    (K,)              bool, True if Γ_h was used for that
                                     target (False if fallback fired).
    """
    log10M_targets = np.atleast_1d(np.asarray(log10M_targets, dtype=np.float64))
    r_out = np.asarray(r_out, dtype=np.float64)

    if grid_path is not None:
        return _interpolate_xi_tree_from_grid(grid_path, r_out, log10M_targets)

    K = len(log10M_targets)
    n_r = len(r_out)

    if k_grid is None:
        k_grid = np.logspace(-4.0, np.log10(5e2), 1024)
    k_grid = np.asarray(k_grid, dtype=np.float64)

    used_de = log10M_targets >= gamma_min_log10M

    xi_lin_at_r = None
    if (~used_de).any() and fallback == "linear_bias":
        P_lin_z = np.asarray(P_lin_func(k_grid), dtype=np.float64)
        r_lin, xi_lin = Pk_to_xi_gg(k_grid, P_lin_z)
        good = (r_lin > 0) & np.isfinite(xi_lin)
        xi_lin_at_r = np.interp(np.log(r_out),
                                 np.log(np.asarray(r_lin)[good]),
                                 np.asarray(xi_lin)[good])

    xi_tree = np.full((K, K, n_r), np.nan)

    de_idx = np.where(used_de)[0]
    if len(de_idx) > 0:
        # DE propagator Γ_h absorbs the growth factor, so P_lin must be at z=0.
        P_lin_z0 = np.asarray(emu.get_pklin(k_grid), dtype=np.float64).ravel()

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
                P_mm = gm[ii] * gm[jj] * P_lin_z0
                P_mp = gm[ii] * gp[jj] * P_lin_z0
                P_pm = gp[ii] * gm[jj] * P_lin_z0
                P_pp = gp[ii] * gp[jj] * P_lin_z0
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

    NaN handling: where ξ_dir is NaN (e.g. r > r_max_pycorr) the result is
    pure ξ_tree; where ξ_tree is NaN the result is pure ξ_dir; where both
    are NaN the result is NaN. Without this, `xi_dir * w_eff` is NaN even
    at r ≫ r_switch where `w_eff` is essentially zero, which silently
    truncates the stitched grid at r_max_pycorr.

    Shapes: r is (n_r,), xi_dir / xi_tree broadcast over any leading axes.
    """
    r = np.asarray(r, dtype=np.float64)
    w = np.exp(-(r / float(r_switch)) ** 4)
    dir_nan = ~np.isfinite(xi_dir)
    tree_nan = ~np.isfinite(xi_tree)
    xi_dir_safe = np.where(dir_nan, 0.0, xi_dir)
    xi_tree_safe = np.where(tree_nan, 0.0, xi_tree)
    # All weight to the surviving side when the other is NaN.
    w_dir = np.where(dir_nan, 0.0, np.where(tree_nan, 1.0, w))
    w_tree = np.where(tree_nan, 0.0, np.where(dir_nan, 1.0, 1.0 - w))
    out = xi_dir_safe * w_dir + xi_tree_safe * w_tree
    return np.where(dir_nan & tree_nan, np.nan, out)


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
    max_per_sample: Optional[int] = None,
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

    cache_max = -1 if max_per_sample is None else int(max_per_sample)

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
                and int(d["max_per_sample"]) == cache_max
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
        los=los, max_per_sample=max_per_sample, verbose=verbose,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(
            cache_path,
            r=res["r"], log10M_th=res["log10M_th"],
            N_above=res["N_above"], xi=res["xi"],
            r_bins=r_bins, Lbox=float(Lbox), los=str(los),
            catalog_sha=catalog_sha,
            max_per_sample=cache_max,
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
) -> Dict[str, np.ndarray]:
    """
    Cheap post-processing: combine a threshold ξ_hh tabulation with a
    target mass set, eps, and a P_nl callable to produce β^NL(r, M_i, M_j).

    `threshold_result` is either:
      - the output of `tabulate_xi_hh_thresholds` (raw pycorr threshold ξ), or
      - the output of `load_xi_threshold_grid` (precomputed + already-stitched
        threshold ξ from `precompute_xi_tree_grid.py`).

    Both must carry the same keys: `r`, `log10M_th`, `N_above`, `xi`.

    Stitching (small + large scales) is the responsibility of whoever
    produced `threshold_result` — this function only does the four-corner
    FD and the β^NL math.
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
    xi_hh = fixed["xi_hh"]

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

    return {
        "r": r,
        "M_targets": fixed["M_targets"],
        "xi_hh": xi_hh,
        "xi_mm": xi_mm,
        "b_hh": b_hh,
        "beta_nl": beta,
        "thresholds": np.asarray(log10M_th),
        "N_above": np.asarray(threshold_result["N_above"]),
        "bias_window": np.asarray(bias_window),
        "eps": eps,
    }


def load_xi_threshold_grid(path: str) -> Dict[str, np.ndarray]:
    """
    Load a precomputed threshold ξ_hh grid (output of
    `precompute_xi_tree_grid.py`) and reshape it into the dict form
    expected by `beta_nl_xi_from_tabulation` and `xi_hh_at_mass`:

        {"r": (n_r,), "log10M_th": (T,), "N_above": (T,), "xi": (T, T, n_r)}

    Picks the stitched curve (`xi_threshold`) if present, falling back
    to `xi_pycorr`. Raises if `N_above` was not saved (older runs).
    """
    d = np.load(path, allow_pickle=False)
    xi = d["xi_threshold"] if "xi_threshold" in d.files else d["xi_pycorr"]
    if "N_above" not in d.files:
        raise KeyError(
            f"{path} has no N_above array — regenerate with the current "
            "precompute_xi_tree_grid.py (N_above is required by the "
            "four-corner FD in xi_hh_at_mass)."
        )
    return {
        "r": np.asarray(d["r"]),
        "log10M_th": np.asarray(d["log10M_arr"]),
        "N_above": np.asarray(d["N_above"]),
        "xi": np.asarray(xi),
    }


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
) -> Dict[str, np.ndarray]:
    """
    Top-level driver: measure β^NL(r, M_i, M_j) from a halo catalog at
    delta-function target masses, mirroring Dark Emulator's convention.

    Stitching (small + large scales) is now owned by
    `precompute_xi_tree_grid.py`. Use that script if you want a
    pre-stitched threshold ξ; otherwise this driver returns β^NL built
    from the raw pycorr threshold tabulation.
    """
    if r_bins is None:
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
    )


def xi_hh_to_Pk_hh(
    r: np.ndarray,
    xi_hh: np.ndarray,
    k_out: Optional[np.ndarray] = None,
    n_int: int = 2048,
    N_extrap_low: int = 1024,
    N_extrap_high: int = 1024,
    fftlog_nu: float = 1.01,
) -> tuple:
    """
    Inverse Hankel transform ξ_hh(r, M_i, M_j) → P_hh(k, M_i, M_j) using
    Dark Emulator's `fftlog.xi2pk` (ν=1.01), matching the convention DE
    uses internally for `get_pknl` (de_interface.py:228-231).

    Each (M_i, M_j) slice is cubic-spline interpolated onto a uniform
    log-r grid spanning the measured range; FFTlog's `N_extrap_*`
    log-linear extension (default 1024 each side, matching DE's
    `xs = logspace(-3, 3, 2000)` + `N_extrap_low=1024` convention) then
    pads the transform to the asymptotic support required.

    The input ξ_hh **must already be stitched** (small-scale direct +
    large-scale tree) — i.e. produced by `precompute_xi_tree_grid.py`
    and loaded via `load_xi_threshold_grid`, then collapsed to fixed
    mass with `xi_hh_at_mass`. Stitching at fixed mass would corrupt
    the four-corner FD, so it must happen at threshold level.

    Parameters
    ----------
    r       : (n_r,) measured separations, Mpc/h. Need not be log-spaced.
    xi_hh   : (..., n_r). Trailing axis is r; leading axes broadcast.
    k_out   : optional output wavenumbers, h/Mpc. If None, the native
              FFTlog k grid is returned.
    n_int   : number of points on the uniform log-r grid fed to fftlog.
    N_extrap_low, N_extrap_high : FFTlog log-linear extrapolation widths
              (DE default 1024). These extend the integral support far
              enough that the answer over the native k window converges.
    fftlog_nu : FFTlog bias parameter (DE uses 1.01).

    Returns
    -------
    k       : 1D array of wavenumbers, h/Mpc.
    P_hh    : array with same leading shape as xi_hh, trailing axis k.
    """
    try:
        from dark_emulator import fftlog as _fftlog
    except ImportError as e:
        raise ImportError(
            "dark_emulator is required for xi_hh_to_Pk_hh — its fftlog module "
            "implements the same ξ↔P convention used by the emulator."
        ) from e
    from scipy.interpolate import CubicSpline

    r = np.asarray(r, dtype=np.float64)
    xi_hh = np.asarray(xi_hh, dtype=np.float64)
    assert xi_hh.shape[-1] == r.size, "trailing axis of xi_hh must match r"

    lead_shape = xi_hh.shape[:-1]
    xi_flat = xi_hh.reshape(-1, r.size)
    n_slices = xi_flat.shape[0]

    k_out_native = None
    Pk_list = []

    for s in range(n_slices):
        xi_s = xi_flat[s]
        finite = np.isfinite(xi_s)
        # Skip slices where the FD produced no usable values (e.g. dN=0 rows
        # because the target log10M isn't on the threshold ladder).
        if finite.sum() < 4:
            Pk_list.append(None)
            continue

        # Build a uniform log-r grid that lies *inside* the finite support of
        # this slice. fftlog then handles all extension to small/large r
        # via its log-linear N_extrap_* padding (DE convention). ξ can be
        # negative around BAO so spline on the value (not log ξ); isolated
        # NaN bins inside the finite range are bridged by the spline.
        r_fin = r[finite]
        xi_fin = xi_s[finite]
        r_int = np.logspace(np.log10(r_fin.min()), np.log10(r_fin.max()), n_int)
        cs = CubicSpline(np.log(r_fin), xi_fin, extrapolate=False)
        xi_int = cs(np.log(r_int))

        k_s, Pk_s = _fftlog.xi2pk(
            r_int, xi_int, fftlog_nu,
            N_extrap_low=N_extrap_low,
            N_extrap_high=N_extrap_high,
        )
        # Each slice has its own native k grid (different r support → different
        # k extent). To keep a single (k, P) array across slices, interpolate
        # each onto a common grid built from the first valid slice.
        if k_out_native is None:
            k_out_native = np.asarray(k_s)
            Pk_list.append(np.asarray(Pk_s))
        else:
            k_s = np.asarray(k_s); Pk_s = np.asarray(Pk_s)
            log_k0 = np.log(k_out_native)
            log_ks = np.log(k_s)
            good = np.isfinite(Pk_s) & (Pk_s > 0)
            if good.sum() < 4:
                Pk_list.append(np.interp(log_k0, log_ks, Pk_s))
            else:
                Pk_list.append(np.exp(np.interp(log_k0, log_ks[good],
                                                 np.log(Pk_s[good]))))

    if k_out_native is None:
        raise ValueError(
            "xi_hh_to_Pk_hh: every (M_i, M_j) slice of xi_hh is non-finite. "
            "The four-corner FD upstream produced no valid output — typically "
            "because the target log10M values don't lie on the threshold "
            "ladder of the precomputed grid (so (1±eps)·M snaps to the same "
            "ladder rung and dN = 0)."
        )
    n_k_native = k_out_native.size
    Pk_arr = np.full((n_slices, n_k_native), np.nan)
    for s, Pk in enumerate(Pk_list):
        if Pk is not None:
            Pk_arr[s] = Pk
    Pk_arr = Pk_arr.reshape(*lead_shape, n_k_native)

    if k_out is None:
        return k_out_native, Pk_arr

    k_out = np.asarray(k_out, dtype=np.float64)
    Pk_at_kout = np.full(lead_shape + (k_out.size,), np.nan)
    log_k_native = np.log(k_out_native)
    log_k_out = np.log(k_out)
    flat_kout = Pk_arr.reshape(-1, n_k_native)
    out_flat = Pk_at_kout.reshape(-1, k_out.size)
    for s in range(flat_kout.shape[0]):
        Pk_s = flat_kout[s]
        if not np.all(np.isfinite(Pk_s)):
            continue
        # interpolate |P| in log-log when positive; fall back to linear in P
        # for negative-P excursions (numerical ringing at very small/large k).
        good = np.isfinite(Pk_s)
        if Pk_s[good].min() > 0:
            out_flat[s] = np.exp(np.interp(log_k_out,
                                             log_k_native[good],
                                             np.log(Pk_s[good])))
        else:
            out_flat[s] = np.interp(log_k_out, log_k_native[good], Pk_s[good])
    return k_out, Pk_at_kout


def _xi_threshold_to_pk_threshold(
    threshold_result: Dict[str, np.ndarray],
    k_out: Optional[np.ndarray] = None,
    r_excl: float = 0.01,
    r_far: float = 2000.0,
    xi_far: float = 1e-12,
    n_int: int = 4000,
    N_extrap_low: int = 1024,
    N_extrap_high: int = 1024,
    fftlog_nu: float = 1.01,
) -> tuple:
    """
    Inverse Hankel each cumulative-threshold ξ_hh(>M_a, >M_b, r) slice to
    P_hh(k, >M_a, >M_b). Mirrors DE's `_get_phh_direct` (de_interface.py:403-409):
    fftlog is fed ξ over a wide r grid (DE uses logspace(-3, 3, 4000)) so
    the transform converges with the physical small/large-r asymptotics.

    Two anchors enforce those asymptotics before the cubic spline + fftlog:

      * Exclusion anchor: ξ(r=r_excl) = -1. Halos cannot get closer than a
        couple of virial radii; below the smallest pycorr bin ξ → -1.
      * Far-field anchor:  ξ(r=r_far)  =  0. Beyond the box the correlation
        is zero.

    Without these anchors, `fftlog.N_extrap_low` does log-linear extension
    from r_min, locking in whatever value/noise sits in the smallest bin —
    which under the IHT becomes a constant P(k) tail at high k.
    """
    from scipy.interpolate import CubicSpline
    try:
        from dark_emulator import fftlog as _fftlog
    except ImportError as e:
        raise ImportError(
            "dark_emulator is required for the fftlog convention used by DE."
        ) from e

    r = np.asarray(threshold_result["r"], dtype=np.float64)
    xi_thr = np.asarray(threshold_result["xi"], dtype=np.float64)
    T = xi_thr.shape[0]

    r_int = np.logspace(np.log10(r_excl), np.log10(r_far), n_int)
    log_r_int = np.log(r_int)

    k_out_native = None
    P_thr_native = None

    for i in range(T):
        for j in range(T):
            xi_s = xi_thr[i, j]
            finite = np.isfinite(xi_s)
            if finite.sum() < 4:
                continue

            r_fin = r[finite]
            # Physical floor at -1 (clip residual numerical undershoot
            # from the four-corner stitch or pycorr noise).
            xi_fin = np.clip(xi_s[finite], -1.0, None)

            r_anc = np.concatenate(([r_excl], r_fin, [r_far]))
            xi_anc = np.concatenate(([-1.0], xi_fin, [xi_far]))
            cs = CubicSpline(np.log(r_anc), xi_anc, extrapolate=False)
            xi_int = cs(log_r_int)
            # Floor: physical exclusion limit at small r.
            xi_int = np.clip(xi_int, -1.0, None)
            # Ceiling near the far anchor: cubic spline can overshoot
            # below the anchor (positive→0 transition). fftlog's
            # log_extrap(fx, N_extrap_high>0) then computes
            # log(fx[-1]/fx[-2]); if fx[-2] ≤ 0 the ratio is non-positive
            # and the extension fills with NaN that propagates to the
            # whole P(k) slice. Clamp the trailing region to xi_far so
            # fx stays strictly positive on the upper tail.
            n_tail = max(2, n_int // 20)
            xi_int[-n_tail:] = np.clip(xi_int[-n_tail:], xi_far, None)

            k_s, Pk_s = _fftlog.xi2pk(
                r_int, xi_int, fftlog_nu,
                N_extrap_low=N_extrap_low,
                N_extrap_high=N_extrap_high,
            )
            if k_out_native is None:
                k_out_native = np.asarray(k_s)
                P_thr_native = np.full((T, T, k_out_native.size), np.nan)
            P_thr_native[i, j] = np.asarray(Pk_s)

    if k_out_native is None:
        raise ValueError(
            "_xi_threshold_to_pk_threshold: every threshold ξ slice is "
            "non-finite — nothing to transform."
        )

    if k_out is None:
        return k_out_native, P_thr_native

    k_out = np.asarray(k_out, dtype=np.float64)
    P_thr_kout = np.full((T, T, k_out.size), np.nan)
    log_kn = np.log(k_out_native)
    log_ko = np.log(k_out)
    for i in range(T):
        for j in range(T):
            Ps = P_thr_native[i, j]
            if not np.all(np.isfinite(Ps)):
                continue
            if (Ps > 0).all():
                P_thr_kout[i, j] = np.exp(np.interp(log_ko, log_kn, np.log(Ps)))
            else:
                P_thr_kout[i, j] = np.interp(log_ko, log_kn, Ps)
    return k_out, P_thr_kout


def _pk_mass_from_pk_threshold(
    P_thr: np.ndarray,
    log10M_th: np.ndarray,
    N_above: np.ndarray,
    log10M_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Four-corner FD in k-space (DE's `get_phh_mass`, de_interface.py:502-519).
    Combines threshold P_hh(k, >M_a, >M_b) into fixed-mass P_hh(k, M_i, M_j).
    Returns array of shape (K, K, n_k).
    """
    K = len(log10M_targets)
    n_k = P_thr.shape[-1]
    P_hh = np.full((K, K, n_k), np.nan)

    for a, lMa in enumerate(log10M_targets):
        ima, ipa = _pair_indices_for_target(log10M_th, lMa, eps)
        Nma = float(N_above[ima]); Npa = float(N_above[ipa])
        for b, lMb in enumerate(log10M_targets):
            imb, ipb = _pair_indices_for_target(log10M_th, lMb, eps)
            Nmb = float(N_above[imb]); Npb = float(N_above[ipb])

            num = (Nma * Nmb * P_thr[ima, imb]
                   - Nma * Npb * P_thr[ima, ipb]
                   - Npa * Nmb * P_thr[ipa, imb]
                   + Npa * Npb * P_thr[ipa, ipb])
            denom = Nma * Nmb - Nma * Npb - Npa * Nmb + Npa * Npb
            if abs(denom) < 1e-30:
                continue
            P_hh[a, b, :] = num / denom
    return P_hh


def beta_nl_k_from_tabulation(
    threshold_result: Dict[str, np.ndarray],
    log10M_targets: Sequence[float],
    P_lin_func: Callable[[np.ndarray], np.ndarray],
    k_out: Optional[np.ndarray] = None,
    eps: float = 0.01,
    k_lin: float = 0.02,
    force_to_zero: str = "additive",
    n_int: int = 4000,
    N_extrap_low: int = 1024,
    N_extrap_high: int = 1024,
    fftlog_nu: float = 1.01,
    r_excl: float = 0.01,
    r_far: float = 2000.0,
) -> Dict[str, np.ndarray]:
    """
    Build β^NL(k, M_i, M_j) from a **stitched** threshold ξ_hh tabulation,
    mirroring `HOD_analytical.emu.BetaNLInterpolator` but with the
    emulator's `get_phh_mass` replaced by the simulation measurement.

    Pipeline:
        threshold ξ_hh(>M_a,>M_b,r)   (already stitched dir + tree)
          ──FD (xi_hh_at_mass)──▶     ξ_hh(r, M_i, M_j)
          ──xi2pk (DE fftlog)──▶      P_hh(k, M_i, M_j)
          ──halo-halo bias──▶         b(M) = √[P_hh(k_lin, M, M)/P_lin(k_lin)]
          ──force-to-zero──▶          β^NL = P_hh/(b_i b_j P_lin) - 1

    `threshold_result` must come from `load_xi_threshold_grid` (which loads
    the stitched grid built by `precompute_xi_tree_grid.py`) or otherwise
    carry pre-stitched threshold ξ. Stitching at fixed mass would corrupt
    the four-corner FD, so it must happen at threshold level.

    Parameters
    ----------
    threshold_result : dict with keys r, log10M_th, N_above, xi.
    log10M_targets   : (K,) target log10(M / [M_sun/h]).
    P_lin_func       : callable, k (h/Mpc) → P_lin(k, z) in (Mpc/h)^3,
                       at the snapshot redshift.
    k_out            : optional output wavenumbers in h/Mpc. If None,
                       returns on the native FFTlog k grid.
    eps              : four-corner FD half-step (matches emulator's 0.01).
    k_lin            : "linear" pivot for bias / force-to-zero (DE: 0.02).
    force_to_zero    : 'additive' (default), 'multiplicative',
                       'exponential', or 'none'. Same definitions as
                       `BetaNLInterpolator._apply_force_to_zero`.
    n_int, N_extrap_*, fftlog_nu : see `xi_hh_to_Pk_hh`.

    Returns
    -------
    dict with:
        k         (n_k,)              wavenumbers, h/Mpc
        M_targets (K,)                M_sun/h
        xi_hh     (K, K, n_r)         fixed-mass ξ from the FD (echoed)
        r         (n_r,)              r grid of xi_hh (echoed)
        P_hh      (K, K, n_k)         halo-halo P(k, M_i, M_j)
        P_lin     (n_k,)              linear P(k) at the output k
        b_hh      (K,)                halo-halo bias b(M_i)
        beta_nl   (K, K, n_k)         β^NL after force-to-zero
        eps, k_lin, force_to_zero     echoed
    """
    if force_to_zero not in ("none", "additive", "multiplicative", "exponential"):
        raise ValueError(f"unknown force_to_zero: {force_to_zero!r}")

    log10M_targets = np.atleast_1d(np.asarray(log10M_targets, dtype=np.float64))
    log10M_th = np.asarray(threshold_result["log10M_th"], dtype=np.float64)
    N_above = np.asarray(threshold_result["N_above"], dtype=np.float64)
    K = len(log10M_targets)
    M_targets = 10.0 ** log10M_targets

    # 1. HT each cumulative-threshold ξ slice (DE convention:
    #    de_interface.py:411-442). Wide r grid + exclusion/zero anchors
    #    so the inverse HT converges to P→0 at high k.
    k_native, P_thr_native = _xi_threshold_to_pk_threshold(
        threshold_result, k_out=None,
        r_excl=r_excl, r_far=r_far,
        n_int=n_int,
        N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high,
        fftlog_nu=fftlog_nu,
    )

    # 2. Four-corner FD in k-space → fixed-mass P_hh(k, M_i, M_j)
    P_hh_native = _pk_mass_from_pk_threshold(
        P_thr_native, log10M_th, N_above, log10M_targets, eps,
    )

    if k_out is None:
        k = k_native
        P_hh = P_hh_native
    else:
        k = np.asarray(k_out, dtype=np.float64)
        log_kn = np.log(k_native); log_ko = np.log(k)
        P_hh = np.full((K, K, k.size), np.nan)
        for i in range(K):
            for j in range(K):
                Ps = P_hh_native[i, j]
                if not np.all(np.isfinite(Ps)):
                    continue
                if (Ps > 0).all():
                    P_hh[i, j] = np.exp(np.interp(log_ko, log_kn, np.log(Ps)))
                else:
                    P_hh[i, j] = np.interp(log_ko, log_kn, Ps)

    # 3. linear power on the output grid + at k_lin
    P_lin = np.asarray(P_lin_func(k), dtype=np.float64).ravel()
    P_lin_klin = float(np.asarray(P_lin_func(np.array([k_lin]))).ravel()[0])

    # 4. halo-halo bias from the diagonal P_hh at k_lin
    log_kn = np.log(k_native)
    log_klin = np.log(k_lin)
    P_hh_klin_diag = np.array([
        np.interp(log_klin, log_kn, P_hh_native[a, a, :]) for a in range(K)
    ])
    with np.errstate(invalid="ignore"):
        b_hh = np.sqrt(P_hh_klin_diag / P_lin_klin)

    # 5. β^NL = P_hh / (b_i b_j P_lin) - 1, then force-to-zero
    beta = np.full_like(P_hh, np.nan)
    for i in range(K):
        if not np.isfinite(b_hh[i]):
            continue
        for j in range(K):
            if not np.isfinite(b_hh[j]):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                beta_ij = P_hh[i, j, :] / (b_hh[i] * b_hh[j] * P_lin) - 1.0

            if force_to_zero in ("additive", "multiplicative"):
                P_hh_ij_klin = np.interp(log_klin, log_kn, P_hh_native[i, j, :])
                offset = P_hh_ij_klin / (b_hh[i] * b_hh[j] * P_lin_klin) - 1.0
                if force_to_zero == "additive":
                    beta_ij = beta_ij - offset
                else:
                    beta_ij = (beta_ij + 1.0) / (offset + 1.0) - 1.0
            elif force_to_zero == "exponential":
                beta_ij = beta_ij * (1.0 - np.exp(-(k / k_lin) ** 2))
            # 'none' → leave as is

            beta[i, j, :] = beta_ij

    # Echo fixed-mass ξ for backward-compat consumers (built via the FD-in-r
    # path; same operator as before, just kept for reporting/plotting).
    fixed = xi_hh_at_mass(threshold_result, log10M_targets, eps=eps)

    return {
        "k": k,
        "M_targets": M_targets,
        "r": fixed["r"],
        "xi_hh": fixed["xi_hh"],
        "P_hh": P_hh,
        "P_lin": P_lin,
        "b_hh": b_hh,
        "beta_nl": beta,
        "eps": eps,
        "k_lin": k_lin,
        "force_to_zero": force_to_zero,
    }


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
    "load_xi_threshold_grid",
    "xi_mm_from_pk",
    "fit_bias_largescale",
    "compute_beta_nl_xi",
    "xi_hh_to_Pk_hh",
    "beta_nl_k_from_tabulation",
]
