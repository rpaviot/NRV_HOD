"""
Measure beta^NL(k, M1, M2) directly from a halo catalog.

Mirrors the conventions in `emu.py` (BetaNLInterpolator):

    P_hh(k, M1, M2) = b(M1) * b(M2) * P_lin(k) * [1 + beta^NL]
    b(M)            = sqrt(P_hh(k_lin, M, M) / P_lin(k_lin))    [halo-halo]
    beta^NL_corr    = beta^NL_raw - beta^NL_raw(k_lin)          [additive force-to-zero]

So measurements from a snapshot can be plotted directly against the emulator.

All masses in M_sun/h, k in h/Mpc, lengths in Mpc/h.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from ..utilsf.measure_pk import PowerSpectrumEstimator


FORCE_TO_ZERO_METHODS = ("none", "additive", "multiplicative", "exponential")


def _interp_at_klin(k: np.ndarray, P: np.ndarray, k_lin: float) -> float:
    """Log-log linear interpolation of P(k) at k = k_lin."""
    good = np.isfinite(P) & (P > 0) & np.isfinite(k) & (k > 0)
    return float(np.exp(np.interp(np.log(k_lin), np.log(k[good]), np.log(P[good]))))


def measure_beta_nl_from_catalog(
    halo_positions: np.ndarray,
    halo_log10M: np.ndarray,
    Lbox: float,
    Nmesh: int,
    log10M_bins: np.ndarray,
    P_lin_func,
    kbins: Optional[np.ndarray] = None,
    k_lin: float = 0.02,
    force_to_zero: str = "additive",
    threads: int = 32,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Measure beta^NL(k, M1, M2) from a halo catalog.

    Parameters
    ----------
    halo_positions : (N, 3) array
        Halo positions in Mpc/h, in [0, Lbox).
    halo_log10M : (N,) array
        log10(M_halo) in M_sun/h.
    Lbox : float
        Box size in Mpc/h.
    Nmesh : int
        Mesh resolution.
    log10M_bins : (n_M+1,) array
        Mass-bin edges in log10(M_sun/h).
    P_lin_func : callable
        P_lin_func(k) -> P_lin(k, z) at the snapshot redshift, in (Mpc/h)^3,
        with k in h/Mpc. Pass e.g. `lambda k: cosmo.linear_power(k, z)`.
    kbins : array, optional
        k-bin edges. Defaults to log-spaced bins from k_fund to k_nyq.
    k_lin : float
        "Linear regime" scale used for halo-halo bias and force-to-zero
        (default 0.02 h/Mpc, matches emu.py).
    force_to_zero : {'none','additive','multiplicative','exponential'}
        Large-scale correction (default 'additive', matches emu.py).
    threads : int
        FFT thread count.

    Returns
    -------
    dict with:
        k          (n_k,)              bin centers, h/Mpc
        n_modes    (n_k,)
        M_centers  (n_M,)              10**(0.5*(edge_i + edge_{i+1})), M_sun/h
        N_per_bin  (n_M,)
        P_hh       (n_M, n_M, n_k)     shot-noise subtracted on diagonal
        b_hh       (n_M,)              halo-halo bias at k_lin
        P_lin      (n_k,)
        beta_nl    (n_M, n_M, n_k)     force-to-zero applied
    """
    if force_to_zero not in FORCE_TO_ZERO_METHODS:
        raise ValueError(f"force_to_zero must be in {FORCE_TO_ZERO_METHODS}")

    halo_positions = np.asarray(halo_positions)
    halo_log10M = np.asarray(halo_log10M)

    bin_idx = np.digitize(halo_log10M, log10M_bins) - 1
    n_M = len(log10M_bins) - 1
    M_centers = 10.0 ** (0.5 * (log10M_bins[:-1] + log10M_bins[1:]))

    estimator = PowerSpectrumEstimator(Nmesh=Nmesh, Lbox=Lbox, threads=threads)
    if kbins is None:
        from ..utilsf.measure_pk import log_kbins
        kbins = log_kbins(Lbox, Nmesh, n_bins=20)

    # Build delta_k per mass bin
    deltaks: list = [None] * n_M
    N_per_bin = np.zeros(n_M, dtype=np.int64)
    for i in range(n_M):
        sel = bin_idx == i
        N_per_bin[i] = int(sel.sum())
        if N_per_bin[i] < 2:
            if verbose:
                print(f"  bin {i}: only {N_per_bin[i]} halos, skipping")
            continue
        if verbose:
            print(f"  bin {i}: {N_per_bin[i]} halos, computing delta_k...")
        deltaks[i] = estimator.delta_k(halo_positions[sel])

    # Auto and cross spectra
    P_hh = np.full((n_M, n_M, len(kbins) - 1), np.nan)
    k_centers = None
    n_modes = None
    for i in range(n_M):
        if deltaks[i] is None:
            continue
        k_centers, P_ii, n_modes = estimator.auto_pk(
            deltaks[i], kbins, n_tracer=int(N_per_bin[i])
        )
        P_hh[i, i, :] = P_ii
        for j in range(i + 1, n_M):
            if deltaks[j] is None:
                continue
            _, P_ij, _ = estimator.cross_pk(deltaks[i], deltaks[j], kbins)
            P_hh[i, j, :] = P_ij
            P_hh[j, i, :] = P_ij

    # Linear power on the same k grid
    P_lin = np.asarray(P_lin_func(k_centers))
    P_lin_klin = float(P_lin_func(np.array([k_lin]))[0])

    # Halo-halo bias from auto-spectra at k_lin
    b_hh = np.full(n_M, np.nan)
    for i in range(n_M):
        if not np.isfinite(P_hh[i, i, :]).any():
            continue
        Phh_klin = _interp_at_klin(k_centers, P_hh[i, i, :], k_lin)
        if Phh_klin > 0 and P_lin_klin > 0:
            b_hh[i] = np.sqrt(Phh_klin / P_lin_klin)

    # Raw beta^NL
    beta_raw = np.full_like(P_hh, np.nan)
    for i in range(n_M):
        for j in range(n_M):
            if not (np.isfinite(b_hh[i]) and np.isfinite(b_hh[j])):
                continue
            denom = b_hh[i] * b_hh[j] * P_lin
            with np.errstate(divide="ignore", invalid="ignore"):
                beta_raw[i, j, :] = P_hh[i, j, :] / denom - 1.0

    # Force-to-zero correction (mirrors emu.py:_apply_force_to_zero)
    beta_nl = beta_raw.copy()
    if force_to_zero != "none":
        for i in range(n_M):
            for j in range(n_M):
                bij = beta_raw[i, j, :]
                if not np.isfinite(bij).any():
                    continue
                if force_to_zero == "exponential":
                    beta_nl[i, j, :] = bij * (1.0 - np.exp(-(k_centers / k_lin) ** 2))
                else:
                    Phh_ij_klin = _interp_at_klin(k_centers, P_hh[i, j, :], k_lin)
                    offset = Phh_ij_klin / (b_hh[i] * b_hh[j] * P_lin_klin) - 1.0
                    if force_to_zero == "additive":
                        beta_nl[i, j, :] = bij - offset
                    elif force_to_zero == "multiplicative":
                        beta_nl[i, j, :] = (bij + 1.0) / (offset + 1.0) - 1.0

    return {
        "k": k_centers,
        "n_modes": n_modes,
        "M_centers": M_centers,
        "N_per_bin": N_per_bin,
        "P_hh": P_hh,
        "b_hh": b_hh,
        "P_lin": P_lin,
        "beta_nl": beta_nl,
        "kbins": kbins,
        "k_lin": k_lin,
        "force_to_zero": force_to_zero,
    }
