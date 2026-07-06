"""
Numerical DeltaSigma Regression Test
=====================================

Fast regression test for the numerical HOD pipeline (population engine,
NFW profiles, two-point calculator).  Uses the optimal downsampling settings
from benchmark_results.json (5% particles, 10% galaxies, 5 realizations)
to run in ~3 s per realization instead of ~120 s.

The mean DeltaSigma is compared against the full-resolution baseline stored
in baseline_dsigma_cache.npz (10 realizations, 100% particles/galaxies).

Prerequisites:
  1. baseline_dsigma_cache.npz  (from numerical_dsigma_example.py)
  2. benchmark_results.json     (optimal settings source)

Data: Flamingo L1000N1800
  - Halos:     parquet_halo_catalogue_L1000N1800.parquet
  - Particles: particle_catalogue_L1000N1800_downsampled.parquet
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
    HaloCenterLensingCache, precompute_halo_center_lensing, TabulatedDeltaSigma,
)
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration (matches other common_test scripts)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

BASELINE_CACHE = "baseline_dsigma_cache.npz"
BENCHMARK_JSON = "benchmark_results.json"
# xi_gm-tabulated halo-center cache (full halo catalog, 5% particles).
# Built once on first run (~minutes); delete after changing particle
# fraction/seed, rp_bins, or BINS_COMP.
TABULATED_CACHE = "cross_check_tabulated_cache.h5"

PLOT_OUTPUT = "cross_check_numerical.png"
RESULTS_OUTPUT = "cross_check_numerical_results.txt"

Lbox = 681.0        # Mpc/h
zeff = 1.0
mass_definition = "MassDef200m"

column_mapping = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}

dict_cosmo = {
    "h": 0.681,
    "Omc": 0.306 - 0.0486 - 1.39e-3,
    "Omb": 0.0486,
    "A_s": 2.099e-9,
    "n_s": 0.967,
    "Omnu": 1.39e-3,
}

base_hod_params = {
    "As": 0.3,
    "Mmin": 12.7,
    "sig_M": 0.3,
    "M1": 13.0,
    "gamma":5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}
target_ngal = 2e-4  # (Mpc/h)^-3

# Lensing bins (same as baseline)
rp_bins = np.geomspace(0.1, 50.0, 16)
BINS_COMP = np.geomspace(5e-3, 120, 151)

# Use a different base seed from baseline (1000) to test statistical agreement
BASE_SEED = 2000
PARTICLE_SUBSAMPLE_SEED = 99

# Pass/fail threshold
THRESHOLD = 0.05  # 5% maximum deviation

# ============================================================================
# Helpers
# ============================================================================

def compute_deviation_metrics(ds_test, ds_ref):
    """Compute fractional deviation metrics between test and reference."""
    mask = np.abs(ds_ref) > 1e-10
    frac_dev = np.abs((ds_test[mask] - ds_ref[mask]) / ds_ref[mask])
    return {
        "median_frac_dev": float(np.median(frac_dev)),
        "max_frac_dev": float(np.max(frac_dev)),
        "rms_frac_dev": float(np.sqrt(np.mean(frac_dev**2))),
    }


def fmt_time(seconds):
    """Pretty-print a duration."""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds/60:.1f} min"


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Numerical DeltaSigma Regression Test")
    print("=" * 70)

    # --- Load optimal settings from benchmark ---
    if not os.path.exists(BENCHMARK_JSON):
        raise FileNotFoundError(
            f"Benchmark results not found: {BENCHMARK_JSON}\n"
            f"Run the convergence benchmark first to generate it."
        )

    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)

    optimal = bench["optimal"]
    particle_fraction = optimal["particle_fraction"]
    galaxy_fraction = optimal["galaxy_fraction"]
    n_real= 10#optimal["n_realizations"]

    print(f"\nOptimal settings from {BENCHMARK_JSON}:")
    print(f"  Particle fraction: {particle_fraction*100:.0f}%")
    print(f"  Galaxy fraction:   {galaxy_fraction*100:.0f}%")
    print(f"  N realizations:    {n_real}")
    print(f"  Pass threshold:    {THRESHOLD*100:.0f}% max deviation")

    # --- Load baseline ---
    if not os.path.exists(BASELINE_CACHE):
        raise FileNotFoundError(
            f"Baseline cache not found: {BASELINE_CACHE}\n"
            f"Run numerical_dsigma_example.py first to generate it."
        )

    print(f"\nLoading baseline from {BASELINE_CACHE}...")
    baseline = np.load(BASELINE_CACHE)
    ds_baseline_all = baseline["ds_all"]
    rp_baseline = baseline["rp"]

    n_baseline = len(ds_baseline_all)
    ds_baseline_mean = ds_baseline_all.mean(axis=0)
    print(f"  {n_baseline} baseline realizations loaded")

    # --- Load catalogs ---
    print("\nLoading halo catalog...")
    t0 = time.perf_counter()
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo):,} halos loaded")

    print("Loading particle catalog...")
    df_part = pd.read_parquet(PARTICLE_PATH)
    load_time = time.perf_counter() - t0
    print(f"  {len(df_part):,} particles loaded")
    print(f"  Load time: {fmt_time(load_time)}")

    # --- Initialize HaloOccupation ---
    print("\nInitializing HaloOccupation...")
    halo = HaloOccupation(
        cosmology=dict_cosmo,
        zeff=zeff,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition=mass_definition,
        DataFrame=df_halo,
        DataFrame_part=df_part,
        assembly_bias=False,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False,
    )

    # --- Set HOD model and rescale Ac/As ---
    print("\nSetting HOD model: ELG_mHMQ")
    halo.set_halo_model("ELG_mHMQ")

    print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, base_hod_params, target_ngal=target_ngal
    )

    hod_params = base_hod_params.copy()
    hod_params["Ac"] = Ac_rescaled
    hod_params["As"] = As_rescaled

    print(f"  Ac = {Ac_rescaled[0]:.6f}")
    print(f"  As = {As_rescaled[0]:.6f}")

    # --- Downsample particles ---
    n_part_full = len(halo.positions_part)
    halo.positions_part = halo._subsample_array(
        halo.positions_part, particle_fraction, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticle downsampling: {n_part_full:,} -> "
          f"{len(halo.positions_part):,} ({particle_fraction*100:.0f}%)")

    # --- Tabulated halo-center cache (before JIT warmup: fork-safe) ---
    if os.path.exists(TABULATED_CACHE):
        print(f"\nLoading tabulated cache from {TABULATED_CACHE}...")
        tab_cache = HaloCenterLensingCache.load(TABULATED_CACHE)
        if (not tab_cache.has_tabulation
                or len(tab_cache.positions) != len(halo.logM)):
            raise ValueError(
                f"Stale tabulated cache {TABULATED_CACHE} — delete and re-run.")
    else:
        print(f"\nPrecomputing tabulated halo-center cache (one-time)...")
        t0 = time.perf_counter()
        tab_cache = precompute_halo_center_lensing(
            halo_positions=np.asarray(halo.positions),
            particle_positions=np.asarray(halo.positions_part),
            Lbox=Lbox,
            rsd_axis=halo.rsd_axis,
            RHO_M=halo.RHO_M,
            rp_bins=rp_bins,
            bins_comp=BINS_COMP,
            halo_logM=np.asarray(halo.logM),
        )
        tab_cache.save(TABULATED_CACHE)
        print(f"  Precompute time: {fmt_time(time.perf_counter() - t0)}")

    # --- JIT warmup ---
    print("\nWarming up JAX JIT (throw-away population)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup time: {fmt_time(warmup_time)}")

    # --- Run optimized realizations ---
    print(f"\nRunning {n_real} optimized realizations (seed base={BASE_SEED})...")
    t0 = time.perf_counter()
    rp_centers, ds_opt_mean, ds_opt_std = halo.compute_avg_lensing(
        hod_params, n_real, rp_bins,
        galaxy_fraction=galaxy_fraction,
        base_seed=BASE_SEED,
        bins_comp=BINS_COMP,
    )
    total_time = time.perf_counter() - t0

    # --- Tabulated prediction (noise-free, no population step) ---
    print("\nRunning tabulated prediction (TabulatedDeltaSigma)...")
    tab = TabulatedDeltaSigma(tab_cache, halo)
    t0 = time.perf_counter()
    rp_tab, ds_tab, tab_info = tab.predict(hod_params)
    tab_time = time.perf_counter() - t0
    print(f"  Predict time: {fmt_time(tab_time)}  "
          f"(fsat={tab_info['fsat']:.4f}, ngal={tab_info['ngal']:.3e})")

    # --- Comparison metrics ---
    print("\n" + "=" * 70)
    print("Comparison: Optimized MC & Tabulated vs Full-Resolution Baseline")
    print("=" * 70)

    metrics = compute_deviation_metrics(ds_opt_mean, ds_baseline_mean)
    metrics_tab = compute_deviation_metrics(ds_tab, ds_baseline_mean)

    print(f"\n  Settings: {particle_fraction*100:.0f}% particles, "
          f"{galaxy_fraction*100:.0f}% galaxies, {n_real} realizations")
    print(f"  Baseline: {n_baseline} realizations (100% particles/galaxies)")
    print(f"\n  Deviation metrics vs baseline:")
    print(f"    {'':12s}  {'Optimized MC':>14s}  {'Tabulated':>14s}")
    print(f"    {'Median':12s}  {metrics['median_frac_dev']*100:13.2f}%  "
          f"{metrics_tab['median_frac_dev']*100:13.2f}%")
    print(f"    {'Max':12s}  {metrics['max_frac_dev']*100:13.2f}%  "
          f"{metrics_tab['max_frac_dev']*100:13.2f}%")
    print(f"    {'RMS':12s}  {metrics['rms_frac_dev']*100:13.2f}%  "
          f"{metrics_tab['rms_frac_dev']*100:13.2f}%")

    # Per-bin deviation table
    mask = np.abs(ds_baseline_mean) > 1e-10
    per_bin_dev = np.zeros_like(ds_opt_mean)
    per_bin_dev[mask] = np.abs(
        (ds_opt_mean[mask] - ds_baseline_mean[mask]) / ds_baseline_mean[mask]
    )
    per_bin_dev_tab = np.zeros_like(ds_tab)
    per_bin_dev_tab[mask] = np.abs(
        (ds_tab[mask] - ds_baseline_mean[mask]) / ds_baseline_mean[mask]
    )

    header = (f"  {'rp [Mpc/h]':>14s}  {'Baseline':>14s}  {'Optimized':>14s}  "
              f"{'Dev':>8s}  {'Tabulated':>14s}  {'Dev':>8s}")
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for r, ds_b, ds_o, dev, ds_t, dev_t in zip(
            rp_centers, ds_baseline_mean, ds_opt_mean, per_bin_dev,
            ds_tab, per_bin_dev_tab):
        print(f"    {r:12.4f}  {ds_b:14.6e}  {ds_o:14.6e}  {dev*100:6.2f}%  "
              f"{ds_t:14.6e}  {dev_t*100:6.2f}%")

    # Timing
    print(f"\n  Timing:")
    print(f"    Optimized MC total ({n_real} realizations): {fmt_time(total_time)}")
    print(f"    Optimized MC per realization: {fmt_time(total_time / n_real)}")
    print(f"    Tabulated per call: {fmt_time(tab_time)}")

    # --- PASS/FAIL verdict ---
    max_dev = metrics["max_frac_dev"]
    max_dev_tab = metrics_tab["max_frac_dev"]
    passed_opt = max_dev < THRESHOLD
    passed_tab = max_dev_tab < THRESHOLD
    passed = passed_opt and passed_tab

    print(f"\n  {'=' * 40}")
    print(f"  Optimized MC: {'PASS' if passed_opt else 'FAIL'}  "
          f"(max deviation {max_dev*100:.2f}% vs {THRESHOLD*100:.0f}% threshold)")
    print(f"  Tabulated:    {'PASS' if passed_tab else 'FAIL'}  "
          f"(max deviation {max_dev_tab*100:.2f}% vs {THRESHOLD*100:.0f}% threshold)")
    print(f"  Overall:      {'PASS' if passed else 'FAIL'}")
    print(f"  {'=' * 40}")

    # --- Save results text ---
    with open(RESULTS_OUTPUT, "w") as f:
        f.write("# Numerical DeltaSigma Regression Test Results\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Settings: {particle_fraction*100:.0f}% particles, "
                f"{galaxy_fraction*100:.0f}% galaxies, {n_real} realizations\n")
        f.write(f"# Baseline: {n_baseline} realizations (100% particles/galaxies)\n")
        f.write(f"# Base seed: {BASE_SEED}\n")
        f.write(f"# Threshold: {THRESHOLD*100:.0f}%\n")
        f.write(f"# Result: {'PASS' if passed else 'FAIL'} "
                f"(optimized MC: {'PASS' if passed_opt else 'FAIL'}, "
                f"tabulated: {'PASS' if passed_tab else 'FAIL'})\n")
        f.write(f"#\n")
        f.write(f"# Deviations vs baseline      Optimized MC   Tabulated\n")
        f.write(f"# Median deviation: {metrics['median_frac_dev']*100:10.2f}%  "
                f"{metrics_tab['median_frac_dev']*100:9.2f}%\n")
        f.write(f"# Max deviation:    {metrics['max_frac_dev']*100:10.2f}%  "
                f"{metrics_tab['max_frac_dev']*100:9.2f}%\n")
        f.write(f"# RMS deviation:    {metrics['rms_frac_dev']*100:10.2f}%  "
                f"{metrics_tab['rms_frac_dev']*100:9.2f}%\n")
        f.write(f"#\n")
        f.write(f"# {'rp [Mpc/h]':>14s}  {'Baseline':>14s}  {'Optimized':>14s}  "
                f"{'Dev':>8s}  {'Tabulated':>14s}  {'Dev':>8s}\n")
        for r, ds_b, ds_o, dev, ds_t, dev_t in zip(
                rp_centers, ds_baseline_mean, ds_opt_mean, per_bin_dev,
                ds_tab, per_bin_dev_tab):
            f.write(f"  {r:14.6f}  {ds_b:14.6e}  {ds_o:14.6e}  {dev*100:6.2f}%  "
                    f"{ds_t:14.6e}  {dev_t*100:6.2f}%\n")
    print(f"\n  Saved results: {RESULTS_OUTPUT}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )

    # Top panel: DeltaSigma comparison
    ax1.loglog(rp_centers, ds_baseline_mean, "k-", lw=2,
               label=f"Raw baseline (mean of {n_baseline}, full res)")
    ax1.loglog(rp_centers, ds_opt_mean, "ro--", lw=1.5, ms=5,
               label=f"Optimized MC (mean of {n_real})")
    ax1.loglog(rp_tab, ds_tab, "b^-", lw=1.5, ms=5,
               label="Tabulated (noise-free)")

    # ±1σ band across realizations
    ax1.fill_between(rp_centers, ds_opt_mean - ds_opt_std, ds_opt_mean + ds_opt_std,
                     color="red", alpha=0.2)

    ax1.set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot/\mathrm{pc}^2$]")
    ax1.legend(loc="upper right")
    ax1.set_title("Numerical Regression Test: Optimized MC & Tabulated vs Baseline")

    # Bottom panel: fractional difference
    frac_diff = np.where(
        np.abs(ds_baseline_mean) > 1e-10,
        (ds_opt_mean - ds_baseline_mean) / ds_baseline_mean,
        0.0,
    )
    frac_diff_tab = np.where(
        np.abs(ds_baseline_mean) > 1e-10,
        (ds_tab - ds_baseline_mean) / ds_baseline_mean,
        0.0,
    )
    ax2.semilogx(rp_centers, frac_diff * 100, "ro-", ms=4, label="Optimized MC")
    ax2.semilogx(rp_centers, frac_diff_tab * 100, "b^-", ms=4, label="Tabulated")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.axhspan(-THRESHOLD * 100, THRESHOLD * 100, color="green", alpha=0.1,
                label=rf"$\pm {THRESHOLD*100:.0f}\%$")
    ax2.set_xlabel(r"$r_p$ [Mpc/h]")
    ax2.set_ylabel("Fractional diff [%]")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {PLOT_OUTPUT}")

    print("\nDone.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
