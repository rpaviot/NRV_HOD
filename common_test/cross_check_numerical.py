"""
Numerical DeltaSigma Regression Test
=====================================

Fast regression test for the numerical HOD pipeline (population engine,
NFW profiles, two-point calculator).  Uses the optimal downsampling settings
from benchmark_results.json (5% particles, 10% galaxies, 5 realizations,
use_fast=True) to run in ~3 s per realization instead of ~120 s.

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
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
    compute_galaxy_lensing,
)
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration (matches other common_test scripts)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

BASELINE_CACHE = "baseline_dsigma_cache.npz"
BENCHMARK_JSON = "benchmark_results.json"

PLOT_OUTPUT = "cross_check_numerical.png"
RESULTS_OUTPUT = "cross_check_numerical_results.txt"

Lbox = 681.0        # Mpc/h
zeff = 1.0
mass_definition = "200m"

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

cosmo_params = {
    "H0": dict_cosmo["h"] * 100,
    "Om0": dict_cosmo["Omc"] + dict_cosmo["Omb"] + dict_cosmo["Omnu"],
    "Ob0": dict_cosmo["Omb"],
    "sigma8": 0.807,
    "ns": 0.967,
}

base_hod_params = {
    "As": 0.3,
    "Mmin": 13.0,
    "sig_M": 0.3,
    "M1": 13.0,
    "alpha": 0.80,
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

def subsample_array(positions, fraction, seed):
    """Subsample positions array with sorted indices for cache locality."""
    if fraction >= 1.0:
        return positions
    n_keep = int(len(positions) * fraction)
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(positions), n_keep, replace=False))
    return positions[indices]


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
    n_real = optimal["n_realizations"]

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
        cosmology=cosmo_params,
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
    print("\nSetting HOD model: ELG_GHOD")
    halo.set_halo_model("ELG_GHOD")

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
    positions_part_full = np.array(halo.positions_part)
    positions_part_sub = subsample_array(
        positions_part_full, particle_fraction, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticle downsampling: {len(positions_part_full):,} -> "
          f"{len(positions_part_sub):,} ({particle_fraction*100:.0f}%)")

    # --- JIT warmup ---
    print("\nWarming up JAX JIT (throw-away population)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0, use_fast=True)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup time: {fmt_time(warmup_time)}")

    # --- Run optimized realizations ---
    print(f"\nRunning {n_real} optimized realizations (seed base={BASE_SEED})...")
    ds_opt_all = []
    timing_all = []
    ngal_all = []
    sat_frac_all = []

    for i in range(n_real):
        seed = BASE_SEED + i
        print(f"  Realization {i+1}/{n_real} (seed={seed})...", end=" ", flush=True)

        # Populate with use_fast=True
        t_pop = time.perf_counter()
        halo.populate_haloes(hod_params, random_seed=seed, use_fast=True)
        t_pop = time.perf_counter() - t_pop

        n_gal = len(halo.positions_gal)
        f_sat = halo.satellite_fraction
        ngal_all.append(n_gal)
        sat_frac_all.append(f_sat)

        # Subsample galaxies
        gal_pos = np.array(halo.positions_gal)
        gal_pos_sub = subsample_array(gal_pos, galaxy_fraction, seed=88 + i)

        # Compute lensing
        t_lens = time.perf_counter()
        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part_sub,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
        )
        t_lens = time.perf_counter() - t_lens

        ds_opt_all.append(ds)
        timing_all.append(t_pop + t_lens)

        print(f"N_gal={n_gal:,}, f_sat={f_sat:.3f}, "
              f"pop={fmt_time(t_pop)}, lens={fmt_time(t_lens)}, "
              f"total={fmt_time(t_pop + t_lens)}")

    ds_opt_all = np.array(ds_opt_all)
    timing_all = np.array(timing_all)
    ds_opt_mean = ds_opt_all.mean(axis=0)

    # --- Comparison metrics ---
    print("\n" + "=" * 70)
    print("Comparison: Optimized vs Full-Resolution Baseline")
    print("=" * 70)

    metrics = compute_deviation_metrics(ds_opt_mean, ds_baseline_mean)

    print(f"\n  Settings: {particle_fraction*100:.0f}% particles, "
          f"{galaxy_fraction*100:.0f}% galaxies, {n_real} realizations")
    print(f"  Baseline: {n_baseline} realizations (100% particles/galaxies)")
    print(f"\n  Mean ngal:      {np.mean(ngal_all):,.0f}")
    print(f"  Mean sat_frac:  {np.mean(sat_frac_all):.4f}")

    print(f"\n  Deviation metrics (mean of {n_real} vs mean of {n_baseline}):")
    print(f"    Median fractional:  {metrics['median_frac_dev']*100:.2f}%")
    print(f"    Max fractional:     {metrics['max_frac_dev']*100:.2f}%")
    print(f"    RMS fractional:     {metrics['rms_frac_dev']*100:.2f}%")

    # Per-bin deviation table
    mask = np.abs(ds_baseline_mean) > 1e-10
    per_bin_dev = np.zeros_like(ds_opt_mean)
    per_bin_dev[mask] = np.abs(
        (ds_opt_mean[mask] - ds_baseline_mean[mask]) / ds_baseline_mean[mask]
    )

    print(f"\n  {'rp [Mpc/h]':>14s}  {'Baseline':>14s}  {'Optimized':>14s}  {'Deviation':>10s}")
    print("  " + "-" * 58)
    for r, ds_b, ds_o, dev in zip(rp_centers, ds_baseline_mean, ds_opt_mean, per_bin_dev):
        print(f"    {r:12.4f}  {ds_b:14.6e}  {ds_o:14.6e}  {dev*100:8.2f}%")

    # Timing
    print(f"\n  Timing:")
    print(f"    Mean per realization: {fmt_time(timing_all.mean())}")
    print(f"    Total ({n_real} realizations): {fmt_time(timing_all.sum())}")

    # --- PASS/FAIL verdict ---
    max_dev = metrics["max_frac_dev"]
    passed = max_dev < THRESHOLD

    print(f"\n  {'=' * 40}")
    if passed:
        print(f"  PASS  (max deviation {max_dev*100:.2f}% < {THRESHOLD*100:.0f}% threshold)")
    else:
        print(f"  FAIL  (max deviation {max_dev*100:.2f}% >= {THRESHOLD*100:.0f}% threshold)")
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
        f.write(f"# Result: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"#\n")
        f.write(f"# Median deviation: {metrics['median_frac_dev']*100:.2f}%\n")
        f.write(f"# Max deviation:    {metrics['max_frac_dev']*100:.2f}%\n")
        f.write(f"# RMS deviation:    {metrics['rms_frac_dev']*100:.2f}%\n")
        f.write(f"#\n")
        f.write(f"# {'rp [Mpc/h]':>14s}  {'Baseline':>14s}  {'Optimized':>14s}  {'Deviation':>10s}\n")
        for r, ds_b, ds_o, dev in zip(rp_centers, ds_baseline_mean, ds_opt_mean, per_bin_dev):
            f.write(f"  {r:14.6f}  {ds_b:14.6e}  {ds_o:14.6e}  {dev*100:8.2f}%\n")
    print(f"\n  Saved results: {RESULTS_OUTPUT}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )

    # Top panel: DeltaSigma comparison
    ax1.loglog(rp_centers, ds_baseline_mean, "k-", lw=2,
               label=f"Baseline (mean of {n_baseline})")
    ax1.loglog(rp_centers, ds_opt_mean, "ro--", lw=1.5, ms=5,
               label=f"Optimized (mean of {n_real})")

    # Individual realizations as faint lines
    for i in range(n_real):
        ax1.loglog(rp_centers, ds_opt_all[i], color="red", alpha=0.15, lw=0.5)

    ax1.set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot/\mathrm{pc}^2$]")
    ax1.legend(loc="upper right")
    ax1.set_title("Numerical Regression Test: Optimized vs Baseline DeltaSigma")

    # Bottom panel: fractional difference
    frac_diff = np.where(
        np.abs(ds_baseline_mean) > 1e-10,
        (ds_opt_mean - ds_baseline_mean) / ds_baseline_mean,
        0.0,
    )
    ax2.semilogx(rp_centers, frac_diff * 100, "ko-", ms=4)
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
