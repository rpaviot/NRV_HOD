"""
Optimized vs Baseline DeltaSigma Benchmark
============================================

Runs compute_galaxy_lensing_optimized() using the precomputed halo-center
cache across 5 HOD realizations and compares against baseline_dsigma_cache.npz.

Prerequisites:
  1. baseline_dsigma_cache.npz  (from numerical_dsigma_example.py)
  2. halo_center_lensing_cache.h5  (from precompute_halo_center_cache.py)

Data: Flamingo L1000N1800
  - Halos:     parquet_halo_catalogue_L1000N1800.parquet
  - Particles: particle_catalogue_L1000N1800_downsampled.parquet
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import HaloCenterLensingCache
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration (matches numerical_dsigma_example.py)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

N_REAL = 10
BASE_SEED = 1000
PARTICLE_FRACTION = 0.05
PARTICLE_SUBSAMPLE_SEED = 99

# Load galaxy fraction from benchmark results
with open("benchmark_results.json") as _f:
    #GALAXY_FRACTION = json.load(_f)["optimal"]["galaxy_fraction"]
    GALAXY_FRACTION = 0.2


CACHE_INPUT = "halo_center_lensing_cache.h5"
BASELINE_CACHE = "baseline_dsigma_cache.npz"
PLOT_OUTPUT = "optimized_vs_baseline_comparison.png"

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
    "Mmin": 12.7,
    "sig_M": 0.3,
    "M1": 13.0,
    "gamma":5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}
target_ngal = 2e-4  # (Mpc/h)^-3

# Lensing bins (same as example)
rp_bins = np.geomspace(0.1, 50.0, 16)


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
    print("=" * 60)
    print("Optimized vs Baseline DeltaSigma Benchmark")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading halo catalog...")
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo)} halos loaded")

    print("Loading particle catalog...")
    df_part = pd.read_parquet(PARTICLE_PATH)
    print(f"  {len(df_part)} particles loaded")

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

    # --- Downsample particles (same as precomputation) ---
    positions_part_full = np.array(halo.positions_part)
    positions_part_sub = subsample_array(
        positions_part_full, PARTICLE_FRACTION, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticle downsampling: {len(positions_part_full):,} -> {len(positions_part_sub):,} ({PARTICLE_FRACTION*100:.0f}%)")
    print(f"Galaxy downsampling:   {GALAXY_FRACTION*100:.0f}%")

    # Overwrite positions_part so compute_galaxy_lensing_optimized uses downsampled particles
    halo.positions_part = positions_part_sub

    # --- Load precomputed cache ---
    print(f"\nLoading precomputed cache from {CACHE_INPUT}...")
    cache = HaloCenterLensingCache.load(CACHE_INPUT)

    # --- Load baseline ---
    print(f"Loading baseline from {BASELINE_CACHE}...")
    baseline = np.load(BASELINE_CACHE)
    ds_baseline_all = baseline["ds_all"]
    rp_baseline = baseline["rp"]
    timing_baseline = baseline["timing_per_real"]

    n_baseline_available = len(ds_baseline_all)
    n_compare = min(N_REAL, n_baseline_available)
    ds_baseline_mean = ds_baseline_all[:n_compare].mean(axis=0)

    print(f"  Baseline: {n_baseline_available} realizations available, using first {n_compare}")
    print(f"  Baseline timing: {timing_baseline.mean():.1f}s/realization")

    # --- JIT warmup ---
    print("\nWarming up JAX JIT (throw-away population)...")
    halo.populate_haloes(hod_params, random_seed=0)

    # --- Run optimized realizations ---
    print(f"\nRunning {N_REAL} optimized realizations...")
    ds_opt_all = []
    timing_opt = []
    rp_centers = None

    for i in range(N_REAL):
        seed = BASE_SEED + i
        print(f"  Realization {i+1}/{N_REAL} (seed={seed})...", end=" ", flush=True)

        halo.populate_haloes(hod_params, random_seed=seed)
        n_gal = len(halo.positions_gal)
        f_sat = halo.satellite_fraction

        t0 = time.perf_counter()
        rp_centers, ds = halo.compute_galaxy_lensing_optimized(
            rp_bins, cache,
            galaxy_subsample_fraction=GALAXY_FRACTION,
            galaxy_subsample_seed=77 + i,
        )
        elapsed = time.perf_counter() - t0

        ds_opt_all.append(ds)
        timing_opt.append(elapsed)

        print(f"N_gal={n_gal:,}, f_sat={f_sat:.3f}, time={fmt_time(elapsed)}")

    ds_opt_all = np.array(ds_opt_all)
    timing_opt = np.array(timing_opt)
    ds_opt_mean = ds_opt_all.mean(axis=0)

    # --- Comparison metrics ---
    print("\n" + "=" * 60)
    print("Comparison: Optimized vs Baseline")
    print("=" * 60)

    metrics = compute_deviation_metrics(ds_opt_mean, ds_baseline_mean)

    print(f"\n  Deviation (mean of {n_compare} realizations):")
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

    # Timing comparison
    baseline_mean_time = float(timing_baseline.mean())
    opt_mean_time = float(timing_opt.mean())
    speedup = baseline_mean_time / opt_mean_time if opt_mean_time > 0 else float("inf")

    print(f"\n  Timing:")
    print(f"    Baseline:    {fmt_time(baseline_mean_time)}/realization")
    print(f"    Optimized:   {fmt_time(opt_mean_time)}/realization")
    print(f"    Speedup:     {speedup:.1f}x")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    # Top panel: DeltaSigma comparison
    ax1.semilogx(rp_centers, rp_centers*ds_baseline_mean, "k-", lw=2, label=f"Baseline (mean of {n_compare})")
    ax1.semilogx(rp_centers, rp_centers*ds_opt_mean, "ro--", lw=1.5, ms=5, label=f"Optimized (mean of {N_REAL})")

    # Show individual realizations as faint lines
    for i in range(N_REAL):
        ax1.loglog(rp_centers, rp_centers*ds_opt_all[i], color="red", alpha=0.15, lw=0.5)

    ax1.set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot/\mathrm{pc}^2$]")
    ax1.legend(loc="upper right")
    ax1.set_title("Optimized (Halo-Center Cache) vs Baseline DeltaSigma")

    # Bottom panel: fractional difference
    frac_diff = np.where(
        np.abs(ds_baseline_mean) > 1e-10,
        (ds_opt_mean - ds_baseline_mean) / ds_baseline_mean,
        0.0,
    )
    ax2.semilogx(rp_centers, frac_diff * 100, "ko-", ms=4)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.axhspan(-5, 5, color="green", alpha=0.1, label=r"$\pm 5\%$")
    ax2.set_xlabel(r"$r_p$ [Mpc/h]")
    ax2.set_ylabel("Fractional diff [%]")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"\n  Saved comparison plot: {PLOT_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
