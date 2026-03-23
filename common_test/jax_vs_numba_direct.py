"""
JAX vs Numba Direct Diagnostic Script
=======================================

Runs both the JAX and Numba population backends side-by-side with the same
seeds and HOD parameters, then computes ΔΣ for each via pycorr.  The goal is
to isolate the source of the ~1.5-2% small-scale bias in the Numba backend.

Hypothesis
----------
The JAX NFW grid (NFW_jax.py line 55) is float32 when jax_enable_x64 is off
(the default in the numerical module), while the Numba grid (NFW.py) is
float64.  This produces systematically different satellite radii and hence a
scale-dependent bias in ΔΣ.

Two test cases
--------------
full_params : normal As — satellites present
As_zero     : As = 0 — centrals only (both backends must agree to ~noise)

Expected result
---------------
  As_zero  : max|(Numba-JAX)/JAX| < 0.5%   (confirms centrals agree)
  full_params : small-scale deviation ~-1.5% (confirms satellite NFW origin)

Pass threshold : 1% (the user's target inter-backend agreement).

Data: Flamingo L1000N1800
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.HOD.population_engine_numba import populate_haloes_full_numba
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
    compute_galaxy_lensing,
)
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration (same as other common_test scripts)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

BENCHMARK_JSON = "benchmark_results.json"
PLOT_OUTPUT = "jax_vs_numba_direct.png"
RESULTS_OUTPUT = "jax_vs_numba_direct_results.txt"

Lbox = 681.0
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
    "gamma": 5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}
target_ngal = 2e-4  # (Mpc/h)^-3

rp_bins = np.geomspace(0.1, 50.0, 16)
BINS_COMP = np.geomspace(5e-3, 120, 151)

N_REAL = 10               # enough to resolve ~0.5% systematics
BASE_SEED = 3000          # different from other cross-checks to avoid seed collision
PARTICLE_SUBSAMPLE_SEED = 99
THRESHOLD = 0.01          # 1% — target inter-backend agreement


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


def signed_deviation(ds_test, ds_ref):
    """Return signed fractional deviation (test - ref) / ref per bin."""
    mask = np.abs(ds_ref) > 1e-10
    dev = np.zeros_like(ds_test)
    dev[mask] = (ds_test[mask] - ds_ref[mask]) / ds_ref[mask]
    return dev


def compute_deviation_metrics(ds_test, ds_ref):
    """Compute fractional deviation metrics (absolute) for PASS/FAIL."""
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
    print("JAX vs Numba Direct Diagnostic")
    print("=" * 70)
    print(f"\nN realizations: {N_REAL}  |  Base seed: {BASE_SEED}")
    print(f"Threshold: {THRESHOLD*100:.0f}%  |  Test cases: full_params, As_zero")

    # --- Load benchmark settings for downsampling fractions ---
    if not os.path.exists(BENCHMARK_JSON):
        raise FileNotFoundError(
            f"Benchmark results not found: {BENCHMARK_JSON}\n"
            "Run the convergence benchmark first to generate it."
        )
    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    optimal = bench["optimal"]
    particle_fraction = 0.1 #optimal["particle_fraction"]
    galaxy_fraction = 0.2 # optimal["galaxy_fraction"]

    print(f"\nDownsampling settings from {BENCHMARK_JSON}:")
    print(f"  Particle fraction: {particle_fraction*100:.0f}%")
    print(f"  Galaxy fraction:   {galaxy_fraction*100:.0f}%")

    # --- Load catalogs ---
    print("\nLoading halo catalog...")
    t0 = time.perf_counter()
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo):,} halos loaded")

    print("Loading particle catalog...")
    df_part = pd.read_parquet(PARTICLE_PATH)
    load_time = time.perf_counter() - t0
    print(f"  {len(df_part):,} particles loaded  ({fmt_time(load_time)})")

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

    # --- Extract halo arrays as numpy (once) ---
    positions_h = np.array(halo.positions)
    velocities_h = np.array(halo.velocities)
    mass_h = np.array(halo.mass)
    radius_h = np.array(halo.radius)
    concentration_h = np.array(halo.concentration)
    vrms_h = np.array(halo.vrms)
    logM_h = np.array(halo.logM)

    # --- Fixed particle subsample (same for all runs) ---
    positions_part_full = np.array(halo.positions_part)
    positions_part_sub = subsample_array(
        positions_part_full, particle_fraction, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticles: {len(positions_part_full):,} -> "
          f"{len(positions_part_sub):,} ({particle_fraction*100:.0f}%)")

    # --- Warmup (JIT compilation) ---
    print("\nWarming up JAX JIT ...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0)
    print(f"  JAX warmup: {fmt_time(time.perf_counter() - t0)}")

    print("Warming up Numba JIT ...")
    t0 = time.perf_counter()
    populate_haloes_full_numba(
        positions_h, velocities_h, mass_h, radius_h,
        concentration_h, vrms_h, logM_h,
        halo.HOD, hod_params,
        triaxial_NFW=False, apply_rsd=False,
        rsd_factor=halo.rsd_factor, rsd_axis_index=halo.rsd_axis_index,
        Lbox=Lbox, random_seed=0,
    )
    print(f"  Numba warmup: {fmt_time(time.perf_counter() - t0)}")

    # --- Define test cases ---
    cases = {
        "full_params": hod_params,
        "As_zero": {**hod_params, "As": 0.0},
    }

    # -----------------------------------------------------------------------
    # Run realizations for each case
    # -----------------------------------------------------------------------
    case_results = {}

    for case_name, case_params in cases.items():
        print(f"\n{'=' * 70}")
        print(f"Case: {case_name}")
        print(f"{'=' * 70}")
        print(f"  {'Seed':>6s}  {'N_gal JAX':>10s}  {'f_sat JAX':>9s}  "
              f"{'N_gal Numba':>12s}  {'f_sat Numba':>11s}  "
              f"{'JAX tot':>8s}  {'Numba tot':>9s}")
        print("  " + "-" * 78)

        ds_jax_all = []
        ds_numba_all = []

        for i in range(N_REAL):
            seed = BASE_SEED + i
            gal_fraction_seed = 88 + i  # same for both backends in this realization

            # ---- JAX population ----
            t_jax = time.perf_counter()
            halo.populate_haloes(case_params, random_seed=seed)
            n_gal_jax = len(halo.positions_gal)
            f_sat_jax = halo.satellite_fraction

            gal_jax = subsample_array(
                np.array(halo.positions_gal), galaxy_fraction, gal_fraction_seed
            )
            rp_centers, ds_jax = compute_galaxy_lensing(
                gal_jax, positions_part_sub,
                Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
            )
            t_jax = time.perf_counter() - t_jax
            ds_jax_all.append(ds_jax)

            # ---- Numba population ----
            t_numba = time.perf_counter()
            pos_gal_n, _, f_sat_numba, _ = populate_haloes_full_numba(
                positions_h, velocities_h, mass_h, radius_h,
                concentration_h, vrms_h, logM_h,
                halo.HOD, case_params,
                triaxial_NFW=False, apply_rsd=False,
                rsd_factor=halo.rsd_factor, rsd_axis_index=halo.rsd_axis_index,
                Lbox=Lbox, random_seed=seed,
            )
            n_gal_numba = len(pos_gal_n)

            gal_numba = subsample_array(pos_gal_n, galaxy_fraction, gal_fraction_seed)
            _, ds_numba = compute_galaxy_lensing(
                gal_numba, positions_part_sub,
                Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
            )
            t_numba = time.perf_counter() - t_numba
            ds_numba_all.append(ds_numba)

            print(f"  {seed:>6d}  {n_gal_jax:>10,}  {f_sat_jax:>9.4f}  "
                  f"{n_gal_numba:>12,}  {f_sat_numba:>11.4f}  "
                  f"{fmt_time(t_jax):>8s}  {fmt_time(t_numba):>9s}")

        ds_jax_all = np.array(ds_jax_all)
        ds_numba_all = np.array(ds_numba_all)

        ds_jax_mean = ds_jax_all.mean(axis=0)
        ds_numba_mean = ds_numba_all.mean(axis=0)

        case_results[case_name] = {
            "ds_jax_all": ds_jax_all,
            "ds_numba_all": ds_numba_all,
            "ds_jax_mean": ds_jax_mean,
            "ds_numba_mean": ds_numba_mean,
            "rp_centers": rp_centers,
        }

    # -----------------------------------------------------------------------
    # Report deviation tables
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Deviation Tables: (Numba - JAX) / JAX  [signed]")
    print(f"{'=' * 70}")

    results_lines = [
        "# JAX vs Numba Direct Diagnostic Results",
        f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# N realizations: {N_REAL}  Base seed: {BASE_SEED}",
        f"# Particle fraction: {particle_fraction*100:.0f}%  "
        f"Galaxy fraction: {galaxy_fraction*100:.0f}%",
        f"# Threshold: {THRESHOLD*100:.0f}%",
        "#",
    ]

    verdicts = {}
    for case_name, res in case_results.items():
        rp_c = res["rp_centers"]
        ds_j = res["ds_jax_mean"]
        ds_n = res["ds_numba_mean"]
        dev_signed = signed_deviation(ds_n, ds_j)
        metrics = compute_deviation_metrics(ds_n, ds_j)
        max_dev = metrics["max_frac_dev"]
        passed = max_dev < THRESHOLD
        verdicts[case_name] = passed

        print(f"\n  Case: {case_name}")
        print(f"  {'rp [Mpc/h]':>12s}  {'JAX mean':>14s}  "
              f"{'Numba mean':>14s}  {'(N-J)/J':>10s}")
        print("  " + "-" * 58)
        for r, dsj, dsn, dev in zip(rp_c, ds_j, ds_n, dev_signed):
            print(f"    {r:10.4f}  {dsj:14.6e}  {dsn:14.6e}  {dev*100:+9.3f}%")

        print(f"\n  Metrics: median={metrics['median_frac_dev']*100:.3f}%  "
              f"max={metrics['max_frac_dev']*100:.3f}%  "
              f"rms={metrics['rms_frac_dev']*100:.3f}%")
        verdict_str = "PASS" if passed else "FAIL"
        print(f"  {verdict_str}  (max {max_dev*100:.3f}% vs {THRESHOLD*100:.0f}% threshold)")

        results_lines += [
            f"#",
            f"# Case: {case_name}",
            f"# Result: {verdict_str}  max_dev={max_dev*100:.3f}%",
            f"# median_dev={metrics['median_frac_dev']*100:.3f}%  "
            f"rms_dev={metrics['rms_frac_dev']*100:.3f}%",
            f"# {'rp [Mpc/h]':>12s}  {'JAX_mean':>14s}  "
            f"{'Numba_mean':>14s}  {'(N-J)/J':>10s}",
        ]
        for r, dsj, dsn, dev in zip(rp_c, ds_j, ds_n, dev_signed):
            results_lines.append(
                f"  {r:12.6f}  {dsj:14.6e}  {dsn:14.6e}  {dev*100:+9.3f}%"
            )

    # Overall verdict
    all_passed = all(verdicts.values())
    print(f"\n{'=' * 70}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    for case_name, passed in verdicts.items():
        print(f"  {case_name}: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")

    # -----------------------------------------------------------------------
    # Save results text
    # -----------------------------------------------------------------------
    with open(RESULTS_OUTPUT, "w") as f:
        f.write("\n".join(results_lines) + "\n")
    print(f"\nSaved results: {RESULTS_OUTPUT}")

    # -----------------------------------------------------------------------
    # 4-panel figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_tl, ax_tr), (ax_bl, ax_br)) = axes

    panel_map = {
        "full_params": (ax_tl, ax_bl),
        "As_zero": (ax_tr, ax_br),
    }
    titles = {
        "full_params": "Full params (As > 0)",
        "As_zero": "As = 0 (centrals only)",
    }

    for case_name, (ax_top, ax_bot) in panel_map.items():
        res = case_results[case_name]
        rp_c = res["rp_centers"]
        ds_j_mean = res["ds_jax_mean"]
        ds_n_mean = res["ds_numba_mean"]
        ds_j_all = res["ds_jax_all"]
        ds_n_all = res["ds_numba_all"]

        # Top: ΔΣ comparison
        for i in range(N_REAL):
            ax_top.semilogx(rp_c, rp_c * ds_j_all[i], color="black", alpha=0.12, lw=0.5)
            ax_top.semilogx(rp_c, rp_c * ds_n_all[i], color="royalblue", alpha=0.12, lw=0.5)
        ax_top.semilogx(rp_c, rp_c * ds_j_mean, "k-", lw=2.0, label=f"JAX (mean {N_REAL})")
        ax_top.semilogx(rp_c, rp_c * ds_n_mean, "b--", lw=2.0,
                        label=f"Numba (mean {N_REAL})")
        ax_top.set_ylabel(r"$r_p \Delta\Sigma$ [$h\,M_\odot/\mathrm{pc}^2 \cdot h^{-1}\mathrm{Mpc}$]")
        ax_top.set_title(titles[case_name])
        ax_top.legend(loc="upper right", fontsize=9)

        # Bottom: signed fractional deviation
        dev = signed_deviation(ds_n_mean, ds_j_mean)
        ax_bot.axhline(0, color="gray", ls="--", lw=0.8)
        ax_bot.axhspan(-THRESHOLD * 100, THRESHOLD * 100, color="green", alpha=0.1,
                       label=rf"$\pm {THRESHOLD*100:.0f}\%$")
        ax_bot.semilogx(rp_c, dev * 100, "bo-", ms=4, lw=1.5, label="(Numba-JAX)/JAX")
        ax_bot.set_xlabel(r"$r_p$ [Mpc/h]")
        ax_bot.set_ylabel("Fractional diff [%]")
        ax_bot.legend(loc="upper right", fontsize=9)

        # Symmetric y-axis around 0 (minimum ±2%)
        yabs = max(np.abs(dev * 100).max() * 1.3, 2.0)
        ax_bot.set_ylim(-yabs, yabs)

    fig.suptitle(
        f"JAX vs Numba Direct Diagnostic  —  {N_REAL} realizations, seed base={BASE_SEED}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"Saved plot:    {PLOT_OUTPUT}")

    print("\nDone.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
