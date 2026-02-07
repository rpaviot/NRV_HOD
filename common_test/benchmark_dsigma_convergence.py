"""
DeltaSigma Performance Optimization Benchmark
==============================================

This script benchmarks the numerical DeltaSigma calculation to determine
optimal downsampling parameters that balance accuracy and computational cost.

Tests:
1. Baseline: Full resolution with multiple HOD realizations (cached)
2. Particle downsampling: Find maximum downsample preserving <5% deviation
3. Galaxy downsampling: Find maximum downsample preserving <5% deviation
4. Realization convergence: Find minimum N_real for stable averaged results

Output: benchmark_results.json with optimal settings and speedup factor

Data: Flamingo L1000N1800
  - Halos:     parquet_halo_catalogue_L1000N1800.parquet
  - Particles: particle_catalogue_L1000N1800_downsampled.parquet
"""

import os
import json
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
# Configuration (matches numerical_dsigma_example.py)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

Lbox = 681.0  # Mpc/h
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

# Lensing bins (same as example)
rp_bins = np.geomspace(0.1, 50.0, 16)

# Use the class-method default for bins_comp (HOD_catalogue.py:434)
BINS_COMP = np.geomspace(5e-3, 120, 151)

# ============================================================================
# Benchmark configuration
# ============================================================================

BASE_SEED = 1000
N_BASELINE = 10              # Number of realizations for baseline
N_TOTAL = 20                 # Total realizations for convergence test

# Skip 100% particles in Part 2 (already computed in baseline)
PARTICLE_FRACTIONS = [0.75, 0.50, 0.25, 0.10, 0.05]
GALAXY_FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.10]
N_REAL_SUBSETS = [3, 5, 10, 15, 20]

TOLERANCE = 0.01             # 5% tolerance for convergence

BASELINE_CACHE = "baseline_dsigma_cache.npz"
RESULTS_FILE = "benchmark_results.json"

# Fixed seeds for subsampling (isolates noise sources)
PARTICLE_SUBSAMPLE_SEED = 99
GALAXY_SUBSAMPLE_SEED = 88


# ============================================================================
# Helper functions
# ============================================================================

def subsample_array(positions, fraction, seed):
    """
    Subsample positions array with sorted indices for cache locality.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions, shape (N, 3)
    fraction : float
        Fraction of points to keep (0 < fraction <= 1)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Subsampled positions array
    """
    if fraction >= 1.0:
        return positions
    n_keep = int(len(positions) * fraction)
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(positions), n_keep, replace=False))
    return positions[indices]


def compute_deviation_metrics(ds_test, ds_ref):
    """
    Compute deviation metrics between test and reference DeltaSigma.

    Parameters
    ----------
    ds_test : np.ndarray
        Test DeltaSigma values
    ds_ref : np.ndarray
        Reference DeltaSigma values

    Returns
    -------
    dict
        Dictionary with median, max, and rms fractional deviations
    """
    # Avoid division by zero
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


def find_plateau_fraction(fractions, deviations, threshold=0.02):
    """
    Find optimal fraction based on plateau detection.

    Scans from largest to smallest fraction. Returns the smallest fraction
    where the change in deviation from the previous (larger) fraction is
    less than threshold (i.e., deviation has plateaued).

    Parameters
    ----------
    fractions : list
        List of fractions tested (e.g., [0.75, 0.50, 0.25, 0.10])
    deviations : dict
        Dictionary mapping fraction -> deviation value
    threshold : float
        Maximum allowed change in deviation to be considered a plateau

    Returns
    -------
    float
        Optimal fraction (smallest within plateau)
    """
    sorted_fracs = sorted(fractions, reverse=True)  # e.g., [0.75, 0.50, 0.25, 0.10]

    for i in range(1, len(sorted_fracs)):
        prev_frac = sorted_fracs[i - 1]
        curr_frac = sorted_fracs[i]

        delta = abs(deviations[curr_frac] - deviations[prev_frac])
        if delta > threshold:
            # Deviation jumped - previous fraction was the plateau
            return prev_frac

    # All within plateau - return smallest
    return sorted_fracs[-1]


def plot_downsampling_comparison(rp_centers, ds_by_fraction, ds_ref, title, filename):
    """
    Plot DeltaSigma for each downsampling level vs reference.

    Parameters
    ----------
    rp_centers : np.ndarray
        Projected radial bin centers
    ds_by_fraction : dict
        Dictionary mapping fraction -> DeltaSigma array
    ds_ref : np.ndarray
        Reference DeltaSigma (100% fraction)
    title : str
        Plot title
    filename : str
        Output filename for the plot
    """
    plt.figure(figsize=(10, 6))

    # Reference
    plt.loglog(rp_centers, ds_ref, "k-", lw=2, label="Reference (100%)")

    # Each fraction
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ds_by_fraction)))
    for (frac, ds), color in zip(sorted(ds_by_fraction.items(), reverse=True), colors):
        plt.loglog(rp_centers, ds, "o-", color=color, label=f"{float(frac)*100:.0f}%")

    plt.xlabel(r"$r_p$ [Mpc/h]")
    plt.ylabel(r"$\Delta\Sigma$ [M$_\odot$/pc$^2$]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved plot: {filename}")


def print_deviation_table(rp_centers, deviations_by_bin):
    """
    Print per-bin fractional deviation table.

    Parameters
    ----------
    rp_centers : np.ndarray
        Projected radial bin centers
    deviations_by_bin : dict
        Dictionary mapping fraction -> array of per-bin deviations
    """
    fractions = sorted(deviations_by_bin.keys(), reverse=True)

    # Header
    header = f"{'rp [Mpc/h]':>12}"
    for frac in fractions:
        header += f"  {float(frac)*100:>5.0f}%"
    print(header)
    print("  " + "-" * len(header))

    # Rows
    for i, rp in enumerate(rp_centers):
        row = f"{rp:>12.3f}"
        for frac in fractions:
            dev = deviations_by_bin[frac][i] * 100
            row += f"  {dev:>5.2f}%"
        print(row)


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# Main benchmark
# ============================================================================

def main():
    print_header("DeltaSigma Performance Optimization Benchmark")

    results = {
        "config": {
            "Lbox": Lbox,
            "zeff": zeff,
            "target_ngal": target_ngal,
            "rp_bins": rp_bins.tolist(),
            "n_baseline": N_BASELINE,
            "n_total": N_TOTAL,
            "tolerance": TOLERANCE,
            "particle_fractions_tested": PARTICLE_FRACTIONS,
            "galaxy_fractions_tested": GALAXY_FRACTIONS,
            "n_real_subsets_tested": N_REAL_SUBSETS,
        },
        "baseline": {},
        "particle_downsampling": {},
        "galaxy_downsampling": {},
        "realization_convergence": {},
        "optimal": {},
    }

    # ========================================================================
    # Load data
    # ========================================================================
    print("\nLoading catalogs...")
    t0 = time.perf_counter()
    df_halo = pd.read_parquet(HALO_PATH)
    df_part = pd.read_parquet(PARTICLE_PATH)
    load_time = time.perf_counter() - t0
    print(f"  Halos: {len(df_halo):,}")
    print(f"  Particles: {len(df_part):,}")
    print(f"  Load time: {fmt_time(load_time)}")

    # ========================================================================
    # Initialize HaloOccupation
    # ========================================================================
    print("\nInitializing HaloOccupation...")
    t0 = time.perf_counter()
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
    init_time = time.perf_counter() - t0
    print(f"  Init time: {fmt_time(init_time)}")

    # ========================================================================
    # Set HOD model and rescale parameters
    # ========================================================================
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

    # ========================================================================
    # JIT warmup
    # ========================================================================
    print("\nWarming up JAX JIT (throw-away population)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup time: {fmt_time(warmup_time)}")

    # Get full particle positions as numpy array
    positions_part_full = np.array(halo.positions_part)

    # ========================================================================
    # Part 1: Baseline (with caching)
    # ========================================================================
    print_header("Part 1: Baseline (100% particles, 100% galaxies)")

    if os.path.exists(BASELINE_CACHE):
        print(f"  Loading cached baseline from {BASELINE_CACHE}")
        data = np.load(BASELINE_CACHE)
        ds_all_baseline = data["ds_all"]
        rp_centers = data["rp"]
        timing_per_real = data["timing_per_real"]
        print(f"  Loaded {len(ds_all_baseline)} realizations")
    else:
        print(f"  Computing {N_BASELINE} realizations...")
        ds_all_baseline = []
        timing_per_real = []
        rp_centers = None

        for i in range(N_BASELINE):
            seed = BASE_SEED + i
            print(f"    Realization {i+1}/{N_BASELINE} (seed={seed})...", end=" ", flush=True)

            # Populate
            t_pop = time.perf_counter()
            halo.populate_haloes(hod_params, random_seed=seed)
            t_pop = time.perf_counter() - t_pop

            n_gal = len(halo.positions_gal)

            # Compute lensing
            t_lens = time.perf_counter()
            rp_centers, ds = compute_galaxy_lensing(
                halo.positions_gal, halo.positions_part,
                Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
            )
            t_lens = time.perf_counter() - t_lens

            ds_all_baseline.append(ds)
            timing_per_real.append(t_lens)

            print(f"N_gal={n_gal:,}, pop={fmt_time(t_pop)}, lens={fmt_time(t_lens)}")

        ds_all_baseline = np.array(ds_all_baseline)
        timing_per_real = np.array(timing_per_real)

        # Save cache
        np.savez(
            BASELINE_CACHE,
            ds_all=ds_all_baseline,
            rp=rp_centers,
            timing_per_real=timing_per_real,
        )
        print(f"  Saved cache to {BASELINE_CACHE}")

    # Compute baseline statistics
    ds_mean_baseline = ds_all_baseline.mean(axis=0)
    ds_std_baseline = ds_all_baseline.std(axis=0, ddof=1)

    results["baseline"] = {
        "n_realizations": int(len(ds_all_baseline)),
        "timing_per_real_mean": float(timing_per_real.mean()),
        "timing_per_real_std": float(timing_per_real.std()),
        "timing_total": float(timing_per_real.sum()),
        "rp_centers": rp_centers.tolist(),
        "ds_mean": ds_mean_baseline.tolist(),
        "ds_std": ds_std_baseline.tolist(),
    }

    print(f"\n  Baseline timing: {fmt_time(timing_per_real.mean())} +/- {fmt_time(timing_per_real.std())} /realization")
    print(f"  Total baseline time: {fmt_time(timing_per_real.sum())}")

    # ========================================================================
    # Part 2: Particle downsampling convergence
    # ========================================================================
    print_header("Part 2: Particle Downsampling Convergence")

    particle_results = {}
    ds_means_by_particle_frac = {}  # Store ds_mean for plotting
    deviations_by_bin_particle = {}  # Store per-bin deviations for table
    optimal_particle_frac = 1.0

    for part_frac in PARTICLE_FRACTIONS:
        print(f"\n  Testing particle fraction = {part_frac*100:.0f}%...")

        # Subsample particles (same seed for all realizations)
        positions_part_sub = subsample_array(
            positions_part_full, part_frac, PARTICLE_SUBSAMPLE_SEED
        )
        print(f"    N_particles: {len(positions_part_sub):,}")

        ds_all = []
        timings = []

        for i in range(N_BASELINE):
            seed = BASE_SEED + i

            # Populate with same seed as baseline
            halo.populate_haloes(hod_params, random_seed=seed)

            t0 = time.perf_counter()
            _, ds = compute_galaxy_lensing(
                halo.positions_gal, positions_part_sub,
                Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
            )
            timings.append(time.perf_counter() - t0)
            ds_all.append(ds)

        ds_all = np.array(ds_all)
        ds_mean = ds_all.mean(axis=0)

        # Store ds_mean for plotting
        ds_means_by_particle_frac[part_frac] = ds_mean

        # Compute per-bin fractional deviations
        mask = np.abs(ds_mean_baseline) > 1e-10
        per_bin_dev = np.zeros_like(ds_mean)
        per_bin_dev[mask] = np.abs(
            (ds_mean[mask] - ds_mean_baseline[mask]) / ds_mean_baseline[mask]
        )
        deviations_by_bin_particle[part_frac] = per_bin_dev

        # Compare to baseline
        metrics = compute_deviation_metrics(ds_mean, ds_mean_baseline)

        particle_results[str(part_frac)] = {
            "n_particles": len(positions_part_sub),
            "timing_mean": float(np.mean(timings)),
            "timing_std": float(np.std(timings)),
            **metrics,
        }

        print(f"    Timing: {fmt_time(np.mean(timings))} +/- {fmt_time(np.std(timings))}")
        print(f"    Median frac. deviation: {metrics['median_frac_dev']*100:.2f}%")

    # Plot downsampling comparison
    plot_downsampling_comparison(
        rp_centers,
        ds_means_by_particle_frac,
        ds_mean_baseline,
        "Particle Downsampling: DeltaSigma Comparison",
        "particle_downsampling_comparison.png",
    )

    # Print per-bin deviation table
    print("\n  Per-bin fractional deviations:")
    print_deviation_table(rp_centers, deviations_by_bin_particle)

    # Use plateau detection to find optimal fraction
    median_devs = {
        frac: particle_results[str(frac)]["median_frac_dev"]
        for frac in PARTICLE_FRACTIONS
    }
    optimal_particle_frac = find_plateau_fraction(
        PARTICLE_FRACTIONS, median_devs, threshold=TOLERANCE
    )

    results["particle_downsampling"] = {
        "results_by_fraction": particle_results,
        "optimal_fraction": optimal_particle_frac,
    }

    print(f"\n  Optimal particle fraction (plateau-based): {optimal_particle_frac*100:.0f}%")

    # ========================================================================
    # Part 3: Galaxy downsampling convergence
    # ========================================================================
    print_header("Part 3: Galaxy Downsampling Convergence")
    print(f"  (Using particle fraction = {optimal_particle_frac*100:.0f}%)")

    # Prepare optimal particle subsample
    positions_part_opt = subsample_array(
        positions_part_full, optimal_particle_frac, PARTICLE_SUBSAMPLE_SEED
    )

    galaxy_results = {}
    ds_means_by_galaxy_frac = {}  # Store ds_mean for plotting
    deviations_by_bin_galaxy = {}  # Store per-bin deviations for table
    optimal_galaxy_frac = 1.0

    # First, get reference with 100% galaxies at optimal particle fraction
    print(f"\n  Computing reference (100% galaxies, {optimal_particle_frac*100:.0f}% particles)...")
    ds_ref_gal = []

    for i in range(N_BASELINE):
        seed = BASE_SEED + i
        halo.populate_haloes(hod_params, random_seed=seed)

        _, ds = compute_galaxy_lensing(
            halo.positions_gal, positions_part_opt,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
        )
        ds_ref_gal.append(ds)

    ds_ref_gal = np.array(ds_ref_gal)
    ds_mean_ref_gal = ds_ref_gal.mean(axis=0)

    # Test galaxy fractions (skip 1.0 as it's the reference)
    for gal_frac in GALAXY_FRACTIONS:
        if gal_frac >= 1.0:
            continue

        print(f"\n  Testing galaxy fraction = {gal_frac*100:.0f}%...")

        ds_all = []
        timings = []

        for i in range(N_BASELINE):
            seed = BASE_SEED + i
            halo.populate_haloes(hod_params, random_seed=seed)

            # Subsample galaxies (different seed offset per realization for variety)
            gal_sub = subsample_array(
                np.array(halo.positions_gal), gal_frac, GALAXY_SUBSAMPLE_SEED + i
            )

            if i == 0:
                print(f"    N_galaxies (first real): {len(gal_sub):,}")

            t0 = time.perf_counter()
            _, ds = compute_galaxy_lensing(
                gal_sub, positions_part_opt,
                Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
            )
            timings.append(time.perf_counter() - t0)
            ds_all.append(ds)

        ds_all = np.array(ds_all)
        ds_mean = ds_all.mean(axis=0)

        # Store ds_mean for plotting
        ds_means_by_galaxy_frac[gal_frac] = ds_mean

        # Compute per-bin fractional deviations
        mask = np.abs(ds_mean_ref_gal) > 1e-10
        per_bin_dev = np.zeros_like(ds_mean)
        per_bin_dev[mask] = np.abs(
            (ds_mean[mask] - ds_mean_ref_gal[mask]) / ds_mean_ref_gal[mask]
        )
        deviations_by_bin_galaxy[gal_frac] = per_bin_dev

        # Compare to reference (100% galaxies, optimal particles)
        metrics = compute_deviation_metrics(ds_mean, ds_mean_ref_gal)

        galaxy_results[str(gal_frac)] = {
            "timing_mean": float(np.mean(timings)),
            "timing_std": float(np.std(timings)),
            **metrics,
        }

        print(f"    Timing: {fmt_time(np.mean(timings))} +/- {fmt_time(np.std(timings))}")
        print(f"    Median frac. deviation: {metrics['median_frac_dev']*100:.2f}%")

    # Plot downsampling comparison
    plot_downsampling_comparison(
        rp_centers,
        ds_means_by_galaxy_frac,
        ds_mean_ref_gal,
        "Galaxy Downsampling: DeltaSigma Comparison",
        "galaxy_downsampling_comparison.png",
    )

    # Print per-bin deviation table
    print("\n  Per-bin fractional deviations:")
    print_deviation_table(rp_centers, deviations_by_bin_galaxy)

    # Use plateau detection to find optimal fraction
    # Only consider fractions < 1.0 (the ones we tested)
    tested_fracs = [f for f in GALAXY_FRACTIONS if f < 1.0]
    median_devs = {
        frac: galaxy_results[str(frac)]["median_frac_dev"] for frac in tested_fracs
    }
    optimal_galaxy_frac = find_plateau_fraction(
        tested_fracs, median_devs, threshold=TOLERANCE
    )

    results["galaxy_downsampling"] = {
        "particle_fraction_used": optimal_particle_frac,
        "results_by_fraction": galaxy_results,
        "optimal_fraction": optimal_galaxy_frac,
    }

    print(f"\n  Optimal galaxy fraction (plateau-based): {optimal_galaxy_frac*100:.0f}%")

    # ========================================================================
    # Part 4: Number of realizations convergence
    # ========================================================================
    print_header("Part 4: Realization Convergence")
    print(f"  (Using particle={optimal_particle_frac*100:.0f}%, galaxy={optimal_galaxy_frac*100:.0f}%)")

    # Compute N_TOTAL realizations with optimal settings
    print(f"\n  Computing {N_TOTAL} realizations with optimal settings...")

    ds_all_opt = []
    timings_opt = []

    for i in range(N_TOTAL):
        seed = BASE_SEED + i

        halo.populate_haloes(hod_params, random_seed=seed)

        # Subsample galaxies
        gal_sub = subsample_array(
            np.array(halo.positions_gal), optimal_galaxy_frac, GALAXY_SUBSAMPLE_SEED + i
        )

        t0 = time.perf_counter()
        _, ds = compute_galaxy_lensing(
            gal_sub, positions_part_opt,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
        )
        timings_opt.append(time.perf_counter() - t0)
        ds_all_opt.append(ds)

        if (i + 1) % 5 == 0:
            print(f"    Completed {i+1}/{N_TOTAL} realizations")

    ds_all_opt = np.array(ds_all_opt)
    timings_opt = np.array(timings_opt)

    # Analyze convergence for different N_real
    realization_results = {}
    optimal_n_real = N_TOTAL

    print("\n  Analyzing convergence by number of realizations:")
    print(f"\n  {'N_real':>8s}  {'Med RMS%':>10s}  {'Med SE%':>10s}  {'Max SE%':>10s}")
    print("  " + "-" * 48)

    for n_real in N_REAL_SUBSETS:
        if n_real > N_TOTAL:
            continue

        ds_subset = ds_all_opt[:n_real]
        ds_mean_subset = ds_subset.mean(axis=0)
        ds_std_subset = ds_subset.std(axis=0, ddof=1)

        # Standard error of mean
        se_mean = ds_std_subset / np.sqrt(n_real)

        # Fractional metrics relative to mean
        mask = np.abs(ds_mean_subset) > 1e-10
        frac_rms = ds_std_subset[mask] / np.abs(ds_mean_subset[mask])
        frac_se = se_mean[mask] / np.abs(ds_mean_subset[mask])

        median_frac_rms = float(np.median(frac_rms))
        median_frac_se = float(np.median(frac_se))
        max_frac_se = float(np.max(frac_se))

        realization_results[str(n_real)] = {
            "median_frac_rms": median_frac_rms,
            "median_frac_se": median_frac_se,
            "max_frac_se": max_frac_se,
        }

        flag = ""
        if median_frac_se < TOLERANCE and optimal_n_real == N_TOTAL:
            optimal_n_real = n_real
            flag = " <--"

        print(f"  {n_real:>8d}  {median_frac_rms*100:>9.2f}%  {median_frac_se*100:>9.2f}%  {max_frac_se*100:>9.2f}%{flag}")

    # Find smallest N_real meeting tolerance (rescan to be sure)
    for n_real in N_REAL_SUBSETS:
        if n_real > N_TOTAL:
            continue
        if realization_results[str(n_real)]["median_frac_se"] < TOLERANCE:
            optimal_n_real = n_real
            break

    results["realization_convergence"] = {
        "particle_fraction_used": optimal_particle_frac,
        "galaxy_fraction_used": optimal_galaxy_frac,
        "results_by_n_real": realization_results,
        "optimal_n_real": optimal_n_real,
        "timing_per_real_mean": float(np.mean(timings_opt)),
        "timing_per_real_std": float(np.std(timings_opt)),
    }

    print(f"\n  Optimal N_realizations: {optimal_n_real}")

    # ========================================================================
    # Part 5: Summary
    # ========================================================================
    print_header("Part 5: Summary")

    # Calculate speedup
    baseline_time_per_real = results["baseline"]["timing_per_real_mean"]
    optimized_time_per_real = float(np.mean(timings_opt))

    baseline_total_time = baseline_time_per_real * N_BASELINE
    optimized_total_time = optimized_time_per_real * optimal_n_real

    speedup_per_real = baseline_time_per_real / optimized_time_per_real if optimized_time_per_real > 0 else float("inf")
    speedup_total = baseline_total_time / optimized_total_time if optimized_total_time > 0 else float("inf")

    results["optimal"] = {
        "particle_fraction": optimal_particle_frac,
        "galaxy_fraction": optimal_galaxy_frac,
        "n_realizations": optimal_n_real,
        "speedup_per_realization": float(speedup_per_real),
        "speedup_total": float(speedup_total),
        "baseline_time_per_real": float(baseline_time_per_real),
        "optimized_time_per_real": float(optimized_time_per_real),
        "baseline_total_time": float(baseline_total_time),
        "optimized_total_time": float(optimized_total_time),
    }

    print(f"\n  Optimal settings:")
    print(f"    Particle fraction:  {optimal_particle_frac*100:.0f}%")
    print(f"    Galaxy fraction:    {optimal_galaxy_frac*100:.0f}%")
    print(f"    N_realizations:     {optimal_n_real}")

    print(f"\n  Timing comparison:")
    print(f"    Baseline:   {fmt_time(baseline_time_per_real)}/real x {N_BASELINE} = {fmt_time(baseline_total_time)}")
    print(f"    Optimized:  {fmt_time(optimized_time_per_real)}/real x {optimal_n_real} = {fmt_time(optimized_total_time)}")

    print(f"\n  Speedup:")
    print(f"    Per realization: {speedup_per_real:.1f}x")
    print(f"    Total:           {speedup_total:.1f}x")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
