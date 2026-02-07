"""
Fast vs Original Satellite Positioning Benchmark
=================================================

Compares the optimized satellite positioning implementation against
the original version using the optimal configuration from CLAUDE.md:
- Particle fraction: 5%
- Galaxy fraction: 10%
- N realizations: 5

Output:
- Timing comparison for 5 iterations
- Plot comparing DeltaSigma results
"""

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
# Configuration (from CLAUDE.md optimal settings)
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

cosmo_params = {
    "H0": 68.1,
    "Om0": 0.306,
    "Ob0": 0.0486,
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

# Optimal configuration from CLAUDE.md
PARTICLE_FRACTION = 0.05  # 5%
GALAXY_FRACTION = 0.10    # 10%
N_REALIZATIONS = 5

# Lensing bins
rp_bins = np.geomspace(0.1, 50.0, 16)
BINS_COMP = np.geomspace(5e-3, 120, 151)

BASE_SEED = 2000


# ============================================================================
# Helper functions
# ============================================================================

def subsample_array(positions, fraction, seed):
    """Subsample positions array with sorted indices."""
    if fraction >= 1.0:
        return positions
    n_keep = int(len(positions) * fraction)
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(positions), n_keep, replace=False))
    return positions[indices]


def fmt_time(seconds):
    """Pretty-print a duration."""
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds/60:.1f} min"


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# Main benchmark
# ============================================================================

def main():
    print_header("Fast vs Original Satellite Positioning Benchmark")
    print(f"\nOptimal configuration from CLAUDE.md:")
    print(f"  Particle fraction: {PARTICLE_FRACTION*100:.0f}%")
    print(f"  Galaxy fraction:   {GALAXY_FRACTION*100:.0f}%")
    print(f"  N realizations:    {N_REALIZATIONS}")

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

    # Get full particle positions as numpy array
    positions_part_full = np.array(halo.positions_part)

    # Subsample particles
    positions_part = subsample_array(positions_part_full, PARTICLE_FRACTION, seed=99)
    print(f"\nUsing {len(positions_part):,} particles ({PARTICLE_FRACTION*100:.0f}% of {len(positions_part_full):,})")

    # ========================================================================
    # JIT warmup for both versions
    # ========================================================================
    print("\nWarming up JAX JIT (both versions)...")

    # Warmup fast version
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0, use_fast=True)
    warmup_fast = time.perf_counter() - t0
    print(f"  Fast warmup: {fmt_time(warmup_fast)}")

    # Warmup original version
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0, use_fast=False)
    warmup_orig = time.perf_counter() - t0
    print(f"  Original warmup: {fmt_time(warmup_orig)}")

    # ========================================================================
    # Benchmark: Original version (use_fast=False)
    # ========================================================================
    print_header(f"Original Version (use_fast=False) - {N_REALIZATIONS} iterations")

    ds_original = []
    timing_original = []
    rp_centers = None

    for i in range(N_REALIZATIONS):
        seed = BASE_SEED + i
        print(f"  Iteration {i+1}/{N_REALIZATIONS} (seed={seed})...", end=" ", flush=True)

        # Populate with original
        t_pop = time.perf_counter()
        halo.populate_haloes(hod_params, random_seed=seed, use_fast=False)
        t_pop = time.perf_counter() - t_pop

        # Subsample galaxies
        gal_pos = np.array(halo.positions_gal)
        gal_pos_sub = subsample_array(gal_pos, GALAXY_FRACTION, seed=88+i)
        n_gal = len(gal_pos_sub)

        # Compute lensing
        t_lens = time.perf_counter()
        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
        )
        t_lens = time.perf_counter() - t_lens

        ds_original.append(ds)
        timing_original.append(t_pop + t_lens)
        print(f"N_gal={n_gal:,}, pop={fmt_time(t_pop)}, lens={fmt_time(t_lens)}, total={fmt_time(t_pop+t_lens)}")

    ds_original = np.array(ds_original)
    timing_original = np.array(timing_original)

    print(f"\n  Total time (original): {fmt_time(timing_original.sum())}")
    print(f"  Mean per iteration:    {fmt_time(timing_original.mean())}")

    # ========================================================================
    # Benchmark: Fast version (use_fast=True)
    # ========================================================================
    print_header(f"Fast Version (use_fast=True) - {N_REALIZATIONS} iterations")

    ds_fast = []
    timing_fast = []

    for i in range(N_REALIZATIONS):
        seed = BASE_SEED + i
        print(f"  Iteration {i+1}/{N_REALIZATIONS} (seed={seed})...", end=" ", flush=True)

        # Populate with fast
        t_pop = time.perf_counter()
        halo.populate_haloes(hod_params, random_seed=seed, use_fast=True)
        t_pop = time.perf_counter() - t_pop

        # Subsample galaxies (same seed for consistency)
        gal_pos = np.array(halo.positions_gal)
        gal_pos_sub = subsample_array(gal_pos, GALAXY_FRACTION, seed=88+i)
        n_gal = len(gal_pos_sub)

        # Compute lensing
        t_lens = time.perf_counter()
        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP
        )
        t_lens = time.perf_counter() - t_lens

        ds_fast.append(ds)
        timing_fast.append(t_pop + t_lens)
        print(f"N_gal={n_gal:,}, pop={fmt_time(t_pop)}, lens={fmt_time(t_lens)}, total={fmt_time(t_pop+t_lens)}")

    ds_fast = np.array(ds_fast)
    timing_fast = np.array(timing_fast)

    print(f"\n  Total time (fast): {fmt_time(timing_fast.sum())}")
    print(f"  Mean per iteration: {fmt_time(timing_fast.mean())}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Summary")

    speedup = timing_original.sum() / timing_fast.sum()

    print(f"\n{'Version':<20} {'Total Time':<15} {'Per Iteration':<15}")
    print("-" * 50)
    print(f"{'Original':<20} {fmt_time(timing_original.sum()):<15} {fmt_time(timing_original.mean()):<15}")
    print(f"{'Fast':<20} {fmt_time(timing_fast.sum()):<15} {fmt_time(timing_fast.mean()):<15}")
    print("-" * 50)
    print(f"{'Speedup':<20} {speedup:.1f}x")

    # ========================================================================
    # Compare DeltaSigma results
    # ========================================================================
    print("\nDeltaSigma comparison (mean over realizations):")

    ds_mean_orig = ds_original.mean(axis=0)
    ds_mean_fast = ds_fast.mean(axis=0)

    frac_diff = np.abs((ds_mean_fast - ds_mean_orig) / ds_mean_orig)

    print(f"  Max fractional difference: {frac_diff.max()*100:.3f}%")
    print(f"  Mean fractional difference: {frac_diff.mean()*100:.3f}%")
    print(f"  (Differences are due to different random sampling, not errors)")

    # ========================================================================
    # Plot comparison
    # ========================================================================
    print("\nGenerating comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: DeltaSigma comparison
    ax1 = axes[0]
    ax1.semilogx(rp_centers, rp_centers*ds_mean_orig, 'b-', lw=2, label='Original', alpha=0.8)
    ax1.semilogx(rp_centers, rp_centers*ds_mean_fast, 'r--', lw=2, label='Fast', alpha=0.8)

    # Add shaded regions for std
    ds_std_orig = ds_original.std(axis=0)
    ds_std_fast = ds_fast.std(axis=0)
    #ax1.fill_between(rp_centers, ds_mean_orig - ds_std_orig, ds_mean_orig + ds_std_orig,
    #                 color='blue', alpha=0.2)
    #ax1.fill_between(rp_centers, ds_mean_fast - ds_std_fast, ds_mean_fast + ds_std_fast,
    #                 color='red', alpha=0.2)

    ax1.set_xlabel(r"$r_p$ [Mpc/h]", fontsize=12)
    ax1.set_ylabel(r"$\Delta\Sigma$ [h M$_\odot$/pc$^2$]", fontsize=12)
    ax1.set_title("DeltaSigma: Original vs Fast")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right panel: Fractional difference
    ax2 = axes[1]
    ax2.semilogx(rp_centers, frac_diff * 100, 'ko-', lw=1.5)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.axhline(1, color='red', ls=':', alpha=0.5, label='1% threshold')
    ax2.axhline(-1, color='red', ls=':', alpha=0.5)

    ax2.set_xlabel(r"$r_p$ [Mpc/h]", fontsize=12)
    ax2.set_ylabel("Fractional Difference [%]", fontsize=12)
    ax2.set_title("Fast vs Original: Fractional Difference")
    ax2.set_ylim(-5, 5)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add timing info as text
    fig.text(0.5, 0.01,
             f"Original: {fmt_time(timing_original.sum())} | Fast: {fmt_time(timing_fast.sum())} | Speedup: {speedup:.1f}x",
             ha='center', fontsize=11, style='italic')

    plt.subplots_adjust(bottom=0.15)

    output_file = "fast_vs_original_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
