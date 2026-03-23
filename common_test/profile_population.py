"""
Population Pipeline Profiler
=============================

Decomposes `populate_haloes_full` into individually-timed sub-steps to identify
bottlenecks. Uses `time.perf_counter()` + `jax.block_until_ready()` on every
output to measure actual compute time (not just async dispatch time).

Key design decisions:
  - Same seed across all 5 timed iterations to avoid JIT recompilation
    (N_s_tot is a static_argnum in NFW positioning functions)
  - Two warmup calls: seed=0 (general JIT) + seed=RANDOM_SEED (exact N_s_tot)
  - Reports array sizes for cross-machine comparison (laptop vs cluster)

Usage:
    python common_test/profile_population.py
"""

import time
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jrandom

jax.config.update("jax_enable_x64", True)

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.HOD.population_engine import (
    populate_centrals, populate_satellites, combine_galaxy_populations,
    apply_rsd_to_galaxies, filter_halo_data,
)
from HOD_NRV.HOD_numerical.satellites.NFW_jax import (
    position_satellites, dispersion_velocities_satellites,
)
from HOD_NRV.utilsf.utils_functions import random_uniform_jax, random_poisson_jax
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal


# ============================================================================
# Configuration
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

Lbox = 681.0  # Mpc/h
zeff = 1.0
mass_definition = "MassDef200m"

column_mapping = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}

cosmo_params = {
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

RANDOM_SEED = 42
N_ITER = 5


# ============================================================================
# Helpers
# ============================================================================

def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    return f"{seconds:.3f} s"


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def timed_block_until_ready(*arrays):
    """Call block_until_ready on all JAX arrays and return elapsed time."""
    for a in arrays:
        if hasattr(a, 'block_until_ready'):
            a.block_until_ready()


# ============================================================================
# Sub-step profiling
# ============================================================================

def profile_population_steps(halo, hod_params, seed):
    """
    Replicate the logic of populate_haloes_full but time each sub-step.
    Returns dict of {step_name: elapsed_seconds}.
    """
    timings = {}

    # --- Step 1: Key setup ---
    t0 = time.perf_counter()
    key = jrandom.key(seed)
    key_c, key_s, key_s_pos, key_s_vel = jrandom.split(key, num=4)
    key_c.block_until_ready()
    timings["1. key setup"] = time.perf_counter() - t0

    # --- Step 2: compute_central_occupation ---
    t0 = time.perf_counter()
    probC = halo.HOD.compute_central_occupation(halo.logM, hod_params)
    probC.block_until_ready()
    timings["2. compute_central_occ"] = time.perf_counter() - t0

    # --- Step 3: populate_centrals (uniform + bool indexing) ---
    t0 = time.perf_counter()
    cent_positions, cent_velocities, is_cent = populate_centrals(
        key_c, halo.positions, halo.velocities, probC
    )
    cent_positions.block_until_ready()
    timings["3. populate_centrals"] = time.perf_counter() - t0

    N_centrals = len(cent_positions)

    # --- Step 4: jnp.where(is_cent) ---
    t0 = time.perf_counter()
    cent_halo_indices = jnp.where(is_cent)[0]
    cent_halo_indices.block_until_ready()
    timings["4. jnp.where(is_cent)"] = time.perf_counter() - t0

    # --- Step 5: compute_satellite_occupation ---
    t0 = time.perf_counter()
    probS = halo.HOD.compute_satellite_occupation(halo.logM, hod_params)
    probS.block_until_ready()
    timings["5. compute_satellite_occ"] = time.perf_counter() - t0

    # --- Step 6a: Pre-filter probS > 0 ---
    t0 = time.perf_counter()
    pre_cond = probS > 0
    probS_filtered = probS[pre_cond]
    probS_filtered.block_until_ready()
    timings["6a. pre-filter probS>0"] = time.perf_counter() - t0

    N_filtered = len(probS_filtered)

    # --- Step 6b: random_poisson_jax ---
    t0 = time.perf_counter()
    N_s = random_poisson_jax(key_s, probS_filtered)
    N_s.block_until_ready()
    timings["6b. random_poisson"] = time.perf_counter() - t0

    # --- Step 6c: Filter N_s > 0 + int(jnp.sum()) ---
    t0 = time.perf_counter()
    has_sat = N_s > 0
    N_s_filtered = N_s[has_sat]
    N_s_tot = int(jnp.sum(N_s_filtered))
    timings["6c. filter+sum N_s"] = time.perf_counter() - t0

    N_halos_with_sat = len(N_s_filtered)

    # --- Step 6d: filter_halo_data (double bool indexing) ---
    t0 = time.perf_counter()
    filtered = filter_halo_data(
        pre_cond, has_sat,
        positions=halo.positions,
        velocities=halo.velocities,
        radius=halo.radius,
        concentration=halo.concentration,
        vrms=halo.vrms,
        shapes=None,
        ratios=None,
    )
    # Block on all filtered arrays
    for v in filtered.values():
        if v is not None and hasattr(v, 'block_until_ready'):
            v.block_until_ready()
    timings["6d. filter_halo_data"] = time.perf_counter() - t0

    # --- Step 6e: position_satellites (NFW fast) ---
    if N_s_tot > 0:
        t0 = time.perf_counter()
        sat_positions = position_satellites(
            key=key_s_pos,
            halo_centers=filtered['positions'],
            Rvir=filtered['radius'],
            c=filtered['concentration'],
            N_s=N_s_filtered,
            N_s_tot=N_s_tot,
            triaxial_NFW=False,
            shapes=None,
            ratios=None,
            f_exp=0.0,
            tau=6.0,
            lambda_NFW=1.0,
        )
        sat_positions.block_until_ready()
        timings["6e. position_satellites"] = time.perf_counter() - t0
    else:
        timings["6e. position_satellites"] = 0.0

    # --- Step 6f: dispersion_velocities_satellites ---
    if N_s_tot > 0:
        t0 = time.perf_counter()
        sat_velocities = dispersion_velocities_satellites(
            key_s_vel, filtered['velocities'], filtered['vrms'],
            N_s_filtered, N_s_tot,
        )
        sat_velocities.block_until_ready()
        timings["6f. dispersion_velocities"] = time.perf_counter() - t0
    else:
        timings["6f. dispersion_velocities"] = 0.0

    # --- Step 7: combine_galaxy_populations ---
    if N_s_tot == 0:
        sat_positions = jnp.array([]).reshape(0, 3)
        sat_velocities = jnp.array([]).reshape(0, 3)

    t0 = time.perf_counter()
    positions_gal, velocities_gal = combine_galaxy_populations(
        cent_positions, cent_velocities, sat_positions, sat_velocities
    )
    positions_gal.block_until_ready()
    timings["7. combine_populations"] = time.perf_counter() - t0

    # --- Step 8: PBC wrapping ---
    t0 = time.perf_counter()
    positions_gal = (positions_gal + Lbox) % Lbox
    positions_gal.block_until_ready()
    timings["8. PBC wrapping"] = time.perf_counter() - t0

    # --- Monolithic reference ---
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=seed)
    halo.positions_gal.block_until_ready()
    timings["MONOLITHIC populate_haloes"] = time.perf_counter() - t0

    sizes = {
        "N_halos": len(halo.logM),
        "N_filtered": N_filtered,
        "N_halos_with_sat": N_halos_with_sat,
        "N_s_tot": N_s_tot,
        "N_centrals": N_centrals,
    }

    return timings, sizes


# ============================================================================
# Raw JAX random benchmarks
# ============================================================================

def benchmark_raw_jax_random(sizes, n_iter=10):
    """
    Time isolated JAX random operations at observed array sizes.
    Returns dict of {op_name: mean_ms}.
    """
    N_halos = sizes["N_halos"]
    N_filtered = sizes["N_filtered"]
    N_s_tot = sizes["N_s_tot"]

    key = jrandom.key(99)
    prob_array = jnp.ones(N_filtered) * 0.5  # dummy probs

    benchmarks = {}

    # 1. uniform for central Bernoulli
    times = []
    for i in range(n_iter):
        k = jrandom.fold_in(key, i)
        t0 = time.perf_counter()
        r = jrandom.uniform(k, (N_halos,))
        r.block_until_ready()
        times.append(time.perf_counter() - t0)
    benchmarks[f"uniform({N_halos:,})  [central Bernoulli]"] = np.mean(times) * 1000

    # 2. poisson for satellite counts
    times = []
    for i in range(n_iter):
        k = jrandom.fold_in(key, 100 + i)
        t0 = time.perf_counter()
        r = jrandom.poisson(k, prob_array, (N_filtered,))
        r.block_until_ready()
        times.append(time.perf_counter() - t0)
    benchmarks[f"poisson({N_filtered:,}) [satellite counts]"] = np.mean(times) * 1000

    if N_s_tot > 0:
        # 3. uniform for CDF inversion
        times = []
        for i in range(n_iter):
            k = jrandom.fold_in(key, 200 + i)
            t0 = time.perf_counter()
            r = jrandom.uniform(k, (N_s_tot,))
            r.block_until_ready()
            times.append(time.perf_counter() - t0)
        benchmarks[f"uniform({N_s_tot:,})  [CDF inversion]"] = np.mean(times) * 1000

        # 4. uniform x2 for sphere sampling
        times = []
        for i in range(n_iter):
            k1, k2 = jrandom.split(jrandom.fold_in(key, 300 + i))
            t0 = time.perf_counter()
            r1 = jrandom.uniform(k1, (N_s_tot,))
            r2 = jrandom.uniform(k2, (N_s_tot,))
            r1.block_until_ready()
            r2.block_until_ready()
            times.append(time.perf_counter() - t0)
        benchmarks[f"uniform({N_s_tot:,})x2 [sphere sampling]"] = np.mean(times) * 1000

        # 5. normal x3 for velocity components
        times = []
        for i in range(n_iter):
            k1, k2, k3 = jrandom.split(jrandom.fold_in(key, 400 + i), 3)
            t0 = time.perf_counter()
            r1 = jrandom.normal(k1, shape=(N_s_tot,))
            r2 = jrandom.normal(k2, shape=(N_s_tot,))
            r3 = jrandom.normal(k3, shape=(N_s_tot,))
            r1.block_until_ready()
            r2.block_until_ready()
            r3.block_until_ready()
            times.append(time.perf_counter() - t0)
        benchmarks[f"normal({N_s_tot:,})x3  [velocity comps]"] = np.mean(times) * 1000

    return benchmarks


# ============================================================================
# Main
# ============================================================================

def main():
    print_header("Population Pipeline Profiler")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading catalogs...")
    t0 = time.perf_counter()
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  Halos: {len(df_halo):,}  (loaded in {fmt_time(time.perf_counter() - t0)})")

    # ------------------------------------------------------------------
    # Initialize HaloOccupation (no particles needed for population)
    # ------------------------------------------------------------------
    print("\nInitializing HaloOccupation...")
    t0 = time.perf_counter()
    halo = HaloOccupation(
        cosmology=cosmo_params,
        zeff=zeff,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition=mass_definition,
        DataFrame=df_halo,
        assembly_bias=False,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False,
    )
    print(f"  Init: {fmt_time(time.perf_counter() - t0)}")

    # ------------------------------------------------------------------
    # Set HOD model and rescale parameters
    # ------------------------------------------------------------------
    print("\nSetting HOD model: ELG_mHMQ")
    halo.set_halo_model("ELG_mHMQ")

    print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, base_hod_params, target_ngal=target_ngal
    )
    hod_params = base_hod_params.copy()
    hod_params["Ac"] = Ac_rescaled
    hod_params["As"] = As_rescaled
    print(f"  Ac = {Ac_rescaled[0]:.6f},  As = {As_rescaled[0]:.6f}")

    # ------------------------------------------------------------------
    # JIT warmup
    # ------------------------------------------------------------------
    print_header("JIT Warmup")

    # Warmup 1: general JIT compilation
    print("  Warmup 1 (seed=0, general JIT)...", end=" ", flush=True)
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=0)
    halo.positions_gal.block_until_ready()
    print(f"{fmt_time(time.perf_counter() - t0)}")

    # Warmup 2: same seed as profiling (compile for exact N_s_tot)
    print(f"  Warmup 2 (seed={RANDOM_SEED}, exact N_s_tot)...", end=" ", flush=True)
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params, random_seed=RANDOM_SEED)
    halo.positions_gal.block_until_ready()
    print(f"{fmt_time(time.perf_counter() - t0)}")

    # ------------------------------------------------------------------
    # Profile sub-steps
    # ------------------------------------------------------------------
    print_header(f"Sub-Step Profiling ({N_ITER} iterations, seed={RANDOM_SEED})")

    all_timings = []
    sizes = None
    for i in range(N_ITER):
        timings, sizes = profile_population_steps(halo, hod_params, RANDOM_SEED)
        all_timings.append(timings)

    # Compute mean timings
    step_names = [k for k in all_timings[0].keys() if k != "MONOLITHIC populate_haloes"]
    mono_key = "MONOLITHIC populate_haloes"

    mean_timings = {}
    for k in list(step_names) + [mono_key]:
        vals = [t[k] for t in all_timings]
        mean_timings[k] = np.mean(vals)

    total_substeps = sum(mean_timings[k] for k in step_names)
    mono_time = mean_timings[mono_key]

    # Print array sizes
    print(f"\n  Array sizes (seed={RANDOM_SEED}):")
    for name, val in sizes.items():
        print(f"    {name:<20s} = {val:>10,}")

    # Print breakdown table
    print(f"\n  {'Step':<30s} {'Mean (ms)':>10s} {'% of sum':>10s}")
    print("  " + "-" * 52)
    for k in step_names:
        ms = mean_timings[k] * 1000
        pct = mean_timings[k] / total_substeps * 100 if total_substeps > 0 else 0
        print(f"  {k:<30s} {ms:>10.2f} {pct:>9.1f}%")
    print("  " + "-" * 52)
    print(f"  {'SUM of sub-steps':<30s} {total_substeps*1000:>10.2f}")
    print(f"  {'MONOLITHIC populate_haloes':<30s} {mono_time*1000:>10.2f}")
    print(f"  {'Overhead (mono - sum)':<30s} {(mono_time - total_substeps)*1000:>10.2f}")

    # ------------------------------------------------------------------
    # Raw JAX random benchmarks
    # ------------------------------------------------------------------
    print_header("Raw JAX Random Benchmarks (10 iterations each)")

    jax_benchmarks = benchmark_raw_jax_random(sizes)

    print(f"\n  {'Operation':<45s} {'Mean (ms)':>10s}")
    print("  " + "-" * 57)
    for name, ms in jax_benchmarks.items():
        print(f"  {name:<45s} {ms:>10.2f}")

    # ------------------------------------------------------------------
    # Varying-seed test
    # ------------------------------------------------------------------
    print_header(f"Varying-Seed Test (seeds 2000-{2000+N_ITER-1}, like benchmark)")
    print("  Different seeds -> different N_s_tot -> potential JIT recompilation")
    print("  Different seeds -> different N_s_tot -> potential JIT recompilation.\n")

    vary_timings = []
    vary_sizes = []
    for i in range(N_ITER):
        seed = 2000 + i
        t0 = time.perf_counter()
        halo.populate_haloes(hod_params, random_seed=seed)
        halo.positions_gal.block_until_ready()
        elapsed = time.perf_counter() - t0

        n_sat = len(halo.positions_gal) - int(jnp.sum(halo.cent_halo_indices >= 0))
        # Rough N_s_tot from satellite count
        n_gal = len(halo.positions_gal)
        n_cent = len(halo.cent_halo_indices)
        n_sat = n_gal - n_cent
        vary_timings.append(elapsed)
        vary_sizes.append(n_sat)
        print(f"  seed={seed}: {fmt_time(elapsed):>12s}  (N_gal={n_gal:,}, N_sat={n_sat:,})")

    print(f"\n  Mean:   {fmt_time(np.mean(vary_timings))}")
    print(f"  Median: {fmt_time(np.median(vary_timings))}")
    print(f"  Min:    {fmt_time(np.min(vary_timings))}")
    print(f"  Max:    {fmt_time(np.max(vary_timings))}")

    # Also time same-seed for direct comparison
    same_seed_times = []
    for i in range(N_ITER):
        t0 = time.perf_counter()
        halo.populate_haloes(hod_params, random_seed=RANDOM_SEED)
        halo.positions_gal.block_until_ready()
        same_seed_times.append(time.perf_counter() - t0)

    print(f"\n  Same-seed (seed={RANDOM_SEED}) comparison:")
    print(f"    Mean: {fmt_time(np.mean(same_seed_times))}")
    print(f"    Recompilation overhead: {fmt_time(np.mean(vary_timings) - np.mean(same_seed_times))}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("Summary")
    print(f"\n  Total sub-step time:    {fmt_time(total_substeps)}")
    print(f"  Monolithic time:        {fmt_time(mono_time)}")
    print(f"  N_halos:                {sizes['N_halos']:,}")
    print(f"  N_s_tot:                {sizes['N_s_tot']:,}")
    print(f"  N_centrals:             {sizes['N_centrals']:,}")

    # Top 3 bottlenecks
    sorted_steps = sorted(step_names, key=lambda k: mean_timings[k], reverse=True)
    print(f"\n  Top 3 bottlenecks:")
    for i, k in enumerate(sorted_steps[:3]):
        ms = mean_timings[k] * 1000
        pct = mean_timings[k] / total_substeps * 100
        print(f"    {i+1}. {k:<28s} {ms:>8.2f} ms  ({pct:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
