"""
Precompute Halo-Center Lensing Cache
=====================================

Runs precompute_halo_center_lensing() on all halos (with downsampled particles)
and saves the HaloCenterLensingCache to HDF5.

This is a one-time precomputation. The resulting cache is consumed by
benchmark_optimized_vs_baseline.py for fast DeltaSigma calculations.

Data: Flamingo L1000N1800
  - Halos:     parquet_halo_catalogue_L1000N1800.parquet
  - Particles: particle_catalogue_L1000N1800_downsampled.parquet
"""

import argparse
import atexit
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import precompute_halo_center_lensing

# ============================================================================
# Configuration (matches numerical_dsigma_example.py; override via CLI)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

# Use >=5x the grid-evaluation fraction (0.05) so the per-halo cached DeltaSigma
# profiles are not noisy; the cache is computed once, so the extra cost is fine.
PARTICLE_FRACTION = 0.25
PARTICLE_SUBSAMPLE_SEED = 99
CACHE_OUTPUT = "halo_center_lensing_cache.h5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--halo_path", default=HALO_PATH)
    p.add_argument("--particle_path", default=PARTICLE_PATH)
    p.add_argument("--output", default=CACHE_OUTPUT)
    p.add_argument("--particle_fraction", type=float, default=PARTICLE_FRACTION)
    p.add_argument("--particle_seed", type=int, default=PARTICLE_SUBSAMPLE_SEED)
    p.add_argument("--n_workers", type=int, default=-1)
    p.add_argument("--tabulate", action="store_true",
                   help="Also tabulate mean xi_gm per (logM [, AB]) bin — "
                        "required by TabulatedDeltaSigma. Free byproduct.")
    p.add_argument("--ab_column", default=None,
                   help="AB environment column (e.g. fs_norm) for the fI "
                        "tabulation dimension; mapped as fE to match the "
                        "B_cent/B_sat fit convention.")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory for per-batch phase-3 checkpoints. Rerun "
                        "with the same value to resume a run that was cut "
                        "short: completed batches are read back instead of "
                        "recomputed. Costs ~2.5 GB at 25%% particles.")
    p.add_argument("--n_logM_bins", type=int, default=40)
    p.add_argument("--n_fI_bins", type=int, default=8)
    p.add_argument("--_dump_inputs", default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--mass_weight", action="store_true",
                   help="Weight particles by their 'mass' column. Required "
                        "for a hydro particle set (DM/gas/star masses differ "
                        "by orders of magnitude); a no-op for DMO.")
    return p.parse_args()


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


# Lensing bins (same as example)
rp_bins = np.geomspace(0.1, 50.0, 26)

# ============================================================================
# Helper
# ============================================================================

def subsample_indices(n, fraction, seed):
    """Sorted subsample indices (sorted for cache locality).

    Returned rather than applied so that positions and masses are guaranteed
    to be subsampled with the *same* draw.
    """
    if fraction >= 1.0:
        return None
    n_keep = int(n * fraction)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, n_keep, replace=False))



def _rss_gb() -> float:
    """Resident set size of this process, in GB."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return float("nan")



def _dump_inputs(args, outdir):
    """Load particles and halos, write them to `outdir`, and exit.

    Runs as a child process launched by main(). Everything this touches --
    pyarrow's decode pools, JAX's XLA backend (pulled in by HaloOccupation),
    and the ~17 GB the parquet decode never returns to the OS -- dies with the
    process instead of being carried into phase 3's fork(). See main().
    """
    cols = ['x', 'y', 'z'] + (['mass'] if args.mass_weight else [])
    print(f"\nLoading particle catalog ({','.join(cols)})...")
    df_part = pd.read_parquet(args.particle_path, columns=cols)
    n_full = len(df_part)
    print(f"  {n_full:,} particles loaded")

    positions_part_full = df_part[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    masses_part_full = (df_part['mass'].to_numpy(dtype=np.float64)
                        if args.mass_weight else None)
    del df_part
    gc.collect()

    keep = subsample_indices(n_full, args.particle_fraction, args.particle_seed)
    positions_part_sub = (positions_part_full if keep is None
                          else positions_part_full[keep])
    masses_part_sub = None if masses_part_full is None else (
        masses_part_full if keep is None else masses_part_full[keep])
    # Apply periodic boundary correction (mirrors HaloOccupation's treatment)
    positions_part_sub = (positions_part_sub + Lbox) % Lbox
    del positions_part_full, masses_part_full, keep
    gc.collect()

    if masses_part_sub is not None:
        print(f"  mass-weighted: m in [{masses_part_sub.min():.3e}, "
              f"{masses_part_sub.max():.3e}] Msun/h, "
              f"ratio {masses_part_sub.max() / masses_part_sub.min():.1f}")
    print(f"  downsampled: {len(positions_part_sub):,} "
          f"({args.particle_fraction * 100:.0f}%)")

    np.save(os.path.join(outdir, "positions_part.npy"), positions_part_sub)
    if masses_part_sub is not None:
        np.save(os.path.join(outdir, "masses_part.npy"), masses_part_sub)
    del positions_part_sub, masses_part_sub
    gc.collect()

    print("\nLoading halo catalog...")
    df_halo = pd.read_parquet(args.halo_path)
    print(f"  {len(df_halo)} halos loaded")

    cmap = dict(column_mapping)
    use_ab = args.tabulate and args.ab_column is not None
    if use_ab:
        cmap["fE"] = args.ab_column

    print("Initializing HaloOccupation (halos only)...")
    halo = HaloOccupation(
        cosmology=dict_cosmo,
        zeff=zeff,
        Lbox=Lbox,
        column_mapping=cmap,
        mass_definition=mass_definition,
        DataFrame=df_halo,
        DataFrame_part=None,
        assembly_bias=use_ab,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False,
    )

    np.save(os.path.join(outdir, "halo_positions.npy"), np.asarray(halo.positions))
    if args.tabulate:
        np.save(os.path.join(outdir, "halo_logM.npy"), np.asarray(halo.logM))
    if use_ab:
        np.save(os.path.join(outdir, "halo_fI.npy"), np.asarray(halo.fE))
    np.savez(os.path.join(outdir, "scalars.npz"),
             RHO_M=halo.RHO_M, rsd_axis=halo.rsd_axis, n_full=n_full)
    print(f"  inputs written to {outdir}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args._dump_inputs:
        _dump_inputs(args, args._dump_inputs)
        return

    print("=" * 60)
    print("Precompute Halo-Center Lensing Cache")
    print("=" * 60)

    # Load in a throwaway child, then memory-map what it wrote.
    #
    # Phase 3 forks 30 workers, and fork() clones only the calling thread: a
    # lock held by any other thread at that instant is inherited locked with no
    # owner, so the child hangs on its first malloc. pyarrow's decode pools and
    # JAX's XLA backend both leave threads alive for the life of the process,
    # and the decode leaves ~17 GB that pool-releasing does not hand back
    # (measured: "19.1 -> 19.1 GB"). Jobs 57402361, 57598215, 57612102 and
    # 57618410 all died silently at that fork -- only 53344937 ever got through,
    # and capping every thread pool at 1 did not help either.
    #
    # Letting a child do the loading and exit returns its threads and its memory
    # to the OS, so the parent reaches the fork single-threaded and small.
    # ~2 GB of .npy at 25% particles: prefer node-local TMPDIR, else sit beside
    # the output (a real filesystem) rather than risk a small /tmp.
    scratch_root = os.environ.get("TMPDIR") or os.path.dirname(os.path.abspath(args.output))
    scratch = tempfile.mkdtemp(prefix="hclc_inputs_", dir=scratch_root)
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)

    print(f"\nLoading inputs in a child process (scratch: {scratch})...")
    t_load = time.perf_counter()
    subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:],
         "--_dump_inputs", scratch],
        check=True,
    )
    print(f"  child exited after {time.perf_counter() - t_load:.1f}s; "
          f"its threads and decode memory are gone")

    # Read into RAM, not mmap_mode="r". The fork only ever needed the parent to
    # be clean of foreign threads and pyarrow arenas, which the child already
    # guarantees; ~2 GB of plain buffers costs the fork nothing. Mapping them
    # instead made every worker fault its way through the file: job 57628352
    # took 62.0s to build the KD-tree over data that took 28.8s in RAM
    # (53344937), and phase 3 ran at 1.22 s/halo against 0.235 s/halo, with
    # 1.3M random-row gathers per halo hitting the mapping.
    def _read(name):
        path = os.path.join(scratch, name)
        return np.load(path) if os.path.exists(path) else None

    positions_part_sub = _read("positions_part.npy")
    masses_part_sub = _read("masses_part.npy")
    halo_positions = _read("halo_positions.npy")
    halo_logM = _read("halo_logM.npy")
    halo_fI = _read("halo_fI.npy")

    scal = np.load(os.path.join(scratch, "scalars.npz"))
    RHO_M = float(scal["RHO_M"])
    rsd_axis = scal["rsd_axis"].item()
    n_full = int(scal["n_full"])
    n_sub = len(positions_part_sub)
    shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nParticle downsampling:")
    print(f"  Full:         {n_full:,}")
    print(f"  Downsampled:  {n_sub:,} ({args.particle_fraction * 100:.0f}%)")
    print(f"  Halos:        {len(halo_positions):,}")
    print(f"  parent RSS before fork: {_rss_gb():.2f} GB")

    # --- Precompute (GC disabled to avoid COW on Python object headers in workers) ---
    print(f"\nPrecomputing DeltaSigma at {len(halo_positions)} halo centers...")
    t0 = time.perf_counter()

    gc.disable()
    cache = precompute_halo_center_lensing(
        halo_positions=halo_positions,
        particle_positions=positions_part_sub,
        particle_masses=masses_part_sub,
        Lbox=Lbox,
        rsd_axis=rsd_axis,
        RHO_M=RHO_M,
        rp_bins=rp_bins,
        verbose=True,
        n_workers=args.n_workers,
        prequery_all=False,
        checkpoint_dir=args.checkpoint_dir,
        halo_logM=halo_logM,
        halo_fI=halo_fI,
        n_logM_bins=args.n_logM_bins,
        n_fI_bins=args.n_fI_bins,
    )
    gc.enable()
    gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"\nPrecomputation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Save ---
    cache.save(args.output)

    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"  N_halos:             {len(halo_positions):,}")
    print(f"  N_particles (full):  {n_full:,}")
    print(f"  N_particles (used):  {n_sub:,}")
    print(f"  Particle fraction:   {args.particle_fraction}")
    print(f"  rp bins:             {len(rp_bins)-1} bins, [{rp_bins[0]:.2f}, {rp_bins[-1]:.1f}] Mpc/h")
    print(f"  Output file:         {args.output}")
    print(f"  Time:                {elapsed:.1f}s")

    # Print sample DeltaSigma values
    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    ds_mean = np.mean(cache.deltasigma, axis=0)
    print(f"\n  Mean DeltaSigma across all halos:")
    print(f"  {'rp [Mpc/h]':>14s}  {'DeltaSigma [h Msun/pc^2]':>26s}")
    print("  " + "-" * 44)
    for r, ds in zip(rp_centers, ds_mean):
        print(f"    {r:12.4f}    {ds:22.6e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
