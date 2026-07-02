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

import gc
import time
import numpy as np
import pandas as pd

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import precompute_halo_center_lensing

# ============================================================================
# Configuration (matches numerical_dsigma_example.py)
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

# Use >=5x the grid-evaluation fraction (0.05) so the per-halo cached DeltaSigma
# profiles are not noisy; the cache is computed once, so the extra cost is fine.
PARTICLE_FRACTION = 0.25
PARTICLE_SUBSAMPLE_SEED = 99
CACHE_OUTPUT = "halo_center_lensing_cache.h5"

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

def subsample_array(positions, fraction, seed):
    """Subsample positions array with sorted indices for cache locality."""
    if fraction >= 1.0:
        return positions
    n_keep = int(len(positions) * fraction)
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(positions), n_keep, replace=False))
    return positions[indices]


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Precompute Halo-Center Lensing Cache")
    print("=" * 60)

    # --- Load particles directly: only x,y,z, then subsample and free full array ---
    print("\nLoading particle catalog (x,y,z only)...")
    df_part = pd.read_parquet(PARTICLE_PATH, columns=['x', 'y', 'z'])
    n_full = len(df_part)
    print(f"  {n_full:,} particles loaded")

    positions_part_full = df_part[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    del df_part
    gc.collect()

    positions_part_sub = subsample_array(
        positions_part_full, PARTICLE_FRACTION, PARTICLE_SUBSAMPLE_SEED
    )
    # Apply periodic boundary correction (mirrors HaloOccupation's treatment)
    positions_part_sub = (positions_part_sub + Lbox) % Lbox
    n_sub = len(positions_part_sub)
    del positions_part_full
    gc.collect()

    print(f"\nParticle downsampling:")
    print(f"  Full:         {n_full:,}")
    print(f"  Downsampled:  {n_sub:,} ({PARTICLE_FRACTION*100:.0f}%)")

    # --- Load halos via HaloOccupation (no particle data) ---
    print("\nLoading halo catalog...")
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo)} halos loaded")

    print("Initializing HaloOccupation (halos only)...")
    halo = HaloOccupation(
        cosmology=dict_cosmo,
        zeff=zeff,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition=mass_definition,
        DataFrame=df_halo,
        DataFrame_part=None,
        assembly_bias=False,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False,
    )

    halo_positions = np.array(halo.positions)
    RHO_M = halo.RHO_M
    rsd_axis = halo.rsd_axis
    del halo, df_halo
    gc.collect()

    # --- Precompute (GC disabled to avoid COW on Python object headers in workers) ---
    print(f"\nPrecomputing DeltaSigma at {len(halo_positions)} halo centers...")
    t0 = time.perf_counter()

    gc.disable()
    cache = precompute_halo_center_lensing(
        halo_positions=halo_positions,
        particle_positions=positions_part_sub,
        Lbox=Lbox,
        rsd_axis=rsd_axis,
        RHO_M=RHO_M,
        rp_bins=rp_bins,
        verbose=True,
        prequery_all=False,
    )
    gc.enable()
    gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"\nPrecomputation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Save ---
    cache.save(CACHE_OUTPUT)

    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"  N_halos:             {len(halo_positions):,}")
    print(f"  N_particles (full):  {n_full:,}")
    print(f"  N_particles (used):  {n_sub:,}")
    print(f"  Particle fraction:   {PARTICLE_FRACTION}")
    print(f"  rp bins:             {len(rp_bins)-1} bins, [{rp_bins[0]:.2f}, {rp_bins[-1]:.1f}] Mpc/h")
    print(f"  Output file:         {CACHE_OUTPUT}")
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
