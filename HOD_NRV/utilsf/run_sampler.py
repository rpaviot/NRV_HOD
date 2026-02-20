"""
run_sampler.py — Nautilus nested sampler for DeltaSigma HOD fitting.

Wires up the standard Flamingo L1000N1800 configuration and runs
NumericalDeltaSigmaFitter. Edit the paths and settings in the sections
below, then run:

    python HOD_NRV/utilsf/run_sampler.py
"""

import numpy as np
import pandas as pd

from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import HaloCenterLensingCache
from HOD_NRV.utilsf.numerical_sampler import NumericalDeltaSigmaFitter, FitCase


# ============================================================================
# Section 1 — Paths
# ============================================================================

HALO_PATH     = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
CACHE_PATH    = "halo_center_lensing_cache.h5"
DATA_PATH     = ""   # TODO: fill in path to observed DeltaSigma .npz
OUTPUT_PATH   = "sampler_results.npz"


# ============================================================================
# Section 2 — Simulation configuration
# ============================================================================

Lbox = 681.0          # Mpc/h
zeff = 1.0
mass_definition = "200m"

cosmo_params = {
    "H0": 68.1,
    "Om0": 0.306,
    "Ob0": 0.0486,
    "sigma8": 0.807,
    "ns": 0.967,
}

column_mapping = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}


# ============================================================================
# Section 3 — Sampler settings
# ============================================================================

PARTICLE_FRACTION = 0.05   # optimal: 5% → <0.5% DeltaSigma error
GALAXY_FRACTION   = 0.10   # optimal: 10%
N_REALIZATIONS    = 5
TARGET_NGAL       = 2.3e-4
M1_FIXED          = 13.0
BASE_SEED         = 1000

RP_MIN = None   # e.g. 0.3  — set to apply min scale cut [Mpc/h]
RP_MAX = None   # e.g. 30.0 — set to apply max scale cut [Mpc/h]

FIT_CASE   = FitCase.STANDARD_NFW   # or EXTENDED_PROFILE / CONFORMITY

N_LIVE     = 500
N_EFF      = 5000
CHECKPOINT = "sampler_checkpoint.hdf5"


# ============================================================================
# Section 4 — Main
# ============================================================================

def main():
    if not DATA_PATH:
        raise ValueError("DATA_PATH is not set. Edit run_sampler.py and provide the path to the observed DeltaSigma .npz file.")

    # --- Load catalogs ---
    print("Loading halo catalog ...")
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo):,} halos loaded")

    print("Loading particle catalog ...")
    df_part = pd.read_parquet(PARTICLE_PATH)
    print(f"  {len(df_part):,} particles loaded")

    # --- Load precomputed lensing cache ---
    print(f"Loading lensing cache from {CACHE_PATH} ...")
    precomputed_cache = HaloCenterLensingCache.load(CACHE_PATH)

    # --- Build fitter ---
    print("Initializing NumericalDeltaSigmaFitter ...")
    fitter = NumericalDeltaSigmaFitter(
        # HaloOccupation args
        cosmology=cosmo_params,
        zeff=zeff,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition=mass_definition,
        DataFrame=df_halo,
        DataFrame_part=df_part,
        assembly_bias=False,
        NFW_scaled=True,
        outerprofile=True,
        apply_rsd=False,
        triaxial_NFW=False,
        rsd_axis='z',
        do_test=False,
        particle_fraction=PARTICLE_FRACTION,
        particle_subsample_seed=42,
        # Cache
        precomputed_cache=precomputed_cache,
        # Fitter args
        fit_case=FIT_CASE,
        data_path=DATA_PATH,
        target_ngal=TARGET_NGAL,
        n_realizations=N_REALIZATIONS,
        galaxy_fraction=GALAXY_FRACTION,
        M1_fixed=M1_FIXED,
        base_seed=BASE_SEED,
        rp_min=RP_MIN,
        rp_max=RP_MAX,
    )

    print(f"\nFit case      : {FIT_CASE.name}")
    print(f"Free params   : {fitter.param_names}")
    print(f"Data bins     : {fitter.n_bins}  (rp in [{fitter.rp[0]:.3f}, {fitter.rp[-1]:.3f}] Mpc/h)")
    print(f"N realizations: {N_REALIZATIONS}")
    print(f"Galaxy frac   : {GALAXY_FRACTION}")
    print(f"Particle frac : {PARTICLE_FRACTION}")
    print(f"Checkpoint    : {CHECKPOINT}")
    print()

    # --- Run sampler ---
    points, weights, log_l, log_z = fitter.run(
        n_live=N_LIVE,
        n_eff=N_EFF,
        filepath=CHECKPOINT,
        verbose=True,
    )

    # --- Save results ---
    fitter.save_results(OUTPUT_PATH, points, weights, log_l, log_z)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # --- MAP best-fit ---
    best = fitter.get_best_fit(points, log_l)
    print("\nMAP best-fit parameters:")
    for k, v in best.items():
        print(f"  {k:<15s} = {v:.6g}")

    # --- Posterior summary table ---
    summary = fitter.get_posterior_summary(points, weights)
    print(f"\n{'Parameter':<15s} {'mean':>10s} {'std':>10s} {'median':>10s} {'q16':>10s} {'q84':>10s}")
    print("-" * 57)
    for name, s in summary.items():
        print(f"{name:<15s} {s['mean']:>10.4f} {s['std']:>10.4f} {s['median']:>10.4f} {s['q16']:>10.4f} {s['q84']:>10.4f}")

    print(f"\nlog Z = {log_z:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
