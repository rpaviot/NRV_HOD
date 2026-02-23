"""
diagnostic_prior_sample.py — Prior sample → DeltaSigma trace

Samples N_SAMPLES random theta vectors from the prior and traces every
intermediate step to pinpoint where the pipeline breaks.

Run from the repo root:
    python common_test/diagnostic_prior_sample.py

Diagnostic output per sample:
    1. Raw theta (param name → value → [low, high])
    2. Rescaling step: ngal_fid, rescale_factor, Ac_new, As_new, Ac/As
    3. Physicality check
    4. Full params dict passed to lensing
    5. compute_avg_lensing result (ds_mean) or exception traceback
    6. PASS / FAIL summary with reason
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (mirrors run_sampler.py exactly)
# ---------------------------------------------------------------------------

HALO_PATH      = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH  = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
CACHE_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "halo_center_lensing_cache.h5")

TARGET_NGAL       = 2.3e-4
M1_FIXED          = 13.0
PARTICLE_FRACTION = 0.05
GALAXY_FRACTION   = 0.10
BASE_SEED         = 1000
N_SAMPLES         = 5

Lbox             = 681.0
zeff             = 1.0
mass_definition  = "200m"

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

# Prior bounds — STANDARD_NFW case (matches _BASE_PARAMS in numerical_sampler.py)
PRIOR_BOUNDS = [
    ("As",         0.002, 0.05),
    ("Mmin",       11.5,  13.5),
    ("sig_M",      0.1,   2.0),
    ("gamma",      0.0,   10.0),
    ("alpha",      0.1,   2.0),
    ("kappa",      0.1,   2.0),
    ("lambda_NFW", 0.1,   2.0),
]

# rp bins: simple default, no need to match data
RP_BINS = np.geomspace(0.1, 50.0, 16)

AC_FID = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char="=", width=72):
    print(char * width)


def _sample_prior(rng):
    """Draw one theta vector uniformly from PRIOR_BOUNDS."""
    return np.array([
        rng.uniform(low, high)
        for _, low, high in PRIOR_BOUNDS
    ])


def _print_theta(theta):
    print("  Raw theta:")
    for (name, low, high), val in zip(PRIOR_BOUNDS, theta):
        print(f"    {name:<12s} = {val:10.5f}   [{low}, {high}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Imports (deferred so startup errors are visible) ---
    from HOD_NRV.HOD_numerical.HOD import HaloOccupation
    from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
        HaloCenterLensingCache,
    )
    from HOD_NRV.utilsf.emulator_utils import (
        rescale_Ac_to_target_ngal,
        compute_ngal_with_fiducial_Ac,
    )

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
    print(f"  Cache rp_bins: {precomputed_cache.rp_bins}")

    # --- Build HaloOccupation ---
    print("Initialising HaloOccupation ...")
    halo = HaloOccupation(
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
    )
    halo.set_halo_model("ELG_mHMQ", conformity=False)
    print("  HaloOccupation ready.\n")

    # --- Sample prior ---
    rng = np.random.default_rng(42)
    thetas = [_sample_prior(rng) for _ in range(N_SAMPLES)]

    pass_count = 0
    fail_count = 0

    for s_idx, theta in enumerate(thetas):
        _sep()
        print(f"SAMPLE {s_idx + 1} / {N_SAMPLES}")
        _sep("-")

        # ---- Step 1: raw theta ----
        _print_theta(theta)

        theta_dict = {name: val for (name, _, _), val in zip(PRIOR_BOUNDS, theta)}

        As_sampled = theta_dict["As"]

        params_for_rescale = {
            "Mmin":  theta_dict["Mmin"],
            "sig_M": theta_dict["sig_M"],
            "gamma": theta_dict["gamma"],
            "M1":    M1_FIXED,
            "alpha": theta_dict["alpha"],
            "kappa": theta_dict["kappa"],
            "As":    As_sampled,
        }

        # ---- Step 2: rescaling ----
        print("\n  [Step 2] Rescaling ...")
        try:
            ngal_fid = compute_ngal_with_fiducial_Ac(
                halo.HOD, params_for_rescale, Ac_fiducial=AC_FID
            )
            rescale_factor = TARGET_NGAL / ngal_fid if ngal_fid != 0 else float("nan")
            Ac_new = AC_FID * rescale_factor
            As_new = As_sampled * rescale_factor

            print(f"    ngal_fid       = {ngal_fid[0]:.4e}  (target = {TARGET_NGAL:.4e})")
            print(f"    rescale_factor = {rescale_factor[0]:.4e}")
            print(f"    Ac_new         = {Ac_new[0]:.4e}")
            print(f"    As_new         = {As_new[0]:.4e}")
            print(f"    Ac/As ratio    = {(Ac_new[0] / As_new[0] if As_new != 0 else float('nan')):.4f}")
        except Exception:
            print("    EXCEPTION during rescaling:")
            traceback.print_exc(file=sys.stdout)
            fail_count += 1
            print(f"\n  >> FAIL (rescaling exception)")
            continue

        # ---- Step 3: physicality ----
        print("\n  [Step 3] Physicality check ...")
        ac_ok = Ac_new > 0 and np.isfinite(Ac_new)
        as_ok = As_new > 0 and np.isfinite(As_new)
        print(f"    Ac > 0 and finite: {ac_ok}   (Ac = {Ac_new[0]:.4e})")
        print(f"    As > 0 and finite: {as_ok}   (As = {As_new[0]:.4e})")

        if not (ac_ok and as_ok):
            fail_count += 1
            reason = []
            if not ac_ok:
                reason.append(f"Ac = {Ac_new:.4e}")
            if not as_ok:
                reason.append(f"As = {As_new:.4e}")
            print(f"\n  >> FAIL (unphysical: {', '.join(reason)})")
            continue

        # ---- Step 4: full params dict ----
        full_params = {
            "Ac":         Ac_new,
            "As":         As_new,
            "Mmin":       theta_dict["Mmin"],
            "sig_M":      theta_dict["sig_M"],
            "gamma":      theta_dict["gamma"],
            "M1":         M1_FIXED,
            "alpha":      theta_dict["alpha"],
            "kappa":      theta_dict["kappa"],
            "lambda_NFW": theta_dict["lambda_NFW"],
        }
        # print("\n  [Step 4] Full params dict:")
        # for k, v in full_params.items():
        #     print(f"    {k:<12s} = {v[9]:.6g}")

        # ---- Step 5: compute_avg_lensing ----
        print("\n  [Step 5] compute_avg_lensing ...")
        try:
            rp_out, ds_mean, ds_std = halo.compute_avg_lensing(
                full_params,
                n_realizations=1,
                bins1=RP_BINS,
                method="optimized",
                precomputed_cache=precomputed_cache,
                galaxy_fraction=GALAXY_FRACTION,
                base_seed=BASE_SEED,
            )
            print(ds_mean)
            all_finite    = bool(np.all(np.isfinite(ds_mean)))
            all_positive  = bool(np.all(ds_mean > 0))
            #print(f"    ds_mean = {np.array2string(ds_mean, precision=3, floatmode='sci')}")
            print(f"    all finite   : {all_finite}")
            print(f"    all positive : {all_positive}")
        except Exception:
            print("    EXCEPTION during compute_avg_lensing:")
            traceback.print_exc(file=sys.stdout)
            fail_count += 1
            print(f"\n  >> FAIL (lensing exception)")
            continue

        # ---- Step 6: summary ----
        if all_finite:
            pass_count += 1
            print(f"\n  >> PASS")
        else:
            fail_count += 1
            print(f"\n  >> FAIL (ds_mean not all finite)")

    _sep()
    print(f"SUMMARY: {pass_count} PASS  /  {fail_count} FAIL  (out of {N_SAMPLES})")
    _sep()


if __name__ == "__main__":
    main()
