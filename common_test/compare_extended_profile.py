"""
Extended vs Standard NFW Profile DeltaSigma Comparison
=======================================================

Compares DeltaSigma(rp) computed with:
  - Standard NFW satellite profiles (defaults: f_exp=0, lambda_NFW=1)
  - Extended NFW satellite profiles (lambda_NFW=0.6, f_exp=0.5, tau=5)

Both variants use identical HOD setup (ELG_mHMQ, Flamingo L1000N1800) and
identical random seeds, so profile shape differences are isolated from shot noise.

Extended profile physics:
  - lambda_NFW=0.6 : NFW core shrunk to 60% of Rvir (suppresses small-scale signal)
  - f_exp=0.5      : 50% of satellites follow exponential tail beyond Rvir
  - tau=5          : exponential decay scale = 5 * Rs

No pass/fail threshold — purely exploratory diagnostic.

Downsampling: benchmark-optimal (5% particles, 10% galaxies, 5 realizations).
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
# Configuration
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
PLOT_OUTPUT = "extended_vs_standard_profile.png"

Lbox = 681.0
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
    "gamma": 5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}
target_ngal = 2e-4  # (Mpc/h)^-3

# Downsampling (benchmark optimum)
N_REAL = 5
PARTICLE_FRACTION = 0.05
GALAXY_FRACTION = 0.10
PARTICLE_SUBSAMPLE_SEED = 99
BASE_SEED = 1000

# Lensing bins
rp_bins = np.geomspace(0.1, 50.0, 16)
BINS_COMP = np.geomspace(5e-3, 120, 151)

# Extended profile parameters
EXTENDED_PARAMS = {"lambda_NFW": 1.0, "f_exp": 0.5, "tau": 5}


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
    print("=" * 65)
    print("Extended vs Standard NFW Profile DeltaSigma Comparison")
    print("=" * 65)
    print(f"\nExtended profile params: {EXTENDED_PARAMS}")
    print(f"N realizations: {N_REAL}  |  Particle fraction: {PARTICLE_FRACTION*100:.0f}%  "
          f"|  Galaxy fraction: {GALAXY_FRACTION*100:.0f}%")

    # --- Load data ---
    print("\nLoading halo catalog...")
    t0 = time.perf_counter()
    df_halo = pd.read_parquet(HALO_PATH)
    print(f"  {len(df_halo):,} halos loaded")

    print("Loading particle catalog...")
    df_part = pd.read_parquet(PARTICLE_PATH)
    load_time = time.perf_counter() - t0
    print(f"  {len(df_part):,} particles loaded  ({fmt_time(load_time)})")

    # --- Initialize HaloOccupation ---
    print("\nInitializing HaloOccupation (numba backend)...")
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
        population_backend='numba',
    )

    # --- Set HOD model and rescale Ac/As once (shared by both variants) ---
    print("\nSetting HOD model: ELG_mHMQ")
    halo.set_halo_model("ELG_mHMQ")

    print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, base_hod_params, target_ngal=target_ngal
    )
    hod_params_std = {**base_hod_params, "Ac": Ac_rescaled, "As": As_rescaled}
    hod_params_ext = {**hod_params_std, **EXTENDED_PARAMS}

    print(f"  Ac = {Ac_rescaled[0]:.6f}")
    print(f"  As = {As_rescaled[0]:.6f}")

    # --- Downsample particles ---
    positions_part_full = np.array(halo.positions_part)
    positions_part_sub = subsample_array(
        positions_part_full, PARTICLE_FRACTION, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticle downsampling: {len(positions_part_full):,} -> "
          f"{len(positions_part_sub):,} ({PARTICLE_FRACTION*100:.0f}%)")

    # Overwrite so compute_galaxy_lensing uses downsampled particles
    halo.positions_part = positions_part_sub

    # --- Numba warmup ---
    print("\nWarming up Numba JIT (throw-away population)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params_std, random_seed=0)
    print(f"  Warmup: {fmt_time(time.perf_counter() - t0)}")

    # --- Standard variant ---
    print(f"\nRunning {N_REAL} standard NFW realizations...")
    ds_std_all = []
    rp_centers = None

    for i in range(N_REAL):
        seed = BASE_SEED + i
        print(f"  Realization {i+1}/{N_REAL} (seed={seed})...", end=" ", flush=True)
        t0 = time.perf_counter()

        halo.populate_haloes(hod_params_std, random_seed=seed)
        n_gal = len(halo.positions_gal)
        f_sat = halo.satellite_fraction
        gal_pos_sub = subsample_array(np.array(halo.positions_gal), GALAXY_FRACTION, seed=88 + i)

        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part_sub,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP,
        )
        ds_std_all.append(ds)
        print(f"N_gal={n_gal:,}, f_sat={f_sat:.3f}, time={fmt_time(time.perf_counter() - t0)}")

    ds_std_all = np.array(ds_std_all)
    ds_std_mean = ds_std_all.mean(axis=0)

    # --- Extended variant ---
    print(f"\nRunning {N_REAL} extended NFW realizations "
          f"(lambda_NFW={EXTENDED_PARAMS['lambda_NFW']}, "
          f"f_exp={EXTENDED_PARAMS['f_exp']}, tau={EXTENDED_PARAMS['tau']})...")
    ds_ext_all = []

    for i in range(N_REAL):
        seed = BASE_SEED + i  # identical seeds — isolates profile difference
        print(f"  Realization {i+1}/{N_REAL} (seed={seed})...", end=" ", flush=True)
        t0 = time.perf_counter()

        halo.populate_haloes(hod_params_ext, random_seed=seed)
        n_gal = len(halo.positions_gal)
        f_sat = halo.satellite_fraction
        gal_pos_sub = subsample_array(np.array(halo.positions_gal), GALAXY_FRACTION, seed=88 + i)

        _, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part_sub,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP,
        )
        ds_ext_all.append(ds)
        print(f"N_gal={n_gal:,}, f_sat={f_sat:.3f}, time={fmt_time(time.perf_counter() - t0)}")

    ds_ext_all = np.array(ds_ext_all)
    ds_ext_mean = ds_ext_all.mean(axis=0)

    # --- Per-bin table ---
    ratio = np.where(np.abs(ds_std_mean) > 1e-10, ds_ext_mean / ds_std_mean, np.nan)

    print("\n" + "=" * 65)
    print("Per-bin comparison: Extended vs Standard NFW")
    print("=" * 65)
    print(f"\n  {'rp [Mpc/h]':>14s}  {'DS_std':>14s}  {'DS_ext':>14s}  {'ratio (ext/std)':>16s}")
    print("  " + "-" * 64)
    for r, ds_s, ds_e, rat in zip(rp_centers, ds_std_mean, ds_ext_mean, ratio):
        print(f"    {r:12.4f}  {ds_s:14.6e}  {ds_e:14.6e}  {rat:14.4f}")

    # --- 2-panel plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )

    # Faint individual realizations
    for i in range(N_REAL):
        ax1.semilogx(rp_centers, rp_centers * ds_std_all[i],
                     color="steelblue", alpha=0.20, lw=0.6)
        ax1.semilogx(rp_centers, rp_centers * ds_ext_all[i],
                     color="tomato", alpha=0.20, lw=0.6)

    # Mean curves
    ax1.semilogx(rp_centers, rp_centers * ds_std_mean,
                 "b-o", lw=2, ms=5, label=f"Standard NFW (mean of {N_REAL})")
    ax1.semilogx(rp_centers, rp_centers * ds_ext_mean,
                 "r--s", lw=2, ms=5,
                 label=rf"Extended NFW: $\lambda$={EXTENDED_PARAMS['lambda_NFW']}, "
                       rf"$f_{{exp}}$={EXTENDED_PARAMS['f_exp']}, "
                       rf"$\tau$={EXTENDED_PARAMS['tau']} (mean of {N_REAL})")

    ax1.set_ylabel(r"$r_p \cdot \Delta\Sigma\ [h\,M_\odot/\mathrm{pc}^2 \cdot \mathrm{Mpc}/h]$")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("DeltaSigma: Extended vs Standard NFW Profile")

    # Ratio panel
    ax2.semilogx(rp_centers, ratio, "ko-", ms=4)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.axhspan(0.95, 1.05, color="green", alpha=0.10, label=r"$\pm 5\%$")
    ax2.set_xlabel(r"$r_p$ [Mpc/h]")
    ax2.set_ylabel("Ratio (ext / std)")
    ax2.set_ylim(0.5, 1.5)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {PLOT_OUTPUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
