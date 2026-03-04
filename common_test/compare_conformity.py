"""
Standard vs Conformity HOD DeltaSigma Comparison
=================================================

Compares DeltaSigma(rp) computed with:
  - Standard HOD  (ELG_mHMQ, conformity=False)
  - Conformity HOD (ELG_mHMQ, conformity=True, kappa_EE=0.7)

With conformity, satellite occupation depends on whether a central galaxy is
actually realised (not just its probability). kappa_EE scales the satellite
mass threshold: log10(M1_EE) = log10(M1) + log10(kappa_EE), so kappa_EE < 1
lowers M1 for halos that host a central, producing more satellites around
centrals relative to the no-central case.

Both variants use identical random seeds to isolate the conformity signal
from shot noise. Ac/As are re-rescaled independently for each variant to
enforce the same target galaxy number density.

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
PLOT_OUTPUT = "conformity_vs_standard.png"

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

# Conformity parameter
KAPPA_EE = 0.7


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


def run_realizations(halo, hod_params, positions_part_sub, label):
    """Run N_REAL populate+lensing realizations and return stacked DeltaSigma."""
    print(f"\nRunning {N_REAL} {label} realizations...")
    ds_all = []
    rp_centers = None

    for i in range(N_REAL):
        seed = BASE_SEED + i
        print(f"  Realization {i+1}/{N_REAL} (seed={seed})...", end=" ", flush=True)
        t0 = time.perf_counter()

        halo.populate_haloes(hod_params, random_seed=seed)
        n_gal = len(halo.positions_gal)
        f_sat = halo.satellite_fraction
        gal_pos_sub = subsample_array(
            np.array(halo.positions_gal), GALAXY_FRACTION, seed=88 + i
        )

        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, positions_part_sub,
            Lbox, "z", halo.RHO_M, rp_bins, bins_comp=BINS_COMP,
        )
        ds_all.append(ds)
        print(f"N_gal={n_gal:,}, f_sat={f_sat:.3f}, time={fmt_time(time.perf_counter() - t0)}")

    return rp_centers, np.array(ds_all)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("Standard vs Conformity HOD DeltaSigma Comparison")
    print("=" * 65)
    print(f"\nConformity parameter: kappa_EE = {KAPPA_EE}")
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

    # --- Downsample particles once (shared by both variants) ---
    positions_part_full = np.array(halo.positions_part)
    positions_part_sub = subsample_array(
        positions_part_full, PARTICLE_FRACTION, PARTICLE_SUBSAMPLE_SEED
    )
    print(f"\nParticle downsampling: {len(positions_part_full):,} -> "
          f"{len(positions_part_sub):,} ({PARTICLE_FRACTION*100:.0f}%)")
    halo.positions_part = positions_part_sub

    # -----------------------------------------------------------------------
    # Standard variant (conformity=False)
    # -----------------------------------------------------------------------
    print("\n--- Setting up standard HOD (conformity=False) ---")
    halo.set_halo_model("ELG_mHMQ", conformity=False)

    print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
    Ac_std, As_std = rescale_Ac_to_target_ngal(
        halo.HOD, base_hod_params, target_ngal=target_ngal
    )
    hod_params_std = {**base_hod_params, "Ac": Ac_std, "As": As_std}
    print(f"  Ac = {Ac_std[0]:.6f},  As = {As_std[0]:.6f}")

    print("Warming up Numba JIT (standard)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params_std, random_seed=0)
    print(f"  Warmup: {fmt_time(time.perf_counter() - t0)}")

    rp_centers, ds_std_all = run_realizations(
        halo, hod_params_std, positions_part_sub, label="standard NFW"
    )
    ds_std_mean = ds_std_all.mean(axis=0)

    # -----------------------------------------------------------------------
    # Conformity variant (conformity=True)
    # -----------------------------------------------------------------------
    print("\n--- Setting up conformity HOD (conformity=True, kappa_EE={}) ---".format(KAPPA_EE))
    halo.set_halo_model("ELG_mHMQ", conformity=True)

    # Include kappa_EE in params for correct ngal rescaling
    base_hod_params_conf = {**base_hod_params, "kappa_EE": KAPPA_EE}
    print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
    Ac_conf, As_conf = rescale_Ac_to_target_ngal(
        halo.HOD, base_hod_params_conf, target_ngal=target_ngal
    )
    hod_params_conf = {**base_hod_params_conf, "Ac": Ac_conf, "As": As_conf}
    print(f"  Ac = {Ac_conf[0]:.6f},  As = {As_conf[0]:.6f}")

    print("Warming up Numba JIT (conformity)...")
    t0 = time.perf_counter()
    halo.populate_haloes(hod_params_conf, random_seed=0)
    print(f"  Warmup: {fmt_time(time.perf_counter() - t0)}")

    _, ds_conf_all = run_realizations(
        halo, hod_params_conf, positions_part_sub, label=f"conformity (kappa_EE={KAPPA_EE})"
    )
    ds_conf_mean = ds_conf_all.mean(axis=0)

    # --- Per-bin table ---
    ratio = np.where(
        np.abs(ds_std_mean) > 1e-10, ds_conf_mean / ds_std_mean, np.nan
    )

    print("\n" + "=" * 65)
    print("Per-bin comparison: Conformity vs Standard HOD")
    print("=" * 65)
    print(f"\n  {'rp [Mpc/h]':>14s}  {'DS_std':>14s}  {'DS_conf':>14s}  {'ratio (conf/std)':>17s}")
    print("  " + "-" * 65)
    for r, ds_s, ds_c, rat in zip(rp_centers, ds_std_mean, ds_conf_mean, ratio):
        print(f"    {r:12.4f}  {ds_s:14.6e}  {ds_c:14.6e}  {rat:15.4f}")

    # --- 2-panel plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )

    # Faint individual realizations
    for i in range(N_REAL):
        ax1.semilogx(rp_centers, rp_centers * ds_std_all[i],
                     color="steelblue", alpha=0.20, lw=0.6)
        ax1.semilogx(rp_centers, rp_centers * ds_conf_all[i],
                     color="tomato", alpha=0.20, lw=0.6)

    # Mean curves
    ax1.semilogx(rp_centers, rp_centers * ds_std_mean,
                 "b-o", lw=2, ms=5, label=f"Standard HOD (mean of {N_REAL})")
    ax1.semilogx(rp_centers, rp_centers * ds_conf_mean,
                 "r--s", lw=2, ms=5,
                 label=rf"Conformity HOD: $\kappa_{{EE}}={KAPPA_EE}$ (mean of {N_REAL})")

    ax1.set_ylabel(r"$r_p \cdot \Delta\Sigma\ [h\,M_\odot/\mathrm{pc}^2 \cdot \mathrm{Mpc}/h]$")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("DeltaSigma: Conformity vs Standard HOD")

    # Ratio panel
    ax2.semilogx(rp_centers, ratio, "ko-", ms=4)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.axhspan(0.95, 1.05, color="green", alpha=0.10, label=r"$\pm 5\%$")
    ax2.set_xlabel(r"$r_p$ [Mpc/h]")
    ax2.set_ylabel("Ratio (conf / std)")
    ax2.set_ylim(0.5, 1.5)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {PLOT_OUTPUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
