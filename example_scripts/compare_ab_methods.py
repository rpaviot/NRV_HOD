"""
Baseline vs Assembly Bias HOD: direct vs variant vs mass comparison
====================================================================

Compares DeltaSigma(rp) computed with:
  1. Baseline  — no assembly bias (loaded from baseline_dsigma_cache.npz)
  2. AB direct  — Ncen/Nsat modified by step-function fE  (ab_method='direct')
  3. AB variant — paper Eq. 12-13 with continuous fE  (ab_method='variant')
  4. AB mass    — Mmin/M1 shifted by A*fE  (ab_method='mass')

The AB catalogue must already be computed via compute_assembly_bias.py;
it supplies the 'delta_norm' column used as fE.

Output: compare_ab_methods.png
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import compute_galaxy_lensing
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration
# ============================================================================

HALO_AB_PATH  = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800_AB.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
BASELINE_CACHE = os.path.join(os.path.dirname(__file__), "baseline_dsigma_cache.npz")
PLOT_OUTPUT = os.path.join(os.path.dirname(__file__), "compare_ab_methods.png")


LBOX             = 681.0
ZEFF             = 1.0
MASS_DEFINITION  = "MassDef200m"
N_REAL           = 10
BASE_SEED        = 1000
PARTICLE_FRACTION        = 0.05
PARTICLE_SUBSAMPLE_SEED  = 99
GALAXY_FRACTION  = 0.20

TARGET_NGAL = 2e-4   # (Mpc/h)^-3

B_CENT = 0.5
B_SAT  = 0.

column_mapping = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
    "fE": "fs_norm",   # primary environmental property (required for AB)
    # "fB": "fs_norm",    # uncomment to include tidal shear as secondary property
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
    "As":    0.3,
    "Mmin":  12.7,
    "sig_M": 0.3,
    "M1":    13.0,
    "gamma": 5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}

rp_bins = np.geomspace(0.1, 50.0, 16)
BINS_COMP = np.geomspace(5e-3, 120, 151)


# ============================================================================
# Helpers
# ============================================================================

def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds/60:.1f} min"


def run_ab_variant(halo, ab_method, ab_params, label):
    """Set up HOD model with given ab_method and run N_REAL realizations.

    ab_params: dict of AB parameters to merge into hod_params
               (e.g. {"B_cent": 0.5, "B_sat": -0.5} for direct/variant,
                     {"A_cent": 0.5, "A_sat": -0.5} for mass)

    Returns rp_centers, ds_mean, ds_std, ngal_all
    where ngal_all is the number density (Mpc/h)^-3 per realization.
    """
    print(f"\n--- {label} ---")
    halo.set_halo_model("ELG_mHMQ", ab_method=ab_method)

    # Ac/As rescaling uses compute_ngal; mean fE = 0 by construction (step function)
    Ac, As = rescale_Ac_to_target_ngal(halo, {**base_hod_params, **ab_params}, target_ngal=TARGET_NGAL)
    hod_params = {**base_hod_params, "Ac": Ac, "As": As, **ab_params}

    # JIT warmup
    print("  Warming up Numba JIT...")
    halo.populate_haloes(hod_params, random_seed=0)

    print(f"  Running {N_REAL} realizations...")
    t0 = time.perf_counter()
    ds_all = []
    ngal_all = []
    for i in range(N_REAL):
        seed = BASE_SEED + i
        halo.populate_haloes(hod_params, random_seed=seed)
        ngal_all.append(len(halo.positions_gal) / LBOX**3)
        gal_pos = np.array(halo.positions_gal)
        gal_pos_sub = halo._subsample_array(
            gal_pos, GALAXY_FRACTION, seed=BASE_SEED + N_REAL + i
        )
        rp_centers, ds = compute_galaxy_lensing(
            gal_pos_sub, halo.positions_part, halo.Lbox,
            halo.rsd_axis, halo.RHO_M, rp_bins,
            bins_comp=BINS_COMP,
        )
        ds_all.append(ds)

    ds_all = np.array(ds_all)
    ngal_all = np.array(ngal_all)
    print(f"  Done in {fmt_time(time.perf_counter() - t0)}")
    print(f"  ngal: {ngal_all.mean():.4e} ± {ngal_all.std():.2e} (Mpc/h)^-3")
    return rp_centers, ds_all.mean(axis=0), ds_all.std(axis=0, ddof=1), ngal_all


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("Baseline vs AB direct vs AB variant: DeltaSigma comparison")
    print("=" * 65)
    print(f"\nB_CENT={B_CENT}, B_SAT={B_SAT}")
    print(f"N realizations: {N_REAL}  |  Particle fraction: {PARTICLE_FRACTION*100:.0f}%"
          f"  |  Galaxy fraction: {GALAXY_FRACTION*100:.0f}%")

    # --- Load baseline ---
    print(f"\nLoading baseline from {BASELINE_CACHE}...")
    baseline = np.load(BASELINE_CACHE)
    ds_baseline_all  = baseline["ds_all"]
    ngal_baseline    = baseline["ngal"]          # number density per realization
    n_base = len(ds_baseline_all)
    ds_baseline_mean = ds_baseline_all.mean(axis=0)
    ds_baseline_std  = ds_baseline_all.std(axis=0, ddof=1)
    print(f"  {n_base} baseline realizations loaded")
    print(f"  baseline ngal: {ngal_baseline.mean():.4e} ± {ngal_baseline.std():.2e} (Mpc/h)^-3")

    # --- Load AB halo catalogue ---
    print(f"\nLoading AB halo catalogue: {HALO_AB_PATH}")
    df_halo = pd.read_parquet(HALO_AB_PATH)
    print(f"  {len(df_halo):,} halos")

    print(f"Loading particle catalogue: {PARTICLE_PATH}")
    df_part = pd.read_parquet(PARTICLE_PATH)
    print(f"  {len(df_part):,} particles")

    # --- Initialize HaloOccupation (assembly_bias=True) ---
    print("\nInitializing HaloOccupation (assembly_bias=True, numba backend)...")
    halo = HaloOccupation(
        cosmology=dict_cosmo,
        zeff=ZEFF,
        Lbox=LBOX,
        column_mapping=column_mapping,
        mass_definition=MASS_DEFINITION,
        DataFrame=df_halo,
        DataFrame_part=df_part,
        assembly_bias=True,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False,
        population_backend='numba',
        particle_fraction=PARTICLE_FRACTION,
        mass_function='Despali16',
        particle_subsample_seed=PARTICLE_SUBSAMPLE_SEED,
    )
    print(f"  fE (delta_norm) range: [{float(halo.fE.min()):.3f}, {float(halo.fE.max()):.3f}]")

    # --- Run AB direct ---
    rp_centers, ds_direct_mean, ds_direct_std, ngal_direct = run_ab_variant(
        halo, 'direct',
        {"B_cent": B_CENT, "B_sat": B_SAT},
        f"AB direct (B_CENT={B_CENT}, B_SAT={B_SAT})",
    )

    # --- Run AB variant ---
    _, ds_variant_mean, ds_variant_std, ngal_variant = run_ab_variant(
        halo, 'variant',
        {"B_cent": B_CENT, "B_sat": B_SAT},
        f"AB variant (B_CENT={B_CENT}, B_SAT={B_SAT})",
    )

    # --- Run AB mass ---
    _, ds_mass_mean, ds_mass_std, ngal_mass = run_ab_variant(
        halo, 'mass',
        {"B_cent": B_CENT, "B_sat": B_SAT},
        f"AB mass (B_CENT={B_CENT}, B_SAT={B_SAT})",
    )

    # --- ngal summary ---
    print("\n" + "=" * 55)
    print("Galaxy number density comparison  [(Mpc/h)^-3]")
    print("=" * 55)
    print(f"  Baseline   : {ngal_baseline.mean():.4e} ± {ngal_baseline.std():.2e}"
          f"  (ratio 1.000)")
    for label, ngal in [("AB direct ", ngal_direct), ("AB variant", ngal_variant),
                         ("AB mass   ", ngal_mass)]:
        ratio = ngal.mean() / ngal_baseline.mean()
        print(f"  {label} : {ngal.mean():.4e} ± {ngal.std():.2e}"
              f"  (ratio {ratio:.4f})")

    # --- Per-bin table ---
    rp = rp_centers
    safe = np.abs(ds_baseline_mean) > 1e-10
    ratio_direct  = np.where(safe, ds_direct_mean  / ds_baseline_mean, np.nan)
    ratio_variant = np.where(safe, ds_variant_mean / ds_baseline_mean, np.nan)
    ratio_mass    = np.where(safe, ds_mass_mean    / ds_baseline_mean, np.nan)

    print("\n" + "=" * 100)
    print("Per-bin ΔΣ: Baseline | AB direct | AB variant | AB mass  (ratio vs baseline)")
    print("=" * 100)
    hdr = (f"  {'rp [Mpc/h]':>12s}  {'Baseline':>14s}  {'AB direct':>14s}  "
           f"{'AB variant':>14s}  {'AB mass':>14s}  "
           f"{'direct/base':>12s}  {'variant/base':>13s}  {'mass/base':>10s}")
    print(hdr)
    print("  " + "-" * 98)
    for r, db, dd, dv, dm, rd, rv, rm in zip(
            rp, ds_baseline_mean, ds_direct_mean, ds_variant_mean, ds_mass_mean,
            ratio_direct, ratio_variant, ratio_mass):
        print(f"  {r:12.4f}  {db:14.6e}  {dd:14.6e}  {dv:14.6e}  {dm:14.6e}"
              f"  {rd:12.4f}  {rv:13.4f}  {rm:10.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 11),
                             height_ratios=[3, 1, 1.2],
                             gridspec_kw={"hspace": 0.08})
    ax1, ax2, ax3 = axes

    ax1.fill_between(rp, rp * (ds_baseline_mean - ds_baseline_std),
                     rp * (ds_baseline_mean + ds_baseline_std),
                     color="k", alpha=0.12)
    ax1.fill_between(rp, rp * (ds_direct_mean - ds_direct_std),
                     rp * (ds_direct_mean + ds_direct_std),
                     color="tomato", alpha=0.18)
    ax1.fill_between(rp, rp * (ds_variant_mean - ds_variant_std),
                     rp * (ds_variant_mean + ds_variant_std),
                     color="forestgreen", alpha=0.18)
    ax1.fill_between(rp, rp * (ds_mass_mean - ds_mass_std),
                     rp * (ds_mass_mean + ds_mass_std),
                     color="steelblue", alpha=0.18)

    ax1.semilogx(rp, rp * ds_baseline_mean, "k-",   lw=2, ms=5,
                 label=f"Baseline (no AB, {n_base} real.)")
    ax1.semilogx(rp, rp * ds_direct_mean,   "r--s", lw=2, ms=5,
                 label=rf"AB direct ($B_{{cen}}={B_CENT},\,B_{{sat}}={B_SAT}$, {N_REAL} real.)")
    ax1.semilogx(rp, rp * ds_variant_mean,  "g-^",  lw=2, ms=5,
                 label=rf"AB variant ($B_{{cen}}={B_CENT},\,B_{{sat}}={B_SAT}$, {N_REAL} real.)")
    ax1.semilogx(rp, rp * ds_mass_mean,     "b-D",  lw=2, ms=5,
                 label=rf"AB mass ($B_{{cen}}={B_CENT},\,B_{{sat}}={B_SAT}$, {N_REAL} real.)")

    ax1.set_ylabel(r"$r_p \cdot \Delta\Sigma\ [h\,M_\odot/\mathrm{pc}^2 \cdot \mathrm{Mpc}/h]$")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(rf"Assembly Bias: direct vs variant vs mass")
    ax1.set_xticklabels([])

    # ΔΣ ratio panel
    ax2.semilogx(rp, ratio_direct,  "r--s", ms=4, label="AB direct / baseline")
    ax2.semilogx(rp, ratio_variant, "g-^",  ms=4, label="AB variant / baseline")
    ax2.semilogx(rp, ratio_mass,    "b-D",  ms=4, label="AB mass / baseline")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.axhspan(0.95, 1.05, color="green", alpha=0.10)
    ax2.set_ylabel(r"$\Delta\Sigma$ ratio")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xticklabels([])

    # ngal panel
    jitter = 0.08
    rng = np.random.default_rng(0)
    ax3.scatter(np.zeros(n_base)   + rng.uniform(-jitter, jitter, n_base),
                ngal_baseline * 1e4, color="k",         alpha=0.7, s=25, zorder=3)
    ax3.scatter(np.ones(N_REAL)    + rng.uniform(-jitter, jitter, N_REAL),
                ngal_direct   * 1e4, color="tomato",    alpha=0.7, s=25, zorder=3)
    ax3.scatter(np.full(N_REAL, 2) + rng.uniform(-jitter, jitter, N_REAL),
                ngal_variant  * 1e4, color="forestgreen", alpha=0.7, s=25, zorder=3)
    ax3.scatter(np.full(N_REAL, 3) + rng.uniform(-jitter, jitter, N_REAL),
                ngal_mass     * 1e4, color="steelblue", alpha=0.7, s=25, zorder=3)
    for x, ng, col in [(0, ngal_baseline, "k"), (1, ngal_direct, "tomato"),
                        (2, ngal_variant, "forestgreen"), (3, ngal_mass, "steelblue")]:
        ax3.hlines(ng.mean() * 1e4, x - 0.2, x + 0.2, colors=col, lw=2)

    ax3.axhline(TARGET_NGAL * 1e4, color="gray", ls="--", lw=0.8, label="Target")
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["Baseline", "AB direct", "AB variant", "AB mass"])
    ax3.set_ylabel(r"$n_\mathrm{gal}\ [\times 10^{-4}\ (h/\mathrm{Mpc})^3]$")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {PLOT_OUTPUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
