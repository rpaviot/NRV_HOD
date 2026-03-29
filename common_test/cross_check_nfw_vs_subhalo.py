"""
Cross-check: NFW satellite placement vs subhalo satellite placement.

Populates the same halo catalogue with two satellite models at identical HOD
parameters and lensing method, then compares the resulting DeltaSigma.

  - Method A (NFW):     satellites sampled from analytic NFW profiles
                        + precomputed halo-centre lensing cache (optimized)
  - Method B (subhalo): satellites placed at real N-body subhalo positions
                        (from precompute_subhalo_catalogue.py output)
                        + same lensing method as Method A

Both arms run with the same seeds, halo catalogue, and HOD parameters so
that only the satellite placement differs.

Usage
-----
    python cross_check_nfw_vs_subhalo.py \\
        --halo_path  /data/flamingo/catalogues/host_catalogue.parquet \\
        --subhalo_path /data/flamingo/catalogues/subhalo_catalogue.npz \\
        [--cache_path halo_center_lensing_cache.h5] \\
        [--n_realizations 5] \\
        [--output_dir .]

Outputs
-------
    cross_check_nfw_vs_subhalo.png  — 2-panel comparison plot
    cross_check_nfw_vs_subhalo.txt  — per-bin table
"""

import argparse
import os
import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default data paths and cosmology (mirror run_emulator_grid.py)
# ---------------------------------------------------------------------------

_DEFAULT_HALO_PATH = (
    "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
)
_DEFAULT_PARTICLE_PATH = (
    "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
)

LBOX            = 681.0          # Mpc/h
ZEFF            = 1.0
MASS_DEFINITION = "MassDef200m"
TARGET_NGAL     = 2e-4           # (Mpc/h)^-3

COSMO_PARAMS = {
    "h":    0.681,
    "Omc":  0.306 - 0.0486 - 1.39e-3,
    "Omb":  0.0486,
    "A_s":  2.099e-9,
    "n_s":  0.967,
    "Omnu": 1.39e-3,
}

COLUMN_MAPPING = {
    "x":      "x",    "y":    "y",    "z":    "z",
    "vx":     "vx",   "vy":   "vy",   "vz":   "vz",
    "mass":   "mass", "radius": "rvir", "c":  "c",
    "vrms":   "vrms",
}

RP_BINS = np.geomspace(0.1, 50.0, 16)   # 15 bins, Mpc/h


def fmt_time(seconds):
    """Pretty-print a duration."""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds/60:.1f} min"

# Base HOD parameters — same as benchmark_optimized_vs_baseline.py.
# Ac and As are rescaled to TARGET_NGAL via rescale_Ac_to_target_ngal()
# after the HOD model is loaded, so only the shape parameters are set here.
BASE_HOD_PARAMS = {
    "As":    0.3,
    "Mmin":  12.7,
    "sig_M": 0.3,
    "M1":    13.0,
    "gamma": 5.0,
    "alpha": 1.10,
    "kappa": 0.80,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare NFW vs subhalo satellite placement for DeltaSigma"
    )
    p.add_argument(
        "--halo_path", default=_DEFAULT_HALO_PATH,
        help="Halo catalogue parquet — used by BOTH methods. Should be the "
             "host_catalogue.parquet produced by precompute_subhalo_catalogue.py "
             "so that the subhalo CSR indices align. "
             f"Default: {_DEFAULT_HALO_PATH}",
    )
    p.add_argument(
        "--particle_path", default=_DEFAULT_PARTICLE_PATH,
        help=f"Particle catalogue parquet. Default: {_DEFAULT_PARTICLE_PATH}",
    )
    p.add_argument(
        "--subhalo_path", required=True,
        help="Path to subhalo_catalogue.npz from precompute_subhalo_catalogue.py",
    )
    p.add_argument(
        "--cache_path", default=None,
        help="Path to halo_center_lensing_cache.h5 (from precompute_halo_center_cache.py). "
             "Enables method='optimized' for both arms (recommended for speed). "
             "If omitted, both arms use method='standard' (pycorr pair counting).",
    )
    p.add_argument(
        "--n_realizations", type=int, default=5,
        help="HOD realizations to average per method (default: 5)",
    )
    p.add_argument(
        "--galaxy_fraction", type=float, default=0.10,
        help="Galaxy subsampling fraction per realization (default: 0.10)",
    )
    p.add_argument(
        "--particle_fraction", type=float, default=0.05,
        help="Particle subsampling fraction (default: 0.05)",
    )
    p.add_argument(
        "--particle_seed", type=int, default=42,
        help="Seed for particle subsampling (default: 42)",
    )
    p.add_argument(
        "--base_seed", type=int, default=1000,
        help="Base seed for HOD realizations (default: 1000)",
    )
    p.add_argument(
        "--output_dir", default=".",
        help="Output directory for plot and table (default: current dir)",
    )
    p.add_argument(
        "--baseline_cache",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_dsigma_cache.npz"),
        help="Path to baseline_dsigma_cache.npz (from benchmark_optimized_vs_baseline.py). "
             "Overlaid on the plot as a sanity-check for the NFW arm. "
             "Pass empty string to disable.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from HOD_NRV.HOD_numerical.HOD import HaloOccupation
    from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
        HaloCenterLensingCache,
    )
    from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

    os.makedirs(args.output_dir, exist_ok=True)

    lensing_method = "optimized" if args.cache_path else "standard"
    print(f"Lensing method : {lensing_method}")
    print(f"N realizations : {args.n_realizations}")
    print(f"Galaxy fraction: {args.galaxy_fraction}")
    print(f"Particle frac  : {args.particle_fraction}")
    print()

    # -------------------------------------------------------------------
    # Load particle catalogue once — shared between both arms
    # -------------------------------------------------------------------
    print(f"Loading particles from {args.particle_path} ...")
    t0 = time.time()
    df_part = pd.read_parquet(args.particle_path)
    print(f"  {len(df_part):,} particles loaded in {time.time()-t0:.1f}s")
    print()

    _common_kwargs = dict(
        cosmology=COSMO_PARAMS,
        zeff=ZEFF,
        Lbox=LBOX,
        column_mapping=COLUMN_MAPPING,
        mass_definition=MASS_DEFINITION,
        halo_path=args.halo_path,
        DataFrame_part=df_part,
        apply_rsd=False,     # lensing uses real-space positions
        do_test=False,
        particle_fraction=args.particle_fraction,
        particle_subsample_seed=args.particle_seed,
        population_backend="numba",
    )

    # -------------------------------------------------------------------
    # Method A: NFW satellites
    # -------------------------------------------------------------------
    print("Building Method A (NFW) ...")
    t0 = time.time()
    halo_nfw = HaloOccupation(subhalo_path=None, **_common_kwargs)
    halo_nfw.set_halo_model("ELG_mHMQ")
    print(f"  {halo_nfw.n_halos:,} halos loaded in {time.time()-t0:.1f}s")
    print()

    # -------------------------------------------------------------------
    # Method B: subhalo satellites
    # -------------------------------------------------------------------
    print(f"Building Method B (subhalo) from {args.subhalo_path} ...")
    t0 = time.time()
    halo_sub = HaloOccupation(subhalo_path=args.subhalo_path, **_common_kwargs)
    halo_sub.set_halo_model("ELG_mHMQ")
    print(f"  {halo_sub.n_halos:,} halos, "
          f"{len(halo_sub.sub_positions):,} subhalos loaded in {time.time()-t0:.1f}s")
    print()

    # -------------------------------------------------------------------
    # Rescale Ac/As to match target n_gal (same logic as benchmark)
    # -------------------------------------------------------------------
    print(f"Rescaling Ac/As to target n_gal = {TARGET_NGAL:.2e} (Mpc/h)^-3 ...")
    Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
        halo_nfw.HOD, BASE_HOD_PARAMS, target_ngal=TARGET_NGAL
    )
    hod_params = BASE_HOD_PARAMS.copy()
    hod_params["Ac"] = float(Ac_rescaled)
    hod_params["As"] = float(As_rescaled)
    print(f"  Ac = {hod_params['Ac']:.6f}")
    print(f"  As = {hod_params['As']:.6f}")
    print()

    # -------------------------------------------------------------------
    # Load cache (shared between both arms if optimized)
    # -------------------------------------------------------------------
    cache = None
    if args.cache_path:
        if not os.path.exists(args.cache_path):
            raise FileNotFoundError(
                f"Cache not found: {args.cache_path}\n"
                "Run common_test/precompute_halo_center_cache.py first."
            )
        print(f"Loading halo-centre lensing cache from {args.cache_path} ...")
        cache = HaloCenterLensingCache.load(args.cache_path)
        print(f"  {len(cache.deltasigma):,} precomputed profiles")
        print()

    # -------------------------------------------------------------------
    # Load baseline cache (optional — for NFW sanity overlay)
    # -------------------------------------------------------------------
    ds_baseline_mean = ds_baseline_std = None
    if args.baseline_cache and os.path.exists(args.baseline_cache):
        bl = np.load(args.baseline_cache)
        ds_baseline_mean = bl["ds_all"].mean(axis=0)
        ds_baseline_std  = bl["ds_all"].std(axis=0)
        print(f"Baseline cache loaded: {args.baseline_cache}  "
              f"({bl['ds_all'].shape[0]} realizations)")
    else:
        print("No baseline cache found — skipping baseline overlay.")
    print()

    print("HOD parameters:")
    for k, v in hod_params.items():
        print(f"  {k:12s} = {v}")
    print()

    # -------------------------------------------------------------------
    # Burn-in: one realization each to trigger JIT / Numba compilation
    # before the timed runs
    # -------------------------------------------------------------------
    print("=" * 60)
    print("Burn-in (1 realization each — discarded) ...")
    halo_nfw.compute_avg_lensing(
        hod_params,
        n_realizations=1,
        bins1=RP_BINS,
        method=lensing_method,
        precomputed_cache=cache,
        galaxy_fraction=args.galaxy_fraction,
        base_seed=args.base_seed,
    )
    halo_sub.compute_avg_lensing(
        hod_params,
        n_realizations=1,
        bins1=RP_BINS,
        method=lensing_method,
        precomputed_cache=cache,
        galaxy_fraction=args.galaxy_fraction,
        base_seed=args.base_seed,
    )
    print("  Burn-in done.\n")

    # -------------------------------------------------------------------
    # Run Method A
    # -------------------------------------------------------------------
    print("=" * 60)
    print("Running Method A — NFW satellite profiles ...")
    t0 = time.perf_counter()
    rp, ds_nfw_mean, ds_nfw_std = halo_nfw.compute_avg_lensing(
        hod_params,
        n_realizations=args.n_realizations,
        bins1=RP_BINS,
        method=lensing_method,
        precomputed_cache=cache,
        galaxy_fraction=args.galaxy_fraction,
        base_seed=args.base_seed,
    )
    t_nfw = time.perf_counter() - t0
    print(f"  Done in {fmt_time(t_nfw)}  "
          f"({fmt_time(t_nfw / args.n_realizations)}/realization,  "
          f"f_sat ~ {halo_nfw.satellite_fraction:.3f})")

    # -------------------------------------------------------------------
    # Run Method B
    # -------------------------------------------------------------------
    print("Running Method B — subhalo satellite placement ...")
    t0 = time.perf_counter()
    rp, ds_sub_mean, ds_sub_std = halo_sub.compute_avg_lensing(
        hod_params,
        n_realizations=args.n_realizations,
        bins1=RP_BINS,
        method=lensing_method,
        precomputed_cache=cache,
        galaxy_fraction=args.galaxy_fraction,
        base_seed=args.base_seed,
    )
    t_sub = time.perf_counter() - t0
    print(f"  Done in {fmt_time(t_sub)}  "
          f"({fmt_time(t_sub / args.n_realizations)}/realization,  "
          f"f_sat ~ {halo_sub.satellite_fraction:.3f})")
    print()

    # -------------------------------------------------------------------
    # Galaxy count / satellite fraction consistency check
    # (last realization — same seed for both, so counts must match)
    # -------------------------------------------------------------------
    n_gal_nfw = len(halo_nfw.positions_gal)
    n_gal_sub = len(halo_sub.positions_gal)
    f_sat_nfw = halo_nfw.satellite_fraction
    f_sat_sub = halo_sub.satellite_fraction
    counts_match = (n_gal_nfw == n_gal_sub)

    print("Galaxy count / f_sat (last realization):")
    print(f"  NFW    : N_gal = {n_gal_nfw:,}   f_sat = {f_sat_nfw:.4f}")
    print(f"  Subhalo: N_gal = {n_gal_sub:,}   f_sat = {f_sat_sub:.4f}")
    print(f"  Counts match: {counts_match}")
    if not counts_match:
        print("  WARNING: N_gal differs — check that both arms use the same seed "
              "and halo catalogue.")
    print()

    # -------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------
    frac_diff = (ds_sub_mean - ds_nfw_mean) / ds_nfw_mean   # (subhalo - NFW) / NFW

    print("=" * 60)
    print(f"{'rp [Mpc/h]':>12}  {'DS_NFW':>12}  {'DS_sub':>12}  {'frac diff':>12}")
    print("-" * 54)
    for i in range(len(rp)):
        print(f"{rp[i]:12.3f}  {ds_nfw_mean[i]:12.4f}  "
              f"{ds_sub_mean[i]:12.4f}  {100*frac_diff[i]:+11.2f}%")
    print("-" * 54)
    print(f"Max |frac diff| : {100*np.nanmax(np.abs(frac_diff)):.2f}%")
    print(f"Mean frac diff  : {100*np.nanmean(frac_diff):.2f}%")
    print()

    # -------------------------------------------------------------------
    # Save table
    # -------------------------------------------------------------------
    table_path = os.path.join(args.output_dir, "cross_check_nfw_vs_subhalo.txt")
    with open(table_path, "w") as f:
        f.write("# NFW vs subhalo satellite cross-check\n")
        f.write(f"# lensing_method = {lensing_method}\n")
        f.write(f"# n_realizations = {args.n_realizations}\n")
        f.write(f"# galaxy_fraction = {args.galaxy_fraction}\n")
        f.write(f"# subhalo_path = {args.subhalo_path}\n")
        f.write(f"# halo_path = {args.halo_path}\n")
        f.write(f"#\n")
        f.write(f"# {'rp':>10}  {'DS_NFW':>12}  {'DS_NFW_std':>12}"
                f"  {'DS_sub':>12}  {'DS_sub_std':>12}  {'frac_diff':>12}\n")
        for i in range(len(rp)):
            f.write(
                f"  {rp[i]:10.4f}  {ds_nfw_mean[i]:12.6f}  {ds_nfw_std[i]:12.6f}"
                f"  {ds_sub_mean[i]:12.6f}  {ds_sub_std[i]:12.6f}"
                f"  {frac_diff[i]:+12.6f}\n"
            )
    print(f"Table saved: {table_path}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    _make_plot(
        rp, ds_nfw_mean, ds_nfw_std, ds_sub_mean, ds_sub_std, frac_diff,
        lensing_method, args.n_realizations, args.output_dir,
        ds_baseline=ds_baseline_mean, ds_baseline_std=ds_baseline_std,
    )


def _make_plot(rp, ds_nfw, ds_nfw_std, ds_sub, ds_sub_std, frac_diff,
               lensing_method, n_real, output_dir,
               ds_baseline=None, ds_baseline_std=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    # --- top panel: DeltaSigma ---
    ax1.errorbar(
        rp, ds_nfw, yerr=ds_nfw_std,
        fmt="o-", color="steelblue", label=f"NFW profiles ({lensing_method})",
        capsize=3, lw=1.5, ms=5,
    )
    ax1.errorbar(
        rp, ds_sub, yerr=ds_sub_std,
        fmt="s--", color="tomato", label="Subhalo placement",
        capsize=3, lw=1.5, ms=5,
    )
    if ds_baseline is not None:
        ax1.errorbar(
            rp, ds_baseline, yerr=ds_baseline_std,
            fmt="^:", color="gray", label="Baseline (full-res NFW)",
            capsize=3, lw=1.2, ms=5, alpha=0.7,
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\Delta\Sigma\ [M_\odot\,h\,\mathrm{pc}^{-2}]$", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title(
        f"NFW vs subhalo satellite placement  "
        f"(N_real={n_real})",
        fontsize=11,
    )

    # --- bottom panel: fractional difference ---
    ax2.axhline(0, color="k", lw=1.0)
    ax2.axhline(+0.05, color="k", lw=0.7, ls=":", alpha=0.5, label="±5%")
    ax2.axhline(-0.05, color="k", lw=0.7, ls=":", alpha=0.5)
    ax2.plot(rp, frac_diff, "D-", color="purple", lw=1.5, ms=5,
             label="(subhalo − NFW) / NFW")
    ax2.fill_between(rp, frac_diff, alpha=0.15, color="purple")
    if ds_baseline is not None:
        frac_base = (ds_baseline - ds_nfw) / ds_nfw
        ax2.plot(rp, frac_base, "^:", color="gray", lw=1.2, ms=5,
                 alpha=0.8, label="(baseline − NFW) / NFW")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r_p\ [\mathrm{Mpc}/h]$", fontsize=12)
    ax2.set_ylabel("Fractional diff.", fontsize=11)
    ax2.set_ylim(-0.6, 0.6)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    plot_path = os.path.join(output_dir, "cross_check_nfw_vs_subhalo.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot  saved: {plot_path}")


if __name__ == "__main__":
    main()
