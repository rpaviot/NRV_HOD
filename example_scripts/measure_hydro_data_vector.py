#!/usr/bin/env python3
"""
measure_hydro_data_vector.py

Direct in-box measurement of the hydro Flamingo data vector, to reproduce (and
cross-check) the reference `data_bin0_finalbinning.npz` that was produced on
another cluster, and to *produce the wgg data vector* on the same binning for
the joint fit.

  - DeltaSigma(rp): NISP galaxies x hydro matter particles, real-space 3D xi_gm
    (pycorr) -> Sigma / DeltaSigma projection. Hydro particles are multi-species
    (DM/gas/star) with very different per-particle masses, so the cross-count is
    MASS-WEIGHTED (weights2 = particle mass) to trace the true matter field.
    Particles are randomly subsampled for tractability (233M is too many for a
    direct pair count to r~100); DeltaSigma is per-galaxy so subsampling only
    adds shot noise (~<0.5% at a few percent, per the convergence benchmark).
  - wgg(rp): NISP galaxy auto projected clustering (rp-pi, integrated to pi_max),
    galaxies shifted to redshift space along the LOS.

Both use the reference binning (rp_bins from the npz). Results are compared to
the reference DeltaSigma and wgg.

`--hod` switches to a third, independent measurement: the *truth* HOD of the
same NISP sample, <N_cen>(M) and <N_sat>(M). The NISP catalogue carries the
host M200m (Msun/h) and a central/satellite `type` flag, so the numerator is a
direct histogram; the denominator is the host halo mass function, read from the
hydro SOAP file (SOAP/HostHaloIndex == -1 selects hosts, SO/200_mean/TotalMass
gives M200m). This is what the posterior HOD profiles of the chains should be
compared against -- but note the two normalisation caveats printed by the run.

`--hod_env` splits that truth HOD by halo environment quantile at fixed mass,
a direct measurement of the assembly bias the chains parametrise with
B_cent/B_sat.

`--truth_ds` predicts DeltaSigma from the *measured per-halo occupation* --
the actual galaxy counts of each host, taken from the galaxy->host link --
instead of from an HOD. That removes halo mass as the only channel by which
galaxies choose halos, and so separates the two ways the forward model can be
short of the data: a wrong number of galaxies per unit mass (which an HOD can
fix) versus galaxies preferring halos an HOD cannot distinguish. The mass-only
control -- the same counts randomly permuted among halos of the same mass --
is what an HOD can reach at best, so the gap between the two IS the
assembly-bias contribution to the lensing amplitude.

`--profile` measures the satellite RADIAL profile of the same sample around
its true hosts and fits the NFW+exponential family (f_exp, tau, lambda_NFW)
that the chains use for satellite positioning, then propagates the residual
mis-specification into DeltaSigma_sat. It needs the exact SOAP host link, so
it reads the catalogue rebuilt by `precompute_subhalo_catalogue.py --nisp`
(which carries r_host/rvir_host/c_host), not NISP_catalogue_flamingo.parquet.

Usage (cluster):
    python example_scripts/measure_hydro_data_vector.py \
        --particle_fraction 0.02 --pi_max 100 \
        --output /sps/euclid/Users/rpaviot/flamingo/hydro_measured_data_vector.npz

    python example_scripts/precompute_subhalo_catalogue.py --nisp \
        --soap_path /sps/euclid/Users/rpaviot/flamingo/snapshots_hydro/halo_properties_0058.hdf5 \
        --output_dir /sps/euclid/Users/rpaviot/flamingo/snapshots_hydro
    python example_scripts/measure_hydro_data_vector.py --profile \
        --nisp_path /sps/euclid/Users/rpaviot/flamingo/snapshots_hydro/NISP_catalogue_rebuilt.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from HOD_NRV.HOD_analytical.pycosmo import Cosmology
from HOD_NRV.HOD_numerical.HOD_models import (
    LRG_Zheng07, ELG_GHOD, ELG_SFR, ELG_mHMQ, HOD_satellite,
    ELG_satellite_cutoff)
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
    compute_galaxy_lensing, compute_galaxy_clustering)
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
    HaloCenterLensingCache, TabulatedDeltaSigma)
from HOD_NRV.utilsf.numerical_sampler import TabulatedFitter, FitCase
from example_scripts.run_tabulated_chains import (
    COSMO_PARAMS, ZEFF, LBOX, MASS_DEFINITION, AC_FIDUCIAL, TARGET_NGAL,
    build_halo_occupation, build_param_config, _make_rescale_occupation,
    _print_per_bin)

HYDRO_DIR = "/sps/euclid/Users/rpaviot/flamingo/snapshots_hydro"
NISP_PATH = os.path.join(HYDRO_DIR, "NISP_catalogue_flamingo.parquet")
PART_PATH = os.path.join(HYDRO_DIR,
                         "hydro_flamingo_0058_downsampled_0.1percent.parquet")
REF_PATH = "/sps/euclid/Users/rpaviot/flamingo/data_bin0_finalbinning.npz"
SOAP_PATH = os.path.join(HYDRO_DIR, "halo_properties_0058.hdf5")
DMO_HOST_PATH = ("/sps/euclid/Users/rpaviot/flamingo/snapshots_DMO/"
                 "host_catalogue_ab.parquet")
HOST_MASS_CACHE = "/sps/euclid/Users/rpaviot/flamingo/hydro_host_mass.npz"
HYDRO_HOST_PATH = os.path.join(HYDRO_DIR, "host_catalogue_ab.parquet")
LINK_CACHE = "/sps/euclid/Users/rpaviot/flamingo/nisp_host_link.npz"
MASS_THRESHOLD = 1e11          # same host cut as the DMO catalogue the fits use


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nisp_path", default=NISP_PATH)
    p.add_argument("--part_path", default=PART_PATH)
    p.add_argument("--ref_path", default=REF_PATH)
    p.add_argument("--output", default=os.path.join(
        os.path.dirname(REF_PATH), "hydro_measured_data_vector.npz"))
    p.add_argument("--particle_fraction", type=float, default=0.02,
                   help="Random subsample fraction of the 233M hydro particles.")
    p.add_argument("--particle_seed", type=int, default=42)
    p.add_argument("--rsd_axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--pi_max", type=float, default=100.0,
                   help="wgg LOS integration half-length [Mpc/h].")
    p.add_argument("--chi_max", type=float, default=100.0,
                   help="DeltaSigma LOS projection half-length [Mpc/h] "
                        "(100 matches the tabulated forward model the fits use).")
    p.add_argument("--rp_min", type=float, default=None,
                   help="Override the reference rp binning (with --rp_max/--n_rp). "
                        "Use 0.1/50/26 for the tabulation grid.")
    p.add_argument("--rp_max", type=float, default=None)
    p.add_argument("--n_rp", type=int, default=None, help="Number of bin EDGES.")
    p.add_argument("--gal_type", type=int, default=None,
                   help="Optional NISP 'type' selection (default: all galaxies).")
    p.add_argument("--no_mass_weight", action="store_true",
                   help="Count particles unweighted (diagnostic).")
    p.add_argument("--hod", action="store_true",
                   help="Measure the truth HOD of the NISP sample instead of "
                        "the (DeltaSigma, wgg) data vector.")
    p.add_argument("--soap_path", default=SOAP_PATH,
                   help="Hydro SOAP file, for the host halo mass function.")
    p.add_argument("--host_mass_cache", default=HOST_MASS_CACHE,
                   help="npz cache of the hydro host logM (written on first "
                        "run; the SOAP pass costs ~30s).")
    p.add_argument("--dmo_host_path", default=DMO_HOST_PATH,
                   help="DMO host catalogue -- the halo population the "
                        "forward model actually lives on.")
    p.add_argument("--chain_bestfits", default=None,
                   help="Optional bestfits_*.npz from run_tabulated_chains, "
                        "overlaid (shape-matched) on the measured HOD.")
    p.add_argument("--dlogM", type=float, default=0.1,
                   help="HOD mass bin width [dex].")
    p.add_argument("--mass_column", default="",
                   help="Galaxy host-mass column for the --hod numerator. "
                        "Default: mass_200m if the catalogue has it, else "
                        "mass (which is M200c on the original NISP file, and "
                        "inconsistent with the M200m denominator).")
    p.add_argument("--hod_env", action="store_true",
                   help="Measure the truth HOD SPLIT BY HALO ENVIRONMENT: "
                        "<N_cen>(M | q) and <N_sat>(M | q) in quantiles of a "
                        "halo environment column, taken at fixed mass. A "
                        "direct measurement of the assembly bias that "
                        "B_cent/B_sat parametrise in the fits.")
    p.add_argument("--hydro_host_path", default=HYDRO_HOST_PATH,
                   help="Hydro host catalogue carrying the environment "
                        "columns (delta_h, delta_norm, qr2, fs_norm).")
    p.add_argument("--env_column", default="qr2",
                   choices=["qr2", "fs_norm", "delta_h", "delta_norm"],
                   help="Environment column to split on. qr2 is the raw tidal "
                        "shear q_R^2 at R = 2.25 Rvir; fs_norm is its "
                        "per-mass-bin rank-normalised form (what the chains "
                        "map as fE). The split is by quantile within each mass "
                        "bin either way, so raw vs normalised only changes the "
                        "mass binning used to rank.")
    p.add_argument("--n_env_bins", type=int, default=4,
                   help="Number of environment quantiles per mass bin.")
    p.add_argument("--link_cache", default=LINK_CACHE,
                   help="npz cache of the galaxy->host KD-tree link.")
    p.add_argument("--fit_hod", default=None, metavar="MEASURED_HOD_NPZ",
                   help="Fit the repo's own HOD forms straight to a measured "
                        "truth occupation (the npz written by --hod) and "
                        "report what the parametrisation can reach, with no "
                        "lensing in the loop.")
    p.add_argument("--fit_hod_min_host", type=int, default=500,
                   help="Minimum halos in a mass bin for it to enter the fit.")
    p.add_argument("--fit_hod_amp_scale", type=float, default=0.1,
                   help="Amplitude convention of the chain priors relative to "
                        "the measured occupation. The fits deliberately run "
                        "with Ac and As divided by 10 (target_ngal 2.3e-4 vs "
                        "the sample's 2.357e-3), and DeltaSigma/wgg depend "
                        "only on the Ac/As ratio, so the As prior (0.001,0.1) "
                        "means (0.01,1.0) in physical units. The 'chain' "
                        "configuration therefore fits scale*<N>. fsat and Meff "
                        "are invariant; ngal scales.")
    p.add_argument("--truth_ds", action="store_true",
                   help="Predict DeltaSigma from the MEASURED per-halo "
                        "occupation (the galaxy->host link) instead of from "
                        "an HOD, plus the mass-only control, and compare both "
                        "with the data. Isolates the assembly-bias "
                        "contribution to the large-scale lensing amplitude.")
    p.add_argument("--cache_path", default=os.path.join(
                       os.path.dirname(REF_PATH), "tabulated_cache_HYDRO.h5"),
                   help="Tabulated halo-center lensing cache, built on "
                        "--hydro_host_path.")
    p.add_argument("--data_path", default=REF_PATH,
                   help="Data vector the prediction is compared against.")
    p.add_argument("--ds_rp_min", type=float, default=0.2,
                   help="Scale cut for the --truth_ds comparison ('--rp_min' "
                        "is already taken by the measurement binning).")
    p.add_argument("--n_shuffle", type=int, default=5,
                   help="Realisations of the mass-only control (counts "
                        "permuted within a mass bin). 0 disables it.")
    p.add_argument("--shuffle_dlogM", type=float, default=0.02,
                   help="Mass-bin width the control permutes within. Narrow "
                        "enough that N(M) is preserved, wide enough that the "
                        "permutation has halos to work with.")
    p.add_argument("--shuffle_within", nargs="+", default=[],
                   metavar="COL",
                   help="Halo columns to hold FIXED alongside mass when "
                        "permuting the truth counts. Each one asks: if an HOD "
                        "could depend on mass AND this property, how much of "
                        "the measured assembly bias could it reach? The "
                        "answer decides whether a given AB column is worth "
                        "tabulating on at all.")
    p.add_argument("--n_prop_bins", type=int, default=5,
                   help="Quantiles of each --shuffle_within column, taken "
                        "WITHIN each mass bin.")
    p.add_argument("--truth_profile", type=float, nargs=3,
                   metavar=("F_EXP", "TAU", "LAMBDA_NFW"),
                   default=(0.0, 5.0, 1.0),
                   help="Satellite profile for the --truth_ds prediction. "
                        "Irrelevant above rp ~ 2 (identical to 4 significant "
                        "figures), which is where the deficit lives.")
    p.add_argument("--profile", action="store_true",
                   help="Measure the SATELLITE RADIAL PROFILE of the NISP "
                        "sample around its true hosts, and fit the "
                        "NFW+exponential family (f_exp, tau, lambda_NFW) the "
                        "chains use to it. Answers whether that family can "
                        "describe the truth at all.")
    p.add_argument("--profile_dlogM", type=float, default=0.5,
                   help="Host-mass bin width for the profile fit [dex].")
    p.add_argument("--profile_logM_min", type=float, default=11.5)
    p.add_argument("--profile_logM_max", type=float, default=14.5)
    p.add_argument("--profile_xmax", type=float, default=5.0,
                   help="Largest r/Rvir kept in the measurement.")
    p.add_argument("--rmax_fac", type=float, default=3.0,
                   help="Exponential truncation radius of the MODEL, in units "
                        "of Rvir. 3.0 is what NFW_jax and halo_center_lensing "
                        "hard-code; Rocher+23 apply no truncation, so 0 means "
                        "untruncated here.")
    return p.parse_args()


# ============================================================================
# Truth HOD of the NISP sample
# ============================================================================

def _hydro_host_logM(args):
    """Host M200m (log10, Msun/h) of every hydro halo above MASS_THRESHOLD.

    SOAP/HostHaloIndex == -1 flags a host (a satellite points at its host), and
    SO/200_mean/TotalMass is the M200m the NISP `mass` column was taken from.
    Only those two columns are read, so the 214 GB file costs ~30s.
    """
    if args.host_mass_cache and os.path.exists(args.host_mass_cache):
        print(f"host mass function from cache: {args.host_mass_cache}")
        return np.load(args.host_mass_cache)["logM"].astype(np.float64)

    import h5py
    print(f"reading host masses from SOAP: {args.soap_path}")
    with h5py.File(args.soap_path, "r") as f:
        is_host = f["SOAP/HostHaloIndex"][:] == -1
        mass = f["SO/200_mean/TotalMass"][:]
    cosmo_h = COSMO_PARAMS["h"]
    # SOAP TotalMass is in 1e10 Msun; the NISP/DMO catalogues are in Msun/h.
    mass = cosmo_h * np.asarray(mass, dtype=np.float64) * 1e10
    logM = np.log10(mass[is_host & (mass > MASS_THRESHOLD)])
    del is_host, mass
    if args.host_mass_cache:
        np.savez(args.host_mass_cache, logM=logM.astype(np.float32))
        print(f"  cached -> {args.host_mass_cache}")
    return logM


def measure_hod(args):
    """<N_cen>(M), <N_sat>(M) of the NISP sample, measured in the hydro box."""
    # The NISP catalogue on disk stores M200*c* in its `mass` column (the
    # builder read spherical_overdensity_200_crit), while the denominator below
    # is M200*m* from SOAP -- mixing them shifts <N>(M) horizontally by
    # ~0.03 dex and changes <N_cen> by a median factor 0.845. Prefer an
    # explicit M200m column when the catalogue carries one (the rebuild written
    # by precompute_subhalo_catalogue.py --nisp does), and say so loudly when
    # falling back.
    have = pq.ParquetFile(args.nisp_path).schema_arrow.names
    if args.mass_column:
        mass_col = args.mass_column
    elif "mass_200m" in have:
        mass_col = "mass_200m"
    else:
        mass_col = "mass"
    if mass_col not in have:
        raise SystemExit(f"{args.nisp_path} has no column {mass_col!r} "
                         f"(has: {have})")
    g = pd.read_parquet(args.nisp_path, columns=[mass_col, "type"])
    if args.gal_type is not None:
        g = g[g["type"] == args.gal_type]
    logM_gal = np.log10(g[mass_col].values)    # host mass [Msun/h]
    if mass_col == "mass":
        print("  *** WARNING: binning galaxies on the `mass` column, which is "
              "M200c in NISP_catalogue_flamingo.parquet, against an M200m "
              "denominator. Pass a catalogue with mass_200m (see "
              "precompute_subhalo_catalogue.py --nisp) or --mass_column. ***")
    print(f"galaxy host mass column: {mass_col}")
    is_sat = g["type"].values == 1
    ngal = len(g)
    print(f"galaxies: {ngal:,}  ngal = {ngal/LBOX**3:.3e} (Mpc/h)^-3  "
          f"fsat = {is_sat.mean():.4f}")

    logM_host = _hydro_host_logM(args)
    logM_dmo = np.log10(pd.read_parquet(
        args.dmo_host_path, columns=["mass"])["mass"].values)
    print(f"hosts: hydro {len(logM_host):,}  DMO {len(logM_dmo):,} "
          f"(>{MASS_THRESHOLD:.0e} Msun/h)")

    edges = np.arange(np.log10(MASS_THRESHOLD),
                      max(logM_host.max(), logM_dmo.max()) + args.dlogM,
                      args.dlogM)
    logM = 0.5 * (edges[1:] + edges[:-1])
    n_host, _ = np.histogram(logM_host, bins=edges)
    n_dmo, _ = np.histogram(logM_dmo, bins=edges)
    n_cen, _ = np.histogram(logM_gal[~is_sat], bins=edges)
    n_sat, _ = np.histogram(logM_gal[is_sat], bins=edges)

    def _occ(num, den):
        out = np.full(len(num), np.nan)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    ncen, nsat = _occ(n_cen, n_host), _occ(n_sat, n_host)
    ncen_dmo, nsat_dmo = _occ(n_cen, n_dmo), _occ(n_sat, n_dmo)

    # Meff / fsat by direct galaxy sum -- no mass-function integral involved.
    Meff = float(np.average(10.0**logM_gal[~is_sat]))
    Meff_all = float(np.average(10.0**logM_gal))
    fsat = float(is_sat.mean())

    print(f"\n{'logM':>6} {'N_host':>9} {'N_cen':>8} {'N_sat':>8} "
          f"{'<Ncen>':>9} {'<Nsat>':>9}")
    for i in np.where(n_host > 0)[0]:
        print(f"{logM[i]:6.2f} {n_host[i]:9d} {n_cen[i]:8d} {n_sat[i]:8d} "
              f"{ncen[i]:9.4f} {nsat[i]:9.4f}")
    print(f"\npeak <Ncen> = {np.nanmax(ncen):.4f} at "
          f"logM = {logM[np.nanargmax(ncen)]:.2f}")
    print(f"Meff(cen) = {Meff:.3e}   Meff(all) = {Meff_all:.3e}   "
          f"fsat = {fsat:.4f}")
    print(f"DMO denominator instead of hydro: peak <Ncen> = "
          f"{np.nanmax(ncen_dmo):.4f}  ({100*(np.nanmax(ncen_dmo)/np.nanmax(ncen)-1):+.1f}%)")

    print("\n--- normalisation caveats when comparing to a chain posterior ---")
    print("  1. The fits target ngal = 2.3e-4, ten times below this sample's "
          f"{ngal/LBOX**3:.2e}: Ac,As are divided by 10 by convention and "
          "DeltaSigma/wgg only see the Ac/As ratio. Compare shapes at matched "
          "ngal (or compare fsat and Meff), not absolute <N>.")
    print("  2. This is measured against the *hydro* M200m function; the model "
          "lives on the DMO one. That is the difference quoted just above.")
    print(f"  3. Numerator and denominator are both M200m here "
          f"(column {mass_col!r})."
          if mass_col != "mass" else
          "  3. MIXED mass definitions: numerator M200c, denominator M200m.")

    out = args.output
    np.savez(out, logM=logM, logM_edges=edges, n_host=n_host, n_dmo=n_dmo,
             n_cen=n_cen, n_sat=n_sat, ncen=ncen, nsat=nsat,
             ncen_dmo=ncen_dmo, nsat_dmo=nsat_dmo,
             ngal=ngal / LBOX**3, fsat=fsat, Meff_cen=Meff, Meff_all=Meff_all,
             mass_column=mass_col, nisp_path=args.nisp_path)
    print(f"\nSaved measured HOD -> {out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    M = 10.0**logM
    for ax, occ, occ_dmo, title in [
            (axes[0], ncen, ncen_dmo, r"$\langle N_{\rm cen}\rangle(M)$"),
            (axes[1], nsat, nsat_dmo, r"$\langle N_{\rm sat}\rangle(M)$")]:
        ax.plot(M, occ, "k.-", lw=1.8, label="hydro truth (hydro halo MF)")
        ax.plot(M, occ_dmo, color="tab:gray", ls="--", lw=1.2,
                label="same galaxies / DMO halo MF")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm 200m}\;[M_\odot/h]$")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3, which="both", ls=":")

    if args.chain_bestfits:
        d = np.load(args.chain_bestfits)
        logM_fit = d["logM_bins"]
        for key in [k for k in d.files if k.endswith("_ncen_med")]:
            pre = key[:-9]
            # The chain HOD is normalised to its own ngal; rescale it onto the
            # measured ngal so the *shapes* are on the same axes (caveat 1).
            scale = (ngal / LBOX**3) / float(d[f"{pre}_ngal"])
            axes[0].plot(10.0**logM_fit, scale * d[key], lw=1.5,
                         label=f"{pre} (x{scale:.1f})")
            axes[1].plot(10.0**logM_fit, scale * d[f"{pre}_nsat_med"], lw=1.5,
                         label=f"{pre} (x{scale:.1f})")
            print(f"  {pre}: fsat = {float(d[f'{pre}_fsat']):.4f} "
                  f"(truth {fsat:.4f})   Meff = {float(d[f'{pre}_Meff']):.3e} "
                  f"(truth {Meff:.3e})")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_ylim(1e-4, 20)
    fig.suptitle("NISP truth HOD in the Flamingo hydro box", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.splitext(out)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved HOD plot -> {plot_path}")


# ---------------------------------------------------------------------------
# Truth HOD split by halo environment (direct assembly-bias measurement)
# ---------------------------------------------------------------------------


def _link_galaxies_to_hosts(args, halo_pos, gal_pos):
    """Nearest host halo for every NISP galaxy, in the periodic box.

    The NISP catalogue carries a host M200m but no host index, and M200m is
    quantised in units of the particle mass (9.18M halos share only 5,559
    distinct values), so the host cannot be recovered by matching on mass.
    Position works instead: a central sits on its halo centre to ~1 kpc/h.
    """
    if args.link_cache and os.path.exists(args.link_cache):
        z = np.load(args.link_cache)
        if len(z["idx"]) == len(gal_pos):
            print(f"galaxy->host link from cache: {args.link_cache}")
            return z["idx"], z["dist"]
        print("  (cached link has the wrong length -- rebuilding)")
    print("linking galaxies to host halos (periodic KD-tree) ...")
    tree = cKDTree(halo_pos, boxsize=LBOX)
    dist, idx = tree.query(gal_pos, k=1, workers=-1)
    if args.link_cache:
        np.savez(args.link_cache, idx=idx.astype(np.int64),
                 dist=dist.astype(np.float32))
        print(f"  cached -> {args.link_cache}")
    return idx, dist


def predict_truth_deltasigma(args):
    """DeltaSigma at the occupation the box actually realised.

    Every forward-model prediction so far has gone through an HOD: <N>(M),
    optionally tilted by one assembly-bias parameter. That makes halo mass the
    only channel by which galaxies choose halos, and it is exactly the
    assumption under suspicion -- at the *measured* truth occupation the model
    is a flat 17% below the data for rp > 3 Mpc/h, with the cache itself
    cleared by cross_check_tabulated.py --direct (0.2%).

    Here the occupation is not modelled at all. `TabulatedDeltaSigma` is linear
    in the per-halo occupation, so handing it the measured counts -- for each
    host, how many NISP centrals and satellites actually live there -- gives
    the lensing of the true galaxy sample with nothing else changed: same
    halos, same cache, same projection.

    Three predictions are printed against the same data:

    truth     measured per-halo counts. Only the satellite radial profile is
              still modelled, and that is irrelevant above rp ~ 2.
    mass-only the same counts randomly permuted among halos in a narrow mass
              bin. N(M) is preserved exactly; every secondary dependence is
              destroyed. This is the ceiling of what *any* mass-only HOD can
              reach, so `truth / mass-only` is the assembly-bias contribution
              to the amplitude, measured rather than parametrised.
    (the HOD at the truth-fit parameters is what `run_tabulated_chains.py
     --predict` already gives, and is the 17%-low curve.)

    Reading: truth matches the data => the occupation-to-lensing step is sound
    and the deficit is entirely the mass-only restriction. truth is short too
    => the fault is upstream of the occupation, in the halo catalogue or the
    galaxy-halo link.
    """
    cols = ["x", "y", "z", "mass", "rvir"] + list(args.shuffle_within)
    halo_df = pd.read_parquet(args.hydro_host_path, columns=cols)
    gal = pd.read_parquet(args.nisp_path, columns=["x", "y", "z", "type"])
    print(f"hosts: {len(halo_df):,}   galaxies: {len(gal):,}")

    hp = np.ascontiguousarray(halo_df[["x", "y", "z"]].values,
                              dtype=np.float64) % LBOX
    gp = np.ascontiguousarray(gal[["x", "y", "z"]].values,
                              dtype=np.float64) % LBOX
    idx, dist = _link_galaxies_to_hosts(args, hp, gp)
    del hp, gp

    is_sat = gal["type"].values == 1
    rvir = halo_df["rvir"].values / 1e3
    inside = dist < rvir[idx]
    print(f"link quality: centrals median {np.median(dist[~is_sat]):.4f} Mpc/h, "
          f"satellites median {np.median(dist[is_sat]):.4f} Mpc/h; "
          f"{100 * inside.mean():.3f}% inside the matched Rvir")

    n_halo = len(halo_df)
    N_cen = np.bincount(idx[~is_sat], minlength=n_halo).astype(np.float64)
    N_sat = np.bincount(idx[is_sat], minlength=n_halo).astype(np.float64)
    multi = int((N_cen > 1).sum())
    if multi:
        print(f"  NOTE: {multi:,} halos matched more than one central "
              f"({N_cen.sum() - N_cen.clip(max=1).sum():.0f} extra) -- the "
              f"link is nearest-neighbour, not the SOAP index")
    print(f"true occupation: {N_cen.sum():,.0f} centrals + {N_sat.sum():,.0f} "
          f"satellites, fsat {N_sat.sum() / (N_cen.sum() + N_sat.sum()):.4f}, "
          f"{(N_cen + N_sat > 0).sum():,} occupied halos")

    # ---- forward model on exactly these halos ------------------------------
    cache = HaloCenterLensingCache.load(args.cache_path)
    if not cache.has_tabulation:
        raise SystemExit("Cache has no xi_gm tabulation.")
    halo = build_halo_occupation(FitCase.EXTENDED_PROFILE, args.hydro_host_path)

    # The counts are indexed by parquet row; the cache and the HaloOccupation
    # must be the same rows in the same order or every weight lands on the
    # wrong halo. Mass is the cheap witness.
    logM_h = np.log10(halo_df["mass"].values)
    if len(halo.logM) != n_halo:
        raise SystemExit(
            f"halo row mismatch: HaloOccupation has {len(halo.logM):,} rows, "
            f"the parquet {n_halo:,} -- the per-halo counts cannot be aligned "
            f"with the cache.")
    # Tolerance is set by float32: the catalogues store mass in single
    # precision and HaloOccupation keeps logM in whatever dtype the column
    # had, so a *correctly aligned* pair still differs by ~1e-5 dex. A
    # permutation of 9M rows spanning logM 11-15 would differ by O(1) dex, so
    # 1e-3 separates the two by three orders of magnitude either way.
    dmax = float(np.abs(np.asarray(halo.logM, dtype=np.float64) - logM_h).max())
    if dmax > 1e-3:
        raise SystemExit(
            f"halo ROW ORDER mismatch: max |logM difference| = {dmax:.3e} dex "
            f"between the parquet and HaloOccupation -- every per-halo count "
            f"would land on the wrong halo.")
    print(f"row alignment verified: max |dlogM| = {dmax:.2e} dex "
          f"(float32 round-trip; a permutation would give O(1))")
    tab = TabulatedDeltaSigma(cache, halo)
    f_exp, tau, lambda_NFW = args.truth_profile
    print(f"satellite profile: f_exp={f_exp} tau={tau} lambda_NFW={lambda_NFW}")

    fitter = TabulatedFitter(
        tabulated_ds=tab,
        occupation_rescale=_make_rescale_occupation(halo,
                                                    FitCase.EXTENDED_PROFILE),
        target_ngal=TARGET_NGAL, fit_case=FitCase.EXTENDED_PROFILE,
        data_path=args.data_path, rp_min=args.ds_rp_min, rp_max=None,
        param_config=build_param_config(FitCase.EXTENDED_PROFILE,
                                        assembly_bias=False),
        Ac_fiducial=AC_FIDUCIAL, ngal_anchor="mass_function",
    )
    print(f"data: {fitter.n_bins} bins in [{fitter.rp_obs[0]:.3f}, "
          f"{fitter.rp_obs[-1]:.2f}] Mpc/h from {args.data_path}")

    def _predict(nc, ns, label):
        rp_full, ds_full, info = tab.predict_occupation(
            nc, ns, f_exp=f_exp, tau=tau, lambda_NFW=lambda_NFW)
        sel = np.isin(np.round(rp_full, 8), np.round(fitter.rp_obs, 8))
        ds = np.asarray(ds_full)[sel]
        if len(ds) != fitter.n_bins:
            raise SystemExit(f"bin mismatch: model {len(ds)} vs data "
                             f"{fitter.n_bins}")
        resid = ds - fitter.ds_obs
        chi2 = float(resid @ fitter.cov_inv @ resid)
        print(f"\n  {label}: ngal = {info['ngal']:.4e}, "
              f"fsat = {info['fsat']:.4f}, chi2 = {chi2:.2f} "
              f"over {fitter.n_bins} bins (no free parameters)")
        return ds, chi2

    ds_truth, chi2_truth = _predict(N_cen, N_sat, "truth")
    _print_per_bin("truth occupation", fitter.rp_obs, fitter.ds_obs,
                   ds_truth, fitter.cov_inv)

    # ---- mass-only control -------------------------------------------------
    ds_shuf = None
    if args.n_shuffle > 0:
        edges = np.arange(logM_h.min(), logM_h.max() + args.shuffle_dlogM,
                          args.shuffle_dlogM)
        mbin = np.clip(np.digitize(logM_h, edges) - 1, 0, len(edges) - 2)
        order = np.argsort(mbin, kind="stable")
        starts = np.searchsorted(mbin[order], np.arange(len(edges) - 1), "left")
        stops = np.searchsorted(mbin[order], np.arange(len(edges) - 1), "right")
        groups = [order[a:b] for a, b in zip(starts, stops) if b - a > 1]
        print(f"\nmass-only control: {len(groups)} mass bins of width "
              f"{args.shuffle_dlogM} dex, {args.n_shuffle} realisations")

        ds_shuf = np.empty((args.n_shuffle, fitter.n_bins))
        chi2_shuf = np.empty(args.n_shuffle)
        for r in range(args.n_shuffle):
            rng = np.random.default_rng(1000 + r)
            # Centrals and satellites are permuted INDEPENDENTLY: that is the
            # HOD's own assumption (N_cen and N_sat drawn separately given M),
            # so the control is the best a mass-only HOD could do, conformity
            # included in what is being destroyed.
            nc, ns = N_cen.copy(), N_sat.copy()
            for g in groups:
                nc[g] = N_cen[rng.permutation(g)]
                ns[g] = N_sat[rng.permutation(g)]
            ds_shuf[r], chi2_shuf[r] = _predict(nc, ns, f"mass-only {r + 1}")
        ds_mean = ds_shuf.mean(axis=0)
        _print_per_bin("mass-only control (mean)", fitter.rp_obs,
                       fitter.ds_obs, ds_mean, fitter.cov_inv)
        print(f"  mass-only chi2 = {chi2_shuf.mean():.2f} +/- "
              f"{chi2_shuf.std():.2f}  (truth {chi2_truth:.2f})")

        # ---- conditional controls: mass AND one halo property held fixed --
        # The mass-only control is the floor (no secondary information) and
        # the truth is the ceiling (all of it). Permuting within (mass,
        # property) cells asks how much of the gap a property recovers, which
        # is exactly what an HOD that depends on that property could reach.
        ladder = {}
        for col in args.shuffle_within:
            prop = np.asarray(halo_df[col].values, dtype=np.float64)
            cells = []
            for g in groups:
                if len(g) < 10 * args.n_prop_bins:
                    cells.append(g)
                    continue
                q = np.quantile(prop[g],
                                np.linspace(0, 1, args.n_prop_bins + 1)[1:-1])
                b = np.searchsorted(q, prop[g], side="right")
                for k in range(args.n_prop_bins):
                    sub = g[b == k]
                    if len(sub) > 1:
                        cells.append(sub)
            ds_c = np.empty((args.n_shuffle, fitter.n_bins))
            for r in range(args.n_shuffle):
                rng = np.random.default_rng(2000 + r)
                nc, ns = N_cen.copy(), N_sat.copy()
                for c in cells:
                    nc[c] = N_cen[rng.permutation(c)]
                    ns[c] = N_sat[rng.permutation(c)]
                ds_c[r], _ = _predict(nc, ns, f"mass+{col} {r + 1}")
            ladder[col] = ds_c.mean(axis=0)
            print(f"  mass+{col}: {len(cells):,} cells")

        ratio = ds_truth / ds_mean
        big = fitter.rp_obs > 3.0
        print(f"\n  {'rp':>8} {'truth':>12} {'mass-only':>12} "
              f"{'truth/mass':>11} {'data/truth':>11}")
        for j in range(fitter.n_bins):
            print(f"  {fitter.rp_obs[j]:8.3f} {ds_truth[j]:12.4f} "
                  f"{ds_mean[j]:12.4f} {ratio[j]:11.4f} "
                  f"{fitter.ds_obs[j] / ds_truth[j]:11.4f}")
        print(f"\n  ASSEMBLY BIAS, rp > 3: truth / mass-only = "
              f"{ratio[big].mean():.4f} +/- {ratio[big].std():.4f}")
        print(f"  RESIDUAL,      rp > 3: data  / truth     = "
              f"{(fitter.ds_obs / ds_truth)[big].mean():.4f} +/- "
              f"{(fitter.ds_obs / ds_truth)[big].std():.4f}")
        print("  The first number is what a mass-only HOD cannot buy; the "
              "second is what is left over once the true occupation is used.")

        if ladder:
            gap = (ds_truth - ds_mean)[big]
            print(f"\n  --- how much of the gap each property recovers "
                  f"(rp > 3) ---")
            print(f"  {'property':>14} {'ratio to mass-only':>20} "
                  f"{'fraction of truth':>19}")
            for col, dsc in ladder.items():
                rec = (dsc - ds_mean)[big]
                frac = float(np.sum(rec) / np.sum(gap))
                rat = float(np.mean(dsc[big] / ds_mean[big]))
                print(f"  {col:>14} {rat:20.4f} {100 * frac:18.1f}%")
            print("  100% means an HOD in (mass, that property) could reach "
                  "the measured amplitude; ~0% means it is the wrong "
                  "variable and no B_cent/B_sat on it will help.")

    out = args.output
    np.savez(out, rp=fitter.rp_obs, ds_obs=fitter.ds_obs, ds_truth=ds_truth,
             ds_shuffled=ds_shuf if ds_shuf is not None else np.zeros(0),
             N_cen=N_cen.astype(np.int32), N_sat=N_sat.astype(np.int32),
             chi2_truth=chi2_truth, logM=logM_h.astype(np.float32),
             shuffle_dlogM=args.shuffle_dlogM,
             profile=np.asarray(args.truth_profile),
             **{f"ds_cond_{c}": v for c, v in
                (ladder.items() if args.n_shuffle > 0 else [])})
    print(f"\nSaved -> {out}")


def measure_hod_environment(args):
    """<N_cen>(M | env quantile), <N_sat>(M | env quantile) in the hydro box.

    Both numerator and denominator are keyed on the SAME halo row, so mass and
    environment come from one catalogue with one definition. That matters twice
    over: the environment split is only meaningful if the galaxies and the
    halos they normalise against are ranked by the same field, and it sidesteps
    the ~0.03 dex offset between the NISP catalogue's stored host mass and the
    SOAP M200m that `--hod` compares across.

    The quantiles are taken WITHIN each mass bin, so the denominator is flat by
    construction: every mass bin contributes the same number of halos to each
    environment quantile, and any asymmetry left in <N> is occupation, not
    a mass-function effect.
    """
    halo = pd.read_parquet(
        args.hydro_host_path,
        columns=["x", "y", "z", "mass", "rvir", args.env_column])
    gal = pd.read_parquet(args.nisp_path, columns=["x", "y", "z", "type"])
    print(f"hosts: {len(halo):,}   galaxies: {len(gal):,}")

    hp = np.ascontiguousarray(halo[["x", "y", "z"]].values, dtype=np.float64) % LBOX
    gp = np.ascontiguousarray(gal[["x", "y", "z"]].values, dtype=np.float64) % LBOX
    idx, dist = _link_galaxies_to_hosts(args, hp, gp)
    del hp, gp

    is_sat = gal["type"].values == 1
    rvir = halo["rvir"].values / 1e3      # the catalogue stores kpc/h
    inside = dist < rvir[idx]
    print(f"link quality: centrals median {np.median(dist[~is_sat]):.4f} Mpc/h, "
          f"satellites median {np.median(dist[is_sat]):.4f} Mpc/h; "
          f"{100*inside.mean():.3f}% inside the matched Rvir")
    if inside.mean() < 0.99:
        print("  WARNING: >1% of galaxies fall outside their matched halo's "
              "Rvir -- the link is not clean, treat the split with caution")

    logM_h = np.log10(halo["mass"].values)
    env = np.asarray(halo[args.env_column].values, dtype=np.float64)

    edges = np.arange(np.log10(MASS_THRESHOLD), logM_h.max() + args.dlogM,
                      args.dlogM)
    logM = 0.5 * (edges[1:] + edges[:-1])
    nM, nE = len(logM), args.n_env_bins
    mbin = np.digitize(logM_h, edges) - 1

    # Environment quantile within each mass bin.
    ebin = np.full(len(halo), -1, dtype=np.int16)
    order = np.argsort(mbin, kind="stable")
    starts = np.searchsorted(mbin[order], np.arange(nM), side="left")
    stops = np.searchsorted(mbin[order], np.arange(nM), side="right")
    for m in range(nM):
        sel = order[starts[m]:stops[m]]
        if len(sel) < 10 * nE:
            continue
        cuts = np.quantile(env[sel], np.linspace(0.0, 1.0, nE + 1)[1:-1])
        ebin[sel] = np.searchsorted(cuts, env[sel], side="right")

    ok = (mbin >= 0) & (mbin < nM) & (ebin >= 0)
    flat = np.where(ok, mbin * nE + np.maximum(ebin, 0), -1)

    def _counts(f):
        return np.bincount(f[f >= 0], minlength=nM * nE).reshape(nM, nE)

    flat_g = flat[idx]
    n_host = _counts(flat)
    n_cen = _counts(flat_g[~is_sat])
    n_sat = _counts(flat_g[is_sat])

    with np.errstate(divide="ignore", invalid="ignore"):
        ncen = np.where(n_host > 0, n_cen / n_host, np.nan)
        nsat = np.where(n_host > 0, n_sat / n_host, np.nan)
        ncen_err = np.where(n_host > 0, np.sqrt(n_cen) / n_host, np.nan)
        nsat_err = np.where(n_host > 0, np.sqrt(n_sat) / n_host, np.nan)

    # ---- residual-mass confounder -----------------------------------------
    # <N_cen> rises very steeply with mass here (dln<N>/dlnM ~ 6 below the
    # peak), so a tiny correlation between the environment and mass WITHIN a
    # bin fakes an occupation signal: at that slope, 0.01 dex of mass offset
    # between the top and bottom quantile already produces a 14% ratio. Measure
    # the offset and quote the ratio it would produce on its own.
    sum_logM = np.bincount(flat[flat >= 0], weights=logM_h[flat >= 0],
                           minlength=nM * nE).reshape(nM, nE)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_logM = np.where(n_host > 0, sum_logM / n_host, np.nan)
    dlogM_env = mean_logM[:, -1] - mean_logM[:, 0]

    # Local slope dln<N>/dlnM from the measured curve (mass-bin to mass-bin,
    # quantile-averaged), by centred differences in log10 M.
    with np.errstate(divide="ignore", invalid="ignore"):
        ncen_all = np.where(n_host.sum(1) > 0,
                            n_cen.sum(1) / n_host.sum(1), np.nan)
        nsat_all = np.where(n_host.sum(1) > 0,
                            n_sat.sum(1) / n_host.sum(1), np.nan)
    def _slope(y):
        ly = np.log10(np.where(y > 0, y, np.nan))
        return np.gradient(ly, logM)
    slope_cen, slope_sat = _slope(ncen_all), _slope(nsat_all)
    ratio_from_mass_cen = 10.0**(slope_cen * dlogM_env)
    ratio_from_mass_sat = 10.0**(slope_sat * dlogM_env)

    qname = {"qr2": "tidal shear q_R^2", "fs_norm": "normalised tidal shear",
             "delta_h": "overdensity delta", "delta_norm": "normalised delta"}
    print(f"\n=== <N>(M) split by {qname[args.env_column]} "
          f"({args.env_column}), {nE} quantiles at fixed mass ===")
    hdr = f"{'logM':>6} {'N_host/q':>9}"
    for e in range(nE):
        hdr += f" {'<Ncen>q%d' % (e + 1):>10}"
    hdr += (f" {'hi/lo':>7} {'nsig':>6} {'dlogM':>8} {'slope':>6}"
            f" {'massfake':>8} {'corr':>6}")
    print(hdr)
    for m in range(nM):
        if n_host[m].min() == 0 or n_cen[m].sum() < 20:
            continue
        line = f"{logM[m]:6.2f} {n_host[m].mean():9.0f}"
        for e in range(nE):
            line += f" {ncen[m, e]:10.5f}"
        lo, hi = ncen[m, 0], ncen[m, -1]
        elo, ehi = ncen_err[m, 0], ncen_err[m, -1]
        if lo > 0 and hi > 0:
            r = hi / lo
            sig = (hi - lo) / np.sqrt(ehi**2 + elo**2)
            line += (f" {r:7.3f} {sig:6.1f} {dlogM_env[m]:+8.4f}"
                     f" {slope_cen[m]:6.2f} {ratio_from_mass_cen[m]:8.3f}"
                     f" {r / ratio_from_mass_cen[m]:6.3f}")
        print(line)
    print("  dlogM    = <logM>(top quantile) - <logM>(bottom quantile)")
    print("  massfake = the hi/lo ratio that mass offset ALONE would produce")
    print("  corr     = hi/lo divided by it: the occupation signal that "
          "survives")

    # Mass-integrated contrast. The denominator is balanced by construction,
    # so summing the raw counts over mass already controls for the halo mass
    # function -- no reweighting needed.
    def _contrast(n):
        lo, hi = n[:, 0].sum(), n[:, -1].sum()
        r = hi / lo if lo > 0 else np.nan
        sig = (hi - lo) / np.sqrt(hi + lo) if (hi + lo) > 0 else np.nan
        return lo, hi, r, sig

    print(f"\n--- mass-integrated contrast, top vs bottom quantile "
          f"(equal halo counts per quantile at fixed mass) ---")
    for tag, n in (("centrals", n_cen), ("satellites", n_sat)):
        lo, hi, r, sig = _contrast(n)
        print(f"  {tag:<11} N(lowest q) = {lo:8d}   N(highest q) = {hi:8d}   "
              f"ratio = {r:.4f}   ({sig:+.1f} sigma)")
    print("  ratio = 1 means the occupation does not depend on this "
          "environment at fixed halo mass (no assembly bias in this variable).")

    # Mass-controlled version: divide out, bin by bin, the ratio that the
    # residual within-bin mass offset would produce on its own, then re-sum.
    for tag, n, occ, fake in (("centrals", n_cen, ncen, ratio_from_mass_cen),
                              ("satellites", n_sat, nsat, ratio_from_mass_sat)):
        good = np.isfinite(fake) & (n_host[:, 0] > 0) & (n[:, 0] > 0)
        lo = n[good, 0].astype(float)
        hi = n[good, -1] / fake[good]
        r = hi.sum() / lo.sum()
        sig = (hi.sum() - lo.sum()) / np.sqrt(hi.sum() + lo.sum())
        print(f"  {tag:<11} mass-offset-corrected ratio = {r:.4f} "
              f"({sig:+.1f} sigma)")
    print(f"  median |<logM> offset| between top and bottom quantile = "
          f"{np.nanmedian(np.abs(dlogM_env)):.4f} dex")

    print("\n--- caveats ---")
    print(f"  1. {args.env_column} was painted from a NUMBER-weighted mesh of "
          "the hydro particles (compute_assembly_bias.py takes positions "
          "only). Hydro particles are multi-species: DM is ~50% by number but "
          "~5x the mass of a gas particle, so gas is over-weighted ~3x "
          "relative to mass. The ranking is still a real environmental "
          "ranking; a mass-weighted field needs a weighted TSC (pysco's has "
          "no weights) and would be the check on this result.")
    print("  2. Satellites are assigned their host halo's environment, which "
          "is what the HOD means; their own position is inside Rvir either "
          "way (100% of the sample).")

    np.savez(args.output, logM=logM, logM_edges=edges,
             env_column=args.env_column, n_env_bins=nE,
             n_host=n_host, n_cen=n_cen, n_sat=n_sat,
             ncen=ncen, nsat=nsat, ncen_err=ncen_err, nsat_err=nsat_err,
             mean_logM=mean_logM, dlogM_env=dlogM_env,
             slope_cen=slope_cen, slope_sat=slope_sat,
             link_dist=dist.astype(np.float32))
    print(f"\nSaved environment-split HOD -> {args.output}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    M = 10.0**logM
    cmap = plt.get_cmap("viridis")
    for ax, occ, err, title in [
            (axes[0], ncen, ncen_err, r"$\langle N_{\rm cen}\rangle(M)$"),
            (axes[1], nsat, nsat_err, r"$\langle N_{\rm sat}\rangle(M)$")]:
        for e in range(nE):
            ax.errorbar(M, occ[:, e], yerr=err[:, e], lw=1.5,
                        color=cmap(e / max(nE - 1, 1)),
                        label=f"q{e+1} ({'low' if e == 0 else 'high' if e == nE-1 else 'mid'})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm 200m}\;[M_\odot/h]$"); ax.set_ylabel(title)
        ax.set_ylim(1e-4, 20)
        ax.grid(alpha=0.3, which="both", ls=":"); ax.legend(fontsize=8)
    ax = axes[2]
    for occ, err, tag, c in ((ncen, ncen_err, "centrals", "tab:blue"),
                             (nsat, nsat_err, "satellites", "tab:red")):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = occ[:, -1] / occ[:, 0]
            re = r * np.sqrt((err[:, -1] / occ[:, -1])**2
                             + (err[:, 0] / occ[:, 0])**2)
        ax.errorbar(M, r, yerr=re, lw=1.5, color=c, label=tag)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel(r"$M_{\rm 200m}\;[M_\odot/h]$")
    ax.set_ylabel(r"$\langle N\rangle$(highest q) / $\langle N\rangle$(lowest q)")
    ax.set_ylim(0, 3); ax.grid(alpha=0.3, which="both", ls=":"); ax.legend(fontsize=8)
    fig.suptitle(f"NISP truth HOD vs {qname[args.env_column]} "
                 f"at fixed mass (Flamingo hydro)", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


# ============================================================================
# Satellite radial profile of the NISP sample, and the NFW+exponential family
# ============================================================================
#
# The chains position satellites with the Rocher+23 (2306.06319) prescription:
# a fraction f_exp of them follow their Eq. 7.7, dN/dr = exp(-r/(tau*r_s)), and
# the rest an NFW with r_s -> r_s/lambda_NFW truncated at Rvir. Every fit so
# far -- DMO, baryonified, and the hydro cache where the matter field is exact
# -- reaches a good chi2 while recovering an occupation nothing like the
# measured truth, so the question is whether that three-parameter radial family
# can describe the real ELG satellite profile at all. If it cannot, the fit has
# to buy the missing 1-halo shape with occupation parameters instead, which is
# precisely the observed pathology.
#
# The measurement uses the SOAP host index (`r_host` from build_nisp_catalogue),
# not a nearest-halo match: satellites beyond Rvir are the part of the profile
# under test, and a nearest-halo link would reassign them to whichever small
# halo they sit next to and truncate the answer at ~Rvir by construction.


def _profile_pdf(r, Rvir, Rs, f_exp, tau, lam, rmax):
    """dN/dr of the sampler, per satellite (its own host's Rvir, Rs).

    Mirrors NFW_jax.extended_NFW_satellites_positions and
    halo_center_lensing.satellite_radial_nodes exactly: NFW component with
    Rs/lambda and c*lambda truncated at Rvir, exponential component
    exp(-r/(tau*Rs)) truncated at ``rmax``. All radii in Mpc/h.
    """
    Rs_n = Rs / lam
    c_n = Rvir / Rs_n
    x = r / Rs_n
    m_c = np.log1p(c_n) - c_n / (1.0 + c_n)
    p_nfw = np.where(r <= Rvir, x / (1.0 + x) ** 2 / (Rs_n * m_c), 0.0)

    s = tau * Rs
    p_exp = np.where(r <= rmax,
                     np.exp(-r / s) / (s * -np.expm1(-rmax / s)), 0.0)
    return (1.0 - f_exp) * p_nfw + f_exp * p_exp


def _profile_cdf(r, Rvir, Rs, f_exp, tau, lam, rmax):
    """CDF of :func:`_profile_pdf` (for the KS test and for sampling)."""
    Rs_n = Rs / lam
    c_n = Rvir / Rs_n
    x = np.minimum(r, Rvir) / Rs_n
    m_c = np.log1p(c_n) - c_n / (1.0 + c_n)
    F_nfw = (np.log1p(x) - x / (1.0 + x)) / m_c
    s = tau * Rs
    F_exp = np.expm1(-np.minimum(r, rmax) / s) / np.expm1(-rmax / s)
    return (1.0 - f_exp) * F_nfw + f_exp * F_exp


_P_LO = np.array([1e-4, 0.05, 0.02])          # f_exp, tau, lambda_NFW
_P_HI = np.array([1.0 - 1e-4, 500.0, 50.0])   # deliberately far outside the
#                                               chain priors, so a fit that
#                                               wants to leave them can say so


def _unpack(q):
    return _P_LO + (_P_HI - _P_LO) / (1.0 + np.exp(-q))


def _pack(p):
    u = np.clip((np.asarray(p, dtype=np.float64) - _P_LO) / (_P_HI - _P_LO),
                1e-9, 1 - 1e-9)
    return np.log(u / (1.0 - u))


def _fit_profile(r, Rvir, Rs, rmax, fix_f_exp=None):
    """Unbinned maximum likelihood for (f_exp, tau, lambda_NFW).

    Unbinned because every satellite carries its own host Rvir and Rs, so the
    family is not self-similar across a mass bin and a stacked histogram would
    already have smeared out the shape under test.
    """
    from scipy.optimize import minimize

    def nll(q):
        p = _unpack(q)
        f = p[0] if fix_f_exp is None else fix_f_exp
        pdf = _profile_pdf(r, Rvir, Rs, f, p[1], p[2], rmax)
        if not np.all(pdf > 0):
            return 1e12
        return -float(np.sum(np.log(pdf)))

    best = None
    for start in ([0.5, 6.0, 0.67], [0.2, 3.0, 1.0], [0.8, 12.0, 0.4]):
        res = minimize(nll, _pack(start), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
        if best is None or res.fun < best.fun:
            best = res
    p = _unpack(best.x)
    if fix_f_exp is not None:
        p[0] = fix_f_exp
    return p, -best.fun


def _ks(r, Rvir, Rs, params, rmax):
    """One-sample KS statistic against the fitted (per-satellite) CDFs."""
    F = np.sort(_profile_cdf(r, Rvir, Rs, params[0], params[1], params[2], rmax))
    n = len(F)
    k = np.arange(1, n + 1)
    return float(max(np.max(k / n - F), np.max(F - (k - 1) / n)))


def _sample_model_x(conc, params, rmax_fac, rng, n_cbin=64, n_draw=8):
    """Draw r/Rvir from the fitted family, one inverse CDF per c bin.

    In units of x = r/Rvir the family depends on the host only through the
    concentration (Rs = Rvir/c), so 64 inverse CDFs cover the whole catalogue.
    Each host gets ``n_draw`` stratified draws so the model side of the
    DeltaSigma comparison is not itself the noisiest term.

    Returns (len(conc), n_draw).
    """
    f_exp, tau, lam = params
    xmax = rmax_fac if np.isfinite(rmax_fac) else 30.0
    x_grid = np.geomspace(1e-5, xmax, 3000)
    cb = np.geomspace(conc.min(), conc.max() * 1.0001, n_cbin + 1)
    cmid = np.sqrt(cb[1:] * cb[:-1])
    idx = np.clip(np.searchsorted(cb, conc) - 1, 0, n_cbin - 1)
    u = (np.arange(n_draw)[None, :]
         + rng.random((len(conc), n_draw))) / n_draw
    out = np.empty((len(conc), n_draw))
    for k in range(n_cbin):
        m = idx == k
        if not m.any():
            continue
        F = _profile_cdf(x_grid, 1.0, 1.0 / cmid[k], f_exp, tau, lam, rmax_fac)
        out[m] = np.interp(u[m], F, x_grid)
    return out


def _nfw_sigma_shape(R, Rs):
    """Wright & Brainerd Sigma(R) / (2 rho_s Rs) for an NFW halo."""
    x = np.asarray(R, dtype=np.float64) / Rs
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl, xh = x[lo], x[hi]
    out[lo] = (1.0 - 2.0 / np.sqrt(1 - xl ** 2)
               * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))) / (xl ** 2 - 1)
    out[hi] = (1.0 - 2.0 / np.sqrt(xh ** 2 - 1)
               * np.arctan(np.sqrt((xh - 1) / (xh + 1)))) / (xh ** 2 - 1)
    out[mid] = 1.0 / 3.0
    return out


def _offset_sigma(r_sample, Rs, R_grid, n_r=160, n_mu=24, n_phi=32):
    """Sigma(R) of an NFW halo averaged over a population of 3D offsets.

    Same convolution as TabulatedDeltaSigma.predict: 3D offsets are projected
    isotropically to rho = r sqrt(1-mu^2), then Sigma is averaged over the
    azimuth of the offset. In units of 2 rho_s Rs.

    The offsets are compressed to a weighted radial histogram first. The
    convolution costs O(R_grid x rho x phi) and 1e5 satellites would make it
    hopeless, while 160 radial nodes already resolve a distribution this
    smooth -- the model side uses the same compression, so the two are
    compared on equal footing either way.
    """
    r_pos = r_sample[r_sample > 0]
    hb = np.geomspace(max(r_pos.min(), 1e-5), r_pos.max() * 1.0001, n_r + 1)
    w, _ = np.histogram(r_pos, hb)
    rc = np.sqrt(hb[1:] * hb[:-1])
    ok = w > 0
    rc, w = rc[ok], w[ok].astype(np.float64)

    mu = (np.arange(n_mu) + 0.5) / n_mu
    rho = (rc[:, None] * np.sqrt(1.0 - mu[None, :] ** 2)).ravel()
    w_rho = np.repeat(w, n_mu) / (w.sum() * n_mu)
    cphi = np.cos((np.arange(n_phi) + 0.5) * np.pi / n_phi)
    Sig_grid = _nfw_sigma_shape(R_grid, Rs)

    Sigma = np.zeros_like(R_grid)
    chunk = max(1, int(4e6 // (len(R_grid) * n_phi)))
    for i in range(0, len(rho), chunk):
        rh, wh = rho[i:i + chunk], w_rho[i:i + chunk]
        d = np.sqrt(np.maximum(
            R_grid[:, None, None] ** 2 + rh[None, :, None] ** 2
            + 2.0 * R_grid[:, None, None] * rh[None, :, None]
            * cphi[None, None, :], 0.0))
        Sigma += np.interp(d, R_grid, Sig_grid,
                           left=Sig_grid[0], right=0.0).mean(axis=2) @ wh
    return Sigma


def _deltasigma_from_sigma(Sigma, R_grid):
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (Sigma[1:] * R_grid[1:] + Sigma[:-1] * R_grid[:-1])
        * np.diff(R_grid))])
    return 2.0 * integ / R_grid ** 2 - Sigma


def measure_satellite_profile(args):
    """dN/dr of the NISP satellites, and the best the fitted family can do."""
    need = ["r_host", "rvir_host", "c_host", "mass_200m", "type"]
    have = pq.ParquetFile(args.nisp_path).schema_arrow.names
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(
            f"{args.nisp_path} lacks {missing}.\nThe profile needs the exact "
            f"SOAP host link. Rebuild the catalogue with:\n"
            f"  python example_scripts/precompute_subhalo_catalogue.py --nisp "
            f"--soap_path {args.soap_path} "
            f"--output_dir {os.path.dirname(args.nisp_path)}\n"
            f"then point --nisp_path at the NISP_catalogue_rebuilt.parquet "
            f"it writes.")

    g = pd.read_parquet(args.nisp_path, columns=need)
    sat = g[(g["type"].values == 1) & (g["c_host"].values > 0)
            & (g["mass_200m"].values > MASS_THRESHOLD)]
    r = sat["r_host"].values.astype(np.float64)               # Mpc/h
    Rvir = sat["rvir_host"].values.astype(np.float64) / 1e3   # kpc/h -> Mpc/h
    conc = sat["c_host"].values.astype(np.float64)
    Rs = Rvir / conc
    logM = np.log10(sat["mass_200m"].values.astype(np.float64))
    x = r / Rvir

    n_drop = int((g["type"].values == 1).sum()) - len(sat)
    print(f"satellites: {len(sat):,}  ({n_drop:,} dropped: c<=0 or host below "
          f"{MASS_THRESHOLD:.0e} Msun/h)")
    print(f"host concentration: median {np.median(conc):.2f}")
    for t in (1.0, 2.0, 3.0, 5.0):
        print(f"  beyond {t:.0f} Rvir: {100 * (x > t).mean():7.3f}%")
    print(f"  median r/Rvir = {np.median(x):.3f}   "
          f"median r/Rs = {np.median(r / Rs):.3f}")

    keep = x <= args.profile_xmax
    print(f"\nkeeping r/Rvir <= {args.profile_xmax}: {keep.sum():,} satellites "
          f"({100 * keep.mean():.2f}%)")

    # The model's own truncation is the first thing to test, before any fit:
    # a satellite beyond rmax has exactly zero probability under the sampler,
    # so it is not a bad fit but an impossible one.
    rmax_fac = np.inf if args.rmax_fac == 0 else args.rmax_fac
    if np.isfinite(rmax_fac):
        n_imp = int((x > rmax_fac).sum())
        print(f"\n*** {n_imp:,} satellites ({100 * n_imp / len(x):.3f}%) lie "
              f"beyond the model's exponential cut-off at {rmax_fac:g} Rvir; "
              f"the sampler gives them probability zero everywhere in "
              f"parameter space. Rocher+23 apply no such cut "
              f"(--rmax_fac 0 here). ***")

    edges = np.arange(args.profile_logM_min,
                      args.profile_logM_max + args.profile_dlogM / 2,
                      args.profile_dlogM)
    groups = [("all", keep)] + [
        (f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
         keep & (logM >= edges[i]) & (logM < edges[i + 1]))
        for i in range(len(edges) - 1)]

    print(f"\n{'logM bin':>12} {'N_sat':>8} {'f_exp':>7} {'tau':>7} "
          f"{'lambda':>7} {'KS':>8} {'KS_5%':>8} {'dlnL(NFW)':>10}  chain prior")
    rows, fits = [], {}
    for name, sel in groups:
        if int(sel.sum()) < 500:
            continue
        m = sel & (x <= rmax_fac) if np.isfinite(rmax_fac) else sel
        rr, RR, SS = r[m], Rvir[m], Rs[m]
        n = len(rr)
        rmax = rmax_fac * RR
        p, lnL = _fit_profile(rr, RR, SS, rmax)
        pn, lnL0 = _fit_profile(rr, RR, SS, rmax, fix_f_exp=1e-4)
        ks = _ks(rr, RR, SS, p, rmax)
        inprior = ((0.1 <= p[0] <= 0.9) and (1.0 <= p[1] <= 10.0)
                   and (0.1 <= p[2] <= 2.0))
        print(f"{name:>12} {n:8,} {p[0]:7.3f} {p[1]:7.2f} {p[2]:7.3f} "
              f"{ks:8.4f} {1.36 / np.sqrt(n):8.4f} {lnL - lnL0:10.1f}  "
              f"{'inside' if inprior else 'OUTSIDE <- rail'}")
        rows.append((name, n, p[0], p[1], p[2], ks, lnL, lnL0))
        fits[name] = (p, pn, m)

    if not rows:
        raise SystemExit("no mass bin had enough satellites to fit")

    print("\nKS_5% is the 5% critical value 1.36/sqrt(N). With this many "
          "satellites almost any mis-specification rejects,")
    print("so read the SIZE of KS and the residual panel, not the p-value. "
          "dlnL(NFW) is what the exponential buys over")
    print("a pure rescaled NFW. 'chain prior' flags a best fit outside "
          "f_exp[0.1,0.9] / tau[1,10] / lambda[0.1,2].")

    # ---- what the mis-specification costs DeltaSigma ------------------------
    # The satellite term of the tabulated model is Sigma_halo convolved with the
    # projected offset distribution and nothing else, so swapping the true
    # offsets for the fitted family's isolates the profile error in the
    # observable the chains fit. Done per mass bin with that bin's own Rs, and
    # stacked with the bin's satellite count and 2 rho_s Rs amplitude.
    R_grid = np.geomspace(1e-4, 60.0, 600)
    rng = np.random.default_rng(0)
    Sig_t = np.zeros_like(R_grid)
    Sig_f = np.zeros_like(R_grid)
    wsum = 0.0
    for name, sel in groups[1:]:
        if name not in fits:
            continue
        p, _, m = fits[name]
        rr, RR, CC = r[m], Rvir[m], conc[m]
        Rs_b = float(np.median(RR / CC))
        c_b = float(np.median(CC))
        M_b = float(10 ** np.median(logM[m]))
        # 2 rho_s Rs, the amplitude _nfw_sigma_shape was divided by
        rho_s = M_b / (4 * np.pi * Rs_b ** 3
                       * (np.log1p(c_b) - c_b / (1 + c_b)))
        amp = 2.0 * rho_s * Rs_b * len(rr)
        r_fit = (_sample_model_x(CC, p, rmax_fac, rng) * RR[:, None]).ravel()
        Sig_t += amp * _offset_sigma(rr, Rs_b, R_grid)
        Sig_f += amp * _offset_sigma(r_fit, Rs_b, R_grid)
        wsum += amp
    ds_true = _deltasigma_from_sigma(Sig_t / wsum, R_grid)
    ds_fit = _deltasigma_from_sigma(Sig_f / wsum, R_grid)

    R_out = np.geomspace(0.1, 50.0, 26)
    R_out = np.sqrt(R_out[1:] * R_out[:-1])          # the tabulation binning
    dt = np.interp(R_out, R_grid, ds_true)
    df_ = np.interp(R_out, R_grid, ds_fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 100.0 * (dt / df_ - 1.0)
    print("\nDeltaSigma_sat with the TRUE satellite offsets vs the best-fit "
          "family's, same halo Sigma:")
    print(f"{'rp':>8} {'true/fit-1':>12}")
    for i in range(0, len(R_out), 2):
        print(f"{R_out[i]:8.3f} {frac[i]:11.2f}%")
    print(f"  max |deviation| over {R_out[0]:.2f}-{R_out[-1]:.1f} Mpc/h: "
          f"{np.nanmax(np.abs(frac)):.2f}%")

    np.savez(args.output,
             x=x[keep].astype(np.float32), logM=logM[keep].astype(np.float32),
             Rvir=Rvir[keep].astype(np.float32), conc=conc[keep].astype(np.float32),
             fit_names=np.array([q[0] for q in rows]),
             fit_n=np.array([q[1] for q in rows]),
             fit_params=np.array([q[2:5] for q in rows]),
             fit_ks=np.array([q[5] for q in rows]),
             fit_lnL=np.array([q[6] for q in rows]),
             fit_lnL_nfw=np.array([q[7] for q in rows]),
             rmax_fac=args.rmax_fac, R_out=R_out, ds_true=dt, ds_fit=df_,
             ds_frac=frac, nisp_path=args.nisp_path)
    print(f"\nSaved -> {args.output}")

    # ---- plot ---------------------------------------------------------------
    show = [n for n, _ in groups if n in fits][:8]
    ncol = min(4, len(show))
    nrow = int(np.ceil(len(show) / ncol))
    fig = plt.figure(figsize=(3.9 * ncol, 4.2 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.34,
                             top=0.90, bottom=0.07, left=0.06, right=0.98)
    xb = np.geomspace(1e-2, args.profile_xmax, 41)
    xc = np.sqrt(xb[1:] * xb[:-1])
    w = np.diff(np.log(xb))
    for j, name in enumerate(show):
        row, col = divmod(j, ncol)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1],
                                            hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        p, pn, m = fits[name]
        rr, RR, SS = r[m], Rvir[m], Rs[m]
        h, _ = np.histogram(rr / RR, xb)
        rmax = rmax_fac * RR
        F = np.array([_profile_cdf(e * RR, RR, SS, p[0], p[1], p[2], rmax)
                      for e in xb])
        mod = np.diff(F, axis=0).sum(axis=1)
        Fn = np.array([_profile_cdf(e * RR, RR, SS, 0.0, pn[1], pn[2], rmax)
                       for e in xb])
        nfw = np.diff(Fn, axis=0).sum(axis=1)

        ax.plot(xc, h / w, "k-", lw=1.5, label="NISP satellites")
        ax.plot(xc, mod / w, "r-", lw=1.2, label="best-fit NFW+exp")
        ax.plot(xc, nfw / w, "b--", lw=1.0,
                label=rf"pure NFW ($\lambda$={pn[2]:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"d$N$/dln$r$")
        ax.set_title(rf"logM {name}   $f_{{exp}}$={p[0]:.2f}, "
                     rf"$\tau$={p[1]:.1f}, $\lambda$={p[2]:.2f}", fontsize=8)
        ax.legend(fontsize=6)
        plt.setp(ax.get_xticklabels(), visible=False)

        err = np.sqrt(np.maximum(h, 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.plot(xc, h / mod - 1.0, "r-", lw=1.2)
            axr.plot(xc, h / nfw - 1.0, "b--", lw=1.0)
            axr.fill_between(xc, -err / mod, err / mod, color="grey", alpha=0.3,
                             lw=0)
        axr.axhline(0.0, color="k", lw=0.8)
        axr.set_ylim(-0.6, 0.6); axr.set_xscale("log")
        axr.set_xlabel(r"$r/R_{\rm vir}$")
        axr.set_ylabel("data/model$-1$", fontsize=7)
        for a in (ax, axr):
            a.axvline(1.0, color="grey", lw=0.8, ls=":")
            if np.isfinite(rmax_fac):
                a.axvline(rmax_fac, color="orange", lw=1.0, ls="--")
            a.grid(alpha=0.3, which="both", ls=":")
    fig.suptitle("NISP satellite radial profile vs the fitted "
                 "NFW+exponential family (Flamingo hydro). "
                 "Grey band = Poisson. Orange = the model's 3 Rvir cut-off.",
                 fontsize=12)
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


# ============================================================================
# What the HOD parametrisation itself can reach
# ============================================================================
#
# The companion question to --profile. Every DeltaSigma fit so far recovers an
# occupation nothing like the measured truth, and there are only two ways that
# happens: the likelihood cannot distinguish the truth from the wrong answer
# (a degeneracy), or the parametrisation cannot express the truth at all (a
# cage). This separates them without any lensing in the loop -- it fits the
# repo's OWN HOD forms straight to the measured <N_cen>(M), <N_sat>(M) and asks
# what the best possible fit's fsat and Meff are. If the best achievable form
# already lands on the truth, the parametrisation is exonerated and the problem
# is on the likelihood side; if it does not, no amount of forward-model work
# recovers the occupation.
#
# Fitted in two configurations, so the two costs separate:
#   free  -- every parameter of the form free, generous bounds. The FORM's
#            error alone.
#   chain -- exactly what run_tabulated_chains samples: M1 and Mmax fixed at
#            13.0 / 15.0 and every free parameter clipped to PRIOR_RANGES.
#            The form's error plus what the priors cost on top.
#
# chi2 is reported but is not the readable number here: with ~9M halos the
# Poisson errors are ~0.5%, so chi2/dof of 100 is a few-percent shape error.
# Read the fractional residuals and, above all, the derived fsat / Meff.

_CEN_FORMS = {
    # name: (function, param names, p0, (lo, hi))
    "ELG_mHMQ": (ELG_mHMQ, ["Ac", "Mmin", "sig_M", "gamma"],
                 [0.2, 11.8, 0.5, 4.0],
                 ([1e-3, 10.5, 0.02, -5.0], [10.0, 14.0, 3.0, 30.0])),
    "ELG_SFR": (ELG_SFR, ["Ac", "Mmin", "sig_M", "gamma"],
                [0.15, 11.9, 0.2, -1.0],
                ([1e-3, 10.5, 0.02, -5.0], [10.0, 14.0, 3.0, 30.0])),
    "ELG_GHOD": (ELG_GHOD, ["Ac", "Mmin", "sig_M"],
                 [0.2, 11.9, 0.3],
                 ([1e-3, 10.5, 0.02], [10.0, 14.0, 3.0])),
    "LRG": (LRG_Zheng07, ["Ac", "Mmin", "sig_M"],
            [0.2, 12.5, 0.3],
            ([1e-3, 10.5, 0.02], [10.0, 14.5, 3.0])),
}

_SAT_FORMS = {
    "ELG_satellite_cutoff": (ELG_satellite_cutoff,
                             ["As", "M1", "alpha", "Mcut", "Mmax"],
                             [0.7, 13.0, 0.7, 12.2, 15.0],
                             ([1e-4, 11.0, 0.05, 10.0, 13.0],
                              [50.0, 15.0, 4.0, 14.0, 18.0])),
    "HOD_satellite": (None,      # needs Mmin from the central fit
                      ["As", "Mmin", "M1", "alpha", "kappa"],
                      [0.5, 11.9, 13.0, 0.9, 1.0],
                      ([1e-4, 10.5, 11.0, 0.05, 0.05],
                       [50.0, 14.0, 15.0, 4.0, 5.0])),
}

# run_tabulated_chains.py: what it samples, and where it pins the rest.
_CHAIN_FIXED = {"M1": 13.0, "Mmax": 15.0}
_CHAIN_PRIORS = {
    "As": (0.001, 0.1), "Mmin": (11.1, 13.0), "sig_M": (0.05, 2.0),
    "gamma": (0.0, 10.0), "alpha": (0.1, 2.0), "Mcut": (11.5, 13.5),
}


def _to_np(fn):
    """HOD_models functions are jnp; curve_fit needs float64 numpy."""
    def wrapped(logM, *p):
        return np.asarray(fn(jnp.asarray(logM, dtype=jnp.float64),
                             *[float(v) for v in p]), dtype=np.float64)
    return wrapped


def _derived(logM, ncen, nsat, n_host):
    """ngal, fsat, Meff from an occupation curve and the measured host counts."""
    M = 10.0 ** np.asarray(logM)
    wc, ws = n_host * np.asarray(ncen), n_host * np.asarray(nsat)
    tot = wc + ws
    return dict(
        ngal=tot.sum() / LBOX ** 3,
        fsat=ws.sum() / tot.sum(),
        Meff_cen=(wc * M).sum() / wc.sum(),
        Meff_all=(tot * M).sum() / tot.sum(),
    )


def _fit_one(fn, x, y, yerr, p0, bounds, names, fixed=None, priors=None):
    """curve_fit with a subset of parameters pinned and the rest prior-clipped."""
    from scipy.optimize import curve_fit
    fixed = fixed or {}
    free = [i for i, n in enumerate(names) if n not in fixed]
    lo = list(bounds[0]); hi = list(bounds[1])
    if priors:
        for i, n in enumerate(names):
            if n in priors:
                lo[i] = max(lo[i], priors[n][0])
                hi[i] = min(hi[i], priors[n][1])
    full = np.array([fixed.get(n, p0[i]) for i, n in enumerate(names)],
                    dtype=np.float64)
    full = np.clip(full, lo, hi)

    def model(xx, *q):
        p = full.copy()
        p[free] = q
        return fn(xx, *p)

    q0 = np.clip(full[free], np.array(lo)[free], np.array(hi)[free])
    popt, _ = curve_fit(model, x, y, p0=q0, sigma=yerr, absolute_sigma=True,
                        bounds=(np.array(lo)[free], np.array(hi)[free]),
                        maxfev=200000)
    full[free] = popt
    resid = (y - model(x, *popt)) / yerr
    return full, float(np.sum(resid ** 2)), len(free)


def _rails(params, names, fixed, priors):
    """Which free parameters ended up pinned against their prior edge."""
    out = []
    for i, n in enumerate(names):
        if n in fixed or n not in (priors or {}):
            continue
        lo, hi = priors[n]
        if abs(params[i] - lo) < 1e-3 * max(abs(lo), 1) or \
           abs(params[i] - hi) < 1e-3 * max(abs(hi), 1):
            out.append(n)
    return out


def fit_hod_forms(args):
    """Fit the repo's HOD forms to the measured truth occupation."""
    z = np.load(args.fit_hod, allow_pickle=True)
    logM, n_host = z["logM"], z["n_host"].astype(np.float64)
    ncen, nsat = z["ncen"], z["nsat"]
    n_cen, n_sat = z["n_cen"].astype(np.float64), z["n_sat"].astype(np.float64)
    print(f"measured HOD: {args.fit_hod}")
    print(f"  mass column {str(z['mass_column'])}, source {str(z['nisp_path'])}")

    # Binomial for centrals (one per halo, so <N_cen> <= 1), Poisson for
    # satellites -- the rebuild measured sqrt(Var N_sat)/sqrt(<N_sat>) = 1.011.
    with np.errstate(divide="ignore", invalid="ignore"):
        cen_err = np.sqrt(np.maximum(n_cen * (1.0 - ncen), 1.0)) / n_host
        sat_err = np.sqrt(np.maximum(n_sat, 1.0)) / n_host

    ok_c = (n_host > args.fit_hod_min_host) & (n_cen > 10)
    ok_s = (n_host > args.fit_hod_min_host) & (n_sat > 10)
    print(f"  fitting {ok_c.sum()} central bins, {ok_s.sum()} satellite bins "
          f"(n_host > {args.fit_hod_min_host})")

    truth = _derived(logM, ncen, nsat, n_host)
    print(f"\nTRUTH (re-derived from the same binned curves):")
    print(f"  ngal {truth['ngal']:.4e}   fsat {truth['fsat']:.4f}   "
          f"Meff_cen {truth['Meff_cen']:.3e}   Meff_all {truth['Meff_all']:.3e}")
    print(f"  (catalogue values in the npz: ngal {float(z['ngal']):.4e}, "
          f"fsat {float(z['fsat']):.4f}, Meff_cen {float(z['Meff_cen']):.3e})")

    results = {}
    for cfg_name, fixed, priors in (("free", {}, None),
                                    ("chain", _CHAIN_FIXED, _CHAIN_PRIORS)):
        # The chain priors are stated in the divided-by-10 amplitude
        # convention, so the truth has to be put into that convention before
        # they mean anything -- otherwise As rails at its upper edge purely
        # because the units differ by 10x.
        sc = 1.0 if priors is None else args.fit_hod_amp_scale
        y_c, ye_c = ncen * sc, cen_err * sc
        y_s, ye_s = nsat * sc, sat_err * sc
        print(f"\n{'=' * 78}\n{cfg_name.upper()} configuration"
              f"{'  (M1=13.0, Mmax=15.0 fixed; PRIOR_RANGES enforced)' if fixed else '  (all parameters free)'}"
              f"\n{'=' * 78}")

        # ---- centrals ----
        print(f"{'central form':>22} {'chi2/dof':>12} {'|resid| med':>11} "
              f"{'max':>7}   best fit")
        cen_fits = {}
        for name, (fn, names, p0, bnd) in _CEN_FORMS.items():
            p0 = [v * sc if n in ("Ac", "As") else v for n, v in zip(names, p0)]
            if cfg_name == "chain" and name != "ELG_mHMQ":
                continue                      # the chains only run mHMQ
            f = _to_np(fn)
            p, chi2, nfree = _fit_one(f, logM[ok_c], y_c[ok_c], ye_c[ok_c],
                                      p0, bnd, names, fixed=None,
                                      priors=priors)
            m = f(logM, *p)
            good = m > 1e-3 * np.nanmax(y_c[ok_c])
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.abs(np.where(good, y_c / m - 1.0, np.nan))[ok_c]
            dof = int(ok_c.sum()) - nfree
            rail = _rails(p, names, {}, priors)
            print(f"{name:>22} {chi2 / dof:12.1f} {np.nanmedian(fr):11.3f} "
                  f"{np.nanmax(fr):7.3f}   "
                  + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, p))
                  + (f"   RAIL: {','.join(rail)}" if rail else ""))
            cen_fits[name] = (p, m, chi2 / dof)

        # ---- satellites ----
        print(f"{'satellite form':>22} {'chi2/dof':>12} {'|resid| med':>11} "
              f"{'max':>7}   best fit")
        sat_fits = {}
        for name, (fn, names, p0, bnd) in _SAT_FORMS.items():
            p0 = [v * sc if n in ("Ac", "As") else v for n, v in zip(names, p0)]
            if cfg_name == "chain" and name != "ELG_satellite_cutoff":
                continue                      # the chains run elg_satellite=True
            f = _to_np(fn if fn is not None else HOD_satellite)
            p, chi2, nfree = _fit_one(f, logM[ok_s], y_s[ok_s], ye_s[ok_s],
                                      p0, bnd, names, fixed=fixed,
                                      priors=priors)
            m = f(logM, *p)
            good = m > 1e-3 * np.nanmax(y_s[ok_s])
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.abs(np.where(good, y_s / m - 1.0, np.nan))[ok_s]
            dof = int(ok_s.sum()) - nfree
            rail = _rails(p, names, fixed, priors)
            print(f"{name:>22} {chi2 / dof:12.1f} {np.nanmedian(fr):11.3f} "
                  f"{np.nanmax(fr):7.3f}   "
                  + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, p))
                  + (f"   RAIL: {','.join(rail)}" if rail else ""))
            sat_fits[name] = (p, m, chi2 / dof)

        # ---- what those best fits imply for the numbers the chains report ----
        cen_key = "ELG_mHMQ"
        sat_key = "ELG_satellite_cutoff"
        d = _derived(logM, np.clip(cen_fits[cen_key][1] / sc, 0, 1),
                     np.maximum(sat_fits[sat_key][1] / sc, 0), n_host)
        print(f"\n  {cen_key} + {sat_key} -> "
              f"ngal {d['ngal']:.4e}  fsat {d['fsat']:.4f}  "
              f"Meff_cen {d['Meff_cen']:.3e}  Meff_all {d['Meff_all']:.3e}")
        print(f"  vs truth:                     "
              f"ngal x{d['ngal'] / truth['ngal']:.3f}  "
              f"fsat x{d['fsat'] / truth['fsat']:.3f}  "
              f"Meff_cen x{d['Meff_cen'] / truth['Meff_cen']:.3f}  "
              f"Meff_all x{d['Meff_all'] / truth['Meff_all']:.3f}")
        results[cfg_name] = (cen_fits, sat_fits, d)

    # ---- the verdict, stated in the terms the fits are judged on -----------
    d_free, d_chain = results["free"][2], results["chain"][2]
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print("Best achievable occupation from the parametrisation the chains run, "
          "with no lensing in the loop:")
    for lab, d in (("form free ", d_free), ("chain cfg ", d_chain)):
        print(f"  {lab}: fsat {d['fsat']:.4f} (truth {truth['fsat']:.4f}, "
              f"x{d['fsat'] / truth['fsat']:.2f})   "
              f"Meff_cen {d['Meff_cen']:.3e} (truth {truth['Meff_cen']:.3e}, "
              f"x{d['Meff_cen'] / truth['Meff_cen']:.2f})")
    print("Compare with the chain posteriors, which land at fsat ~ 0.005 and "
          "Meff_cen ~ 3.7e12 (x4.3).")
    print("If the numbers above sit ON the truth, the parametrisation is "
          "exonerated and the failure is in the likelihood")
    print("(degeneracy / scale cuts / covariance), not in the HOD form.")

    np.savez(args.output, logM=logM, n_host=n_host, ncen=ncen, nsat=nsat,
             cen_err=cen_err, sat_err=sat_err, truth=np.array([truth]),
             **{f"{cfg}_{kind}_{name}": v[0]
                for cfg, (cf, sf, _) in results.items()
                for kind, dd in (("cen", cf), ("sat", sf))
                for name, v in dd.items()},
             **{f"{cfg}_derived": np.array([r[2]])
                for cfg, r in results.items()})
    print(f"\nSaved -> {args.output}")

    # ---- plot ---------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.06,
                          wspace=0.24, top=0.90, bottom=0.09,
                          left=0.08, right=0.98)
    for col, (kind, y, ye, okm, fitkey) in enumerate((
            ("centrals", ncen, cen_err, ok_c, 0),
            ("satellites", nsat, sat_err, ok_s, 1))):
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        ax.errorbar(10 ** logM[okm], y[okm], yerr=ye[okm], fmt="k.", ms=4,
                    lw=1, label="NISP truth")
        for cfg, style in (("free", "-"), ("chain", "--")):
            fits = results[cfg][fitkey]
            name = "ELG_mHMQ" if col == 0 else "ELG_satellite_cutoff"
            sc = 1.0 if cfg == "free" else args.fit_hod_amp_scale
            m = fits[name][1] / sc
            ax.plot(10 ** logM, np.maximum(m, 1e-6), style, lw=1.4,
                    label=f"{name} ({cfg}, "
                          f"$\\chi^2_r$={fits[name][2]:.0f})")
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.plot(10 ** logM[okm], (y / m - 1.0)[okm], style, lw=1.3)
        if col == 0:      # the alternative central forms, free config only
            for name in ("ELG_SFR", "ELG_GHOD"):
                if name in results["free"][0]:
                    ax.plot(10 ** logM, np.maximum(results["free"][0][name][1],
                                                   1e-6), ":", lw=1,
                            label=f"{name} (free)")
        axr.axhline(0, color="k", lw=0.8)
        axr.fill_between(10 ** logM[okm], -(ye / np.maximum(y, 1e-12))[okm],
                         (ye / np.maximum(y, 1e-12))[okm], color="grey",
                         alpha=0.35, lw=0)
        for a in (ax, axr):
            a.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(max(1e-4, np.nanmin(y[okm]) * 0.3), np.nanmax(y[okm]) * 3)
        ax.set_ylabel(rf"$\langle N_{{\rm {kind[:3]}}}\rangle$")
        ax.set_title(kind, fontsize=10)
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), visible=False)
        axr.set_ylim(-0.5, 0.5)
        axr.set_xlabel(r"$M_{200m}\ [M_\odot/h]$")
        axr.set_ylabel("data/fit$-1$", fontsize=8)
        for a in (ax, axr):
            a.grid(alpha=0.3, which="both", ls=":")
    fig.suptitle("Can the HOD parametrisation reach the measured truth? "
                 "(no lensing in the loop; grey band = measurement error)",
                 fontsize=12)
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


def main():
    args = parse_args()

    default_out = os.path.join(os.path.dirname(REF_PATH),
                               "hydro_measured_data_vector.npz")

    if args.hod:
        if args.output == default_out:
            args.output = os.path.join(os.path.dirname(REF_PATH),
                                       "hydro_measured_hod.npz")
        measure_hod(args)
        return

    if args.fit_hod:
        if args.output == default_out:
            args.output = os.path.join(os.path.dirname(REF_PATH),
                                       "hydro_hod_form_fits.npz")
        fit_hod_forms(args)
        return

    if args.profile:
        if args.output == default_out:
            args.output = os.path.join(os.path.dirname(REF_PATH),
                                       "hydro_measured_satellite_profile.npz")
        measure_satellite_profile(args)
        return

    if args.truth_ds:
        if args.output == default_out:
            args.output = os.path.join(os.path.dirname(REF_PATH),
                                       "hydro_truth_occupation_ds.npz")
        predict_truth_deltasigma(args)
        return

    if args.hod_env:
        if args.output == default_out:
            args.output = os.path.join(
                os.path.dirname(REF_PATH),
                f"hydro_measured_hod_env_{args.env_column}.npz")
        measure_hod_environment(args)
        return

    # ---- cosmology (RHO_M, RSD factor) -------------------------------------
    cosmo = Cosmology(COSMO_PARAMS, mass_function="Tinker08",
                      mass_definition=MASS_DEFINITION, use_dark_emulator=False,
                      verbose=False, units_per_h=True)
    h = cosmo.h
    a = 1.0 / (1.0 + ZEFF)
    RHO_M = cosmo.get_rho_m()
    rsd_factor = h / (cosmo.Hz(ZEFF) * a)   # km/s -> Mpc/h (as in HaloOccupation)
    print(f"h={h}  RHO_M={RHO_M:.4e}  rsd_factor={rsd_factor:.5e}")

    # ---- reference binning + data ------------------------------------------
    ref = np.load(args.ref_path)
    if args.rp_min is not None:
        rp_bins = np.geomspace(args.rp_min, args.rp_max, args.n_rp)
    else:
        rp_bins = np.asarray(ref["rp_bins"])
    custom_bins = args.rp_min is not None
    rp_cen = np.sqrt(rp_bins[1:] * rp_bins[:-1])
    rp_ref = np.asarray(ref["rp_centers"])
    ds_ref = np.asarray(ref["delta_sigma"])
    ds_err = np.asarray(ref["delta_sigma_err"])
    wgg_ref = np.asarray(ref["wgg"])
    wgg_err = np.asarray(ref["wgg_err"])
    print(f"{'custom' if custom_bins else 'reference'} binning: {len(rp_bins)-1} bins "
          f"[{rp_bins[0]:.2f}, {rp_bins[-1]:.1f}] Mpc/h")

    # ---- galaxies ----------------------------------------------------------
    g = pd.read_parquet(args.nisp_path,
                        columns=["x", "y", "z", "vx", "vy", "vz", "type"])
    if args.gal_type is not None:
        g = g[g["type"] == args.gal_type]
    pos_g = np.ascontiguousarray(g[["x", "y", "z"]].values, dtype=np.float64)
    vel_g = np.ascontiguousarray(g[["vx", "vy", "vz"]].values, dtype=np.float64)
    ngal = len(pos_g)
    print(f"galaxies: {ngal:,}  (ngal = {ngal/LBOX**3:.3e} (Mpc/h)^-3)")

    # ---- particles (subsampled, mass-weighted) -----------------------------
    # Streamed so that only the subsample is ever materialised: reading all
    # 233M rows to keep 2% costs ~15 GB of peak RSS for nothing.
    rng = np.random.default_rng(args.particle_seed)
    cols = ["x", "y", "z", "mass"]
    chunks = []
    pf = pq.ParquetFile(args.part_path)
    for batch in pf.iter_batches(batch_size=5_000_000, columns=cols):
        arr = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False)
                               for c in cols]).astype(np.float64)
        if args.particle_fraction < 1.0:
            arr = arr[rng.random(len(arr)) < args.particle_fraction]
        chunks.append(arr)
    p = np.concatenate(chunks); del chunks
    pos_p = np.ascontiguousarray(p[:, :3])
    w_p = None if args.no_mass_weight else np.ascontiguousarray(p[:, 3])
    del p
    print(f"particles: {len(pos_p):,} "
          f"(fraction {args.particle_fraction}, "
          f"{'unweighted' if w_p is None else 'mass-weighted'})")

    # ---- DeltaSigma (real space, mass-weighted galaxy x particle) ----------
    print("Measuring DeltaSigma ...")
    _, ds_meas = compute_galaxy_lensing(
        pos_g, pos_p, LBOX, args.rsd_axis, RHO_M, rp_bins,
        weights_part=w_p, chi_max=args.chi_max,
        bins_comp=np.geomspace(5e-3, 120, 201))  # matches tabulated + DMO pipeline

    # ---- wgg (redshift space galaxy auto) ----------------------------------
    print("Measuring wgg ...")
    ax = {"x": 0, "y": 1, "z": 2}[args.rsd_axis]
    pos_g_rsd = pos_g.copy()
    pos_g_rsd[:, ax] = (pos_g_rsd[:, ax] + vel_g[:, ax] * rsd_factor) % LBOX
    pi_bins = np.linspace(0.0, args.pi_max, int(args.pi_max) + 1)
    _, wgg_meas = compute_galaxy_clustering(
        pos_g_rsd, LBOX, args.rsd_axis, "rppi", rp_bins,
        bins2=pi_bins, output="wp")
    wgg_meas = np.asarray(wgg_meas)

    # ---- cross-check -------------------------------------------------------
    def _report(name, meas, refv, err):
        dev = meas / refv - 1.0
        nsig = (meas - refv) / err
        print(f"\n=== {name}: measured vs reference ===")
        print(f"{'rp':>8} {'meas':>11} {'ref':>11} {'dev%':>8} {'nsig':>7}")
        for j in range(len(refv)):
            print(f"{rp_ref[j]:8.3f} {meas[j]:11.4f} {refv[j]:11.4f} "
                  f"{100*dev[j]:8.2f} {nsig[j]:7.2f}")
        print(f"  median|dev| = {100*np.median(np.abs(dev)):.2f}%   "
              f"max|dev| = {100*np.max(np.abs(dev)):.2f}%   "
              f"median|nsig| = {np.median(np.abs(nsig)):.2f}")
        return dev, nsig

    if custom_bins:
        print("\ncustom binning: skipping the reference comparison "
              "(the reference is on a different grid)")
        ds_dev = wgg_dev = None
    else:
        ds_dev, _ = _report("DeltaSigma", ds_meas, ds_ref, ds_err)
        wgg_dev, _ = _report("wgg", wgg_meas, wgg_ref, wgg_err)

    # ---- save + plot -------------------------------------------------------
    ref_fields = {} if custom_bins else {"delta_sigma_ref": ds_ref,
                                         "wgg_ref": wgg_ref}
    np.savez(args.output, rp_centers=rp_cen, rp_bins=rp_bins,
             delta_sigma=ds_meas, wgg=wgg_meas, **ref_fields,
             particle_fraction=args.particle_fraction, pi_max=args.pi_max,
             chi_max=args.chi_max, ngal=ngal)
    print(f"\nSaved measured data vector -> {args.output}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (name, meas, refv, err, dev) in enumerate([
            (r"$\Delta\Sigma$", ds_meas, ds_ref, ds_err, ds_dev),
            (r"$w_{gg}$", wgg_meas, wgg_ref, wgg_err, wgg_dev)]):
        top, bot = axes[0, col], axes[1, col]
        if not custom_bins:
            top.errorbar(rp_ref, rp_ref * refv, yerr=rp_ref * err, fmt="ko",
                         ms=4, label="reference")
        top.plot(rp_cen, rp_cen * meas, "r.-", label="measured (in box)")
        top.set_xscale("log"); top.set_ylabel(rf"$r_p\,${name}")
        top.legend(); top.set_title(name); top.grid(alpha=0.3, which="both", ls=":")
        bot.axhline(0, color="k", lw=0.8)
        bot.axhspan(-5, 5, color="green", alpha=0.1)
        if not custom_bins:
            bot.plot(rp_ref, 100 * dev, "r.-")
        bot.set_xscale("log"); bot.set_xlabel(r"$r_p$ [Mpc/$h$]")
        bot.set_ylabel("dev [%]"); bot.grid(alpha=0.3, which="both", ls=":")
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot -> {plot_path}")


if __name__ == "__main__":
    main()
