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

Usage (cluster):
    python example_scripts/measure_hydro_data_vector.py \
        --particle_fraction 0.02 --pi_max 100 \
        --output /sps/euclid/Users/rpaviot/flamingo/hydro_measured_data_vector.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HOD_NRV.HOD_analytical.pycosmo import Cosmology
from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
    compute_galaxy_lensing, compute_galaxy_clustering)
from example_scripts.run_tabulated_chains import (
    COSMO_PARAMS, ZEFF, LBOX, MASS_DEFINITION)

HYDRO_DIR = "/sps/euclid/Users/rpaviot/flamingo/snapshots_hydro"
NISP_PATH = os.path.join(HYDRO_DIR, "NISP_catalogue_flamingo.parquet")
PART_PATH = os.path.join(HYDRO_DIR,
                         "hydro_flamingo_0058_downsampled_0.1percent.parquet")
REF_PATH = "/sps/euclid/Users/rpaviot/flamingo/data_bin0_finalbinning.npz"
SOAP_PATH = os.path.join(HYDRO_DIR, "halo_properties_0058.hdf5")
DMO_HOST_PATH = ("/sps/euclid/Users/rpaviot/flamingo/snapshots_DMO/"
                 "host_catalogue_ab.parquet")
HOST_MASS_CACHE = "/sps/euclid/Users/rpaviot/flamingo/hydro_host_mass.npz"
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
    g = pd.read_parquet(args.nisp_path, columns=["mass", "type"])
    if args.gal_type is not None:
        g = g[g["type"] == args.gal_type]
    logM_gal = np.log10(g["mass"].values)      # host M200m [Msun/h]
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

    out = args.output
    np.savez(out, logM=logM, logM_edges=edges, n_host=n_host, n_dmo=n_dmo,
             n_cen=n_cen, n_sat=n_sat, ncen=ncen, nsat=nsat,
             ncen_dmo=ncen_dmo, nsat_dmo=nsat_dmo,
             ngal=ngal / LBOX**3, fsat=fsat, Meff_cen=Meff, Meff_all=Meff_all)
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


def main():
    args = parse_args()

    if args.hod:
        default_out = os.path.join(os.path.dirname(REF_PATH),
                                   "hydro_measured_data_vector.npz")
        if args.output == default_out:
            args.output = os.path.join(os.path.dirname(REF_PATH),
                                       "hydro_measured_hod.npz")
        measure_hod(args)
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
