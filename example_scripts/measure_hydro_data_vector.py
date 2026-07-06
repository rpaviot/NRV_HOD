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
    p.add_argument("--gal_type", type=int, default=None,
                   help="Optional NISP 'type' selection (default: all galaxies).")
    p.add_argument("--no_mass_weight", action="store_true",
                   help="Count particles unweighted (diagnostic).")
    return p.parse_args()


def main():
    args = parse_args()

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
    rp_bins = np.asarray(ref["rp_bins"])
    rp_ref = np.asarray(ref["rp_centers"])
    ds_ref = np.asarray(ref["delta_sigma"])
    ds_err = np.asarray(ref["delta_sigma_err"])
    wgg_ref = np.asarray(ref["wgg"])
    wgg_err = np.asarray(ref["wgg_err"])
    print(f"reference binning: {len(rp_bins)-1} bins "
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
    p = pd.read_parquet(args.part_path, columns=["x", "y", "z", "mass"])
    if args.particle_fraction < 1.0:
        rng = np.random.default_rng(args.particle_seed)
        keep = rng.random(len(p)) < args.particle_fraction
        p = p[keep]
    pos_p = np.ascontiguousarray(p[["x", "y", "z"]].values, dtype=np.float64)
    w_p = None if args.no_mass_weight else \
        np.ascontiguousarray(p["mass"].values, dtype=np.float64)
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

    ds_dev, _ = _report("DeltaSigma", ds_meas, ds_ref, ds_err)
    wgg_dev, _ = _report("wgg", wgg_meas, wgg_ref, wgg_err)

    # ---- save + plot -------------------------------------------------------
    np.savez(args.output, rp_centers=rp_ref, rp_bins=rp_bins,
             delta_sigma=ds_meas, wgg=wgg_meas,
             delta_sigma_ref=ds_ref, wgg_ref=wgg_ref,
             particle_fraction=args.particle_fraction, pi_max=args.pi_max,
             chi_max=args.chi_max, ngal=ngal)
    print(f"\nSaved measured data vector -> {args.output}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (name, meas, refv, err, dev) in enumerate([
            (r"$\Delta\Sigma$", ds_meas, ds_ref, ds_err, ds_dev),
            (r"$w_{gg}$", wgg_meas, wgg_ref, wgg_err, wgg_dev)]):
        top, bot = axes[0, col], axes[1, col]
        top.errorbar(rp_ref, rp_ref * refv, yerr=rp_ref * err, fmt="ko",
                     ms=4, label="reference")
        top.plot(rp_ref, rp_ref * meas, "r.-", label="measured (in box)")
        top.set_xscale("log"); top.set_ylabel(rf"$r_p\,${name}")
        top.legend(); top.set_title(name); top.grid(alpha=0.3, which="both", ls=":")
        bot.axhline(0, color="k", lw=0.8)
        bot.axhspan(-5, 5, color="green", alpha=0.1)
        bot.plot(rp_ref, 100 * dev, "r.-")
        bot.set_xscale("log"); bot.set_xlabel(r"$r_p$ [Mpc/$h$]")
        bot.set_ylabel("dev [%]"); bot.grid(alpha=0.3, which="both", ls=":")
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot -> {plot_path}")


if __name__ == "__main__":
    main()
