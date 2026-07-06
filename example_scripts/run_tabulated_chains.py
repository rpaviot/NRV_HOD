#!/usr/bin/env python3
"""
run_tabulated_chains.py

Nautilus DeltaSigma fits on the Flamingo L1000N1800 ELG data using the
TabCorr-style TabulatedDeltaSigma forward model — no LHS grid, no NN
emulator. Each likelihood call is the exact expectation of the MC pipeline,
so chains run directly against the tabulated cache.

Sampling is single-process with nautilus ``vectorized=True`` on the
jit/vmap-batched likelihood (TabulatedFitter.build_batched_loglike):
no multiprocessing pool, hence no JAX-after-fork deadlock (which stalled
jobs 52827669-75 at zero likelihood calls). Parallelism comes from XLA
threading; float64 is enabled below for parity with the NumPy path
(validated by cross_check_tabulated.py --jax).

Mirrors run_emulator_chains.py (same priors, fixed params, ngal rescaling,
scale cuts, outputs) so posteriors are directly comparable to the
grid+emulator chains (chains_BARYON / FULL_* generations).

Usage (cluster):
    python example_scripts/run_tabulated_chains.py NFW \
        --cache_path /sps/euclid/Users/rpaviot/flamingo/tabulated_cache_DMO.h5 \
        --output_dir /sps/euclid/Users/rpaviot/flamingo/chains_TABULATED_DMO
"""

import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HOD_NRV.utilsf.numerical_sampler import TabulatedFitter, FitCase
from HOD_NRV.HOD_numerical.HOD_models import Occupation, rescale_Ac_to_target_ngal
from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
    HaloCenterLensingCache, TabulatedDeltaSigma,
)

# ============================================================================
# Paths & global configuration (mirrors run_emulator_chains.py)
# ============================================================================

FLAMINGO_DIR = "/sps/euclid/Users/rpaviot/flamingo"
DATA_PATH_DEFAULT = os.path.join(FLAMINGO_DIR, "data_bin0_finalbinning.npz")
HALO_PATH_DEFAULT = os.path.join(FLAMINGO_DIR, "snapshots_DMO/host_catalogue_ab.parquet")

RP_MIN_VALUES     = [0.2, 0.5, 1.0, 2.0]
N_LIVE            = 1_000
N_EFF             = 30_000
N_PROFILE_SAMPLES = 2_000

M1_FIXED      = 13.0
MMAX_FIXED    = 15.0
TARGET_NGAL   = 2.3e-4
ZEFF          = 1.0
LBOX          = 681.0
MASS_DEFINITION = "MassDef200m"
AC_FIDUCIAL   = 0.01
AB_COLUMN     = "fs_norm"

COSMO_PARAMS = {
    "h":    0.681,
    "Omc":  0.306 - 0.0486 - 1.39e-3,
    "Omb":  0.0486,
    "A_s":  2.099e-9,
    "n_s":  0.967,
    "Omnu": 1.39e-3,
}

COLUMN_MAPPING = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

PRIOR_RANGES = {
    "As":         (0.001, 0.1),
    "Mmin":       (11.1,  13.0),
    "sig_M":      (0.05,  2.0),
    "gamma":      (0.0,   10.0),
    "alpha":      (0.1,   2.0),
    "Mcut":       (11.5,  13.5),
    "lambda_NFW": (0.1,   2.0),
    "f_exp":      (0.0,   0.9),
    "tau":        (1.0,   10.0),
    "kappa_EE":   (0.5,   1.0),
    "B_cent":     (-0.5,  0.5),
    "B_sat":      (-0.5,  0.5),
}

FIXED_DEFAULTS = {
    "M1":     M1_FIXED,
    "Mmax":   MMAX_FIXED,
    "A_cent": 0.0,
    "A_sat":  0.0,
}

FIT_CASE_OF = {
    "NFW":  FitCase.STANDARD_NFW,
    "EXT":  FitCase.EXTENDED_PROFILE,
    "CONF": FitCase.CONFORMITY,
}
ALL_CASE_NAMES = ["NFW", "EXT", "CONF", "NFW_AB", "EXT_AB", "CONF_AB"]


def _base_name(case_name):
    return case_name[:-3] if case_name.endswith("_AB") else case_name


def build_param_config(fit_case, *, assembly_bias, gaussian_ab=False):
    """Same prior structure as run_emulator_chains.py --full_bb (elg_satellite)."""
    cfg = dict(FIXED_DEFAULTS)
    active = ["As", "Mmin", "sig_M", "gamma", "alpha", "Mcut", "lambda_NFW"]

    if fit_case in (FitCase.EXTENDED_PROFILE, FitCase.CONFORMITY):
        active += ["f_exp", "tau"]
    else:
        cfg["f_exp"], cfg["tau"] = 0.0, 5.0

    if fit_case == FitCase.CONFORMITY:
        active.append("kappa_EE")
    else:
        cfg["kappa_EE"] = 1.0

    if assembly_bias:
        if gaussian_ab:
            cfg["B_cent"] = (0.0, 0.15, "gaussian")
            cfg["B_sat"]  = (0.0, 0.15, "gaussian")
        else:
            active += ["B_cent", "B_sat"]
    else:
        cfg["B_cent"], cfg["B_sat"] = 0.0, 0.0

    for name in active:
        cfg[name] = PRIOR_RANGES[name]
    return cfg


# ============================================================================
# Model builders
# ============================================================================

def build_halo_occupation(fit_case, halo_path):
    """HaloOccupation with fs_norm mapped as fE — matches the FULL grid runs.

    assembly_bias is always on so the fI tabulation bins resolve; non-AB
    cases simply fix B_cent = B_sat = 0.
    """
    cmap = dict(COLUMN_MAPPING)
    cmap["fE"] = AB_COLUMN
    halo = HaloOccupation(
        cosmology=COSMO_PARAMS,
        zeff=ZEFF,
        Lbox=LBOX,
        column_mapping=cmap,
        mass_definition=MASS_DEFINITION,
        halo_path=halo_path,
        DataFrame_part=None,
        assembly_bias=True,
        apply_rsd=False,
        do_test=False,
    )
    halo.set_halo_model(
        "ELG_mHMQ",
        conformity=(fit_case == FitCase.CONFORMITY),
        elg_satellite=True,
    )
    return halo


def _make_rescale_occupation(halo, fit_case):
    """Non-AB Occupation for the (Ac, As) -> ngal rescale (grid convention)."""
    return Occupation(
        "ELG_mHMQ", halo.logM_bins, halo.mass_function,
        conformity=(fit_case == FitCase.CONFORMITY),
        elg_satellite=True,
    )


# ============================================================================
# Posterior HOD profiles / Meff / fsat (same as run_emulator_chains.py)
# ============================================================================

def compute_meff_fsat(occupation, theta_best, fitter):
    full = fitter.full_params(theta_best)
    # compute_Meff / compute_fsat return scalars (not arrays) — no [0] indexing.
    return occupation.compute_Meff(full), occupation.compute_fsat(full)


def compute_hod_profiles_from_chain(points, weights, fitter, occupation, halo,
                                    n_samples=N_PROFILE_SAMPLES):
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(w), size=n_samples, replace=True, p=w)
    samples = points[idx]

    logM = np.asarray(halo.logM_bins)
    ncen_all = np.empty((n_samples, len(logM)))
    nsat_all = np.empty((n_samples, len(logM)))
    for i, theta in enumerate(samples):
        theta_dict = dict(zip(fitter.param_names, theta))
        full = fitter.full_params(theta_dict)
        nc, ns = occupation.compute_HOD_occupation(logM, full)
        ncen_all[i] = np.asarray(nc)
        nsat_all[i] = np.asarray(ns)

    def _med_sig(arr):
        q16, q50, q84 = np.percentile(arr, [16, 50, 84], axis=0)
        return q50, (q84 - q16) / 2.0

    ncen_med, ncen_sig = _med_sig(ncen_all)
    nsat_med, nsat_sig = _med_sig(nsat_all)
    return logM, ncen_med, ncen_sig, nsat_med, nsat_sig


def plot_hod_profiles(case_name, logM, profiles_by_rp_min, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    ax_cen, ax_sat = axes
    M = 10**logM
    for (rp_min, vals), color in zip(profiles_by_rp_min.items(), COLORS):
        label = f"$r_{{\\rm min}}={rp_min}$ Mpc/$h$"
        for ax, med_key, sig_key in [(ax_cen, 'ncen_med', 'ncen_sig'),
                                     (ax_sat, 'nsat_med', 'nsat_sig')]:
            ax.plot(M, vals[med_key], color=color, lw=1.8, label=label)
            ax.fill_between(M, vals[med_key] - vals[sig_key],
                            vals[med_key] + vals[sig_key],
                            color=color, alpha=0.25)
    for ax, title in zip(axes, [r'$\langle N_{\rm cen}\rangle(M)$',
                                r'$\langle N_{\rm sat}\rangle(M)$']):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$M\;[M_\odot/h]$', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, which='both', ls=':')
    fig.suptitle(f'HOD occupation — {case_name} (tabulated)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, f"hod_profiles_{case_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  HOD profile plot saved: {path}")


# ============================================================================
# Run one case
# ============================================================================

def run_case(case_name, fit_case, halo, tab, args):
    print(f"\n{'='*60}\n  Case: {case_name}  ({fit_case.name}, tabulated)\n{'='*60}")
    os.makedirs(args.output_dir, exist_ok=True)

    assembly_bias = case_name.endswith("_AB")
    occupation_rescale = _make_rescale_occupation(halo, fit_case)
    occupation_full = halo.HOD

    data = np.load(args.data_path)
    rp_all = data['rp_centers']
    ds_all = data['delta_sigma']
    ds_err = data['delta_sigma_err']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(rp_all, rp_all * ds_all, yerr=rp_all * ds_err,
                fmt='ko', ms=4, zorder=10, label='Flamingo ELG data')

    bestfits = {}
    for rp_min, color in zip(args.rp_min_values, COLORS):
        print(f"\n  rp_min = {rp_min} Mpc/h")
        chain_path = os.path.join(args.output_dir,
                                  f"chain_{case_name}_rmin{rp_min}.npz")

        param_config = build_param_config(
            fit_case, assembly_bias=assembly_bias, gaussian_ab=args.gaussian_ab)

        fitter = TabulatedFitter(
            tabulated_ds=tab,
            occupation_rescale=occupation_rescale,
            target_ngal=TARGET_NGAL,
            fit_case=fit_case,
            data_path=args.data_path,
            rp_min=rp_min,
            rp_max=None,
            param_config=param_config,
            Ac_fiducial=AC_FIDUCIAL,
            **({"max_fsat": args.max_fsat, "hod_occupation": halo.HOD}
               if args.max_fsat is not None else {}),
        )
        print(f"  {fitter.n_bins} bins in [{fitter.rp_obs[0]:.3f}, "
              f"{fitter.rp_obs[-1]:.2f}] Mpc/h, {fitter.n_params} free params")

        if args.postprocess:
            # Reuse an already-sampled chain: recompute Meff/fsat/chi2/plots
            # without re-running Nautilus (the sampling is the expensive part).
            if not os.path.exists(chain_path):
                print(f"  [skip] no saved chain at {chain_path}")
                continue
            saved = np.load(chain_path)
            points, weights, log_l, log_z = (
                saved['points'], saved['weights'],
                saved['log_l'], float(saved['log_z']),
            )
            print(f"  Loaded saved chain: {chain_path} "
                  f"({len(points)} pts, logZ = {log_z:.2f})")
        else:
            print(f"  Running Nautilus (n_live={N_LIVE}, vectorized jit/vmap "
                  "likelihood, single process) ...")
            checkpoint = os.path.join(args.output_dir,
                                      f"checkpoint_{case_name}_rmin{rp_min}.h5")
            points, weights, log_l, log_z = fitter.run(
                n_eff=args.n_eff, n_live=N_LIVE, verbose=True,
                vectorized=True, filepath=checkpoint,
            )
            fitter.save_results(chain_path, points, weights, log_l, log_z)
            print(f"  Chain saved: {chain_path}")

        theta_best = fitter.get_best_fit(points, log_l)
        ds_map = fitter.predict_at_obs(theta_best)
        residual = ds_map - fitter.ds_obs
        chi2 = float(residual @ fitter.cov_inv @ residual)
        chi2_red = chi2 / (len(fitter.ds_obs) - fitter.n_params)
        ds_map_full = fitter.predict_at_obs(theta_best, rp_eval=rp_all)
        Meff, fsat = compute_meff_fsat(occupation_full, theta_best, fitter)
        log10Meff = np.log10(Meff)

        print(f"  chi2_red = {chi2_red:.3f}")
        print(f"  log10(Meff / [Msun/h]) = {log10Meff:.3f}")
        print(f"  fsat = {fsat:.3f}")

        logM_bins, ncen_med, ncen_sig, nsat_med, nsat_sig = \
            compute_hod_profiles_from_chain(points, weights, fitter,
                                            occupation_full, halo)

        bestfits[rp_min] = {
            'ds': ds_map_full, 'chi2_red': chi2_red,
            'Meff': Meff, 'fsat': fsat,
            'ncen_med': ncen_med, 'ncen_sig': ncen_sig,
            'nsat_med': nsat_med, 'nsat_sig': nsat_sig,
        }

        label = (f"$r_{{\\rm min}}={rp_min}$ Mpc/$h$   "
                 f"$\\chi^2_\\nu={chi2_red:.2f}$\n"
                 f"$\\log_{{10}}M_{{\\rm eff}}={log10Meff:.2f}$   "
                 f"$f_{{\\rm sat}}={fsat:.3f}$")
        ax.plot(rp_all, rp_all * ds_map_full, color=color, lw=1.8, label=label)

    ax.set_xscale('log')
    ax.set_xlabel(r'$r_p\;[\mathrm{Mpc}/h]$', fontsize=12)
    ax.set_ylabel(r'$r_p\,\Delta\Sigma\;[M_\odot\,\mathrm{pc}^{-2}\cdot\mathrm{Mpc}/h]$',
                  fontsize=12)
    ax.set_title(f'HOD model: {case_name} (tabulated)', fontsize=13)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.3, which='both', ls=':')
    fig.tight_layout()
    # scope filenames by rp_min when a single cut runs (jobs split per rp_min)
    rp_tag = ("" if len(args.rp_min_values) > 1 else
              f"_rmin{str(args.rp_min_values[0]).replace('.', 'p')}")
    plot_path = os.path.join(args.output_dir, f"fit_{case_name}{rp_tag}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")

    profiles_by_rp_min = {
        rp_min: {k: bestfits[rp_min][k]
                 for k in ('ncen_med', 'ncen_sig', 'nsat_med', 'nsat_sig')}
        for rp_min in bestfits
    }
    plot_hod_profiles(case_name + rp_tag, logM_bins, profiles_by_rp_min,
                      args.output_dir)

    return rp_all, bestfits, logM_bins


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Nautilus DeltaSigma fits with the tabulated forward model")
    p.add_argument("case", nargs="?", choices=ALL_CASE_NAMES,
                   help="HOD case (default: all)")
    p.add_argument("--cache_path", required=True,
                   help="HaloCenterLensingCache .h5 WITH xi_gm tabulation "
                        "(precompute_halo_center_cache.py --tabulate)")
    p.add_argument("--halo_path", default=HALO_PATH_DEFAULT)
    p.add_argument("--data_path", default=DATA_PATH_DEFAULT)
    p.add_argument("--output_dir",
                   default=os.path.join(FLAMINGO_DIR, "chains_TABULATED_DMO"))
    p.add_argument("--n_workers", type=int, default=20,
                   help="Unused (kept for CLI compatibility); sampling is "
                        "single-process vectorized")
    p.add_argument("--n_eff", type=int, default=N_EFF)
    p.add_argument("--rp_min_values", type=float, nargs="+",
                   default=RP_MIN_VALUES)
    p.add_argument("--max_fsat", type=float, default=None)
    p.add_argument("--gaussian_ab", action="store_true")
    p.add_argument("--postprocess", action="store_true",
                   help="Skip sampling: load the saved chain_*.npz and "
                        "(re)compute Meff/fsat/chi2, plots, and the aggregate. "
                        "Use to recover the outputs when a run crashed in "
                        "post-processing after the chains were saved.")
    return p.parse_args()


def main():
    args = parse_args()
    names = [args.case] if args.case else ALL_CASE_NAMES

    cache = HaloCenterLensingCache.load(args.cache_path)
    if not cache.has_tabulation:
        raise SystemExit("Cache has no xi_gm tabulation — regenerate with "
                         "precompute_halo_center_cache.py --tabulate.")

    all_bestfits_arrays = {}
    rp_centers_saved = None
    logM_bins_saved = None

    for case_name in names:
        fit_case = FIT_CASE_OF[_base_name(case_name)]
        print(f"\nLoading halo catalogue for case {case_name} ...")
        halo = build_halo_occupation(fit_case, args.halo_path)
        tab = TabulatedDeltaSigma(cache, halo)
        print(f"  {tab}")

        rp_all, bestfits, logM_bins = run_case(case_name, fit_case, halo,
                                               tab, args)
        if rp_centers_saved is None:
            rp_centers_saved = rp_all
            logM_bins_saved = logM_bins

        for rp_min, vals in bestfits.items():
            prefix = f"{case_name}_rmin{str(rp_min).replace('.', 'p')}"
            all_bestfits_arrays[f"{prefix}_ds"]       = vals['ds']
            all_bestfits_arrays[f"{prefix}_chi2_red"] = np.array(vals['chi2_red'])
            all_bestfits_arrays[f"{prefix}_Meff"]     = np.array(vals['Meff'])
            all_bestfits_arrays[f"{prefix}_fsat"]     = np.array(vals['fsat'])
            all_bestfits_arrays[f"{prefix}_ncen_med"] = vals['ncen_med']
            all_bestfits_arrays[f"{prefix}_ncen_sig"] = vals['ncen_sig']
            all_bestfits_arrays[f"{prefix}_nsat_med"] = vals['nsat_med']
            all_bestfits_arrays[f"{prefix}_nsat_sig"] = vals['nsat_sig']

    if all_bestfits_arrays:
        tag = names[0] if len(names) == 1 else "all"
        if len(args.rp_min_values) == 1:
            tag += f"_rmin{str(args.rp_min_values[0]).replace('.', 'p')}"
        bestfits_path = os.path.join(args.output_dir, f"bestfits_{tag}.npz")
        np.savez(bestfits_path,
                 rp_centers=rp_centers_saved,
                 logM_bins=logM_bins_saved,
                 **all_bestfits_arrays)
        print(f"\nAll bestfits saved: {bestfits_path}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
