#!/usr/bin/env python3
"""
run_emulator_chains.py

For each HOD model complexity (NFW / Extended profile / Conformity, optionally
with assembly bias), runs Nautilus nested sampling at four rp_min scale cuts
on the Flamingo L1000N1800 ELG dataset using pre-trained emulators.

Usage:
    python run_emulator_chains.py                       # all default cases
    python run_emulator_chains.py NFW                   # one case
    python run_emulator_chains.py --full_bb             # full-BB emulator
    python run_emulator_chains.py --full_bb --subhalo   # subhalo-placement full-BB
    python run_emulator_chains.py --max_fsat 0.2 NFW    # f_sat truncation prior
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HOD_NRV.utilsf.numerical_sampler import EmulatorFitter, FitCase
from HOD_NRV.utilsf.emulator_nn_flax import predict_dsigma
from HOD_NRV.HOD_numerical.HOD_models import Occupation, rescale_Ac_to_target_ngal
from HOD_NRV.HOD_numerical.HOD import HaloOccupation


# ============================================================================
# Paths & global configuration
# ============================================================================

FLAMINGO_DIR = "/sps/euclid/Users/rpaviot/flamingo"
DATA_PATH    = os.path.join(FLAMINGO_DIR, "data_bin0_finalbinning.npz")
OUTPUT_DIR   = os.path.join(FLAMINGO_DIR, "chains_subhalo_AB")

HALO_PATH = os.path.join(FLAMINGO_DIR, "snapshots_DMO/host_catalogue_ab.parquet")

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


# ============================================================================
# Single source of truth for priors
# ============================================================================

# Uniform prior ranges. Every free parameter referenced anywhere lives here.
PRIOR_RANGES = {
    "As":         (0.002, 0.08),
    "Mmin":       (11.1,  13.0),
    "sig_M":      (0.05,  1.5),
    "gamma":      (0.1,   10.0),
    "alpha":      (0.05,  1.5),
    "kappa":      (0.1,   2.0),     # standard satellite cutoff
    "Mcut":       (11.5,  13.5),    # ELG satellite cutoff (replaces kappa)
    "lambda_NFW": (0.05,  1.0),
    "f_exp":      (0.0,   0.9),
    "tau":        (1.0,   10.0),
    "kappa_EE":   (0.05,  1.0),
    "B_cent":     (-0.5,  0.5),
    "B_sat":      (-0.5,  0.5),
}

# Parameters that are always fixed (matching training conditions).
FIXED_DEFAULTS = {
    "M1":     M1_FIXED,
    "Mmax":   MMAX_FIXED,
    "A_cent": 0.0,
    "A_sat":  0.0,
}


def build_param_config(fit_case, *, elg_satellite, assembly_bias,
                       subhalo=False, gaussian_ab=False, overrides=None):
    """Return a {name: (low, high) | (mu, sig, 'gaussian') | scalar} dict
    consumable by EmulatorFitter.

    Active parameters are pulled from PRIOR_RANGES; everything else is fixed.
    """
    cfg = dict(FIXED_DEFAULTS)
    active = ["As", "Mmin", "sig_M", "gamma", "alpha"]
    active.append("Mcut" if elg_satellite else "kappa")

    if subhalo:
        cfg["lambda_NFW"] = 1.0
    else:
        active.append("lambda_NFW")

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

    if overrides:
        cfg.update(overrides)
    return cfg


# ============================================================================
# Emulator path / case selection
# ============================================================================

def _emu(subdir):
    return os.path.join(FLAMINGO_DIR, subdir, "emulator_dsigma.npz")

FIT_CASE_OF = {
    "NFW":  FitCase.STANDARD_NFW,
    "EXT":  FitCase.EXTENDED_PROFILE,
    "CONF": FitCase.CONFORMITY,
}

ALL_CASE_NAMES = ["NFW", "EXT", "CONF", "NFW_AB", "EXT_AB", "CONF_AB"]


def _base_name(case_name):
    return case_name[:-3] if case_name.endswith("_AB") else case_name


def select_cases(*, full_bb, subhalo, elg_satellite, requested):
    """Return {case_name: (emu_path, fit_case)} for the requested mode."""
    if full_bb and subhalo:
        path_for = lambda _: _emu("emulator_grid_FULL_SUBHALO_BB")
        names = ["NFW", "NFW_AB"]
    elif full_bb:
        path_for = lambda _: _emu("emulator_grid_FULL_BB")
        names = ALL_CASE_NAMES
    elif subhalo:
        path_for = lambda _: _emu("emulator_grid_FULL_SUBHALO_BB")
        names = ["NFW"]
    elif elg_satellite:
        names = ALL_CASE_NAMES
        path_for = lambda n: _emu(
            f"emulator_grid_{_base_name(n)}_NEWSAT" + ("_AB" if n.endswith("_AB") else "")
        )
    else:
        names = ["NFW", "EXT", "CONF"]
        path_for = lambda n: _emu(f"emulator_grid_{n}")

    cases = {n: (path_for(n), FIT_CASE_OF[_base_name(n)]) for n in names}
    if requested:
        if requested not in cases:
            raise SystemExit(f"ERROR: case {requested!r} not available in this mode")
        cases = {requested: cases[requested]}
    return cases


# ============================================================================
# HaloOccupation builder
# ============================================================================

def build_halo_occupation(fit_case):
    """Return a HaloOccupation matching the emulator-training pipeline."""
    halo = HaloOccupation(
        cosmology=COSMO_PARAMS,
        zeff=ZEFF,
        Lbox=LBOX,
        column_mapping=COLUMN_MAPPING,
        mass_definition=MASS_DEFINITION,
        halo_path=HALO_PATH,
        DataFrame_part=None,
        apply_rsd=False,
        do_test=False,
    )
    halo.set_halo_model(
        "ELG_mHMQ",
        conformity=(fit_case == FitCase.CONFORMITY),
        elg_satellite=True,
    )
    return halo


# ============================================================================
# Helpers: prediction, chi2, M_eff/fsat, posterior HOD profiles
# ============================================================================

def map_prediction(fitter, theta_best, rp_eval=None):
    """ΔΣ at MAP, interpolated onto rp_eval (defaults to fitter.rp_obs)."""
    if rp_eval is None:
        rp_eval = fitter.rp_obs
    theta_vec = np.array([
        fitter.fixed_params_dict[p] if p in fitter.fixed_params_dict else theta_best[p]
        for p in fitter.emulator_param_order
    ], dtype=np.float32)
    ds_pred = np.asarray(predict_dsigma(fitter.model, fitter.norm_stats, theta_vec))
    log_ds = np.interp(np.log(rp_eval), np.log(fitter.emulator_rp),
                       np.log(np.maximum(ds_pred, 1e-30)))
    return np.exp(log_ds)


def reduced_chi2(fitter, ds_map):
    residual = ds_map - fitter.ds_obs
    chi2 = float(residual @ fitter.cov_inv @ residual)
    return chi2, chi2 / (len(fitter.ds_obs) - fitter.n_params)


def _make_occupation(halo, fit_case):
    return Occupation(
        "ELG_mHMQ", halo.logM_bins, halo.mass_function,
        conformity=(fit_case == FitCase.CONFORMITY),
        elg_satellite=True,
    )


def _full_rescaled_params(occupation, theta_dict, fixed_dict):
    """Merge free + fixed params and rescale (Ac, As) to TARGET_NGAL."""
    merged = {**theta_dict, **fixed_dict}
    merged.pop("Ac", None)
    Ac, As = rescale_Ac_to_target_ngal(occupation, merged, TARGET_NGAL,
                                       Ac_fiducial=AC_FIDUCIAL)
    return {**merged, "Ac": Ac, "As": As}


def compute_meff_fsat(halo, theta_best, fitter, fit_case):
    occupation = _make_occupation(halo, fit_case)
    full = _full_rescaled_params(occupation, theta_best, fitter.fixed_params_dict)
    return occupation.compute_Meff(full)[0], occupation.compute_fsat(full)[0]


def compute_hod_profiles_from_chain(points, weights, fitter, halo, fit_case,
                                    n_samples=N_PROFILE_SAMPLES):
    occupation = _make_occupation(halo, fit_case)

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
        full = _full_rescaled_params(occupation, theta_dict, fitter.fixed_params_dict)
        nc, ns = occupation.compute_HOD_occupation(logM, full)
        ncen_all[i] = np.asarray(nc)
        nsat_all[i] = np.asarray(ns)

    def _med_sig(arr):
        q16, q50, q84 = np.percentile(arr, [16, 50, 84], axis=0)
        return q50, (q84 - q16) / 2.0

    ncen_med, ncen_sig = _med_sig(ncen_all)
    nsat_med, nsat_sig = _med_sig(nsat_all)
    return logM, ncen_med, ncen_sig, nsat_med, nsat_sig


# ============================================================================
# Plotting
# ============================================================================

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

    fig.suptitle(f'HOD occupation — {case_name}', fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, f"hod_profiles_{case_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  HOD profile plot saved: {path}")


# ============================================================================
# Run one case (all rp_min values)
# ============================================================================

def run_case(case_name, emulator_path, fit_case, halo, *,
             full_bb=False, subhalo=False, max_fsat=None):
    print(f"\n{'='*60}\n  Case: {case_name}  ({fit_case.name})\n{'='*60}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assembly_bias = case_name.endswith("_AB")
    gaussian_ab = full_bb  # full-BB historical default

    data = np.load(DATA_PATH)
    rp_all = data['rp_centers']
    ds_all = data['delta_sigma']
    ds_err = data['delta_sigma_err']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(rp_all, rp_all * ds_all, yerr=rp_all * ds_err,
                fmt='ko', ms=4, zorder=10, label='Flamingo ELG data')

    bestfits = {}

    for rp_min, color in zip(RP_MIN_VALUES, COLORS):
        print(f"\n  rp_min = {rp_min} Mpc/h")
        chain_path = os.path.join(OUTPUT_DIR, f"chain_{case_name}_rmin{rp_min}.npz")

        param_config = build_param_config(
            fit_case,
            elg_satellite=True,
            assembly_bias=assembly_bias,
            subhalo=subhalo,
            gaussian_ab=gaussian_ab,
        )

        fitter_kwargs = dict(
            emulator_path=emulator_path,
            data_path=DATA_PATH,
            rp_min=rp_min,
            rp_max=None,
            fit_case=fit_case,
            param_config=param_config,
        )
        if max_fsat is not None:
            fitter_kwargs.update(
                max_fsat=max_fsat,
                hod_occupation=halo.HOD,
                Ac_fiducial=AC_FIDUCIAL,
            )

        fitter = EmulatorFitter(**fitter_kwargs)
        print(f"  {fitter.n_bins} bins in [{fitter.rp_obs[0]:.3f}, "
              f"{fitter.rp_obs[-1]:.2f}] Mpc/h, {fitter.n_params} free params"
              + (f", max_fsat={max_fsat}" if max_fsat is not None else ""))

        print(f"  Running Nautilus (n_live={N_LIVE}) ...")
        points, weights, log_l, log_z = fitter.run(
            n_eff=N_EFF, n_live=N_LIVE, verbose=True,
        )
        fitter.save_results(chain_path, points, weights, log_l, log_z)
        print(f"  Chain saved: {chain_path}")

        theta_best = fitter.get_best_fit(points, log_l)
        ds_map = map_prediction(fitter, theta_best)
        _, chi2_red = reduced_chi2(fitter, ds_map)
        ds_map_full = map_prediction(fitter, theta_best, rp_eval=rp_all)
        Meff, fsat = compute_meff_fsat(halo, theta_best, fitter, fit_case)
        log10Meff = np.log10(Meff)

        print(f"  chi2_red = {chi2_red:.3f}")
        print(f"  log10(Meff / [Msun/h]) = {log10Meff:.3f}")
        print(f"  fsat = {fsat:.3f}")

        print(f"  Computing N_cen/N_sat profiles from {N_PROFILE_SAMPLES} posterior draws ...")
        logM_bins, ncen_med, ncen_sig, nsat_med, nsat_sig = \
            compute_hod_profiles_from_chain(points, weights, fitter, halo, fit_case)

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
    ax.set_title(f'HOD model: {case_name}', fontsize=13)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.3, which='both', ls=':')

    plot_path = os.path.join(OUTPUT_DIR, f"fit_{case_name}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")

    profiles_by_rp_min = {
        rp_min: {k: bestfits[rp_min][k]
                 for k in ('ncen_med', 'ncen_sig', 'nsat_med', 'nsat_sig')}
        for rp_min in RP_MIN_VALUES
    }
    plot_hod_profiles(case_name, logM_bins, profiles_by_rp_min, OUTPUT_DIR)

    return rp_all, bestfits, logM_bins


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Nautilus nested sampling with pre-trained ΔΣ emulators",
    )
    p.add_argument("case", nargs="?", choices=ALL_CASE_NAMES,
                   help="HOD case to run (default: all available)")
    p.add_argument("--elg_satellite", type=lambda x: bool(int(x)), default=0,
                   metavar="{0,1}",
                   help="Use ELG satellite cutoff (1) vs standard kappa (0). "
                        "Ignored when --subhalo or --full_bb is set.")
    p.add_argument("--subhalo", action="store_true",
                   help="Use the subhalo-placement emulator (lambda_NFW fixed to 1).")
    p.add_argument("--full_bb", action="store_true",
                   help="Use the single full-BB emulator covering CONF+AB space "
                        "(or FULL_SUBHALO_BB with --subhalo).")
    p.add_argument("--max_fsat", type=float, default=None,
                   help="Apply the f_sat truncation prior (mirrors the rejection "
                        "applied at grid-build time). Reject samples with "
                        "f_sat > MAX_FSAT.")
    return p.parse_args()


def main():
    args = parse_args()
    requested = args.case.upper() if args.case else None
    use_subhalo = args.subhalo
    use_full_bb = args.full_bb
    elg_satellite = True if (use_subhalo or use_full_bb) else args.elg_satellite

    cases_to_run = select_cases(
        full_bb=use_full_bb,
        subhalo=use_subhalo,
        elg_satellite=elg_satellite,
        requested=requested,
    )

    all_bestfits_arrays = {}
    rp_centers_saved = None
    logM_bins_saved  = None

    for case_name, (emu_path, fit_case) in cases_to_run.items():
        print(f"\nLoading halo catalogue for case {case_name} ...")
        halo = build_halo_occupation(fit_case)
        print(f"  logM in [{halo.logM_bins[0]:.2f}, {halo.logM_bins[-1]:.2f}], "
              f"{len(halo.logM_bins)} points")

        rp_all, bestfits, logM_bins = run_case(
            case_name, emu_path, fit_case, halo,
            full_bb=use_full_bb,
            subhalo=use_subhalo,
            max_fsat=args.max_fsat,
        )

        if rp_centers_saved is None:
            rp_centers_saved = rp_all
        if logM_bins_saved is None:
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
        bestfits_path = os.path.join(OUTPUT_DIR, "all_bestfits.npz")
        np.savez(bestfits_path,
                 rp_centers=rp_centers_saved,
                 logM_bins=logM_bins_saved,
                 **all_bestfits_arrays)
        print(f"\nAll bestfits saved: {bestfits_path}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
