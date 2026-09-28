"""
Semi-analytical joint DeltaSigma + w_gg fit of the hydro NISP (in-R200m) data
vector, with beta^NL, and optionally free concentration rescalings f_h (matter)
and f_s (satellites) -- Dvornik+23 eq. 15. The question: does f_h absorb the
baryonic feedback the DMO-based halo model lacks?

Same data, covariance (no cross term), n_gal and scale cuts as the tabulated
chains in chains_TABULATED_*_R1_inR200m; w_gg uses the measurement's pi_max.

    python internal/run_analytical_fh_fit.py --rp_min_ds 0.1 --free_f
    python internal/run_analytical_fh_fit.py --rp_min_ds 0.5 --free_f --minuit
    python internal/run_analytical_fh_fit.py --test --no_beta_nl   # laptop smoke test
"""
import argparse
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from HOD_NRV.HOD_analytical.halo_model import HaloModel
from HOD_NRV.HOD_analytical.analytical_sampler import AnalyticalHODFitter

FLAM = "/sps/euclid/Users/rpaviot/flamingo"

# Flamingo D3A (as in example_scripts/cross_check_analytical_numerical.py)
COSMO = {"h": 0.681, "Omc": 0.306 - 0.0486 - 1.39e-3, "Omb": 0.0486,
         "A_s": 2.099e-9, "n_s": 0.967, "Omnu": 1.39e-3}
BETA_NL = {"n_k": 50, "n_mass": 30, "k_min": 1e-2, "k_max": 30.0,
           "log_M_min": 11.5, "log_M_max": 15.0, "method": "linear",
           "bias_method": "halo-halo", "force_to_zero": "additive",
           "k_lin": 0.02, "constant_low": False, "verbose": True}

# Hydro truth: HOD-form fit of the measured occupation (hydro_hod_form_fits_inR200m.npz,
# free_cen_ELG_mHMQ + free_sat_HOD_satellite) and the measured fsat / Meff(cen).
TRUTH = {"log10Mmin": 11.626, "sig_M": 0.599, "gamma": 4.08, "log10M1": 13.04,
         "alpha": 0.859, "kappa": 1.01, "fsat": 0.2002, "log10Meff": 11.935}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{FLAM}/hydro_measured_tabbins_inR200m.npz")
    ap.add_argument("--cov", default=f"{FLAM}/tabbins_newcov_inR200m.npz")
    ap.add_argument("--out_dir", default=f"{FLAM}/chains_ANALYTICAL_inR200m")
    ap.add_argument("--rp_min_ds", type=float, default=0.1)
    ap.add_argument("--rp_min_wgg", type=float, default=0.1)
    ap.add_argument("--no_wgg", action="store_true")
    ap.add_argument("--free_f", action="store_true", help="free f_h and f_s")
    ap.add_argument("--no_beta_nl", action="store_true")
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_eff", type=int, default=5000)
    ap.add_argument("--n_derived", type=int, default=400)
    ap.add_argument("--minuit", action="store_true",
                    help="MIGRAD/HESSE best fit instead of a Nautilus chain")
    ap.add_argument("--n_starts", type=int, default=3)
    ap.add_argument("--test", action="store_true",
                    help="evaluate chi2 at the truth HOD and time it; no sampling")
    args = ap.parse_args()

    d, c = np.load(args.data), np.load(args.cov)
    rp, edges = d["rp_centers"], d["rp_bins"]
    ngal = float(c["n_gal"])
    pi_max = float(d["pi_max"])

    t0 = time.time()
    model = HaloModel(COSMO, z=float(c["z_eff"]), hod_type="ELG_MHMQ",
                      units_per_h=True, mass_definition="MassDef200m",
                      include_beta_nl=not args.no_beta_nl, beta_nl_kwargs=BETA_NL,
                      verbose=True)
    if not args.no_beta_nl and len(model._beta_nl_gl_cache) == 0:
        raise RuntimeError("beta^NL caches empty (dark_emulator / interpax missing)")
    print(f"HaloModel built in {time.time() - t0:.0f} s")

    # As is relative to Ac_fiducial = 1: the truth ratio As/Ac ~ 2.6 (fsat 0.26 at the
    # truth HOD on the Tinker10 mass function); the fitter's default (0.001, 0.1)
    # caps fsat below 0.01. (0.05, 10) spans fsat ~ 0.01-0.6.
    cfg = {"As": (0.05, 10.0)}
    if args.free_f:
        cfg.update({"f_h": (0.1, 2.0), "f_s": (0.1, 1.0)})
    wgg_kw = {} if args.no_wgg else dict(
        rp_obs_wgg=rp, wgg_obs=d["wgg"], cov_wgg=c["cov_wgg"],
        rp_min_wgg=args.rp_min_wgg)
    fitter = AnalyticalHODFitter(
        model, ngal, rp, d["delta_sigma"], c["cov_delta_sigma"],
        rp_min=args.rp_min_ds, param_config=cfg, pi_max=pi_max, **wgg_kw)

    tag = (f"mHMQ{'_bnl' if not args.no_beta_nl else ''}"
           f"{'_fhfs' if args.free_f else ''}{'' if args.no_wgg else '_wggjoint'}"
           f"_rmin{args.rp_min_ds:g}")

    if args.test:
        start = {n: TRUTH[n] for n in fitter.free_names if n in TRUTH}
        start.update({"As": 2.55, "f_h": 1.0, "f_s": 1.0})
        start = {n: start[n] for n in fitter.free_names}
        t = time.time()
        chi2 = fitter.chi2(start)
        print(f"chi2 at truth HOD = {chi2:.1f} / {fitter.n_data}  "
              f"({time.time() - t:.2f} s/call)  derived {fitter.derived_quantities(start)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if args.minuit:
        run_minuit(args, fitter, model, d, rp, edges, pi_max, ngal, tag)
        return
    t = time.time()
    res = fitter.run(n_live=args.n_live, n_eff=args.n_eff, n_workers=1, verbose=True)
    print(f"sampling took {(time.time() - t) / 3600:.2f} h, log Z = {res.sampler.log_z:.2f}")

    names = fitter.free_names
    post = res.posterior
    P = np.column_stack([post[n] for n in names])
    w = np.exp(post["log_w"] - post["log_w"].max())
    w /= w.sum()

    # Derived fsat / Meff on a weighted resample of the posterior
    rng = np.random.default_rng(1)
    idx = rng.choice(len(w), size=args.n_derived, p=w)
    der = {"fsat": [], "log10Meff": []}
    for i in idx:
        free = dict(zip(names, P[i]))
        full, f_h, f_s = fitter._build_params(free)
        model.update_f(f_h=f_h, f_s=f_s)
        model.set_hod_params(full)
        der["fsat"].append(float(np.asarray(model.satellite_fraction())))
        der["log10Meff"].append(float(np.log10(np.asarray(model.effective_halo_mass()))))

    # Best-fit model over the full rp range (scale cuts shown, not applied)
    full, f_h, f_s = fitter._build_params(res.best_fit)
    model.update_f(f_h=f_h, f_s=f_s)
    model.set_hod_params(full)
    _, ds_bf = model.DeltaSigma(rp, rp_bins=edges, method="direct", include_stellar=False)
    _, wgg_bf = model.wgg(rp, rp_bins=edges, pi_max=pi_max)
    bf_fsat = float(np.asarray(model.satellite_fraction()))
    bf_meff = float(np.log10(np.asarray(model.effective_halo_mass())))

    def q(x, wt):
        i = np.argsort(x)
        cw = np.cumsum(wt[i])
        cw /= cw[-1]
        return np.interp([0.16, 0.5, 0.84], cw, x[i])

    print(f"\n==== {tag} ====")
    print(f"best-fit chi2 = {res.chi2:.2f} / {fitter.n_data} data, {len(names)} free "
          f"(chi2_red {res.chi2 / res.ndof:.3f});  log Z = {res.sampler.log_z:.2f}")
    print(f"{'param':>10s} {'median':>9s} {'-1sig':>8s} {'+1sig':>8s} {'best':>9s} {'truth':>8s}")
    for j, n in enumerate(names):
        lo, me, hi = q(P[:, j], w)
        print(f"{n:>10s} {me:9.3f} {me - lo:8.3f} {hi - me:8.3f} "
              f"{res.best_fit[n]:9.3f} {TRUTH.get(n, np.nan):8.3f}")
    for n, bfv in (("fsat", bf_fsat), ("log10Meff", bf_meff)):
        lo, me, hi = np.percentile(der[n], [16, 50, 84])
        print(f"{n:>10s} {me:9.3f} {me - lo:8.3f} {hi - me:8.3f} {bfv:9.3f} {TRUTH[n]:8.3f}")
    print("\n  rp       DS model/data-1 [%]   wgg model/data-1 [%]")
    for i in range(len(rp)):
        print(f"{rp[i]:7.3f}   {100 * (ds_bf[i] / d['delta_sigma'][i] - 1):8.2f}"
              f"            {100 * (wgg_bf[i] / d['wgg'][i] - 1):8.2f}")

    np.savez(os.path.join(args.out_dir, f"chain_{tag}.npz"),
             points=P, log_w=post["log_w"], log_l=post["log_l"], param_names=names,
             log_z=res.sampler.log_z, best_fit=np.array([res.best_fit[n] for n in names]),
             chi2=res.chi2, ndof=res.ndof, n_data=fitter.n_data,
             rp=rp, ds_data=d["delta_sigma"], wgg_data=d["wgg"],
             ds_bf=ds_bf, wgg_bf=wgg_bf, bf_fsat=bf_fsat, bf_log10Meff=bf_meff,
             derived_fsat=np.array(der["fsat"]),
             derived_log10Meff=np.array(der["log10Meff"]),
             rp_min_ds=args.rp_min_ds, rp_min_wgg=args.rp_min_wgg, pi_max=pi_max,
             target_ngal=ngal, beta_nl=not args.no_beta_nl)
    print("saved", os.path.join(args.out_dir, f"chain_{tag}.npz"))


def run_minuit(args, fitter, model, d, rp, edges, pi_max, ngal, tag):
    """MIGRAD + HESSE from the truth HOD and from ``--n_starts - 1`` random
    starts inside the priors; keeps the lowest chi2. fsat / Meff errors come
    from draws of the HESSE covariance (clipped to the priors)."""
    names = fitter.free_names
    lims = {n: (a, b) for n, a, b, _ in fitter.priors}
    truth_start = {n: TRUTH.get(n, 0.5 * sum(lims[n])) for n in names}
    truth_start.update({k: v for k, v in (("As", 2.55), ("f_h", 1.0), ("f_s", 0.9))
                        if k in names})
    rng = np.random.default_rng(2)
    starts = [truth_start] + [
        {n: rng.uniform(*lims[n]) if n in ("f_h", "f_s") else
            np.clip(truth_start[n] + 0.15 * (lims[n][1] - lims[n][0]) * rng.normal(),
                    *lims[n])
         for n in names}
        for _ in range(args.n_starts - 1)]

    best = None
    for k, s in enumerate(starts):
        t = time.time()
        res = fitter.minimize(start=s)
        m = res.sampler
        print(f"start {k}: chi2 {res.chi2:.2f}  valid {m.valid}  nfcn {m.nfcn}  "
              f"({(time.time() - t) / 60:.1f} min)  "
              + " ".join(f"{n}={res.best_fit[n]:.3f}" for n in names), flush=True)
        if best is None or res.chi2 < best.chi2:
            best = res
    m = best.sampler
    m.hesse()
    bf = best.best_fit
    err = {n: float(m.errors[n]) for n in names}
    cov = np.array(m.covariance)

    full, f_h, f_s = fitter._build_params(bf)
    model.update_f(f_h=f_h, f_s=f_s)
    model.set_hod_params(full)
    _, ds_bf = model.DeltaSigma(rp, rp_bins=edges, method="direct", include_stellar=False)
    _, wgg_bf = model.wgg(rp, rp_bins=edges, pi_max=pi_max)
    bf_fsat = float(np.asarray(model.satellite_fraction()))
    bf_meff = float(np.log10(np.asarray(model.effective_halo_mass())))

    x0 = np.array([bf[n] for n in names])
    draws = rng.multivariate_normal(x0, cov, size=args.n_derived)
    lo_b = np.array([lims[n][0] for n in names])
    hi_b = np.array([lims[n][1] for n in names])
    der = {"fsat": [], "log10Meff": []}
    for x in np.clip(draws, lo_b, hi_b):
        built = fitter._build_params(dict(zip(names, x)))
        if built is None:
            continue
        model.set_hod_params(built[0])
        der["fsat"].append(float(np.asarray(model.satellite_fraction())))
        der["log10Meff"].append(float(np.log10(np.asarray(model.effective_halo_mass()))))

    print(f"\n==== {tag} (minuit) ====")
    print(f"chi2 = {best.chi2:.2f} / {fitter.n_data} data, {len(names)} free "
          f"(chi2_red {best.chi2 / best.ndof:.3f}); valid {m.valid}")
    print(f"{'param':>10s} {'best':>9s} {'hesse':>8s} {'truth':>8s}")
    for n in names:
        print(f"{n:>10s} {bf[n]:9.3f} {err[n]:8.3f} {TRUTH.get(n, np.nan):8.3f}")
    for n, v in (("fsat", bf_fsat), ("log10Meff", bf_meff)):
        print(f"{n:>10s} {v:9.3f} {np.std(der[n]):8.3f} {TRUTH[n]:8.3f}")
    print("\n  rp       DS model/data-1 [%]   wgg model/data-1 [%]")
    for i in range(len(rp)):
        print(f"{rp[i]:7.3f}   {100 * (ds_bf[i] / d['delta_sigma'][i] - 1):8.2f}"
              f"            {100 * (wgg_bf[i] / d['wgg'][i] - 1):8.2f}")

    out = os.path.join(args.out_dir, f"minuit_{tag}.npz")
    np.savez(out, param_names=names, best_fit=x0, errors=[err[n] for n in names],
             covariance=cov, chi2=best.chi2, ndof=best.ndof, n_data=fitter.n_data,
             valid=m.valid, rp=rp, ds_data=d["delta_sigma"], wgg_data=d["wgg"],
             ds_bf=ds_bf, wgg_bf=wgg_bf, bf_fsat=bf_fsat, bf_log10Meff=bf_meff,
             fsat_err=np.std(der["fsat"]), log10Meff_err=np.std(der["log10Meff"]),
             rp_min_ds=args.rp_min_ds, rp_min_wgg=args.rp_min_wgg, pi_max=pi_max,
             target_ngal=ngal, beta_nl=not args.no_beta_nl)
    print("saved", out)


if __name__ == "__main__":
    main()
