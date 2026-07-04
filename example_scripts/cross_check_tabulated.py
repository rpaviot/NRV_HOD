"""
Cross-check: TabCorr-style tabulated DeltaSigma vs Monte-Carlo population.

Compares TabulatedDeltaSigma.predict() (occupation-weighted sums over the
xi_gm-tabulated halo-center cache + analytic satellite offset convolution)
against the existing MC pipeline (populate_haloes + pycorr satellites via
compute_galaxy_lensing_optimized), on a random halo subsample of the local
Flamingo L1000N1800 catalog. The per-galaxy DeltaSigma expectation is
invariant under halo subsampling, so a small subsample gives an unbiased
comparison at tractable cache-precompute cost.

Usage::

    python cross_check_tabulated.py                       # ~114k halos
    python cross_check_tabulated.py --assembly_bias       # + fs_norm AB
    python cross_check_tabulated.py --halo_fraction 0.02  # faster smoke test

The tabulated cache is saved and reused between runs (delete it after
changing --halo_fraction / --particle_fraction / bin counts).
"""

import argparse
import os
import time
import numpy as np

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800_AB.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

LBOX = 681.0
ZEFF = 1.0
MASS_DEFINITION = "MassDef200m"
COLUMN_MAPPING = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}
COSMO_PARAMS = {
    "h": 0.681,
    "Omc": 0.306 - 0.0486 - 1.39e-3,
    "Omb": 0.0486,
    "A_s": 2.099e-9,
    "n_s": 0.967,
    "Omnu": 1.39e-3,
}
RP_BINS = np.geomspace(0.1, 50.0, 16)

BASE_PARAMS = {
    "Ac": 0.05, "Mmin": 11.8, "sig_M": 0.35, "gamma": 3.0,
    "As": 0.02, "M1": 13.0, "alpha": 0.9, "kappa": 0.7,
}

CASES = {
    "NFW": {},
    "EXT_compact": {"f_exp": 0.5, "tau": 4.0, "lambda_NFW": 0.7},
    "EXT_diffuse": {"f_exp": 0.8, "tau": 8.0, "lambda_NFW": 1.5},
}
AB_CASE = {"A_cent": 0.3, "A_sat": -0.3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--halo_fraction", type=float, default=0.05)
    p.add_argument("--particle_fraction", type=float, default=0.05)
    p.add_argument("--n_real", type=int, default=8,
                   help="MC realizations per case")
    p.add_argument("--n_logM_bins", type=int, default=40)
    p.add_argument("--n_fI_bins", type=int, default=8)
    p.add_argument("--assembly_bias", action="store_true",
                   help="Enable fs_norm assembly bias (adds an AB test case)")
    p.add_argument("--cache_path", type=str, default=None,
                   help="Default: tabulated_cache_check[_AB].h5 in output_dir")
    p.add_argument("--output_dir", type=str, default="tabulated_crosscheck")
    p.add_argument("--wgg", action="store_true",
                   help="Cross-check tabulated wgg instead of DeltaSigma")
    p.add_argument("--jax", action="store_true",
                   help="Validate the pure-JAX predict/loglike (make_predict_jax"
                        " + build_batched_loglike) against the NumPy path")
    p.add_argument("--jax_n_points", type=int, default=64,
                   help="Random prior draws per case for --jax validation")
    p.add_argument("--n_logM_bins_wgg", type=int, default=24,
                   help="24 validated; 16 leaves ~6% cc binning errors")
    p.add_argument("--n_fI_bins_wgg", type=int, default=8)
    return p.parse_args()


PI_BINS = np.linspace(0.0, 60.0, 61)


# ── --jax validation: chains' prior structure (run_tabulated_chains.py) ──
JAX_PRIORS = {
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
JAX_FIXED = {"M1": 13.0, "Mmax": 15.0, "A_cent": 0.0, "A_sat": 0.0}
JAX_TARGET_NGAL = 2.3e-4
JAX_AC_FIDUCIAL = 0.01


def _jax_param_config(case, assembly_bias):
    base = ["As", "Mmin", "sig_M", "gamma", "alpha", "Mcut", "lambda_NFW"]
    cfg = dict(JAX_FIXED)
    if case == "NFW":
        free = base
        cfg["f_exp"], cfg["tau"], cfg["kappa_EE"] = 0.0, 5.0, 1.0
    elif case == "EXT":
        free = base + ["f_exp", "tau"]
        cfg["kappa_EE"] = 1.0
    else:  # CONF
        free = base + ["f_exp", "tau", "kappa_EE"]
    if assembly_bias:
        free = free + ["B_cent", "B_sat"]
    else:
        cfg["B_cent"], cfg["B_sat"] = 0.0, 0.0
    for name in free:
        cfg[name] = JAX_PRIORS[name]
    return cfg


def run_jax_check(args, halo, cache):
    """Validate make_predict_jax + build_batched_loglike vs the NumPy path."""
    from HOD_NRV.HOD_numerical.HOD_models import Occupation
    from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
        TabulatedDeltaSigma,
    )
    from HOD_NRV.utilsf.numerical_sampler import TabulatedFitter, FitCase

    fit_case_of = {"NFW": FitCase.STANDARD_NFW, "EXT": FitCase.EXTENDED_PROFILE,
                   "CONF": FitCase.CONFORMITY}
    cases = ["NFW", "EXT", "CONF"]
    rng = np.random.default_rng(7)
    all_pass = True

    for case in cases:
        conformity = case == "CONF"
        halo.set_halo_model("ELG_mHMQ", conformity=conformity,
                            elg_satellite=True)
        tab = TabulatedDeltaSigma(cache, halo)
        occ_rescale = Occupation(
            "ELG_mHMQ", halo.logM_bins, halo.mass_function,
            conformity=conformity, elg_satellite=True)

        cfg = _jax_param_config(case, args.assembly_bias)

        # Synthetic data: NumPy prediction at prior midpoint, 5% diag errors
        mid = {k: 0.5 * (v[0] + v[1]) for k, v in cfg.items()
               if isinstance(v, tuple)}
        fitter0 = TabulatedFitter(
            tabulated_ds=tab, occupation_rescale=occ_rescale,
            target_ngal=JAX_TARGET_NGAL, fit_case=fit_case_of[case],
            ds_obs=np.ones(len(cache.rp_centers)),
            cov_inv=np.eye(len(cache.rp_centers)),
            rp_obs=np.asarray(cache.rp_centers),
            param_config=cfg, Ac_fiducial=JAX_AC_FIDUCIAL,
        )
        _, ds_ref, _ = tab.predict(fitter0.full_params(mid))
        err = 0.05 * np.abs(ds_ref) + 1e-3
        fitter = TabulatedFitter(
            tabulated_ds=tab, occupation_rescale=occ_rescale,
            target_ngal=JAX_TARGET_NGAL, fit_case=fit_case_of[case],
            ds_obs=ds_ref, cov_inv=np.diag(1.0 / err**2),
            rp_obs=np.asarray(cache.rp_centers),
            rp_min=0.2, param_config=cfg, Ac_fiducial=JAX_AC_FIDUCIAL,
        )

        bounds = np.array([(a, b) for _, a, b, _ in fitter.free_params])
        pts = rng.uniform(bounds[:, 0], bounds[:, 1],
                          size=(args.jax_n_points, len(bounds)))

        # ── predict-level parity on a subset ──
        predict_fn = tab.make_predict_jax()
        dev_ds, dev_ngal, dev_fsat = [], [], []
        t_np = 0.0
        for theta in pts[:16]:
            full = fitter.full_params(dict(zip(fitter.param_names, theta)))
            t0 = time.time()
            _, ds_np, info = tab.predict(full)
            t_np += time.time() - t0
            ds_j, ngal_j, fsat_j = (np.asarray(x) for x in predict_fn(full))
            dev_ds.append(np.max(np.abs(ds_j / ds_np - 1)))
            dev_ngal.append(abs(ngal_j / info['ngal'] - 1))
            dev_fsat.append(abs(fsat_j / info['fsat'] - 1))
        t_np /= 16

        # ── loglike-level parity on all points ──
        loglike_fn, free_names = fitter.build_batched_loglike()
        assert free_names == fitter.param_names
        t0 = time.time()
        ll_jax = loglike_fn(pts)          # includes compile
        t_compile = time.time() - t0
        t0 = time.time()
        ll_jax = loglike_fn(pts)
        t_jax = (time.time() - t0) / len(pts)
        ll_np = np.array([fitter.log_likelihood(p) for p in pts])
        d_ll = np.abs(ll_jax - ll_np)
        # compare rejected points only by agreement of the rejection
        both_ok = (ll_np > -1e99) & (ll_jax > -1e99)
        agree_rej = np.array_equal(ll_np > -1e99, ll_jax > -1e99)

        ok = (max(dev_ds) < 5e-3 and max(dev_fsat) < 5e-3
              and np.max(d_ll[both_ok], initial=0.0) < 0.5 and agree_rej)
        all_pass &= ok
        ab_tag = " +AB" if args.assembly_bias else ""
        print(f"\n=== jax {case}{ab_tag} ===  [{'PASS' if ok else 'FAIL'}]")
        print(f"  predict : max|ds dev|={max(dev_ds):.2e}  "
              f"ngal dev={max(dev_ngal):.2e}  fsat dev={max(dev_fsat):.2e}")
        print(f"  loglike : max|dlogL|={np.max(d_ll[both_ok], initial=0.0):.3e} "
              f"over {both_ok.sum()}/{len(pts)} finite pts  "
              f"(rejection agreement: {agree_rej})")
        print(f"  timing  : numpy {t_np*1e3:.0f} ms/pt | jax "
              f"{t_jax*1e3:.1f} ms/pt (compile {t_compile:.1f} s) | "
              f"speedup x{t_np/max(t_jax, 1e-9):.0f}")

    print(f"\nOVERALL (--jax): {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# Denser sample for wgg (clustering noise scales with 1/ngal; probC peak
# must stay < 1: mHMQ peak = Ac * 2 / (sqrt(2pi) * sig_M))
WGG_BASE_PARAMS = {**BASE_PARAMS, "Ac": 0.35, "As": 0.1}


def run_wgg_check(args, halo, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HOD_NRV.HOD_numerical.twopoint_calculator.tabulated_wgg import (
        WggTabulation, precompute_wgg_tabulation, TabulatedWgg, refine_rp_edges,
    )

    tab_path = os.path.join(
        args.output_dir,
        f"wgg_tabulation{'_AB' if args.assembly_bias else ''}.npz")
    if os.path.exists(tab_path):
        tab = WggTabulation.load(tab_path)
    else:
        pos_rsd = np.array(halo.positions, dtype=np.float64)
        pos_rsd[:, halo.rsd_axis_index] += (
            np.asarray(halo.velocities)[:, halo.rsd_axis_index] * halo.rsd_factor)
        pos_rsd = (pos_rsd + LBOX) % LBOX
        tab = precompute_wgg_tabulation(
            pos_rsd, np.asarray(halo.logM), LBOX, halo.rsd_axis,
            halo_fI=np.asarray(halo.fI) if args.assembly_bias else None,
            n_logM_bins=args.n_logM_bins_wgg, n_fI_bins=args.n_fI_bins_wgg,
            rp_edges=refine_rp_edges(RP_BINS))
        tab.save(tab_path)

    tabw = TabulatedWgg(tab, halo)
    print(tabw)

    results = {}
    all_pass = True
    for name, extra in cases.items():
        params = {**WGG_BASE_PARAMS, **extra}
        if args.assembly_bias:
            params.setdefault("A_cent", 0.0)
            params.setdefault("A_sat", 0.0)

        t0 = time.time()
        rp, wgg_tab, info = tabw.predict(params, RP_BINS)
        t_tab = time.time() - t0

        wgg_mc = []
        t0 = time.time()
        for i in range(args.n_real):
            halo.populate_haloes(params, random_seed=1000 + i)
            _, w_i = halo.compute_galaxy_clustering(
                'rppi', RP_BINS, bins2=PI_BINS, output='wp')
            wgg_mc.append(w_i)
        t_mc = (time.time() - t0) / args.n_real
        wgg_mc = np.array(wgg_mc)
        mc_mean = wgg_mc.mean(axis=0)
        mc_se = wgg_mc.std(axis=0, ddof=1) / np.sqrt(args.n_real)

        dev = wgg_tab / mc_mean - 1
        nsig = (wgg_tab - mc_mean) / mc_se
        # per bin: deviation must be small OR statistically insignificant
        ok = bool(np.all((np.abs(dev) < 0.05) | (np.abs(nsig) < 3))
                  and np.abs(nsig).max() < 4)
        all_pass &= ok

        print(f"\n=== wgg {name} ===  [{'PASS' if ok else 'FAIL'}]")
        print(f"  tabulated: {t_tab*1e3:.0f} ms/call | MC: {t_mc:.2f} s/real "
              f"x {args.n_real} | fsat tab={info['fsat']:.4f}")
        print(f"  max|dev|={np.abs(dev).max()*100:.2f}%  "
              f"max|dev|/SE={np.abs(nsig).max():.1f}")
        for j in range(len(rp)):
            print(f"    rp={rp[j]:7.3f}  tab={wgg_tab[j]:9.3f}  "
                  f"mc={mc_mean[j]:9.3f} ± {mc_se[j]:7.3f}  "
                  f"dev={dev[j]*100:+6.2f}%")
        results[name] = dict(rp=rp, tab=wgg_tab, mc=mc_mean, se=mc_se)

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    if n == 1:
        axes = axes[:, None]
    for k, (name, r) in enumerate(results.items()):
        ax, axr = axes[0, k], axes[1, k]
        ax.errorbar(r["rp"], r["rp"] * r["mc"], yerr=r["rp"] * r["se"],
                    fmt="o", ms=4, color="k", label=f"MC x{args.n_real}")
        ax.plot(r["rp"], r["rp"] * r["tab"], "-", color="crimson", label="tabulated")
        ax.set_xscale("log")
        ax.set_title(f"wgg {name}")
        ax.legend(fontsize=8)
        axr.axhline(0, color="k", lw=0.5)
        axr.fill_between(r["rp"], -100 * r["se"] / r["mc"], 100 * r["se"] / r["mc"],
                         alpha=0.3, color="gray")
        axr.plot(r["rp"], 100 * (r["tab"] / r["mc"] - 1), "o-", ms=3, color="crimson")
        axr.set_ylim(-8, 8)
        axr.set_xlabel(r"$r_p$ [Mpc/h]")
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "cross_check_tabulated_wgg.png")
    fig.savefig(plot_path, dpi=130)
    print(f"\nPlot: {plot_path}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.jax:
        # float64 parity with the NumPy path; must precede any JAX array use
        import jax
        jax.config.update("jax_enable_x64", True)

    import pandas as pd
    from HOD_NRV.HOD_numerical.HOD import HaloOccupation
    from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
        HaloCenterLensingCache, precompute_halo_center_lensing, TabulatedDeltaSigma,
    )

    column_mapping = dict(COLUMN_MAPPING)
    if args.assembly_bias:
        column_mapping["fI"] = "fs_norm"

    print(f"Loading halo catalog ({args.halo_fraction:.0%} subsample)...")
    df = pd.read_parquet(HALO_PATH)
    rng = np.random.default_rng(42)
    keep = rng.random(len(df)) < args.halo_fraction
    df = df[keep].reset_index(drop=True)
    print(f"  {len(df)} halos kept")

    cache_path = args.cache_path or os.path.join(
        args.output_dir,
        f"tabulated_cache_check{'_AB' if args.assembly_bias else ''}.h5")
    # particles are only needed to (pre)compute the lensing cache
    need_particles = not args.wgg and not (args.jax and os.path.exists(cache_path))

    halo = HaloOccupation(
        cosmology=COSMO_PARAMS, zeff=ZEFF, Lbox=LBOX,
        column_mapping=column_mapping, mass_definition=MASS_DEFINITION,
        DataFrame=df,
        DataFrame_part=pd.read_parquet(PARTICLE_PATH) if need_particles else None,
        assembly_bias=args.assembly_bias, apply_rsd=True, do_test=False,
        particle_fraction=args.particle_fraction,
        population_backend="numba", mass_function="Despali16",
    )
    halo.set_halo_model("ELG_mHMQ")

    if args.wgg:
        cases = dict(CASES)
        if args.assembly_bias:
            cases["NFW_AB"] = dict(AB_CASE)
            cases["EXT_AB"] = {**CASES["EXT_compact"], **AB_CASE}
        run_wgg_check(args, halo, cases)
        return

    if os.path.exists(cache_path):
        cache = HaloCenterLensingCache.load(cache_path)
        if not cache.has_tabulation or len(cache.positions) != len(halo.logM):
            raise ValueError(f"Stale cache {cache_path} — delete and re-run.")
    else:
        print("Precomputing tabulated halo-center cache (one-time)...")
        cache = precompute_halo_center_lensing(
            halo_positions=np.asarray(halo.positions),
            particle_positions=np.asarray(halo.positions_part),
            Lbox=LBOX, rsd_axis=halo.rsd_axis, RHO_M=halo.RHO_M,
            rp_bins=RP_BINS,
            halo_logM=np.asarray(halo.logM),
            halo_fI=np.asarray(halo.fI) if args.assembly_bias else None,
            n_logM_bins=args.n_logM_bins, n_fI_bins=args.n_fI_bins,
        )
        cache.save(cache_path)

    if args.jax:
        run_jax_check(args, halo, cache)
        return

    tab = TabulatedDeltaSigma(cache, halo)
    print(tab)

    cases = dict(CASES)
    if args.assembly_bias:
        cases["NFW_AB"] = dict(AB_CASE)
        cases["EXT_AB"] = {**CASES["EXT_compact"], **AB_CASE}

    results = {}
    all_pass = True
    for name, extra in cases.items():
        params = {**BASE_PARAMS, **extra}
        if args.assembly_bias:
            params.setdefault("A_cent", 0.0)
            params.setdefault("A_sat", 0.0)

        t0 = time.time()
        rp, ds_tab, info = tab.predict(params)
        t_tab = time.time() - t0

        ds_mc, fsat_mc, ngal_mc = [], [], []
        t0 = time.time()
        for i in range(args.n_real):
            halo.populate_haloes(params, random_seed=1000 + i)
            _, ds_i = halo.compute_galaxy_lensing_optimized(RP_BINS, cache)
            ds_mc.append(ds_i)
            fsat_mc.append(float(halo.satellite_fraction))
            ngal_mc.append(len(halo.positions_gal) / LBOX**3)
        t_mc = (time.time() - t0) / args.n_real
        ds_mc = np.array(ds_mc)
        mc_mean = ds_mc.mean(axis=0)
        mc_se = ds_mc.std(axis=0, ddof=1) / np.sqrt(args.n_real)

        dev = ds_tab / mc_mean - 1
        nsig = (ds_tab - mc_mean) / mc_se
        # per bin: deviation must be small OR statistically insignificant
        ok = bool(np.all((np.abs(dev) < 0.05) | (np.abs(nsig) < 3))
                  and np.abs(nsig).max() < 4)
        all_pass &= ok

        print(f"\n=== {name} ===  [{'PASS' if ok else 'FAIL'}]")
        print(f"  tabulated: {t_tab*1e3:.0f} ms/call | MC: {t_mc:.2f} s/realization "
              f"x {args.n_real}")
        print(f"  fsat  tab={info['fsat']:.4f}  MC={np.mean(fsat_mc):.4f}")
        print(f"  ngal  tab={info['ngal']:.3e}  MC={np.mean(ngal_mc):.3e}")
        print(f"  max|dev|={np.abs(dev).max()*100:.2f}%  "
              f"max|dev|/SE={np.abs(nsig).max():.1f}")
        for j in range(len(rp)):
            print(f"    rp={rp[j]:7.3f}  tab={ds_tab[j]:9.4f}  "
                  f"mc={mc_mean[j]:9.4f} ± {mc_se[j]:7.4f}  "
                  f"dev={dev[j]*100:+6.2f}%")

        results[name] = dict(rp=rp, ds_tab=ds_tab, mc_mean=mc_mean, mc_se=mc_se,
                             fsat_tab=info['fsat'], fsat_mc=np.mean(fsat_mc))

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    if n == 1:
        axes = axes[:, None]
    for k, (name, r) in enumerate(results.items()):
        ax, axr = axes[0, k], axes[1, k]
        ax.errorbar(r["rp"], r["rp"] * r["mc_mean"], yerr=r["rp"] * r["mc_se"],
                    fmt="o", ms=4, label=f"MC x{args.n_real}", color="k")
        ax.plot(r["rp"], r["rp"] * r["ds_tab"], "-", color="crimson",
                label="tabulated")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_ylabel(r"$r_p \Delta\Sigma$") if k == 0 else None
        ax.legend(fontsize=8)
        axr.axhline(0, color="k", lw=0.5)
        axr.fill_between(r["rp"], -100 * r["mc_se"] / r["mc_mean"],
                         100 * r["mc_se"] / r["mc_mean"], alpha=0.3, color="gray")
        axr.plot(r["rp"], 100 * (r["ds_tab"] / r["mc_mean"] - 1), "o-",
                 ms=3, color="crimson")
        axr.set_ylim(-6, 6)
        axr.set_xlabel(r"$r_p$ [Mpc/h]")
        axr.set_ylabel("dev [%]") if k == 0 else None
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "cross_check_tabulated.png")
    fig.savefig(plot_path, dpi=130)
    np.savez(os.path.join(args.output_dir, "cross_check_tabulated.npz"),
             **{f"{k}_{q}": v for k, r in results.items() for q, v in r.items()})
    print(f"\nPlot: {plot_path}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
