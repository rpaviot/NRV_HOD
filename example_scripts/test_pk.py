"""
Validate the matter power spectrum implementation.

Measures P(k) of the matter field directly from the Flamingo particle
catalogue using `PowerSpectrumEstimator` and compares to the linear and
non-linear pyccl predictions from `Cosmology`.

Usage
-----
    python test_pk.py --Nmesh 512 --output_dir pk_validation

    # Quick smoke test
    python test_pk.py --Nmesh 256 --subsample 0.05 \
        --output_dir pk_validation_smoke
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HOD_NRV.HOD_analytical.pycosmo import Cosmology
from HOD_NRV.utilsf.measure_pk import PowerSpectrumEstimator, log_kbins


_DEFAULT_PARTICLE_PATH = (
    "/Users/ler13nrv/Documents/flamingo_data/"
    "particle_catalogue_L1000N1800_downsampled.parquet"
)

LBOX = 681.0  # Mpc/h

COSMO_PARAMS = {
    "h":    0.681,
    "Omc":  0.306 - 0.0486 - 1.39e-3,
    "Omb":  0.0486,
    "A_s":  2.099e-9,
    "n_s":  0.967,
    "Omnu": 1.39e-3,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--particle_path", default=_DEFAULT_PARTICLE_PATH)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--Lbox", type=float, default=LBOX)
    p.add_argument("--Nmesh", type=int, default=512)
    p.add_argument("--n_kbins", type=int, default=25)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Further downsample particle catalogue by this fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="pk_validation")
    return p.parse_args()


def load_particles(path, subsample, seed):
    print(f"Loading particles from {path} ...")
    df = pd.read_parquet(path)
    pos = np.column_stack([df["x"].values, df["y"].values, df["z"].values]
                          ).astype(np.float64)
    print(f"  {len(pos):,} particles loaded")
    if subsample < 1.0:
        rng = np.random.default_rng(seed)
        n_keep = int(len(pos) * subsample)
        idx = rng.choice(len(pos), size=n_keep, replace=False)
        pos = pos[idx]
        print(f"  subsampled to {len(pos):,} particles ({subsample:.3g})")
    return pos


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pos = load_particles(args.particle_path, args.subsample, args.seed)
    N = len(pos)
    n_bar = N / args.Lbox**3
    shot_noise = 1.0 / n_bar
    print(f"  n_bar = {n_bar:.4g} (Mpc/h)^-3")
    print(f"  shot noise = Lbox^3 / N = {shot_noise:.4g} (Mpc/h)^3")

    print("\nBuilding cosmology + theory P(k) ...")
    cosmo = Cosmology(cosmo_params=COSMO_PARAMS, units_per_h=True, verbose=True)
    k_theory = np.geomspace(1e-3, 5.0, 400)
    P_lin_th = np.asarray(cosmo.linear_power(k_theory, z=args.z))
    P_nl_th = np.asarray(cosmo.nonlinear_power(k_theory, z=args.z))

    print("\nMeasuring matter P(k) ...")
    est = PowerSpectrumEstimator(Nmesh=args.Nmesh, Lbox=args.Lbox,
                                 threads=args.threads)
    deltak = est.delta_k(pos)
    kbins = log_kbins(args.Lbox, args.Nmesh, n_bins=args.n_kbins)
    k, P_meas, n_modes = est.auto_pk(deltak, kbins, n_tracer=N,
                                     subtract_shot_noise=True)

    out_npz = os.path.join(args.output_dir, "pk_matter_measured.npz")
    np.savez(
        out_npz,
        k=k, P=P_meas, n_modes=n_modes,
        k_theory=k_theory, P_lin=P_lin_th, P_nl=P_nl_th,
        z=args.z, Lbox=args.Lbox, Nmesh=args.Nmesh,
        N_part=N, shot_noise=shot_noise,
    )
    print(f"\nSaved measurements to {out_npz}")

    k_nyq = np.pi * args.Nmesh / args.Lbox
    k_cut = (2.0 / 3.0) * k_nyq

    valid = np.isfinite(k) & np.isfinite(P_meas) & (k > 0)
    k_v = k[valid]
    P_v = P_meas[valid]
    P_nl_at_k = np.interp(np.log(k_v), np.log(k_theory), P_nl_th)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_top.loglog(k_theory, P_lin_th, "--", color="C2", lw=1.4, label="linear (pyccl)")
    ax_top.loglog(k_theory, P_nl_th, "-", color="C3", lw=1.6, label="non-linear (Halofit)")
    ax_top.loglog(k_v, P_v, "o", color="C0", ms=5, label="measured")
    ax_top.axhline(shot_noise, color="gray", lw=0.8, ls=":",
                   label=f"shot noise = {shot_noise:.2g}")
    ax_top.axvspan(k_cut, k_v.max() * 1.5, color="gray", alpha=0.15,
                   label=r"$k > \frac{2}{3} k_{\rm Nyq}$")
    ax_top.set_ylabel(r"$P(k)$  [$(h^{-1}\,{\rm Mpc})^3$]")
    ax_top.set_title(f"Matter P(k) — z={args.z}, Lbox={args.Lbox}, Nmesh={args.Nmesh}")
    ax_top.legend(fontsize=9, loc="lower left")
    ax_top.set_xlim(k_v.min() * 0.7, k_v.max() * 1.5)

    ratio = P_v / P_nl_at_k
    ax_bot.semilogx(k_v, ratio, "o-", color="C0", lw=1.0, ms=4)
    ax_bot.axhline(1.0, color="k", lw=0.8)
    ax_bot.axhline(1.05, color="gray", lw=0.6, ls="--")
    ax_bot.axhline(0.95, color="gray", lw=0.6, ls="--")
    ax_bot.axvspan(k_cut, k_v.max() * 1.5, color="gray", alpha=0.15)
    ax_bot.set_ylim(0.7, 1.3)
    ax_bot.set_xlabel(r"$k$  [$h$/Mpc]")
    ax_bot.set_ylabel(r"$P_{\rm meas} / P_{\rm NL}$")

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "pk_matter_validation.png")
    fig.savefig(out_png, dpi=140)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
