"""
Run Nautilus chains on a pre-trained DeltaSigma (+ optional wgg) emulator.

Edit PARAM_CONFIG below before launching:
  (low, high) tuple  →  free parameter (sampled)
  scalar float       →  fixed parameter (held constant)

Preset configurations
---------------------
Full CONFORMITY (all extended + conformity params free):
    f_exp:    (0.0, 0.9)
    tau:      (1.0, 10.0)
    kappa_EE: (0.05, 1.0)

CONF-only (standard NFW profile, conformity free):
    lambda_NFW: 1.0
    f_exp:      0.0
    tau:        5.0      # arbitrary fixed value, not sampled
    kappa_EE:   (0.05, 1.0)

NFW-only (no conformity, no extended profile):
    lambda_NFW: (0.05, 1.0)
    f_exp:      0.0
    tau:        5.0
    kappa_EE:   1.0

Examples
--------
python run_emulator_chains.py \\
    --emulator_path /path/to/emulator_CONFORMITY.npz \\
    --data_path /path/to/observed_ds.npz \\
    --output_path chains_CONF_full.npz \\
    --n_live 500 --n_eff 10000 --n_workers 4
"""

import argparse
import os
import sys

import numpy as np

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HOD_NRV.utilsf.numerical_sampler import EmulatorFitter

# ---------------------------------------------------------------------------
# CONFIGURE PRIORS — edit this block before launching
# ---------------------------------------------------------------------------
PARAM_CONFIG = {
    # Core HOD parameters
    "As":         (0.002, 0.08),
    "Mmin":       (11.1, 13.0),
    "sig_M":      (0.05, 1.5),
    "gamma":      (0.1, 10.0),
    "alpha":      (0.05, 1.5),
    "kappa":      (0.1, 2.0),
    # NFW profile scaling — set to 1.0 to fix (CONF-only case)
    "lambda_NFW": (0.05, 1.0),
    # Extended profile — set f_exp=0.0 and tau=5.0 to disable
    "f_exp":      (0.0, 0.9),
    "tau":        (1.0, 10.0),
    # Conformity — set to 1.0 to disable (NFW-only case)
    "kappa_EE":   (0.05, 1.0),
    # Assembly bias — uncomment to include
    # "A_cent": (-0.5, 0.5),
    # "B_cent": (-0.5, 0.5),
    # "A_sat":  (-0.5, 0.5),
    # "B_sat":  (-0.5, 0.5),
    # M1 is always fixed at 13.0 unless overridden here
    # "M1": 13.0,
}
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Nautilus chains on a pre-trained DeltaSigma emulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--emulator_path", required=True,
                   help="Path to DeltaSigma emulator .npz file")
    p.add_argument("--data_path", required=True,
                   help="Path to observed DeltaSigma .npz file")
    p.add_argument("--output_path", default="chains_results.npz",
                   help="Output .npz path for posterior samples")
    p.add_argument("--n_live",    type=int, default=500,
                   help="Nautilus live points")
    p.add_argument("--n_eff",     type=int, default=10000,
                   help="Nautilus target effective sample size")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel likelihood workers (1 is usually fine for emulator)")
    p.add_argument("--rp_min",  type=float, default=None,
                   help="Minimum rp scale cut [Mpc/h]")
    p.add_argument("--rp_max",  type=float, default=None,
                   help="Maximum rp scale cut [Mpc/h]")
    # Optional wgg joint fit
    p.add_argument("--emulator_wgg_path", default="",
                   help="Path to wgg emulator .npz (enables joint DS+wgg fit)")
    p.add_argument("--data_path_wgg",     default="",
                   help="Path to observed wgg .npz")
    p.add_argument("--rp_min_wgg", type=float, default=None)
    p.add_argument("--rp_max_wgg", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = args.output_path.replace(".npz", "_checkpoint.hdf5")

    fitter = EmulatorFitter(
        emulator_path=args.emulator_path,
        data_path=args.data_path,
        param_config=PARAM_CONFIG,
        rp_min=args.rp_min,
        rp_max=args.rp_max,
        emulator_wgg_path=args.emulator_wgg_path,
        data_path_wgg=args.data_path_wgg,
        rp_min_wgg=args.rp_min_wgg,
        rp_max_wgg=args.rp_max_wgg,
    )

    print(f"Free params ({fitter.n_params}): {fitter.param_names}")
    print(f"Fixed params: {fitter.fixed_params_dict}")
    print(f"Checkpoint:   {checkpoint}")
    print()

    points, weights, log_l, log_z = fitter.run(
        n_live=args.n_live,
        n_eff=args.n_eff,
        n_workers=args.n_workers,
        filepath=checkpoint,
    )

    fitter.save_results(args.output_path, points, weights, log_l, log_z)
    print(f"\nlog Z = {log_z:.3f}")
    print(f"Results saved to {args.output_path}")

    summary = fitter.get_posterior_summary(points, weights)
    print("\nPosterior summary:")
    for name, s in summary.items():
        lo = s["median"] - s["q16"]
        hi = s["q84"] - s["median"]
        print(f"  {name:12s}  {s['median']:.4f}  +{hi:.4f}  -{lo:.4f}")


if __name__ == "__main__":
    main()
