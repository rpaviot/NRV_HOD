#!/usr/bin/env python3
"""
precompute_wgg_tabulation.py

Precompute + save the wgg wp_ij(rp) tabulation between all (logM [x fI]) bins of
RSD halo centers (pycorr, Kaiser + exclusion built in). This is the wgg analog of
example_scripts/precompute_halo_center_cache.py.

The tabulation is *field-independent* — it depends only on the halo catalogue
(positions, velocities, logM, fs_norm), not on the matter field — so a single
wgg tabulation is reused for BOTH the DMO and baryonified DeltaSigma joint fits.

The fI dimension bins by ``halo.fE`` (the fs_norm environment column, mapped as
fE exactly as in precompute_halo_center_cache.py), so the fI-quantile bins match
the DeltaSigma cache and the fitter's AB convention.

Usage (cluster):
    python example_scripts/precompute_wgg_tabulation.py \
        --output /sps/euclid/Users/rpaviot/flamingo/wgg_tabulation_AB.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_scripts.run_tabulated_chains import (
    build_halo_occupation, HALO_PATH_DEFAULT, FLAMINGO_DIR, LBOX)
from HOD_NRV.utilsf.numerical_sampler import FitCase
from HOD_NRV.HOD_numerical.twopoint_calculator.tabulated_wgg import (
    precompute_wgg_tabulation, refine_rp_edges)


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute + save the wgg wp_ij tabulation (RSD halo "
                    "centers, logM x fI bins); reusable for DMO and baryon.")
    p.add_argument("--halo_path", default=HALO_PATH_DEFAULT)
    p.add_argument("--output",
                   default=os.path.join(FLAMINGO_DIR, "wgg_tabulation_AB.npz"))
    # Defaults match the DeltaSigma cache (40 logM x 8 fI, rp geomspace(0.1,50,26))
    # so both observables share the same (logM, fI, rp) binning schema.
    p.add_argument("--n_logM_bins", type=int, default=40)
    p.add_argument("--n_fI_bins", type=int, default=8)
    p.add_argument("--no_ab", action="store_true",
                   help="Tabulate without fI bins (logM only).")
    # analysis rp binning the tabulation must nest (refine_rp_edges); the tab
    # aggregates onto any sub-binning of this at predict time.
    p.add_argument("--rp_min", type=float, default=0.1)
    p.add_argument("--rp_max", type=float, default=50.0)
    p.add_argument("--n_rp", type=int, default=26)
    p.add_argument("--pi_max", type=float, default=100.0)
    p.add_argument("--n_pi", type=int, default=101)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading halo catalogue: {args.halo_path}")
    halo = build_halo_occupation(FitCase.STANDARD_NFW, args.halo_path)

    # RSD halo centers: shift centers by v_los * rsd_factor, periodic-wrap.
    # (Kaiser enters the tabulation; centrals inherit this exactly, satellite
    #  LOS offsets integrate out of wp up to pi_max edge leakage.)
    pos_rsd = np.array(halo.positions, dtype=np.float64)
    ax = halo.rsd_axis_index
    pos_rsd[:, ax] += np.asarray(halo.velocities)[:, ax] * halo.rsd_factor
    pos_rsd = (pos_rsd + LBOX) % LBOX

    # bin the fI dimension on halo.fE (= fs_norm), matching the DeltaSigma cache
    halo_fI = None if args.no_ab else np.asarray(halo.fE)
    rp_bins = np.geomspace(args.rp_min, args.rp_max, args.n_rp)
    pi_bins = np.linspace(0.0, args.pi_max, args.n_pi)

    print(f"  {len(halo.logM):,} halos, Lbox={LBOX}, rsd_axis={halo.rsd_axis}")
    print(f"  bins: {args.n_logM_bins} logM x "
          f"{1 if args.no_ab else args.n_fI_bins} fI; "
          f"rp {args.rp_min}-{args.rp_max} ({args.n_rp}), pi_max={args.pi_max}")

    tab = precompute_wgg_tabulation(
        pos_rsd, np.asarray(halo.logM), LBOX, halo.rsd_axis,
        halo_fI=halo_fI,
        n_logM_bins=args.n_logM_bins,
        n_fI_bins=args.n_fI_bins,
        rp_edges=refine_rp_edges(rp_bins),
        pi_bins=pi_bins,
        verbose=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tab.save(args.output)
    print(f"Saved wgg tabulation -> {args.output}")


if __name__ == "__main__":
    main()
