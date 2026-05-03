"""
Compare Dark Emulator beta^NL against beta^NL measured directly from
Flamingo DMO and Hydro halo catalogs.

The measured curves use the same conventions as
`HOD_NRV.HOD_analytical.emu.BetaNLInterpolator`:
  - halo-halo bias at k_lin = 0.02 h/Mpc
  - additive force-to-zero correction
so the two are directly comparable in (k, M1, M2, z).

Usage
-----
    python compare_beta_nl_emulator_vs_sim.py \
        --z 1.0 --Lbox 681 --Nmesh 512 \
        --logM_edges 12.5 13.0 13.5 14.0 14.5 \
        --output_dir beta_nl_validation
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from HOD_NRV.HOD_analytical.pycosmo import Cosmology
from HOD_NRV.HOD_analytical.measure_beta_nl import measure_beta_nl_from_catalog
from HOD_NRV.utilsf.measure_pk import log_kbins
from HOD_NRV.utilsf.data_reader import read_halo_catalog


# Same as run_chains_all.py
COSMO_PARAMS = {
    "h":    0.681,
    "Omc":  0.306 - 0.0486 - 1.39e-3,
    "Omb":  0.0486,
    "A_s":  2.099e-9,
    "n_s":  0.967,
    "Omnu": 1.39e-3,
}

DEFAULT_DMO_HALO = (
    "/sps/euclid/Users/rpaviot/flamingo/snapshots_DMO/halo_catalogue_flamingo.parquet"
)
DEFAULT_HYDRO_HALO = (
    "/sps/euclid/Users/rpaviot/flamingo/snapshots_hydro/"
    "NISP_catalogue_flamingo_withSHEAR.parquet"
)


def load_halos(path, x_col="x", y_col="y", z_col="z", mass_col="mass"):
    """Return (positions [N,3] in Mpc/h, log10(M_h) [N] in M_sun/h)."""
    df = read_halo_catalog(halo_path=path)
    pos = np.column_stack([df[x_col].values, df[y_col].values, df[z_col].values]).astype(np.float64)
    M = np.asarray(df[mass_col].values, dtype=np.float64)
    log10M = np.log10(M)
    return pos, log10M


def measure_one(label, halo_path, log10M_edges, kbins, P_lin_func,
                Lbox, Nmesh, k_lin, threads):
    print(f"\n=== {label}: {halo_path} ===")
    pos, log10M = load_halos(halo_path)
    print(f"  loaded {len(log10M)} halos, log10M in [{log10M.min():.2f}, {log10M.max():.2f}]")
    return measure_beta_nl_from_catalog(
        halo_positions=pos,
        halo_log10M=log10M,
        Lbox=Lbox, Nmesh=Nmesh,
        log10M_bins=log10M_edges,
        P_lin_func=P_lin_func,
        kbins=kbins,
        k_lin=k_lin,
        force_to_zero="additive",
        threads=threads,
        verbose=True,
    )


def emulator_beta_nl(cosmo, z, k, M1, M2):
    """Query the BetaNLInterpolator at (k, M1, M2, z)."""
    if cosmo.beta_nl_interp is None:
        cosmo.compute_beta_nl([z])
    interp = cosmo.beta_nl_interp
    return np.asarray(interp(np.asarray(k), float(M1), float(M2), z=float(z)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--Lbox", type=float, default=681.0)
    p.add_argument("--Nmesh", type=int, default=512)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--k_lin", type=float, default=0.02)
    p.add_argument("--n_kbins", type=int, default=20)
    p.add_argument("--logM_edges", type=float, nargs="+",
                   default=[12.5, 13.0, 13.5, 14.0, 14.5])
    p.add_argument("--dmo_path", default=DEFAULT_DMO_HALO)
    p.add_argument("--hydro_path", default=DEFAULT_HYDRO_HALO)
    p.add_argument("--skip_dmo", action="store_true")
    p.add_argument("--skip_hydro", action="store_true")
    p.add_argument("--output_dir", default="beta_nl_validation")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log10M_edges = np.asarray(args.logM_edges)

    # Cosmology + linear power callback (h-units)
    cosmo = Cosmology(
        cosmo_params=COSMO_PARAMS,
        units_per_h=True,
        use_dark_emulator=True,
        verbose=True,
    )
    P_lin = lambda k: cosmo.linear_power(np.asarray(k), z=args.z)

    kbins = log_kbins(args.Lbox, args.Nmesh, n_bins=args.n_kbins)

    results = {}
    if not args.skip_dmo:
        results["DMO"] = measure_one(
            "DMO", args.dmo_path, log10M_edges, kbins, P_lin,
            args.Lbox, args.Nmesh, args.k_lin, args.threads,
        )
    if not args.skip_hydro:
        results["Hydro"] = measure_one(
            "Hydro", args.hydro_path, log10M_edges, kbins, P_lin,
            args.Lbox, args.Nmesh, args.k_lin, args.threads,
        )

    # Save raw arrays
    out_npz = os.path.join(args.output_dir, "beta_nl_measured.npz")
    save_dict = {"logM_edges": log10M_edges, "z": args.z, "Lbox": args.Lbox,
                 "Nmesh": args.Nmesh, "k_lin": args.k_lin}
    for label, r in results.items():
        for key in ("k", "M_centers", "N_per_bin", "P_hh", "b_hh", "P_lin", "beta_nl"):
            save_dict[f"{label}_{key}"] = r[key]
    np.savez(out_npz, **save_dict)
    print(f"\nSaved measurements to {out_npz}")

    # Emulator predictions on the same (k, M_centers) grid
    M_centers = next(iter(results.values()))["M_centers"]
    k_meas = next(iter(results.values()))["k"]
    cosmo.compute_beta_nl([args.z])
    emu_beta = np.zeros((len(M_centers), len(M_centers), len(k_meas)))
    for i, M1 in enumerate(M_centers):
        for j, M2 in enumerate(M_centers):
            emu_beta[i, j, :] = emulator_beta_nl(cosmo, args.z, k_meas, M1, M2)

    # Plot: rows = mass-pair selection, cols = (DMO, Hydro)
    pair_idx = [(0, 0), (0, len(M_centers) - 1), (len(M_centers) - 1, len(M_centers) - 1)]
    fig, axes = plt.subplots(len(pair_idx), len(results),
                             figsize=(5 * len(results), 3.5 * len(pair_idx)),
                             sharex=True, squeeze=False)

    for col, (label, r) in enumerate(results.items()):
        for row, (i, j) in enumerate(pair_idx):
            ax = axes[row, col]
            beta_meas = r["beta_nl"][i, j, :]
            beta_emu = emu_beta[i, j, :]
            ax.semilogx(k_meas, beta_meas, "o", ms=4, label="measured")
            ax.semilogx(k_meas, beta_emu, "-", lw=1.6, label="Dark Emulator")
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_title(f"{label}: log10(M1, M2) = "
                         f"({np.log10(M_centers[i]):.2f}, {np.log10(M_centers[j]):.2f})")
            ax.set_ylabel(r"$\beta^{\rm NL}$")
            if row == len(pair_idx) - 1:
                ax.set_xlabel(r"$k$ [$h$/Mpc]")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "beta_nl_emulator_vs_sim.png")
    fig.savefig(out_png, dpi=140)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
