"""
One-time preprocessing: build host + subhalo catalogues from a FLAMINGO SOAP file.

Usage
-----
    python precompute_subhalo_catalogue.py \\
        --soap_path /data/flamingo/flamingo_z1.hdf5 \\
        --h 0.681 \\
        --output_dir /data/flamingo/catalogues/

Outputs
-------
    <output_dir>/host_catalogue.parquet   — host halo catalogue
    <output_dir>/subhalo_catalogue.npz    — CSR subhalo catalogue

Then use in HaloOccupation::

    halo = HaloOccupation(
        ...,
        halo_path="<output_dir>/host_catalogue.parquet",
        subhalo_path="<output_dir>/subhalo_catalogue.npz",
    )
"""

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build host + subhalo catalogues from a FLAMINGO SOAP HDF5 file"
    )
    parser.add_argument(
        "--soap_path", type=str, required=True,
        help="Path to SOAP HDF5 file"
    )
    parser.add_argument(
        "--h", type=float, default=0.681,
        help="Dimensionless Hubble parameter (default: 0.681)"
    )
    parser.add_argument(
        "--Lbox", type=float, default=681.0,
        help="Box size in Mpc/h (default: 681.0)"
    )
    parser.add_argument(
        "--mass_threshold", type=float, default=1e11,
        help="Minimum host M200m in Msun/h (default: 1e11)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory (created if absent)"
    )
    parser.add_argument(
        "--ranking",
        choices=["r_ascending", "r_descending", "vpeak_descending", "hybrid"],
        default="r_descending",
        help=(
            "Subhalo selection ranking baked into the CSR: "
            "r_descending (ELG outermost first), "
            "vpeak_descending (SHAM most massive first), "
            "hybrid (combined r + vpeak rank), "
            "r_ascending (innermost first). Default: r_descending."
        ),
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=0.5,
        help="Weight of r_rank in hybrid score (1-alpha goes to vpeak_rank). Default: 0.5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from HOD_NRV.utilsf.subhalo_catalogue import (
        build_halo_and_subhalo_catalogues,
        save_catalogues,
    )

    print(f"Loading SOAP file: {args.soap_path}")
    print(f"  h={args.h}, Lbox={args.Lbox} Mpc/h, M_min={args.mass_threshold:.1e} Msun/h")
    t0 = time.time()

    host_cat, sub_cat, csr = build_halo_and_subhalo_catalogues(
        filepath=args.soap_path,
        h=args.h,
        Lbox=args.Lbox,
        mass_threshold=args.mass_threshold,
        ranking=args.ranking,
        hybrid_alpha=args.hybrid_alpha,
    )

    t1 = time.time()
    print(f"Catalogue built in {t1-t0:.1f}s")

    # An empty catalogue is a unit or threshold error, not a valid result, and
    # it used to be written out with exit code 0 -- indistinguishable from
    # success until the fits downstream came out nonsense.
    if len(host_cat['mass']) == 0:
        raise SystemExit(
            f"No hosts above {args.mass_threshold:.1e} Msun/h. Check the SOAP "
            f"mass units (swiftsimio returns the file's internal unit) and h.")
    print(f"  N_host = {len(host_cat['mass']):,}")
    print(f"  N_sub  = {len(sub_cat['cop']):,}")
    print(f"  max subhalos per host = {csr['max_subs']:,}")

    save_catalogues(host_cat, sub_cat, csr, args.output_dir,
                    ranking=args.ranking, hybrid_alpha=args.hybrid_alpha)
    print(f"Done in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
