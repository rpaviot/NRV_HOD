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
import os
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
    parser.add_argument(
        "--nisp", action="store_true",
        help="Instead of the host/subhalo catalogues, rebuild the NISP-like "
             "ELG galaxy catalogue from the same SOAP file and measure its "
             "occupation on BOTH M200m and M200c."
    )
    parser.add_argument("--log_mstar_min", type=float, default=10.1,
                        help="log10 stellar mass floor [Msun]. The catalogue "
                             "on disk used 10.1; the notebook used 10.0.")
    parser.add_argument("--log_mstar_max", type=float, default=12.1)
    parser.add_argument("--sfr_min", type=float, default=5.0,
                        help="SFR floor [Msun/yr].")
    parser.add_argument("--disc_frac_min", type=float, default=0.7)
    parser.add_argument("--hod_mmin", type=float, default=1e11)
    parser.add_argument("--hod_mmax", type=float, default=1e14)
    parser.add_argument("--hod_nbins", type=int, default=50,
                        help="Number of mass bin EDGES (notebook used 50).")
    parser.add_argument("--compare_nisp", type=str, default="",
                        help="NISP parquet on disk to compare the rebuild to.")
    return parser.parse_args()


def _occupation(mass_gal, is_sat_gal, mass_host, counts_per_host, bins):
    """<N_cen>(M), <N_sat>(M) and the Poisson ratio, exactly as the notebook's
    get_HOD does it: numerator histogrammed on the galaxy's host mass,
    denominator on the mass of every host, no threshold beyond the binning."""
    import numpy as np
    n_cen, _ = np.histogram(mass_gal[~is_sat_gal], bins=bins)
    n_sat, _ = np.histogram(mass_gal[is_sat_gal], bins=bins)
    n_host, _ = np.histogram(mass_host, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ncen = np.where(n_host > 0, n_cen / n_host, np.nan)
        nsat = np.where(n_host > 0, n_sat / n_host, np.nan)

    # sqrt(Var N_sat)/sqrt(<N_sat>) per mass bin: 1 for a Poisson satellite
    # count, and the notebook's test of that assumption.
    hbin = np.digitize(mass_host, bins) - 1
    nb = len(bins) - 1
    ratio = np.full(nb, np.nan)
    for i in range(nb):
        Ni = counts_per_host[hbin == i]
        if Ni.size > 1 and Ni.mean() > 0:
            ratio[i] = np.sqrt(Ni.var(ddof=1)) / np.sqrt(Ni.mean())
    return ncen, nsat, ratio, n_cen, n_sat, n_host


def run_nisp(args):
    """Rebuild the NISP ELG catalogue from SOAP and measure its occupation."""
    import numpy as np
    import pandas as pd
    from HOD_NRV.utilsf.subhalo_catalogue import build_nisp_catalogue

    print(f"Building NISP catalogue from {args.soap_path}")
    cat = build_nisp_catalogue(
        args.soap_path, h=args.h, Lbox=args.Lbox,
        stellar_mass_min=10.0**args.log_mstar_min,
        stellar_mass_max=10.0**args.log_mstar_max,
        sfr_min=args.sfr_min, disc_frac_min=args.disc_frac_min)

    host_m200m = cat.pop("_host_m200m")
    host_m200c = cat.pop("_host_m200c")
    cat.pop("_host_rows")
    counts_per_host = cat.pop("_counts_per_host")

    df = pd.DataFrame(cat)
    ngal = len(df)
    is_sat_gal = df["type"].values == 1
    print(f"\nrebuilt: {ngal:,} galaxies, ngal = {ngal/args.Lbox**3:.4e} "
          f"(Mpc/h)^-3, fsat = {is_sat_gal.mean():.4f}")

    # ---- the two mass definitions ------------------------------------------
    r = np.log10(df["mass_200m"].values / df["mass_200c"].values)
    print(f"log10(M200m/M200c) over the sample: median {np.median(r):+.4f} dex "
          f"({100*(10**np.median(r)-1):+.1f}%), 16-84% "
          f"[{np.percentile(r,16):+.4f}, {np.percentile(r,84):+.4f}]")

    # ---- compare with the catalogue on disk ---------------------------------
    if args.compare_nisp and os.path.exists(args.compare_nisp):
        ref = pd.read_parquet(args.compare_nisp)
        print(f"\n--- vs {args.compare_nisp} ---")
        print(f"  rows: rebuilt {ngal:,}   on disk {len(ref):,}   "
              f"({ngal - len(ref):+,})")
        print(f"  fsat: rebuilt {is_sat_gal.mean():.4f}   "
              f"on disk {(ref['type'].values == 1).mean():.4f}")
        lref = np.log10(np.sort(ref["mass"].values))
        for tag, m in (("M200m", df["mass_200m"].values),
                       ("M200c", df["mass_200c"].values)):
            if len(m) != len(lref):
                print(f"  {tag}: row counts differ, comparing sorted "
                      f"quantiles instead")
                q = np.linspace(0.01, 0.99, 99)
                d = (np.quantile(np.log10(m), q) - np.quantile(lref, q))
            else:
                d = np.sort(np.log10(m)) - lref
            print(f"  sorted log10 mass, rebuilt {tag} - on-disk 'mass': "
                  f"median {np.median(d):+.4f} dex, "
                  f"max|d| {np.max(np.abs(d)):.4f}")

    # ---- occupation, both mass definitions ----------------------------------
    bins = np.geomspace(args.hod_mmin, args.hod_mmax, args.hod_nbins)
    centers = np.sqrt(bins[1:] * bins[:-1])
    results = {}
    for tag, mgal, mhost in (("M200m", df["mass_200m"].values, host_m200m),
                             ("M200c", df["mass_200c"].values, host_m200c)):
        ncen, nsat, pratio, n_c, n_s, n_h = _occupation(
            mgal, is_sat_gal, mhost, counts_per_host, bins)
        results[tag] = (ncen, nsat, pratio, n_c, n_s, n_h)
        peak = np.nanargmax(ncen)
        print(f"\n=== occupation on {tag} ===")
        print(f"  peak <Ncen> = {ncen[peak]:.4f} at logM = "
              f"{np.log10(centers[peak]):.3f}")
        Mw = mgal[~is_sat_gal]
        print(f"  Meff(cen) = {np.mean(Mw):.4e}   Meff(all) = "
              f"{np.mean(mgal):.4e} Msun/h")
        ok = np.isfinite(pratio) & (n_h > 100)
        print(f"  Poisson ratio sqrt(Var Nsat)/sqrt(<Nsat>): median "
              f"{np.nanmedian(pratio[ok]):.4f} over {ok.sum()} bins")

    ncen_m, ncen_c = results["M200m"][0], results["M200c"][0]
    peak_m = np.log10(centers[np.nanargmax(ncen_m)])
    peak_c = np.log10(centers[np.nanargmax(ncen_c)])
    print(f"\n>>> peak of <Ncen> shifts {peak_m - peak_c:+.3f} dex when the "
          f"HOD is built on M200m instead of M200c "
          f"(logM {peak_c:.3f} -> {peak_m:.3f})")

    print(f"\n{'logM':>7} {'N_host':>9} {'<Ncen>200m':>11} {'<Ncen>200c':>11}"
          f" {'<Nsat>200m':>11} {'<Nsat>200c':>11} {'Poiss200m':>10}")
    for i in range(len(centers)):
        if results["M200m"][5][i] < 50:
            continue
        print(f"{np.log10(centers[i]):7.3f} {results['M200m'][5][i]:9d} "
              f"{ncen_m[i]:11.5f} {ncen_c[i]:11.5f} "
              f"{results['M200m'][1][i]:11.5f} {results['M200c'][1][i]:11.5f} "
              f"{results['M200m'][2][i]:10.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_cat = os.path.join(args.output_dir, "NISP_catalogue_rebuilt.parquet")
    df.to_parquet(out_cat, index=False)
    out_hod = os.path.join(args.output_dir, "NISP_occupation_rebuilt.npz")
    np.savez(out_hod, bins=bins, centers=centers,
             host_m200m=host_m200m.astype(np.float32),
             host_m200c=host_m200c.astype(np.float32),
             **{f"{k}_{tag}": v
                for tag in ("M200m", "M200c")
                for k, v in zip(("ncen", "nsat", "poisson", "n_cen", "n_sat",
                                 "n_host"), results[tag])})
    print(f"\nSaved rebuilt catalogue -> {out_cat}")
    print(f"Saved occupation       -> {out_hod}")


def main():
    args = parse_args()

    if args.nisp:
        run_nisp(args)
        return

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
