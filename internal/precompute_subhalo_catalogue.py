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
    parser.add_argument("--distinct", action="store_true",
                        help="Write host_catalogue_distinct.parquet: distinct-"
                             "halo (Rockstar-like) convention, satellites beyond "
                             "their FOF host's R200m promoted with an aperture "
                             "M200m (see build_distinct_halo_catalogue).")
    parser.add_argument("--aperture", choices=["inclusive", "exclusive"],
                        default="inclusive",
                        help="With --distinct: aperture M200m for promoted "
                             "halos. exclusive = bound-only, corrected by the "
                             "centrals' median offset to SO/200_mean; written "
                             "to host_catalogue_distinct_excl.parquet.")
    parser.add_argument("--distinct_nisp", type=str, default="",
                        help="With --distinct: a rebuilt NISP catalogue "
                             "(precompute --nisp) to re-host under the distinct "
                             "convention; writes NISP_catalogue_distinct.parquet "
                             "and prints its occupation next to the FOF one.")
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


def _nisp_soap_rows(args):
    """The rebuilt NISP catalogue plus each galaxy's OWN SOAP row.

    The catalogue stores host_row, which is the galaxy's own row only for
    centrals; a satellite's is recovered by matching its stellar centre of
    mass (ExclusiveSphere/300kpc, what the builder stored as x, y, z) and
    checked against its HostHaloIndex.
    """
    import h5py
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    gal = pd.read_parquet(args.distinct_nisp)
    is_sat = gal["type"].values == 1
    rows = gal["host_row"].values.astype(np.int64).copy()
    with h5py.File(args.soap_path, "r") as f:
        ms = f["ExclusiveSphere/300kpc/StellarMass"][:]           # 1e10 Msun
        pool = np.nonzero(ms >= 10.0 ** (args.log_mstar_min - 10.0) * 0.999)[0]
        del ms
        com = f["ExclusiveSphere/300kpc/StellarCentreOfMass"][:][pool]  # cMpc
        hidx = f["SOAP/HostHaloIndex"][:]
    com = (args.h * com) % args.Lbox                               # Mpc/h
    tree = cKDTree(com, boxsize=args.Lbox)
    xyz = gal[["x", "y", "z"]].values[is_sat].astype(np.float64) % args.Lbox
    dist, j = tree.query(xyz, k=1, workers=-1)
    own = pool[j]
    ok = (dist < 1e-4) & (hidx[own] == rows[is_sat])
    print(f"  satellite -> own SOAP row: {100 * ok.mean():.3f}% matched "
          f"(stellar CoM within 1e-4 Mpc/h AND same FOF host); "
          f"max matched offset {dist[ok].max():.2e} Mpc/h")
    if not ok.all():
        raise SystemExit(f"{(~ok).sum():,} satellites unmatched")
    rows[is_sat] = own
    return gal, rows


def run_distinct_nisp(args, gal, rows, halos, assign):
    """NISP occupation re-hosted under the distinct-halo convention."""
    import numpy as np
    import h5py

    fof_sat = gal["type"].values == 1
    row, cen_d, case = assign["row"], assign["central"], assign["case"]
    sat_d = ~cen_d
    ok = row >= 0
    m_d = np.full(len(gal), np.nan)
    m_d[ok] = halos["mass"].values[row[ok]]
    # a centre of a sub-threshold halo: its own M200m is below the threshold
    # (or undefined); it cannot enter a HOD built on the listed halos
    print(f"\n=== NISP galaxies re-hosted under the distinct convention ===")
    print(f"  placed by: own candidate {np.sum(case == 0):,}, inside FOF host "
          f"R200m {np.sum(case == 1):,}, containment {np.sum(case == 2):,}")
    print(f"  {'FOF -> distinct':>16} {'central':>9} {'satellite':>10} "
          f"{'sub-thr cen':>12}")
    for lab, s in (("central", ~fof_sat), ("satellite", fof_sat)):
        print(f"  {lab:>16} {np.sum(s & cen_d & ok):9,} {np.sum(s & sat_d):10,} "
              f"{np.sum(s & ~ok):12,}")
    print(f"  satellites whose host changes: "
          f"{np.sum(fof_sat & sat_d & (halos['soap_index'].values[np.maximum(row, 0)] != gal['host_row'].values)):,}")

    use = ok
    fsat_f, fsat_d = fof_sat.mean(), sat_d[use].mean()
    mf = gal["mass_200m"].values
    print(f"  fsat: FOF {fsat_f:.4f}  distinct {fsat_d:.4f} "
          f"({np.sum(~ok):,} sub-threshold centres dropped)")
    print(f"  Meff(cen): FOF {np.mean(mf[~fof_sat]):.4e}  distinct "
          f"{np.mean(m_d[use & cen_d]):.4e} Msun/h")
    print(f"  Meff(all): FOF {np.mean(mf):.4e}  distinct "
          f"{np.mean(m_d[use]):.4e} Msun/h")

    with h5py.File(args.soap_path, "r") as f:
        hidx = f["SOAP/HostHaloIndex"][:]
        m_fof = args.h * 1e10 * f["SO/200_mean/TotalMass"][:][hidx == -1].astype(np.float64)
    bins = np.geomspace(args.hod_mmin, args.hod_mmax, args.hod_nbins)
    centers = np.sqrt(bins[1:] * bins[:-1])
    cnt_f = np.zeros(len(m_fof), np.int64)       # Poisson ratio not needed here
    occ_f = _occupation(mf, fof_sat, m_fof, cnt_f, bins)
    cnt_d = np.bincount(row[use & sat_d], minlength=len(halos))
    occ_d = _occupation(m_d[use], sat_d[use], halos["mass"].values, cnt_d, bins)
    for tag, o in (("FOF", occ_f), ("distinct", occ_d)):
        p = np.nanargmax(o[0])
        print(f"  peak <Ncen> {tag}: {o[0][p]:.4f} at logM {np.log10(centers[p]):.3f}")
    print(f"\n{'logM':>7} {'Nh FOF':>9} {'Nh dist':>9} {'<Nc> FOF':>9} "
          f"{'<Nc> dist':>9} {'ratio':>6} {'<Ns> FOF':>9} {'<Ns> dist':>9} "
          f"{'ratio':>6} {'Poiss dist':>10}")
    for i in range(len(centers)):
        if occ_d[5][i] < 50:
            continue
        rc = occ_d[0][i] / occ_f[0][i] if occ_f[0][i] > 0 else np.nan
        rs = occ_d[1][i] / occ_f[1][i] if occ_f[1][i] > 0 else np.nan
        print(f"{np.log10(centers[i]):7.3f} {occ_f[5][i]:9d} {occ_d[5][i]:9d} "
              f"{occ_f[0][i]:9.5f} {occ_d[0][i]:9.5f} {rc:6.3f} "
              f"{occ_f[1][i]:9.5f} {occ_d[1][i]:9.5f} {rs:6.3f} "
              f"{occ_d[2][i]:10.4f}")

    out = gal.copy()
    out["own_row"] = rows
    out["type_distinct"] = sat_d.astype(np.int32)
    out["mass_200m_distinct"] = m_d
    out["distinct_row"] = row
    out["distinct_case"] = case
    p = os.path.join(args.output_dir, "NISP_catalogue_distinct.parquet")
    out.to_parquet(p, index=False)
    q = os.path.join(args.output_dir, "NISP_occupation_distinct.npz")
    np.savez(q, bins=bins, centers=centers,
             ncen_fof=occ_f[0], nsat_fof=occ_f[1], nhost_fof=occ_f[5],
             ncen_distinct=occ_d[0], nsat_distinct=occ_d[1],
             nhost_distinct=occ_d[5], poisson_distinct=occ_d[2],
             fsat_fof=fsat_f, fsat_distinct=fsat_d)
    print(f"\nSaved -> {p}\n      -> {q}")


def run_nisp(args):
    """Rebuild the NISP ELG catalogue from SOAP and measure its occupation."""
    import numpy as np
    import pandas as pd
    from internal.subhalo_catalogue import build_nisp_catalogue

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

    if args.distinct:
        from internal.subhalo_catalogue import build_distinct_halo_catalogue
        t0 = time.time()
        print(f"Distinct-halo catalogue from {args.soap_path} "
              f"(M200m >= {args.mass_threshold:.1e} Msun/h)")
        gal = rows = None
        if args.distinct_nisp:
            gal, rows = _nisp_soap_rows(args)
        res = build_distinct_halo_catalogue(args.soap_path, h=args.h,
                                            Lbox=args.Lbox,
                                            mass_threshold=args.mass_threshold,
                                            gal_rows=rows,
                                            aperture=args.aperture)
        df, assign = res if gal is not None else (res, None)
        if len(df) == 0:
            raise SystemExit("empty distinct-halo catalogue")
        os.makedirs(args.output_dir, exist_ok=True)
        tag = "_excl" if args.aperture == "exclusive" else ""
        out = os.path.join(args.output_dir, f"host_catalogue_distinct{tag}.parquet")
        df.to_parquet(out, index=False)
        print(f"Saved {len(df):,} halos -> {out}  ({time.time() - t0:.0f}s)")
        if gal is not None:
            run_distinct_nisp(args, gal, rows, df, assign)
        return

    from internal.subhalo_catalogue import (
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
