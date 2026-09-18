#!/usr/bin/env python3
"""
compute_gaussian_covariance.py

Gaussian covariance for the in-box DeltaSigma + wgg data vector, following
Marian, Smith & Angulo 2015 (arXiv:1410.3468) Eq. 32 (wgg) and Eq. 53 (DeltaSigma),
generalised to a periodic box where the wp projection length Pi_max and the
DeltaSigma projection length chi_max are shorter than the box.

Box generalisation
------------------
Marian et al. write V_s = 2 chi_max A_s, i.e. survey depth == projection depth.
In a periodic box the mode density is set by the *box* volume while the LOS
window is set by the projection, so the prefactors become

    wgg:        4 Pi_max / V_box        (== Eq. 32's 2/A_s when 2 Pi_max = L_box)
    DeltaSigma: 2 chi_max_ds / V_box    (== Eq. 53's 1/A_s when 2 chi_max_ds = L_box)

Eq. 53 carries no chi_max because Marian et al.'s lenses fill the same slab they
project through (V_s = 2 chi_max A_s, so N_g = n_g 2 chi_max A_s): the depth
cancels between "more particles per cylinder" and "more lenses". In the box the
lenses fill all of L_box while Sigma is truncated at |chi| < chi_max, so the
cancellation is incomplete and the prefactor is 2 chi_max / V_box. Using
1/A_s with A_s = L_box^2 is the same expression with 2 chi_max = L_box, i.e. it
assumes Sigma is projected through the whole box. Our measurement and the
tabulated forward model both use chi_max = 100 Mpc/h, hence CHI_MAX_DS below.

What this fixes relative to the original in-notebook script
-----------------------------------------------------------
1. P_lin was taken from `hm.linear_power()`, whose signature defaults to z=0.0
   (HaloModel does NOT forward its own z). Here the redshift is passed explicitly.
2. b1 = 1.05 did not describe the sample: DeltaSigma / [b=1 prediction with the
   nonlinear P_mm] gives b = 1.713, flat to 0.3% over rp = 8-37 Mpc/h.
3. Particle shot noise: the hydro particles are MASS-WEIGHTED (3 species,
   ~5:1 mass ratio), so the effective density is n_eff = <m>^2/<m^2> * N/V
   (= 0.68309 * N/V for this catalogue), and N is the number actually used in
   the measurement (particle_fraction of the 233M catalogue), not the full set.
4. Eq. 32's Kronecker term carries [w_gg(R_i) + 2 Pi_max]/n_g^2, not just the
   Poisson piece: the w_gg(R_i) part was missing. It is ~79% of the Poisson term
   at rp = 0.2 and ~24% at rp = 1.3, i.e. dominant exactly where a joint fit
   gains information.
5. The bin-averaged Bessel filters are done analytically instead of by quad
   (the original 2D quad grid threw tolerance warnings and then needed cubic
   `fill_value='extrapolate'` out to k=100):
       int_r1^r2 2 pi r J0(kr) dr = 2 pi / k^2 [ x J1(x) ]
       int_r1^r2 2 pi r J2(kr) dr = 2 pi / k^2 [ -x J1(x) - 2 J0(x) ]
   with x = k r, so no interpolation of the filters is involved anywhere.
6. Adds the DeltaSigma x wgg cross-block, which a joint fit needs and which is
   not zero: Cov[P_gm, P_gg] = 2 P_gm (P_gg + 1/n_g) / N_modes.
7. The k integral ran to k=100 h/Mpc. The shot-noise terms are k-independent,
   so the integral only converges once k >> 1/(bin width): at k_max=100 the
   narrow inner annuli are up to 27% short (checked against the exact
   orthogonality relation int dk k/(2pi) Jbar_i Jbar_j = delta_ij / A_i, which
   recovers to 0.998-1.000 with k_max=1e4).

NOTE on the prefactor: the original script's wgg normalisation is CORRECT and
reproduces the reference cov_wgg exactly; only the inputs and items 4/7 change it.

Usage
-----
    python example_scripts/compute_gaussian_covariance.py            # tabulation binning
    python example_scripts/compute_gaussian_covariance.py --validate # reproduce the old cov
"""

import argparse
import os
import sys

import numpy as np
from scipy.special import jv
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HOD_NRV.HOD_analytical.halo_model import HaloModel
from example_scripts.run_tabulated_chains import COSMO_PARAMS, ZEFF, LBOX

FLAM = "/sps/euclid/Users/rpaviot/flamingo"
N_PART_CATALOGUE = 232982044          # rows in hydro_..._0.1percent.parquet
MASS_WEIGHT_EFF = 0.68309             # <m>^2 / <m^2>, exact over that catalogue
NGAL_NISP = 2.37e-3                   # (Mpc/h)^-3, all 744368 NISP galaxies
B1_MEASURED = 1.713                   # DeltaSigma / [b=1, nonlinear P_mm], rp > 8
CHI_MAX_DS = 100.0                    # LOS half-length of the DeltaSigma projection


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rp_min", type=float, default=0.1)
    p.add_argument("--rp_max", type=float, default=50.0)
    p.add_argument("--n_rp", type=int, default=26,
                   help="Number of bin EDGES (tabulation grid: 0.1-50, 26).")
    p.add_argument("--b1", type=float, default=B1_MEASURED)
    p.add_argument("--particle_fraction", type=float, default=0.02,
                   help="Fraction of the 233M particles used in the DeltaSigma "
                        "measurement (measure_hydro_data_vector.py default).")
    p.add_argument("--no_mass_weight", action="store_true",
                   help="Drop the <m>^2/<m^2> effective-density correction.")
    p.add_argument("--nbar", type=float, default=NGAL_NISP)
    p.add_argument("--pi_max", type=float, default=100.0)
    p.add_argument("--chi_max_ds", type=float, default=CHI_MAX_DS,
                   help="Half-length of the DeltaSigma LOS projection, set by the "
                        "measurement (measure_hydro_data_vector.py) and the forward "
                        "model (halo_center_lensing.py), both chi_max=100.")
    p.add_argument("--k_max", type=float, default=1e4,
                   help="Upper limit of the k integral. The shot-noise terms are "
                        "k-independent, so the narrow inner annuli need "
                        "k >> 1/bin-width; 100 h/Mpc leaves them up to 27% short.")
    p.add_argument("--linear", action="store_true",
                   help="Use the linear P_mm instead of the nonlinear one.")
    p.add_argument("--data_path", default=os.path.join(FLAM, "hydro_measured_data_vector.npz"),
                   help="Source of w_gg(R) for Eq. 32's Kronecker term.")
    p.add_argument("--output", default=os.path.join(FLAM, "gaussian_covariance.npz"))
    p.add_argument("--ref_path", default=os.path.join(FLAM, "data_bin0_finalbinning.npz"),
                   help="Reference file used by --validate.")
    p.add_argument("--merge_data", action="store_true",
                   help="Copy delta_sigma / wgg from --data_path into the output, "
                        "producing a fit-ready file for run_tabulated_chains.py "
                        "(requires the measurement to be on the same binning).")
    p.add_argument("--validate", action="store_true",
                   help="Reproduce the ORIGINAL script (z=0, b=1.05, 10%% of "
                        "particles unweighted, full-box projection) on the "
                        "reference binning and compare to data_bin0_finalbinning.npz.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Analytic bin-averaged Bessel filters
# ---------------------------------------------------------------------------

def bar_J(nu, k, rp_bins):
    """
    Annulus-averaged J_nu filter, \bar{J}_nu(k R_i), shape (n_k, n_bins).

    int_r1^r2 2 pi r J_0(kr) dr = (2 pi / k^2) [ x J1(x) ]_{k r1}^{k r2}
    int_r1^r2 2 pi r J_2(kr) dr = (2 pi / k^2) [ -x J1(x) - 2 J0(x) ]_{k r1}^{k r2}
    divided by the annulus area pi (r2^2 - r1^2).
    """
    k = np.atleast_1d(k)[:, None]
    r1, r2 = rp_bins[:-1][None, :], rp_bins[1:][None, :]
    area = np.pi * (r2**2 - r1**2)
    x1, x2 = k * r1, k * r2
    if nu == 0:
        prim = lambda x: x * jv(1, x)
    elif nu == 2:
        prim = lambda x: -x * jv(1, x) - 2.0 * jv(0, x)
    else:
        raise ValueError(f"nu={nu} not implemented")
    return (2.0 * np.pi / k**2) * (prim(x2) - prim(x1)) / area


# ---------------------------------------------------------------------------
# Covariance blocks
# ---------------------------------------------------------------------------

def integrate(k, integrand):
    """int dk k/(2 pi) f(k), done as int dlnk k^2/(2 pi) f(k) on the log grid."""
    kk = k[:, None] if integrand.ndim == 2 else k
    return np.trapezoid(integrand * kk**2 / (2.0 * np.pi), x=np.log(k), axis=0)


def build_covariances(k, P_mm, b1, n_part, n_gal, rp_bins, wgg,
                      V, pi_max, chi_max_ds):
    """Return (cov_ds, cov_wgg, cov_cross) with cov_cross[i, j] = Cov[DS_i, wgg_j]."""
    J0 = bar_J(0, k, rp_bins)                     # (n_k, n_rp)
    J2 = bar_J(2, k, rp_bins)
    n_rp = J0.shape[1]

    P_gg = b1**2 * P_mm
    P_gm = b1 * P_mm
    P_mm_s = P_mm + 1.0 / n_part                  # discreteness
    P_gg_s = P_gg + 1.0 / n_gal

    # LOS window factors int dk_z/(2pi) W_a W_b: 2a for an auto, 2 min(a,b) for
    # the cross. The Gaussian pairing factor (2 for an auto-spectrum, 2 for the
    # gm x gg cross) is carried in the brackets below, NOT here.
    pref_ds = 2.0 * chi_max_ds / V                # == 1/A_s when 2 chi_max_ds = L
    pref_w = 2.0 * pi_max / V
    pref_x = 2.0 * min(chi_max_ds, pi_max) / V

    cov_ds = np.zeros((n_rp, n_rp))
    cov_wg = np.zeros((n_rp, n_rp))
    cov_x = np.zeros((n_rp, n_rp))

    brack_ds = (P_mm_s * P_gg_s + P_gm**2)[:, None]
    brack_w = 2.0 * (P_gg_s**2)[:, None]
    brack_x = 2.0 * (P_gm * P_gg_s)[:, None]

    for i in range(n_rp):
        cov_ds[i] = pref_ds * integrate(k, brack_ds * J2[:, i:i + 1] * J2)
        cov_wg[i] = pref_w * integrate(k, brack_w * J0[:, i:i + 1] * J0)
        cov_x[i] = pref_x * integrate(k, brack_x * J2[:, i:i + 1] * J0)

    cov_ds = 0.5 * (cov_ds + cov_ds.T)
    cov_wg = 0.5 * (cov_wg + cov_wg.T)

    # Eq. 32 Kronecker term: delta_ij / n_g^2 * [w_gg(R_i) + 2 Pi_max].
    # The Poisson piece (2 Pi_max) is already inside the (P_gg + 1/n)^2 integral
    # above, since the bin-averaged Bessels are orthogonal; what is missing is
    # the signal x shot-noise piece, smaller by w_gg(R_i) / (2 Pi_max).
    area_bins = np.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2)
    cov_wg[np.diag_indices(n_rp)] += (2.0 / (V * n_gal**2)) * wgg / area_bins

    return cov_ds, cov_wg, cov_x


def main():
    args = parse_args()

    rp_bins = np.geomspace(args.rp_min, args.rp_max, args.n_rp)
    rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])
    V = LBOX**3
    chi_max_ds = args.chi_max_ds

    # ---- power spectrum (z passed EXPLICITLY; the default is z=0) -----------
    hm = HaloModel(COSMO_PARAMS, hod_type="ELG_GHOD", mass_function="Tinker08",
                   halo_bias="Tinker10", z=ZEFF, mass_definition="MassDef200m",
                   k_array=np.geomspace(1e-5, 100, 512), M_min=1e9, M_max=1e16,
                   units_per_h=True, include_beta_nl=False, verbose=False)
    k_model = hm.get_k()
    z_pk = 0.0 if args.validate else ZEFF
    P_model = (hm.linear_power(z=z_pk) if (args.linear or args.validate)
               else hm.nonlinear_power(z=z_pk))

    # The shot-noise terms are k-independent, so the k integral only converges
    # once k >> 1/(bin width): truncating at the model's k_max=100 h/Mpc leaves
    # the narrow inner annuli up to 27% short. Extend the integration grid with
    # a log-log (power-law) extrapolation of P, which is safe because P is
    # positive and monotonically falling there and is negligible next to 1/n.
    k = np.geomspace(k_model[0], 100.0 if args.validate else args.k_max, 4096)
    P_mm = np.exp(interp1d(np.log(k_model), np.log(P_model), kind="linear",
                           fill_value="extrapolate")(np.log(k)))
    rho_m = hm.get_rho_m() / 1e12                 # -> h Msun / pc^2 per Mpc/h

    # ---- shot noise --------------------------------------------------------
    if args.validate:
        n_part = int(N_PART_CATALOGUE / 10) / V
        b1, chi_max_ds = 1.05, LBOX / 2.0
        rp_bins = np.geomspace(0.2, 40.0, 31)
        rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])
    else:
        n_used = args.particle_fraction * N_PART_CATALOGUE
        if not args.no_mass_weight:
            n_used *= MASS_WEIGHT_EFF
        n_part = n_used / V
        b1 = args.b1

    print(f"z(P_mm)      = {z_pk}   ({'linear' if (args.linear or args.validate) else 'nonlinear'})")
    print(f"b1           = {b1}")
    print(f"n_part       = {n_part:.4e} (Mpc/h)^-3   1/n_part = {1/n_part:.2f}")
    print(f"n_gal        = {args.nbar:.4e} (Mpc/h)^-3   1/n_gal  = {1/args.nbar:.2f}")
    print(f"Pi_max       = {args.pi_max}   chi_max_ds = {chi_max_ds}")
    print(f"rp_bins      = geomspace({rp_bins[0]}, {rp_bins[-1]}, {len(rp_bins)})")

    # ---- w_gg(R) for the Kronecker term ------------------------------------
    d = np.load(args.data_path)
    key = "rp_centers" if "rp_wgg" not in d.files else "rp_wgg"
    wgg = np.exp(interp1d(np.log(d[key]), np.log(d["wgg"]), kind="cubic",
                          bounds_error=False,
                          fill_value=(np.log(d["wgg"][0]), np.log(d["wgg"][-1])))(np.log(rp)))

    cov_ds, cov_wg, cov_x = build_covariances(
        k, P_mm, b1, n_part, args.nbar, rp_bins, wgg, V, args.pi_max, chi_max_ds)
    cov_ds *= rho_m**2
    cov_x *= rho_m

    n = len(rp)
    joint = np.zeros((2 * n, 2 * n))
    joint[:n, :n], joint[n:, n:] = cov_ds, cov_wg
    joint[:n, n:], joint[n:, :n] = cov_x, cov_x.T

    e_ds, e_w = np.sqrt(np.diag(cov_ds)), np.sqrt(np.diag(cov_wg))
    r_x = np.diag(cov_x) / (e_ds * e_w)
    print(f"\n{'rp':>8s} {'err_DS':>10s} {'err_wgg':>10s} {'r(DS,wgg)':>10s}")
    for i in range(0, n, max(1, n // 10)):
        print(f"{rp[i]:8.3f} {e_ds[i]:10.5f} {e_w[i]:10.4f} {r_x[i]:10.3f}")

    ev = np.linalg.eigvalsh(joint)
    print(f"\njoint matrix {joint.shape}, min eigenvalue {ev.min():.3e} "
          f"({'positive definite' if ev.min() > 0 else 'NOT positive definite'})")

    if args.validate:
        ref = np.load(args.ref_path)
        rd = e_ds / np.sqrt(np.diag(ref["cov_delta_sigma"]))
        rw = e_w / np.sqrt(np.diag(ref["cov_wgg"]))
        print(f"\nvs data_bin0_finalbinning.npz (should be ~1.0 for DS, ~sqrt(2) for wgg):")
        print(f"  DeltaSigma  median {np.median(rd):.4f}  min {rd.min():.4f}  max {rd.max():.4f}")
        print(f"  wgg         median {np.median(rw):.4f}  min {rw.min():.4f}  max {rw.max():.4f}")
        return

    extra = {}
    if args.merge_data:
        if not np.allclose(d["rp_bins"], rp_bins, rtol=1e-6):
            raise ValueError(
                f"--merge_data needs the measurement on the same binning: "
                f"data has {len(d['rp_bins'])-1} bins "
                f"[{d['rp_bins'][0]}, {d['rp_bins'][-1]}], "
                f"requested {len(rp_bins)-1} [{rp_bins[0]}, {rp_bins[-1]}]")
        extra = {"delta_sigma": d["delta_sigma"], "wgg": d["wgg"],
                 "delta_sigma_err": np.sqrt(np.diag(cov_ds)),
                 "wgg_err": np.sqrt(np.diag(cov_wg))}
        print("merged delta_sigma + wgg from", args.data_path)

    np.savez(args.output, rp_centers=rp, rp_bins=rp_bins, **extra,
             cov_delta_sigma=cov_ds, cov_wgg=cov_wg, cov_cross=cov_x,
             cov_joint=joint, b1=b1, n_part=n_part, n_gal=args.nbar,
             pi_max=args.pi_max, chi_max_ds=chi_max_ds, z_eff=ZEFF,
             particle_fraction=args.particle_fraction)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
