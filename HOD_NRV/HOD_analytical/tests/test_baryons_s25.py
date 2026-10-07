"""Schneider+25 JAX profiles against BaryonForge (upstream fafc0f3) and against
exact / brute-force Fourier transforms.

The reference ``data/s25_bfg_reference.npz`` was made on cc with
``~/flamingo_analysis/scripts/make_s25_reference.py``: BaryonForge real-space
profiles for M200c = 10^11.5, 10^12, 10^13, 10^14 Msun/h at z = 1 with the
DMO_B2 box parameters.

Run with the NRV env:
    ~/miniconda3/envs/NRV_ENV/bin/python -m pytest HOD_NRV/HOD_analytical/tests/test_baryons_s25.py
"""
from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp
import pytest

from HOD_NRV.HOD_analytical.baryons_s25 import (
    S25Grid, S25_PARAM_NAMES, complete_params, s25_flamingo_m8_z1,
    s25_mass_profiles, s25_density, s25_fourier, m200m_to_m200c, _fftlog_j0,
)

REF = np.load(os.path.join(os.path.dirname(__file__), "data", "s25_bfg_reference.npz"))
GRID = S25Grid()


def _ref_inputs():
    p = complete_params({n[4:]: float(REF[n]) for n in REF.files
                         if n.startswith("par_") and n[4:] in S25_PARAM_NAMES})
    f_bar = float(REF["Omega_b"] / REF["Omega_m"])
    return (REF["M200c"], REF["R200c"], REF["c200c"], REF["nu"], float(REF["z"]), f_bar, p)


@pytest.fixture(scope="module")
def prof_bf():
    """Profiles with BaryonForge's 10 Mpc normalisation window."""
    M, R, c, nu, z, f_bar, p = _ref_inputs()
    return s25_mass_profiles(GRID, M, R, c, nu, z, f_bar, p, 10.0)


def _on_ref_r(rho, i):
    xr = REF["r"] / REF["R200c"][i]
    return np.exp(np.interp(np.log(xr), np.asarray(GRID.lnx), np.log(np.maximum(rho[i], 1e-300))))


def test_flamingo_params_match_reference_box():
    """The module's FLAMINGO m8 set is the one the DMO_B2 box was made with."""
    ours = s25_flamingo_m8_z1(float(REF["h"]))
    for name, v in ours.items():
        np.testing.assert_allclose(v, float(REF["par_" + name]), rtol=1e-12, err_msg=name)


def test_unknown_parameter_raises():
    with pytest.raises(KeyError, match="mu"):
        complete_params({**s25_flamingo_m8_z1(0.68), "mu": 0.7})


def test_fractions_match_baryonforge(prof_bf):
    for key in ("f_star", "f_cga", "f_sga", "f_iga"):
        np.testing.assert_allclose(np.asarray(prof_bf[key]), REF[key], rtol=1e-10, err_msg=key)
    # f_hga = f_bar - f_star - f_iga; REF's Omega_m adds Omega_nu by hand,
    # BaryonForge takes it from CCL: 1e-5 difference in f_bar.
    np.testing.assert_allclose(np.asarray(prof_bf["f_hga"]), REF["f_hga"], rtol=5e-5)
    np.testing.assert_allclose(np.asarray(prof_bf["beta"]), REF["beta"], rtol=1e-10)


@pytest.mark.parametrize("key, ref_key, tol", [
    ("dmo", "rho_DarkMatter", 1e-3),
    ("cga", "rho_Stars", 5e-3),
    ("hga", "rho_HotGas", 5e-3),
    ("iga", "rho_InnerGas", 5e-3),
    ("clm", "rho_CollisionlessMatter", 5e-3),
])
def test_components_match_baryonforge(prof_bf, key, ref_key, tol):
    """Each component, undoing our DMB renormalisation factor (see next test)."""
    rho = np.asarray(s25_density(prof_bf, GRID, REF["R200c"], key))
    if key != "dmo":
        rho = rho / np.asarray(prof_bf["factor"])[:, None]
    r = REF["r"]
    for i in range(len(REF["M200c"])):
        ref = REF[ref_key][i]
        sel = (r > 0.02) & (r < 8.0) & (ref > 1e-6 * ref.max())
        np.testing.assert_allclose(_on_ref_r(rho, i)[sel], ref[sel], rtol=tol,
                                   err_msg=f"{key}, mass index {i}")


def test_dmb_conserves_mass_where_baryonforge_rescales(prof_bf):
    """Our one-halo DMB integrates to M_tot to 0.2%. BaryonForge's
    DarkMatterBaryon integrates from 1e-5 Mpc only and misses the dark matter
    the Q1 (central-galaxy) term contracts below ~10 pc (xi -> inf as r -> 0),
    so its renormalisation factor is 1.001-1.008 and its profile sits that much
    high at every radius. The shapes agree once both factors are divided out."""
    np.testing.assert_allclose(np.asarray(prof_bf["factor"]), 1.0, atol=2e-3)
    comp = REF["rho_CollisionlessMatter"] + REF["rho_Stars"] + REF["rho_HotGas"] + REF["rho_InnerGas"]
    bf_factor = np.median(REF["rho_DMB_1h"] / comp, axis=1)
    assert np.all((bf_factor > 1.0) & (bf_factor < 1.01))
    rho = np.asarray(s25_density(prof_bf, GRID, REF["R200c"], "dmb")) / np.asarray(prof_bf["factor"])[:, None]
    r = REF["r"]
    sel = (r > 0.02) & (r < 8.0)
    for i in range(len(REF["M200c"])):
        np.testing.assert_allclose(_on_ref_r(rho, i)[sel], (REF["rho_DMB_1h"][i] / bf_factor[i])[sel],
                                   rtol=5e-3, err_msg=f"mass index {i}")


def test_fftlog_gaussian():
    """int sqrt(2/pi) x^3 e^{-x^2/2} j0(kx) dlnx = e^{-k^2/2} exactly."""
    x = np.asarray(GRID.x)
    f = np.sqrt(2.0 / np.pi) * x ** 3 * np.exp(-x ** 2 / 2.0)
    G = np.asarray(_fftlog_j0(jnp.asarray(f), GRID))
    y = np.exp(np.asarray(GRID.lny))
    sel = (y > 1e-3) & (y < 6.0)
    np.testing.assert_allclose(G[sel], np.exp(-y[sel] ** 2 / 2.0), atol=1e-4)


def test_fourier_against_direct_quadrature():
    """FFTLog u(k) of the S25 profiles vs brute-force j0 quadrature."""
    M, R, c, nu, z, f_bar, p = _ref_inputs()
    k = np.geomspace(0.05, 50.0, 12)                       # 1/Mpc
    u_dmb, u_dmo = (np.asarray(a) for a in s25_fourier(GRID, k, M, R, c, nu, z, f_bar, p))
    prof = s25_mass_profiles(GRID, M, R, c, nu, z, f_bar, p)
    lnx = np.asarray(GRID.lnx)
    sel = lnx < np.log(1e3)                                # f ~ x^-4 beyond; avoids j0 aliasing
    for i in range(len(M)):
        kx = k[:, None] * R[i] * np.exp(lnx[sel])[None, :]
        j0 = np.sinc(kx / np.pi)
        for name, u in (("dmb", u_dmb), ("dmo", u_dmo)):
            f = np.asarray(prof[name])[i, sel] / float(prof["M_tot"][i])
            direct = np.trapezoid(f[None, :] * j0, lnx[sel], axis=1)
            if name == "dmb":
                direct = direct + float(prof["point"][i] / prof["M_tot"][i])
            np.testing.assert_allclose(u[:, i], direct, rtol=2e-3, atol=1e-4,   # FFTLog floor ~6e-5
                                       err_msg=f"{name}, mass index {i}")


def test_fourier_normalised_and_suppressed():
    """u -> 1 at k -> 0 for both; baryons push mass out, so u_DMB < u_DMO
    at k R200c ~ 1-10 for the low-mass halos."""
    M, R, c, nu, z, f_bar, p = _ref_inputs()
    k = np.array([1e-4, 3.0, 10.0])
    u_dmb, u_dmo = (np.asarray(a) for a in s25_fourier(GRID, k, M, R, c, nu, z, f_bar, p))
    np.testing.assert_allclose(u_dmo[0], 1.0, atol=1e-4)
    np.testing.assert_allclose(u_dmb[0], 1.0, atol=2e-3)
    assert np.all(u_dmb[1:, :2] < u_dmo[1:, :2])


def test_m200m_to_m200c_inverts_colossus_ratio():
    """At z = 1 with Omega_m(z) ~ 0.77 a c200m = 5 halo has M200c/M200m ~ 0.92."""
    M200c, r_ratio, c200c = m200m_to_m200c(1e12, 5.0, 0.774)
    np.testing.assert_allclose(M200c / 1e12, 0.919, atol=2e-3)
    np.testing.assert_allclose(c200c, 5.0 * r_ratio, rtol=1e-12)


def test_halomodel_baryon_hook():
    """HaloModel.set_baryons: S25 lowers DeltaSigma at 0.2-1 Mpc/h for an ELG HOD
    (z=1, M200m), leaves large scales alone, and None restores P_gm bit for bit."""
    import warnings
    warnings.filterwarnings("ignore")
    from HOD_NRV.HOD_analytical.halo_model import HaloModel
    cosmo = {"h": 0.681, "Omc": 0.306 - 0.0486 - 1.39e-3, "Omb": 0.0486,
             "A_s": 2.099e-9, "n_s": 0.967, "Omnu": 1.39e-3}
    hod = {"Ac": 1.0, "As": 2.0, "log10Mmin": 11.626, "sig_M": 0.599, "gamma": 4.08,
           "log10M1": 13.04, "alpha": 0.859, "kappa": 1.01}
    m = HaloModel(cosmo, z=1.0, hod_type="ELG_MHMQ", units_per_h=True,
                  mass_definition="MassDef200m", verbose=False)
    m.set_hod_params(hod)
    edges = np.geomspace(0.1, 50, 26)
    rc = np.sqrt(edges[1:] * edges[:-1])
    P0 = np.asarray(m.Pgm())
    _, d0 = m.DeltaSigma(rc, rp_bins=edges, include_stellar=False)
    m.set_baryons("s25")
    _, d1 = m.DeltaSigma(rc, rp_bins=edges, include_stellar=False)
    ratio = d1 / d0
    assert np.all(ratio[(rc > 0.2) & (rc < 1.0)] < 0.98)
    np.testing.assert_allclose(ratio[rc > 3.0], 1.0, atol=2e-3)
    m.set_baryons(None)
    assert np.array_equal(np.asarray(m.Pgm()), P0)
