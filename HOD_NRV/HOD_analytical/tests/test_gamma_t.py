"""gamma_t projection of DeltaSigma and the lens-magnification term.

Run with:  python -m pytest -o addopts="" HOD_NRV/HOD_analytical/tests/test_gamma_t.py
"""
from __future__ import annotations

import numpy as np
import pyccl as ccl
import pytest
from scipy.special import jv

from HOD_NRV.HOD_analytical.gamma_t import (
    FOUR_PI_G_OVER_C2, GammaTProjector, sigma_crit_inv_comoving)

H = 0.6766
ARCMIN = np.pi / 180.0 / 60.0


@pytest.fixture(scope="module")
def cosmo():
    return ccl.Cosmology(Omega_c=0.26, Omega_b=0.049, h=H, sigma8=0.81, n_s=0.9665,
                         transfer_function="eisenstein_hu",
                         matter_power_spectrum="halofit")


def _spike(z0, width=0.004, n=41):
    z = np.linspace(z0 - width, z0 + width, n)
    return z, np.exp(-0.5 * ((z - z0) / (width / 4)) ** 2)


def _broad(z0, sig, zmax=3.0, n=600):
    z = np.linspace(0.0, zmax, n)
    return z, np.exp(-0.5 * ((z - z0) / sig) ** 2)


def test_sigma_crit_matches_physical_definition(cosmo):
    """Sigma_c,com^-1 = (1+z_l)^2 * 4 pi G/c^2 * D_l D_ls / D_s (physical D_A)."""
    z_l, z_s = 0.4, 1.1
    a_l, a_s = 1 / (1 + z_l), 1 / (1 + z_s)
    d_l = ccl.angular_diameter_distance(cosmo, a_l)
    d_s = ccl.angular_diameter_distance(cosmo, a_s)
    d_ls = ccl.angular_diameter_distance(cosmo, a_l, a_s)
    expected = (1 + z_l) ** 2 * FOUR_PI_G_OVER_C2 * 1e12 * d_l * d_ls / d_s  # pc^2/Msun
    got = sigma_crit_inv_comoving(cosmo, z_l, z_s, units_per_h=False)[0, 0]
    assert got == pytest.approx(expected, rel=1e-6)
    # h-units: distances in Mpc/h, so the value is h times larger.
    got_h = sigma_crit_inv_comoving(cosmo, z_l, z_s, units_per_h=True)[0, 0]
    assert got_h == pytest.approx(expected * H, rel=1e-6)
    # Sources in front of the lens do not contribute.
    assert sigma_crit_inv_comoving(cosmo, 0.8, 0.5)[0, 0] == 0.0


def test_thin_lens_thin_source_limit(cosmo):
    """Narrow n(z): gamma_t(theta) = Sigma_c^-1(z_l, z_s) DeltaSigma(theta chi_l)."""
    z_l, z_s = 0.4, 1.1
    theta_bins = np.geomspace(1.0, 100.0, 11)
    proj = GammaTProjector(cosmo, theta_bins, *_spike(z_l), *_spike(z_s))
    ds = 30.0 * proj.r_grid ** -0.8                      # power law, R in Mpc/h
    chi_l = ccl.comoving_radial_distance(cosmo, 1 / (1 + z_l)) * H
    sci = sigma_crit_inv_comoving(cosmo, z_l, z_s)[0, 0]
    # exact area-weighted bin average of a power law
    a, b = theta_bins[:-1] * ARCMIN, theta_bins[1:] * ARCMIN
    avg = 2 / (b ** 2 - a ** 2) * (b ** 1.2 - a ** 1.2) / 1.2
    expected = sci * 30.0 * chi_l ** -0.8 * avg
    np.testing.assert_allclose(proj.gamma_t(ds), expected, rtol=2e-3)
    assert proj.z_eff == pytest.approx(z_l, abs=2e-3)


def test_point_mass_with_broad_distributions(cosmo):
    """DeltaSigma = M/(pi R^2): the z_l integral separates from the angle."""
    theta_bins = np.geomspace(2.0, 60.0, 8)
    z_lg, n_lg = _broad(0.5, 0.08)
    z_sg, n_sg = _broad(1.2, 0.3)
    proj = GammaTProjector(cosmo, theta_bins, z_lg, n_lg, z_sg, n_sg)
    m = 1e13
    ds = m / (np.pi * proj.r_grid ** 2) / 1e12           # h Msun/pc^2, R in Mpc/h

    # Independent quadrature on the full grids (z = 0 node carries no weight).
    zl, zs = z_lg[1:], z_sg
    nl = n_lg[1:] / np.trapezoid(n_lg, z_lg)
    ns = n_sg / np.trapezoid(n_sg, z_sg)
    sci = np.trapezoid(sigma_crit_inv_comoving(cosmo, zl, zs) * ns[None, :], zs, axis=1)
    chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + zl)) * H
    z_int = np.trapezoid(nl * sci / chi ** 2, zl)
    a, b = theta_bins[:-1] * ARCMIN, theta_bins[1:] * ARCMIN
    avg = 2 / (b ** 2 - a ** 2) * np.log(b / a)          # <theta^-2> over the annulus
    expected = m / np.pi / 1e12 * z_int * avg
    np.testing.assert_allclose(proj.gamma_t(ds), expected, rtol=3e-3)


def test_units_per_h_gives_the_same_shear(cosmo):
    """gamma_t is dimensionless: h-units and natural units must agree."""
    theta_bins = np.geomspace(1.0, 100.0, 7)
    args = (theta_bins, *_broad(0.5, 0.08), *_broad(1.2, 0.3))
    p_h = GammaTProjector(cosmo, *args, units_per_h=True)
    p_n = GammaTProjector(cosmo, *args, units_per_h=False)
    np.testing.assert_allclose(p_h.r_grid, p_n.r_grid * H, rtol=1e-10)
    ds_n = 20.0 * p_n.r_grid ** -1.0                     # Msun/pc^2 at R [Mpc]
    ds_h = ds_n / H                                      # same profile in h Msun/pc^2
    np.testing.assert_allclose(p_h.gamma_t(ds_h), p_n.gamma_t(ds_n), rtol=1e-8)


def test_theta_unit(cosmo):
    args = (*_spike(0.4), *_spike(1.1))
    p_am = GammaTProjector(cosmo, np.array([6.0, 12.0, 30.0]), *args)
    p_deg = GammaTProjector(cosmo, np.array([0.1, 0.2, 0.5]), *args, theta_unit="deg")
    np.testing.assert_allclose(p_am.r_grid, p_deg.r_grid, rtol=1e-10)
    np.testing.assert_allclose(p_am.matrix, p_deg.matrix, rtol=1e-8, atol=1e-18)


def test_no_source_behind_lens_raises(cosmo):
    with pytest.raises(ValueError, match="behind"):
        GammaTProjector(cosmo, np.array([1.0, 10.0]), *_spike(1.0), *_spike(0.3))


def test_lens_magnification_matches_direct_limber(cosmo):
    """Unit-amplitude term vs a direct Limber + J_2 integral for thin planes.

    Thin lens plane at z_l, thin source plane at z_s:
      C(l) = (3 H0^2 Om / 2 c^2)^2 int_0^chi_l dchi
             (chi_l - chi)(chi_s - chi) / (chi_l chi_s) / a^2 P_nl(l / chi)
    """
    z_l, z_s = 0.6, 1.2
    theta_bins = np.array([10.0, 12.0, 40.0, 48.0])      # arcmin
    proj = GammaTProjector(cosmo, theta_bins, *_spike(z_l), *_spike(z_s))
    got = proj.lens_magnification()

    chi_l = ccl.comoving_radial_distance(cosmo, 1 / (1 + z_l))
    chi_s = ccl.comoving_radial_distance(cosmo, 1 / (1 + z_s))
    pref = 1.5 * (100.0 * H / 299792.458) ** 2 * (cosmo["Omega_c"] + cosmo["Omega_b"])
    chi = np.linspace(1.0, chi_l, 400)
    a = ccl.scale_factor_of_chi(cosmo, chi)
    ell = np.geomspace(2.0, 2e5, 4000)
    pk = np.array([ccl.nonlin_power(cosmo, (ell + 0.5) / c, ai)
                   for c, ai in zip(chi, a)])            # (n_chi, n_ell)
    kern = (chi_l - chi) * (chi_s - chi) / (chi_l * chi_s) / a ** 2
    cl = pref ** 2 * np.trapezoid(kern[:, None] * pk, chi, axis=0)

    def gamma_at(theta_rad):
        return np.trapezoid(ell ** 2 * cl * jv(2, ell * theta_rad), np.log(ell)) / (2 * np.pi)

    expected = []
    for lo, hi in zip(theta_bins[:-1], theta_bins[1:]):
        t = np.linspace(lo, hi, 21) * ARCMIN
        g = np.array([gamma_at(x) for x in t])
        expected.append(2 / (t[-1] ** 2 - t[0] ** 2) * np.trapezoid(g * t, t))
    expected = np.array(expected)
    assert np.all(got > 0)
    np.testing.assert_allclose(got[[0, 2]], expected[[0, 2]], rtol=0.03)
