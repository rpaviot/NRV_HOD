"""Tangential shear gamma_t(theta) from the halo-model DeltaSigma(R).

DeltaSigma needs a redshift for every lens (to turn angles into R and to get
Sigma_crit). gamma_t needs only the redshift *distribution* of the lens
sample, so it stays usable when individual lens photo-z are poor.

Galaxy-matter term
------------------
In the Limber, flat-sky approximation (Zacharegkas et al. 2022,
arXiv:2106.08438, Eqs. 14-18) the galaxy-matter angular spectrum of a lens at
z_l is C_gm(l) = rho_m Sigma_c^-1 / (a^2 chi^2) P_gm(l/chi), and gamma_t is its
J_2 Hankel transform. Since DeltaSigma(R) = rho_m int k dk/(2 pi) P_gm(k)
J_2(k R) in comoving units, this is exactly

    gamma_t(theta) = int dz_l n_l(z_l) <Sigma_c,com^-1>(z_l)
                     DeltaSigma(R = theta f_K(z_l))

    <Sigma_c,com^-1>(z_l) = (4 pi G / c^2) (1 + z_l) f_K(z_l)
                            int dz_s n_s(z_s) f_K(z_l, z_s) / f_K(z_s)

with comoving DeltaSigma, comoving R and comoving transverse distances f_K.
:class:`GammaTProjector` turns that integral, bin-averaged over the theta
bins, into one fixed matrix acting on DeltaSigma sampled on a log R grid.

Approximation: one DeltaSigma(R) profile (the model at the bin's effective
redshift) is used for the whole lens bin. The geometry is integrated over the
full lens n(z); the evolution of P_gm across the bin is neglected.

Lens magnification
------------------
Magnification by foreground matter changes the lens counts by
2 (alpha - 1) kappa_l, which correlates with the source shear
(arXiv:2106.08438 Eqs. 22-24, following Prat et al. 2022):

    gamma_t^lmag(theta) = 2 (alpha - 1) int l dl/(2 pi) C_{kappa_l kappa_s}(l)
                          J_2(l theta)

C_{kappa_l kappa_s} is the Limber cross-spectrum of the convergence of the
lens n(z) and of the source n(z) with the non-linear matter power spectrum.
It does not depend on the HOD, so it is computed once with pyccl.
"""
from __future__ import annotations

import numpy as np
import pyccl as ccl

# 4 pi G / c^2 in Mpc / Msun  (G = 4.30091727e-9 Mpc km^2 s^-2 Msun^-1)
FOUR_PI_G_OVER_C2 = 4.0 * np.pi * 4.30091727e-9 / 299792.458 ** 2
_C_KMS = 299792.458
_THETA_TO_RAD = {"arcmin": np.pi / 180.0 / 60.0, "deg": np.pi / 180.0, "rad": 1.0}


def _trapz_weights(x):
    """Trapezoid weights w with sum(w * f) = int f dx on the grid x."""
    x = np.asarray(x, dtype=float)
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def _f_K(chi, omega_k, hubble_dist):
    """Comoving transverse distance for comoving radial distance chi."""
    if abs(omega_k) < 1e-8:
        return chi
    s = np.sqrt(abs(omega_k)) / hubble_dist
    return np.sinh(s * chi) / s if omega_k > 0 else np.sin(s * chi) / s


def sigma_crit_inv_comoving(ccl_cosmo, z_lens, z_source, units_per_h=True):
    """Comoving inverse critical surface density, 0 where z_source <= z_lens.

    Returns an array (n_lens, n_source) in pc^2/Msun, or pc^2/(h Msun) if
    ``units_per_h``: it multiplies a comoving DeltaSigma in Msun/pc^2
    (h Msun/pc^2) to give the shear.
    """
    z_l = np.atleast_1d(np.asarray(z_lens, dtype=float))
    z_s = np.atleast_1d(np.asarray(z_source, dtype=float))
    h = ccl_cosmo["h"]
    unit = h if units_per_h else 1.0
    d_h = _C_KMS / (100.0 * h) * unit
    omega_k = ccl_cosmo["Omega_k"]
    chi_l = ccl.comoving_radial_distance(ccl_cosmo, 1.0 / (1.0 + z_l)) * unit
    chi_s = ccl.comoving_radial_distance(ccl_cosmo, 1.0 / (1.0 + z_s)) * unit
    f_l = _f_K(chi_l, omega_k, d_h)[:, None]
    f_s = _f_K(chi_s, omega_k, d_h)[None, :]
    f_ls = _f_K(np.clip(chi_s[None, :] - chi_l[:, None], 0.0, None), omega_k, d_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(f_s > 0, f_ls / f_s, 0.0)
    return FOUR_PI_G_OVER_C2 * 1e12 * (1.0 + z_l)[:, None] * f_l * ratio


class GammaTProjector:
    """DeltaSigma(R) -> bin-averaged gamma_t(theta) for one lens/source pair.

    Parameters
    ----------
    ccl_cosmo : pyccl.Cosmology
    theta_bins : array (nbin + 1,)
        Angular bin edges, in ``theta_unit``.
    z_lens, n_lens : arrays
        Redshift distribution of the lens sample (any normalisation).
    z_source, n_source : arrays
        Redshift distribution of the source sample (any normalisation).
    units_per_h : bool
        True: R in Mpc/h and DeltaSigma in h Msun/pc^2 (comoving), as
        ``HaloModel(units_per_h=True)``. False: Mpc and Msun/pc^2.
    theta_unit : {'arcmin', 'deg', 'rad'}
    n_r : int
        Points of the log R grid on which DeltaSigma is sampled.
    n_gl : int
        Gauss-Legendre nodes per theta bin for the area-weighted average.

    Attributes
    ----------
    r_grid : array (n_r,)
        Comoving R at which the model DeltaSigma must be evaluated.
    matrix : array (nbin, n_r)
        ``gamma_t = matrix @ DeltaSigma(r_grid)``.
    theta : array (nbin,)
        Area-weighted mean angle of each bin, in ``theta_unit``.
    z_eff : float
        Lens redshift weighted by n_l(z) <Sigma_c^-1>(z): the redshift the
        galaxy-matter signal comes from.
    sigma_crit_inv : array
        <Sigma_c,com^-1>(z_l) on ``z_lens_grid``.
    """

    def __init__(self, ccl_cosmo, theta_bins, z_lens, n_lens, z_source, n_source,
                 units_per_h=True, theta_unit="arcmin", n_r=300, n_gl=12):
        if theta_unit not in _THETA_TO_RAD:
            raise ValueError(f"theta_unit must be one of {list(_THETA_TO_RAD)}")
        self.ccl_cosmo = ccl_cosmo
        self.units_per_h = units_per_h
        self.theta_unit = theta_unit
        self.theta_bins = np.asarray(theta_bins, dtype=float)
        if self.theta_bins.ndim != 1 or self.theta_bins.size < 2 \
                or np.any(np.diff(self.theta_bins) <= 0) or self.theta_bins[0] <= 0:
            raise ValueError("theta_bins must be increasing, positive bin edges")
        self.nbin = self.theta_bins.size - 1

        self.z_lens_full, self.n_lens_full = self._normalised(z_lens, n_lens, "lens")
        self.z_source, self.n_source = self._normalised(z_source, n_source, "source")

        # Lens quadrature nodes: drop z <= 0 and nodes of negligible weight
        # (they would only stretch the R grid).
        w_l = _trapz_weights(self.z_lens_full) * self.n_lens_full
        keep = (w_l > 1e-10 * w_l.max()) & (self.z_lens_full > 0)
        self.z_lens_grid = self.z_lens_full[keep]
        w_l = w_l[keep] / w_l[keep].sum()

        w_s = _trapz_weights(self.z_source) * self.n_source
        sci = sigma_crit_inv_comoving(ccl_cosmo, self.z_lens_grid, self.z_source,
                                      units_per_h)
        self.sigma_crit_inv = sci @ w_s                      # (n_l,)
        q = w_l * self.sigma_crit_inv                        # lensing weight per z_l
        if not q.sum() > 0:
            raise ValueError("no source lies behind the lens distribution")
        self.z_eff = float((q * self.z_lens_grid).sum() / q.sum())
        self.mean_sigma_crit_inv = float(q.sum())

        h = ccl_cosmo["h"]
        unit = h if units_per_h else 1.0
        chi = ccl.comoving_radial_distance(
            ccl_cosmo, 1.0 / (1.0 + self.z_lens_grid)) * unit
        f_k = _f_K(chi, ccl_cosmo["Omega_k"], _C_KMS / (100.0 * h) * unit)

        # Area-weighted Gauss-Legendre nodes inside each theta bin:
        # <g> = 2 / (b^2 - a^2) int_a^b g(t) t dt.
        x, w = np.polynomial.legendre.leggauss(n_gl)
        a, b = self.theta_bins[:-1], self.theta_bins[1:]
        self._theta_nodes = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (a + b)[:, None]
        self._theta_w = (w[None, :] * self._theta_nodes
                         * ((b - a) / (b ** 2 - a ** 2))[:, None])       # rows sum to 1
        self.theta = (self._theta_w * self._theta_nodes).sum(axis=1)

        theta_rad = self._theta_nodes * _THETA_TO_RAD[theta_unit]         # (nbin, n_gl)
        r = theta_rad[:, :, None] * f_k[None, None, :]                    # (nbin, n_gl, n_l)
        self.r_grid = np.geomspace(r.min(), r.max(), n_r)
        dln = np.log(self.r_grid[1] / self.r_grid[0])
        t = np.log(r / self.r_grid[0]) / dln
        j = np.clip(np.floor(t).astype(int), 0, n_r - 2)
        frac = t - j
        coeff = self._theta_w[:, :, None] * q[None, None, :]
        rows = np.broadcast_to(np.arange(self.nbin)[:, None, None], r.shape)
        self.matrix = np.zeros((self.nbin, n_r))
        np.add.at(self.matrix, (rows, j), coeff * (1.0 - frac))
        np.add.at(self.matrix, (rows, j + 1), coeff * frac)

    @staticmethod
    def _normalised(z, n, name):
        z = np.asarray(z, dtype=float)
        n = np.asarray(n, dtype=float)
        if z.shape != n.shape or z.ndim != 1 or z.size < 2:
            raise ValueError(f"{name} n(z): z and n must be 1-D arrays of equal length")
        if np.any(np.diff(z) <= 0) or np.any(n < 0):
            raise ValueError(f"{name} n(z): z must increase and n must be >= 0")
        norm = (_trapz_weights(z) * n).sum()
        if not norm > 0:
            raise ValueError(f"{name} n(z) integrates to zero")
        return z, n / norm

    def gamma_t(self, delta_sigma):
        """Bin-averaged gamma_t from DeltaSigma evaluated on ``r_grid``."""
        return self.matrix @ np.asarray(delta_sigma)

    def lens_magnification(self, ell_max=2.0e5, n_ell=400):
        """Bin-averaged lens-magnification term for unit 2 (alpha - 1).

        The full contribution is ``2 * (alpha - 1) * lens_magnification()``.
        Uses the non-linear matter power spectrum of ``ccl_cosmo``.
        """
        t_l = ccl.WeakLensingTracer(self.ccl_cosmo,
                                    dndz=(self.z_lens_full, self.n_lens_full))
        t_s = ccl.WeakLensingTracer(self.ccl_cosmo,
                                    dndz=(self.z_source, self.n_source))
        ell = np.unique(np.concatenate([np.arange(2.0, 50.0),
                                        np.geomspace(50.0, ell_max, n_ell)]))
        cl = ccl.angular_cl(self.ccl_cosmo, t_l, t_s, ell)
        theta_deg = np.degrees(self._theta_nodes * _THETA_TO_RAD[self.theta_unit])
        g = ccl.correlation(self.ccl_cosmo, ell=ell, C_ell=cl,
                            theta=theta_deg.ravel(), type="NG", method="FFTLog")
        return (self._theta_w * g.reshape(theta_deg.shape)).sum(axis=1)
