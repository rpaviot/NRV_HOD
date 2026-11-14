"""End to end: HaloModel.GammaT and the 'GammaT' observable of CSMFFitter.

Run with:  python -m pytest -o addopts="" HOD_NRV/HOD_analytical/tests/test_gamma_t_fitter.py
"""
from __future__ import annotations

import numpy as np
import pyccl as ccl
import pytest

from HOD_NRV.HOD_analytical import CSMFFitter, HaloModel
from HOD_NRV.HOD_analytical.gamma_t import sigma_crit_inv_comoving
from HOD_NRV.HOD_analytical.sampler import DEFAULT_COSMO_PARAMS, FIDUCIAL_CSMF_PARAMS

ARCMIN = np.pi / 180.0 / 60.0
HOD = {k: FIDUCIAL_CSMF_PARAMS[k] for k in
       ('M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1')}
Z_EFF = (0.3, 0.6)
LOGMS = ((10.2, 12.5), (10.8, 12.5))
LOGMS_MED = (10.6, 11.1)
THETA_BINS = np.geomspace(1.0, 100.0, 11)
ALPHA = {0: 1.6, 1: 2.4}


def _gauss(z0, sig, zmax=3.0, n=400):
    z = np.linspace(0.0, zmax, n)
    return z, np.exp(-0.5 * ((z - z0) / sig) ** 2)


@pytest.fixture(scope="module")
def model():
    hm = HaloModel(cosmo_params=DEFAULT_COSMO_PARAMS, z=np.array(Z_EFF),
                   hod_type='CSMF',
                   Mstar_min=np.array([10 ** m[0] for m in LOGMS]),
                   Mstar_max=np.array([10 ** m[1] for m in LOGMS]),
                   median_Mstar=np.array([10 ** m for m in LOGMS_MED]),
                   units_per_h=True, masses_are_log10=True, verbose=False,
                   mass_function='Tinker08')
    hm.set_hod_params(HOD)
    hm.update_f(f_h=1.0, f_s=1.0)
    return hm


def test_gamma_t_thin_planes_equals_sigma_crit_times_delta_sigma(model):
    """Narrow lens and source planes: gamma_t(theta) bins are DeltaSigma
    averaged over the matching R bins, times Sigma_c^-1."""
    z_s = 1.2
    projs, expected = [], []
    _, ds_dummy = model.DeltaSigma(rp=np.array([1.0]), rp_bins=np.array([0.9, 1.1]))
    for iz, z_l in enumerate(Z_EFF):
        zl = np.linspace(z_l - 0.002, z_l + 0.002, 21)
        zs = np.linspace(z_s - 0.002, z_s + 0.002, 21)
        tophat = np.ones(21)
        projs.append(model.make_gamma_t_projector(THETA_BINS, zl, tophat, zs, tophat))
        chi = ccl.comoving_radial_distance(model.ccl_cosmo, 1 / (1 + z_l)) * model.h
        rp_bins = THETA_BINS * ARCMIN * chi
        rp = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        _, ds = model.DeltaSigma(rp=rp, rp_bins=rp_bins)
        sci = sigma_crit_inv_comoving(model.ccl_cosmo, z_l, z_s)[0, 0]
        expected.append(sci * ds[iz])
    got = model.GammaT(projs)
    for g, e in zip(got, expected):
        assert np.all(g > 0)
        np.testing.assert_allclose(g, e, rtol=5e-3)


def _write_mock(tmp_path, model, with_magnification):
    """Noise-free gamma_t files for the two bins; returns the projectors."""
    projs = []
    for iz in range(2):
        z_l, n_l = _gauss(Z_EFF[iz], 0.06)
        z_s, n_s = _gauss(1.1, 0.3)
        projs.append(model.make_gamma_t_projector(THETA_BINS, z_l, n_l, z_s, n_s))
    model.set_hod_params(HOD)
    model.update_f(f_h=1.0, f_s=1.0)
    gt = model.GammaT(projs)
    for iz in range(2):
        g = np.array(gt[iz])
        if with_magnification:
            g = g + 2.0 * (ALPHA[iz] - 1.0) * projs[iz].lens_magnification()
        cov = np.diag((0.05 * np.abs(gt[iz])) ** 2)
        np.savez(tmp_path / f"gt_bin{iz}.npz", z_eff=Z_EFF[iz],
                 logmstar_min=LOGMS[iz][0], logmstar_max=LOGMS[iz][1],
                 logmstar_median=LOGMS_MED[iz],
                 theta=projs[iz].theta, theta_bins=THETA_BINS,
                 gamma_t=g, cov_gamma_t=cov,
                 z_lens=projs[iz].z_lens_full, nz_lens=projs[iz].n_lens_full,
                 z_source=projs[iz].z_source, nz_source=projs[iz].n_source)
    return projs


def _fitter(tmp_path, alpha_values=None, **kwargs):
    fitter = CSMFFitter(observables=['GammaT'], verbose=False, **kwargs)
    fitter.load_data(str(tmp_path), 'gt_bin{}.npz', [0, 1], name='mock',
                     alpha_values=alpha_values)
    fitter.set_priors(fixed_params={'f_h': 1.0, 'f_s': 1.0})
    fitter._initialize_halo_model()      # as minimize() / run() do
    return fitter


def test_fitter_recovers_truth_on_noise_free_gamma_t(tmp_path, model):
    _write_mock(tmp_path, model, with_magnification=False)
    fitter = _fitter(tmp_path)
    logl_true = fitter.log_likelihood(HOD)
    assert logl_true == pytest.approx(0.0, abs=1e-6)
    assert fitter._n_data_points == 20
    shifted = dict(HOD, M1=HOD['M1'] + 0.3)
    assert fitter.log_likelihood(shifted) < -1.0
    pred = fitter.compute_model_prediction(HOD)
    assert pred[('mock', 0)]['gamma_t_model'].shape == (10,)
    assert pred[('mock', 0)]['delta_sigma_model'] is None


def test_theta_cuts(tmp_path, model):
    _write_mock(tmp_path, model, with_magnification=False)
    fitter = _fitter(tmp_path, theta_min=3.0, theta_max=50.0)
    fitter.log_likelihood(HOD)
    theta = fitter.mass_bins[0].theta
    n_kept = int(np.sum((theta >= 3.0) & (theta <= 50.0)))
    assert 0 < n_kept < 10
    assert fitter._n_data_points == 2 * n_kept


def test_lens_magnification_is_removed_when_alpha_is_given(tmp_path, model):
    """Data = galaxy-matter + 2(alpha-1) magnification. With the alphas the
    fit sits at the truth; without them the truth is disfavoured."""
    _write_mock(tmp_path, model, with_magnification=True)
    with_alpha = _fitter(tmp_path, alpha_values=ALPHA)
    assert with_alpha.log_likelihood(HOD) == pytest.approx(0.0, abs=1e-6)
    mag = [mb.alpha_minus_one * mb.mag_gamma_t for mb in with_alpha.mass_bins]
    assert all(np.all(m > 0) for m in mag)
    without = _fitter(tmp_path)
    assert without.log_likelihood(HOD) < -1e-3


def test_gamma_t_needs_gamma_t_data(tmp_path):
    np.savez(tmp_path / "ds_bin0.npz", z_eff=0.3, logmstar_min=10.2,
             logmstar_max=12.5, rp=np.array([1.0, 2.0]),
             rp_bins=np.array([0.7, 1.4, 2.8]), delta_sigma=np.array([10.0, 5.0]),
             delta_sigma_err=np.array([1.0, 1.0]), cov_delta_sigma=np.eye(2))
    fitter = CSMFFitter(observables=['GammaT'], verbose=False)
    fitter.load_data(str(tmp_path), 'ds_bin{}.npz', [0])
    fitter.set_priors(fixed_params={'f_h': 1.0, 'f_s': 1.0})
    with pytest.raises(ValueError, match="no gamma_t data"):
        fitter._initialize_halo_model()
