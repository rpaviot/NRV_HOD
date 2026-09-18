"""
Test suite for extended NFW profiles (exponential radial distribution + ellipticity).

Validates:
1. Exponential radial distribution (eq. 7.7: dN/dr ∝ exp(-r/(τ·rs))) in JAX and Numba
2. Elliptical geometry (inner elliptical, outer isotropic) in JAX and Numba
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from scipy.stats import kstest
from HOD_NRV.HOD_numerical.satellites.NFW_jax import (
    extended_NFW_satellites_positions,
    extended_elliptical_NFW_satellites_positions,
)
from HOD_NRV.HOD_numerical.satellites.NFW import (
    extended_NFW_positions_numba,
    extended_elliptical_NFW_positions_numba,
)
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
    satellite_radial_nodes,
)

# Shared halo fixture
_Rvir   = 200.0              # kpc/h
_c      = 10.0
_Rs     = _Rvir / _c         # 20.0 kpc/h
_tau    = 6.0
_Rmax   = 3.0 * _Rvir        # 600.0 kpc/h
_center = np.array([0.0, 0.0, 0.0])  # Mpc/h
_b_a    = 0.7
_c_a    = 0.5
_N_s    = 200_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analytical_fraction_inside(Rvir, Rs, tau, Rmax):
    """Analytical CDF(Rvir) for exponential profile on [0, Rmax]."""
    u_max = 1.0 - np.exp(-Rmax / (tau * Rs))
    return (1.0 - np.exp(-Rvir / (tau * Rs))) / u_max


def _exponential_cdf(r, Rs, tau, Rmax):
    """Theoretical CDF for dN/dr ∝ exp(-r/(tau*Rs)) on [0, Rmax]."""
    u_max = 1.0 - np.exp(-Rmax / (tau * Rs))
    return (1.0 - np.exp(-r / (tau * Rs))) / u_max


def _check_exponential_radii(radii, label):
    """Three shared assertions for the exponential radial distribution."""
    Rvir = _Rvir
    Rs   = _Rs
    tau  = _tau
    Rmax = _Rmax

    # Assertion 1: radii start at r ≈ 0 (not at Rvir)
    assert np.min(radii) < Rvir, (
        f"{label}: min radius {np.min(radii):.2f} kpc/h >= Rvir={Rvir} — "
        "exponential profile should start at r=0"
    )

    # Assertion 2: fraction inside Rvir matches analytical CDF(Rvir)
    F_Rvir = _analytical_fraction_inside(Rvir, Rs, tau, Rmax)
    frac_inside = np.mean(radii <= Rvir)
    assert abs(frac_inside - F_Rvir) < 0.01, (
        f"{label}: fraction inside Rvir = {frac_inside:.4f}, "
        f"expected {F_Rvir:.4f} (tolerance 0.01)"
    )

    # Assertion 3: KS test against theoretical CDF
    stat, pvalue = kstest(radii, lambda r: _exponential_cdf(r, Rs, tau, Rmax))
    assert pvalue > 0.01, (
        f"{label}: KS test failed — stat={stat:.4f}, p-value={pvalue:.4f}"
    )

    print(f"  Min radius:           {np.min(radii):.2f} kpc/h  (< {Rvir} required)")
    print(f"  Fraction inside Rvir: {frac_inside:.4f}  (expected {F_Rvir:.4f})")
    print(f"  KS stat={stat:.4f}, p-value={pvalue:.4f}")


def _check_ellipticity(sat_pos_np, label):
    """Two shared assertions for inner ellipticity and outer isotropy."""
    radii_kpch = np.linalg.norm(sat_pos_np - _center, axis=1) * 1000.0
    inner_mask = radii_kpch <= _Rvir
    outer_mask = ~inner_mask

    inner_pos = sat_pos_np[inner_mask]
    outer_pos = sat_pos_np[outer_mask]

    print(f"  Total: {len(sat_pos_np)},  "
          f"inner (r<=Rvir): {inner_mask.sum()} ({100*inner_mask.mean():.1f}%),  "
          f"outer (r>Rvir): {outer_mask.sum()} ({100*outer_mask.mean():.1f}%)")

    # Assertion 1: outer satellites are isotropic (all axis ratios within 3% of 1.0)
    assert len(outer_pos) > 100, f"{label}: too few outer satellites ({len(outer_pos)})"
    outer_inertia = outer_pos.T @ outer_pos
    outer_evals   = np.sort(np.linalg.eigvalsh(outer_inertia))[::-1]
    outer_lengths = np.sqrt(outer_evals)
    outer_ratios  = outer_lengths / outer_lengths[0]
    assert np.all(np.abs(outer_ratios - 1.0) < 0.03), (
        f"{label}: outer axis ratios {outer_ratios} not within 3% of isotropic"
    )

    # Assertion 2: inner satellites are elliptical (b/a and c/a within 5%)
    assert len(inner_pos) > 100, f"{label}: too few inner satellites ({len(inner_pos)})"
    inner_inertia = inner_pos.T @ inner_pos
    inner_evals   = np.sort(np.linalg.eigvalsh(inner_inertia))[::-1]
    inner_lengths = np.sqrt(inner_evals)
    inner_ratios  = inner_lengths / inner_lengths[0]
    assert abs(inner_ratios[1] - _b_a) < 0.05, (
        f"{label}: inner b/a = {inner_ratios[1]:.4f}, expected {_b_a} (tolerance 0.05)"
    )
    assert abs(inner_ratios[2] - _c_a) < 0.05, (
        f"{label}: inner c/a = {inner_ratios[2]:.4f}, expected {_c_a} (tolerance 0.05)"
    )

    print(f"  Outer axis ratios: "
          f"[{outer_ratios[0]:.3f}, {outer_ratios[1]:.3f}, {outer_ratios[2]:.3f}]"
          f"  (expected [1.000, 1.000, 1.000])")
    print(f"  Inner axis ratios: "
          f"[{inner_ratios[0]:.3f}, {inner_ratios[1]:.3f}, {inner_ratios[2]:.3f}]"
          f"  (expected [1.000, {_b_a:.3f}, {_c_a:.3f}])")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exponential_radial_distribution_jax():
    print("\n[JAX EXP PROFILE]")
    key = jrandom.PRNGKey(42)

    sat_positions = extended_NFW_satellites_positions(
        key,
        jnp.array([_center]),
        jnp.array([_Rvir]),
        jnp.array([_c]),
        jnp.array([_N_s]),
        _N_s,
        f_exp=1.0,
        tau=_tau,
    )

    radii = np.linalg.norm(np.array(sat_positions) - _center, axis=1) * 1000.0
    _check_exponential_radii(radii, "JAX")
    print("  PASSED")


def test_exponential_radial_distribution_numba():
    print("\n[NUMBA EXP PROFILE]")
    np.random.seed(42)

    sat_positions = extended_NFW_positions_numba(
        np.array([_center]),
        np.array([_Rvir]),
        np.array([_c]),
        np.array([_N_s]),
        f_exp=1.0,
        tau=_tau,
    )

    radii = np.linalg.norm(sat_positions - _center, axis=1) * 1000.0
    _check_exponential_radii(radii, "Numba")
    print("  PASSED")


def test_ellipticity_jax():
    print("\n[JAX ELLIPTICAL PROFILE]")
    key = jrandom.PRNGKey(123)

    shape_matrix = jnp.eye(3)[None, :, :]
    ratios = jnp.array([[_b_a, _c_a]])

    sat_positions = extended_elliptical_NFW_satellites_positions(
        key,
        jnp.array([_center]),
        jnp.array([_Rvir]),
        jnp.array([_c]),
        shape_matrix,
        ratios,
        jnp.array([_N_s]),
        _N_s,
        f_exp=0.3,
        tau=_tau,
        lambda_NFW=1.0,
    )

    _check_ellipticity(np.array(sat_positions), "JAX elliptical")
    print("  PASSED")


def test_ellipticity_numba():
    print("\n[NUMBA ELLIPTICAL PROFILE]")
    np.random.seed(123)

    shape_matrix = np.eye(3)[None, :, :]
    axis_ratios  = np.array([[_b_a, _c_a]])

    sat_positions = extended_elliptical_NFW_positions_numba(
        np.array([_center]),
        np.array([_Rvir]),
        np.array([_c]),
        shape_matrix,
        axis_ratios,
        np.array([_N_s]),
        f_exp=0.3,
        tau=_tau,
        lambda_NFW=1.0,
    )

    _check_ellipticity(sat_positions, "Numba elliptical")
    print("  PASSED")


def test_lambda_NFW_rescales_rs_not_the_cutoff():
    """The NFW component must be Rs/lambda truncated at Rvir.

    Rocher+23 sec 7.5 states the prescription as "squeezing the proxy for r_s
    by a factor lambda_NFW, namely r_s -> r_s/lambda_NFW", with the same r_vir.
    Their Fig 16 plots the other reading of the same sentence (r_s kept, the
    cut-off moved to lambda*r_vir); the two differ by an overall radial stretch
    of 1/lambda, so which one is coded is not cosmetic. This pins the code to
    the text: satellites reach Rvir, and the radial CDF is the NFW one with an
    effective concentration c*lambda.
    """
    print("\n[LAMBDA_NFW CONVENTION]")
    lam = 0.67
    key = jrandom.PRNGKey(7)
    sat = extended_NFW_satellites_positions(
        key, jnp.array([_center]), jnp.array([_Rvir]), jnp.array([_c]),
        jnp.array([_N_s]), _N_s, f_exp=0.0, tau=_tau, lambda_NFW=lam,
    )
    radii = np.linalg.norm(np.array(sat) - _center, axis=1) * 1000.0

    assert radii.max() <= _Rvir * 1.001, (
        f"NFW component reaches {radii.max():.1f} kpc/h > Rvir={_Rvir}: the "
        f"cut-off moved, so lambda was applied to the truncation radius")
    c_eff = _c * lam
    def cdf(r):
        x = r * c_eff / _Rvir
        return (np.log1p(x) - x / (1 + x)) / (np.log1p(c_eff) - c_eff / (1 + c_eff))
    stat, pvalue = kstest(radii, cdf)
    assert pvalue > 0.01, (
        f"NFW component is not NFW(c*lambda={c_eff:.2f}) truncated at Rvir: "
        f"KS stat={stat:.4f}, p={pvalue:.4f}")
    # and the *other* reading must be excluded, not merely disfavoured
    frac_beyond_lamRvir = float(np.mean(radii > lam * _Rvir))
    assert frac_beyond_lamRvir > 0.05, (
        "no satellites beyond lambda*Rvir -- the Fig 16 convention is coded")
    print(f"  max radius {radii.max():.1f} kpc/h (Rvir={_Rvir})")
    print(f"  KS vs NFW(c*lambda={c_eff:.2f}): stat={stat:.4f}, p={pvalue:.4f}")
    print(f"  {100 * frac_beyond_lamRvir:.1f}% beyond lambda*Rvir")
    print("  PASSED")


def test_radial_nodes_match_sampler():
    """halo_center_lensing's analytic nodes must equal the mock's sampling.

    The chains never populate a box: TabulatedDeltaSigma convolves the halo
    Sigma with satellite_radial_nodes(). If that quadrature and the JAX
    sampler ever drift apart, the fitted profile parameters mean one thing in
    the likelihood and another in any mock made from the posterior.
    """
    print("\n[NODES vs SAMPLER]")
    x_norm = np.geomspace(1e-4, 1.0, 1000)
    t, w = np.polynomial.legendre.leggauss(400)
    u_nodes, u_w = 0.5 * (t + 1), 0.5 * w
    Rvir_mpc = _Rvir / 1000.0

    for f_exp, tau, lam in ((0.58, 6.14, 0.67), (0.2, 3.0, 1.0),
                            (0.9, 10.0, 1.5)):
        key = jrandom.PRNGKey(11)
        sat = extended_NFW_satellites_positions(
            key, jnp.array([_center]), jnp.array([_Rvir]), jnp.array([_c]),
            jnp.array([_N_s]), _N_s, f_exp=f_exp, tau=tau, lambda_NFW=lam)
        radii = np.linalg.norm(np.array(sat) - _center, axis=1)   # Mpc/h

        r_nodes, w_nodes = satellite_radial_nodes(
            Rvir_mpc, _c, f_exp, tau, lam, u_nodes, u_w, x_norm)
        order = np.argsort(r_nodes)
        r_nodes, w_nodes = r_nodes[order], w_nodes[order]
        cdf_nodes = np.cumsum(w_nodes)

        grid = np.geomspace(1e-4, 3 * Rvir_mpc, 200)
        emp = np.searchsorted(np.sort(radii), grid) / len(radii)
        mod = np.interp(grid, r_nodes, cdf_nodes)
        dmax = float(np.max(np.abs(emp - mod)))
        assert dmax < 0.01, (
            f"nodes vs sampler CDF differ by {dmax:.4f} at "
            f"f_exp={f_exp}, tau={tau}, lambda={lam}")
        print(f"  f_exp={f_exp}, tau={tau}, lambda={lam}: "
              f"max |dCDF| = {dmax:.4f}")
    print("  PASSED")


def run_all_tests():
    test_exponential_radial_distribution_jax()
    test_exponential_radial_distribution_numba()
    test_lambda_NFW_rescales_rs_not_the_cutoff()
    test_radial_nodes_match_sampler()
    test_ellipticity_jax()
    test_ellipticity_numba()


if __name__ == "__main__":
    print("Testing Extended NFW Profiles (JAX + Numba)")
    print("=" * 50)
    run_all_tests()
