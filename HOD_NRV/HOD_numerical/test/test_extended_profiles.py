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


def run_all_tests():
    test_exponential_radial_distribution_jax()
    test_exponential_radial_distribution_numba()
    test_ellipticity_jax()
    test_ellipticity_numba()


if __name__ == "__main__":
    print("Testing Extended NFW Profiles (JAX + Numba)")
    print("=" * 50)
    run_all_tests()
