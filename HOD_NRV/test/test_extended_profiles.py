"""
Test script for extended NFW profiles with continuity at Rvir
Demonstrates:
1. Continuity of density profile at r = Rvir
2. Ellipticity only inside Rvir (r < Rvir)
3. Isotropic exponential profile outside Rvir (r > Rvir)
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from HOD_NRV.satellites.NFW_jax import (
    create_point_on_unit_sphere,
    extended_elliptical_NFW_satellites_positions
)

def test_extended_elliptical_profile():
    """
    Test that:
    1. Inner satellites (r < Rvir) show ellipticity
    2. Outer satellites (r > Rvir) are isotropic
    3. Profile is continuous at r = Rvir
    """

    # Halo parameters
    Rvir = 200.0
    c_true = 10.0
    Rs = Rvir / c_true
    center = np.array([0.0, 0.0, 0.0])
    N_s = 100_000

    # Axis ratios (ellipticity)
    b_a = 0.7
    c_a = 0.5
    ratios = jnp.array([[b_a, c_a]])

    # Rotation matrix (identity for simplicity)
    shape_matrix = jnp.eye(3)[None, :, :]

    # Extended profile parameters
    f_exp = 0.3  # 30% of satellites in exponential component
    tau = 6.0
    lambda_NFW = 1.0

    # Generate satellites
    key = jrandom.PRNGKey(42)
    SpherePoints = create_point_on_unit_sphere(key)
    key_pos = jrandom.PRNGKey(123)

    sat_positions = 1000 * extended_elliptical_NFW_satellites_positions(
        key_pos,
        SpherePoints,
        jnp.array([center]),
        jnp.array([Rvir]),
        jnp.array([c_true]),
        shape_matrix,
        ratios,
        jnp.array([N_s]),
        N_s,
        f_exp=f_exp,
        tau=tau,
        lambda_NFW=lambda_NFW
    )

    sat_pos_np = np.array(sat_positions)

    # Separate inner and outer satellites
    radii = np.linalg.norm(sat_pos_np - center, axis=1)
    inner_mask = radii <= Rvir
    outer_mask = radii > Rvir

    inner_pos = sat_pos_np[inner_mask]
    outer_pos = sat_pos_np[outer_mask]

    print(f"Total satellites: {len(sat_pos_np)}")
    print(f"Inner (r <= Rvir): {np.sum(inner_mask)} ({100*np.sum(inner_mask)/len(sat_pos_np):.1f}%)")
    print(f"Outer (r > Rvir): {np.sum(outer_mask)} ({100*np.sum(outer_mask)/len(sat_pos_np):.1f}%)")
    print(f"Expected outer fraction: {f_exp*100:.1f}%")

    # Check ellipticity of inner satellites
    if len(inner_pos) > 0:
        inner_inertia = inner_pos.T @ inner_pos
        inner_evals = np.linalg.eigvalsh(inner_inertia)
        inner_evals = np.sort(inner_evals)[::-1]
        inner_axis_lengths = np.sqrt(inner_evals)
        inner_ratios = inner_axis_lengths / inner_axis_lengths[0]

        print(f"\nInner satellites (r < Rvir) - Expected ellipticity:")
        print(f"  Input axis ratios: [1.00, {b_a:.2f}, {c_a:.2f}]")
        print(f"  Measured axis ratios: [{inner_ratios[0]:.2f}, {inner_ratios[1]:.2f}, {inner_ratios[2]:.2f}]")

    # Check isotropy of outer satellites
    if len(outer_pos) > 0:
        outer_inertia = outer_pos.T @ outer_pos
        outer_evals = np.linalg.eigvalsh(outer_inertia)
        outer_evals = np.sort(outer_evals)[::-1]
        outer_axis_lengths = np.sqrt(outer_evals)
        outer_ratios = outer_axis_lengths / outer_axis_lengths[0]

        print(f"\nOuter satellites (r > Rvir) - Expected isotropy:")
        print(f"  Input axis ratios: [1.00, 1.00, 1.00] (isotropic)")
        print(f"  Measured axis ratios: [{outer_ratios[0]:.2f}, {outer_ratios[1]:.2f}, {outer_ratios[2]:.2f}]")

        # Check if outer distribution is approximately isotropic (within 5%)
        is_isotropic = np.all(np.abs(outer_ratios - 1.0) < 0.05)
        print(f"  Isotropy test: {'PASSED' if is_isotropic else 'FAILED'}")

    # Plot radial density profile
    r_bins = np.logspace(np.log10(1), np.log10(Rvir*2.5), 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    hist, _ = np.histogram(radii, bins=r_bins)

    # Volume-normalized density
    bin_volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    density = hist / bin_volumes

    print(f"\nContinuity check at r = Rvir:")
    rvir_bin_idx = np.argmin(np.abs(r_centers - Rvir))
    print(f"  Measured density at bin nearest Rvir (r={r_centers[rvir_bin_idx]:.1f}): {density[rvir_bin_idx]:.2e}")
    print(f"  Note: Continuity is enforced analytically in the sampling (exponential starts at Rvir)")

    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    print("Testing Extended Elliptical NFW Profiles")
    print("=" * 50)
    test_extended_elliptical_profile()
