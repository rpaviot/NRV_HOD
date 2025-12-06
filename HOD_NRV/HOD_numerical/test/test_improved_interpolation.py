"""
Test improved interpolation methods on the same galaxy sample from test_fast_dsigma.py

This is a direct follow-up to test_fast_dsigma.py that uses the SAME:
- Pre-computed lensing grid
- Galaxy positions (from HOD population)
- Ground truth ΔΣ (from standard method)

Goal: Find which interpolation method reduces the ~98% error from original IDW.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.twopoint_calculator.improved_interpolation import (
    ImprovedDeltaSigmaInterpolator,
    AdaptiveDeltaSigmaInterpolator
)
from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import load_precomputed_lensing
from HOD_NRV.HOD_numerical.twopoint_calculator.fast_two_point import FastDeltaSigmaCalculator
from test_utils import print_header


def test_improved_interpolation():
    """
    Test improved interpolation methods using data from test_fast_dsigma.py
    """

    print_header("Improved Interpolation Method Test")

    # Paths (same as test_fast_dsigma.py)
    output_dir = Path(__file__).parent / 'output'
    precomputed_file = output_dir / 'precomputed_lensing_grid.h5'
    comparison_file = output_dir / 'lensing_comparison.npz'

    if not precomputed_file.exists():
        print(f"\n❌ ERROR: Pre-computed grid not found: {precomputed_file}")
        print("Please run test_fast_dsigma.py first!")
        return

    if not comparison_file.exists():
        print(f"\n❌ ERROR: Comparison data not found: {comparison_file}")
        print("Please run the comparison step in test_fast_dsigma.py first!")
        return

    # ========================================================================
    # Step 1: Load Comparison Data
    # ========================================================================
    print_header("Step 1: Load Comparison Data")

    print("\n  Loading comparison results from test_fast_dsigma.py...")
    data = np.load(comparison_file, allow_pickle=True)

    # Extract saved data
    rp_centers = data['rp']
    ds_standard = data['ds_standard']  # Ground truth
    ds_fast_original = data['ds_fast']  # Original IDW result

    # Reconstruct rp_bins from centers
    rp_bins = np.zeros(len(rp_centers) + 1)
    rp_bins[1:-1] = np.sqrt(rp_centers[:-1] * rp_centers[1:])
    rp_bins[0] = rp_centers[0]**2 / rp_bins[1]
    rp_bins[-1] = rp_centers[-1]**2 / rp_bins[-2]

    print(f"    ✓ Loaded ground truth ΔΣ: {len(ds_standard)} radial bins")
    print(f"    ✓ Loaded original fast ΔΣ: {len(ds_fast_original)} radial bins")
    print(f"    ✓ rp range: [{rp_centers[0]:.3f}, {rp_centers[-1]:.3f}] Mpc/h")

    # Compute original error for reference
    rel_diff_original = 100 * np.abs(ds_fast_original - ds_standard) / (ds_standard + 1e-10)
    rms_original = np.sqrt(np.mean(rel_diff_original**2))

    print(f"\n  Original IDW (k=8) performance:")
    print(f"    RMS error: {rms_original:.2f}%")
    print(f"    Mean error: {np.mean(rel_diff_original):.2f}%")
    print(f"    Median error: {np.median(rel_diff_original):.2f}%")

    # ========================================================================
    # Step 2: Load Pre-computed Grid and Galaxy Positions
    # ========================================================================
    print_header("Step 2: Load Pre-computed Grid")

    print("\n  Loading pre-computed lensing grid...")
    positions, deltasigma, rp_bins_pre, metadata = load_precomputed_lensing(str(precomputed_file))
    Lbox = metadata.get('Lbox', 681.0)  # Default from Flamingo

    print(f"    ✓ Pre-computed positions: {len(positions):,}")
    print(f"    ✓ ΔΣ values shape: {deltasigma.shape}")
    print(f"    ✓ Box size: {Lbox} Mpc/h")

    # Get galaxy positions - need to re-populate with same parameters
    print("\n  Note: Galaxy positions must be regenerated with same HOD parameters")
    print("        Loading galaxy positions from saved HOD population...")

    # Try to load galaxy positions from npz if available
    if 'galaxy_positions' in data:
        galaxy_positions = data['galaxy_positions']
        print(f"    ✓ Loaded {len(galaxy_positions):,} galaxy positions")
    else:
        print(f"    ⚠  Galaxy positions not saved in comparison file")
        print(f"       Will use HOD to regenerate...")
        galaxy_positions = regenerate_galaxy_positions(data, Lbox)

    n_gal = len(galaxy_positions)

    # ========================================================================
    # Step 3: Initialize Interpolation Methods
    # ========================================================================
    print_header("Step 3: Initialize Interpolation Methods")

    methods = {}

    print("\n  1. Original IDW (k=8) - for reference")
    methods['Original_IDW_k8'] = FastDeltaSigmaCalculator(
        str(precomputed_file),
        interpolation_method='idw',
        k_neighbors=8,
        Lbox=Lbox
    )

    print("\n  2. Kernel Smoothing (k=32, auto-bandwidth, Gaussian)")
    methods['Kernel_k32_auto'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=32,
        bandwidth=None,  # Auto-compute
        kernel='gaussian'
    )

    print("\n  3. Kernel Smoothing (k=32, bandwidth=10 Mpc/h)")
    methods['Kernel_k32_bw10'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=32,
        bandwidth=10.0,
        kernel='gaussian'
    )

    print("\n  4. Kernel Smoothing (k=64, bandwidth=15 Mpc/h)")
    methods['Kernel_k64_bw15'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=64,
        bandwidth=15.0,
        kernel='gaussian'
    )

    print("\n  5. Shepard's Method (k=32, power=3)")
    methods['Shepard_k32_p3'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='shepard',
        k_neighbors=32
    )

    print("\n  6. Adaptive Kernel (k=16-64)")
    methods['Adaptive_16_64'] = AdaptiveDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        k_min=16, k_max=64
    )

    # ========================================================================
    # Step 4: Test Each Method
    # ========================================================================
    print_header("Step 4: Test Interpolation Methods")

    results = {}

    for name, interpolator in methods.items():
        print(f"\n  Testing {name}...")
        t0 = time.time()

        if name == 'Original_IDW_k8':
            # Use FastDeltaSigmaCalculator interface
            _, ds = interpolator.compute_deltasigma_for_galaxies(
                galaxy_positions, rp_bins=rp_bins
            )
        else:
            # Use ImprovedDeltaSigmaInterpolator interface
            ds_sum = np.zeros(len(rp_bins) - 1)
            for i, gal_pos in enumerate(galaxy_positions):
                if (i + 1) % max(1, n_gal // 10) == 0:
                    print(f"      Progress: {i+1}/{n_gal} ({100*(i+1)/n_gal:.0f}%)")
                ds_i = interpolator.interpolate_at_position(gal_pos)
                ds_sum += ds_i
            ds = ds_sum / n_gal

        elapsed = time.time() - t0

        # Compute errors
        rel_diff = 100 * np.abs(ds - ds_standard) / (ds_standard + 1e-10)
        mean_err = np.mean(rel_diff)
        median_err = np.median(rel_diff)
        max_err = np.max(rel_diff)
        rms_err = np.sqrt(np.mean(rel_diff**2))

        results[name] = {
            'ds': ds,
            'time': elapsed,
            'mean_err': mean_err,
            'median_err': median_err,
            'max_err': max_err,
            'rms_err': rms_err,
            'rel_diff': rel_diff
        }

        print(f"      Time: {elapsed:.2f} s")
        print(f"      RMS error: {rms_err:.1f}%")
        print(f"      Mean error: {mean_err:.1f}%")

    # ========================================================================
    # Step 5: Create Comparison Plots
    # ========================================================================
    print_header("Step 5: Create Comparison Plots")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: ΔΣ profiles
    ax = axes[0, 0]
    ax.plot(rp_centers, ds_standard, 'k-', linewidth=3, label='Ground Truth', zorder=10)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for (name, res), color in zip(results.items(), colors):
        label = f"{name} ({res['rms_err']:.1f}%)"
        ax.plot(rp_centers, res['ds'], 'o-', label=label, alpha=0.7, color=color, markersize=4)
    ax.set_xlabel(r'$r_p$ [Mpc/h]', fontsize=12)
    ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('ΔΣ Profiles Comparison', fontsize=13, fontweight='bold')

    # Plot 2: Relative errors vs radius
    ax = axes[0, 1]
    for (name, res), color in zip(results.items(), colors):
        ax.plot(rp_centers, res['rel_diff'], 'o-', label=name, alpha=0.7, color=color, markersize=4)
    ax.set_xlabel(r'$r_p$ [Mpc/h]', fontsize=12)
    ax.set_ylabel('Relative Error [%]', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=2)
    ax.set_title('Relative Error vs Radius', fontsize=13, fontweight='bold')

    # Plot 3: Error metrics bar chart
    ax = axes[1, 0]
    method_names = list(results.keys())
    mean_errors = [results[n]['mean_err'] for n in method_names]
    median_errors = [results[n]['median_err'] for n in method_names]
    rms_errors = [results[n]['rms_err'] for n in method_names]

    x = np.arange(len(method_names))
    width = 0.25

    ax.bar(x - width, mean_errors, width, label='Mean', alpha=0.8)
    ax.bar(x, median_errors, width, label='Median', alpha=0.8)
    ax.bar(x + width, rms_errors, width, label='RMS', alpha=0.8)

    ax.set_ylabel('Error [%]', fontsize=12)
    ax.set_title('Error Metrics by Method', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Speed vs Accuracy
    ax = axes[1, 1]
    times = [results[n]['time'] for n in method_names]
    ax.scatter(times, rms_errors, s=150, alpha=0.7, c=colors)
    for i, name in enumerate(method_names):
        ax.annotate(name, (times[i], rms_errors[i]),
                   fontsize=8, ha='left', va='bottom')
    ax.set_xlabel('Computation Time [s]', fontsize=12)
    ax.set_ylabel('RMS Error [%]', fontsize=12)
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = output_dir / 'improved_interpolation_comparison.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: {output_plot}")

    # ========================================================================
    # Step 6: Summary and Recommendations
    # ========================================================================
    print_header("Summary and Recommendations")

    print(f"\n  {'Method':<25s} {'Time [s]':>10s} {'RMS %':>10s} {'Mean %':>10s} {'Median %':>10s}")
    print(f"  {'-'*70}")
    for name in method_names:
        res = results[name]
        print(f"  {name:<25s} {res['time']:>10.1f} {res['rms_err']:>10.1f} {res['mean_err']:>10.1f} {res['median_err']:>10.1f}")

    # Find best methods
    best_accuracy = min(method_names, key=lambda n: results[n]['rms_err'])
    best_speed = min(method_names, key=lambda n: results[n]['time'])
    best_tradeoff = min(method_names,
                       key=lambda n: results[n]['rms_err'] * np.sqrt(results[n]['time']))

    print(f"\n  ✅ Best accuracy: {best_accuracy}")
    print(f"      RMS error: {results[best_accuracy]['rms_err']:.1f}%")
    print(f"      Improvement: {rms_original / results[best_accuracy]['rms_err']:.1f}× better than original")

    print(f"\n  ⚡ Fastest: {best_speed}")
    print(f"      Time: {results[best_speed]['time']:.1f} s")
    print(f"      RMS error: {results[best_speed]['rms_err']:.1f}%")

    print(f"\n  ⭐ Best trade-off: {best_tradeoff}")
    print(f"      RMS error: {results[best_tradeoff]['rms_err']:.1f}%")
    print(f"      Time: {results[best_tradeoff]['time']:.1f} s")
    print(f"      Improvement: {rms_original / results[best_tradeoff]['rms_err']:.1f}× better than original")

    print("\n" + "="*70)
    print("  Test completed successfully!")
    print("="*70)


def regenerate_galaxy_positions(data, Lbox):
    """Regenerate galaxy positions from same HOD parameters."""
    print("    Regenerating galaxy positions from HOD...")

    # This would require re-running the HOD - for now, return None
    # In practice, we should save galaxy_positions in the comparison file
    raise ValueError("Galaxy positions not found in comparison file. "
                    "Please re-run test_fast_dsigma.py and ensure it saves galaxy_positions.")


if __name__ == "__main__":
    test_improved_interpolation()
