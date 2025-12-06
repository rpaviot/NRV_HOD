"""
Test different interpolation methods for noisy ΔΣ data.

This script compares:
1. Original IDW (k=8)
2. Kernel smoothing (Gaussian, k=32, auto-bandwidth)
3. Kernel smoothing (larger k=64)
4. Adaptive kernel smoothing
5. Modified Shepard (k=32, power=3)

Goal: Find which method best handles the noisy downsampled data.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import load_precomputed_lensing
from HOD_NRV.HOD_numerical.twopoint_calculator.improved_interpolation import (
    ImprovedDeltaSigmaInterpolator,
    AdaptiveDeltaSigmaInterpolator
)
from HOD_NRV.HOD_numerical.twopoint_calculator.fast_two_point import FastDeltaSigmaCalculator


def test_interpolation_methods(
    precomputed_file: str,
    galaxy_positions: np.ndarray,
    ground_truth: np.ndarray,
    rp_bins: np.ndarray
):
    """
    Compare different interpolation methods.

    Parameters
    ----------
    precomputed_file : str
        Path to precomputed lensing HDF5 file
    galaxy_positions : np.ndarray
        Galaxy positions for testing
    ground_truth : np.ndarray
        Ground truth ΔΣ from standard method
    rp_bins : np.ndarray
        Radial bins
    """
    print("=" * 70)
    print("Testing Interpolation Methods")
    print("=" * 70)

    # Load pre-computed data
    positions, deltasigma, rp_bins_pre, metadata = load_precomputed_lensing(precomputed_file)
    Lbox = metadata.get('Lbox', 1000.0)

    # Initialize all methods
    methods = {}

    print("\n1. Original IDW (k=8)")
    methods['IDW_k8'] = FastDeltaSigmaCalculator(
        precomputed_file,
        interpolation_method='idw',
        k_neighbors=100,
        Lbox=Lbox
    )

    print("\n2. Kernel Smoothing (Gaussian, k=32, auto-bandwidth)")
    methods['Kernel_k32'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=32,
        bandwidth=None,  # Auto-compute
        kernel='gaussian'
    )

    print("\n3. Kernel Smoothing (Gaussian, k=64)")
    methods['Kernel_k64'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=64,
        bandwidth=None,
        kernel='gaussian'
    )

    print("\n4. Kernel Smoothing (Gaussian, k=32, bandwidth=10 Mpc/h)")
    methods['Kernel_k32_bw10'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='kernel_smooth',
        k_neighbors=32,
        bandwidth=10.0,
        kernel='gaussian'
    )

    print("\n5. Modified Shepard (k=32, power=3)")
    methods['Shepard_k32_p3'] = ImprovedDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        method='shepard',
        k_neighbors=32
    )

    print("\n6. Adaptive Kernel Smoothing (k=16-64)")
    methods['Adaptive'] = AdaptiveDeltaSigmaInterpolator(
        positions, deltasigma, Lbox=Lbox,
        k_min=16, k_max=64
    )

    # Test each method
    results = {}
    n_gal = len(galaxy_positions)

    print("\n" + "=" * 70)
    print("Running Interpolation Tests")
    print("=" * 70)

    for name, interpolator in methods.items():
        print(f"\nTesting {name}...")
        t0 = time.time()

        # Compute ΔΣ
        if name == 'IDW_k8':
            # Use FastDeltaSigmaCalculator interface
            _, ds = interpolator.compute_deltasigma_for_galaxies(
                galaxy_positions, rp_bins=rp_bins
            )
        else:
            # Use ImprovedDeltaSigmaInterpolator interface
            ds_sum = np.zeros(len(rp_bins) - 1)
            for gal_pos in galaxy_positions:
                ds_i = interpolator.interpolate_at_position(gal_pos)
                ds_sum += ds_i
            ds = ds_sum / n_gal

        elapsed = time.time() - t0

        # Compute accuracy metrics
        rel_diff = 100 * np.abs(ds - ground_truth) / (np.abs(ground_truth) + 1e-10)
        mean_err = np.mean(rel_diff)
        median_err = np.median(rel_diff)
        max_err = np.max(rel_diff)
        rms_err = np.sqrt(np.mean(rel_diff ** 2))

        results[name] = {
            'ds': ds,
            'time': elapsed,
            'mean_err': mean_err,
            'median_err': median_err,
            'max_err': max_err,
            'rms_err': rms_err,
            'rel_diff': rel_diff
        }

        print(f"  Time: {elapsed:.3f} s")
        print(f"  Mean error: {mean_err:.2f}%")
        print(f"  Median error: {median_err:.2f}%")
        print(f"  Max error: {max_err:.2f}%")
        print(f"  RMS error: {rms_err:.2f}%")

    # Create comparison plots
    print("\n" + "=" * 70)
    print("Creating Comparison Plots")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

    # Plot 1: ΔΣ profiles
    ax = axes[0, 0]
    ax.plot(rp_centers, ground_truth, 'k-', linewidth=2, label='Ground Truth')
    for name, res in results.items():
        ax.plot(rp_centers, res['ds'], marker='o', label=name, alpha=0.7)
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('ΔΣ Profiles')

    # Plot 2: Relative differences
    ax = axes[0, 1]
    for name, res in results.items():
        ax.plot(rp_centers, res['rel_diff'], marker='o', label=name, alpha=0.7)
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel('Relative Difference [%]')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Relative Error vs Radius')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Plot 3: Error summary
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

    ax.set_ylabel('Error [%]')
    ax.set_title('Error Metrics by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Speed vs Accuracy
    ax = axes[1, 1]
    times = [results[n]['time'] for n in method_names]

    ax.scatter(times, mean_errors, s=100, alpha=0.7)
    for i, name in enumerate(method_names):
        ax.annotate(name, (times[i], mean_errors[i]),
                   fontsize=8, ha='right', alpha=0.7)

    ax.set_xlabel('Computation Time [s]')
    ax.set_ylabel('Mean Error [%]')
    ax.set_title('Speed vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('interpolation_method_comparison.png', dpi=150)
    print("\nSaved comparison plot: interpolation_method_comparison.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Method':<25} {'Time [s]':>10} {'Mean Err %':>12} {'Median Err %':>14} {'RMS Err %':>12}")
    print("-" * 70)
    for name in method_names:
        res = results[name]
        print(f"{name:<25} {res['time']:>10.3f} {res['mean_err']:>12.2f} "
              f"{res['median_err']:>14.2f} {res['rms_err']:>12.2f}")

    # Find best method
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)

    best_accuracy = min(method_names, key=lambda n: results[n]['rms_err'])
    best_speed = min(method_names, key=lambda n: results[n]['time'])

    # Best trade-off: minimize RMS_err * sqrt(time)
    best_tradeoff = min(method_names,
                       key=lambda n: results[n]['rms_err'] * np.sqrt(results[n]['time']))

    print(f"Best accuracy: {best_accuracy} (RMS error: {results[best_accuracy]['rms_err']:.2f}%)")
    print(f"Fastest: {best_speed} (time: {results[best_speed]['time']:.3f} s)")
    print(f"Best trade-off: {best_tradeoff}")
    print(f"  RMS error: {results[best_tradeoff]['rms_err']:.2f}%")
    print(f"  Time: {results[best_tradeoff]['time']:.3f} s")

    return results


if __name__ == "__main__":
    # Paths to test data
    output_dir = Path(__file__).parent / 'output'
    precomputed_file = output_dir / 'precomputed_lensing_grid.h5'
    comparison_file = output_dir / 'lensing_comparison.npz'

    # Check if files exist
    if not precomputed_file.exists():
        print(f"ERROR: Pre-computed file not found: {precomputed_file}")
        print("Please run test_fast_dsigma.py first!")
        sys.exit(1)

    if not comparison_file.exists():
        print(f"ERROR: Comparison file not found: {comparison_file}")
        print("Please run test_fast_dsigma.py with compare_lensing_pipelines()!")
        sys.exit(1)

    # Load comparison data
    print("Loading test data...")
    data = np.load(comparison_file, allow_pickle=True)

    rp_centers = data['rp']
    ds_standard = data['ds_standard']  # Ground truth

    # Reconstruct rp_bins from centers
    rp_bins = np.zeros(len(rp_centers) + 1)
    rp_bins[1:-1] = np.sqrt(rp_centers[:-1] * rp_centers[1:])
    rp_bins[0] = rp_centers[0]**2 / rp_bins[1]
    rp_bins[-1] = rp_centers[-1]**2 / rp_bins[-2]

    # Get galaxy positions
    if 'galaxy_positions' in data:
        galaxy_positions = data['galaxy_positions']
        print(f"Loaded {len(galaxy_positions):,} galaxy positions")
    else:
        print("ERROR: Galaxy positions not found in comparison file")
        print("Please re-run test_fast_dsigma.py and ensure galaxy_positions are saved")
        sys.exit(1)

    # Run test
    results = test_interpolation_methods(
        str(precomputed_file),
        galaxy_positions,
        ds_standard,
        rp_bins
    )
