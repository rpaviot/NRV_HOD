"""
Performance comparison test for different ΔΣ pre-computation methods.

This script compares:
1. Original serial version (single-point KDTree queries)
2. NEW batched version (multi-point KDTree queries)
3. Joblib parallel version (multiprocessing)
4. Numba parallel version (prange with shared memory)

Run this to determine which approach is fastest for your dataset.
"""

import numpy as np
import time
from scipy.spatial import cKDTree

# Import different implementations
from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    precompute_lensing_grid as precompute_serial
)
from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma_batched import (
    precompute_lensing_grid_batched,
    precompute_lensing_grid_simple_batched
)

try:
    from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma_parallel import (
        precompute_lensing_grid_parallel
    )
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False
    print("Warning: joblib version not available")

try:
    from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma_numba import (
        precompute_lensing_grid_numba_nokdtree,
        precompute_lensing_grid_numba
    )
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    print("Warning: Numba version not available")


def create_test_data(n_halos=100, n_particles=100000, Lbox=1000.0, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)

    # Create random halos
    halo_positions = np.random.rand(n_halos, 3) * Lbox
    halo_rvir = np.random.uniform(1.0, 10.0, n_halos)  # 1-10 Mpc/h

    # Create random particles
    particle_positions = np.random.rand(n_particles, 3) * Lbox

    # Cosmological parameters
    RHO_M = 2.775e11 * 0.3 * (0.7)**2  # Mean matter density [Msun/h / (Mpc/h)^3]

    # Radial bins
    rp_bins = np.logspace(-1, 1.5, 15)  # 0.1 to ~30 Mpc/h

    return halo_positions, halo_rvir, particle_positions, RHO_M, rp_bins, Lbox


def benchmark_method(name, func, *args, **kwargs):
    """Benchmark a single method."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    try:
        start = time.time()
        positions, deltasigma = func(*args, **kwargs)
        elapsed = time.time() - start

        if len(positions) > 0:
            rate = len(positions) / elapsed
            print(f"\nRESULTS:")
            print(f"  Success: {len(positions):,} positions computed")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Rate: {rate:.1f} positions/s")
            print(f"  Memory: {deltasigma.nbytes / 1e6:.1f} MB")
            return elapsed, len(positions), rate
        else:
            print(f"\nFAILED: No positions computed")
            return None, 0, 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0


def run_comparison(
    n_halos=50,
    n_particles=50000,
    test_serial=False,  # Serial is slow, disable by default
    test_batched=True,
    test_simple_batched=True,
    test_joblib=False,
    test_numba=True
):
    """
    Run performance comparison on different methods.

    Parameters
    ----------
    n_halos : int
        Number of halos to test (default: 50 for quick test)
    n_particles : int
        Number of particles to test (default: 50,000)
    test_serial : bool
        Test original serial version (SLOW! Only for small datasets)
    test_batched : bool
        Test new batched version
    test_simple_batched : bool
        Test simplified batched version
    test_joblib : bool
        Test joblib parallel version
    test_numba : bool
        Test Numba parallel version
    """
    print(f"\n{'#'*70}")
    print(f"# PERFORMANCE COMPARISON")
    print(f"#{'#'*69}")
    print(f"Dataset size:")
    print(f"  Halos: {n_halos:,}")
    print(f"  Particles: {n_particles:,}")
    print(f"{'#'*70}\n")

    # Create test data
    print("Creating test data...")
    halo_pos, halo_rvir, particle_pos, RHO_M, rp_bins, Lbox = create_test_data(
        n_halos=n_halos,
        n_particles=n_particles
    )

    # Pre-select particles near halos (for fair comparison)
    print("Pre-selecting particles near halos...")
    kdtree = cKDTree(particle_pos, boxsize=Lbox)
    selected_indices = set()
    for halo, rvir in zip(halo_pos, halo_rvir):
        nearby = kdtree.query_ball_point(halo, r=3.0 * rvir)
        selected_indices.update(nearby)

    selected_positions = particle_pos[list(selected_indices)]
    print(f"  {len(selected_positions):,} particles selected\n")

    results = {}

    # Test methods
    if test_serial:
        print("\nWARNING: Testing serial version (this will be SLOW)...")
        time.sleep(2)
        elapsed, n_pos, rate = benchmark_method(
            "Original Serial (single-point queries)",
            precompute_serial,
            halo_pos, halo_rvir, particle_pos,
            RHO_M, rp_bins, Lbox,
            r_factor=3.0,
            method='spherical',
            verbose=True
        )
        results['serial'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

    if test_simple_batched:
        elapsed, n_pos, rate = benchmark_method(
            "Simple Batched (multi-point queries)",
            precompute_lensing_grid_simple_batched,
            halo_pos, halo_rvir, particle_pos,
            RHO_M, rp_bins, Lbox,
            r_factor=3.0,
            batch_size=5000,
            method='spherical',
            verbose=True
        )
        results['simple_batched'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

    if test_batched:
        elapsed, n_pos, rate = benchmark_method(
            "Full Batched (halo + particle batching)",
            precompute_lensing_grid_batched,
            halo_pos, halo_rvir, particle_pos,
            RHO_M, rp_bins, Lbox,
            r_factor=3.0,
            halo_batch_size=20,
            particle_batch_size=5000,
            method='spherical',
            verbose=True
        )
        results['batched'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

    if test_joblib and HAVE_JOBLIB:
        elapsed, n_pos, rate = benchmark_method(
            "Joblib Parallel (multiprocessing)",
            precompute_lensing_grid_parallel,
            selected_positions, particle_pos,
            RHO_M, rp_bins, Lbox,
            method='spherical',
            n_jobs=-1,
            batch_size=1000,
            backend='loky',
            use_memmap=True,
            verbose=True
        )
        results['joblib'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

    if test_numba and HAVE_NUMBA:
        # Test no-KDTree version (usually fastest)
        elapsed, n_pos, rate = benchmark_method(
            "Numba Parallel (NO KDTree)",
            precompute_lensing_grid_numba_nokdtree,
            selected_positions, particle_pos,
            RHO_M, rp_bins, Lbox,
            batch_size=5000,
            verbose=True
        )
        results['numba_nokdtree'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

        # Test with KDTree version
        elapsed, n_pos, rate = benchmark_method(
            "Numba Parallel (with KDTree)",
            precompute_lensing_grid_numba,
            selected_positions, particle_pos,
            RHO_M, rp_bins, Lbox,
            batch_size=5000,
            verbose=True
        )
        results['numba_kdtree'] = {'time': elapsed, 'n_positions': n_pos, 'rate': rate}

    # Print summary
    print(f"\n\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"#{'#'*69}")
    print(f"\n{'Method':<35} {'Time (s)':<12} {'Rate (pos/s)':<15} {'Speedup':<10}")
    print(f"{'-'*70}")

    baseline_time = None
    if 'serial' in results and results['serial']['time']:
        baseline_time = results['serial']['time']
    elif 'simple_batched' in results and results['simple_batched']['time']:
        baseline_time = results['simple_batched']['time']

    for method, data in results.items():
        if data['time'] is not None:
            speedup = baseline_time / data['time'] if baseline_time else 1.0
            print(f"{method:<35} {data['time']:<12.2f} {data['rate']:<15.1f} {speedup:<10.1f}×")

    print(f"{'#'*70}\n")

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 70)

    if HAVE_NUMBA and 'numba_nokdtree' in results and results['numba_nokdtree']['time']:
        print("✓ For best performance: Use Numba (NO KDTree) version")
        print("  from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma_numba import \\")
        print("      precompute_lensing_grid_numba_nokdtree")
    elif test_simple_batched and 'simple_batched' in results:
        print("✓ For good performance without Numba: Use Simple Batched version")
        print("  from HOD_NRV.HOD_numerical.twopoint_calculator.precompute_deltasigma_batched import \\")
        print("      precompute_lensing_grid_simple_batched")

    print("\nNotes:")
    print("- Numba versions require particles to be pre-selected near halos")
    print("- Batched versions work directly with halo catalog")
    print("- Larger datasets show bigger speedups")
    print("-" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark different ΔΣ pre-computation methods"
    )
    parser.add_argument('--n-halos', type=int, default=50,
                        help='Number of halos (default: 50)')
    parser.add_argument('--n-particles', type=int, default=50000,
                        help='Number of particles (default: 50,000)')
    parser.add_argument('--test-serial', action='store_true',
                        help='Test original serial version (SLOW!)')
    parser.add_argument('--test-all', action='store_true',
                        help='Test all available methods')
    parser.add_argument('--large', action='store_true',
                        help='Use larger dataset (200 halos, 200k particles)')

    args = parser.parse_args()

    if args.large:
        n_halos = 200
        n_particles = 200000
    else:
        n_halos = args.n_halos
        n_particles = args.n_particles

    run_comparison(
        n_halos=n_halos,
        n_particles=n_particles,
        test_serial=args.test_serial,
        test_batched=True,
        test_simple_batched=True,
        test_joblib=args.test_all,
        test_numba=True
    )
