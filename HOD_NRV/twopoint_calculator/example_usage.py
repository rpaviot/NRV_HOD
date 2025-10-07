"""
Example usage of the fast two-point calculator.

This script demonstrates how to:
1. Pre-compute ΔΣ at particle positions (expensive, once per snapshot)
2. Use the fast calculator for HOD realizations (cheap, many times)
"""

import numpy as np
import matplotlib.pyplot as plt
from HOD_NRV.twopoint_calculator import (
    precompute_lensing_grid,
    save_precomputed_lensing,
    FastDeltaSigmaCalculator
)


def example_precomputation():
    """
    Example: Pre-compute ΔΣ for a simulation snapshot.

    This is expensive but only needs to be done once per snapshot.
    """
    print("=" * 70)
    print("EXAMPLE: Pre-computation Phase")
    print("=" * 70)

    # NOTE: Replace these with your actual simulation data
    # For this example, we'll use dummy data

    # Load halo catalog
    print("\n1. Loading halo catalog...")
    # halo_positions = ...  # Load from your simulation
    # halo_rvir = ...
    # Example with dummy data:
    n_halos = 1000
    halo_positions = np.random.rand(n_halos, 3) * 1000  # 1000 Mpc/h box
    halo_rvir = np.random.uniform(0.5, 2.0, n_halos)  # Mpc/h
    print(f"   Loaded {n_halos} halos")

    # Load particle catalog
    print("\n2. Loading particle catalog...")
    # particle_positions = ...  # Load from your simulation
    # Example with dummy data:
    n_particles = 50000  # In real simulations this would be millions
    particle_positions = np.random.rand(n_particles, 3) * 1000
    print(f"   Loaded {n_particles} particles")

    # Define parameters
    print("\n3. Setting up parameters...")
    RHO_M = 8.6e10  # Msun/h / (Mpc/h)^3 for Planck cosmology
    Lbox = 1000.0   # Mpc/h
    rp_bins = np.logspace(-1, 1.5, 15)  # Mpc/h
    r_factor = 3.0  # Search within 3×R_vir

    print(f"   RHO_M = {RHO_M:.2e} Msun/h / (Mpc/h)^3")
    print(f"   Lbox = {Lbox} Mpc/h")
    print(f"   rp bins: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"   Search radius: {r_factor}×R_vir")

    # Pre-compute ΔΣ
    print("\n4. Pre-computing ΔΣ at particle positions...")
    print("   (This may take a while...)")

    positions, deltasigma = precompute_lensing_grid(
        halo_positions=halo_positions,
        halo_rvir=halo_rvir,
        particle_positions=particle_positions,
        RHO_M=RHO_M,
        rp_bins=rp_bins,
        Lbox=Lbox,
        r_factor=r_factor,
        verbose=True
    )

    print(f"\n   Successfully computed ΔΣ at {len(positions)} positions")
    print(f"   Storage size: ~{positions.nbytes + deltasigma.nbytes} bytes")

    # Save to disk
    print("\n5. Saving pre-computed data to disk...")
    metadata = {
        'cosmology': 'Planck2018',
        'redshift': 0.5,
        'RHO_M': RHO_M,
        'Lbox': Lbox,
        'n_halos': n_halos,
        'n_particles': n_particles,
        'r_factor': r_factor
    }

    output_file = 'precomputed_lensing_example.h5'
    save_precomputed_lensing(
        output_file,
        positions,
        deltasigma,
        rp_bins,
        metadata
    )

    print(f"\n   Saved to: {output_file}")
    print("\n" + "=" * 70)
    print("Pre-computation complete!")
    print("You can now use this file for fast lensing calculations.")
    print("=" * 70 + "\n")

    return output_file


def example_fast_evaluation(precomputed_file):
    """
    Example: Use fast calculator for HOD realizations.

    This is fast and can be called many times for different HOD parameters.
    """
    print("=" * 70)
    print("EXAMPLE: Fast Evaluation Phase")
    print("=" * 70)

    # Initialize fast calculator
    print("\n1. Loading pre-computed data...")
    calc = FastDeltaSigmaCalculator(
        precomputed_file,
        interpolation_method='idw',
        k_neighbors=8
    )

    # Simulate galaxy positions (in real usage, these come from HOD)
    print("\n2. Simulating galaxy catalog...")
    n_galaxies = 500
    galaxy_positions = np.random.rand(n_galaxies, 3) * 1000  # Random positions
    print(f"   Generated {n_galaxies} galaxies")

    # Compute lensing signal (FAST!)
    print("\n3. Computing ΔΣ for galaxies (this should be fast)...")
    import time
    start = time.time()

    rp, delta_sigma = calc.compute_deltasigma_for_galaxies(
        galaxy_positions,
        verbose=False
    )

    elapsed = time.time() - start
    print(f"   Computed in {elapsed:.3f} seconds")
    print(f"   (Compare to ~minutes for legacy method)")

    # Display results
    print("\n4. Results:")
    print(f"   rp range: {rp[0]:.2f} - {rp[-1]:.2f} Mpc/h")
    print(f"   ΔΣ range: {delta_sigma.min():.2e} - {delta_sigma.max():.2e} Msun h/pc²")

    # Plot
    print("\n5. Plotting results...")
    plt.figure(figsize=(8, 6))
    plt.loglog(rp, delta_sigma, 'o-', label='Fast calculator')
    plt.xlabel(r'$r_p$ [Mpc/h]', fontsize=12)
    plt.ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]', fontsize=12)
    plt.title('Galaxy-Galaxy Lensing (Fast Calculator)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fast_lensing_example.png', dpi=150)
    print("   Saved plot to: fast_lensing_example.png")

    print("\n" + "=" * 70)
    print("Fast evaluation complete!")
    print("=" * 70 + "\n")

    return rp, delta_sigma


def compare_with_multiple_realizations(precomputed_file, n_realizations=10):
    """
    Example: Demonstrate speed advantage with multiple HOD realizations.
    """
    print("=" * 70)
    print("EXAMPLE: Multiple HOD Realizations")
    print("=" * 70)

    calc = FastDeltaSigmaCalculator(precomputed_file)

    n_galaxies = 500
    results = []

    import time
    start = time.time()

    print(f"\nComputing ΔΣ for {n_realizations} different HOD realizations...")
    for i in range(n_realizations):
        # Generate different galaxy positions (simulating different HOD parameters)
        galaxy_positions = np.random.rand(n_galaxies, 3) * 1000

        rp, delta_sigma = calc.compute_deltasigma_for_galaxies(
            galaxy_positions, verbose=False
        )
        results.append(delta_sigma)

        if (i + 1) % max(1, n_realizations // 5) == 0:
            print(f"  Completed {i+1}/{n_realizations} realizations")

    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Time per realization: {elapsed/n_realizations:.3f} seconds")
    print(f"Estimated legacy time: ~{n_realizations * 100:.0f} seconds (100s per realization)")
    print(f"Speedup: ~{n_realizations * 100 / elapsed:.0f}x")

    # Plot distribution
    results = np.array(results)
    mean_ds = results.mean(axis=0)
    std_ds = results.std(axis=0)

    plt.figure(figsize=(8, 6))
    plt.loglog(rp, mean_ds, 'o-', label='Mean')
    plt.fill_between(rp, mean_ds - std_ds, mean_ds + std_ds, alpha=0.3, label='±1σ')
    plt.xlabel(r'$r_p$ [Mpc/h]', fontsize=12)
    plt.ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]', fontsize=12)
    plt.title(f'ΔΣ from {n_realizations} HOD Realizations', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fast_lensing_multiple.png', dpi=150)
    print("\nSaved plot to: fast_lensing_multiple.png")

    print("\n" + "=" * 70)
    print("This demonstrates the power of the fast calculator for")
    print("MCMC sampling or parameter grid exploration!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run the examples
    print("\n" + "=" * 70)
    print("FAST TWO-POINT CALCULATOR - USAGE EXAMPLES")
    print("=" * 70 + "\n")

    # Step 1: Pre-computation (expensive, once)
    precomputed_file = example_precomputation()

    input("\nPress Enter to continue to fast evaluation...")

    # Step 2: Fast evaluation (cheap, many times)
    example_fast_evaluation(precomputed_file)

    input("\nPress Enter to demonstrate multiple realizations...")

    # Step 3: Multiple realizations
    compare_with_multiple_realizations(precomputed_file, n_realizations=20)

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
