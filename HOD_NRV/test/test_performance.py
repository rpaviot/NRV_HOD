"""
Performance test for HOD pipeline.

Tests the full HOD workflow with timing for each step:
1. Find Ac for target number density
2. Compute HOD occupation probabilities
3. Populate central galaxies
4. Populate satellite galaxies
5. Compute galaxy clustering (wp)
6. Compute galaxy-galaxy lensing

Fixed parameters:
- Mmin = 12.0
- sig_M = 0.4
- As = 0.2
- M1 = 13.0
- alpha = 0.8
- kappa = 0.8
- Target ngal = 2e-3 (h/Mpc)^3

Using Flamingo L1000N1800 simulation:
- Lbox = 681 Mpc/h
- Halo catalog: parquet_halo_catalogue_L1000N1800.parquet
- Particle catalog: particle_catalogue_L1000N1800_downsampled.parquet
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HOD_NRV.HOD.HOD_catalogue import HaloOccupation
from HOD_NRV.utils.emulator_utils import rescale_Ac_to_target_ngal


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_timing(step_name, elapsed_time, units="s"):
    """Print timing result."""
    if units == "ms":
        print(f"  ⏱  {step_name:40s} {elapsed_time*1000:8.2f} ms")
    else:
        print(f"  ⏱  {step_name:40s} {elapsed_time:8.3f} s")


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("Performance Test Configuration")

    # Paths
    halo_path = '/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet'
    particle_path = '/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet'

    # Cosmology (Flamingo L1000N1800)
    cosmo_params = {
        'H0': 67.74,
        'Om0': 0.3089,
        'Ob0': 0.0486,
        'sigma8': 0.8159,
        'ns': 0.9667
    }

    # Column mapping
    column_mapping = {
        "x": "x", "y": "y", "z": "z",
        "vx": "vx", "vy": "vy", "vz": "vz",
        "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms"
    }

    # Box size
    Lbox = 681  # Mpc/h

    # HOD parameters (fixed, except Ac)
    hod_params_base = {
        'Mmin': 12.0,
        'sig_M': 0.4,
        'As': 0.2,
        'M1': 13.0,
        'alpha': 0.8,
        'kappa': 0.8
    }

    # Target number density
    target_ngal = 2e-3  # (h/Mpc)^3

    print(f"\n  Halo catalog: {Path(halo_path).name}")
    print(f"  Particle catalog: {Path(particle_path).name}")
    print(f"  Box size: {Lbox} Mpc/h")
    print(f"  Target ngal: {target_ngal:.2e} (h/Mpc)^3")
    print(f"\n  Fixed HOD parameters:")
    for key, val in hod_params_base.items():
        print(f"    {key:10s} = {val:.4f}")

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print_header("Step 1: Load Halo and Particle Catalogs")

    start = time.time()
    df_halos = pd.read_parquet(halo_path)
    elapsed_halos = time.time() - start
    print_timing("Load halo catalog", elapsed_halos)
    print(f"    → Loaded {len(df_halos):,} halos")

    start = time.time()
    df_particles = pd.read_parquet(particle_path)
    elapsed_particles = time.time() - start
    print_timing("Load particle catalog", elapsed_particles)
    print(f"    → Loaded {len(df_particles):,} particles")

    # ========================================================================
    # Step 2: Initialize HaloOccupation
    # ========================================================================
    print_header("Step 2: Initialize HaloOccupation")

    start = time.time()
    halo = HaloOccupation(
        cosmology=cosmo_params,
        zeff=1.0,
        Lbox=Lbox,
        column_mapping=column_mapping,
        mass_definition="vir",
        DataFrame=df_halos,
        DataFrame_part=df_particles,
        assembly_bias=False,
        apply_rsd=False,
        triaxial_NFW=False,
        do_test=False
    )
    elapsed_init = time.time() - start
    print_timing("Initialize HaloOccupation", elapsed_init)

    # ========================================================================
    # Step 3: Find Ac for Target Number Density
    # ========================================================================
    print_header("Step 3: Find Ac for Target Number Density")

    halo.set_halo_model('ELG_GHOD')

    start = time.time()
    Ac_rescaled = rescale_Ac_to_target_ngal(
        halo.HOD, hod_params_base, target_ngal, Ac_fiducial=1.0
    )
    elapsed_rescale = time.time() - start

    # Compute achieved ngal with rescaled Ac
    hod_params_check = hod_params_base.copy()
    hod_params_check['Ac'] = Ac_rescaled
    ngal_achieved = float(halo.HOD.compute_ngal(hod_params_check))

    print_timing("Rescale Ac to target ngal", elapsed_rescale)
    print(f"    → Rescaled Ac: {float(Ac_rescaled):.6f}")
    print(f"    → Achieved ngal: {ngal_achieved:.6e} (h/Mpc)^3")
    print(f"    → Target ngal: {target_ngal:.6e} (h/Mpc)^3")
    print(f"    → Relative error: {abs(ngal_achieved - target_ngal)/target_ngal * 100:.2e}%")

    # Add Ac to parameters
    hod_params = hod_params_base.copy()
    hod_params['Ac'] = Ac_rescaled

    # ========================================================================
    # Step 4: Compute HOD Occupation Probabilities
    # ========================================================================
    print_header("Step 4: Compute HOD Occupation Probabilities")

    start = time.time()
    probC, probS = halo.HOD.compute_HOD_occupation(halo.logM, hod_params)
    elapsed_probs = time.time() - start

    print_timing("Compute HOD probabilities", elapsed_probs)
    print(f"    → Central occupation (mean): {probC.mean():.6f}")
    print(f"    → Satellite occupation (mean): {probS.mean():.6f}")
    print(f"    → Expected N_centrals: {probC.sum():.0f}")
    print(f"    → Expected N_satellites: {probS.sum():.0f}")
    print(f"    → Expected N_galaxies: {(probC.sum() + probS.sum()):.0f}")

    # ========================================================================
    # Step 5: Populate Halos (Centrals + Satellites)
    # ========================================================================
    print_header("Step 5: Populate Halos with Galaxies")

    # Time the full population
    start = time.time()
    halo.populate_haloes(hod_params, random_seed=42)
    elapsed_populate = time.time() - start

    n_galaxies = len(halo.positions_gal)
    n_satellites = int(n_galaxies * halo.satellite_fraction)
    n_centrals = n_galaxies - n_satellites
    ngal_actual = n_galaxies / Lbox**3

    print_timing("Total population time", elapsed_populate)
    print(f"    → Total galaxies: {n_galaxies:,}")
    print(f"    → Centrals: {n_centrals:,}")
    print(f"    → Satellites: {n_satellites:,}")
    print(f"    → Satellite fraction: {halo.satellite_fraction:.4f}")
    print(f"    → Actual ngal: {ngal_actual:.6e} (h/Mpc)^3")
    print(f"    → Relative error vs target: {abs(ngal_actual - target_ngal)/target_ngal * 100:.2f}%")

    # ========================================================================
    # Step 6: Compute Galaxy Clustering (wp)
    # ========================================================================
    print_header("Step 6: Compute Galaxy Clustering (projected)")

    # Define bins
    rp_bins = np.logspace(-1, np.log10(80), 16)  # 0.1 to 80 Mpc/h
    pi_bins = np.linspace(-100, 100, 21)  # -100 to 100 Mpc/h

    print(f"  Bins:")
    print(f"    rp: {len(rp_bins)-1} bins from {rp_bins[0]:.2f} to {rp_bins[-1]:.2f} Mpc/h")
    print(f"    pi: {len(pi_bins)-1} bins from {pi_bins[0]:.0f} to {pi_bins[-1]:.0f} Mpc/h")

    start = time.time()
    rp, wp = halo.compute_galaxy_clustering('rppi', bins1=rp_bins, bins2=pi_bins, output='wp')
    elapsed_clustering = time.time() - start

    print_timing("Compute galaxy clustering (wp)", elapsed_clustering)
    print(f"    → Computed wp at {len(rp)} radial bins")
    print(f"    → wp range: [{wp.min():.2f}, {wp.max():.2f}] Mpc/h")

    # ========================================================================
    # Step 7: Compute Galaxy-Galaxy Lensing
    # ========================================================================
    print_header("Step 7: Compute Galaxy-Galaxy Lensing")

    # Define bins for lensing
    rp_lens_bins = np.logspace(-1, 1.5, 16)  # 0.1 to ~31 Mpc/h
    bins_comp = np.geomspace(5e-2, 100, 31)  # Completeness bins

    print(f"  Bins:")
    print(f"    rp: {len(rp_lens_bins)-1} bins from {rp_lens_bins[0]:.2f} to {rp_lens_bins[-1]:.2f} Mpc/h")
    print(f"    Completeness: {len(bins_comp)-1} bins from {bins_comp[0]:.2f} to {bins_comp[-1]:.2f} Mpc/h")

    start = time.time()
    rp_lens, delta_sigma = halo.compute_galaxy_lensing(rp_lens_bins, bins_comp=bins_comp)
    elapsed_lensing = time.time() - start

    print_timing("Compute galaxy-galaxy lensing", elapsed_lensing)
    print(f"    → Computed ΔΣ at {len(rp_lens)} radial bins")
    print(f"    → ΔΣ range: [{delta_sigma.min():.2e}, {delta_sigma.max():.2e}] Msun/pc^2")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Performance Summary")

    total_time = (elapsed_halos + elapsed_particles + elapsed_init +
                  elapsed_rescale + elapsed_probs + elapsed_populate +
                  elapsed_clustering + elapsed_lensing)

    print(f"\n  {'Step':<45s} {'Time':>10s}  {'%':>6s}")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'1. Load halo catalog':<45s} {elapsed_halos:>9.3f}s  {elapsed_halos/total_time*100:>5.1f}%")
    print(f"  {'2. Load particle catalog':<45s} {elapsed_particles:>9.3f}s  {elapsed_particles/total_time*100:>5.1f}%")
    print(f"  {'3. Initialize HaloOccupation':<45s} {elapsed_init:>9.3f}s  {elapsed_init/total_time*100:>5.1f}%")
    print(f"  {'4. Find Ac (rescale to target ngal)':<45s} {elapsed_rescale:>9.3f}s  {elapsed_rescale/total_time*100:>5.1f}%")
    print(f"  {'5. Compute HOD probabilities':<45s} {elapsed_probs:>9.3f}s  {elapsed_probs/total_time*100:>5.1f}%")
    print(f"  {'6. Populate halos (centrals + satellites)':<45s} {elapsed_populate:>9.3f}s  {elapsed_populate/total_time*100:>5.1f}%")
    print(f"  {'7. Compute galaxy clustering (wp)':<45s} {elapsed_clustering:>9.3f}s  {elapsed_clustering/total_time*100:>5.1f}%")
    print(f"  {'8. Compute galaxy-galaxy lensing':<45s} {elapsed_lensing:>9.3f}s  {elapsed_lensing/total_time*100:>5.1f}%")
    print(f"  {'-'*45} {'-'*10}  {'-'*6}")
    print(f"  {'TOTAL':<45s} {total_time:>9.3f}s  100.0%")

    print(f"\n  Key metrics:")
    print(f"    Number of halos: {len(df_halos):,}")
    print(f"    Number of particles: {len(df_particles):,}")
    print(f"    Number of galaxies: {n_galaxies:,}")
    print(f"    Time per galaxy (population): {elapsed_populate/n_galaxies*1e6:.2f} μs")
    print(f"    Galaxies per second (population): {n_galaxies/elapsed_populate:,.0f}")

    print("\n" + "="*70)
    print("  Performance test completed successfully!")
    print("="*70 + "\n")

    # Save results to file
    results = {
        'target_ngal': target_ngal,
        'achieved_ngal': ngal_actual,
        'Ac': Ac_rescaled,
        'n_halos': len(df_halos),
        'n_particles': len(df_particles),
        'n_galaxies': n_galaxies,
        'satellite_fraction': halo.satellite_fraction,
        'time_load_halos': elapsed_halos,
        'time_load_particles': elapsed_particles,
        'time_init': elapsed_init,
        'time_rescale_Ac': elapsed_rescale,
        'time_compute_probs': elapsed_probs,
        'time_populate': elapsed_populate,
        'time_clustering': elapsed_clustering,
        'time_lensing': elapsed_lensing,
        'time_total': total_time,
        'wp_min': wp.min(),
        'wp_max': wp.max(),
        'delta_sigma_min': delta_sigma.min(),
        'delta_sigma_max': delta_sigma.max(),
    }

    results_df = pd.DataFrame([results])
    output_path = Path(__file__).parent / 'performance_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
