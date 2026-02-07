"""
Numerical DeltaSigma Example
=============================

Canonical producer of baseline numerical DeltaSigma results.

This script:
1. Loads Flamingo L1000N1800 halo + particle catalogs
2. Populates ELG_GHOD galaxies with rescaled Ac/As to target n_gal
3. Computes galaxy-galaxy lensing (DeltaSigma) for N_BASELINE realizations
4. Saves results to baseline_dsigma_cache.npz

Other scripts (benchmark_dsigma_convergence.py, cross_check_analytical_numerical.py)
consume this cache rather than recomputing the baseline.

Data: Flamingo L1000N1800
  - Halos:     /Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet
  - Particles: /Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet
"""

import time
import numpy as np
import pandas as pd

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration
# ============================================================================

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"

# Baseline configuration
N_BASELINE = 10
BASE_SEED = 1000
CACHE_FILE = "baseline_dsigma_cache.npz"

Lbox = 681.0        # Mpc/h
zeff = 1.0
mass_definition = "200m"

column_mapping = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}


dict_cosmo = {
        'h': 0.681,
        'Omc': 0.306-0.0486-1.39e-3,
        'Omb': 0.0486,
        'A_s': 2.099e-9,
        'n_s': 0.967,
        'Omnu':1.39e-3
    }


cosmo_params = {'H0': dict_cosmo['h']*100, 'Om0': dict_cosmo['Omc']+dict_cosmo['Omb'] + dict_cosmo['Omnu'], 'Ob0': dict_cosmo['Omb'], 'sigma8': 0.807, 'ns': 0.967}
# Base HOD parameters for ELG_GHOD (Ac and As will be rescaled)
base_hod_params = {
    "As": 0.3,
    "Mmin": 13.0,
    "sig_M": 0.3,
    "M1": 13.0,
    "alpha": 0.80,
    "kappa": 0.80,
}

target_ngal = 2e-4  # (Mpc/h)^-3

# Lensing bins
rp_bins = np.geomspace(0.1, 50.0, 16)

# ============================================================================
# Load data
# ============================================================================

print("=" * 60)
print("Numerical DeltaSigma Example -- Flamingo L1000N1800")
print("=" * 60)

print("\nLoading halo catalog...")
df_halo = pd.read_parquet(HALO_PATH)
print(f"  {len(df_halo)} halos loaded")

print("Loading particle catalog...")
df_part = pd.read_parquet(PARTICLE_PATH)
print(f"  {len(df_part)} particles loaded")

# ============================================================================
# Initialize HaloOccupation
# ============================================================================

print("\nInitializing HaloOccupation...")
halo = HaloOccupation(
    cosmology=cosmo_params,
    zeff=zeff,
    Lbox=Lbox,
    column_mapping=column_mapping,
    mass_definition=mass_definition,
    DataFrame=df_halo,
    DataFrame_part=df_part,
    assembly_bias=False,
    apply_rsd=False,
    triaxial_NFW=False,
    do_test=False,
)

# ============================================================================
# Set HOD model and rescale Ac/As
# ============================================================================

print("\nSetting HOD model: ELG_GHOD")
halo.set_halo_model("ELG_GHOD")

print(f"Rescaling Ac/As to target n_gal = {target_ngal:.2e} (Mpc/h)^-3 ...")
Ac_rescaled, As_rescaled = rescale_Ac_to_target_ngal(
    halo.HOD, base_hod_params, target_ngal=target_ngal
)

hod_params = base_hod_params.copy()
hod_params["Ac"] = Ac_rescaled
hod_params["As"] = As_rescaled

print(f"  Ac = {Ac_rescaled[0]:.6f}")
print(f"  As = {As_rescaled[0]:.6f}")

# ============================================================================
# Run multiple realizations and cache results
# ============================================================================

print(f"\nRunning {N_BASELINE} realizations...")
ds_all = []
sat_fracs = []
ngals = []
timing_per_real = []
rp_centers = None

for i in range(N_BASELINE):
    seed = BASE_SEED + i
    print(f"  Realization {i+1}/{N_BASELINE} (seed={seed})...", end=" ", flush=True)

    halo.populate_haloes(hod_params, random_seed=seed)
    sat_fracs.append(halo.satellite_fraction)
    ngals.append(len(halo.positions_gal) / Lbox**3)

    t0 = time.perf_counter()
    rp_centers, ds = halo.compute_galaxy_lensing(rp_bins)
    timing_per_real.append(time.perf_counter() - t0)
    ds_all.append(ds)

    print(f"done ({timing_per_real[-1]:.1f}s)")

# Convert to arrays
ds_all = np.array(ds_all)
timing_per_real = np.array(timing_per_real)
sat_fracs = np.array(sat_fracs)
ngals = np.array(ngals)

# Save cache
np.savez(
    CACHE_FILE,
    ds_all=ds_all,
    rp=rp_centers,
    timing_per_real=timing_per_real,
    sat_frac=sat_fracs,
    ngal=ngals,
    Ac=Ac_rescaled[0],
    As=As_rescaled[0],
)
print(f"\nSaved cache to {CACHE_FILE}")

# ============================================================================
# Print averaged results
# ============================================================================

ds_mean = np.mean(ds_all, axis=0)
ds_std = np.std(ds_all, axis=0, ddof=1)

print(f"\n--- Averaged DeltaSigma ({N_BASELINE} realizations) ---")
print(f"  ngal (mean)     = {ngals.mean():.6e}  (Mpc/h)^-3")
print(f"  sat_frac (mean) = {sat_fracs.mean():.4f}")
print(f"  timing (mean)   = {timing_per_real.mean():.1f}s per realization")

print(f"\n{'rp [Mpc/h]':>14s}  {'DeltaSigma [h Msun/pc^2]':>26s}  {'std':>14s}")
print("-" * 60)
for r, ds_m, ds_s in zip(rp_centers, ds_mean, ds_std):
    print(f"  {r:12.4f}    {ds_m:22.6e}    {ds_s:12.6e}")

print("\nDone.")
