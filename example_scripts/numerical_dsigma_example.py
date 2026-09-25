"""
Numerical DeltaSigma Example
=============================

Canonical producer of baseline numerical DeltaSigma results.

This script:
1. Loads Flamingo L1000N1800 halo + particle catalogs
2. Populates ELG_mHMQ galaxies with rescaled Ac/As to target n_gal
3. Computes galaxy-galaxy lensing (DeltaSigma) for N_BASELINE realizations
4. Saves results AND the HOD (type + all parameters) to baseline_dsigma_cache.npz,
   so the analytical side evaluates exactly the same occupation

Halos without a concentration (promoted halos of a distinct-halo catalogue,
c = NaN) get the analytical model's c(M) (Duffy08, M200m, z = zeff).

    python numerical_dsigma_example.py \
        --halo_path  .../host_catalogue_distinct_excl.parquet \
        --particle_path .../DMO_flamingo_0058_downsampled_0.5percent.parquet \
        --cache baseline_dsigma_cache_DMO_distinct.npz

cross_check_analytical_numerical.py consumes this cache rather than recomputing
the baseline.

Data: Flamingo L1000N1800
  - Halos:     /Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet
  - Particles: /Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet
"""

import argparse
import time
import numpy as np
import pandas as pd

from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.HOD_models import rescale_Ac_to_target_ngal

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--halo_path", default="/Users/ler13nrv/Documents/flamingo_data/"
                    "parquet_halo_catalogue_L1000N1800.parquet")
parser.add_argument("--particle_path", default="/Users/ler13nrv/Documents/flamingo_data/"
                    "particle_catalogue_L1000N1800_downsampled.parquet")
parser.add_argument("--cache", default="baseline_dsigma_cache.npz")
parser.add_argument("--n_real", type=int, default=10)
args = parser.parse_args()

HALO_PATH = args.halo_path
PARTICLE_PATH = args.particle_path

# Baseline configuration
N_BASELINE = args.n_real
BASE_SEED = 1000
CACHE_FILE = args.cache
HOD_TYPE = "ELG_mHMQ"    # analytical name: ELG_MHMQ

Lbox = 681.0        # Mpc/h
zeff = 1.0
mass_definition = "MassDef200m"

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


# Base HOD parameters for ELG_mHMQ (Ac and As will be rescaled)
base_hod_params = {
    "As": 0.3,
    "Mmin": 12.7,
    "sig_M": 0.3,
    "M1": 13.0,
    "gamma":5.0,
    "alpha": 1.10,
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
no_c = ~np.isfinite(df_halo["c"].to_numpy())
if no_c.any():
    import pyccl as ccl
    cosmo_ccl = ccl.Cosmology(Omega_c=dict_cosmo["Omc"], Omega_b=dict_cosmo["Omb"],
                              h=dict_cosmo["h"], A_s=dict_cosmo["A_s"],
                              n_s=dict_cosmo["n_s"])
    cM = ccl.halos.ConcentrationDuffy08(mass_def=ccl.halos.MassDef200m)
    M_nat = df_halo["mass"].to_numpy()[no_c] / dict_cosmo["h"]   # Msun
    df_halo.loc[no_c, "c"] = cM(cosmo_ccl, M_nat, 1.0 / (1.0 + zeff))
    print(f"  {no_c.sum():,} halos without c -> Duffy08 c(M200m)")
if "promoted" in df_halo:
    print(f"  promoted (distinct-halo) halos: {int(df_halo['promoted'].sum()):,}")

print("Loading particle catalog...")
df_part = pd.read_parquet(PARTICLE_PATH)
print(f"  {len(df_part)} particles loaded")

# ============================================================================
# Initialize HaloOccupation
# ============================================================================

print("\nInitializing HaloOccupation...")
halo = HaloOccupation(
    cosmology=dict_cosmo,
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

print(f"\nSetting HOD model: {HOD_TYPE}")
halo.set_halo_model(HOD_TYPE)

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
    hod_type=HOD_TYPE,
    hod_param_names=np.array(list(base_hod_params)),
    hod_param_values=np.array([float(np.ravel(hod_params[k])[0])
                               for k in base_hod_params]),
    halo_path=HALO_PATH,
    particle_path=PARTICLE_PATH,
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
