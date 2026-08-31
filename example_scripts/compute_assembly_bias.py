"""
Compute assembly bias environmental properties for host_catalogue.parquet.

For each halo, the density field and tidal shear are smoothed at a scale
R = 2.25 * Rvir (per-halo adaptive smoothing). Both raw and mass-binned
normalized values are saved.

Normalization follows Paviot et al. (2024):
    f = normalize(log(1 + property))
clipped to [-1, 1] per mass bin (30 bins, 2nd–98th percentiles).

Output: host_catalogue_ab.parquet — original catalog + 4 new columns:
    delta_h    : raw local overdensity at R = 2.25 Rvir
    delta_norm : log-normalized delta_h in [-1, 1]  (use as fE in HOD)
    qr2        : raw tidal shear q_R^2 at R = 2.25 Rvir
    fs_norm    : log-normalized qr2 in [-1, 1]       (use as fB in HOD)
"""

import argparse

import numpy as np
import pandas as pd
from HOD_NRV.utilsf.fieldmesh import compute_assembly_bias_properties

# ---------------------------------------------------------------------------
# Paths and parameters (consistent with run_emulator_grid_subhalo.py)
# ---------------------------------------------------------------------------

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"
PARTICLE_PATH = "/Users/ler13nrv/Documents/flamingo_data/particle_catalogue_L1000N1800_downsampled.parquet"
OUTPUT_PATH   = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800_AB.parquet"

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--halo_path", default=HALO_PATH)
_p.add_argument("--particle_path", default=PARTICLE_PATH)
_p.add_argument("--output", default=OUTPUT_PATH)
_p.add_argument("--threads", type=int, default=20)
_args = _p.parse_args()
HALO_PATH, PARTICLE_PATH, OUTPUT_PATH = (
    _args.halo_path, _args.particle_path, _args.output)

LBOX         = 681.0   # Mpc/h
NMESH        = 256
RVIR_FACTOR  = 2.25    # R_smooth = 2.25 * Rvir per halo
R_MIN        = 0.5     # Mpc/h  (smoothing scale grid lower bound)
R_MAX        = 6.0     # Mpc/h  (smoothing scale grid upper bound)
DR           = 0.25    # Mpc/h  (grid step)
THREADS      = _args.threads

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


size = 31
low_m = 11.
high_m = 14.
bins_mass = np.linspace(low_m, high_m, size)


print(f"Halo catalog  : {HALO_PATH}")
print(f"Particle path : {PARTICLE_PATH}")
print(f"Smoothing     : {RVIR_FACTOR} × Rvir  (grid {R_MIN}–{R_MAX} Mpc/h, step {DR})")
print(f"Nmesh         : {NMESH},  Lbox = {LBOX} Mpc/h")
print()

results = compute_assembly_bias_properties(
    halo_catalogue=HALO_PATH,
    particle_positions=PARTICLE_PATH,
    compute_shear=True,
    Nmesh=NMESH,
    Lbox=LBOX,
    position_columns=("x", "y", "z"),
    mass_column="mass",
    mass_bins=bins_mass,
    rvir_column="rvir",       # column name in host_catalogue.parquet
    rvir_factor=RVIR_FACTOR,
    r_min=R_MIN,
    r_max=R_MAX,
    dr=DR,
    threads=THREADS,
)

# ---------------------------------------------------------------------------
# Attach results to halo catalog and save
# ---------------------------------------------------------------------------

df = pd.read_parquet(HALO_PATH)
df["delta_h"]    = results["delta_h"]
df["delta_norm"] = results["delta_norm"]
df["qr2"]        = results["qr2"]
df["fs_norm"]    = results["fs_norm"]

df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

print("\n--- Summary ---")
for col in ("delta_h", "delta_norm", "qr2", "fs_norm"):
    arr = df[col].values
    print(f"  {col:12s}  min={arr.min():.4f}  median={np.median(arr):.4f}"
          f"  max={arr.max():.4f}  std={arr.std():.4f}")
