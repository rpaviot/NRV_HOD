"""
Compute assembly bias environmental properties for host_catalogue.parquet.

For each halo, the density field and tidal shear are smoothed at a scale
R = 2.25 * Rvir (per-halo adaptive smoothing). Both raw and mass-binned
normalized values are saved.

Resolution caveat: the adaptive scale is only meaningful where the mesh can
represent it. The cell is Lbox/Nmesh, and a Gaussian at R < cell/2 filters
almost nothing (at Nmesh=256, cell = 2.66 Mpc/h, while the median 2.25*Rvir is
0.33 Mpc/h -- 99.99% of halos below one cell, so "adaptive" meant "the raw
mesh field" for all of them). Nmesh=1024 gives cell = 0.665 Mpc/h and makes
R >~ 0.5 real; genuine adaptivity down to 0.25 Mpc/h needs Nmesh=2048.

--fixed_radii additionally saves the same fields at FIXED smoothing scales.
The grid over R is built anyway for the adaptive interpolation, so a scan over
scale is essentially free -- and since the measured assembly bias is carried
by the DENSITY (delta_norm recovers 73% of it against fs_norm's 35%), scanning
R on delta is how the right column gets chosen instead of guessed.

Normalization follows Paviot et al. (2024):
    f = normalize(log(1 + property))
clipped to [-1, 1] per mass bin (30 bins, 2nd–98th percentiles).

Output: host_catalogue_ab.parquet — original catalog + 4 new columns:
    delta_h    : raw local overdensity at R = 2.25 Rvir
    delta_norm : log-normalized delta_h in [-1, 1]  (use as fE in HOD)
    qr2        : raw tidal shear q_R^2 at R = 2.25 Rvir
    fs_norm    : log-normalized qr2 in [-1, 1]       (use as fB in HOD)
plus, per --fixed_radii entry R:
    delta_h_R<tag>, delta_norm_R<tag>, qr2_R<tag>, fs_norm_R<tag>
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
_p.add_argument("--nmesh", type=int, default=1024,
                help="Mesh per side. Cell = Lbox/Nmesh; a Gaussian at "
                     "R < cell/2 does no filtering, so this sets the smallest "
                     "smoothing scale that means anything.")
_p.add_argument("--r_min", type=float, default=0.5)
_p.add_argument("--r_max", type=float, default=6.0)
_p.add_argument("--dr", type=float, default=0.25)
_p.add_argument("--rvir_scale", type=float, default=1e-3,
                help="Multiplier taking the catalogue's rvir column to Mpc/h. "
                     "Every host catalogue here stores kpc/h, hence 1e-3.")
_p.add_argument("--fixed_radii", type=float, nargs="*",
                default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                help="Fixed smoothing scales to save alongside the adaptive "
                     "one. Free: the R grid is built anyway.")
_args = _p.parse_args()
HALO_PATH, PARTICLE_PATH, OUTPUT_PATH = (
    _args.halo_path, _args.particle_path, _args.output)

LBOX         = 681.0   # Mpc/h
NMESH        = _args.nmesh
RVIR_FACTOR  = 2.25    # R_smooth = 2.25 * Rvir per halo
R_MIN        = _args.r_min   # Mpc/h  (smoothing scale grid lower bound)
R_MAX        = _args.r_max   # Mpc/h  (smoothing scale grid upper bound)
DR           = _args.dr      # Mpc/h  (grid step)
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
print(f"Nmesh         : {NMESH},  Lbox = {LBOX} Mpc/h,  "
      f"cell = {LBOX / NMESH:.3f} Mpc/h")
print(f"Fixed radii   : {_args.fixed_radii}")
if R_MIN < 0.5 * LBOX / NMESH:
    print(f"  WARNING: r_min = {R_MIN} is below half a cell "
          f"({0.5 * LBOX / NMESH:.3f}) -- the Gaussian filters nothing there")
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
    rvir_scale=_args.rvir_scale,   # that column is kpc/h; r_min/r_max are Mpc/h
    rvir_factor=RVIR_FACTOR,
    r_min=R_MIN,
    r_max=R_MAX,
    dr=DR,
    fixed_radii=_args.fixed_radii,
    threads=THREADS,
)

# ---------------------------------------------------------------------------
# Attach results to halo catalog and save
# ---------------------------------------------------------------------------

df = pd.read_parquet(HALO_PATH)
for key, col in results.items():
    df[key] = col
print(f"\nadded {len(results)} columns: {', '.join(sorted(results))}")

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
