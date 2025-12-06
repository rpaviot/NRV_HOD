# Fast Two-Point Calculator

Fast galaxy-galaxy lensing calculator using pre-computed ΔΣ and spatial interpolation.

## Overview

This module implements an optimized lensing calculation method based on Yuan et al. (2021) - AbacusHOD paper ([arXiv:2110.11412](https://arxiv.org/abs/2110.11412)). The key optimization is:

1. **Pre-compute once** (expensive): Calculate ΔΣ at particle positions near halos
2. **Evaluate fast** (cheap): Interpolate ΔΣ for galaxy positions during HOD sampling

This provides **~100-1000× speedup** compared to computing correlations for each HOD realization, enabling full MCMC posterior sampling instead of just optimization.

## Module Contents

The `twopoint_calculator` module contains:

- **Fast calculator** (recommended): `FastDeltaSigmaCalculator`, `precompute_lensing_grid`, etc.
- **Legacy implementation**: `compute_galaxy_lensing`, `DeltaSigmaCalculator` (for backward compatibility and null testing)

## Imports

```python
# Fast calculator (recommended for HOD sampling)
from HOD_NRV.twopoint_calculator import FastDeltaSigmaCalculator
from HOD_NRV.twopoint_calculator import precompute_lensing_grid, save_precomputed_lensing

# Legacy functions (for backward compatibility)
from HOD_NRV.twopoint_calculator import compute_galaxy_lensing, DeltaSigmaCalculator

# Original module still accessible
from HOD_NRV import two_point  # Original location
```

## Workflow

### Step 1: Pre-computation (run once per snapshot)

```python
from HOD_NRV.twopoint_calculator import precompute_lensing_grid, save_precomputed_lensing
import numpy as np

# Load your halo and particle catalogs
halo_positions = ...  # shape (N_halos, 3)
halo_rvir = ...       # shape (N_halos,)
particle_positions = ...  # shape (N_particles, 3)

# Define radial bins
rp_bins = np.logspace(-1, 1.5, 15)  # Mpc/h

# Pre-compute ΔΣ at particle positions
positions, deltasigma = precompute_lensing_grid(
    halo_positions=halo_positions,
    halo_rvir=halo_rvir,
    particle_positions=particle_positions,
    RHO_M=8.6e10,  # Msun/h / (Mpc/h)^3
    rp_bins=rp_bins,
    Lbox=1000.0,   # Mpc/h
    r_factor=3.0,  # Search within 3×R_vir
    verbose=True
)

# Save to disk
metadata = {
    'cosmology': 'Planck2018',
    'redshift': 0.5,
    'RHO_M': 8.6e10,
    'Lbox': 1000.0
}
save_precomputed_lensing(
    'precomputed_lensing_z0.5.h5',
    positions, deltasigma, rp_bins, metadata
)
```

**Note:** This step is expensive (potentially hours for large simulations) but only needs to be done once per snapshot.

### Step 2: Fast Evaluation (per HOD realization)

```python
from HOD_NRV.twopoint_calculator import FastDeltaSigmaCalculator
from HOD_NRV.HOD_catalogue import HaloOccupation

# Initialize fast calculator
calc = FastDeltaSigmaCalculator('precomputed_lensing_z0.5.h5')

# Populate galaxies with HOD
halo = HaloOccupation(...)
halo.set_halo_model("LRG")
halo.populate_haloes(hod_params)

# Compute lensing signal (FAST!)
rp_bins = np.logspace(-1, 1.5, 15)
rp, delta_sigma = calc.compute_deltasigma_for_galaxies(
    halo.positions_gal,
    rp_bins=rp_bins
)

# Plot
import matplotlib.pyplot as plt
plt.loglog(rp, delta_sigma)
plt.xlabel(r'$r_p$ [Mpc/h]')
plt.ylabel(r'$\Delta\Sigma$ [M$_\odot$ h/pc$^2$]')
plt.show()
```

## Key Features

- **Fast neighbor queries**: Uses scipy's cKDTree with periodic boundary conditions
- **Flexible interpolation**: Supports inverse distance weighting (default) or RBF
- **Memory efficient**: HDF5 storage with compression
- **Drop-in replacement**: Compatible interface with legacy `compute_galaxy_lensing`

## Method Details

The lensing signal is computed as a linear sum:

```
ΔΣ_total(rp) = Σ_i ΔΣ_i(rp)
```

where ΔΣ_i is interpolated from k-nearest pre-computed values using inverse distance weighting:

```
ΔΣ_i(rp) = Σ_j w_j * ΔΣ_j(rp) / Σ_j w_j
```

with weights `w_j = 1 / d_j^p` (default p=2) and d_j = distance to neighbor j.

## Limitations

- Only works for galaxies near halos (within 3×R_vir by default)
- Pre-computation is expensive and storage-intensive
- Interpolation accuracy depends on particle density
- Best suited for HOD models where you evaluate many realizations

## Performance

Typical performance on a 2 Gpc/h box:

- **Pre-computation**: ~2-4 hours (once)
- **Per-realization**: ~0.1-1 seconds (vs ~100-1000 seconds legacy)
- **Speedup**: ~100-1000×
- **Storage**: ~1-10 GB (compressed HDF5)

## References

1. Yuan, S., et al. (2021). *AbacusHOD: A highly efficient extended multi-tracer HOD framework*. MNRAS. [arXiv:2110.11412](https://arxiv.org/abs/2110.11412)

2. Mandelbaum, R., et al. (2006). *Precision photometric redshift calibration for galaxy-galaxy lensing*. MNRAS 368, 715.

3. Cacciato, M., et al. (2009). *Cosmological constraints from a combination of galaxy clustering and lensing*. MNRAS 394, 929.
