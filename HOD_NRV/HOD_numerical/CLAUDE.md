# HOD_numerical Module

Numerical HOD framework for populating dark matter halos with galaxies using simulation catalogs, computing galaxy clustering, and galaxy-galaxy lensing via pair counting (pycorr) and direct particle methods.

## Module Map

```
__init__.py                                 Public API: HOD, satellites, twopoint_calculator, test
HOD_models.py                               Occupation class + HOD functions (LRG, ELG, conformity)
HOD/
    __init__.py                             Exports HaloOccupation + population engine functions
    HOD_catalogue.py                        HaloOccupation class - main user interface
    population_engine.py                    Core galaxy population algorithms (centrals, satellites, RSD)
satellites/
    __init__.py                             Exports NFW_jax, NFW
    NFW_jax.py                              JAX NFW profiles: spherical, elliptical, extended, strategy pattern
    NFW.py                                  Legacy NumPy/Numba NFW implementation
twopoint_calculator/
    __init__.py                             Empty; import submodules directly
    standard_two_point_calculator.py        compute_corr, DeltaSigmaCalculator, clustering, lensing
    halo_center_lensing.py                  RECOMMENDED: HaloCenterLensingCache, OptimizedDeltaSigmaCalculator
test/
    __init__.py                             Not auto-imported; import explicitly
    test_satellites.py                      NFW profile null tests (spherical + elliptical)
    test_extended_profiles.py               Extended NFW: continuity, ellipticity inside/outside Rvir
```

## Class and Module Details

### `HOD/HOD_catalogue.py` - `HaloOccupation`

Main entry point. Loads halo/particle catalogs, sets up cosmology, populates galaxies, and delegates to two-point calculators.

**Constructor parameters:**
- `cosmology` (dict: `Om0`, `Ob0`), `zeff`, `Lbox`, `column_mapping`, `mass_definition`
- `DataFrame` / `halo_path` - halo catalog (DataFrame or parquet path)
- `DataFrame_part` - particle catalog for lensing
- `assembly_bias`, `triaxial_NFW`, `apply_rsd`, `rsd_axis`, `NFW_scaled`, `outerprofile`
- `do_test` - run validation on init (calls `test_satellites.run_all_tests()`)

**Key methods:**
- `set_halo_model(hod_type, conformity=False)` - configure HOD model (`"LRG"`, `"ELG_GHOD"`, `"ELG_SFR"`)
- `populate_haloes(dict_params, random_seed=None)` - full population pipeline
- `compute_galaxy_clustering(mode, bins1, ...)` - wrapper for `compute_galaxy_clustering()`
- `compute_galaxy_lensing(bins1, ...)` - wrapper for `compute_galaxy_lensing()`
- `compute_galaxy_lensing_optimized(bins1, precomputed_cache, ...)` - fast lensing using precomputed halo-center profiles

**Key attributes (post-population):**
- `positions_gal`, `velocities_gal` - galaxy positions/velocities (jnp arrays)
- `satellite_fraction` - N_sat / N_total
- `cent_halo_indices` - indices of halos hosting central galaxies (for optimized lensing)
- `positions`, `velocities`, `mass`, `radius`, `concentration`, `vrms`, `logM` - halo arrays
- `RHO_M`, `rsd_factor`, `cosmology` - cosmological quantities

### `HOD/population_engine.py` - Population Algorithms

Stateless functions orchestrated by `HaloOccupation.populate_haloes()`.

- `populate_centrals(key, positions, velocities, probC)` - probabilistic central placement at halo centers; returns `(cent_pos, cent_vel, is_cent)`
- `populate_satellites(key_s, key_s_pos, key_s_vel, ...)` - Poisson-sample N_s, draw NFW positions/velocities; delegates to `NFW_jax.position_satellites()`
- `populate_haloes_full(...)` - complete pipeline: centrals → satellites → combine → RSD; now returns `(positions_gal, velocities_gal, satellite_fraction, cent_halo_indices)`
- `combine_galaxy_populations(cent_pos, cent_vel, sat_pos, sat_vel)` - vstack centrals + satellites
- `apply_rsd_to_galaxies(positions, velocities, rsd_factor, rsd_axis_index, Lbox)` - shift positions along LOS, apply PBC
- `filter_halo_data(pre_cond, has_sat, **arrays)` - utility for double boolean-mask filtering

### `HOD_models.py` - `Occupation` Class and HOD Functions

**HOD functions** (all `@jit`):
- `LRG_Zheng07(logM, Ac, Mmin, sig_M)` - error function central (Zheng+07)
- `ELG_GHOD(logM, Ac, Mmin, sig_M)` - Gaussian central
- `ELG_SFR(logM, Ac, Mmin, sig_M, gamma)` - Gaussian + power-law tail
- `HOD_satellite(logM, As, Mmin, M1, alpha, kappa)` - power-law satellite
- `HOD_satellite_conformity(logM, As, Mmin, M1, alpha, kappa, kappa_EE, has_central)` - AbacusHOD-style conformity with `M1_EE = kappa_EE * M1`

**Helper functions:**
- `compute_ngal_(logM, mass_function, probC, probS)` - galaxy number density via GL integration
- `compute_fsat_(logM, mass_function, probC, probS)` - satellite fraction
- `assembly_bias_mass(logM, A, B, fI, fE)` - shift log mass by `A*fI + B*fE`

**`Occupation` class:**
- `__init__(hod_type, logM_bins, mass_function, assembly_bias, conformity, fI, fE)`
- `set_params(dict_params)` - validate keys, extract parameter lists, apply assembly bias
- `compute_HOD_occupation(logM, dict_params, has_central=None)` - returns `(probC, probS)`
- `compute_central_occupation(logM, dict_params)` / `compute_satellite_occupation(logM, dict_params, has_central=None)`
- `compute_ngal(dict_params)` / `compute_fsat(dict_params)`

**Parameter registry:**
```python
central_funcs = {
    "LRG":      (LRG_Zheng07, ["Ac", "Mmin", "sig_M"]),
    "ELG_GHOD": (ELG_GHOD,    ["Ac", "Mmin", "sig_M"]),
    "ELG_SFR":  (ELG_SFR,     ["Ac", "Mmin", "sig_M", "gamma"]),
}
satellite_params            = ["As", "Mmin", "M1", "alpha", "kappa"]
satellite_conformity_params = ["As", "Mmin", "M1", "alpha", "kappa", "kappa_EE"]
assembly_bias_params        = ["A_cent", "B_cent", "A_sat", "B_sat"]
```

### `satellites/NFW_jax.py` - JAX NFW Profiles

**Core math (all `@jit`):**
- `NFW_CDF(r, Rs, c)` - analytic NFW CDF
- `single_inverse_CDF(u, Rvir, Rs, c)` - inverse CDF via `jnp.interp` on 1000-point grid
- `exponential_profile_CDF_continuous(r, tau, Rs, Rvir)` - exponential CDF for r >= Rvir
- `single_exponential_inverse_CDF_continuous(u, tau, Rs, Rvir, Rmax)` - inverse CDF for exponential component

**Positioning functions** (all `@jit` with `static_argnames=['N_s_tot']`):
- `spherical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, N_s, N_s_tot)` - standard spherical NFW
- `elliptical_NFW_satellites_positions(key, SpherePoints, halo_centers, Rvir, c, shapes, axis_ratios, N_s, N_s_tot)` - triaxial with rotation matrices + axis ratios `[b/a, c/a]`
- `extended_NFW_satellites_positions(...)` - inner NFW (rescaled by `lambda_NFW`) + outer exponential (fraction `f_exp`, scale `tau*Rs`)
- `extended_elliptical_NFW_satellites_positions(...)` - elliptical inside Rvir, isotropic exponential outside

**Strategy pattern:**
- `SatellitePositioningStrategy` (ABC) - abstract base
- `SphericalNFWStrategy`, `EllipticalNFWStrategy`, `ExtendedNFWStrategy`, `ExtendedEllipticalNFWStrategy`
- `position_satellites(key, SpherePoints, ..., triaxial_NFW, f_exp, tau, lambda_NFW)` - unified interface, auto-selects strategy

**Utilities:**
- `sample_unit_sphere_jax(key_theta, key_phi, N)` - uniform points on S^2
- `create_point_on_unit_sphere(key)` - pre-generate 10M sphere points
- `dispersion_velocities_satellites(key, halo_vel, vrms, N_s, N_s_tot)` - Gaussian velocities with `sigma = 0.577 * vrms`

### `satellites/NFW.py` - Legacy NumPy/Numba Implementation

- `NFW_PDF(x)` - universal NFW PDF
- `generate_NFW_profile(key_r, N, x_max=50)` - generate NFW-distributed radii via CDF inversion
- `sample_unit_sphere_jax(key_theta, key_phi, N)` - sphere sampling (JAX-based)
- `get_satellites(...)` - `@njit(parallel=True)` Numba implementation for satellite positioning

### `twopoint_calculator/standard_two_point_calculator.py`

**`compute_corr(mode, catalog1, bins1, ...)`** - pycorr wrapper for Natural estimator (`DD/RR - 1`). Modes: `'s'`, `'smu'`, `'rppi'`. Output: `'auto'`, `'multipoles'`, `'wp'`. Uses all CPU cores.

**`DeltaSigmaCalculator(rr, xi_gm, RHO_M, chi_max=100)`** - surface density contrast from ξ_gm:
- Computes Σ(rp) and ΔΣ(rp) at init via GL quadrature
- `compute_deltasigma(rp)` - evaluate spline at arbitrary radii
- `compute_deltasigma_averaged(r_bins)` - bin-averaged ΔΣ

**`binavg_2D(spline, r_bins)`** - radial bin averaging via GL integration: `<f> = (2/Δr²) ∫ f(r) r dr`

**Convenience wrappers:**
- `compute_galaxy_clustering(positions_gal, Lbox, rsd_axis, mode, bins1, ...)` - clustering
- `compute_galaxy_lensing(positions_gal, positions_part, Lbox, rsd_axis, RHO_M, bins1, ...)` - lensing pipeline (cross-correlate → DeltaSigmaCalculator → bin-average)

### `twopoint_calculator/halo_center_lensing.py` - Optimized Lensing (RECOMMENDED)

Fast galaxy-galaxy lensing by precomputing ΔΣ at halo centers. Since centrals are exactly at halo centers, their lensing contribution can be looked up instantly - no interpolation needed.

**Key Insight:** Central galaxies are at halo centers (`cent_positions = positions[is_cent]`). Precomputing ΔΣ at each halo center allows instant lookup by halo index.

**`HaloCenterLensingCache`** - stores precomputed profiles:
- `__init__(positions, deltasigma, rp_bins, metadata)` - create from arrays
- `save(output_path)` / `load(input_path)` - HDF5 persistence
- Attributes: `positions`, `deltasigma`, `rp_bins`, `rp_centers`, `metadata`

**`precompute_halo_center_lensing(halo_positions, particle_positions, Lbox, rsd_axis, RHO_M, rp_bins, ...)`** - one-time precomputation:
- Returns `HaloCenterLensingCache` with ΔΣ profiles at all halo centers
- Computational cost: O(N_halos) calls to pycorr (~1-2 hours for 200k halos)
- Storage: ~30 MB compressed for 200k halos × 15 rp bins

**`OptimizedDeltaSigmaCalculator`** - runtime calculation:
- `__init__(cache, particle_positions, Lbox, rsd_axis, RHO_M)`
- `compute_deltasigma(cent_halo_indices, sat_positions, satellite_fraction, rp_bins)` - combines lookup (centrals) + runtime (satellites)

**Performance speedup:**
| Scenario    | Standard | Optimized | Speedup |
|-------------|----------|-----------|---------|
| f_sat = 0.3 | 0.64s    | ~0.20s    | ~3×     |
| f_sat = 0.2 | 0.64s    | ~0.13s    | ~5×     |
| f_sat = 0.1 | 0.64s    | ~0.07s    | ~9×     |

### `test/` - Test Modules

| File | Purpose |
|------|---------|
| `test_satellites.py` | NFW null test: fit recovered concentration from spherical + elliptical samples |
| `test_extended_profiles.py` | Extended NFW: ellipticity inside Rvir, isotropy outside, continuity |

## Module Dependency Graph

```
HOD/HOD_catalogue.py
    |-- uses Occupation from HOD_models.py
    |-- uses NFW_jax (create_point_on_unit_sphere)
    |-- uses data_reader from utilsf/ (read_halo_catalog, setup_cosmology, ...)
    |-- calls populate_haloes_full from HOD/population_engine.py
    +-- calls compute_galaxy_clustering / compute_galaxy_lensing from twopoint_calculator/

HOD/population_engine.py
    |-- uses NFW_jax.position_satellites
    +-- uses utils_functions (random_uniform_jax, random_poisson_jax)

HOD_models.py
    +-- uses utils_functions (gauss_legendre_integration)

twopoint_calculator/standard_two_point_calculator.py
    +-- uses pycorr, utils_functions

twopoint_calculator/halo_center_lensing.py
    +-- uses pycorr, scipy (DeltaSigmaCalculator), multiprocessing
```

## External Dependencies

**Required:** JAX, NumPy, SciPy, pandas

**Required for specific features:**
- `pycorr` - all correlation function calculations
- `numba` - legacy NFW implementation (`NFW.py`)
- `colossus` - cosmology and mass function setup (via `data_reader`)
- `h5py` - precomputed lensing HDF5 I/O

## Usage Patterns

### Basic population workflow

```python
from HOD_NRV.HOD_numerical.HOD import HaloOccupation

halo = HaloOccupation(
    cosmology={'Om0': 0.3, 'Ob0': 0.049},
    zeff=1.0, Lbox=681.0,
    column_mapping=column_mapping,
    mass_definition="Mvir",
    DataFrame=df_halo,
    DataFrame_part=df_part,   # needed for lensing
    assembly_bias=False,
    apply_rsd=True,
    triaxial_NFW=False
)

halo.set_halo_model("LRG")

hod_params = {
    "Ac": 1.0, "Mmin": 12.0, "sig_M": 0.4,
    "As": 0.2, "M1": 13.0, "alpha": 0.8, "kappa": 0.8
}
halo.populate_haloes(hod_params, random_seed=42)

print(f"N_gal = {len(halo.positions_gal)}")
print(f"f_sat = {halo.satellite_fraction:.3f}")
```

### Galaxy clustering

```python
import numpy as np

# 3D correlation function xi(s)
s_bins = np.logspace(-1, 2, 20)
s, xi_s = halo.compute_galaxy_clustering('s', s_bins)

# Projected wp(rp)
rp_bins = np.logspace(-1, 1.5, 15)
pi_bins = np.linspace(-50, 50, 51)
rp, wp = halo.compute_galaxy_clustering('rppi', rp_bins, bins2=pi_bins, output='wp')
```

### Galaxy-galaxy lensing (standard)

```python
rp_bins = np.logspace(-1, 1.5, 15)
rp, delta_sigma = halo.compute_galaxy_lensing(rp_bins)
```

### Optimized lensing with halo-center precomputation (RECOMMENDED)

```python
from HOD_NRV.HOD_numerical.twopoint_calculator import (
    HaloCenterLensingCache, precompute_halo_center_lensing
)

# Step 1: One-time precomputation (run once per simulation snapshot)
cache = precompute_halo_center_lensing(
    halo_positions=halo.positions,
    particle_positions=halo.positions_part,
    Lbox=halo.Lbox,
    rsd_axis=halo.rsd_axis,
    RHO_M=halo.RHO_M,
    rp_bins=rp_bins
)
cache.save('halo_center_lensing.h5')

# Step 2: Fast evaluation during HOD sampling (per realization)
cache = HaloCenterLensingCache.load('halo_center_lensing.h5')
rp, delta_sigma = halo.compute_galaxy_lensing_optimized(rp_bins, cache)
```

## Key Technical Notes

- **Units**: all positions in Mpc/h, masses in Msun/h, HOD mass parameters in log10
- **Radii conversion**: NFW sampling outputs radii in kpc/h; divided by 1000 to get Mpc/h in all positioning functions
- **Velocity dispersion**: `sigma = 0.577 * vrms` (isotropic 1D component of 3D dispersion)
- **RSD formula**: `s = r + v / (H(z) * a)` along the chosen LOS axis, with periodic wrapping
- **Strategy selection** in `NFW_jax.position_satellites()`: extended profiles used when `f_exp > 0` or `lambda_NFW != 1`; elliptical when `triaxial_NFW=True`; gives 4 strategy combinations
- **Conformity**: `kappa_EE` scales M1 when central present: `log10(M1_EE) = log10(M1) + log10(kappa_EE)`
- **Precomputed sphere points**: 10M points on S^2 generated at init; permuted for each call to sample directions
- **GL quadrature**: 200-node Gauss-Legendre for all mass/surface density integrals

## HOD Parameter Reference

| Parameter | LRG | ELG_GHOD | ELG_SFR | Description |
|-----------|:---:|:--------:|:-------:|-------------|
| `Ac` | x | x | x | Central amplitude |
| `Mmin` | x | x | x | Minimum mass threshold [log10] |
| `sig_M` | x | x | x | Mass scatter |
| `gamma` | | | x | Power-law slope for SFR tail |
| `As` | x | x | x | Satellite amplitude |
| `M1` | x | x | x | Satellite mass scale [log10] |
| `alpha` | x | x | x | Satellite power-law index |
| `kappa` | x | x | x | Satellite mass cutoff factor |
| `kappa_EE` | * | * | * | Conformity M1 scaling (* = when conformity=True) |
| `A_cent`, `B_cent` | * | * | * | Central assembly bias (* = when assembly_bias=True) |
| `A_sat`, `B_sat` | * | * | * | Satellite assembly bias |
| `f_exp` | + | + | + | Exponential profile fraction [0,1] (+ = optional, default 0) |
| `tau` | + | + | + | Exponential decay scale in Rs units (default 6) |
| `lambda_NFW` | + | + | + | NFW rescaling factor (default 1) |
