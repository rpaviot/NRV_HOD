# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NRVpy is a Python-based Halo Occupation Distribution (HOD) code for modeling galaxy-halo connections in cosmological simulations. It performs halo occupation fitting to projected clustering and galaxy-galaxy lensing 2-point functions.

## Development Commands

This project does not use a formal build system. Development is typically done through:

- **Running Python scripts directly**: `python script_name.py`
- **Jupyter notebooks**: Open `.ipynb` files for analysis and testing
- **Testing NFW profiles**: `python HOD_NRV/test.py`

No formal test suite, linting, or build commands are configured.

## Architecture

### Dual Modeling Approach

1. **Semi-analytical models** (`HOD_analytical/`): Lightcone geometry modeling based on pyccl, JAX, and optionally the Dark Emulator
2. **Numerical models** (`HOD_numerical/`): Direct halo population using parquet catalogs from simulations, with pair-counting via pycorr

### Directory Layout

```
HOD_NRV/
    HOD_analytical/     Semi-analytical power spectra, projected statistics, fitting
    HOD_numerical/      Numerical halo population, clustering, lensing
    utilsf/             Shared utilities (data I/O, assembly bias, Hankel transforms)
```

## Module Documentation

Each sub-module has its own detailed CLAUDE.md with class/method docs, dependency graphs, usage examples, and parameter references.

### HOD_analytical

Semi-analytical HOD framework computing galaxy power spectra (P_gg, P_gm), projected clustering (w_gg), and galaxy-galaxy lensing (DeltaSigma) using pyccl and JAX. Optionally includes beta^NL corrections via the Dark Emulator. Also provides CSMF HOD fitting with Nautilus/iminuit.

**Entry point:** `from HOD_NRV.HOD_analytical import HaloModel`

**Details:** [`HOD_NRV/HOD_analytical/CLAUDE.md`](HOD_NRV/HOD_analytical/CLAUDE.md)

### HOD_numerical

Numerical HOD framework for populating dark matter halos with galaxies from simulation catalogs. Computes galaxy clustering and galaxy-galaxy lensing via pair counting (pycorr) and direct particle methods. Includes standard and fast (KD-tree precomputed) lensing calculators.

**Entry point:** `from HOD_NRV.HOD_numerical.HOD import HaloOccupation`

**Details:** [`HOD_NRV/HOD_numerical/CLAUDE.md`](HOD_NRV/HOD_numerical/CLAUDE.md)

### Shared Utilities (`utilsf/`)

Common functions used by both modules: data I/O (`data_reader.py`), assembly bias environment ranking (`assembly_bias_environment.py`), Hankel transforms (`hankel_transforms.py`), JAX random samplers and GL quadrature (`utils_functions.py`), and emulator helpers (`emulator_utils.py`).

## Technology Stack

- **JAX/JAXlib**: Primary computation framework (JIT compilation, GPU acceleration)
- **NumPy/SciPy**: Core numerical operations
- **Pandas**: Parquet file handling for halo/particle catalogs
- **pyccl**: Cosmological calculations (analytical module)
- **pycorr**: Correlation function pair counting (numerical module)
- **colossus**: Cosmology, mass functions, concentrations
- **Astropy**: Cosmological units and calculations
- **Numba**: Legacy JIT-compiled functions

## Data Handling

- **Input formats**: Parquet files for halo/particle catalogs; HDF5 for precomputed lensing grids
- **Large data files**: `flamingo_0057_downsampled.hdf5` (1.1GB, gitignored)
- **Simulation support**: SWIFT/Flamingo simulation data via swiftsimio

## Important Development Patterns

1. **Cosmological parameters**: Passed as dictionaries to maintain flexibility
2. **Column mapping**: Use `column_mapping` dictionaries to adapt to different simulation formats
3. **Model selection**: HOD models selected via string identifiers (`"LRG"`, `"ELG_GHOD"`, `"ELG_SFR"`)
4. **JAX compatibility**: New numerical functions should be JAX-compatible when possible
5. **Assembly bias**: Supported along any numerical halo property (concentration, merger time, etc.)

## Git Workflow

- **Development branch**: `develop`
- **Main branch**: `main` (use for pull requests)

## Test Status

- **`HOD_analytical/null_test.py`**: Last successful run **2026-02-03**. All tests pass: P_gg and P_gm agree with pyccl reference to <0.4% in both natural and h-units. Note: when `units_per_h=True`, `HaloModel` now expects the input `k_array` in h/Mpc (not 1/Mpc).
- **`HOD_numerical/twopoint_calculator/halo_center_lensing.py`**: Recommended fast-lensing path (`HaloCenterLensingCache` + `precompute_halo_center_lensing`). The older interpolation-based `fast_two_point.py`/`precompute_deltasigma.py` modules were removed.
- **`example_scripts/cross_check_numerical.py`**: Fast numerical regression test. Uses optimal downsampling (5% particles, 10% galaxies, 5 realizations) to validate the numerical DeltaSigma pipeline against the full-resolution baseline in ~3s/realization. Pass threshold: 5% max deviation.

## Data Dependencies

The code expects specific data formats from cosmological simulations. When working with new datasets, ensure proper column mapping and mass definition consistency with the existing framework.
