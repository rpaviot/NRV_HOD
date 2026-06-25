# HOD Emulator Parameter Generation

This guide explains how to use the `emulator_utils.py` module to generate HOD parameter grids with fixed number density for emulator training.

## Overview

When fitting HOD models to observations, we typically fix the galaxy number density to match the observed sample. For independent central and satellite occupations, the clustering and lensing signals depend on the ratio **Ac/As** rather than Ac and As individually. This allows us to:

1. Sample free parameters: `Mmin, sig_M, gamma, As, M1, alpha, kappa`
2. Compute number density with a fiducial `Ac` value
3. Rescale `Ac` to achieve the target number density

This approach **reduces the parameter space dimensionality by 1**, making Ac a derived parameter rather than a free parameter.

## Key Concept

For a given set of HOD parameters (excluding Ac), the number density is:

```
ngal = ∫ [Ncen(M) + Nsat(M)] * (dn/dM) dM
```

Since `ngal ∝ Ac` (when central/satellite are independent), we can rescale:

```
Ac_new = Ac_fid * (ngal_target / ngal_fid)
```

This is exactly the approach used in Rocher et al. for HOD emulation.

## Quick Start

### Basic Usage (Standard HOD)

```python
from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.utilsf.emulator_utils import generate_hod_parameter_grid

# Setup halo catalog
halo = HaloOccupation(
    cosmology=cosmo_params,
    zeff=1.0,
    Lbox=1000,
    column_mapping=column_mapping,
    mass_definition="vir",
    DataFrame=df,
    assembly_bias=False  # Standard HOD
)

# Define parameter ranges (excluding Ac which is auto-computed)
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

# Generate parameter grid - Ac will be rescaled to match target_ngal
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,  # (Mpc/h)^-3
    Ac_fiducial=0.5,
    random_seed=42,
    save_path='hod_params.parquet'
)
```

### With Assembly Bias

```python
# Enable assembly bias in HaloOccupation
halo = HaloOccupation(
    cosmology=cosmo_params,
    zeff=1.0,
    Lbox=1000,
    column_mapping=column_mapping,
    mass_definition="vir",
    DataFrame=df,
    assembly_bias=True  # Enable assembly bias
)

# Base parameters only - assembly bias params will be auto-added!
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

# Generate grid - A_cent, B_cent, A_sat, B_sat will be auto-added
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    auto_add_defaults=True,  # Automatically adds assembly bias parameters
    random_seed=42
)
# Result: 10D parameter space (6 base + 4 assembly bias)
```

### With Conformity

```python
# Conformity doesn't need special HaloOccupation setup
# Uses M1_EE = kappa_EE * M1 parametrization
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

# kappa_EE will be auto-added when conformity=True
# kappa_EE > 1: satellites prefer halos with centrals
# kappa_EE < 1: satellites avoid halos with centrals
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    conformity=True,  # Enable conformity (adds kappa_EE)
    auto_add_defaults=True,
    random_seed=42
)
# Result: 7D parameter space (6 base + kappa_EE)
# M1_EE is computed as M1_EE = kappa_EE * M1 during population
```

### With Extended NFW Profiles

```python
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

# f_exp, tau, lambda_NFW will be auto-added
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    include_nfw_extensions=True,  # Enable NFW extensions
    auto_add_defaults=True,
    random_seed=42
)
# Result: 9D parameter space (6 base + 3 NFW extensions)
```

### All Extensions Combined

```python
# Setup with assembly bias
halo = HaloOccupation(
    cosmology=cosmo_params,
    zeff=1.0,
    Lbox=1000,
    column_mapping=column_mapping,
    mass_definition="vir",
    DataFrame=df,
    assembly_bias=True  # Enable assembly bias
)

# Base parameters only
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

# All extension parameters will be auto-added!
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    conformity=True,              # +1 param (M1_EE)
    include_nfw_extensions=True,  # +3 params (f_exp, tau, lambda_NFW)
    auto_add_defaults=True,       # Auto-add assembly bias (+4 params)
    random_seed=42
)
# Result: 14D parameter space! (6 base + 4 assembly + 1 conformity + 3 NFW)
```

### Manual Control (No Auto-Defaults)

```python
# If you want full control, disable auto_add_defaults
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'M1': (12.5, 14.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5),
    # Manually specify assembly bias ranges
    'A_cent': (-0.3, 0.3),  # Custom range
    'B_cent': (-0.3, 0.3),
    'A_sat': (-0.3, 0.3),
    'B_sat': (-0.3, 0.3)
}

param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    auto_add_defaults=False,  # No auto-adding
    random_seed=42
)
```

## Extension Auto-Detection

The module **automatically detects** which extensions are enabled and adjusts the parameter space accordingly:

| Extension | How Detected | Additional Parameters | Default Ranges |
|-----------|--------------|----------------------|----------------|
| **Assembly Bias** | `halo.assembly_bias==True` | A_cent, B_cent, A_sat, B_sat | (-0.5, 0.5) each |
| **Conformity** | `conformity=True` argument | kappa_EE (M1_EE = kappa_EE × M1) | (0.5, 2.0) |
| **NFW Extensions** | `include_nfw_extensions=True` | f_exp, tau, lambda_NFW | (0, 0.5), (2, 10), (0.5, 2.0) |

When `auto_add_defaults=True` (default), missing extension parameters are automatically added with sensible default ranges from `DEFAULT_PARAM_RANGES`.

**Example output** (verbose mode):
```
Generating 1000 HOD parameter combinations for ELG_GHOD
Target number density: 1.00e-03 (Mpc/h)^-3

Detected extensions:
  Assembly bias: True
  Conformity: True
  NFW extensions: True

Required parameters: ['A_cent', 'A_sat', 'As', 'B_cent', 'B_sat', 'M1', 'kappa_EE',
                      'Mmin', 'alpha', 'f_exp', 'kappa', 'lambda_NFW', 'sig_M', 'tau']

Auto-adding default ranges for missing parameters: ['A_cent', 'A_sat', 'B_cent',
                                                     'B_sat', 'kappa_EE', 'f_exp',
                                                     'lambda_NFW', 'tau']
  A_cent: (-0.5, 0.5)
  A_sat: (-0.5, 0.5)
  B_cent: (-0.5, 0.5)
  B_sat: (-0.5, 0.5)
  kappa_EE: (0.5, 2.0)
  f_exp: (0.0, 0.5)
  lambda_NFW: (0.5, 2.0)
  tau: (2.0, 10.0)
```

## Available Functions

### `generate_hod_parameter_grid()`
Main function that generates the complete parameter grid.

**Parameters:**
- `halo`: HaloOccupation instance
- `hod_type`: 'LRG', 'ELG_GHOD', or 'ELG_SFR'
- `param_ranges`: Dict of (min, max) for each FREE parameter (to be sampled)
- `n_samples`: Number of parameter combinations
- `target_ngal`: Target number density [(Mpc/h)^-3]
- `Ac_fiducial`: Fiducial Ac value for rescaling
- `fixed_params`: Optional dict of parameters to hold constant (not sampled)
  - Note: With conformity, M1_EE = kappa_EE * M1, so fixing M1 still allows varying conformity via kappa_EE
- `random_seed`: Random seed for reproducibility
- `save_path`: Optional path to save the grid

**Returns:**
- DataFrame with all parameters including rescaled Ac and fixed params

**Note:** Parameters in `fixed_params` should NOT be in `param_ranges`.

### `create_latin_hypercube()`
Generate Latin Hypercube Samples for parameter space exploration.

**Parameters:**
- `param_ranges`: Dict of (min, max) for each parameter
- `n_samples`: Number of samples
- `random_seed`: Random seed

**Returns:**
- DataFrame with LHS samples

### `rescale_Ac_to_target_ngal()`
Rescale Ac to achieve target number density.

**Parameters:**
- `hod_model`: Occupation instance
- `params`: HOD parameters (excluding Ac)
- `target_ngal`: Target number density
- `Ac_fiducial`: Initial Ac value

**Returns:**
- `Ac_rescaled`, `ngal_achieved`

### `compute_ngal_with_fiducial_Ac()`
Compute number density with a fiducial Ac value.

**Parameters:**
- `hod_model`: Occupation instance
- `params`: HOD parameters (excluding Ac)
- `Ac_fiducial`: Fiducial Ac value

**Returns:**
- `ngal`: Number density

## Example Workflow

See `example_emulator_parameter_generation.py` for a complete working example.

```bash
python example_emulator_parameter_generation.py
```

This will:
1. Load a halo catalog
2. Generate a parameter grid with 100 samples
3. Save the grid to parquet
4. Create visualization plots
5. Demonstrate halo population with the parameters

## Conformity Parametrization

**Important:** Conformity uses `kappa_EE` instead of `M1_EE` directly:

```
M1_EE = kappa_EE * M1
```

**Benefits:**
- **Can fix M1** while still varying conformity strength via `kappa_EE`
- **Physical interpretation**: `kappa_EE > 1` means satellites prefer halos with centrals, `kappa_EE < 1` means they avoid them
- **Better for emulation** when M1 is constrained by other observations

**Example with fixed M1:**
```python
# Fix M1 but allow conformity to vary
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0),
    'alpha': (0.5, 1.5),
    'kappa': (0.5, 1.5)
}

fixed_params = {'M1': 13.5}  # M1 is fixed

# kappa_EE will be sampled, M1_EE computed as kappa_EE * M1
param_grid = generate_hod_parameter_grid(
    halo=halo,
    hod_type='ELG_GHOD',
    param_ranges=param_ranges,
    n_samples=1000,
    target_ngal=1e-3,
    conformity=True,  # kappa_EE will be auto-added
    fixed_params=fixed_params,
    auto_add_defaults=True,
    random_seed=42
)
# Result: 5D space (4 free + kappa_EE), M1 fixed, M1_EE = kappa_EE * 13.5
```

## Fixed vs Free Parameters

You can control which parameters are sampled in the LHS and which are held constant:

**Free parameters** (in `param_ranges`): These will be sampled using Latin Hypercube Sampling. Each parameter gets a range `(min, max)`.

**Fixed parameters** (in `fixed_params`): These will be constant across all samples. Useful for:
- Reducing parameter space dimensionality
- Focusing on specific parameter variations
- Computational efficiency

**Example:**
```python
# Sample only 3 parameters, fix the rest
param_ranges = {
    'Mmin': (11.5, 12.5),
    'sig_M': (0.2, 0.6),
    'As': (0.1, 1.0)
}

fixed_params = {
    'M1': 13.5,      # Fixed satellite mass scale
    'alpha': 1.0,    # Fixed power law index
    'kappa': 1.0     # Fixed cutoff
}
```

This generates a 3D parameter space (+ Ac rescaled) instead of 6D, making emulator training faster and requiring fewer samples.

## Parameter Ranges

Typical ranges for ELG HODs:

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `Mmin` | Minimum halo mass threshold | (11.5, 12.5) | log10(Msun/h) |
| `sig_M` | Mass scatter | (0.2, 0.6) | - |
| `gamma` | Power law index (ELG_SFR only) | (0.0, 2.0) | - |
| `As` | Satellite amplitude | (0.1, 1.0) | - |
| `M1` | Satellite mass scale | (12.5, 14.0) | log10(Msun/h) |
| `alpha` | Satellite power law index | (0.5, 1.5) | - |
| `kappa` | Satellite cutoff | (0.5, 1.5) | - |
| `kappa_EE` | Conformity scaling (M1_EE = kappa_EE × M1) | (0.5, 2.0) | - |
| `A_cent`, `B_cent` | Assembly bias (centrals) | (-0.5, 0.5) | - |
| `A_sat`, `B_sat` | Assembly bias (satellites) | (-0.5, 0.5) | - |
| `f_exp` | Exponential profile fraction | (0.0, 0.5) | - |
| `tau` | Exponential decay scale | (2.0, 10.0) | Rs units |
| `lambda_NFW` | NFW rescaling factor | (0.5, 2.0) | - |

For LRG HODs, adjust ranges accordingly (typically higher masses).

## Target Number Densities

Choose `target_ngal` based on your science case:

- **Bright ELGs**: ~1e-4 (Mpc/h)^-3
- **Typical ELGs**: ~1e-3 (Mpc/h)^-3
- **LRGs**: ~1e-4 to 1e-5 (Mpc/h)^-3

## Output Format

The generated DataFrame has columns:

```
Ac, Mmin, sig_M, As, M1, alpha, kappa, ngal_achieved
```

where:
- `Ac`: Rescaled central amplitude for target ngal
- Other columns: LHS-sampled parameters
- `ngal_achieved`: Actual ngal achieved (should ≈ target_ngal)

## Next Steps: Emulator Training

Once you have the parameter grid:

1. **Generate training data:**
   ```python
   for idx, row in param_grid.iterrows():
       params = row.to_dict()
       halo.populate_haloes(params)

       # Compute observables
       r, xi = halo.compute_galaxy_clustering('s', bins)
       rp, ds = halo.compute_galaxy_lensing(rp_bins)

       # Store (params, observables) pairs
   ```

2. **Train emulator** (GP, NN, etc.) to map:
   ```
   params → observables
   ```

3. **Use emulator for inference:**
   - MCMC/nested sampling becomes fast
   - No need to populate halos during inference
   - Emulator provides instant predictions

## References

- Rocher et al. (2023) - HOD emulation methodology
- Zheng et al. (2007) - Standard HOD models
- Yuan et al. (2022) - AbacusHOD conformity models

## Troubleshooting

**Issue:** Ac rescaling doesn't converge
- **Solution:** Adjust `Ac_fiducial` or check parameter ranges
- **Cause:** Parameters may produce very small/large ngal

**Issue:** ngal_achieved differs from target_ngal
- **Solution:** Check `tolerance` parameter in `rescale_Ac_to_target_ngal()`
- **Cause:** Numerical precision or integration limits

**Issue:** Memory errors with large n_samples
- **Solution:** Process in batches or reduce n_samples
- **Consider:** Parallelizing the parameter grid generation
