# HOD_analytical Module

Semi-analytical HOD framework for computing galaxy power spectra, projected clustering, and galaxy-galaxy lensing using pyccl, JAX, and optionally the Dark Emulator.

## Module Map

```
__init__.py          Public API: HaloModel, AnalyticalHOD, StandardHOD, CSMF_HOD, create_hod()
halo_model.py        HaloModel class (inherits Cosmology) - main user interface
pycosmo.py           Cosmology base class - pyccl + Dark Emulator integration
hod_analytical.py    HOD occupation functions and wrapper classes (JAX @jit)
power_spectrum.py    Pure JIT-compiled P_gg / P_gm computation functions
emu.py               BetaNLInterpolator - beta^NL(k, M1, M2, z) from Dark Emulator
sampler.py           CSMFFitter - Nautilus / iminuit fitting framework
null_test.py         Validation: compares JAX implementation vs pyccl reference
```

## Class and Module Details

### `halo_model.py` - `HaloModel(Cosmology)`

Main entry point. Inherits from `Cosmology` (pycosmo.py) and wraps HOD + power spectrum computation.

**Key methods:**
- `set_hod_params(hod_params)` - update HOD parameters
- `update_f(f_c, f_s)` - update NFW profile scaling factors
- `ngal(hod_params)` - galaxy number density
- `Pgg(hod_params)` - galaxy-galaxy power spectrum
- `Pgm(hod_params)` - galaxy-matter power spectrum
- `effective_halo_mass(hod_params)` - mean halo mass
- `satellite_fraction(hod_params)` - satellite fraction
- `wgg(rp, rp_bins, hod_params)` - projected clustering (requires pycorr)
- `DeltaSigma(rp, rp_bins, method, hod_params, ...)` - excess surface density
- `diagnose_beta_nl_terms(iz, hod_params)` - individual I^NL contributions

Precomputes mass function, halo bias, linear P(k), NFW profiles, and optionally beta^NL grids at Gauss-Legendre nodes per redshift.

### `pycosmo.py` - `Cosmology`

Base cosmology class. Initializes pyccl cosmology, mass definition (default 200c), concentration, mass function (Tinker10), halo bias (Tinker10), and optionally the Dark Emulator.

**Key methods:**
- `linear_power(k, z)` / `nonlinear_power(k, z)` - power spectra
- `compute_beta_nl(z_values)` - build BetaNLInterpolator
- `beta_nl(k, M1, M2, z)` - query interpolated beta^NL
- `get_k()` / `get_k_h()` - k array in user's or h-units

**Unit conventions:** `units_per_h=True` means user provides k in h/Mpc and gets results in h-units; internally converts to 1/Mpc for pyccl.

### `hod_analytical.py` - HOD Occupation Functions and Classes

**Occupation functions** (all `@jit`):
- `lrg_N_central()` - LRG Zheng+07 (error function)
- `elg_ghod_N_central()` - ELG Gaussian HOD
- `elg_sfr_N_central()` - ELG SFR-based (Gaussian + power-law tail)
- `unified_N_satellite()` - power-law with kappa*Mmin cutoff

**GL integration utilities** (N_GL = 200 nodes):
- `gl_nodes_scaled(a, b)` / `gl_integrate(integrand, a, b)`

**Wrapper classes:**
- `AnalyticalHOD(hod_type)` - primary class for LRG / ELG_GHOD / ELG_SFR
- `StandardHOD` - legacy parametrization (deprecated)
- `CSMF_HOD(params)` - conditional stellar mass function HOD

**Factory:** `create_hod(hod_type, **kwargs)` returns the appropriate class.

**Parameter definitions:**
```python
HOD_PARAM_DEFINITIONS = {
    'LRG':      {'central_params': ['Ac', 'log10Mmin', 'sig_M'],
                 'satellite_params': ['As', 'log10Mmin', 'log10M1', 'alpha', 'kappa']},
    'ELG_GHOD': {same as LRG},
    'ELG_SFR':  {'central_params': [..., 'gamma'], ...}
}
CSMF_HOD_PARAMS = ['M0', 'M1', 'gamma1', 'gamma2', 'sigma_c', 'alpha_s', 'b0', 'b1']
```

### `power_spectrum.py` - Pure JIT Functions

No class state; all functions are `@jit`-compiled and called by `HaloModel`.

- `nfw_fourier_u(k, R_s, c, f_scale)` - NFW Fourier transform (analytic si/ci)
- `_compute_ngal(...)` - galaxy density from mass integrals
- `_compute_Pgg(...)` / `_compute_Pgm(...)` - standard 1h+2h power spectra
- `_compute_Pgg_with_beta_nl(...)` / `_compute_Pgm_with_beta_nl(...)` - beta^NL-enhanced versions
- `_compute_I_NL_22`, `_compute_I_NL_12`, `_compute_I_NL_21` - individual I^NL integral terms

### `emu.py` - `BetaNLInterpolator`

Builds a 3D interpolator on (log k, log M1, log M2) per redshift from the Dark Emulator.

- `__call__(k, M1, M2, z)` - interpolate beta^NL (JAX-compatible)
- `interpolate_to_mass_grid(log10M_target, z, k_target)` - grid interpolation
- Bias extraction: `'traditional'` or `'halo-halo'` (default, self-consistent with emulator)
- Force-to-zero corrections: `'none'`, `'additive'` (default), `'multiplicative'`, `'exponential'`

### `sampler.py` - `CSMFFitter`

CSMF HOD fitting with Nautilus nested sampling and iminuit minimization.

**Data containers:** `MassBinData`, `ParameterPrior`, `MinuitResult`, `DifferentialEvolutionResult`

**Key methods:**
- `load_data(data_dir, mass_bins, file_pattern)` - load NPZ data files
- `set_priors(priors, fixed_params)` - configure parameter space
- `minimize(start_params)` - iminuit MIGRAD + HESSE/MINOS
- `minimize_de(...)` - scipy differential_evolution
- `run(n_live, n_eff)` - Nautilus nested sampling
- `profile_likelihood(param_name, param_range, n_points)`
- `save_results(filepath)` / `create_fitter_from_config()`

## Module Dependency Graph

```
halo_model.py
    |-- inherits Cosmology from pycosmo.py
    |-- uses create_hod() from hod_analytical.py
    |-- calls functions from power_spectrum.py
    +-- optionally uses BetaNLInterpolator from emu.py

pycosmo.py
    +-- optionally uses BetaNLInterpolator from emu.py

sampler.py
    +-- uses HaloModel from halo_model.py
```

## External Dependencies

**Required:** pyccl, JAX, NumPy, SciPy

**Optional:**
- `dark_emulator` + `interpax` - beta^NL support
- `iminuit` - fast minimization in sampler
- `nautilus` - nested sampling in sampler
- `pycorr` - Hankel transforms for wgg / DeltaSigma
- `matplotlib` - plotting in null_test

## Usage Patterns

### Basic power spectrum computation

```python
from HOD_NRV.HOD_analytical import HaloModel

cosmo = {'h': 0.6774, 'Omc': 0.2589, 'Omb': 0.0486, 'n_s': 0.9667, 'A_s': 2.1e-9}
model = HaloModel(cosmo, z=[0.5, 1.0], hod_type='LRG', units_per_h=True)

hod_params = {'Ac': 1.0, 'log10Mmin': 13.0, 'sig_M': 0.3,
              'As': 1.0, 'log10M1': 14.0, 'alpha': 1.0, 'kappa': 1.0}
model.set_hod_params(hod_params)

k = model.get_k()
Pgg = model.Pgg(hod_params)   # shape (n_z, n_k)
Pgm = model.Pgm(hod_params)
n_gal = model.ngal(hod_params)
```

### Projected statistics (requires pycorr)

```python
rp_bins = np.logspace(-1, 1.5, 15)
rp = 0.5 * (rp_bins[:-1] + rp_bins[1:])

wgg = model.wgg(rp, rp_bins, hod_params)
ds = model.DeltaSigma(rp, rp_bins, 'FFTlog', hod_params)
```

### With beta^NL corrections

```python
model = HaloModel(cosmo, z=[0.5], hod_type='LRG',
                  units_per_h=True, includes_beta_nl=True)
# beta^NL grids are precomputed; Pgg/Pgm automatically include corrections
```

### CSMF HOD fitting

```python
from HOD_NRV.HOD_analytical import HaloModel
from HOD_NRV.HOD_analytical.sampler import CSMFFitter

fitter = CSMFFitter(cosmo, z_bins=[0.5], hod_type='CSMF')
fitter.load_data('data/', mass_bins=[(10.0, 10.5), (10.5, 11.0)])
fitter.set_priors(priors_dict)
result = fitter.minimize(start_params)
```

## Key Technical Notes

- All masses in HOD functions use **log10** units (solar masses or solar masses/h depending on `units_per_h`)
- Gauss-Legendre integration with 200 nodes is used for all mass integrals
- Power spectrum functions in `power_spectrum.py` are stateless and pure; `HaloModel` manages all state
- beta^NL support requires both `dark_emulator` and `interpax` packages
- `null_test.py` validates <1% agreement between JAX implementation and pyccl reference
- **Last successful null test run: 2026-02-03** -- all tests pass (<0.4% P_gg, <0.17% P_gm for both natural and h-units). Fixed k_array input to respect `units_per_h=True` convention (k in h/Mpc).
