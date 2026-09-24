# HOD_NRV/utilsf — Shared Utilities

Shared utilities used by both the analytical and numerical HOD modules, plus
the Nautilus sampler for the tabulated ΔΣ / w_gg likelihood.

## Module Map

```
fieldmesh.py          TSC + interlacing + MAS field machinery: PowerSpectrumEstimator,
                      log_kbins, AssemblyBiasEnvironment, compute_assembly_bias_properties
numerical_sampler.py  TabulatedFitter (Nautilus) for DeltaSigma (+ optional w_gg)
data_reader.py        Parquet I/O, cosmology setup (colossus/astropy)
hankel_transforms.py  Hankel transforms (P(k) ↔ ξ(r))
utils_functions.py    JAX samplers, 200-pt Gauss-Legendre quadrature
```

Files held locally but not tracked (in `.gitignore`):

```
subhalo_catalogue.py  Subhalo / NISP catalogue builder (used by internal/precompute_subhalo_catalogue.py)
emulator_utils.py     Deprecated NN-emulator LHS grid tools
emulator_nn_flax.py   Deprecated NN emulator
```

---

## `fieldmesh.py`

Anything that lays a field on a periodic mesh lives here. Built on the same
`pysco` primitives (TSC mass-assignment, FFTs, Morton reordering, MAS
compensation) used throughout the project. Public API:

### `PowerSpectrumEstimator(Nmesh, Lbox, threads=32)`

- `delta_k(positions, normalize=True) → np.ndarray (complex64)` — particle
  positions in `[0, Lbox)` (shape `(N, 3)`, Mpc/h) → Fourier-space
  density-contrast field. TSC + Jing-2005 interlacing + MAS compensation.
- `auto_pk(deltak, kbins, n_tracer, subtract_shot_noise=True) → (k, P, n_modes)`
  — auto P(k), optional Poisson shot noise (`Lbox³ / n_tracer`).
  Normalization: `(Lbox / Nmesh²)³`.
- `cross_pk(deltak1, deltak2, kbins) → (k, P, n_modes)` — cross-power.

### `log_kbins(Lbox, Nmesh, n_bins=20, k_min=None, k_max=None)`

Log-spaced k bins between the fundamental mode `2π/Lbox` and the Orszag 2/3
anti-aliasing cutoff.

### `AssemblyBiasEnvironment(Nmesh=1024, Lbox=681, smoothing_radius=1.0, threads=32)`

Density-contrast δ(x) + tidal-shear q_R² at halo positions, with
mass-binned rank-normalization (Paviot et al. 2024, Eq. 15-16).

- `compute_density_field(particle_positions)` → real-space δ at fixed
  smoothing_radius
- `compute_shear_field(deltak, halo_positions)` → q_R² at halo positions
- `compute_multiscale_properties(...)` → per-halo δ (and q_R²) interpolated
  to R = rvir_factor * r_vir, evaluated on an internal grid of R
- `compute_environmental_properties(...)` → δ, q_R², plus mass-binned
  normalized (f_A, f_B) ranks

### `compute_assembly_bias_properties(halo_catalogue, particle_positions, ...)`

High-level convenience wrapper: load halo + particle parquet files (or pass
arrays/DataFrames), call `AssemblyBiasEnvironment`, return the property dict.

---

## `numerical_sampler.py`

Nautilus wrapper around the tabulated forward model
(`TabulatedDeltaSigma`, and optionally `TabulatedWgg` for joint fits).

### `FitCase` (IntEnum)

| Value | Name | Free params | Description |
|-------|------|-------------|-------------|
| 1 | `STANDARD_NFW` | 7 | Standard NFW satellite profile |
| 2 | `EXTENDED_PROFILE` | 9 | Adds exponential cutoff (`f_exp`, `tau`) |
| 3 | `CONFORMITY` | 10 | Adds AbacusHOD conformity (`kappa_EE`) |

`M1` is fixed (default `M1=13.0`).

### `TabulatedFitter`

```python
TabulatedFitter(
    tabulated_ds=tab,                   # TabulatedDeltaSigma
    occupation_rescale=occ,             # supplies the analytic mass function
    target_ngal=2.3e-4,
    fit_case=FitCase.EXTENDED_PROFILE,
    data_path="data.npz",               # or (ds_obs, cov_inv, rp_obs)
    rp_min=0.1, rp_max=None,
    param_config={"Mmin": (11.0, 13.0), "f_exp": 0.68},
    ngal_anchor="catalogue",            # or "mass_function"
    # optional joint w_gg
    tabulated_wgg=tab_wgg, data_path_wgg="data.npz", rp_min_wgg=0.1,
)
```

- Each call rescales (Ac, As) by a common factor to hit `target_ngal`, so
  ΔΣ and w_gg only see the Ac/As ratio.
- `ngal_anchor="catalogue"` sums the occupation over the tabulation's halo
  cells, assembly bias included. It is required when `B_cent` / `B_sat` are
  sampled. `"mass_function"` integrates against the analytic mass function
  and is only valid with no assembly bias.
- `param_config`: `(low, high)` → uniform free, `(mean, std, "gaussian")` →
  Gaussian free, scalar → fixed.
- `run(n_live, n_eff, vectorized=True)` uses the jit/vmap-batched likelihood
  (`build_batched_loglike`) in a single process. Never put JAX inside a
  forked pool: jitted calls in forked workers deadlock.

---

## Parameter Reference

| Parameter | Range | Description |
|-----------|-------|-------------|
| `As` | (0.002, 0.05) | Satellite amplitude (Ac derived from this) |
| `Mmin` | (11.5, 13.5) | Minimum halo mass threshold [log10 M☉/h] |
| `sig_M` | (0.1, 2.0) | HOD mass scatter |
| `gamma` | (0.0, 10.0) | Power-law slope for SFR tail (ELG_mHMQ) |
| `alpha` | (0.1, 2.0) | Satellite power-law index |
| `kappa` | (0.1, 2.0) | Satellite mass cutoff factor |
| `lambda_NFW` | (0.1, 2.0) | NFW concentration rescaling |
| `f_exp` | (0.1, 0.9) | Exponential profile fraction (EXTENDED only) |
| `tau` | (1.0, 10.0) | Exponential decay scale in Rs units (EXTENDED only) |
| `kappa_EE` | (0.5, 2.0) | Conformity strength: M1_EE = kappa_EE × M1 (CONFORMITY only) |

`M1` is fixed at `13.0` by default. `Ac` / `As` are rescaled together to
the target number density (`rescale_Ac_to_target_ngal()` in
`HOD_numerical/HOD_models.py`).
