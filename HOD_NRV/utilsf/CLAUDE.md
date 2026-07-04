# HOD_NRV/utilsf — Shared Utilities

Shared utilities used by both the analytical and numerical HOD modules, plus
the emulator pipeline for fast Bayesian inference.

## Module Map

```
emulator_utils.py     LHS grid generation, Ac rescaling, run_hod_grid(), merge_grid_chunks()
emulator_nn_flax.py   CosmoPower / Flax emulator (load_emulator, predict_dsigma, predict_wgg)
fieldmesh.py          TSC + interlacing + MAS field machinery: PowerSpectrumEstimator,
                      log_kbins, AssemblyBiasEnvironment, compute_assembly_bias_properties
numerical_sampler.py  EmulatorFitter (Nautilus) for DeltaSigma (+ optional w_gg)
data_reader.py        Parquet I/O, cosmology setup (colossus/astropy)
hankel_transforms.py  Hankel transforms (P(k) ↔ ξ(r))
utils_functions.py    JAX samplers, 200-pt Gauss-Legendre quadrature
```

Files held locally but not tracked (in `.gitignore`):

```
subhalo_catalogue.py  Subhalo catalogue scratch
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

## Emulator Pipeline

The emulator pipeline replaces the slow numerical forward model (~3.2 s/call)
with a neural emulator (~μs/call), making full Nautilus nested sampling
feasible.

### End-to-End Workflow

```
0. precompute_halo_center_cache.py   [example_scripts/]
        ↓  HaloCenterLensingCache saved to .h5 (one-time, ~1-2h)
        ↓  enables method="optimized": 3-9x faster per grid point
1. generate_hod_parameter_grid()     [emulator_utils.py]
        ↓  LHS grid, Ac rescaled to fixed n_gal
2. run_hod_grid(..., precomputed_cache=cache)   [emulator_utils.py]
        ↓  compute_avg_lensing(method="optimized") per row
        ↓  fault-tolerant checkpointing; MPI-parallelised in run_emulator_grid.py
3. merge_grid_chunks()               [emulator_utils.py]
        ↓  concatenate per-rank .npz files, drop failed rows
4. train_emulator()                  [emulator_nn_flax.py]
        ↓  cosmopower_NN, log10(ΔΣ) output, multi-stage LR
5. EmulatorFitter.run()              [numerical_sampler.py]
        ↓  Nautilus on emulator likelihood (~μs/call)
        ↓  full posterior in < 1h
```

---

## `emulator_utils.py`

### `generate_hod_parameter_grid(halo, hod_type, param_ranges, n_samples, target_ngal, ...)`

Latin Hypercube grid of HOD parameters with Ac rescaled so every row achieves
the same galaxy number density. Returns a `pd.DataFrame` with columns
`[Ac, <free params>]`. Call `halo.set_halo_model()` before invocation.

### `rescale_Ac_to_target_ngal(hod_model, params, target_ngal, Ac_fiducial)`

Rescales both `Ac` and `As` by the same factor to hit `target_ngal`,
preserving the `Ac/As` ratio. Returns `(Ac_new, As_new)`.

### `run_hod_grid(...)`

Evaluates `halo.compute_avg_lensing()` for every row in `param_grid`.
Single-process — MPI slicing is the caller's job.

- Pass `precomputed_cache` (a loaded `HaloCenterLensingCache`) for
  `method="optimized"` (3–9× faster).
- Always construct `halo` with `particle_fraction=0.05`.
- Incremental `.npz` checkpoints every `checkpoint_every` points; resume on
  re-run.
- Failed rows stored as `NaN`, dropped by `merge_grid_chunks()`.

### `merge_grid_chunks(directory, n_ranks, output, rank_prefix)`

Concatenates per-rank `.npz` files, drops all-NaN rows, saves
`grid_merged.npz`.

---

## `emulator_nn_flax.py`

CosmoPower / Flax emulator mapping HOD parameters → log10 ΔΣ(rp) (and a
parallel w_gg emulator).

- `train_emulator(...)` — train in log10(ΔΣ) space; 4-stage learning-rate
  schedule, saves weights + `.meta.npz` (rp_centers, param_names).
- `load_emulator(path) → (model, norm_stats)` — returns the model plus the
  metadata dict (`rp_centers`, `param_names`).
- `predict_dsigma(model, norm_stats, params)` / `predict_wgg(...)` — batch
  inference; handles log10/10^x internally.

---

## `numerical_sampler.py`

Nautilus wrapper backed by the emulator.

### `FitCase` (IntEnum)

| Value | Name | Free params | Description |
|-------|------|-------------|-------------|
| 1 | `STANDARD_NFW` | 7 | Standard NFW satellite profile |
| 2 | `EXTENDED_PROFILE` | 9 | Adds exponential cutoff (`f_exp`, `tau`) |
| 3 | `CONFORMITY` | 10 | Adds AbacusHOD conformity (`kappa_EE`) |

`M1` is fixed (default `M1=13.0`). `Ac` / `As` are baked into the training
grid by `rescale_Ac_to_target_ngal()`.

### `EmulatorFitter`

Fast emulator-backed Nautilus sampler.

**Constructor (key args):**
```python
EmulatorFitter(
    emulator_path="emulator_STANDARD_NFW",
    data_path="observed_ds.npz",      # or pass (ds_obs, cov_inv, rp_obs)
    fit_case=FitCase.STANDARD_NFW,
    M1_fixed=13.0,
    rp_min=None, rp_max=None,
    # optional joint w_gg
    emulator_wgg_path="...",
    data_path_wgg="...",
    rp_min_wgg=1.0,
    # optional f_sat truncation matching grid-time rejection
    max_fsat=0.2, hod_occupation=halo.HOD, Ac_fiducial=0.01,
    # optional custom prior dict
    param_config={"Mmin": (12.0, 13.5), "alpha": 1.0},
)
```

- The emulator is trained on the **full HOD parameter vector** (including
  `Ac`, `As`) produced by `generate_hod_parameter_grid()`. The emulator's rp
  grid may differ from the observed rp grid — `EmulatorFitter` handles this
  with log-cubic interpolation (`interpax.Interpolator1D`) inside
  `log_likelihood()`.
- `param_config` is a unified prior dict: `(low, high)` → uniform free,
  `(mean, std, "gaussian")` → Gaussian free, scalar → fixed.
- `max_fsat` applies the same `f_sat > max_fsat` rejection used in
  `generate_hod_parameter_grid()` at grid build time. Needs `hod_occupation`
  (pass `halo.HOD`).

**`run(n_live, n_eff, n_workers=1, ...)`**: emulator calls are so cheap that
`n_workers=1` is usually sufficient.

### Fork-Safety Pattern

`_emulator_fitter_instance` is set module-level before Nautilus creates worker
processes. Workers inherit the instance via fork copy-on-write without
pickling the full object. Reset to `None` in a `finally` block.

---

## Recommended Downsampling Settings

From `example_scripts/benchmark_dsigma_convergence.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `particle_fraction` | **0.05** | Set at `HaloOccupation` init; <0.5% systematic error |
| `galaxy_fraction` | **0.10** | Passed to `compute_avg_lensing()`; <1% error |
| `n_realizations` | **5** | ~0.8% median SE; good precision/cost tradeoff |
| Combined speedup | **382×** | vs full-resolution single realization |

These are the defaults in `example_scripts/run_emulator_grid.py` and should
be used for all grid evaluations.

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

`M1` is fixed at `13.0` by default. `Ac` is derived via
`rescale_Ac_to_target_ngal()`.
