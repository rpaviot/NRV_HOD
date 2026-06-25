# DeltaSigma Convergence Benchmark Results

## Summary

This directory contains benchmarks testing the convergence of numerical DeltaSigma calculations with respect to particle and galaxy downsampling.

## Configuration

- **Box size**: 681 Mpc/h
- **Redshift**: z = 1.0
- **Target ngal**: 2×10⁻⁴ (Mpc/h)⁻³
- **rp range**: 0.1 – 50 Mpc/h (15 bins)
- **Tolerance**: 1% fractional deviation

## Key Results

### Particle Downsampling

| Fraction | N particles | Max deviation | Timing (s) |
|----------|-------------|---------------|------------|
| 0.75     | 43.3M       | 0.08%         | 92.2       |
| 0.50     | 28.9M       | 0.19%         | 63.5       |
| 0.25     | 14.4M       | 0.18%         | 34.4       |
| 0.10     | 5.8M        | 0.39%         | 16.4       |
| **0.05** | **2.9M**    | **0.43%**     | **10.7**   |

**Optimal particle fraction: 5%** (all deviations remain < 1%)

### Galaxy Downsampling (at 5% particles)

| Fraction | Max deviation | Timing (s) |
|----------|---------------|------------|
| 0.75     | 0.21%         | 4.5        |
| 0.50     | 0.27%         | 3.0        |
| 0.25     | 0.51%         | 1.5        |
| **0.10** | **0.76%**     | **0.64**   |

**Optimal galaxy fraction: 10%**

### Realization Convergence (5% particles, 10% galaxies)

| N realizations | Median SE | Max SE |
|----------------|-----------|--------|
| 3              | 1.0%      | 2.2%   |
| **5**          | **0.77%** | **1.7%** |
| 10             | 0.47%     | 1.4%   |
| 20             | 0.39%     | 0.95%  |

**Optimal realizations: 5** (good balance of precision vs cost)

## Optimal Settings

| Parameter          | Value |
|--------------------|-------|
| Particle fraction  | 0.05  |
| Galaxy fraction    | 0.10  |
| N realizations     | 5     |
| **Speedup**        | **382×** |

- Baseline: 122.6 s/realization → Optimized: 0.64 s/realization

## Particle Resolution Limits

### Current Setup

The particle catalogue is already downsampled by a factor of **0.99 × 1/100 ≈ 1%** of the original simulation.

- **Original particle mass**: 6.72 × 10⁹ M☉
- **Effective mass after 1% downsampling**: 6.72 × 10¹¹ M☉ per sampled particle

### Additional Downsampling Limit

From the benchmark, we can further downsample particles to **5%** of the current catalogue while keeping systematic errors below 1%.

**Total downsampling from original simulation:**
```
Total fraction = 0.01 × 0.05 = 0.0005 (0.05%)
```

**Effective particle mass resolution at maximum downsampling:**
```
m_eff = 6.72 × 10⁹ M☉ / 0.0005 = 1.34 × 10¹³ M☉
```

### Practical Recommendation

For a simulation with particle mass resolution of **6.72 × 10⁹ M☉**:

| Downsampling stage | Fraction kept | Effective mass per particle |
|--------------------|---------------|----------------------------|
| Initial (current)  | 1%            | 6.72 × 10¹¹ M☉            |
| **Maximum usable** | **0.05%**     | **1.34 × 10¹³ M☉**        |

This corresponds to keeping approximately **1 in 2000** original particles while maintaining < 1% systematic error on DeltaSigma.

## Cross-Check: Analytical vs Numerical

Agreement between numerical and analytical (NL) DeltaSigma varies with scale:
- **Large scales (rp > 10 Mpc/h)**: Agreement within ~5%
- **Intermediate scales (1–10 Mpc/h)**: Deviations up to 25-30%
- **Small scales (rp < 1 Mpc/h)**: Agreement within ~5%

The intermediate-scale disagreement is expected due to differences in halo profile modeling and baryonic effects.

## Numerical-Only Cross-Check

The script `cross_check_numerical.py` provides a fast regression test for the numerical HOD pipeline (population engine, NFW profiles, standard two-point calculator). It does NOT test `halo_center_lensing.py` or the analytical module.

**How it works:**
1. Loads optimal downsampling settings from `benchmark_results.json`
2. Loads the full-resolution baseline from `baseline_dsigma_cache.npz`
3. Runs N optimized realizations with `use_fast=True` satellite positioning, particle subsampling, and galaxy subsampling
4. Compares mean DeltaSigma against the baseline mean
5. Reports deviation metrics and a PASS/FAIL verdict (5% threshold)

**When to run:** After modifying any file in `HOD_NRV/HOD_numerical/` except `halo_center_lensing.py`.

**Outputs:**
- `cross_check_numerical_results.txt` — Per-bin deviation table and summary
- `cross_check_numerical.png` — 2-panel comparison plot

## Emulator Grid Generation (`run_emulator_grid.py`)

This script pre-computes a DeltaSigma grid over a Latin Hypercube parameter space so that a fast neural emulator (see `HOD_NRV/utilsf/emulator_nn.py`) can replace the slow numerical forward model during Nautilus sampling.

### Motivation

The full numerical forward model costs **~3.2 s/call** (5 realizations, 10% galaxies, 5% particles). Nautilus requires ~10⁵ likelihood evaluations, making direct sampling infeasible (~4 CPU-years). The emulator forward pass costs **~μs/call**, reducing a full Nautilus run to minutes.

### MPI Strategy

The script uses **mpi4py**, not `multiprocessing.Pool`. This matters because `Pool` forks after JAX is already imported and can deadlock. MPI spawns independent Python interpreters, so each rank imports JAX fresh. Pin each rank to its own CPU set:

```bash
# Local cluster (20 cores per rank, 5 ranks)
mpirun -n 5 --bind-to core --map-by node:PE=20 \
    python run_emulator_grid.py --fit_case STANDARD_NFW --n_samples 100000

# SLURM
srun --ntasks=5 --cpus-per-task=20 \
    python run_emulator_grid.py --fit_case STANDARD_NFW --n_samples 100000
```

### Smoke Test (no MPI)

```bash
python run_emulator_grid.py --n_samples 50 --no_mpi
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fit_case` | `STANDARD_NFW` | Model complexity: `STANDARD_NFW` (7D), `EXTENDED_PROFILE` (9D), `CONFORMITY` (10D) |
| `--n_samples` | 100000 | Total LHS grid points across all ranks |
| `--n_realizations` | 5 | HOD realizations averaged per grid point |
| `--galaxy_fraction` | 0.10 | Galaxy downsampling per realization |
| `--particle_fraction` | 0.05 | Particle downsampling at `HaloOccupation` init (benchmark optimum) |
| `--particle_seed` | 42 | Random seed for particle subsampling |
| `--cache_path` | None | Path to `halo_center_lensing_cache.h5` from `precompute_halo_center_cache.py`. Enables `method="optimized"` (3–9× speedup). **Strongly recommended.** |
| `--output_dir` | `emulator_grid/` | Directory for checkpoint files and merged output |
| `--checkpoint_every` | 200 | Save a `.npz` checkpoint every N points (fault-tolerant) |
| `--base_seed` | 42 | Base seed for LHS generation and HOD realizations |
| `--no_mpi` | False | Run single-process (for local testing) |

### How It Works

0. **(Once, before the grid run)** Run `precompute_halo_center_cache.py` to generate `halo_center_lensing_cache.h5`. This is the same cache used by `NumericalDeltaSigmaFitter` and the benchmark scripts. Pass it via `--cache_path` for a 3–9× speedup per grid point. Without it the script falls back to `method="standard"` and prints a warning.
1. **Rank 0** calls `generate_hod_parameter_grid()` to produce the full LHS grid and broadcasts it to all ranks via `comm.bcast()`.
2. **Each rank** loads the HDF5 cache independently (no MPI I/O contention), then evaluates its slice of the grid by calling `run_hod_grid()`. Checkpoints are saved every `checkpoint_every` points — if the job is interrupted, re-running the same command resumes from the last checkpoint.
3. After all ranks finish, **rank 0** calls `merge_grid_chunks()` to concatenate the per-rank `.npz` files, discard failed rows (all-NaN DeltaSigma), and write `grid_merged.npz`.

### Expected Wall-Clock (5 ranks × 20 cores, Flamingo L1000N1800)

| Fit case | Params | Total pts | Per rank | ~Wall-clock |
|----------|--------|-----------|----------|-------------|
| STANDARD_NFW | 7 | 100k | 20k | ~18h |
| EXTENDED_PROFILE | 9 | 100k | 20k | ~18h |
| CONFORMITY | 10 | 100k | 20k | ~18h |

### After the Grid: Train the Emulator

```python
import numpy as np
from HOD_NRV.utilsf.emulator_nn import train_emulator

d = np.load("emulator_grid/grid_merged.npz", allow_pickle=True)
train_emulator(
    d["params_array"], d["dsigma_array"], d["rp_centers"],
    save_path="emulator_STANDARD_NFW.pt",
)
```

### After Training: Sample with `EmulatorFitter`

```python
from HOD_NRV.utilsf.numerical_sampler import EmulatorFitter, FitCase

fitter = EmulatorFitter(
    emulator_path="emulator_STANDARD_NFW.pt",
    ds_obs=ds_obs, cov_inv=cov_inv, rp_obs=rp_centers,
    fit_case=FitCase.STANDARD_NFW,
    target_ngal=2e-4,
    hod_model=halo,
)
points, weights, log_l, log_z = fitter.run(n_live=500, n_eff=5000)
```

### Output Files

| File | Description |
|------|-------------|
| `emulator_grid/param_grid_full.parquet` | Full LHS grid (all ranks, reference) |
| `emulator_grid/grid_rank{i}.npz` | Per-rank checkpoint (params + DeltaSigma) |
| `emulator_grid/grid_merged.npz` | Merged grid after all ranks finish |

## Power Spectrum Validation

### `test_pk.py`

End-to-end validation of `HOD_NRV.utilsf.measure_pk.PowerSpectrumEstimator` on the matter field. Loads the Flamingo particle catalogue (`particle_catalogue_L1000N1800_downsampled.parquet`), measures the matter auto-P(k) with TSC + interlacing + shot-noise subtraction, and overlays the linear and Halofit predictions from `Cosmology` (pyccl). Produces a 2-panel plot (P(k) overlay + ratio to non-linear pyccl) and a `.npz` with the raw measurement.

```bash
# Smoke test
python test_pk.py --Nmesh 256 --subsample 0.05 --output_dir pk_validation_smoke

# Full validation
python test_pk.py --Nmesh 512 --output_dir pk_validation
```

Sanity bar: `P_meas / P_NL ≈ 1` within a few percent for `k ≲ 0.3 h/Mpc`; departure on small scales is dominated by the shot-noise floor (printed by the script as `Lbox³ / N_part`) and aliasing above `(2/3) k_Nyq`.

## Files

- `benchmark_results.json` — Full benchmark data
- `cross_check_results.txt` — Analytical vs numerical comparison
- `cross_check_numerical_results.txt` — Numerical regression test results
- `particle_downsampling_comparison.png` — Particle convergence plot
- `galaxy_downsampling_comparison.png` — Galaxy convergence plot
- `cross_check_dsigma.png` — Analytical/numerical comparison plot
- `cross_check_numerical.png` — Numerical regression test comparison plot
- `run_emulator_grid.py` — MPI grid evaluator for emulator training
