# example_scripts — Validation Tests

Only the validation tests, the downsampling benchmark and the assembly-bias environment builder are tracked here. The production pipeline (catalogue
building, tabulation, covariance, chains, hydro measurements) lives in
[`../internal/`](../internal/); everything else in this directory is local
scratch and gitignored.

| Script | What it validates |
|--------|-------------------|
| `null_test_analytical.py` | Analytical halo model (JAX) vs pyccl's halo model |
| `numerical_dsigma_example.py` | Builds `baseline_dsigma_cache.npz` for the test below |
| `cross_check_analytical_numerical.py` | Analytical vs numerical ΔΣ for one HOD parameter set |
| `cross_check_tabulated.py` | Tabulated (TabCorr-style) ΔΣ / w_gg vs the full Monte-Carlo calculation |
| `benchmark_dsigma_convergence.py` | Particle / galaxy downsampling and number of realizations needed for converged direct ΔΣ |
| `compute_assembly_bias.py` | Not a test: builds the environment columns (δ, tidal shear, mass-binned ranks) that assembly bias uses |

## `null_test_analytical.py` — analytical vs pyccl

Compares P_gg and P_gm from `HaloModel` with pyccl's built-in halo model, in
natural and h-units. Last full pass: agreement < 0.4%. When
`units_per_h=True`, `HaloModel` expects `k` in h/Mpc.

## `cross_check_analytical_numerical.py` — analytical vs numerical

Same HOD parameters and cosmology through both pipelines: the numerical one
(populated Flamingo L1000N1800 halos + pycorr pair counts) and the analytical
one (halo model + Hankel transform).

Run `numerical_dsigma_example.py` first. It populates the box for 10
realizations and writes `baseline_dsigma_cache.npz` (`--cache`), which the
cross-check loads (`--cache`). The cache stores the HOD type and all its
parameters, so the analytical side evaluates the occupation that was
populated. Halos with `c = NaN` (promoted halos of a distinct-halo catalogue)
get Duffy08 c(M200m), the analytical model's default.

Before 2026-09-25 the two scripts used DIFFERENT HODs (numerical ELG_mHMQ
Mmin 12.7, alpha 1.10; analytical ELG_GHOD Mmin 13.0, alpha 0.80 — only
Ac/As were shared), so the old 25–30% gap at 1–10 Mpc/h, attributed to
β^NL, was never a like-for-like comparison. Caches written before then are
rejected.

## `cross_check_tabulated.py` — tabulated vs full calculation

Checks `TabulatedDeltaSigma` (occupation-weighted sums over a precomputed
per-halo ξ_gm tabulation, plus an analytic satellite-offset convolution)
against the Monte-Carlo pipeline (`populate_haloes` + pycorr) on a random halo
subsample. The per-galaxy ΔΣ expectation doesn't change under halo
subsampling, so a small subsample gives an unbiased comparison.

```bash
python cross_check_tabulated.py                        # ~114k halos
python cross_check_tabulated.py --assembly_bias        # with fs_norm assembly bias
python cross_check_tabulated.py --halo_fraction 0.02   # quick smoke test
python cross_check_tabulated.py --direct ...           # tabulation vs direct pair count
python cross_check_tabulated.py --jax ...              # jit/vmap likelihood parity
```

The tabulated cache is saved and reused between runs. Delete it after changing
`--halo_fraction`, `--particle_fraction` or the bin counts.
