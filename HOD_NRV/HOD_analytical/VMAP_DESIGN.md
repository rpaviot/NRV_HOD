# Vectorizing the CSMF likelihood for nautilus (`vectorized=True`)

Goal: replace the serial, single-point `CSMFFitter.log_likelihood` (~3.3 calls/s,
JAX run forks badly under nautilus `pool`) with a batched likelihood so nautilus
can be run with `vectorized=True` — one fused call per proposal batch, all cores
saturated, no `multiprocessing`/fork.

## Forward-pass audit (CSMF ΔΣ path)

Per likelihood call: `log_likelihood` -> `_compute_model_observables` ->
`halo_model.set_hod_params` (stateful) + `halo_model.DeltaSigma`.

| Stage | File | JAX-ready for vmap? |
|---|---|---|
| Cosmology / halo structure (`_Pk_lin_jax`, `_b_h_jax`, `_R_s_jax`, `_c_jax`, `_n_M_jax`) | halo_model | **Precomputed once at init** (fixed cosmology) — constants, fine |
| β^NL emulator templates (`_beta_nl_*_cache`) | halo_model | **Cached once at init** — constants, fine |
| HOD occupation `_get_occupation` | hod_analytical | jnp, BUT reads HOD params from mutable `self.hod.params` (stateful) |
| `_compute_Pgm[_with_beta_nl]` | power_spectrum | jnp ✓ (takes cached arrays + `f_c`,`f_s` as explicit args) |
| Per-z assembly | halo_model `Pgm`/`DeltaSigma` | Python `for iz` loop + `np.asarray` round-trip |
| **P_gm → ΔΣ Hankel transform** (`Pk_to_DeltaSigma_direct`) | **utilsf/hankel_transforms.py** | **NO — pure numpy/scipy: FAST-PT `HT.k_to_r` + `scipy.interpolate.interp1d` (cubic). Not traceable.** |
| stellar term | halo_model | trivial numpy, easily jnp |
| χ² (`np.linalg.inv(cov)` per call) | sampler | numpy; cov is fixed data → invert ONCE at setup |

## The blocker

The expensive non-JAX pieces (dark_emulator, CCL) are **not** in the hot loop —
they are frozen at init. The HOD occupation and the P(k) kernels **are** jnp. So
the *only* thing standing between us and a real `jax.vmap(theta -> ΔΣ)` is the
**final Hankel stage**: `Pk_to_DeltaSigma_direct` uses FAST-PT's FFTLog
(`k_to_r`) and a `scipy` cubic `interp1d`, neither of which JAX can trace.

## Two routes (decision needed before implementing)

### Route A — true vmap (reimplement the Hankel stage in JAX)
- Rewrite `Pk_to_DeltaSigma_direct` as a jnp FFTLog + `interpax` cubic interp
  (interpax is already a dependency), matching FAST-PT bin-for-bin.
- De-state the occupation: thread HOD params (incl. `f_c`,`f_s`) as explicit
  traced args instead of `self.hod.params`/`self.f_c`.
- Replace the per-z Python loops with `vmap`/`scan` over the z axis.
- Precompute per-bin `cov_inv` as jnp constants; build `log_likelihood_batched`.
- `jax.vmap` over the parameter batch; pass `vectorized=True` to nautilus.
- **Pro:** genuine batch speedup (XLA fuses the whole live-point proposal).
  **Con:** delicate numerical reimplementation of FFTLog in a *shared* package;
  must validate ΔΣ equivalence to ~1e-6 bin-for-bin before trusting any run.

### Route B — spawn-based JAX process pool (no vmap, no numerical rewrite)
- The reason joblib/`pool` failed is fork + a live XLA runtime in the parent.
- A `multiprocessing` **spawn** pool, each worker importing JAX fresh and pinning
  `XLA threads = 1` (`OMP/MKL/XLA_FLAGS`), avoids the fork entirely.
- nautilus already accepts `pool=`; pass a spawn pool of N workers.
- **Pro:** small, no FFTLog rewrite, parallelism ~N×. **Con:** N×memory
  (each worker holds its own halo model + caches), and IPC overhead per point.

## Safe-either-way pieces (can land now)
- Precompute per-bin `cov_inv` at setup → drop the per-call `np.linalg.inv`
  (helps minuit/de/nautilus/pool alike; tiny but free).
- A scalar-vs-batch (or scalar-vs-refactor) **equivalence test** harness, so any
  optimisation is guarded against numerical drift.

## Status
- [x] Forward-pass audit; Hankel stage pinned as the vmap blocker.
- [ ] Decision: Route A (JAX FFTLog) vs Route B (spawn pool).
- [ ] Implement chosen route + equivalence test.
