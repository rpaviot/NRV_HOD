"""Equivalence: JAX FFTLog ΔΣ  vs  reference FAST-PT + scipy Pk_to_DeltaSigma_direct.

Run with the NRV venv:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python HOD_NRV/utilsf/tests/test_hankel_jax.py
"""
from __future__ import annotations

import time
import numpy as np
import jax

from HOD_NRV.utilsf.hankel_transforms import Pk_to_DeltaSigma_direct
from HOD_NRV.utilsf.hankel_jax import build_direct_deltasigma


def _mock_pgm(k, knl=0.3, amp=2.0e4):
    """A smooth, P_gm-like spectrum: small-k power law with a soft cut."""
    return amp * (k / 0.1) ** (-1.5) / (1.0 + (k / knl) ** 2)


def _run(method="cubic2"):
    # k-grid matches fit_csmf.py: np.geomspace(1e-5, 300, 1024)
    k = np.geomspace(1e-5, 300.0, 1024)
    rp_bins = np.geomspace(0.1, 60.0, 16)
    rho_m = 2.775e11 * 0.3096  # Msun/h / (Mpc/h)^3, ~Planck18 Om

    builder = build_direct_deltasigma(k, rp_bins, interp_method=method)

    n_curves = 8
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for i in range(n_curves):
        Pk = _mock_pgm(k, knl=0.2 + 0.3 * rng.random(), amp=(1 + rng.random()) * 1e4)
        _, ds_ref = Pk_to_DeltaSigma_direct(k, Pk, rho_m, None, rp_bins=rp_bins)
        ds_jax = np.asarray(builder(Pk, rho_m))
        rel = np.abs(ds_jax - ds_ref) / np.abs(ds_ref)
        max_rel = max(max_rel, rel.max())
        if i == 0:
            print(f"  bin-averaged ΔΣ (curve 0), method={method}:")
            for j in range(len(ds_ref)):
                print(f"    bin{j:2d}  ref={ds_ref[j]: .6e}  jax={ds_jax[j]: .6e}"
                      f"  rel={rel[j]:.2e}")
    print(f"  -> max relative error over {n_curves} curves: {max_rel:.3e}")
    return max_rel, k, rp_bins, rho_m, builder


def _bench(k, rp_bins, rho_m, builder):
    Pk = _mock_pgm(k)
    # batched throughput via vmap over a stack of P_gm curves
    batch = np.repeat(Pk[None, :], 256, axis=0)
    vfun = jax.jit(jax.vmap(builder.transform, in_axes=(0, None)))
    vfun(batch[:2], rho_m).block_until_ready()  # warm compile
    t0 = time.time()
    out = vfun(batch, rho_m)
    out.block_until_ready()
    dt = time.time() - t0
    print(f"  vmap(256) one fused call: {dt*1e3:.1f} ms  "
          f"=> {256/dt:,.0f} ΔΣ-evals/s (transform only)")


if __name__ == "__main__":
    print("== JAX FFTLog ΔΣ vs FAST-PT/scipy reference ==")
    best = None
    for m in ("cubic2", "cubic", "akima"):
        try:
            mr, k, rp_bins, rho_m, builder = _run(m)
        except Exception as e:
            print(f"  method={m}: FAILED ({e})")
            continue
        if best is None or mr < best[0]:
            best = (mr, m, k, rp_bins, rho_m, builder)
    assert best is not None, "no interp method ran"
    mr, m, k, rp_bins, rho_m, builder = best
    print(f"\nBest interp method: {m}  (max rel err {mr:.3e})")
    print("\n== throughput ==")
    _bench(k, rp_bins, rho_m, builder)
    tol = 1e-3
    print(f"\n{'PASS' if mr < tol else 'CHECK'}: max rel err {mr:.3e} "
          f"({'<' if mr < tol else '>='} {tol:g})")
