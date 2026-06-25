"""JAX-traceable FFTLog Hankel transform: P_gm(k) -> ΔΣ(r_p), bin-averaged.

A drop-in, vmap-able replacement for ``hankel_transforms.Pk_to_DeltaSigma_direct``
(FAST-PT ``HT.k_to_r`` + ``scipy`` cubic ``interp1d`` + 200-pt GL ``binavg_2D``),
built specifically so the CSMF ΔΣ forward pass can be ``jax.vmap``'d over a batch
of HOD parameter vectors (nautilus ``vectorized=True``).

Key insight: FAST-PT's FFTLog is called with ``q=0``. In that regime the FFTLog
kernel ``u_m = 2**x * Γ((μ+1+x)/2)/Γ((μ+1−x)/2)`` (x = i·2π·m/L), the output radii
``r``, the index permutation, and the per-bin Gauss-Legendre evaluation radii are
ALL independent of the data ``P_gm``. They depend only on the (fixed) k-grid, the
transform exponents (alpha_k, beta_r, mu, pf) and ``rp_bins``. So we precompute
every one of them ONCE in numpy -- bit-matching FAST-PT via ``scipy.special.gamma``
and ``scipy.special.roots_legendre`` -- and the per-call work reduces to

    f_k  = kpow * Pk                 # kpow = k**alpha_k        (const)
    c_m  = jnp.fft.rfft(f_k)
    A    = jnp.fft.irfft(c_m * u_m, n=N)[perm]    # u_m, perm   (const)
    ds_r = pf * A * rbeta * rho_m / 1e12          # rbeta       (const)
    ds_b = binavg( interpax.cubic(r_sorted, ds_r)(gl_radii) )   # all const grids

all of which is pure jnp (jnp.fft.rfft/irfft + interpax cubic) and therefore
vmap-able. Reference parity is checked in tests/test_hankel_jax.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import gamma as _cgamma, roots_legendre

import jax
# This transform is designed to reproduce FAST-PT's float64 FFTLog to machine
# precision; in float32 the rfft/irfft and the cubic spline solve lose ~6-7
# digits. Enable x64 at import so correctness does not depend on some other
# module (e.g. halo_model) happening to switch it on first.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import interpax


# ---------------------------------------------------------------------------
# FAST-PT FFTLog kernel, replicated in numpy (precompute only; never per-call)
# ---------------------------------------------------------------------------
_CUT = 200  # FAST-PT's switch to the Stirling expansion for |Im(x)| > cut


def _g_m_vals(mu: float, q: np.ndarray) -> np.ndarray:
    """Exact copy of fastpt.misc.HT.g_m_vals (Γ-ratio with Stirling fallback)."""
    imag_q = np.imag(q)
    g_m = np.zeros(q.size, dtype=complex)

    asym_q = q[np.absolute(imag_q) > _CUT]
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.

    good = (np.absolute(imag_q) <= _CUT) & (q != mu + 1 + 0.0j)
    alpha_plus = (mu + 1 + q[good]) / 2.
    alpha_minus = (mu + 1 - q[good]) / 2.
    g_m[good] = _cgamma(alpha_plus) / _cgamma(alpha_minus)

    g_m[np.absolute(imag_q) > _CUT] = np.exp(
        (asym_plus - 0.5) * np.log(asym_plus)
        - (asym_minus - 0.5) * np.log(asym_minus) - asym_q
        + 1. / 12 * (1. / asym_plus - 1. / asym_minus)
        + 1. / 360. * (1. / asym_minus ** 3 - 1. / asym_plus ** 3)
        + 1. / 1260 * (1. / asym_plus ** 5 - 1. / asym_minus ** 5)
    )
    g_m[np.where(q == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    return g_m


def _fftlog_constants(k: np.ndarray, mu: float):
    """Precompute the q=0 FFTLog kernel u_m, output radii r, and reorder perm.

    Mirrors fastpt.misc.HT.fft_log(k, f_k, q=0, mu) exactly, but factors out
    everything that does not depend on f_k.
    """
    k = np.asarray(k, dtype=float)
    N = k.size
    if N % 2 != 0:
        raise ValueError("FFTLog grid length N must be even (matches FAST-PT rfft).")
    # even log-spacing check (same tolerance as FAST-PT)
    dd = np.diff(np.diff(np.log(k)))
    if np.sum(dd) >= 1e-10:
        raise ValueError("k must be evenly spaced in log.")

    L = np.log(k.max()) - np.log(k.min())
    delta_L = L / float(N - 1)
    log_k0 = np.log(k[N // 2])
    k0 = np.exp(log_k0)

    # rfft frequency index m = [0, 1, ..., N/2]
    m = np.fft.rfftfreq(N, d=1.) * float(N)

    # u_m_vals(m, mu, q=0, kr=1, L): u_m = 2**x * g_m_vals(mu, x), x = i 2π m / L
    omega = 1j * 2 * np.pi * m / L
    x = omega  # q = 0
    u_m = (2.0 ** x) * _g_m_vals(mu, x)
    u_m[m.size - 1] = np.real(u_m[m.size - 1])  # Nyquist made real (FAST-PT)

    # output radii r and the gather/reverse permutation (kr=1, r0=1/k0)
    log_r0 = -log_k0
    m_r = np.arange(-N // 2, N // 2)
    m_shift = np.fft.fftshift(m_r)
    s = delta_L * (-m_r) + log_r0
    idx = m_shift % N                  # FAST-PT's A_m[id] with python negative wrap
    r = np.exp(s[idx])                 # 10**(s/log10) == exp(s)
    perm = idx[::-1].copy()            # A = A_m[id][::-1]  -> A_m[perm]
    r = r[::-1].copy()                 # r = r[::-1]  (now ascending)
    return N, u_m, perm, r


# ---------------------------------------------------------------------------
# Builder: returns a jitted, vmap-able  P_gm -> ΔΣ_binned  callable
# ---------------------------------------------------------------------------
@dataclass
class _JaxDirectDS:
    """Carries the precomputed constants + the compiled transform."""
    transform: Callable          # (Pk_gm[...,Nk], rho_m) -> ΔΣ[..., nbin]
    r: np.ndarray                # ascending transform radii (diagnostic)
    n_gl: int

    def __call__(self, Pk_gm, rho_m):
        return self.transform(Pk_gm, rho_m)


def build_direct_deltasigma(
    k: np.ndarray,
    rp_bins: np.ndarray,
    *,
    alpha_k: float = 1.0,
    beta_r: float = -1.0,
    mu: float = 2.0,
    pf: float = 1.0 / (2 * np.pi),
    n_gl: int = 200,
    interp_method: str = "cubic",
) -> _JaxDirectDS:
    """Build a JAX ΔΣ(r_p) transform matching ``Pk_to_DeltaSigma_direct``.

    Parameters mirror the FAST-PT call used in the library:
    ``k_to_r(k, Pk, alpha_k=1, beta_r=-1, mu=2, pf=1/2π)`` followed by
    ``* rho_m / 1e12``, cubic interpolation and 200-pt GL bin-averaging
    over ``rp_bins``.

    ``interp_method`` selects the interpax 1-D scheme used in place of scipy's
    not-a-knot cubic. Use a LOCAL scheme (``cubic`` C1, ``akima``, ``monotonic``)
    -- these never build a global system, so they are robust under jit/vmap, same
    as the β^NL interpolator (interpax ``method='linear'``). Avoid ``cubic2``: it
    is the C2 global spline and solves a tridiagonal system via ``lineax``, which
    raises on any non-finite input (the FFTLog ds_r carries tiny/extreme values at
    radii far outside the data range). On real bin-averaged ΔΣ the local ``cubic``
    reproduces the scipy-cubic reference to ~3e-5 -- ~1000x below the data errors.
    """
    k = np.asarray(k, dtype=float)
    rp_bins = np.asarray(rp_bins, dtype=float)

    N, u_m_np, perm_np, r_np = _fftlog_constants(k, mu)
    kpow_np = k ** alpha_k
    rbeta_np = r_np ** beta_r

    # 200-pt Gauss-Legendre evaluation radii inside each rp bin (data-independent)
    x_gl, w_gl = roots_legendre(n_gl)
    a, b = rp_bins[:-1], rp_bins[1:]
    gl_radii_np = 0.5 * (np.outer(x_gl, (b - a)) + (a + b))     # (n_gl, nbin)
    diff_sq_np = b ** 2 - a ** 2                                # (nbin,)
    half_width_np = 0.5 * (b - a)                               # (nbin,)

    # promote constants to device arrays
    u_m = jnp.asarray(u_m_np)
    perm = jnp.asarray(perm_np)
    kpow = jnp.asarray(kpow_np)
    rbeta = jnp.asarray(rbeta_np)
    r_sorted = jnp.asarray(r_np)
    gl_radii_flat = jnp.asarray(gl_radii_np.reshape(-1))        # (n_gl*nbin,)
    gl_radii_2d = jnp.asarray(gl_radii_np)                      # (n_gl, nbin)
    w_gl_j = jnp.asarray(w_gl)
    diff_sq = jnp.asarray(diff_sq_np)
    half_width = jnp.asarray(half_width_np)
    nbin = a.size

    def _direct(Pk_gm, rho_m):
        # FFTLog: f_k = k**alpha_k * Pk ; A = irfft(rfft(f_k) * u_m)[perm]
        f_k = kpow * Pk_gm
        c_m = jnp.fft.rfft(f_k)
        A = jnp.fft.irfft(c_m * u_m, n=N)[perm]
        ds_r = pf * A * rbeta * (rho_m / 1e12)                  # ΔΣ on r_sorted grid
        # cubic interpolation onto the fixed GL radii, then area-weighted average
        ds_gl = interpax.interp1d(gl_radii_flat, r_sorted, ds_r,
                                  method=interp_method, extrap=True)
        ds_gl = ds_gl.reshape(n_gl, nbin)
        # binavg_2D: <ΔΣ> = (2/Δr²) ∫ ΔΣ(r) r dr  -- note the radial weight r
        integrand = ds_gl * gl_radii_2d
        integral = half_width * jnp.einsum("i,ib->b", w_gl_j, integrand)
        return 2.0 * integral / diff_sq                        # (nbin,)

    transform = jax.jit(_direct)
    return _JaxDirectDS(transform=transform, r=r_np, n_gl=n_gl)
