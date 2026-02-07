"""
Power Spectrum Computation Functions
=====================================

JIT-compiled power spectrum computation functions for the halo model.
All functions are pure (no class state) and operate on JAX arrays.

Contents:
- NFW Fourier transforms
- Standard Pgg/Pgm (no beta^NL)
- beta^NL I^NL correction terms (I^22, I^12, I^21)
- beta^NL-enhanced Pgg/Pgm
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import sici

from HOD_NRV.HOD_analytical.hod_analytical import gl_integrate, GL_W, N_GL

jax.config.update("jax_enable_x64", True)


# ============================================================================
# NFW Fourier Transform
# ============================================================================

@jit
def nfw_fourier_u(k: jnp.ndarray, R_s: jnp.ndarray, c: jnp.ndarray,
                  f_scale: float = 1.0) -> jnp.ndarray:
    """
    NFW Fourier transform. Returns shape (n_k, n_M).
    """
    x = k[:, None] * R_s[None, :]
    norm = 1.0 / (jnp.log(1.0 + c) - c / (1.0 + c))
    si_x, ci_x = sici(x)
    si_cx, ci_cx = sici((1.0 + c) * x)

    term1 = jnp.sin(x) * (si_cx - si_x)
    term2 = jnp.cos(x) * (ci_cx - ci_x)
    term3 = jnp.sin(c[None, :] * x) / ((1.0 + c[None, :]) * x)

    u = norm[None, :] * (term1 + term2 - term3)
    u = jnp.where(x < 1e-8, 1.0, u)
    return u * f_scale


@jit
def nfw_fourier_u_single(k: jnp.ndarray, R_s: float, c: float) -> jnp.ndarray:
    """NFW Fourier transform for single halo. Returns shape (n_k,)."""
    x = k * R_s
    norm = 1.0 / (jnp.log(1.0 + c) - c / (1.0 + c))
    si_x, ci_x = sici(x)
    si_cx, ci_cx = sici((1.0 + c) * x)

    term1 = jnp.sin(x) * (si_cx - si_x)
    term2 = jnp.cos(x) * (ci_cx - ci_x)
    term3 = jnp.sin(c * x) / ((1.0 + c) * x)

    u = norm * (term1 + term2 - term3)
    return jnp.where(x < 1e-8, 1.0, u)


# ============================================================================
# Power Spectrum Functions (Standard - no beta^NL)
# ============================================================================

@jit
def _compute_ngal(N_c, N_s, n_M, log10M_min, log10M_max):
    return gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)


@jit
def _compute_Pgg(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                 log10M_min, log10M_max, f_c, f_s):
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)

    # 1-halo
    integrand_cs = 2.0 * N_c * N_s * u_s * n_M
    I_cs = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_cs)
    integrand_ss = (N_s ** 2) * (u_s ** 2) * n_M
    I_ss = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_ss)
    P_1h = (I_cs + I_ss) / n_gal ** 2

    # 2-halo
    integrand_2h = (N_c * u_c + N_s * u_s) * b_h * n_M
    I_2h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_2h)
    P_2h = Pk_lin * (I_2h / n_gal) ** 2

    return P_1h + P_2h


@jit
def _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                 log10M_min, log10M_max, f_c, f_s):
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, 1.0)

    # 1-halo
    integrand_1h = (N_c + N_s * u_s) * u_m * (M / rho_m) * n_M
    P_1h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_1h) / n_gal

    # 2-halo
    integrand_g = (N_c * u_c + N_s * u_s) * b_h * n_M
    I_g = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_g) / n_gal

    I_m1 = 1.0 - gl_integrate((M / rho_m) * b_h * n_M, log10M_min, log10M_max)
    integrand_m2 = u_m * (M / rho_m) * b_h * n_M
    I_m2 = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_m2)

    return P_1h + Pk_lin * I_g * (I_m1 + I_m2)


# ============================================================================
# beta^NL Correction Functions
# ============================================================================

@jit
def _compute_I_NL_22(beta_nl_gl, H1, H2, b_h, n_M, log10M_min, log10M_max):
    """
    I^22 - Double integral over [M_min, inf] x [M_min, inf].
    H1, H2 are (n_k, n_M) arrays including profiles.
    """
    scale = (0.5 * (log10M_max - log10M_min)) ** 2
    bn_w = b_h * n_M * GL_W
    Hb1 = H1 * bn_w[None, :]
    Hb2 = H2 * bn_w[None, :]
    return scale * jnp.einsum('kij,ki,kj->k', beta_nl_gl, Hb1, Hb2)


@jit
def _compute_I_NL_12(beta_nl_Mmin_row, H2, b_h, n_M, log10M_min, log10M_max):
    """I^12 - First mass at M_min, integrate over second."""
    scale = 0.5 * (log10M_max - log10M_min)
    bn_w = b_h * n_M * GL_W
    integrand = beta_nl_Mmin_row * H2 * bn_w[None, :]
    return scale * jnp.sum(integrand, axis=1)


@jit
def _compute_I_NL_21(beta_nl_Mmin_col, H1, b_h, n_M, log10M_min, log10M_max):
    """I^21 - Second mass at M_min, integrate over first."""
    scale = 0.5 * (log10M_max - log10M_min)
    bn_w = b_h * n_M * GL_W
    integrand = beta_nl_Mmin_col * H1 * bn_w[None, :]
    return scale * jnp.sum(integrand, axis=1)


@jit
def _compute_I_NL_gg(beta_nl_gl, H_g, b_h, n_M, log10M_min, log10M_max):
    """I^NL for galaxy-galaxy (only I^22)."""
    return _compute_I_NL_22(beta_nl_gl, H_g, H_g, b_h, n_M, log10M_min, log10M_max)


@jit
def _compute_I_NL_gm(beta_nl_gl, beta_nl_Mmin_col, H_g, H_m, b_h, n_M,
                      u_m_Mmin, A_Mmin, log10M_min, log10M_max):
    """I^NL for galaxy-matter (I^21 + I^22)."""
    I_22 = _compute_I_NL_22(beta_nl_gl, H_g, H_m, b_h, n_M, log10M_min, log10M_max)
    integral_21 = _compute_I_NL_21(beta_nl_Mmin_col, H_g, b_h, n_M, log10M_min, log10M_max)
    I_21 = A_Mmin * u_m_Mmin * integral_21
    return I_21 + I_22


# ============================================================================
# Power Spectrum with beta^NL
# ============================================================================

@jit
def _compute_Pgg_with_beta_nl(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                               log10M_min, log10M_max, f_c, f_s, beta_nl_gl):
    P_gg = _compute_Pgg(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                        log10M_min, log10M_max, f_c, f_s)

    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal

    I_NL = _compute_I_NL_gg(beta_nl_gl, H_g, b_h, n_M, log10M_min, log10M_max)
    return P_gg + Pk_lin * I_NL


@jit
def _compute_Pgm_with_beta_nl(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                               log10M_min, log10M_max, f_c, f_s,
                               beta_nl_gl, beta_nl_Mmin_col, R_s_Mmin, c_Mmin):
    P_gm = _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                        log10M_min, log10M_max, f_c, f_s)

    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_c = nfw_fourier_u(k, R_s, c, f_c)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, 1.0)
    u_m_Mmin = nfw_fourier_u_single(k, R_s_Mmin, c_Mmin)

    H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal
    W_m = M / rho_m
    H_m = u_m * W_m[None, :]
    A_Mmin = 1.0 - gl_integrate(W_m * b_h * n_M, log10M_min, log10M_max)

    I_NL = _compute_I_NL_gm(beta_nl_gl, beta_nl_Mmin_col, H_g, H_m, b_h, n_M,
                            u_m_Mmin, A_Mmin, log10M_min, log10M_max)
    return P_gm + Pk_lin * I_NL


__all__ = [
    # NFW Fourier transforms
    'nfw_fourier_u', 'nfw_fourier_u_single',
    # Standard power spectrum
    '_compute_ngal', '_compute_Pgg', '_compute_Pgm',
    # beta^NL corrections
    '_compute_I_NL_22', '_compute_I_NL_12', '_compute_I_NL_21',
    '_compute_I_NL_gg', '_compute_I_NL_gm',
    # beta^NL-enhanced power spectrum
    '_compute_Pgg_with_beta_nl', '_compute_Pgm_with_beta_nl',
]
