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

from .hod_analytical import gl_integrate, GL_W, N_GL

jax.config.update("jax_enable_x64", True)


# ============================================================================
# NFW Fourier Transform
# ============================================================================

@jit
def nfw_fourier_u(k: jnp.ndarray, R_s: jnp.ndarray, c: jnp.ndarray,
                  f_scale: float = 1.0) -> jnp.ndarray:
    """
    Normalised NFW Fourier transform u(k|M). Returns shape (n_k, n_M).

    ``f_scale`` renormalises the concentration-mass relation (Dvornik+23 eq. 15):
    c -> f_scale * c, at fixed virial mass so the scale radius follows
    R_s -> R_s / f_scale (R_vir = R_s * c held constant). f_scale = 1 recovers the
    input (Duffy+08) profile. This is a *shape* rescaling that keeps the profile
    normalised to unity at k -> 0 for any f_scale -- unlike a flat amplitude
    factor u * f_scale, which would violate u(k=0|M) = 1 (mass/number
    conservation).
    """
    c = c * f_scale
    R_s = R_s / f_scale
    x = k[:, None] * R_s[None, :]
    norm = 1.0 / (jnp.log(1.0 + c) - c / (1.0 + c))
    si_x, ci_x = sici(x)
    si_cx, ci_cx = sici((1.0 + c) * x)

    term1 = jnp.sin(x) * (si_cx - si_x)
    term2 = jnp.cos(x) * (ci_cx - ci_x)
    term3 = jnp.sin(c[None, :] * x) / ((1.0 + c[None, :]) * x)

    u = norm[None, :] * (term1 + term2 - term3)
    u = jnp.where(x < 1e-8, 1.0, u)
    return u


@jit
def nfw_fourier_u_single(k: jnp.ndarray, R_s: float, c: float,
                         f_scale: float = 1.0) -> jnp.ndarray:
    """NFW Fourier transform for single halo. Returns shape (n_k,).

    ``f_scale`` renormalises the concentration exactly as in :func:`nfw_fourier_u`
    (c -> f_scale * c, R_s -> R_s / f_scale).
    """
    c = c * f_scale
    R_s = R_s / f_scale
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
                 log10M_min, log10M_max, f_h, f_s, A_b=1.0):
    # f_h (halo/matter concentration norm) does not enter P_gg -- there is no
    # matter profile here. Centrals are points (H_c = N_c/n_c, eq. 9), so only
    # the satellite profile u_s carries a concentration rescaling (f_s).
    # A_b rescales the galaxy-side halo bias, A_b * b_h(M) (Liu+25 eq. 7): it
    # absorbs assembly bias in the 2-halo amplitude, so P_2h goes as A_b^2.
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_s = nfw_fourier_u(k, R_s, c, f_s)

    # 1-halo
    integrand_cs = 2.0 * N_c * N_s * u_s * n_M
    I_cs = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_cs)
    integrand_ss = (N_s ** 2) * (u_s ** 2) * n_M
    I_ss = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_ss)
    P_1h = (I_cs + I_ss) / n_gal ** 2

    # 2-halo (central point: u_c = 1)
    integrand_2h = (N_c + N_s * u_s) * b_h * n_M
    I_2h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_2h)
    P_2h = Pk_lin * (A_b * I_2h / n_gal) ** 2

    return P_1h + P_2h


@jit
def _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                 log10M_min, log10M_max, f_h, f_s, resp_m=None, A_b=1.0):
    # Centrals are points (u_c = 1, eq. 9); satellites carry u_s(f_s); the matter
    # profile u_m carries the halo concentration norm f_h (eq. 8, 15) and, if
    # given, the baryonic response resp_m(k, M) = u_DMB / u_DMO (n_k, n_M; e.g.
    # Schneider+25, baryons_s25.py). Both enter the 1-halo and 2-halo terms.
    # A_b rescales the galaxy-side bias only (I_g); the matter side I_m1 + I_m2
    # keeps its consistency relation.
    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, f_h)
    if resp_m is not None:
        u_m = u_m * resp_m

    # 1-halo
    integrand_1h = (N_c + N_s * u_s) * u_m * (M / rho_m) * n_M
    P_1h = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_1h) / n_gal

    # 2-halo (central point: u_c = 1)
    integrand_g = (N_c + N_s * u_s) * b_h * n_M
    I_g = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_g) / n_gal

    I_m1 = 1.0 - gl_integrate((M / rho_m) * b_h * n_M, log10M_min, log10M_max)
    integrand_m2 = u_m * (M / rho_m) * b_h * n_M
    I_m2 = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand_m2)

    return P_1h + Pk_lin * A_b * I_g * (I_m1 + I_m2)


@jit
def _compute_P_strip(N_c, N_s, n_M, R_s, c_t, m_frac, k, M, rho_m,
                     log10M_min, log10M_max):
    """Satellite-strip 1-halo term: each satellite galaxy x its own tidally
    stripped subhalo (Mandelbaum+05, Zacharegkas+22 eq. 19).

    The subhalo is an NFW hard-truncated at c_t * R_s (u_t, normalised, from the
    same closed form as an NFW cut at its virial radius), holding m_frac * M of
    its pre-infall mass M. Pre-infall masses follow the centrals of the same
    sample, p(M) = N_c n(M) / n_c, so the term is

        P_strip(k) = (n_s / n_gal) * int dM p(M) m_frac M u_t(k|M) / rho_m.

    The retained subhalo mass is not removed from the host profile; with
    m_frac ~ 0.5 and a few satellites per host that double counts a few percent
    of the host mass in the satellite 1-halo term.
    """
    n_c = gl_integrate(N_c * n_M, log10M_min, log10M_max)
    n_s = gl_integrate(N_s * n_M, log10M_min, log10M_max)
    u_t = nfw_fourier_u(k, R_s, c_t)
    integrand = N_c * n_M * (m_frac * M / rho_m) * u_t
    I_t = vmap(lambda x: gl_integrate(x, log10M_min, log10M_max))(integrand) / n_c
    return n_s / (n_c + n_s) * I_t


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
                               log10M_min, log10M_max, f_h, f_s, beta_nl_gl,
                               A_b=1.0):
    P_gg = _compute_Pgg(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k,
                        log10M_min, log10M_max, f_h, f_s, A_b)

    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    H_g = (N_c[None, :] + N_s[None, :] * u_s) / n_gal  # central point: u_c = 1

    I_NL = _compute_I_NL_gg(beta_nl_gl, H_g, b_h, n_M, log10M_min, log10M_max)
    return P_gg + Pk_lin * A_b ** 2 * I_NL  # two galaxy-side biases


@jit
def _compute_Pgm_with_beta_nl(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                               log10M_min, log10M_max, f_h, f_s,
                               beta_nl_gl, beta_nl_Mmin_col, R_s_Mmin, c_Mmin,
                               resp_m=None, resp_m_Mmin=None, A_b=1.0):
    """resp_m (n_k, n_M) / resp_m_Mmin (n_k,): baryonic response of the matter
    profile at the GL nodes and at M_min (see _compute_Pgm); both or neither.
    A_b: galaxy-side bias amplitude (see _compute_Pgg)."""
    P_gm = _compute_Pgm(N_c, N_s, n_M, b_h, R_s, c, Pk_lin, k, M, rho_m,
                        log10M_min, log10M_max, f_h, f_s, resp_m, A_b)

    n_gal = gl_integrate((N_c + N_s) * n_M, log10M_min, log10M_max)
    u_s = nfw_fourier_u(k, R_s, c, f_s)
    u_m = nfw_fourier_u(k, R_s, c, f_h)
    u_m_Mmin = nfw_fourier_u_single(k, R_s_Mmin, c_Mmin, f_h)
    if resp_m is not None:
        u_m = u_m * resp_m
        u_m_Mmin = u_m_Mmin * resp_m_Mmin

    H_g = (N_c[None, :] + N_s[None, :] * u_s) / n_gal  # central point: u_c = 1
    W_m = M / rho_m
    H_m = u_m * W_m[None, :]
    A_Mmin = 1.0 - gl_integrate(W_m * b_h * n_M, log10M_min, log10M_max)

    I_NL = _compute_I_NL_gm(beta_nl_gl, beta_nl_Mmin_col, H_g, H_m, b_h, n_M,
                            u_m_Mmin, A_Mmin, log10M_min, log10M_max)
    return P_gm + Pk_lin * A_b * I_NL  # one galaxy-side bias


__all__ = [
    # NFW Fourier transforms
    'nfw_fourier_u', 'nfw_fourier_u_single',
    # Standard power spectrum
    '_compute_ngal', '_compute_Pgg', '_compute_Pgm', '_compute_P_strip',
    # beta^NL corrections
    '_compute_I_NL_22', '_compute_I_NL_12', '_compute_I_NL_21',
    '_compute_I_NL_gg', '_compute_I_NL_gm',
    # beta^NL-enhanced power spectrum
    '_compute_Pgg_with_beta_nl', '_compute_Pgm_with_beta_nl',
]
