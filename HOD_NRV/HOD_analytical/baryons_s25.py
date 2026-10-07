"""
Schneider+25 Baryonic Halo Profiles (JAX)
==========================================

JAX port of the Schneider et al. (2025, arXiv:2507.07892) baryonification
profiles, following the BaryonForge implementation
(``BaryonForge/Profiles/Schneider25.py``, upstream fafc0f3) -- the code that
baryonified the Flamingo DMO box -- so the analytical halo model and the
baryonified simulation share one model.

Where BaryonForge departs from the paper, this module follows BaryonForge:

* truncation radius r_t = eps R200c with eps = max(eps0 - eps1 nu, 1e-3)
  (BaryonForge: S25 Eq. 2.8 has the wrong sign);
* relaxation amplitudes Q_i(z) = q_i + nu_qi z (linear, not (1+z)^nu);
* inner gas r^-3 exp(-r/R200c) with a tanh core at r_min_iga.

The output is the matter profile in Fourier space, normalised by the total
(r -> infinity) mass of the truncated NFW, for both the dark-matter-only
(DMO, truncated NFW) and the dark-matter-baryon (DMB) one-halo profiles. The
two-halo term is NOT included: the halo model supplies its own.

Units are BaryonForge's: comoving Mpc and Msun (no h). Halo masses are M200c.
Callers in h-units convert M -> M/h, R -> R/h, k -> k*h; u(k) is
dimensionless and depends only on k*R, so it is unchanged by the conversion.

Profiles are evaluated on a dimensionless log grid x = r/R200c shared by all
halos. The Fourier transform u(k) = int dM/dlnx j0(k R x) dlnx / M_tot is done
with FFTLog (Hamilton 2000) on that grid; its kernel depends only on the grid,
so it is precomputed once and the whole pipeline stays jit-able in the model
parameters.
"""

from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from scipy.optimize import brentq
from scipy.special import loggamma

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Parameters
# ============================================================================

# Model parameters read by this module (a subset of BaryonForge's
# Schneider25.model_params: no concentration dependence, no pressure terms).
S25_PARAM_NAMES = (
    'epsilon0', 'epsilon1',
    'q0', 'q1', 'q2', 'nu_q0', 'nu_q1', 'nu_q2', 'nstep',
    'theta_c', 'M_c', 'gamma', 'delta', 'alpha', 'mu_beta',
    'mu_theta_c', 'mu_gamma', 'mu_delta', 'mu_alpha',
    'M_theta_c', 'M_gamma', 'M_delta', 'M_alpha',
    'nu_theta_c', 'nu_M_c', 'nu_gamma', 'nu_delta', 'nu_alpha',
    'c_iga', 'nu_c_iga', 'r_min_iga',
    'Nstar', 'Mstar', 'eta', 'eta_delta', 'tau', 'tau_delta', 'epsilon_cga',
)

# BaryonForge's defaults for parameters left unset: mass/redshift slopes 0,
# mass pivots 1e14 Msun.
_DEFAULTS = {n: 0.0 for n in S25_PARAM_NAMES if n.startswith(('mu_', 'nu_'))}
_DEFAULTS.update({n: 1e14 for n in ('M_theta_c', 'M_gamma', 'M_delta', 'M_alpha')})


def s25_flamingo_m8_z1(h: float) -> Dict[str, float]:
    """FLAMINGO m8 at z=1 (S25 Fig. 10 and the reduced-model relations).

    The parameter set of the DMO_B2 baryonified box. M_c and Mstar are quoted
    in Msun/h by S25 and converted here to BaryonForge's Msun.
    """
    return {
        'epsilon0': 4.0, 'epsilon1': 0.5,
        'q0': 0.075, 'q1': 0.25, 'q2': 0.7,
        'nu_q0': 0.0, 'nu_q1': 0.25, 'nu_q2': 0.0, 'nstep': 1.5,
        'theta_c': 0.3, 'nu_theta_c': 0.5,
        'M_c': 10**13.2 / h, 'mu_beta': 0.7,
        'alpha': 1.0, 'gamma': 1.5, 'delta': 7.0,
        'c_iga': 0.1, 'nu_c_iga': 1.5, 'r_min_iga': 1e-3,
        'Nstar': 0.03, 'Mstar': 2.5e11 / h,
        'eta': 0.05, 'eta_delta': 0.25, 'tau': -1.376, 'tau_delta': 0.0,
        'epsilon_cga': 0.03,
    }


def complete_params(params: Dict[str, float]) -> Dict[str, float]:
    """Fill BaryonForge defaults and reject unknown names.

    BaryonForge silently drops misspelt keys (a 'mu' instead of 'mu_beta' left
    the May 2026 box with beta = 1.5 at every mass); here they raise.
    """
    unknown = set(params) - set(S25_PARAM_NAMES)
    if unknown:
        raise KeyError(f"not Schneider25 parameters: {sorted(unknown)}")
    full = dict(_DEFAULTS)
    full.update(params)
    missing = set(S25_PARAM_NAMES) - set(full)
    if missing:
        raise KeyError(f"missing Schneider25 parameters: {sorted(missing)}")
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in full.items()}


# ============================================================================
# Radial grid and FFTLog kernel
# ============================================================================

class S25Grid:
    """Log grid in x = r/R200c plus the FFTLog kernel for the j0 transform.

    Defaults: 4096 points on [1e-7, 1e7] (dlnx = 0.0079) with FFTLog bias
    q = 0.3. The integrand dM/dlnx goes as x^1 (central galaxy) to x^3 (gas)
    at small x and as x^-4 or steeper beyond the truncation, so x^-q dM/dlnx
    vanishes at both ends of the grid and FFTLog's periodic wrap is harmless.
    The output covers k R200c in [1e-7, 1e7]; the absolute accuracy is ~1e-4
    (6e-5 against an exact Gaussian transform).
    """

    def __init__(self, n: int = 4096, x_min: float = 1e-7, x_max: float = 1e7,
                 q: float = 0.3):
        if not 0.0 < q < 2.0:
            raise ValueError("FFTLog bias q must lie in (0, 2) for the j0 kernel")
        lnx = np.linspace(np.log(x_min), np.log(x_max), n)
        dlnx = lnx[1] - lnx[0]
        lny = -lnx[::-1]                       # y = k R200c, centred on x y = 1

        # Mellin transform of j0: int t^(s-1) j0(t) dt
        #   = sqrt(pi) 2^(s-2) Gamma(s/2) / Gamma((3-s)/2),   0 < Re s < 2.
        m = np.arange(n // 2 + 1)
        eta = 2.0 * np.pi * m / (n * dlnx)
        s = q + 1j * eta
        lnU = (0.5 * np.log(np.pi) + (s - 2.0) * np.log(2.0)
               + loggamma(s / 2.0) - loggamma((3.0 - s) / 2.0))
        kernel = np.exp(lnU - 1j * eta * (lnx[0] + lny[0]))
        if n % 2 == 0:
            kernel[-1] = kernel[-1].real        # Nyquist term must be real

        self.n, self.q, self.dlnx = n, q, dlnx
        self.lnx = jnp.asarray(lnx)
        self.x = jnp.exp(self.lnx)
        self.lny = jnp.asarray(lny)
        self.kernel = jnp.asarray(kernel)


def _fftlog_j0(f: jnp.ndarray, grid: S25Grid) -> jnp.ndarray:
    """G(y_j) = int f(x) j0(x y_j) dlnx on grid.lny, for f on grid.lnx (last axis)."""
    c = jnp.fft.rfft(f * jnp.exp(-grid.q * grid.lnx), axis=-1)
    d = c * grid.kernel
    return jnp.exp(-grid.q * grid.lny) * jnp.fft.irfft(jnp.conj(d), n=grid.n, axis=-1)


def _cumtrapz(f: jnp.ndarray, dlnx: float) -> jnp.ndarray:
    """Cumulative integral of f dlnx along the last axis.

    The part below the grid is added as a power law f ~ x^s fitted to the
    first two points, int_0^x0 f dlnx = f0 / s. Starting at exactly 0 instead
    makes the mass ratios in the relaxation term 0/0 at the first point.
    """
    f0, f1 = f[..., :1], f[..., 1:2]
    pos = (f0 > 0) & (f1 > 0)
    s = jnp.where(pos, jnp.log(jnp.where(pos, f1 / jnp.where(pos, f0, 1.0), 1.0)) / dlnx, 1.0)
    head = jnp.where(pos & (s > 0.1), f0 / jnp.clip(s, 0.1, None), 0.0)
    inc = 0.5 * (f[..., 1:] + f[..., :-1]) * dlnx
    return head + jnp.concatenate([jnp.zeros(f.shape[:-1] + (1,)), jnp.cumsum(inc, axis=-1)], axis=-1)


# ============================================================================
# Fractions
# ============================================================================

def s25_fractions(M, z, f_bar, p):
    """Stellar and gas fractions of M_tot (BaryonForge clipping included)."""
    m = M / p['Mstar']
    f_star = p['Nstar'] / (m ** p['tau'] + m ** p['eta'])
    f_cga = p['Nstar'] / (m ** (p['tau'] + p['tau_delta']) + m ** (p['eta'] + p['eta_delta']))
    f_star = jnp.clip(f_star, 1e-10, f_bar)
    f_cga = jnp.clip(f_cga, 1e-10, f_star)
    f_sga = jnp.clip(f_star - f_cga, 1e-10, None)
    f_iga = jnp.clip(f_cga * p['c_iga'] * (1.0 + z) ** p['nu_c_iga'], 1e-10, f_bar - f_star)
    f_hga = jnp.clip(f_bar - f_star - f_iga, 1e-10, f_bar)
    return dict(f_star=f_star, f_cga=f_cga, f_sga=f_sga, f_iga=f_iga, f_hga=f_hga)


def _gas_shape_params(M, z, p):
    """beta, theta_c, delta, gamma, alpha per halo (cdelta = None in BaryonForge)."""
    M_c = p['M_c'] * (1.0 + z) ** p['nu_M_c']
    mm = (M / M_c) ** p['mu_beta']
    beta = 3.0 * mm / (1.0 + mm)

    def scaled(name):
        return (p[name] * (M / p['M_' + name]) ** p['mu_' + name]
                * (1.0 + z) ** p['nu_' + name])

    return beta, scaled('theta_c'), scaled('delta'), scaled('gamma'), scaled('alpha')


# ============================================================================
# Mass profiles on the x grid
# ============================================================================

def s25_mass_profiles(grid: S25Grid, M, R, c, nu, z, f_bar, p,
                      r_norm_max: float = jnp.inf) -> Dict[str, jnp.ndarray]:
    """dM/dlnx of every component on grid.x, shape (n_M, n_x).

    Parameters
    ----------
    M, R, c, nu : (n_M,) arrays
        M200c [Msun], comoving R200c [Mpc], c200c and peak height
        1.686 / sigma(M, a).
    z, f_bar : float
        Redshift and Omega_b / Omega_m.
    p : dict
        Output of :func:`complete_params`.
    r_norm_max : float
        Comoving radius [Mpc] to which the star and gas normalisation
        integrals (and the M_tot they are fractions of) run. BaryonForge uses
        10 Mpc (its r_max_int), which truncates them for halos with
        eps R200c >~ 3 Mpc; inf follows the paper (fractions of the mass at
        infinity). The two agree below ~1e14 Msun.

    Returns
    -------
    dict with 'dmo' (truncated NFW), 'dmb' (relaxed one-halo total, without
    the central point mass) and the components 'clm', 'cga', 'hga', 'iga' as
    dM/dlnx in Msun; 'point' (n_M,), the collisionless mass contracted onto
    the centre; 'M_tot' (n_M,); the fractions. The DMB total is
    sum(dmb) dlnx + point = M_tot.
    """
    x = grid.x[None, :]
    lnx = grid.lnx
    dlnx = grid.dlnx
    Mc, Rc, cc, nuc = (jnp.asarray(a)[:, None] for a in (M, R, c, nu))
    vol = 4.0 * jnp.pi * Rc ** 3 * x ** 3          # dM/dlnx = vol * rho

    # --- truncated NFW, normalised to M200c inside R200c ---------------------
    eps = jnp.clip(p['epsilon0'] - p['epsilon1'] * nuc, 1e-3, None)
    xs = x * cc
    shape_nfw = vol / (xs * (1.0 + xs) ** 2) / (1.0 + (x / eps) ** 2) ** 2
    cum_nfw = _cumtrapz(shape_nfw, dlnx)
    cum_at_R = vmap(lambda cm: jnp.interp(0.0, lnx, cm))(cum_nfw)[:, None]
    dM_nfw = shape_nfw * Mc / cum_at_R
    M_nfw = cum_nfw * Mc / cum_at_R
    M_tot = M_nfw[:, -1]

    # Normalisation window (BaryonForge integrates to r_max_int only).
    in_norm = (x * Rc) <= r_norm_max
    ln_rn = jnp.log(jnp.minimum(r_norm_max / Rc[:, 0], jnp.exp(lnx[-1])))
    M_tot_norm = vmap(jnp.interp, in_axes=(0, None, 0))(ln_rn, lnx, M_nfw)[:, None]

    def normalised(shape, frac):
        norm = jnp.sum(jnp.where(in_norm, shape, 0.0), axis=-1, keepdims=True) * dlnx
        # trapezoid end-correction is irrelevant: shapes vanish at both grid ends
        return shape * frac[:, None] * M_tot_norm / norm

    fr = s25_fractions(M, z, f_bar, p)

    # --- central galaxy: r^-2 exp(-r / (eps_cga R)) ----------------------------
    dM_cga = normalised(vol * x ** -2 * jnp.exp(-x / p['epsilon_cga']), fr['f_cga'])

    # --- hot gas ------------------------------------------------------------------
    beta, theta_c, delta, gamma, alpha = (a[:, None] for a in _gas_shape_params(M, z, p))
    ln_hga = (-(beta / alpha) * jnp.log1p((x / theta_c) ** alpha)
              - (delta / gamma) * jnp.log1p((x / eps) ** gamma))
    dM_hga = normalised(vol * jnp.exp(ln_hga), fr['f_hga'])

    # --- inner (cold) gas: r^-3 exp(-r/R) with a tanh core at r_min_iga --------
    w = 0.5 * (1.0 + jnp.tanh(jnp.log10(x * Rc / p['r_min_iga']) / 0.02))
    dM_iga = normalised(vol * w * x ** -3 * jnp.exp(-x), fr['f_iga'])

    M_cga = _cumtrapz(dM_cga, dlnx)
    M_hga = _cumtrapz(dM_hga, dlnx)
    M_iga = _cumtrapz(dM_iga, dlnx)

    # --- relaxed collisionless matter (S25 Eq. 2.16, BaryonForge form) ---------
    Q0 = p['q0'] + p['nu_q0'] * z
    Q1 = p['q1'] + p['nu_q1'] * z
    Q2 = p['q2'] + p['nu_q2'] * z
    x_step = eps / p['epsilon0']                    # r_step / R200c
    M_i = jnp.maximum(M_nfw, 1e-300)
    xi = (1.0 + Q0 / (1.0 + (x / x_step) ** p['nstep'])
          + Q1 * (M_cga / M_i - fr['f_cga'][:, None])
          + Q1 * (M_iga / M_i - fr['f_iga'][:, None])
          + Q2 * (M_hga / M_i - fr['f_hga'][:, None]))       # xi = r_i / r_f
    f_clm = 1.0 - f_bar + fr['f_sga']
    ln_M_i = jnp.log(M_i)
    ln_M_clm = vmap(jnp.interp, in_axes=(0, None, 0))(lnx[None, :] + jnp.log(xi), lnx, ln_M_i)
    M_clm = f_clm[:, None] * jnp.exp(ln_M_clm)
    # The Q1 term diverges as M_cga/M_i ~ 1/x for x -> 0, so r_f -> 0 at finite
    # r_i: the shells inside r_i = x_p R collapse onto the centre. That mass,
    # M_clm(x -> 0) = f_clm M_nfw(x_p), is a central point mass (0.3-0.9% of
    # M_tot for 1e11.5-1e13 Msun/h). Keep it as one, outside dM/dlnx.
    M_point = M_clm[:, 0]
    dM_clm = jnp.clip(jnp.gradient(M_clm, dlnx, axis=-1), 0.0, None)

    # --- one-halo DMB, rescaled so its total matches the DMO total -------------
    dM_sum = dM_clm + dM_cga + dM_hga + dM_iga
    factor = M_tot / (jnp.sum(dM_sum, axis=-1) * dlnx + M_point)
    dM_dmb = dM_sum * factor[:, None]

    out = dict(dmo=dM_nfw, dmb=dM_dmb, clm=dM_clm * factor[:, None],
               cga=dM_cga * factor[:, None], hga=dM_hga * factor[:, None],
               iga=dM_iga * factor[:, None], point=M_point * factor,
               M_tot=M_tot, factor=factor, beta=beta[:, 0], eps=eps[:, 0])
    out.update(fr)
    return out


def s25_density(prof: Dict[str, jnp.ndarray], grid: S25Grid, R, key: str) -> jnp.ndarray:
    """Real-space density [Msun / Mpc^3] of one component, shape (n_M, n_x)."""
    r = grid.x[None, :] * jnp.asarray(R)[:, None]
    return prof[key] / (4.0 * jnp.pi * r ** 3)


# ============================================================================
# Fourier space
# ============================================================================

def s25_fourier(grid: S25Grid, k, M, R, c, nu, z, f_bar, p,
                r_norm_max: float = jnp.inf):
    """Normalised one-halo Fourier profiles u_DMB(k|M), u_DMO(k|M).

    k is in 1/Mpc (comoving) and R in Mpc, so only k R matters. Both are
    normalised by M_tot (the truncated NFW mass at infinity), so both tend to
    1 as k -> 0. Returns two (n_k, n_M) arrays, the layout of
    ``power_spectrum.nfw_fourier_u``.
    """
    prof = s25_mass_profiles(grid, M, R, c, nu, z, f_bar, p, r_norm_max)
    M_tot = prof['M_tot'][:, None]
    G_dmb = _fftlog_j0(prof['dmb'] / M_tot, grid)
    G_dmo = _fftlog_j0(prof['dmo'] / M_tot, grid)
    ln_kR = jnp.log(jnp.asarray(k)[None, :] * jnp.asarray(R)[:, None])   # (n_M, n_k)
    interp = vmap(jnp.interp, in_axes=(0, None, 0))
    u_point = (prof['point'] / prof['M_tot'])[:, None]                  # j0(0) = 1
    return (interp(ln_kR, grid.lny, G_dmb) + u_point).T, interp(ln_kR, grid.lny, G_dmo).T


s25_mass_profiles_jit = jit(s25_mass_profiles, static_argnums=(0,))
s25_fourier_jit = jit(s25_fourier, static_argnums=(0,))


# ============================================================================
# Mass-definition helper
# ============================================================================

def m200m_to_m200c(M200m, c200m, Omega_m_z):
    """Convert (M200m, c200m) to (M200c, R200c/R200m, c200c) for an NFW halo.

    Omega_m_z = rho_m(z) / rho_c(z). Solves m(x)/x^3 = m(c)/c^3 / Omega_m_z
    for x = R200c / r_s, with m(x) = ln(1+x) - x/(1+x). Numpy, one-off.
    """
    M200m = np.atleast_1d(np.asarray(M200m, dtype=float))
    c200m = np.broadcast_to(np.asarray(c200m, dtype=float), M200m.shape)
    mu = lambda t: np.log1p(t) - t / (1.0 + t)
    x200c = np.array([brentq(lambda t, cm=cm: mu(t) / t ** 3 - mu(cm) / cm ** 3 / Omega_m_z,
                             1e-3 * cm, cm) for cm in c200m])
    M200c = M200m * mu(x200c) / mu(c200m)
    return M200c, x200c / c200m, x200c
