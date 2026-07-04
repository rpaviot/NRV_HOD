"""
Halo Model Power Spectrum Calculator - INTERPAX VERSION
=========================================================

Updated to work with the interpax-based Cosmology class.

Key features:
1. Uses interpax for beta^NL interpolation (via Cosmology.beta_nl_interp)
2. Corrected beta^NL implementation with I^11, I^12, I^21, I^22 terms
3. Clean JAX-based implementation with Gauss-Legendre integration
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Union, List
from .pycosmo import Cosmology
from .emu import HAS_INTERPAX

from .hod_analytical import (
    N_GL, GL_X, GL_W, gl_nodes_scaled, gl_integrate,
    HOD_PARAM_DEFINITIONS, CSMF_HOD_PARAMS,
    get_required_params, validate_hod_params,
    AnalyticalHOD, StandardHOD, CSMF_HOD, create_hod,
    csmf_N_central, csmf_N_satellite,
)

from .power_spectrum import (
    nfw_fourier_u, nfw_fourier_u_single,
    _compute_ngal, _compute_Pgg, _compute_Pgm,
    _compute_Pgg_with_beta_nl, _compute_Pgm_with_beta_nl,
    _compute_I_NL_22, _compute_I_NL_12, _compute_I_NL_21,
    _compute_I_NL_gg, _compute_I_NL_gm,
)

try:
    from HOD_NRV.utilsf.hankel_transforms import (
        Pk_to_wgg_direct,
        Pk_to_DeltaSigma_direct,
        Pk_gm_to_DeltaSigma_traditional,
    )
    HAS_HANKEL = True
except ImportError:
    HAS_HANKEL = False

jax.config.update("jax_enable_x64", True)


# ============================================================================
# HaloModel Class
# ============================================================================

class HaloModel(Cosmology):
    """
    Halo Model with interpax-based beta^NL support.

    Inherits from the simplified Cosmology class that uses interpax
    for beta^NL interpolation.
    """

    def __init__(
        self,
        cosmo_params: Dict[str, float],
        z: Union[float, np.ndarray, List[float]],
        hod_type: str,
        f_c: float = 1.0,
        f_s: float = 1.0,
        M_min: float = 1e9,
        M_max: float = 1e16,
        masses_are_log10: bool = True,
        units_per_h: bool = True,
        k_array: Optional[Union[float, np.ndarray]] = None,
        Mstar_min: Optional[Union[float, np.ndarray]] = None,
        Mstar_max: Optional[Union[float, np.ndarray]] = None,
        median_Mstar: Optional[Union[float, np.ndarray]] = None,
        include_beta_nl: bool = False,
        beta_nl_kwargs: Optional[Dict] = None,
        beta_nl_source: str = 'emulator',
        verbose: bool = True,
        **cosmo_kwargs
    ):
        # Initialize Cosmology base class
        super().__init__(cosmo_params, beta_nl_kwargs=beta_nl_kwargs,
                         beta_nl_source=beta_nl_source,
                         verbose=verbose, k_array=k_array,
                         units_per_h=units_per_h, **cosmo_kwargs)

        self.verbose = verbose

        # Redshift setup
        self.z_array = np.atleast_1d(z)
        self.n_z = len(self.z_array)
        self.is_single_z = (self.n_z == 1)
        self.a_array = 1.0 / (1.0 + self.z_array)

        # HOD setup
        self.hod_type = hod_type.lower()
        self.masses_are_log10 = masses_are_log10
        self.hod = create_hod(self.hod_type, masses_are_log10=masses_are_log10)

        if self.hod_type.upper() == 'CSMF' and (Mstar_min is None or Mstar_max is None):
            raise ValueError("CSMF HOD requires Mstar_min and Mstar_max")

        self._Mstar_min_jax, self._Mstar_max_jax = self._prepare_mstar_arrays(Mstar_min, Mstar_max)
        self.RHO_M = self.get_rho_m()
        self.f_c = f_c
        self.f_s = f_s

        # Mass array at GL nodes
        self.log10M_min = np.log10(M_min)
        self.log10M_max = np.log10(M_max)
        log10M_gl = np.array(gl_nodes_scaled(self.log10M_min, self.log10M_max))
        self.M_array = 10.0 ** log10M_gl
        self._log10M_gl_jax = jnp.array(log10M_gl)

        # Stellar mass for point mass contribution
        self.median_Mstar = self._prepare_median_Mstar(median_Mstar)

        # beta^NL cache
        self.include_beta_nl = include_beta_nl
        self._beta_nl_gl_cache = {}
        self._beta_nl_Mmin_row_cache = {}
        self._beta_nl_Mmin_col_cache = {}
        self._beta_nl_Mmin_Mmin_cache = {}

        # Precompute CCL quantities
        self._precompute_ccl()

        # Compute beta^NL
        if include_beta_nl:
            self._compute_beta_nl()

        if self.verbose:
            print(f"HaloModel: {self.n_z} z, {self.n_k} k, {N_GL} M points")
            print(f"  HOD: {self.hod_type}, f_c={f_c}, f_s={f_s}")
            print(f"  Mass: [10^{self.log10M_min:.1f}, 10^{self.log10M_max:.1f}]")
            if include_beta_nl:
                print(f"  β^NL: enabled (source='{self.beta_nl_source}')")
            else:
                print("  β^NL: disabled")

    def _prepare_mstar_arrays(self, Mstar_min, Mstar_max):
        # Only CSMF needs Mstar arrays
        hod_type_upper = self.hod_type.upper()
        if hod_type_upper != 'CSMF':
            return jnp.zeros(self.n_z), jnp.zeros(self.n_z)

        Mstar_min = jnp.atleast_1d(Mstar_min)
        Mstar_max = jnp.atleast_1d(Mstar_max)

        if len(Mstar_min) == 1:
            Mstar_min = jnp.full(self.n_z, Mstar_min[0])
        if len(Mstar_max) == 1:
            Mstar_max = jnp.full(self.n_z, Mstar_max[0])

        return jnp.asarray(Mstar_min), jnp.asarray(Mstar_max)

    def _prepare_median_Mstar(self, median_Mstar):
        if median_Mstar is None:
            return None
        if self.hod_type.upper() != 'CSMF':
            raise ValueError("median_Mstar only for CSMF")

        arr = np.atleast_1d(median_Mstar)
        if len(arr) == 1:
            return np.full(self.n_z, float(arr[0]))
        elif len(arr) == self.n_z:
            return np.array(arr)
        raise ValueError(f"median_Mstar length must be 1 or {self.n_z}")

    def _precompute_ccl(self):
        """Precompute CCL quantities at GL nodes and M_min.

        When units_per_h=True, user provides HOD masses in Msun/h.
        CCL requires natural units (Msun), so we convert: M_ccl = M_h / h.
        Output quantities are rescaled for h-unit consistency:
        - n_M: multiply by h^3
        - R_s: multiply by h
        """
        M_min_val = 10.0 ** self.log10M_min

        # When units_per_h=True, M_array is in Msun/h (matches user's HOD params)
        # CCL needs natural units (Msun), so convert for CCL calls
        if self.units_per_h:
            M_ccl = self.M_array / self.h  # Msun/h -> Msun
            M_min_ccl = M_min_val / self.h
        else:
            M_ccl = self.M_array
            M_min_ccl = M_min_val

        self._n_M = np.zeros((self.n_z, N_GL))
        self._b_h = np.zeros((self.n_z, N_GL))
        self._R_s = np.zeros((self.n_z, N_GL))
        self._c = np.zeros((self.n_z, N_GL))
        self._Pk_lin = np.zeros((self.n_z, self.n_k))
        self._R_s_Mmin = np.zeros(self.n_z)
        self._c_Mmin = np.zeros(self.n_z)

        for iz, a in enumerate(self.a_array):
            # CCL calls with natural unit masses
            self._n_M[iz] = self.mass_func(self.ccl_cosmo, M_ccl, a)
            self._b_h[iz] = self.halo_bias_model(self.ccl_cosmo, M_ccl, a)

            # Use parent class linear_power() - already handles h-units
            self._Pk_lin[iz] = self.linear_power(z=self.z_array[iz])

            # R_s and concentration from CCL (natural units)
            R_vir = self.mass_def.get_radius(self.ccl_cosmo, M_ccl, a) / a
            self._c[iz] = self.concentration_model(self.ccl_cosmo, M_ccl, a)
            self._R_s[iz] = R_vir / self._c[iz]

            R_vir_Mmin = self.mass_def.get_radius(self.ccl_cosmo, M_min_ccl, a) / a
            self._c_Mmin[iz] = self.concentration_model(self.ccl_cosmo, M_min_ccl, a)
            self._R_s_Mmin[iz] = R_vir_Mmin / self._c_Mmin[iz]

            # Apply h-unit conversions for internal consistency
            if self.units_per_h:
                self._n_M[iz] /= self.h**3   # dn/dlogM factor (includes mass dependence)
                self._R_s[iz] *= self.h       # [Mpc] -> [Mpc/h]
                self._R_s_Mmin[iz] *= self.h

        # Convert to JAX arrays
        self._M_jax = jnp.array(self.M_array)  # Stays in user's units (h-units if units_per_h)
        self.k_np = self.get_k()
        self._k_jax = jnp.array(self.k_np)  # Uses get_k() which handles h-units
        self._n_M_jax = jnp.array(self._n_M)
        self._b_h_jax = jnp.array(self._b_h)
        self._R_s_jax = jnp.array(self._R_s)
        self._c_jax = jnp.array(self._c)
        self._Pk_lin_jax = jnp.array(self._Pk_lin)
        self._R_s_Mmin_jax = jnp.array(self._R_s_Mmin)
        self._c_Mmin_jax = jnp.array(self._c_Mmin)

    def _compute_beta_nl(self):
        """Compute beta^NL using the interpax-based interpolator."""
        if not HAS_INTERPAX:
            if self.verbose:
                print("Warning: interpax not available, skipping β^NL")
            return

        if self.beta_nl_source == 'emulator':
            if self.emu is None:
                if self.verbose:
                    print("Warning: DarkEmulator not available, skipping β^NL")
                return

            beta_nl_opts = {
                'n_k': 100,
                'n_mass': 20,
                'k_min': 1e-2,
                'k_max': 10.0,
                'log_M_min': 12.0,
                'log_M_max': 15.0,
                'method': 'linear',
                'verbose': self.verbose,
            }
            beta_nl_opts.update(self._beta_nl_kwargs)
            self.compute_beta_nl(self.z_array, **beta_nl_opts)
        else:
            # Numerical source: opts owned by NumericalBetaNLInterpolator,
            # do not inject emulator-only keys (n_mass, log_M_min, log_M_max).
            self.compute_beta_nl(self.z_array)

        if self.beta_nl_interp is None:
            return

        if self.verbose:
            print("  Interpolating β^NL to GL nodes...")

        # Convert masses to h-units for emulator
        if self.units_per_h:
            log10M_gl_h=self._log10M_gl_jax
            log10M_min_h = self.log10M_min
        else:
            log10M_gl_h = self._log10M_gl_jax - jnp.log10(self.h)
            log10M_min_h = self.log10M_min - np.log10(self.h)

        k_h = self.get_k_h()
        M_min_h = 10.0 ** log10M_min_h

        for iz, z in enumerate(self.z_array):
            # Get beta^NL on full (k, M, M) grid
            beta_nl_gl = self.beta_nl_interp.interpolate_to_mass_grid(
                log10M_gl_h, float(z), k_target=jnp.array(k_h)
            )
            self._beta_nl_gl_cache[iz] = beta_nl_gl

            # beta^NL at M_min row/column
            beta_nl_Mmin_row = np.zeros((self.n_k, N_GL))
            for j in range(N_GL):
                M2_h = 10.0 ** float(log10M_gl_h[j])
                beta_nl_Mmin_row[:, j] = np.asarray(
                    self.beta_nl_interp(k_h, M_min_h, M2_h, float(z))
                )

            self._beta_nl_Mmin_row_cache[iz] = jnp.array(beta_nl_Mmin_row)
            self._beta_nl_Mmin_col_cache[iz] = self._beta_nl_Mmin_row_cache[iz]  # Symmetric

            # beta^NL at (M_min, M_min)
            self._beta_nl_Mmin_Mmin_cache[iz] = jnp.array(
                self.beta_nl_interp(k_h, M_min_h, M_min_h, float(z))
            )

        if self.verbose:
            print(f"  β^NL cached for {self.n_z} redshifts")

    def _get_occupation(self, iz: int):
        """Get N_c, N_s for redshift index iz."""
        # AnalyticalHOD uses log10(M), legacy StandardHOD uses linear M
        if isinstance(self.hod, AnalyticalHOD):
            return self.hod.N_central(self._log10M_gl_jax), self.hod.N_satellite(self._log10M_gl_jax)
        elif isinstance(self.hod, StandardHOD):
            # Legacy StandardHOD uses linear mass
            return self.hod.N_central(self._M_jax), self.hod.N_satellite(self._M_jax)
        elif isinstance(self.hod, CSMF_HOD):
            # CSMF uses linear mass and Mstar bounds
            return (
                self.hod.N_central(self._M_jax, self._Mstar_min_jax[iz], self._Mstar_max_jax[iz]),
                self.hod.N_satellite(self._M_jax, self._Mstar_min_jax[iz], self._Mstar_max_jax[iz])
            )
        else:
            raise ValueError(f"Unknown HOD type: {type(self.hod)}")

    def _squeeze(self, arr):
        return arr[0] if self.is_single_z else arr

    def set_hod_params(self, hod_params: Dict):
        self.hod.set_params(hod_params)

    def update_f(self, f_c=None, f_s=None):
        if f_c is not None:
            self.f_c = f_c
        if f_s is not None:
            self.f_s = f_s

    def ngal(self, hod_params=None):
        """Compute galaxy number density."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")

        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            n = _compute_ngal(N_c, N_s, self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            result.append(n)

        ngal = jnp.array(result)
        # if self.units_per_h:
        #     ngal *= self.h ** 3
        return self._squeeze(np.asarray(ngal))

    def Pgg(self, hod_params=None):
        """Compute galaxy-galaxy power spectrum."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")

        use_beta_nl = self.include_beta_nl and len(self._beta_nl_gl_cache) > 0

        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)

            if use_beta_nl:
                P = _compute_Pgg_with_beta_nl(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                    self._beta_nl_gl_cache[iz],
                )
            else:
                P = _compute_Pgg(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                )
            result.append(P)

        Pgg = jnp.stack(result)
        # if self.units_per_h:
        #     Pgg *= self.h ** 3
        return self._squeeze(np.asarray(Pgg))

    def Pgm(self, hod_params=None):
        """Compute galaxy-matter power spectrum."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")

        use_beta_nl = self.include_beta_nl and len(self._beta_nl_gl_cache) > 0

        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)

            if use_beta_nl:
                P = _compute_Pgm_with_beta_nl(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self._M_jax, self.RHO_M,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                    self._beta_nl_gl_cache[iz],
                    self._beta_nl_Mmin_col_cache[iz],
                    self._R_s_Mmin_jax[iz], self._c_Mmin_jax[iz],
                )
            else:
                P = _compute_Pgm(
                    N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                    self._R_s_jax[iz], self._c_jax[iz],
                    self._Pk_lin_jax[iz], self._k_jax,
                    self._M_jax, self.RHO_M,
                    self.log10M_min, self.log10M_max, self.f_c, self.f_s,
                )
            result.append(P)

        Pgm = jnp.stack(result)
        # if self.units_per_h:
        #     Pgm *= self.h ** 3
        return self._squeeze(np.asarray(Pgm))


    def get_A_Mmin(self, iz: int = 0) -> float:
        """Get A(M_min) = 1 - integral(M/rho_m)*b*n dM."""
        W_m = self._M_jax / self.RHO_M
        return float(1.0 - gl_integrate(
            W_m * self._b_h_jax[iz] * self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        ))

    def effective_halo_mass(self, hod_params=None):
        """Compute effective halo mass."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")

        result = []
        for iz in range(self.n_z):
            N_c, _ = self._get_occupation(iz)
            n_c = gl_integrate(N_c * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            M_eff = gl_integrate(N_c * self._n_M_jax[iz] * self._M_jax,
                                 self.log10M_min, self.log10M_max) / n_c
            result.append(M_eff)

        M_eff = jnp.array(result)
        if self.units_per_h:
            M_eff *= self.h
        return self._squeeze(np.asarray(M_eff))

    def satellite_fraction(self, hod_params=None):
        """Compute satellite fraction."""
        if hod_params is not None:
            self.set_hod_params(hod_params)
        if self.hod.params is None:
            raise ValueError("HOD parameters not set")

        result = []
        for iz in range(self.n_z):
            N_c, N_s = self._get_occupation(iz)
            n_c = gl_integrate(N_c * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            n_s = gl_integrate(N_s * self._n_M_jax[iz], self.log10M_min, self.log10M_max)
            result.append(n_s / (n_c + n_s))

        return self._squeeze(np.asarray(result))

    def diagnose_beta_nl_terms(self, iz: int = 0, hod_params=None):
        """Diagnostic method to show contribution of each I^NL term."""
        if not self.include_beta_nl:
            print("β^NL not enabled")
            return

        if hod_params is not None:
            self.set_hod_params(hod_params)

        N_c, N_s = self._get_occupation(iz)

        # Compute profiles
        u_c = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], self.f_c)
        u_s = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], self.f_s)
        u_m = nfw_fourier_u(self._k_jax, self._R_s_jax[iz], self._c_jax[iz], 1.0)
        u_m_Mmin = nfw_fourier_u_single(self._k_jax, self._R_s_Mmin_jax[iz], self._c_Mmin_jax[iz])

        n_gal = gl_integrate((N_c + N_s) * self._n_M_jax[iz], self.log10M_min, self.log10M_max)

        # H_g and H_m with profiles
        H_g = (N_c[None, :] * u_c + N_s[None, :] * u_s) / n_gal
        W_m = self._M_jax / self.RHO_M
        H_m = u_m * W_m[None, :]

        A_Mmin = self.get_A_Mmin(iz)

        # Compute individual terms for P_gm
        I_22_gm = _compute_I_NL_22(
            self._beta_nl_gl_cache[iz], H_g, H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )

        integral_21 = _compute_I_NL_21(
            self._beta_nl_Mmin_col_cache[iz], H_g,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_21_gm = A_Mmin * u_m_Mmin * integral_21

        # Compute individual terms for P_gg
        I_22_gg = _compute_I_NL_gg(
            self._beta_nl_gl_cache[iz], H_g,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )

        # Compute individual terms for P_mm
        I_22_mm = _compute_I_NL_22(
            self._beta_nl_gl_cache[iz], H_m, H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )

        I_11_mm = (A_Mmin ** 2) * (u_m_Mmin ** 2) * self._beta_nl_Mmin_Mmin_cache[iz]

        integral_12_mm = _compute_I_NL_12(
            self._beta_nl_Mmin_row_cache[iz], H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_12_mm = A_Mmin * u_m_Mmin * integral_12_mm

        integral_21_mm = _compute_I_NL_21(
            self._beta_nl_Mmin_col_cache[iz], H_m,
            self._b_h_jax[iz], self._n_M_jax[iz],
            self.log10M_min, self.log10M_max
        )
        I_21_mm = A_Mmin * u_m_Mmin * integral_21_mm

        print(f"=== β^NL Diagnostic for z={self.z_array[iz]:.2f} ===")
        print(f"A(M_min) = {A_Mmin:.4f}")
        print(f"  (Mass fraction below M_min = 10^{self.log10M_min:.1f})")
        print()
        print("P_gg:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_gg))):.4e}")
        print()
        print("P_gm:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_gm))):.4e}")
        print(f"  I^21:  mean|I| = {float(jnp.mean(jnp.abs(I_21_gm))):.4e}")
        print(f"  I^21/I^22:     = {float(jnp.mean(jnp.abs(I_21_gm / (I_22_gm + 1e-30)))):.2%}")
        print()
        print("P_mm:")
        print(f"  I^22:  mean|I| = {float(jnp.mean(jnp.abs(I_22_mm))):.4e}")
        print(f"  I^21:  mean|I| = {float(jnp.mean(jnp.abs(I_21_mm))):.4e}")
        print(f"  I^12:  mean|I| = {float(jnp.mean(jnp.abs(I_12_mm))):.4e}")
        print(f"  I^11:  mean|I| = {float(jnp.mean(jnp.abs(I_11_mm))):.4e}")

        return {
            'A_Mmin': A_Mmin,
            'I_22_gg': np.asarray(I_22_gg),
            'I_22_gm': np.asarray(I_22_gm),
            'I_21_gm': np.asarray(I_21_gm),
            'I_22_mm': np.asarray(I_22_mm),
            'I_21_mm': np.asarray(I_21_mm),
            'I_12_mm': np.asarray(I_12_mm),
            'I_11_mm': np.asarray(I_11_mm),
        }

    def wgg(self, rp, rp_bins=None, hod_params=None):
        """Compute projected correlation function."""
        if not HAS_HANKEL:
            raise ImportError("Hankel transform utilities not available")

        Pgg = self.Pgg(hod_params)
        k = self.get_k()

        if self.is_single_z:
            Pgg = Pgg[np.newaxis, :]

        result = []
        for iz in range(self.n_z):
            rp_out, wgg_iz = Pk_to_wgg_direct(k, Pgg[iz], rp, rp_bins=rp_bins)
            result.append(wgg_iz)

        return rp_out, self._squeeze(np.array(result))

    def DeltaSigma(self, rp, rp_bins=None, method='direct', hod_params=None,
                   include_stellar=True):
        """Compute excess surface density."""
        if not HAS_HANKEL:
            raise ImportError("Hankel transform utilities not available")

        Pgm = self.Pgm(hod_params)
        k = self.get_k()

        if self.is_single_z:
            Pgm = Pgm[np.newaxis, :]

        result = []
        for iz in range(self.n_z):
            if method == 'direct':
                rp_out, ds_iz = Pk_to_DeltaSigma_direct(k, Pgm[iz], self.RHO_M, rp, rp_bins=rp_bins)
            elif method == 'traditional':
                rp_out, ds_iz = Pk_gm_to_DeltaSigma_traditional(k, Pgm[iz], self.RHO_M, rp, rp_bins=rp_bins)
            else:
                raise ValueError(f"Unknown method: {method}")

            if include_stellar and self.median_Mstar is not None:
                ds_stellar = self.median_Mstar[iz] / (np.pi * rp_out**2) / 1e12
                if self.units_per_h:
                    ds_stellar *=self.h

                ds_iz = ds_iz + ds_stellar

            result.append(ds_iz)

        return rp_out, self._squeeze(np.array(result))

    # ------------------------------------------------------------------
    # Pure / vmap-able CSMF ΔΣ forward pass
    # ------------------------------------------------------------------
    def make_deltasigma_jax(self, ds_builder, rp_centers):
        """Return a pure, ``jax.vmap``-able function ``theta -> ΔΣ[n_z, nbin]``.

        This is a stateless twin of ``set_hod_params``+``DeltaSigma``: instead of
        mutating ``self.hod.params``/``self.f_c`` and round-tripping through numpy,
        it threads the HOD parameter vector

            theta = [M0, M1, gamma1, gamma2, sigma_c, alpha_s, b0, b1, f_c, f_s]

        as explicit traced arguments and keeps everything in jnp, so the whole
        forward pass can be ``jax.vmap``'d over a batch of theta (nautilus
        ``vectorized=True``). All cosmology / halo-structure / β^NL quantities are
        fixed at init and captured here as constants; only the HOD params vary.

        Parameters
        ----------
        ds_builder : hankel_jax._JaxDirectDS
            Pure-jnp P_gm -> bin-averaged ΔΣ transform (built for this k-grid and
            the data's ``rp_bins``); reproduces ``Pk_to_DeltaSigma_direct``.
        rp_centers : array (nbin,)
            Data ``r_p`` bin centres, used for the 1-halo stellar point term
            exactly as the stateful ``DeltaSigma`` adds it (at ``rp_out``).

        Returns
        -------
        callable : theta[10] -> jnp.ndarray of shape (n_z, nbin)
        """
        if not isinstance(self.hod, CSMF_HOD):
            raise TypeError("make_deltasigma_jax supports the CSMF HOD only "
                            f"(got {type(self.hod).__name__}).")

        use_beta_nl = self.include_beta_nl and len(self._beta_nl_gl_cache) > 0
        n_z = self.n_z
        M = self._M_jax
        rho_m = self.RHO_M
        log10M_min, log10M_max = self.log10M_min, self.log10M_max
        # CSMF_HOD.set_params converts M0,M1 from log10 to linear mass when
        # masses_are_log10 (the rest pass through). The de-stated path calls
        # csmf_N_central/satellite directly, so we must apply the SAME transform.
        masses_are_log10 = self.hod.masses_are_log10

        # Stellar 1-halo point term, added at the data rp centres (matches the
        # stateful DeltaSigma: median_Mstar/(π rp²)/1e12, *h if units_per_h).
        stellar = None
        if self.median_Mstar is not None:
            h_fac = self.h if self.units_per_h else 1.0
            rp_c = jnp.asarray(rp_centers)
            stellar = (jnp.asarray(self.median_Mstar) * h_fac / 1e12)[:, None] \
                / (jnp.pi * rp_c[None, :] ** 2)

        def _predict(theta):
            M0, M1, g1, g2, sig, al, b0, b1, f_c, f_s = (theta[i] for i in range(10))
            if masses_are_log10:
                M0 = 10.0 ** M0
                M1 = 10.0 ** M1
            rows = []
            for iz in range(n_z):
                ms_lo = self._Mstar_min_jax[iz]
                ms_hi = self._Mstar_max_jax[iz]
                N_c = csmf_N_central(M, ms_lo, ms_hi, M0, M1, g1, g2, sig)
                N_s = csmf_N_satellite(M, ms_lo, ms_hi, M0, M1, g1, g2, al, b0, b1)
                if use_beta_nl:
                    Pgm = _compute_Pgm_with_beta_nl(
                        N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                        self._R_s_jax[iz], self._c_jax[iz],
                        self._Pk_lin_jax[iz], self._k_jax, M, rho_m,
                        log10M_min, log10M_max, f_c, f_s,
                        self._beta_nl_gl_cache[iz], self._beta_nl_Mmin_col_cache[iz],
                        self._R_s_Mmin_jax[iz], self._c_Mmin_jax[iz],
                    )
                else:
                    Pgm = _compute_Pgm(
                        N_c, N_s, self._n_M_jax[iz], self._b_h_jax[iz],
                        self._R_s_jax[iz], self._c_jax[iz],
                        self._Pk_lin_jax[iz], self._k_jax, M, rho_m,
                        log10M_min, log10M_max, f_c, f_s,
                    )
                rows.append(ds_builder.transform(Pgm, rho_m))   # (nbin,)
            ds_all = jnp.stack(rows)                            # (n_z, nbin)
            if stellar is not None:
                ds_all = ds_all + stellar
            return ds_all

        return jax.jit(_predict)

    def make_ngal_jax(self):
        """Return a pure, ``jax.vmap``-able function ``theta -> n_gal[n_z]``.

        The de-stated abundance twin of :meth:`ngal`, for the batched likelihood
        (nautilus ``vectorized=True``). Threads the same 10-vector

            theta = [M0, M1, gamma1, gamma2, sigma_c, alpha_s, b0, b1, f_c, f_s]

        and returns the model galaxy number density per mass bin. Units follow
        ``units_per_h``: ``self._n_M_jax`` already carries the 1/h^3 factor
        (set in the HMF prep), so with ``units_per_h=True`` the output is in
        h^3/Mpc^3 -- exactly matching :meth:`ngal` (whose ``*h**3`` line is
        intentionally inert) and the measured n_gal fed to the likelihood.
        f_c/f_s do not enter the occupation integral, so they are ignored here.

        Returns
        -------
        callable : theta[10] -> jnp.ndarray of shape (n_z,)
        """
        if not isinstance(self.hod, CSMF_HOD):
            raise TypeError("make_ngal_jax supports the CSMF HOD only "
                            f"(got {type(self.hod).__name__}).")
        n_z = self.n_z
        M = self._M_jax
        log10M_min, log10M_max = self.log10M_min, self.log10M_max
        masses_are_log10 = self.hod.masses_are_log10

        def _predict(theta):
            M0, M1, g1, g2, sig, al, b0, b1, f_c, f_s = (theta[i] for i in range(10))
            if masses_are_log10:
                M0 = 10.0 ** M0
                M1 = 10.0 ** M1
            rows = []
            for iz in range(n_z):
                ms_lo = self._Mstar_min_jax[iz]
                ms_hi = self._Mstar_max_jax[iz]
                N_c = csmf_N_central(M, ms_lo, ms_hi, M0, M1, g1, g2, sig)
                N_s = csmf_N_satellite(M, ms_lo, ms_hi, M0, M1, g1, g2, al, b0, b1)
                rows.append(_compute_ngal(N_c, N_s, self._n_M_jax[iz],
                                          log10M_min, log10M_max))
            return jnp.stack(rows)                              # (n_z,)

        return jax.jit(_predict)


__all__ = [
    # Re-export from hod_analytical
    'N_GL', 'GL_X', 'GL_W',
    'gl_nodes_scaled', 'gl_integrate',
    'HOD_PARAM_DEFINITIONS', 'CSMF_HOD_PARAMS',
    'get_required_params', 'validate_hod_params',
    'AnalyticalHOD', 'StandardHOD', 'CSMF_HOD', 'create_hod',
    # Re-export from power_spectrum
    'nfw_fourier_u', 'nfw_fourier_u_single',
    '_compute_ngal', '_compute_Pgg', '_compute_Pgm',
    '_compute_Pgg_with_beta_nl', '_compute_Pgm_with_beta_nl',
    '_compute_I_NL_22', '_compute_I_NL_12', '_compute_I_NL_21',
    '_compute_I_NL_gg', '_compute_I_NL_gm',
    # Main class
    'HaloModel',
    # Hankel transforms
    'HAS_HANKEL',
]
