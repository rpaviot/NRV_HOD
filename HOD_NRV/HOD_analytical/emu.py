"""
Emulator: Beta_NL Interpolator using Dark Emulator and interpax
================================================================

This module provides the BetaNLInterpolator class for computing and interpolating
the beyond-linear halo bias correction beta^NL(k, M1, M2) at specific redshifts.

The beyond-linear halo bias beta^NL is defined as:
    P_hh(k, M1, M2) = b(M1) * b(M2) * P_lin(k) * [1 + beta^NL(k, M1, M2)]

where beta^NL -> 0 as k -> 0 (linear regime).

Uses interpax.Interpolator3D for fast, differentiable interpolation on a
precomputed (log k, log M1, log M2) grid built from the Dark Emulator.
"""

import numpy as np
from typing import Dict, Optional, Any, Union, List

try:
    from dark_emulator import darkemu
    HAS_DARK_EMULATOR = True
except ImportError:
    HAS_DARK_EMULATOR = False
    darkemu = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from interpax import Interpolator3D
    HAS_JAX = True
    HAS_INTERPAX = True
    jax.config.update("jax_enable_x64", True)
except ImportError:
    HAS_JAX = False
    HAS_INTERPAX = False
    jnp = np


# ============================================================================
# BetaNLInterpolator
# ============================================================================

class BetaNLInterpolator:
    """
    Interpolator for beta^NL(k, M1, M2) at specific redshifts using interpax.

    Parameters
    ----------
    emu : darkemu.base_class
        Initialized Dark Emulator instance
    h : float
        Dimensionless Hubble parameter
    z_values : float or array-like
        Redshift(s) at which to compute beta^NL
    n_k : int
        Number of k points in the grid (default: 100)
    n_mass : int
        Number of mass points in the grid (default: 30)
    k_min, k_max : float
        k range in h/Mpc (default: 0.01 to 10)
    log_M_min, log_M_max : float
        log10(M) range in Msun/h (default: 12 to 15)
    method : str
        Interpolation method for interpax (default: 'linear')
    bias_method : str
        Method for extracting linear halo bias:
        - 'halo-halo' (default): b(M) = sqrt(P_hh(k_lin, M, M) / P_lin(k_lin))
        - 'traditional': Use emulator's bias function directly
    force_to_zero : str
        Method for correcting beta^NL at large scales:
        - 'additive' (default): Subtract offset measured at k_lin
        - 'multiplicative': Divide by (1 + offset)
        - 'exponential': Multiply by (1 - exp(-(k/k_lin)^2))
        - 'none': No correction
    k_lin : float
        Scale defining "linear" regime in h/Mpc (default: 0.02)
    constant_low : bool
        If True, use constant extrapolation at low mass boundaries instead of
        linear extrapolation. When a requested mass is below log_M_min, the
        interpolator will return the value at log_M_min. (default: False)
    verbose : bool
        Print progress information (default: True)

    Notes
    -----
    The emulator is valid for M in [10^12, 10^15] Msun/h.
    Grid is computed within emulator range; interpax extrapolates beyond.

    The 'halo-halo' bias method is recommended as it ensures self-consistency
    between the bias and the halo power spectrum from the same emulator.

    The 'additive' force_to_zero is recommended as it minimally modifies the
    emulator output while ensuring the correct large-scale behavior.

    When constant_low=True, masses below log_M_min are clamped to log_M_min,
    providing constant extrapolation at the low-mass boundary. This can be
    useful to avoid unphysical behavior from linear extrapolation.
    """

    BIAS_METHODS = ['traditional', 'halo-halo']
    FORCE_TO_ZERO_METHODS = ['none', 'additive', 'multiplicative', 'exponential']

    def __init__(
        self,
        emu,
        h: float,
        z_values: Union[float, np.ndarray, List[float]],
        n_k: int = 100,
        n_mass: int = 30,
        k_min: float = 1e-2,
        k_max: float = 10.0,
        log_M_min: float = 12.0,
        log_M_max: float = 15.0,
        method: str = "linear",
        bias_method: str = "halo-halo",
        force_to_zero: str = "additive",
        k_lin: float = 0.02,
        constant_low: bool = False,
        constant_low_limit: Optional[float] = None,  # log10(M) threshold
        verbose: bool = True,
    ):
        if not HAS_JAX or not HAS_INTERPAX:
            raise RuntimeError("JAX and interpax are required for BetaNLInterpolator")

        if bias_method not in self.BIAS_METHODS:
            raise ValueError(f"bias_method must be one of {self.BIAS_METHODS}, got '{bias_method}'")

        if force_to_zero not in self.FORCE_TO_ZERO_METHODS:
            raise ValueError(f"force_to_zero must be one of {self.FORCE_TO_ZERO_METHODS}, got '{force_to_zero}'")

        self.emu = emu
        self.h = h
        self.verbose = verbose
        self.method = method
        self.bias_method = bias_method
        self.force_to_zero = force_to_zero
        self.k_lin = k_lin
        self.constant_low = constant_low
        self.constant_low_limit = constant_low_limit if constant_low_limit is not None else log_M_min

        # Store redshifts
        self.z_values = np.atleast_1d(z_values)
        self.n_z = len(self.z_values)

        # Grid parameters
        self.n_k = n_k
        self.n_mass = n_mass
        self.k_min = k_min
        self.k_max = k_max
        self.log_M_min = log_M_min
        self.log_M_max = log_M_max

        # Build grid arrays
        self.log_k_arr = np.linspace(np.log10(k_min), np.log10(k_max), n_k)
        self.k_arr = 10**self.log_k_arr
        self.log_M_arr = np.linspace(log_M_min, log_M_max, n_mass)
        self.M_arr = 10**self.log_M_arr

        # Build interpolators
        self._build_interpolators()

    def _get_bias_traditional(self, M: float, z: float) -> float:
        """
        Get linear halo bias using the emulator's bias function.

        This uses a finite difference approach on the mass-to-density relation.
        """
        Mp = M * 1.01
        Mm = M * 0.99
        logdensp = np.log10(self.emu.mass_to_dens(Mp, z))
        logdensm = np.log10(self.emu.mass_to_dens(Mm, z))
        bp = self.emu.get_bias(logdensp, z)
        bm = self.emu.get_bias(logdensm, z)
        return (bm * 10**logdensm - bp * 10**logdensp) / (10**logdensm - 10**logdensp)

    def _get_linear_bias(self, M: float, z: float, Pk_klin: float) -> float:
        """
        Get linear halo bias using the specified method.

        Parameters
        ----------
        M : float
            Halo mass in Msun/h
        z : float
            Redshift
        Pk_klin : float
            Linear power spectrum at k_lin

        Returns
        -------
        float
            Linear halo bias b(M, z)
        """
        if self.bias_method == 'traditional':
            return self._get_bias_traditional(M, z)

        elif self.bias_method == 'halo-halo':
            k_lin_arr = np.array([self.k_lin])
            Phh_klin = self.emu.get_phh_mass(k_lin_arr, M, M, z)[0]
            return np.sqrt(Phh_klin / Pk_klin)

        else:
            raise ValueError(f"Unknown bias method: {self.bias_method}")

    def _get_linear_power(self, k_h: np.ndarray, z: float) -> np.ndarray:
        """Get linear power spectrum from emulator."""
        return self.emu.get_pklin_from_z(k_h, z)

    def _get_halo_power(self, k_h: np.ndarray, M1: float, M2: float, z: float) -> np.ndarray:
        """Get halo-halo power spectrum from emulator."""
        return np.asarray(self.emu.get_phh_mass(np.asarray(k_h), M1, M2, z))

    def _apply_force_to_zero(
        self,
        beta: np.ndarray,
        M1: float,
        M2: float,
        b1: float,
        b2: float,
        Pk_klin: float,
        z: float
    ) -> np.ndarray:
        """
        Apply force_to_zero correction to beta^NL.

        Parameters
        ----------
        beta : np.ndarray
            Raw beta^NL values
        M1, M2 : float
            Halo masses
        b1, b2 : float
            Linear biases for M1, M2
        Pk_klin : float
            Linear power at k_lin
        z : float
            Redshift

        Returns
        -------
        np.ndarray
            Corrected beta^NL values
        """
        if self.force_to_zero == 'none':
            return beta

        k_lin_arr = np.array([self.k_lin])

        if self.force_to_zero == 'additive':
            # Subtract offset measured at k_lin
            Pk_hh_klin = self.emu.get_phh_mass(k_lin_arr, M1, M2, z)[0]
            offset = Pk_hh_klin / (b1 * b2 * Pk_klin) - 1.0
            return beta - offset

        elif self.force_to_zero == 'multiplicative':
            # Divide by (1 + offset)
            Pk_hh_klin = self.emu.get_phh_mass(k_lin_arr, M1, M2, z)[0]
            offset = Pk_hh_klin / (b1 * b2 * Pk_klin) - 1.0
            return (beta + 1.0) / (offset + 1.0) - 1.0

        elif self.force_to_zero == 'exponential':
            # Smooth suppression: beta * (1 - exp(-(k/k_lin)^2))
            suppression = 1.0 - np.exp(-(self.k_arr / self.k_lin)**2)
            return beta * suppression

        else:
            raise ValueError(f"Unknown force_to_zero method: {self.force_to_zero}")

    def _compute_beta_nl_grid(self, z: float) -> np.ndarray:
        """
        Compute beta^NL grid for a single redshift.

        Returns
        -------
        np.ndarray
            Shape: (n_k, n_mass, n_mass)
        """
        beta_grid = np.zeros((self.n_k, self.n_mass, self.n_mass))

        # Linear power at k_lin for bias calculation
        k_lin_arr = np.array([self.k_lin])
        Pk_klin = self._get_linear_power(k_lin_arr, z)[0]

        # Linear power at all k values
        Pk_lin = self._get_linear_power(self.k_arr, z)

        # Precompute linear biases
        biases = np.array([
            self._get_linear_bias(float(M), z, Pk_klin)
            for M in self.M_arr
        ])

        for i1 in range(self.n_mass):
            for i2 in range(i1, self.n_mass):
                M1, M2 = float(self.M_arr[i1]), float(self.M_arr[i2])
                b1, b2 = biases[i1], biases[i2]

                Phh = self._get_halo_power(self.k_arr, M1, M2, z)

                # Compute raw beta^NL
                denom = b1 * b2 * Pk_lin
                with np.errstate(divide='ignore', invalid='ignore'):
                    beta = Phh / denom - 1.0
                    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

                # Apply force_to_zero correction
                beta = self._apply_force_to_zero(beta, M1, M2, b1, b2, Pk_klin, z)

                beta_grid[:, i1, i2] = beta
                beta_grid[:, i2, i1] = beta  # Symmetry

        return beta_grid

    def _build_interpolators(self):
        """Build interpax.Interpolator3D for each redshift."""
        if self.verbose:
            print(f"Building beta^NL interpolators for {self.n_z} redshift(s)")
            print(f"  k: [{self.k_min:.2e}, {self.k_max:.2e}] h/Mpc, n={self.n_k}")
            print(f"  M: [10^{self.log_M_min:.1f}, 10^{self.log_M_max:.1f}] Msun/h, n={self.n_mass}")
            print(f"  bias_method: '{self.bias_method}'")
            print(f"  force_to_zero: '{self.force_to_zero}'")
            print(f"  k_lin: {self.k_lin} h/Mpc")
            print(f"  constant_low: {self.constant_low}")

        self.interpolators = {}
        self.beta_nl_grids = {}

        for iz, z in enumerate(self.z_values):
            if self.verbose:
                print(f"  z = {z:.4f} ({iz+1}/{self.n_z})...")

            beta_grid = self._compute_beta_nl_grid(z)
            self.beta_nl_grids[iz] = beta_grid

            self.interpolators[iz] = Interpolator3D(
                x=self.log_k_arr,
                y=self.log_M_arr,
                z=self.log_M_arr,
                f=beta_grid,
                method=self.method,
                extrap=[[True, True], [True, True], [True, True]],
            )

    def get_z_index(self, z: float) -> int:
        """Get index of closest redshift."""
        z_diff = np.abs(self.z_values - z)
        idx = np.argmin(z_diff)
        if z_diff[idx] > 0.01:
            raise ValueError(f"z={z} not found. Available: {self.z_values}")
        return idx

    def _clamp_log_mass_low(self, log_M: Any) -> Any:
        """
        Clamp log mass values to the minimum grid value if constant_low is True.

        Parameters
        ----------
        log_M : array-like
            log10(M) values

        Returns
        -------
        array-like
            Clamped log10(M) values (if constant_low=True) or original values
        """
        if self.constant_low:
            return jnp.maximum(log_M, self.constant_low_limit)
        return log_M

    def __call__(self, k: Any, M1: Any, M2: Any, z: float = 0.0) -> jnp.ndarray:
        """
        Interpolate beta^NL at (k, M1, M2, z).

        Parameters
        ----------
        k : array-like
            Wavenumber(s) in h/Mpc
        M1, M2 : float
            Halo masses in Msun/h
        z : float
            Redshift (must be close to a pre-computed value)

        Returns
        -------
        jnp.ndarray
            beta^NL values at the requested points

        Notes
        -----
        If constant_low=True, masses below 10^log_M_min are clamped to the
        minimum grid value, providing constant extrapolation at low masses.
        """
        iz = self.get_z_index(z)

        k = jnp.atleast_1d(k)
        log_k = jnp.log10(k)
        log_M1 = jnp.full_like(log_k, jnp.log10(M1))
        log_M2 = jnp.full_like(log_k, jnp.log10(M2))

        # Apply constant_low clamping if enabled
        log_M1 = self._clamp_log_mass_low(log_M1)
        log_M2 = self._clamp_log_mass_low(log_M2)

        return self.interpolators[iz](log_k, log_M1, log_M2)

    def interpolate_to_mass_grid(
        self,
        log10M_target: jnp.ndarray,
        z: float,
        k_target: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Interpolate beta^NL to a target mass grid.

        Parameters
        ----------
        log10M_target : array-like
            Target log10(M) values
        z : float
            Redshift
        k_target : array-like, optional
            Target k values (default: use internal k_arr)

        Returns
        -------
        jnp.ndarray
            Shape (n_k, n_M_target, n_M_target)

        Notes
        -----
        If constant_low=True, masses below 10^log_M_min are clamped to the
        minimum grid value, providing constant extrapolation at low masses.
        """
        iz = self.get_z_index(z)
        interp = self.interpolators[iz]

        log_k_vals = jnp.log10(k_target) if k_target is not None else jnp.array(self.log_k_arr)

        # Apply constant_low clamping if enabled
        log10M_target_clamped = self._clamp_log_mass_low(jnp.asarray(log10M_target))
        n_M = len(log10M_target_clamped)

        @jit
        def eval_at_k(log_k_val):
            M1_grid, M2_grid = jnp.meshgrid(log10M_target_clamped, log10M_target_clamped, indexing='ij')
            k_flat = jnp.full(n_M * n_M, log_k_val)
            return interp(k_flat, M1_grid.ravel(), M2_grid.ravel()).reshape(n_M, n_M)

        return vmap(eval_at_k)(log_k_vals)

    def get_grid_data(self, iz: int = 0) -> Dict[str, np.ndarray]:
        """
        Return grid data for inspection.

        Parameters
        ----------
        iz : int
            Redshift index

        Returns
        -------
        dict
            Contains 'log_k_arr', 'log_M_arr', 'M_arr', 'z', 'beta_nl_grid',
            and 'constant_low'
        """
        return {
            'log_k_arr': self.log_k_arr,
            'k_arr': self.k_arr,
            'log_M_arr': self.log_M_arr,
            'M_arr': self.M_arr,
            'z': self.z_values[iz],
            'beta_nl_grid': self.beta_nl_grids[iz],
            'constant_low': self.constant_low,
        }


__all__ = [
    'BetaNLInterpolator',
    'HAS_DARK_EMULATOR',
    'HAS_JAX',
    'HAS_INTERPAX',
    'darkemu',
]
