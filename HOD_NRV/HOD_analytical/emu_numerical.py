"""
Numerical Beta_NL Interpolator
==============================

Mirrors `emu.BetaNLInterpolator` but consumes a simulation-measured
β^NL grid built from a stitched threshold-ξ cache (output of
`example_scripts/precompute_xi_tree_grid.py`) instead of the Dark
Emulator.

The class is a drop-in replacement for `BetaNLInterpolator` on the
single-z path: it exposes `__call__(k, M1, M2, z)`,
`interpolate_to_mass_grid(log10M_target, z, k_target)`, `get_z_index`,
and `get_grid_data` so `HaloModel._compute_beta_nl()` does not need to
branch on the source.

Convention matches `emu.BetaNLInterpolator`:
- halo-halo bias at k_lin = 0.02 h/Mpc
- additive force-to-zero (default)
- masses in M_sun/h, k in h/Mpc
"""

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

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

from ..utilsf.measure_beta_nl_xi import (
    beta_nl_k_from_tabulation,
    load_xi_threshold_grid,
)


class NumericalBetaNLInterpolator:
    """
    Interpolator for β^NL(k, M1, M2) at a single redshift built from a
    stitched threshold-ξ cache.

    Parameters
    ----------
    xi_grid_path : str
        Path to the .npz produced by `precompute_xi_tree_grid.py`
        (consumed via `load_xi_threshold_grid`).
    P_lin_func : callable
        k [h/Mpc] -> P_lin(k, z) [(Mpc/h)^3] at the snapshot redshift.
    z : float
        Snapshot redshift of the cache.
    log10M_targets : array-like
        Target log10(M / [M_sun/h]) values that define the mass nodes
        of the interpolator.
    n_k, k_min, k_max : int, float, float
        k-grid for the interpolator (log-spaced).
    eps : float
        Four-corner FD step for β^NL build (must be consistent with the
        precompute --eps when interpreting the cache).
    k_lin : float
        Pivot for halo-halo bias / force-to-zero.
    force_to_zero : str
        'additive' (default), 'multiplicative', 'exponential', 'none'.
    method : str
        interpax interpolation method.
    constant_low, constant_low_limit : see BetaNLInterpolator.
    verbose : bool
    """

    FORCE_TO_ZERO_METHODS = ('none', 'additive', 'multiplicative', 'exponential')

    def __init__(
        self,
        xi_grid_path: str,
        P_lin_func: Callable[[np.ndarray], np.ndarray],
        z: float,
        log10M_targets: Sequence[float],
        n_k: int = 100,
        k_min: float = 1e-2,
        k_max: float = 10.0,
        eps: float = 0.02,
        k_lin: float = 0.02,
        force_to_zero: str = "additive",
        method: str = "linear",
        constant_low: bool = False,
        constant_low_limit: Optional[float] = None,
        verbose: bool = True,
    ):
        if not HAS_JAX or not HAS_INTERPAX:
            raise RuntimeError("JAX and interpax are required for NumericalBetaNLInterpolator")
        if force_to_zero not in self.FORCE_TO_ZERO_METHODS:
            raise ValueError(f"force_to_zero must be one of {self.FORCE_TO_ZERO_METHODS}")

        self.xi_grid_path = xi_grid_path
        self.method = method
        self.eps = eps
        self.k_lin = k_lin
        self.force_to_zero = force_to_zero
        self.verbose = verbose

        self.z_values = np.atleast_1d(np.float64(z))
        self.n_z = 1

        log10M_targets = np.asarray(log10M_targets, dtype=np.float64)
        self.log_M_arr = log10M_targets
        self.M_arr = 10.0 ** log10M_targets
        self.n_mass = len(log10M_targets)
        self.log_M_min = float(log10M_targets.min())
        self.log_M_max = float(log10M_targets.max())

        self.n_k = n_k
        self.k_min = k_min
        self.k_max = k_max
        self.log_k_arr = np.linspace(np.log10(k_min), np.log10(k_max), n_k)
        self.k_arr = 10.0 ** self.log_k_arr

        self.constant_low = constant_low
        self.constant_low_limit = (constant_low_limit if constant_low_limit is not None
                                    else self.log_M_min)

        if verbose:
            print(f"Building numerical β^NL interpolator from {xi_grid_path}")
            print(f"  z = {float(z):.4f}")
            print(f"  k: [{k_min:.2e}, {k_max:.2e}] h/Mpc, n={n_k}")
            print(f"  log10M targets: [{self.log_M_min:.2f}, {self.log_M_max:.2f}], n={self.n_mass}")
            print(f"  eps={eps}, k_lin={k_lin}, force_to_zero='{force_to_zero}'")

        thr_res = load_xi_threshold_grid(xi_grid_path)
        res = beta_nl_k_from_tabulation(
            thr_res,
            log10M_targets=log10M_targets,
            P_lin_func=P_lin_func,
            k_out=self.k_arr,
            eps=eps,
            k_lin=k_lin,
            force_to_zero=force_to_zero,
        )

        beta_kij = np.asarray(res["beta_nl"])  # (K, K, n_k)
        if not np.all(np.isfinite(beta_kij)):
            n_bad = int(np.sum(~np.isfinite(beta_kij)))
            if verbose:
                print(f"  warning: {n_bad} non-finite β^NL entries "
                      f"(of {beta_kij.size}); filling with 0.0")
            beta_kij = np.nan_to_num(beta_kij, nan=0.0, posinf=0.0, neginf=0.0)

        # interpax expects (n_x, n_y, n_z) ordering matching (log_k, log_M1, log_M2)
        beta_grid = np.transpose(beta_kij, (2, 0, 1))
        self.beta_nl_grid = beta_grid
        self.b_hh = np.asarray(res["b_hh"])
        self.P_hh = np.asarray(res["P_hh"])
        self.P_lin = np.asarray(res["P_lin"])

        self.interpolators = {
            0: Interpolator3D(
                x=self.log_k_arr,
                y=self.log_M_arr,
                z=self.log_M_arr,
                f=beta_grid,
                method=method,
                extrap=[[True, True], [True, True], [True, True]],
            )
        }

    def get_z_index(self, z: float) -> int:
        if abs(float(z) - float(self.z_values[0])) > 0.01:
            raise ValueError(
                f"z={z} not in cache (single-z interpolator at z={float(self.z_values[0])})"
            )
        return 0

    def _clamp_log_mass_low(self, log_M: Any) -> Any:
        if self.constant_low:
            return jnp.maximum(log_M, self.constant_low_limit)
        return log_M

    def __call__(self, k: Any, M1: Any, M2: Any, z: float = 0.0):
        iz = self.get_z_index(z)
        k = jnp.atleast_1d(k)
        log_k = jnp.log10(k)
        log_M1 = jnp.full_like(log_k, jnp.log10(M1))
        log_M2 = jnp.full_like(log_k, jnp.log10(M2))
        log_M1 = self._clamp_log_mass_low(log_M1)
        log_M2 = self._clamp_log_mass_low(log_M2)
        return self.interpolators[iz](log_k, log_M1, log_M2)

    def interpolate_to_mass_grid(
        self,
        log10M_target: jnp.ndarray,
        z: float,
        k_target: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        iz = self.get_z_index(z)
        interp = self.interpolators[iz]

        log_k_vals = jnp.log10(k_target) if k_target is not None else jnp.array(self.log_k_arr)
        log10M_target_clamped = self._clamp_log_mass_low(jnp.asarray(log10M_target))
        n_M = len(log10M_target_clamped)

        @jit
        def eval_at_k(log_k_val):
            M1_grid, M2_grid = jnp.meshgrid(log10M_target_clamped, log10M_target_clamped,
                                            indexing='ij')
            k_flat = jnp.full(n_M * n_M, log_k_val)
            return interp(k_flat, M1_grid.ravel(), M2_grid.ravel()).reshape(n_M, n_M)

        return vmap(eval_at_k)(log_k_vals)

    def get_grid_data(self, iz: int = 0) -> Dict[str, np.ndarray]:
        return {
            'log_k_arr': self.log_k_arr,
            'k_arr': self.k_arr,
            'log_M_arr': self.log_M_arr,
            'M_arr': self.M_arr,
            'z': float(self.z_values[0]),
            'beta_nl_grid': self.beta_nl_grid,
            'b_hh': self.b_hh,
            'P_hh': self.P_hh,
            'P_lin': self.P_lin,
            'constant_low': self.constant_low,
        }


__all__ = ['NumericalBetaNLInterpolator']
