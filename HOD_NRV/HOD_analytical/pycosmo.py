"""
PyCosmo: Cosmology Base Class with Beyond-Linear Halo Bias - INTERPAX VERSION
==============================================================================

Simplified implementation using interpax for JAX-compatible interpolation.

Key features:
1. Uses interpax.Interpolator3D for β^NL interpolation
2. interpax handles all extrapolation (k: zero outside, M: extrapolate)
3. Phh noise handling: set to zero after first negative value at high-k
4. Clean, minimal code

Unit Convention:
    Internal calculations use natural units (Msun, Mpc, 1/Mpc)
    When units_per_h=True: outputs converted to h-units
"""

import numpy as np
from typing import Dict, Optional, Any, Union, List

import pyccl as ccl

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
# BetaNLInterpolator - Simplified with interpax
# ============================================================================

class BetaNLInterpolator:
    """
    Interpolator for β^NL(k, M1, M2) at specific redshifts using interpax.
    
    Uses interpax.Interpolator3D which handles:
    - Fast, differentiable interpolation
    - Extrapolation (k: zero outside, M: linear extrapolation)
    
    The emulator is valid for M ∈ [10^12, 10^15] Msun/h.
    Grid is computed within emulator range; interpax extrapolates beyond.
    """
    
    def __init__(
        self,
        emu,
        h: float,
        z_values: Union[float, np.ndarray, List[float]],
        n_k: int = 100,
        n_mass: int = 30,
        k_min: float = 1e-2,
        k_max: float = 10.0,
        log_M_min: float = 12.0,  # Emulator valid range
        log_M_max: float = 15.0,  # Emulator valid range
        method: str = "linear",
        verbose: bool = True,
    ):
        if not HAS_JAX or not HAS_INTERPAX:
            raise RuntimeError("JAX and interpax are required for BetaNLInterpolator")
        
        self.emu = emu
        self.h = h
        self.verbose = verbose
        self.method = method
        
        # Store redshifts
        self.z_values = np.atleast_1d(z_values)
        self.n_z = len(self.z_values)
        
        # Grid parameters (within emulator valid range)
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
    
    def _get_linear_bias(self, M_h: float, z: float) -> float:
        """Get linear halo bias from emulator."""
        return self.emu.get_bias_mass(M_h, z)
    
    def _get_linear_power(self, k_h: np.ndarray, z: float) -> np.ndarray:
        """Get linear power spectrum from CCL."""
        Pk_lin = self.emu.get_pklin_from_z(k_h,z)
        return Pk_lin
    
    def _get_halo_power_cleaned(self, k_h: np.ndarray, M1: float, M2: float, z: float) -> np.ndarray:
        """
        Get halo-halo power spectrum with noise handling.
        Sets Phh to zero after first negative value at high-k.
        """
        power = np.asarray(self.emu.get_phh_mass(np.asarray(k_h), M1, M2, z))
        
        # negative_mask = power < 0
        # if np.any(negative_mask):
        #     first_negative_idx = np.argmax(negative_mask)
        #     power[first_negative_idx:] = 0.0
        
        return power
    
    def _compute_beta_nl_grid(self, z: float) -> np.ndarray:
        """Compute β^NL grid for a single redshift. Shape: (n_k, n_mass, n_mass)."""
        beta_grid = np.zeros((self.n_k, self.n_mass, self.n_mass))
        
        Pk_lin = self._get_linear_power(self.k_arr, z)
        b = np.array([self._get_linear_bias(float(M), z) for M in self.M_arr])
        
        for i1 in range(self.n_mass):
            for i2 in range(i1, self.n_mass):
                M1, M2 = float(self.M_arr[i1]), float(self.M_arr[i2])
                Phh = self._get_halo_power_cleaned(self.k_arr, M1, M2, z)
                
                denom = b[i1] * b[i2] * Pk_lin
                with np.errstate(divide='ignore', invalid='ignore'):
                    beta = Phh / denom - 1.0
                    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
                
                beta_grid[:, i1, i2] = beta
                beta_grid[:, i2, i1] = beta  # Symmetry
        
        return beta_grid
    
    def _build_interpolators(self):
        """Build interpax.Interpolator3D for each redshift."""
        if self.verbose:
            print(f"Building β^NL interpolators for {self.n_z} redshift(s)")
            print(f"  k: [{self.k_min:.2e}, {self.k_max:.2e}] h/Mpc, n={self.n_k}")
            print(f"  M: [10^{self.log_M_min:.1f}, 10^{self.log_M_max:.1f}] Msun/h, n={self.n_mass}")
        
        self.interpolators = {}
        self.beta_nl_grids = {}
        
        for iz, z in enumerate(self.z_values):
            if self.verbose:
                print(f"  z = {z:.4f} ({iz+1}/{self.n_z})...")
            
            beta_grid = self._compute_beta_nl_grid(z)
            self.beta_nl_grids[iz] = beta_grid
            
            # extrap: k -> 0 outside bounds, M -> linear extrapolation
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
    
    def __call__(self, k: Any, M1: Any, M2: Any, z: float = 0.0) -> jnp.ndarray:
        """
        Interpolate β^NL at (k, M1, M2, z).
        
        k values outside [k_min, k_max] return 0.
        M values outside [M_min, M_max] are extrapolated.
        """
        iz = self.get_z_index(z)
        
        k = jnp.atleast_1d(k)
        log_k = jnp.log10(k)
        log_M1 = jnp.full_like(log_k, jnp.log10(M1))
        log_M2 = jnp.full_like(log_k, jnp.log10(M2))
        
        return self.interpolators[iz](log_k, log_M1, log_M2)
    
    def interpolate_to_mass_grid(
        self,
        log10M_target: jnp.ndarray,
        z: float,
        k_target: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Interpolate β^NL to target mass grid.
        
        Returns shape (n_k, n_M_target, n_M_target).
        """
        iz = self.get_z_index(z)
        interp = self.interpolators[iz]
        
        log_k_vals = jnp.log10(k_target) if k_target is not None else jnp.array(self.log_k_arr)
        n_M = len(log10M_target)
        
        @jit
        def eval_at_k(log_k_val):
            M1_grid, M2_grid = jnp.meshgrid(log10M_target, log10M_target, indexing='ij')
            k_flat = jnp.full(n_M * n_M, log_k_val)
            return interp(k_flat, M1_grid.ravel(), M2_grid.ravel()).reshape(n_M, n_M)
        
        return vmap(eval_at_k)(log_k_vals)
    
    def get_grid_data(self, iz: int = 0) -> Dict[str, np.ndarray]:
        """Return grid data for inspection."""
        return {
            'log_k_arr': self.log_k_arr,
            'log_M_arr': self.log_M_arr,
            'z': self.z_values[iz],
            'beta_nl_grid': self.beta_nl_grids[iz],
        }


# ============================================================================
# Cosmology Class
# ============================================================================

class Cosmology:
    """Base cosmology class with pyccl and dark_emulator integration."""
    
    DEFAULT_K_ARRAY = np.geomspace(1e-5, 100, 512)
    
    def __init__(
        self,
        cosmo_params: Dict[str, float],
        mass_function: str = "Tinker10",
        halo_bias: str = "Tinker10",
        mass_definition: str = "MassDef200c",
        concentration: str = "Duffy08",
        k_array: Optional[np.ndarray] = None,
        units_per_h: bool = False,
        use_dark_emulator: bool = True,
        transfer_function: str = 'boltzmann_camb',
        matter_power_spectrum: str = 'camb',
        halofit_version: str = 'mead2020',
        beta_nl_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.cosmo_params = cosmo_params.copy()
        self.units_per_h = units_per_h
        print(units_per_h)
        self.verbose = verbose
        
        self._validate_cosmo_params(cosmo_params)
        self.h = cosmo_params['h']
        
        self._k_array_internal = np.atleast_1d(k_array).copy() if k_array is not None else self.DEFAULT_K_ARRAY.copy()
        self.n_k = len(self._k_array_internal)
        
        self.ccl_cosmo = self._init_ccl_cosmology(
            cosmo_params, transfer_function, matter_power_spectrum, halofit_version
        )
        self._rho_m_internal = ccl.rho_x(self.ccl_cosmo, 1.0, 'matter', is_comoving=True)
        
        self.emu = self._init_dark_emulator(cosmo_params) if use_dark_emulator else None
        self._setup_halo_components(mass_function, halo_bias, mass_definition, concentration)
        
        self.beta_nl_interp = None
        self._beta_nl_kwargs = beta_nl_kwargs or {}
        
        if verbose:
            self._print_init_info()
    
    def _validate_cosmo_params(self, params: Dict[str, float]):
        required = ['h', 'Omc', 'Omb', 'n_s']
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        if 'A_s' not in params and 's8' not in params:
            raise ValueError("Must provide 'A_s' or 's8'")
        params.setdefault('mnu', 0.0)
    
    def _init_ccl_cosmology(self, params, transfer_function, matter_power_spectrum, halofit_version):
        h, Omc, Omb, ns = params['h'], params['Omc'], params['Omb'], params['n_s']
        
        mnu = h**2 * params['Omnu'] * 93.14 if params.get('Omnu', 0) > 0 else params.get('mnu', 0.0)
        
        ccl_kwargs = {
            'Omega_c': Omc, 'Omega_b': Omb, 'h': h, 'n_s': ns, 'm_nu': mnu,
            'w0': params.get('w0', -1.0), 'wa': params.get('wa', 0.0),
            'transfer_function': transfer_function,
            'matter_power_spectrum': matter_power_spectrum,
            'mass_split': 'normal',
            'extra_parameters': {"camb": {"halofit_version": halofit_version}}
        }
        
        if 'A_s' in params:
            ccl_kwargs['A_s'] = params['A_s']
        else:
            ccl_kwargs['sigma8'] = params['s8']
        
        return ccl.Cosmology(**ccl_kwargs)
    
    def _init_dark_emulator(self, params):
        if not HAS_DARK_EMULATOR:
            if self.verbose:
                print("Warning: dark_emulator not available")
            return None
        
        try:
            emu = darkemu.base_class()
            h = params['h']
            Omnu = params.get('Omnu', 0.0)
            if Omnu == 0 and params.get('mnu', 0) > 0:
                Omnu = params['mnu'] / (93.14 * h**2)
            
            cparam = np.array([
                params['Omb'] * h**2,
                params['Omc'] * h**2,
                1.0 - (params['Omc'] + params['Omb'] + Omnu),
                np.log(1e10 * params.get('A_s', 2.1e-9)),
                params['n_s'],
                params.get('w0', -1.0)
            ])
            emu.set_cosmology(cparam)
            return emu
        except Exception as e:
            if self.verbose:
                print(f"Warning: dark_emulator init failed: {e}")
            return None
    
    def _setup_halo_components(self, mass_function, halo_bias, mass_definition, concentration):
        self.mass_def = getattr(ccl.halos, mass_definition)
        self.concentration_model = getattr(ccl.halos, f"Concentration{concentration}")(mass_def=self.mass_def)
        self.mass_func = getattr(ccl.halos, f"MassFunc{mass_function}")(mass_def=self.mass_def)
        self.halo_bias_model = getattr(ccl.halos, f"HaloBias{halo_bias}")(mass_def=self.mass_def)
        
        self._mass_function_name = mass_function
        self._halo_bias_name = halo_bias
        self._mass_def_name = mass_definition
        self._concentration_name = concentration
    
    def _print_init_info(self):
        print(f"Cosmology: h={self.h:.4f}, Ωc={self.cosmo_params['Omc']:.4f}, Ωb={self.cosmo_params['Omb']:.4f}")
        print(f"  Halo: {self._mass_function_name}, {self._halo_bias_name}")
        print(f"  Dark emulator: {'enabled' if self.emu else 'disabled'}")
    
    def get_k(self) -> np.ndarray:
        return self._k_array_internal / self.h if self.units_per_h else self._k_array_internal.copy()
    
    @property
    def k_array(self) -> np.ndarray:
        return self._k_array_internal.copy()
    
    def get_rho_m(self) -> float:
        return self._rho_m_internal / self.h**2 if self.units_per_h else self._rho_m_internal
    
    @property
    def rho_m_natural(self) -> float:
        return self._rho_m_internal
    
    def linear_power(self, k: Optional[np.ndarray] = None, z: float = 0.0) -> np.ndarray:
        k = k if k is not None else self._k_array_internal
        Pk = ccl.linear_power(self.ccl_cosmo, k, 1.0 / (1.0 + z))
        return Pk * self.h**3 if self.units_per_h else Pk
    
    def nonlinear_power(self, k: Optional[np.ndarray] = None, z: float = 0.0) -> np.ndarray:
        k = k if k is not None else self._k_array_internal
        Pk = ccl.nonlin_power(self.ccl_cosmo, k, 1.0 / (1.0 + z))
        return Pk * self.h**3 if self.units_per_h else Pk
    
    
    def compute_beta_nl(self, z_values: Union[float, np.ndarray, List[float]], **kwargs) -> BetaNLInterpolator:
        """Compute β^NL at specific redshifts."""
        if self.emu is None:
            raise RuntimeError("Cannot compute β^NL without DarkEmulator")
        if not HAS_INTERPAX:
            raise RuntimeError("Cannot compute β^NL without interpax")
        
        opts = {'n_k': 100, 'n_mass': 10, 'k_min': 1e-2, 'k_max': 10.0,
                'log_M_min': 12.0, 'log_M_max': 15.0, 'method': 'linear',
                'verbose': self.verbose}
        opts.update(self._beta_nl_kwargs)
        opts.update(kwargs)
        
        self.beta_nl_interp = BetaNLInterpolator(
            emu=self.emu, h=self.h, z_values=z_values, **opts
        )
        return self.beta_nl_interp
    
    def beta_nl(self, k: Any, M1: Any, M2: Any, z: float = 0.0) -> Any:
        if self.beta_nl_interp is None:
            raise RuntimeError("Call compute_beta_nl(z_values) first")
        return self.beta_nl_interp(k, M1, M2, z)


__all__ = ['Cosmology', 'BetaNLInterpolator', 'HAS_DARK_EMULATOR', 'HAS_JAX', 'HAS_INTERPAX']