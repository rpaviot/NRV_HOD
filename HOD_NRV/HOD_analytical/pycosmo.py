"""
PyCosmo: Cosmology Base Class with Beyond-Linear Halo Bias - INTERPAX VERSION
==============================================================================

Implementation with:
1. Linear bias extraction: 'traditional' (emulator bias function) or 'halo-halo' (default)
2. force_to_zero corrections: 'none', 'additive' (default), 'multiplicative', 'exponential'
3. Uses interpax.Interpolator3D for fast, differentiable β^NL interpolation

The beyond-linear halo bias β^NL is defined as:
    P_hh(k, M1, M2) = b(M1) * b(M2) * P_lin(k) * [1 + β^NL(k, M1, M2)]

where β^NL → 0 as k → 0 (linear regime).

Unit Convention:
    When units_per_h=False (default):
        - Input k_array is in 1/Mpc (natural units)
        - Outputs are in natural units (Msun, Mpc, 1/Mpc)
    When units_per_h=True:
        - Input k_array is in h/Mpc
        - Outputs are in h-units (Msun/h, Mpc/h, h/Mpc)

    Internally, pyccl uses natural units (1/Mpc), while dark_emulator uses h-units (h/Mpc).
"""

import numpy as np
from typing import Dict, Optional, Any, Union, List

import pyccl as ccl

from .emu import (
    BetaNLInterpolator, HAS_DARK_EMULATOR, HAS_INTERPAX, HAS_JAX, darkemu
)
from .emu_numerical import NumericalBetaNLInterpolator


# ============================================================================
# Cosmology Class
# ============================================================================

class Cosmology:
    """
    Base cosmology class with pyccl and dark_emulator integration.
    
    Parameters
    ----------
    cosmo_params : dict
        Cosmological parameters. Required: 'h', 'Omc', 'Omb', 'n_s'.
        Must also have either 'A_s' or 's8'.
        Optional: 'mnu', 'Omnu', 'w0', 'wa'.
    mass_function : str
        Mass function model (default: 'Tinker10')
    halo_bias : str
        Halo bias model (default: 'Tinker10')
    mass_definition : str
        Mass definition (default: 'MassDef200c')
    concentration : str
        Concentration model (default: 'Duffy08')
    k_array : array-like, optional
        Custom k array for power spectrum calculations.
        If units_per_h=True, this is in h/Mpc.
        If units_per_h=False, this is in 1/Mpc.
    units_per_h : bool
        If True, input k_array is in h/Mpc and outputs are in h-units.
        If False (default), k_array is in 1/Mpc and outputs are in natural units.
    use_dark_emulator : bool
        Enable Dark Emulator (default: True)
    transfer_function : str
        Transfer function for CCL (default: 'boltzmann_camb')
    matter_power_spectrum : str
        Matter power spectrum for CCL (default: 'camb')
    halofit_version : str
        Halofit version (default: 'mead2020')
    beta_nl_kwargs : dict, optional
        Default kwargs for compute_beta_nl()
    verbose : bool
        Print initialization info (default: True)
    
    Notes
    -----
    Unit conventions:
    - pyccl uses natural units internally (k in 1/Mpc, P(k) in Mpc³)
    - dark_emulator uses h-units (k in h/Mpc, M in Msun/h, P(k) in (Mpc/h)³)
    
    When units_per_h=True:
        - Input k_array is interpreted as h/Mpc
        - Internal k_array for pyccl is k_input * h (converting to 1/Mpc)
        - Outputs are in h-units
        - This is convenient since dark_emulator uses h/Mpc natively
    
    When units_per_h=False:
        - Input k_array is interpreted as 1/Mpc (natural units)
        - Internal k_array for pyccl is k_input directly
        - Outputs are in natural units
    """
    
    # Default k array in h/Mpc (for units_per_h=True) or 1/Mpc (for units_per_h=False)
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
        beta_nl_source: str = 'emulator',
        verbose: bool = True,
    ):
        self.cosmo_params = cosmo_params.copy()
        self.units_per_h = units_per_h
        self.verbose = verbose
        if beta_nl_source not in ('emulator', 'numerical'):
            raise ValueError(
                f"beta_nl_source must be 'emulator' or 'numerical', got {beta_nl_source!r}"
            )
        self.beta_nl_source = beta_nl_source
        
        self._validate_cosmo_params(cosmo_params)
        self.h = cosmo_params['h']
        
        # Store the user-provided k array (in user's units)
        k_input = np.atleast_1d(k_array).copy() if k_array is not None else self.DEFAULT_K_ARRAY.copy()
        
        # Internal k array for pyccl (always in 1/Mpc, natural units)
        # k array in h/Mpc for dark_emulator
        if units_per_h:
            # User provided k in h/Mpc
            self._k_array_h = k_input  # h/Mpc (for dark_emulator)
            self._k_array_internal = k_input * self.h  # 1/Mpc (for pyccl)
        else:
            # User provided k in 1/Mpc (natural units)
            self._k_array_internal = k_input  # 1/Mpc (for pyccl)
            self._k_array_h = k_input / self.h  # h/Mpc (for dark_emulator)
        
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

        if mass_function=="Nishimichi19":
            self.mass_func = getattr(ccl.halos, f"MassFunc{mass_function}")(mass_def=self.mass_def,extrapolate=True)
        else:
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
        print(f"  units_per_h: {self.units_per_h}")
        if self.units_per_h:
            print(f"  k_array range: [{self._k_array_h[0]:.2e}, {self._k_array_h[-1]:.2e}] h/Mpc")
        else:
            print(f"  k_array range: [{self._k_array_internal[0]:.2e}, {self._k_array_internal[-1]:.2e}] 1/Mpc")
    
    def get_k(self) -> np.ndarray:
        """
        Get k array in user's preferred units.
        
        Returns
        -------
        np.ndarray
            k array in h/Mpc if units_per_h=True, else in 1/Mpc
        """
        if self.units_per_h:
            return self._k_array_h.copy()
        else:
            return self._k_array_internal.copy()
    
    def get_k_h(self) -> np.ndarray:
        """
        Get k array in h/Mpc (for dark_emulator).
        
        Returns
        -------
        np.ndarray
            k array in h/Mpc
        """
        return self._k_array_h.copy()
    
    @property
    def k_array(self) -> np.ndarray:
        """
        Internal k array in natural units (1/Mpc) for pyccl.
        """
        return self._k_array_internal.copy()
    
    @property
    def k_array_h(self) -> np.ndarray:
        """
        k array in h/Mpc for dark_emulator.
        """
        return self._k_array_h.copy()
    
    def Hz(self, z: float) -> float:
        """Hubble parameter H(z) in km/s/Mpc."""
        a = 1.0 / (1.0 + z)
        return ccl.h_over_h0(self.ccl_cosmo, a) * self.h * 100.0

    def get_rho_m(self) -> float:
        """
        Get matter density in user's preferred units.
        
        Returns
        -------
        float
            Matter density in Msun/h / (Mpc/h)³ if units_per_h=True,
            else in Msun/Mpc³
        """
        if self.units_per_h:
            return self._rho_m_internal / self.h**2
        else:
            return self._rho_m_internal
    
    @property
    def rho_m_natural(self) -> float:
        """Matter density in natural units (Msun/Mpc³)."""
        return self._rho_m_internal
    
    def linear_power(self, k: Optional[np.ndarray] = None, z: float = 0.0) -> np.ndarray:
        """
        Get linear matter power spectrum.
        
        Parameters
        ----------
        k : array-like, optional
            Wavenumbers. If units_per_h=True, expected in h/Mpc.
            If units_per_h=False, expected in 1/Mpc.
            Default: internal k_array
        z : float
            Redshift
            
        Returns
        -------
        np.ndarray
            P_lin(k, z) in (Mpc/h)³ if units_per_h=True, else in Mpc³
        """
        if k is None:
            k_ccl = self._k_array_internal
        else:
            # Convert input k to pyccl units (1/Mpc)
            k_ccl = np.asarray(k) * self.h if self.units_per_h else np.asarray(k)
        
        # pyccl returns P(k) in Mpc³
        Pk = ccl.linear_power(self.ccl_cosmo, k_ccl, 1.0 / (1.0 + z))
        
        # Convert to h-units if requested: P [Mpc³] -> P [(Mpc/h)³] = P * h³
        if self.units_per_h:
            return Pk * self.h**3
        else:
            return Pk
    
    def nonlinear_power(self, k: Optional[np.ndarray] = None, z: float = 0.0) -> np.ndarray:
        """
        Get nonlinear matter power spectrum.
        
        Parameters
        ----------
        k : array-like, optional
            Wavenumbers. If units_per_h=True, expected in h/Mpc.
            If units_per_h=False, expected in 1/Mpc.
            Default: internal k_array
        z : float
            Redshift
            
        Returns
        -------
        np.ndarray
            P_nl(k, z) in (Mpc/h)³ if units_per_h=True, else in Mpc³
        """
        if k is None:
            k_ccl = self._k_array_internal
        else:
            # Convert input k to pyccl units (1/Mpc)
            k_ccl = np.asarray(k) * self.h if self.units_per_h else np.asarray(k)
        
        # pyccl returns P(k) in Mpc³
        Pk = ccl.nonlin_power(self.ccl_cosmo, k_ccl, 1.0 / (1.0 + z))
        
        # Convert to h-units if requested
        if self.units_per_h:
            return Pk * self.h**3
        else:
            return Pk
    
    def compute_beta_nl(
        self, 
        z_values: Union[float, np.ndarray, List[float]], 
        bias_method: str = "halo-halo",
        force_to_zero: str = "additive",
        k_lin: float = 0.02,
        constant_low: bool = False,
        **kwargs
    ) -> BetaNLInterpolator:
        """
        Compute β^NL interpolator at specific redshifts.
        
        Parameters
        ----------
        z_values : float or array-like
            Redshift(s) at which to compute β^NL
        bias_method : str
            Method for extracting linear halo bias:
            - 'halo-halo' (default): b(M) = sqrt(P_hh(k_lin)/P_lin(k_lin))
            - 'traditional': Use emulator's bias function
        force_to_zero : str
            Method for correcting large-scale behavior:
            - 'additive' (default): Subtract offset at k_lin
            - 'multiplicative': Divide by (1 + offset)
            - 'exponential': Multiply by (1 - exp(-(k/k_lin)²))
            - 'none': No correction
        k_lin : float
            Scale defining "linear" regime in h/Mpc (default: 0.02)
        constant_low : bool
            If True, use constant extrapolation at low mass boundaries.
            When a requested mass is below log_M_min (default: 10^12 Msun/h),
            the interpolator will return the value at log_M_min instead of
            extrapolating. (default: False)
        **kwargs
            Additional arguments passed to BetaNLInterpolator:
            n_k, n_mass, k_min, k_max, log_M_min, log_M_max, method
            
        Returns
        -------
        BetaNLInterpolator
            Interpolator for β^NL(k, M1, M2, z)
            Note: BetaNLInterpolator always uses h-units (k in h/Mpc, M in Msun/h)
        """
        if not HAS_INTERPAX:
            raise RuntimeError("Cannot compute β^NL without interpax")

        if self.beta_nl_source == 'numerical':
            return self._compute_beta_nl_numerical(
                z_values, force_to_zero=force_to_zero, k_lin=k_lin,
                constant_low=constant_low, **kwargs
            )

        if self.emu is None:
            raise RuntimeError("Cannot compute β^NL without DarkEmulator")

        # Default options
        opts = {
            'n_k': 100,
            'n_mass': 30,
            'k_min': 1e-2,
            'k_max': 10.0,
            'log_M_min': 12.0,
            'log_M_max': 15.0,
            'method': 'linear',
            'verbose': self.verbose,
            'bias_method': bias_method,
            'force_to_zero': force_to_zero,
            'k_lin': k_lin,
            'constant_low': constant_low,
        }
        # Override with instance defaults
        opts.update(self._beta_nl_kwargs)
        # Override with call arguments
        opts.update(kwargs)

        self.beta_nl_interp = BetaNLInterpolator(
            emu=self.emu, h=self.h, z_values=z_values, **opts
        )
        return self.beta_nl_interp

    def _compute_beta_nl_numerical(
        self,
        z_values,
        force_to_zero: str = "additive",
        k_lin: float = 0.02,
        constant_low: bool = False,
        **kwargs,
    ) -> NumericalBetaNLInterpolator:
        """Build the numerical (sim-measured) β^NL interpolator at one z."""
        z_arr = np.atleast_1d(z_values)
        if len(z_arr) != 1:
            raise ValueError(
                "Numerical β^NL source supports a single redshift per cache; "
                f"got {len(z_arr)} z values."
            )
        z = float(z_arr[0])

        opts = {
            'n_k': 100,
            'k_min': 1e-2,
            'k_max': 10.0,
            'eps': 0.02,
            'method': 'linear',
            'verbose': self.verbose,
            'force_to_zero': force_to_zero,
            'k_lin': k_lin,
            'constant_low': constant_low,
        }
        opts.update(self._beta_nl_kwargs)
        opts.update(kwargs)

        try:
            xi_grid_path = opts.pop('xi_grid_path')
            log10M_targets = opts.pop('log10M_targets')
        except KeyError as exc:
            raise ValueError(
                "beta_nl_source='numerical' requires 'xi_grid_path' and "
                "'log10M_targets' in beta_nl_kwargs"
            ) from exc

        def P_lin_func(k):
            return self.linear_power(np.asarray(k), z=z)

        self.beta_nl_interp = NumericalBetaNLInterpolator(
            xi_grid_path=xi_grid_path,
            P_lin_func=P_lin_func,
            z=z,
            log10M_targets=log10M_targets,
            **opts,
        )
        return self.beta_nl_interp
    
    def beta_nl(self, k: Any, M1: Any, M2: Any, z: float = 0.0) -> Any:
        """
        Get β^NL at (k, M1, M2, z).
        
        Must call compute_beta_nl() first.
        
        Parameters
        ----------
        k : array-like
            Wavenumber(s) in h/Mpc (BetaNLInterpolator always uses h-units)
        M1, M2 : float
            Halo masses in Msun/h
        z : float
            Redshift
            
        Returns
        -------
        array-like
            β^NL values
            
        Notes
        -----
        If the interpolator was created with constant_low=True, masses below
        the minimum grid mass (default: 10^12 Msun/h) will be clamped to the
        minimum, providing constant extrapolation at low masses.
        """
        if self.beta_nl_interp is None:
            raise RuntimeError("Call compute_beta_nl(z_values) first")
        return self.beta_nl_interp(k, M1, M2, z)


__all__ = [
    'Cosmology',
]