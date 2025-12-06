"""
Hankel Transform Utilities using FAST-PT
=========================================

Wrapper functions for converting power spectra to correlation functions
using FAST-PT's FFTLog implementation.

Key Transforms:
- P_gg(k) -> w_p(r_p)       : Projected galaxy clustering (direct)
- P_gm(k) -> DeltaSigma(R)  : Galaxy-galaxy lensing (two methods available)

Requirements:
    pip install FAST-PT

Author: NRVpy Development Team
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    from fastpt import HT
    HAS_FASTPT = True
except ImportError:
    HAS_FASTPT = False
    warnings.warn(
        "FAST-PT not available. Install with: pip install FAST-PT\n"
        "See: https://github.com/JoeMcEwen/FAST-PT"
    )


# ============================================================================
# Power Spectrum to Correlation Function Transforms
# ============================================================================

def Pk_to_xi_gg(k: np.ndarray, Pk_gg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert galaxy power spectrum P_gg(k) to correlation function xi_gg(r).

    Uses FAST-PT's k_to_r with parameters:
    - alpha_k = 1.5
    - beta_r = -1.5
    - mu = 0.5 (standard cosmology transform)

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gg : array
        Galaxy power spectrum [(Mpc/h)³ or Mpc³]

    Returns
    -------
    r : array
        Radial distances [Mpc/h or Mpc]
    xi_gg : array
        Galaxy correlation function

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> r, xi_gg = Pk_to_xi_gg(k, Pk_gg)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Standard cosmology transform
    r, xi_gg = HT.k_to_r(k, Pk_gg, alpha_k=1.5, beta_r=-1.5, mu=0.5)

    return r, xi_gg


def Pk_to_xi_gm(k: np.ndarray, Pk_gm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert galaxy-matter power spectrum P_gm(k) to correlation function xi_gm(r).

    Uses FAST-PT's k_to_r with standard cosmology parameters:
    - alpha_k = 1.5
    - beta_r = -1.5
    - mu = 0.5

    This is the traditional transform used in Method 1 for DeltaSigma calculation.

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gm : array
        Galaxy-matter power spectrum [(Mpc/h)³ or Mpc³]

    Returns
    -------
    r : array
        Radial distances [Mpc/h or Mpc]
    xi_gm : array
        Galaxy-matter correlation function

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> r, xi_gm = Pk_to_xi_gm(k, Pk_gm)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Standard cosmology transform
    r, xi_gm = HT.k_to_r(k, Pk_gm, alpha_k=1.5, beta_r=-1.5, mu=0.5)

    return r, xi_gm


def Pk_to_wgg_direct(k: np.ndarray, Pk_gg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct transform from P_gg(k) to projected correlation w_gg(r).

    Uses FAST-PT's k_to_r with parameters:
    - alpha_k = 1.0
    - beta_r = -1.0
    - mu = 0.0
    - prefactor = 1/(2π)

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gg : array
        Galaxy power spectrum [(Mpc/h)³ or Mpc³]

    Returns
    -------
    r : array
        Projected radii [Mpc/h or Mpc]
    wgg : array
        Projected correlation function [Mpc/h or Mpc]

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> r, wgg = Pk_to_wgg_direct(k, Pk_gg)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Direct transform for w_gg
    r, wgg_full = HT.k_to_r(k, Pk_gg, alpha_k=1., beta_r=-1., mu=0., pf=1./(2*np.pi))

    return r, wgg_full


def Pk_to_DeltaSigma_direct(k: np.ndarray, Pk_gm: np.ndarray,
                             rho_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct transform from P_gm(k) to DeltaSigma(R) (Method 2).

    Uses FAST-PT's k_to_r with IA-type parameters:
    - alpha_k = 1.0
    - beta_r = -1.0
    - mu = 2.0
    - prefactor = 1/(2π)

    Then rescales by rho_m.

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gm : array
        Galaxy-matter power spectrum [(Mpc/h)³ or Mpc³]
    rho_m : float
        Mean matter density in comoving units
        [Msun/h / (Mpc/h)³] if using h-units

    Returns
    -------
    r : array
        Projected radii [Mpc/h or Mpc]
    DeltaSigma : array
        Surface mass density contrast
        [Msun/h / (Mpc/h)²] if using h-units

    Notes
    -----
    This is the direct method (Method 2) which computes DeltaSigma
    in a single Hankel transform.

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> rho_m = 2.775e11  # Msun/h / (Mpc/h)^3
    >>> r, ds = Pk_to_DeltaSigma_direct(k, Pk_gm, rho_m)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Direct IA-type transform
    r, ds_unnormalized = HT.k_to_r(k, Pk_gm, alpha_k=1., beta_r=-1., mu=2., pf=1./(2*np.pi))

    # Rescale by rho_m
    DeltaSigma = ds_unnormalized * rho_m

    return r, DeltaSigma


# ============================================================================
# Traditional DeltaSigma Calculation (Method 1)
# ============================================================================

def Pk_gm_to_DeltaSigma_traditional(k: np.ndarray, Pk_gm: np.ndarray,
                                    R_out: np.ndarray, rho_m: float,
                                    chi_max: float = 100.0) -> np.ndarray:
    """
    Compute DeltaSigma from P_gm using traditional method (Method 1).

    This first transforms P_gm -> xi_gm using standard cosmology parameters,
    then integrates xi_gm to get DeltaSigma using the DeltaSigmaCalculator.

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gm : array
        Galaxy-matter power spectrum [(Mpc/h)³ or Mpc³]
    R_out : array
        Projected radii [Mpc/h or Mpc]
    rho_m : float
        Mean matter density [Msun/h / (Mpc/h)³]
    chi_max : float
        Maximum line-of-sight distance [Mpc/h or Mpc]

    Returns
    -------
    DeltaSigma : array
        Surface mass density contrast [Msun h/pc²]

    See Also
    --------
    Pk_to_DeltaSigma_direct : Direct method (Method 2)

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> R = np.logspace(-1, 1.5, 15)
    >>> rho_m = 2.775e11
    >>> ds = Pk_gm_to_DeltaSigma_traditional(k, Pk_gm, R, rho_m)
    """
    from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import DeltaSigmaCalculator

    # Transform P_gm -> xi_gm using FAST-PT
    r, xi_gm = Pk_to_xi_gm(k, Pk_gm)

    # Use DeltaSigmaCalculator to compute DeltaSigma from xi_gm
    calc = DeltaSigmaCalculator(r, xi_gm, rho_m)
    calc.compute_sigma(r, chi_max=chi_max)
    DeltaSigma = calc.compute_deltasigma(R_out)

    return DeltaSigma


# ============================================================================
# Convenience Wrapper
# ============================================================================

def Pk_gm_to_DeltaSigma(k: np.ndarray, Pk_gm: np.ndarray,
                        R_out: np.ndarray, rho_m: float,
                        method: str = 'direct',
                        chi_max: float = 100.0) -> np.ndarray:
    """
    Compute DeltaSigma from P_gm with choice of method.

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gm : array
        Galaxy-matter power spectrum [(Mpc/h)³ or Mpc³]
    R_out : array
        Projected radii [Mpc/h or Mpc]
    rho_m : float
        Mean matter density [Msun/h / (Mpc/h)³]
    method : str, default='direct'
        'direct' - Direct Hankel transform (Method 2, faster)
        'traditional' - Via xi_gm integration (Method 1, matches numerical code)
    chi_max : float
        Maximum line-of-sight distance (for traditional method)

    Returns
    -------
    DeltaSigma : array
        Surface mass density contrast [Msun h/pc²]

    Examples
    --------
    >>> # Direct method (faster)
    >>> ds = Pk_gm_to_DeltaSigma(k, Pk_gm, R, rho_m, method='direct')
    >>>
    >>> # Traditional method (matches numerical code)
    >>> ds = Pk_gm_to_DeltaSigma(k, Pk_gm, R, rho_m, method='traditional')
    """
    if method == 'direct':
        # Method 2: Direct transform
        r, ds = Pk_to_DeltaSigma_direct(k, Pk_gm, rho_m)
        # Interpolate to R_out
        ds_out = np.interp(R_out, r, ds)
        # Convert to Msun h/pc²
        return ds_out / 1e12

    elif method == 'traditional':
        # Method 1: Via xi_gm
        return Pk_gm_to_DeltaSigma_traditional(k, Pk_gm, R_out, rho_m, chi_max=chi_max)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'direct' or 'traditional'")


# ============================================================================
# Utility Functions
# ============================================================================

def check_k_spacing(k: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if k array is evenly spaced in log (required for FAST-PT).

    Parameters
    ----------
    k : array
        Wavenumber array
    tolerance : float
        Maximum allowed deviation

    Returns
    -------
    is_log_spaced : bool
        True if evenly spaced in log
    """
    diff = np.diff(np.log(k))
    diff2 = np.diff(diff)
    return np.sum(np.abs(diff2)) < tolerance


def make_log_k_array(kmin: float, kmax: float, npts: int) -> np.ndarray:
    """
    Create log-spaced k array suitable for FAST-PT.

    Parameters
    ----------
    kmin, kmax : float
        k range endpoints
    npts : int
        Number of points (recommend power of 2 for FFT efficiency)

    Returns
    -------
    k : array
        Log-spaced wavenumber array
    """
    return np.logspace(np.log10(kmin), np.log10(kmax), npts)


# ============================================================================
# Testing
# ============================================================================

def test_transforms():
    """Test FAST-PT transforms"""
    if not HAS_FASTPT:
        print("FAST-PT not available. Cannot run tests.")
        return

    print("Testing FAST-PT Hankel Transforms")
    print("=" * 70)

    # Create test k array (log-spaced)
    k = make_log_k_array(1e-3, 100, 512)

    print(f"\nk array: {len(k)} points from {k[0]:.3e} to {k[-1]:.3e}")
    print(f"Log-spaced: {check_k_spacing(k)}")

    # Test power spectrum (power law)
    Pk_test = 1000 * k**(-2)

    print("\nTest 1: P_gg(k) -> xi_gg(r)")
    print("-" * 70)
    r_gg, xi_gg = Pk_to_xi_gg(k, Pk_test)
    print(f"  r range: [{r_gg[0]:.3e}, {r_gg[-1]:.3e}] ({len(r_gg)} points)")
    print(f"  xi range: [{xi_gg.min():.3e}, {xi_gg.max():.3e}]")

    print("\nTest 2: P_gm(k) -> xi_gm(r)")
    print("-" * 70)
    r_gm, xi_gm = Pk_to_xi_gm(k, Pk_test)
    print(f"  r range: [{r_gm[0]:.3e}, {r_gm[-1]:.3e}] ({len(r_gm)} points)")
    print(f"  xi range: [{xi_gm.min():.3e}, {xi_gm.max():.3e}]")

    print("\nTest 3: P_gg(k) -> w_gg(r) [direct]")
    print("-" * 70)
    r_wgg, wgg = Pk_to_wgg_direct(k, Pk_test)
    print(f"  r range: [{r_wgg[0]:.3e}, {r_wgg[-1]:.3e}] ({len(r_wgg)} points)")
    print(f"  w_gg range: [{wgg.min():.3e}, {wgg.max():.3e}]")

    print("\nTest 4: P_gm(k) -> ΔΣ(R) [both methods]")
    print("-" * 70)
    R = np.logspace(-1, 1.5, 15)
    rho_m = 2.775e11  # Msun/h / (Mpc/h)^3

    # Direct method
    ds_direct = Pk_gm_to_DeltaSigma(k, Pk_test, R, rho_m, method='direct')
    print(f"  Direct method:")
    print(f"    ΔΣ range: [{ds_direct.min():.3e}, {ds_direct.max():.3e}] Msun h/pc²")

    # Traditional method
    ds_trad = Pk_gm_to_DeltaSigma(k, Pk_test, R, rho_m, method='traditional')
    print(f"  Traditional method:")
    print(f"    ΔΣ range: [{ds_trad.min():.3e}, {ds_trad.max():.3e}] Msun h/pc²")

    # Compare
    frac_diff = np.abs((ds_direct - ds_trad) / ds_trad)
    print(f"  Fractional difference: {frac_diff.max():.2%}")

    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")


if __name__ == "__main__":
    test_transforms()
