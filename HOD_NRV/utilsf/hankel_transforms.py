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
from scipy.interpolate import interp1d

try:
    from fastpt import HT
    HAS_FASTPT = True
except ImportError:
    HAS_FASTPT = False
    warnings.warn(
        "FAST-PT not available. Install with: pip install FAST-PT\n"
        "See: https://github.com/JoeMcEwen/FAST-PT"
    )

# Import DeltaSigmaCalculator and bin averaging utility
try:
    from HOD_NRV.HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
        DeltaSigmaCalculator,
        binavg_2D
    )
    HAS_DELTASIGMA = True
except ImportError:
    HAS_DELTASIGMA = False
    warnings.warn(
        "DeltaSigmaCalculator not available. Traditional method will not work."
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


def Pk_to_wgg_direct(k: np.ndarray, Pk_gg: np.ndarray,
                     r_out, rp_bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    r_out: array
    r_bins : array, optional
        Bin edges for averaging [Mpc/h]. If provided, returns bin-averaged w_gg.
        If None (default), returns point evaluations at transform radii.

    Returns
    -------
    r : array
        Projected radii [Mpc/h or Mpc]
        - If r_bins is None: transform radii
        - If r_bins provided: bin centers (geometric mean)
    wgg : array
        Projected correlation function [Mpc/h or Mpc]
        - If r_bins is None: point evaluations
        - If r_bins provided: bin-averaged values

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> # Point evaluations
    >>> r, wgg = Pk_to_wgg_direct(k, Pk_gg)
    >>> # Bin-averaged
    >>> r_bins = np.logspace(-1, 1.5, 10)
    >>> r_centers, wgg_avg = Pk_to_wgg_direct(k, Pk_gg, r_bins=r_bins)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Direct transform for w_gg
    r, wgg_full = HT.k_to_r(k, Pk_gg, alpha_k=1., beta_r=-1., mu=0., pf=1./(2*np.pi))
    spline_wgg = interp1d(r, wgg_full, kind='cubic', bounds_error=False,
                            fill_value=(wgg_full[0], wgg_full[-1]))

    if rp_bins is not None:
        wgg_avg = binavg_2D(spline_wgg, rp_bins)
        return r_out, wgg_avg
    else:
        return r_out, spline_wgg(r_out)


def Pk_to_DeltaSigma_direct(k: np.ndarray, Pk_gm: np.ndarray,
                             rho_m: float, r_out, 
                             rp_bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    r_out : float 
    r_bins : array, optional
        Bin edges for averaging [Mpc/h]. If provided, returns bin-averaged ΔΣ.
        If None (default), returns point evaluations at transform radii.

    Returns
    -------
    r : array
        Projected radii [Mpc/h or Mpc]
        - If r_bins is None: transform radii
        - If r_bins provided: bin centers (geometric mean)
    DeltaSigma : array
        Surface mass density contrast
        [Msun/h / (Mpc/h)²] if using h-units
        - If r_bins is None: point evaluations
        - If r_bins provided: bin-averaged values

    Notes
    -----
    This is the direct method (Method 2) which computes DeltaSigma
    in a single Hankel transform.

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> rho_m = 2.775e11  # Msun/h / (Mpc/h)^3
    >>> # Point evaluations
    >>> r, ds = Pk_to_DeltaSigma_direct(k, Pk_gm, rho_m)
    >>> # Bin-averaged
    >>> r_bins = np.logspace(-1, 1.5, 10)
    >>> r_centers, ds_avg = Pk_to_DeltaSigma_direct(k, Pk_gm, rho_m, r_bins=r_bins)
    """
    if not HAS_FASTPT:
        raise ImportError("FAST-PT is required. Install with: pip install FAST-PT")

    # Direct IA-type transform
    r, ds_unnormalized = HT.k_to_r(k, Pk_gm, alpha_k=1., beta_r=-1., mu=2., pf=1./(2*np.pi))

    # Rescale by rho_m
    DeltaSigma = ds_unnormalized * rho_m / 1e12 
    spline_ds = interp1d(r, DeltaSigma, kind='cubic', bounds_error=False,
                        fill_value=(DeltaSigma[0], DeltaSigma[-1]))
    if rp_bins is not None:
        # Bin-average using binavg_2D

        ds_avg = binavg_2D(spline_ds, rp_bins)
        return r_out, ds_avg
    else:
        return r_out, spline_ds(r_out)


# ============================================================================
# Traditional DeltaSigma Calculation (Method 1)
# ============================================================================

def Pk_gm_to_DeltaSigma_traditional(k: np.ndarray, Pk_gm: np.ndarray,
                                    rho_m: float,
                                    R_out: np.ndarray,
                                    chi_max: float = 100.0,
                                    rp_bins : np.ndarray = None,
                                    ) -> np.ndarray:
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
        Projected radii or bin edges [Mpc/h or Mpc]
        - If bin_avg=False: radii for point evaluation
        - If bin_avg=True: bin edges for averaging
    rho_m : float
        Mean matter density [Msun/h / (Mpc/h)³]
    chi_max : float, default=100.0
        Maximum line-of-sight distance [Mpc/h or Mpc]
    bin_avg : bool, default=False
        If True, treat R_out as bin edges and return bin-averaged ΔΣ.
        If False, treat R_out as radii and return point evaluations.

    Returns
    -------
    DeltaSigma : array
        Surface mass density contrast [Msun h/pc²]
        - If bin_avg=False: shape matches R_out
        - If bin_avg=True: shape is (len(R_out)-1,)

    See Also
    --------
    Pk_to_DeltaSigma_direct : Direct method (Method 2)

    Examples
    --------
    >>> k = np.logspace(-3, 2, 512)
    >>> R = np.logspace(-1, 1.5, 15)
    >>> rho_m = 2.775e11
    >>> # Point evaluations
    >>> ds = Pk_gm_to_DeltaSigma_traditional(k, Pk_gm, R, rho_m)
    >>> # Bin-averaged
    >>> R_bins = np.logspace(-1, 1.5, 10)
    >>> ds_avg = Pk_gm_to_DeltaSigma_traditional(k, Pk_gm, R_bins, rho_m, bin_avg=True)
    """
    if not HAS_DELTASIGMA:
        raise ImportError("DeltaSigmaCalculator not available. Install HOD_NRV package.")

    # Transform P_gm -> xi_gm using FAST-PT
    r, xi_gm = Pk_to_xi_gm(k, Pk_gm)

    # Use DeltaSigmaCalculator (now computes everything at init)
    calc = DeltaSigmaCalculator(r, xi_gm, rho_m, chi_max=chi_max)

    # Evaluate or average
    if rp_bins is not None:
        DeltaSigma = calc.compute_deltasigma_averaged(rp_bins)
    else:
        DeltaSigma = calc.compute_deltasigma(R_out)

    return R_out, DeltaSigma


# ============================================================================
# Convenience Wrapper
# ============================================================================

def Pk_gm_to_DeltaSigma(k: np.ndarray, Pk_gm: np.ndarray,
                        R_out: np.ndarray, rho_m: float,
                        rp_bins : str = None,
                        method: str = 'direct',
                        chi_max: float = 100.0,
                        bin_avg: bool = False) -> np.ndarray:
    """
    Compute DeltaSigma from P_gm with choice of method.

    Parameters
    ----------
    k : array
        Wavenumbers [h/Mpc or 1/Mpc], must be log-spaced
    Pk_gm : array
        Galaxy-matter power spectrum [(Mpc/h)³ or Mpc³]
    R_out : array
    rho_m : float
        Mean matter density [Msun/h / (Mpc/h)³]
    method : str, default='direct'
        'direct' - Direct Hankel transform (Method 2, faster)
        'traditional' - Via xi_gm integration (Method 1, matches numerical code)
    chi_max : float, default=100.0
        Maximum line-of-sight distance (for traditional method)
    bin_avg : bool, default=False
        If True, treat R_out as bin edges and return bin-averaged ΔΣ.
        If False, treat R_out as radii and return point evaluations.
    rp_bins: Optional

    Returns
    -------
    DeltaSigma : array
        Surface mass density contrast [Msun h/pc²]

    Examples
    --------
    >>> # Direct method (faster), point evaluation
    >>> ds = Pk_gm_to_DeltaSigma(k, Pk_gm, R, rho_m, method='direct')
    >>>
    >>> # Traditional method, bin-averaged
    >>> R_bins = np.logspace(-1, 1.5, 10)
    >>> ds_avg = Pk_gm_to_DeltaSigma(k, Pk_gm, R_bins, rho_m,
    ...                              method='traditional', bin_avg=True)
    """
    if method == 'direct':
        return Pk_to_DeltaSigma_direct(k, Pk_gm, rho_m,R_out,rp_bins=rp_bins)

    elif method == 'traditional':
        return Pk_gm_to_DeltaSigma_traditional(k, Pk_gm, R_out, rho_m,rp_bins=rp_bins,
                                               chi_max=chi_max, bin_avg=bin_avg)

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
