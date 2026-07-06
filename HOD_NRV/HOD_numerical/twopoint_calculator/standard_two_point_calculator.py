"""
Two-Point Correlation Functions and Galaxy-Galaxy Lensing for NRVpy HOD Code

For production use with HOD parameter sampling, consider using the optimized
OptimizedDeltaSigmaCalculator from the twopoint_calculator module, which provides
significant speedup by precomputing ΔΣ at halo centers.

See: HOD_NRV.twopoint_calculator.OptimizedDeltaSigmaCalculator

This module provides functions for computing two-point correlation functions and
galaxy-galaxy lensing signals from populated galaxy catalogs. It includes both
clustering statistics and weak lensing measurements.

Author: NRVpy Development Team
"""

from pycorr import TwoPointCorrelationFunction
from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration
import numpy as np
from scipy.interpolate import interp1d
import multiprocessing
from typing import Tuple, Optional
import jax.numpy as jnp

cpu_count = multiprocessing.cpu_count()

def compute_corr(mode: str,
                catalog1: jnp.ndarray,
                bins1: np.ndarray,
                catalog2: Optional[jnp.ndarray] = None,
                bins2: Optional[np.ndarray] = None,
                boxsize: Optional[float] = None,
                los: str = 'z',
                output: str = 'auto',
                weights1: Optional[np.ndarray] = None,
                weights2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-point correlation functions using the Natural estimator.
    
    This function computes correlation functions using the Natural estimator
    (DD/RR - 1) with RR computed analytically for periodic boundary conditions.
    Supports various binning modes and output formats.
    
    Parameters
    ----------
    mode : str
        Correlation function binning mode:
        - 's': Isotropic 3D correlation function ξ(s)
        - 'smu': 2D correlation function ξ(s,μ) with μ = cos(θ_los)
        - 'rppi': Projected correlation function ξ(rp,π)
    catalog1 : jnp.ndarray, shape (N1, 3)
        Positions of first catalog in Cartesian coordinates [Mpc/h]
    bins1 : np.ndarray
        Primary bin edges. For 's' mode: separation bins [Mpc/h].
        For 'rppi' mode: projected separation bins rp [Mpc/h].
    catalog2 : jnp.ndarray, optional, shape (N2, 3)
        Positions of second catalog for cross-correlation [Mpc/h].
        If None, computes auto-correlation of catalog1.
    bins2 : np.ndarray, optional
        Secondary bin edges:
        - For 'smu' mode: μ bins (typically [-1, 1])
        - For 'rppi' mode: π bins [Mpc/h]
    boxsize : float, optional
        Simulation box size for periodic boundary conditions [Mpc/h].
        Required for natural estimator with analytic randoms.
    los : str, default='z'
        Line-of-sight direction ('x', 'y', or 'z')
    output : str, default='auto'
        Output format:
        - 'auto': Raw correlation function
        - 'multipoles': Legendre multipoles ξ_ℓ(s) 
        - 'wp': Projected correlation function wp(rp)
        
    Returns
    -------
    r : np.ndarray
        Bin centers or separation values
    xi : np.ndarray
        Correlation function values
        
    Raises
    ------
    ValueError
        If boxsize is not provided (required for natural estimator)
        
    Examples
    --------
    >>> # 3D isotropic correlation function
    >>> s_bins = np.logspace(-1, 2, 20)
    >>> s, xi_s = compute_corr('s', galaxy_pos, s_bins, boxsize=1000.0)
    >>> 
    >>> # Projected correlation function
    >>> rp_bins = np.logspace(-1, 1.5, 15)
    >>> pi_bins = np.linspace(-50, 50, 21)
    >>> rp, wp = compute_corr('rppi', galaxy_pos, rp_bins, bins2=pi_bins, 
    ...                      boxsize=1000.0, output='wp')
    >>> 
    >>> # Cross-correlation
    >>> s, xi_cross = compute_corr('s', galaxies, s_bins, catalog2=dark_matter,
    ...                          boxsize=1000.0)
    
    Notes
    -----
    This function uses the pycorr library for efficient correlation function
    computation. The Natural estimator is computed as DD/RR - 1, where DD
    is the data-data pair count and RR is computed analytically for periodic
    boundary conditions.
    
    For galaxy clustering measurements, use mode='s' or mode='smu'.
    For galaxy-galaxy lensing, use mode='s' with cross-correlation.
    
    References
    ----------
    .. [1] Landy & Szalay (1993), ApJ 412, 64
    .. [2] Peebles (1980), "The Large-Scale Structure of the Universe"
    .. [3] Hamilton (1993), ApJ 417, 19
    """
    # Determine auto vs. cross-correlation
    is_cross = catalog2 is not None

    if boxsize is None:
        raise ValueError("Boxsize value need to be pass for natural estimator calculation")

    # Handle 1D vs. 2D binning
    if bins2 is not None:
        edges = (bins1, bins2)
    else:
        edges = bins1


    # Construct TwoPointCorrelationFunction. Optional per-object weights
    # (e.g. particle mass for a mass-weighted matter field in hydro lensing);
    # pycorr normalises the natural estimator by the summed weights.
    corr = TwoPointCorrelationFunction(
        mode=mode,
        edges=edges,
        data_positions1=catalog1.T,
        data_positions2=catalog2.T if is_cross else None,
        data_weights1=weights1,
        data_weights2=weights2 if is_cross else None,
        boxsize=boxsize,
        compute_sepsavg=False,
        los=los,
        nthreads = cpu_count
    )

    r, xi = corr.get_corr(return_sep=True, mode=output)

    return r,xi


def binavg_2D(spline, r_bins):
    """
    Average a 2D spline over radial bins using Gauss-Legendre integration.

    Computes the weighted average: ⟨f⟩ = (2/Δr²) ∫ f(r) r dr for each bin.
    This function is useful for averaging any radially-dependent quantity
    over annular bins.

    Parameters
    ----------
    spline : callable
        Interpolated function to average (e.g., scipy.interpolate.interp1d).
        Must accept array-like input and return array-like output.
    r_bins : array_like
        Bin edges for averaging [Mpc/h], shape (N+1,)

    Returns
    -------
    f_averaged : ndarray
        Average values of the spline in each bin, shape (N,)

    Examples
    --------
    >>> from scipy.interpolate import interp1d
    >>> r = np.logspace(-1, 2, 50)
    >>> f = r**(-2)  # Some radial function
    >>> spline_f = interp1d(r, f, kind='cubic')
    >>> r_bins = np.logspace(-1, 1.5, 10)
    >>> f_avg = binavg_2D(spline_f, r_bins)

    Notes
    -----
    Uses 200-point Gauss-Legendre quadrature for accurate integration.
    The integration is performed in parallel over all bins via the
    gauss_legendre_integration function.
    """
    def integrand(r):
        return spline(r) * r

    # Compute Δr² for each bin
    diff_sq = np.diff(r_bins**2)

    # Integrate f(r) * r over each bin and normalize
    f_averaged = 2 * gauss_legendre_integration(
        integrand, r_bins[:-1], r_bins[1:]
    ) / diff_sq

    return f_averaged


class DeltaSigmaCalculator:
    """
    Calculator for galaxy-galaxy lensing surface density contrast ΔΣ(rp).

    This class computes the projected surface mass density contrast around
    galaxies by integrating the galaxy-matter correlation function. The
    surface density and ΔΣ are computed at initialization time, creating
    spline interpolations that can be efficiently evaluated at any radii
    or averaged over any binning.

    Parameters
    ----------
    rr : np.ndarray
        Separation bins used for correlation function computation [Mpc/h]
    xi_gm : np.ndarray
        Galaxy-matter correlation function ξ_gm(r) values
    RHO_M : float
        Mean matter density in units of [Msun/h / (Mpc/h)^3]
    chi_max : float, optional
        Maximum line-of-sight distance for integration [Mpc/h].
        Default: 120

    Attributes
    ----------
    spline_xigm : scipy.interpolate.interp1d
        Spline interpolation of the correlation function
    SIGMA : np.ndarray
        Surface density values computed at rr [ρ_m units]
    spline_SIGMA : scipy.interpolate.interp1d
        Spline interpolation of surface density
    DeltaSigma : np.ndarray
        Surface density contrast computed at rr [h M☉/pc²]
    spline_DeltaSigma : scipy.interpolate.interp1d
        Spline interpolation of ΔΣ (computed at initialization)

    Examples
    --------
    >>> # Compute galaxy-matter correlation function first
    >>> r, xi_gm = compute_corr('s', galaxy_pos, r_bins, catalog2=matter_pos,
    ...                        boxsize=1000.0)
    >>>
    >>> # Initialize calculator (ΔΣ computed automatically)
    >>> rho_m = 8.6e10  # Msun/h / (Mpc/h)^3
    >>> calc = DeltaSigmaCalculator(r, xi_gm, rho_m)
    >>>
    >>> # Evaluate at specific radii
    >>> rp = np.logspace(-1, 1.5, 15)
    >>> delta_sigma = calc.compute_deltasigma(rp)
    >>>
    >>> # Or compute bin-averaged values
    >>> rp_bins = np.logspace(-1, 1.5, 10)
    >>> delta_sigma_avg = calc.compute_deltasigma_averaged(rp_bins)

    Notes
    -----
    The surface mass density contrast is computed as:
    ΔΣ(rp) = Σ̄(<rp) - Σ̄(rp)

    where Σ̄ is the azimuthally averaged surface mass density:
    Σ(rp) = 2∫₀^χmax ρ_m(1 + ξ_gm(√(rp² + χ²))) dχ

    and the mean within rp is:
    Σ̄(<rp) = (2/rp²) ∫₀^rp Σ(rp') rp' drp'

    All integrals are computed using 200-point Gauss-Legendre quadrature
    for high accuracy. The spline_DeltaSigma is created at initialization
    and can be efficiently evaluated for any subsequent binning or radii.

    Performance: Initialization has a one-time computational cost, but
    multiple calls to compute_deltasigma_averaged() or compute_deltasigma()
    with different binnings/radii are ~50x faster than recomputation.

    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715
    .. [2] Cacciato et al. (2009), MNRAS 394, 929
    .. [3] Singh et al. (2017), MNRAS 471, 3827
    """
    def __init__(self, rr, xi_gm, RHO_M, chi_max=120):
        self.rr = rr
        self.RHO_M = RHO_M
        self.chi_max = chi_max
        self.spline_xigm = interp1d(rr, xi_gm, bounds_error=False, kind='cubic',
                                   fill_value=(xi_gm[0], 0))

        # Compute SIGMA at initialization
        self.SIGMA = self._compute_sigma_values()
        self.spline_SIGMA = interp1d(self.rr, self.SIGMA, kind='cubic',
                                     bounds_error=False, fill_value=(self.SIGMA[0], 0))

        # Compute Delta Sigma at initialization
        self.DeltaSigma = self._compute_deltasigma_values()
        self.spline_DeltaSigma = interp1d(self.rr, self.DeltaSigma,
                                          bounds_error=False, kind='cubic',
                                          fill_value=(self.DeltaSigma[0], self.DeltaSigma[-1]))
    
    def _sigma_integrand(self, chi, r_proj):
        """Integrand for surface density computation."""
        return self.spline_xigm(np.sqrt(r_proj**2 + chi**2))

    def _compute_sigma_values(self):
        """
        Compute surface density SIGMA at internal radii.

        Returns
        -------
        SIGMA : ndarray
            Surface density values at self.rr [ρ_m units]
        """
        SIGMA = 2 * gauss_legendre_integration(
            self._sigma_integrand, 0, self.chi_max, r_proj=self.rr
        )
        return SIGMA

    def _sigma_mean_integrand(self, r):
        """Integrand for mean surface density computation."""
        return self.spline_SIGMA(r) * r

    def _compute_deltasigma_values(self):
        """
        Compute Delta Sigma at internal radii.
        Returns
        -------
        DeltaSigma : ndarray
            Surface density contrast at self.rr [h M☉/pc²]
        """
        # Compute mean surface density inside each radius
        SIGMA_MEAN = 2 * gauss_legendre_integration(
            self._sigma_mean_integrand, 0, self.rr
        ) / self.rr**2

        # Compute excess surface density
        Delta_Sigma = SIGMA_MEAN - self.SIGMA

        # Convert to appropriate units [h M☉/pc²]
        Delta_Sigma = Delta_Sigma * self.RHO_M / 1e12

        return Delta_Sigma
    
    def compute_deltasigma_averaged(self, r_bins):
        """
        Compute bin-averaged Delta Sigma using pre-computed spline.

        Parameters
        ----------
        r_bins : array_like
            Bin edges for averaging [Mpc/h], shape (N+1,)

        Returns
        -------
        Delta_Sigma_averaged : ndarray
            Bin-averaged ΔΣ values [h M☉/pc²], shape (N,)

        Notes
        -----
        Uses the pre-computed spline_DeltaSigma from initialization.
        Multiple calls with different binnings are efficient.

        Examples
        --------
        >>> rp_bins = np.logspace(-1, 1.5, 10)
        >>> delta_sigma_avg = calc.compute_deltasigma_averaged(rp_bins)
        """
        Delta_Sigma_averaged = binavg_2D(self.spline_DeltaSigma, r_bins)
        return Delta_Sigma_averaged
    
    def compute_deltasigma(self, rp):
        """
        Evaluate Delta Sigma at specific radii using pre-computed spline.

        Parameters
        ----------
        rp : array_like
            Projected radii to evaluate [Mpc/h]

        Returns
        -------
        DeltaSigma : ndarray
            ΔΣ evaluated at input radii [h M☉/pc²]

        Notes
        -----
        Uses the pre-computed spline_DeltaSigma from initialization.

        Examples
        --------
        >>> rp = np.logspace(-1, 1.5, 20)
        >>> delta_sigma = calc.compute_deltasigma(rp)
        """
        DeltaSigma = self.spline_DeltaSigma(rp)
        return DeltaSigma


def compute_galaxy_clustering(positions_gal: jnp.ndarray,
                             Lbox: float,
                             rsd_axis: str,
                             mode: str, 
                             bins1: np.ndarray,
                             catalog2: Optional[jnp.ndarray] = None, 
                             bins2: Optional[np.ndarray] = None,
                             output: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute galaxy clustering correlation functions.
    
    Wrapper function for compute_corr() specialized for galaxy clustering
    measurements. Automatically handles boxsize and line-of-sight parameters.
    
    Parameters
    ----------
    positions_gal : jnp.ndarray, shape (N_galaxies, 3)
        Galaxy positions [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    rsd_axis : str
        Line-of-sight axis for RSD ('x', 'y', or 'z')
    mode : str
        Correlation function mode ('s', 'smu', 'rppi')
    bins1 : np.ndarray
        Primary binning edges
    catalog2 : jnp.ndarray, optional, shape (N2, 3)
        Second catalog for cross-correlation
    bins2 : np.ndarray, optional
        Secondary binning edges
    output : str, default='auto'
        Output format ('auto', 'multipoles', 'wp')
        
    Returns
    -------
    r : np.ndarray
        Bin centers or separation values
    xi : np.ndarray  
        Correlation function values
        
    Examples
    --------
    >>> # Galaxy auto-correlation
    >>> s, xi = compute_galaxy_clustering(gal_pos, 1000.0, 'z', 's', s_bins)
    >>> 
    >>> # Projected correlation function  
    >>> rp, wp = compute_galaxy_clustering(gal_pos, 1000.0, 'z', 'rppi', 
    ...                                   rp_bins, bins2=pi_bins, output='wp')
    """
    return compute_corr(
        mode, positions_gal, bins1,
        catalog2=catalog2, bins2=bins2,
        boxsize=Lbox, los=rsd_axis, output=output
    )


def compute_galaxy_lensing(positions_gal: jnp.ndarray,
                          positions_part: jnp.ndarray,
                          Lbox: float,
                          rsd_axis: str,
                          RHO_M: float,
                          bins1: np.ndarray,
                          output: str = 'xi',
                          bins2: Optional[np.ndarray] = None,
                          bins_comp: np.ndarray = np.geomspace(5e-3, 100, 81),
                          weights_part: Optional[np.ndarray] = None,
                          chi_max: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute galaxy-galaxy lensing signal ΔΣ(rp).
    
    This function calculates the projected surface mass density contrast
    around galaxies by cross-correlating the galaxy catalog with a
    matter tracer (typically dark matter particles).
    
    Parameters
    ----------
    positions_gal : jnp.ndarray, shape (N_galaxies, 3)
        Galaxy positions [Mpc/h]
    positions_part : jnp.ndarray, shape (N_particles, 3) 
        Matter tracer positions (e.g., dark matter particles) [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    rsd_axis : str
        Line-of-sight axis ('x', 'y', or 'z')
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    bins1 : np.ndarray
        Projected separation bin edges rp [Mpc/h]
    output : str, default='xi'
        Correlation function output format (passed to compute_corr)
    bins2 : np.ndarray, optional
        Secondary binning (if needed)
    bins_comp : np.ndarray, default=geomspace(5e-3, 100, 81)
        Computation bins for the galaxy-matter correlation function.
        Should span a wider range than bins1 for accurate integration.
        
    Returns
    -------
    rp_centers : np.ndarray
        Projected separation bin centers [Mpc/h]
    DeltaSigma : np.ndarray
        Surface mass density contrast [Msun h/pc²]
        
    Examples
    --------
    >>> # Compute lensing signal
    >>> rp_bins = np.logspace(-1, 1.5, 15) 
    >>> rp, delta_sigma = compute_galaxy_lensing(
    ...     gal_pos, dm_pos, 1000.0, 'z', 8.6e10, rp_bins
    ... )
    
    Notes
    -----
    This calculation computes the galaxy-matter correlation function over
    a wide range of scales (bins_comp) and then integrates to obtain the
    surface density contrast.
    
    The surface mass density contrast is computed as:
    ΔΣ(rp) = Σ̄(<rp) - Σ̄(rp)
    
    where Σ̄ is the azimuthally averaged surface mass density.
    
    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715
    .. [2] Cacciato et al. (2009), MNRAS 394, 929
    """
    # Compute bin centers
    rp_centers = np.sqrt(bins1[:-1] * bins1[1:])
    
    # Compute galaxy-matter correlation function (particles may carry mass
    # weights so ξ_gm traces the mass-weighted matter field, as needed for
    # multi-species hydro particles).
    rr, xi_gm = compute_corr(
        's', positions_gal, bins_comp,
        catalog2=positions_part, bins2=bins2,
        boxsize=Lbox, los=rsd_axis, output=output,
        weights2=weights_part,
    )

    # Calculate surface mass density contrast
    DeltaSigma_pipeline = DeltaSigmaCalculator(rr, xi_gm, RHO_M, chi_max=chi_max)
    DeltaSigma = DeltaSigma_pipeline.compute_deltasigma_averaged(bins1)
    
    return rp_centers, DeltaSigma