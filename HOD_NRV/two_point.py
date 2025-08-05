from pycorr import TwoPointCorrelationFunction
from .utils import gauss_legendre_integration
import numpy as np
from scipy.interpolate import interp1d

def compute_corr(mode, catalog1, bins1, catalog2=None, bins2=None, boxsize=None):
    """
    Compute the 2PCF using the Natural estimator (DD/RR - 1) with RR computed analytically.
    
    Parameters
    ----------
    mode : str
        Binning mode (e.g., 'smu', 'rppi', 's', etc.).
    catalog1 : array_like
        Positions of first dataset (shape: Nx3).
    bins1 : array_like
        Bin edges or tuple of edges (for 1D or 2D binning).
    catalog2 : array_like, optional
        Positions of second dataset (for cross-correlation).
    bins2 : array_like, optional
        Second binning axis if doing 2D binning.
    boxsize : float or array_like, optional
        Size of the periodic box for analytic RR.
        
    Returns
    -------
    TwoPointCorrelationFunction
        The resulting correlation function.
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


    # Construct TwoPointCorrelationFunction
    corr = TwoPointCorrelationFunction(
        mode=mode,
        edges=edges,
        data_positions1=catalog1,
        data_positions2=catalog2 if is_cross else None,
        estimator='natural',
        boxsize=boxsize
    )

    return corr


class DeltaSigmaCalculator:
    """
    Class to compute Delta_Sigma efficiently by reusing splines
    """
    def __init__(self, r, xi_gm, RHO_M):
        self.RHO_M = RHO_M
        self.spline_xigm = interp1d(r, xi_gm, bounds_error=False, kind='cubic',
                                   fill_value=(xi_gm[0], 0))
    
    def sigma_integral(self, chi, r_proj):
        """Function to integrate for surface density"""
        return self.spline_xigm(np.sqrt(r_proj**2 + chi**2))
    
    def compute_sigma(self, r, chi_max=100):
        """Compute surface density at radii rr"""
        SIGMA = 2 * gauss_legendre_integration(self.sigma_integral, 0, chi_max, r)
        
        # Store SIGMA and create its spline
        self.SIGMA = SIGMA
        self.r = r
        self.spline_SIGMA = interp1d(r, SIGMA, bounds_error=False, 
                                    fill_value=(SIGMA[0], 0))
        return SIGMA
    
    def sigma_mean_integrand(self, r_val):
        """Function to integrate for mean surface density"""
        return self.spline_SIGMA(r_val) * r_val
    
    def compute_delta_sigma(self, rr=None):
        """
        Compute Delta_Sigma from previously calculated SIGMA
        If rr is provided, compute_sigma will be called first
        """
        if rr is not None or not hasattr(self, 'SIGMA'):
            self.compute_sigma(rr)
        
        # Compute mean surface density inside R
        SIGMA_MEAN = np.zeros_like(self.rr)
        for i, r_val in enumerate(self.rr):
            if r_val > 0:
                SIGMA_MEAN[i] = 2 * gauss_legendre_integration(
                    lambda r: self.spline_SIGMA(r) * r, 0, r_val) / r_val**2
        
        # Compute excess surface density
        Delta_Sigma = SIGMA_MEAN - self.SIGMA
        
        # Convert to appropriate units
        Delta_Sigma = Delta_Sigma * self.RHO_M / 1e12
        
        return Delta_Sigma
