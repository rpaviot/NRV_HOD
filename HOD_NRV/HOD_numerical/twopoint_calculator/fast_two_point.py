"""
Fast galaxy-galaxy lensing calculator using pre-computed ΔΣ and spatial interpolation.

This module implements the fast evaluation phase of the optimized lensing
calculation, interpolating pre-computed ΔΣ values at galaxy positions.

References
----------
.. [1] Yuan et al. (2021), MNRAS, arXiv:2110.11412 (AbacusHOD paper)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median of values.

    Parameters
    ----------
    values : np.ndarray
        Values to take median of
    weights : np.ndarray
        Weights for each value (must be non-negative)

    Returns
    -------
    float
        Weighted median

    Notes
    -----
    The weighted median is the value where the cumulative weight equals 0.5.
    This is more robust to outliers than weighted mean.
    """
    # Sort by values
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Compute cumulative weights
    cumsum = np.cumsum(sorted_weights)
    cumsum /= cumsum[-1]  # Normalize to [0, 1]

    # Find where cumsum crosses 0.5
    idx = np.searchsorted(cumsum, 0.5)

    return sorted_values[idx]


class FastDeltaSigmaCalculator:
    """
    Fast galaxy-galaxy lensing calculator using pre-computed ΔΣ values.

    This class loads pre-computed ΔΣ(rp) values at particle positions and
    uses spatial interpolation to estimate ΔΣ at arbitrary galaxy positions.

    Parameters
    ----------
    precomputed_file : str
        Path to HDF5 file with pre-computed lensing data
    interpolation_method : str, default='idw'
        Interpolation method: 'idw' (inverse distance weighting) or 'rbf'
        (radial basis function)
    k_neighbors : int, default=8
        Number of nearest neighbors to use for interpolation
    Lbox : float, optional
        Simulation box size for periodic boundary conditions [Mpc/h]
    use_median : bool, default=False
        If True, use weighted median instead of weighted mean for interpolation.
        More robust to outliers and skewed distributions.

    Attributes
    ----------
    positions : np.ndarray, shape (N, 3)
        Pre-computed positions [Mpc/h]
    deltasigma : np.ndarray, shape (N, M)
        Pre-computed ΔΣ values [Msun h/pc²]
    rp_bins : np.ndarray
        Radial bin edges [Mpc/h]
    kdtree : cKDTree
        KD-tree for fast neighbor queries
    metadata : dict
        Metadata from pre-computation

    Examples
    --------
    >>> from HOD_NRV.twopoint_calculator import FastDeltaSigmaCalculator
    >>>
    >>> # Initialize calculator
    >>> calc = FastDeltaSigmaCalculator('precomputed_lensing.h5')
    >>>
    >>> # Compute lensing for a galaxy catalog
    >>> galaxy_positions = halo.positions_gal  # shape (N_gal, 3)
    >>> rp_centers, delta_sigma = calc.compute_deltasigma_for_galaxies(
    ...     galaxy_positions, rp_bins=np.logspace(-1, 1.5, 15)
    ... )
    >>>
    >>> # Result is ~100-1000x faster than traditional method
    >>> print(f"ΔΣ at rp={rp_centers[5]:.2f} Mpc/h: {delta_sigma[5]:.2e} Msun h/pc²")

    Notes
    -----
    The total lensing signal is computed as a sum of contributions from
    all galaxies:

    .. math::
        \\Delta\\Sigma_{total}(r_p) = \\sum_i \\Delta\\Sigma_i(r_p)

    where ΔΣ_i is interpolated from pre-computed values near galaxy i's position.

    References
    ----------
    .. [1] Mandelbaum et al. (2006), MNRAS 368, 715
    .. [2] Yuan et al. (2021), MNRAS, arXiv:2110.11412
    """

    def __init__(
        self,
        precomputed_file: str,
        interpolation_method: str = 'idw',
        k_neighbors: int = 1000,
        Lbox: Optional[float] = None,
        use_median: bool = False
    ):
        from .precompute_deltasigma import load_precomputed_lensing

        # Load pre-computed data
        self.positions, self.deltasigma, self.rp_bins, self.metadata = \
            load_precomputed_lensing(precomputed_file)

        self.interpolation_method = interpolation_method
        self.k_neighbors = k_neighbors
        self.use_median = use_median

        # Get Lbox from metadata if not provided
        if Lbox is None:
            self.Lbox = self.metadata.get('Lbox', None)
        else:
            self.Lbox = Lbox

        # Build KD-tree for fast neighbor searches
        print(f"Building KD-tree for {len(self.positions)} pre-computed positions...")
        if self.Lbox is not None:
            self.kdtree = cKDTree(self.positions, boxsize=self.Lbox)
        else:
            self.kdtree = cKDTree(self.positions)
            print("Warning: No Lbox provided, periodic boundaries not enforced")

        print(f"FastDeltaSigmaCalculator initialized")
        print(f"  Interpolation method: {interpolation_method}")
        print(f"  K-neighbors: {k_neighbors}")
        print(f"  Use median: {use_median}")
        print(f"  Radial bins: {len(self.rp_bins)-1}")

    def interpolate_deltasigma_at_position(
        self,
        position: np.ndarray,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Interpolate ΔΣ(rp) at a single galaxy position.

        Parameters
        ----------
        position : np.ndarray, shape (3,)
            Galaxy position [Mpc/h]
        power : float, default=2.0
            Power for inverse distance weighting (only used if method='idw')

        Returns
        -------
        delta_sigma_interp : np.ndarray, shape (len(rp_bins)-1,)
            Interpolated ΔΣ(rp) at this position [Msun h/pc²]

        Notes
        -----
        Uses inverse distance weighting by default:

        .. math::
            \\Delta\\Sigma(r_p) = \\frac{\\sum_i w_i \\Delta\\Sigma_i(r_p)}{\\sum_i w_i}

        where :math:`w_i = 1 / d_i^p` and d_i is distance to neighbor i.
        """
        # Query k-nearest neighbors
        distances, indices = self.kdtree.query(position, k=self.k_neighbors)

        # Handle case where query point is exactly on a pre-computed point
        if distances[0] < 1e-10:
            return self.deltasigma[indices[0]]

        if self.interpolation_method == 'idw':
            # Inverse distance weighting
            weights = 1.0 / (distances ** power)
            weights /= weights.sum()

            if self.use_median:
                # Weighted median of ΔΣ from neighbors (more robust to outliers)
                delta_sigma_interp = np.zeros(len(self.rp_bins) - 1)
                for i in range(len(self.rp_bins) - 1):
                    delta_sigma_interp[i] = weighted_median(
                        self.deltasigma[indices, i], weights
                    )
            else:
                # Weighted average of ΔΣ from neighbors
                delta_sigma_interp = np.sum(
                    weights[:, np.newaxis] * self.deltasigma[indices],
                    axis=0
                )

        elif self.interpolation_method == 'rbf':
            # Radial basis function interpolation
            # Note: This can be slower than IDW
            neighbor_positions = self.positions[indices]
            neighbor_deltasigma = self.deltasigma[indices]

            # Use multiquadric RBF for each radial bin
            delta_sigma_interp = np.zeros(len(self.rp_bins) - 1)
            for i in range(len(self.rp_bins) - 1):
                # Create RBF interpolator for this radial bin
                rbf = RBFInterpolator(
                    neighbor_positions,
                    neighbor_deltasigma[:, i],
                    kernel='multiquadric',
                    epsilon=1.0
                )
                delta_sigma_interp[i] = rbf(position.reshape(1, -1))[0]

        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")

        return delta_sigma_interp

    def compute_deltasigma_for_galaxies(
        self,
        galaxy_positions: np.ndarray,
        rp_bins: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute averaged ΔΣ from contributions of all galaxies.

        Parameters
        ----------
        galaxy_positions : np.ndarray, shape (N_galaxies, 3)
            Positions of galaxies [Mpc/h]
        rp_bins : np.ndarray, optional
            Projected separation bin edges [Mpc/h]. If None, uses pre-computed bins.
        verbose : bool, default=False
            Print progress information

        Returns
        -------
        rp_centers : np.ndarray
            Bin centers [Mpc/h]
        delta_sigma_avg : np.ndarray
            Averaged surface mass density contrast [Msun h/pc²]

        Examples
        --------
        >>> # After populating galaxies with HOD
        >>> rp, ds = calc.compute_deltasigma_for_galaxies(halo.positions_gal)
        >>> plt.loglog(rp, ds)
        >>> plt.xlabel(r'$r_p$ [Mpc/h]')
        >>> plt.ylabel(r'$\\Delta\\Sigma$ [M$_\\odot$ h/pc$^2$]')

        Notes
        -----
        The lensing signal is the average of individual contributions:

        .. math::
            \\Delta\\Sigma_{avg}(r_p) = \\frac{1}{N_{gal}} \\sum_{i=1}^{N_{gal}} \\Delta\\Sigma_i(r_p)

        where each ΔΣ_i is interpolated from nearby pre-computed values.
        """
        # Use pre-computed bins if not provided
        if rp_bins is None:
            rp_bins = self.rp_bins

        # Check if rp_bins match pre-computed bins
        if not np.allclose(rp_bins, self.rp_bins):
            print("Warning: Provided rp_bins differ from pre-computed bins")
            print("Results may require rebinning")

        n_galaxies = len(galaxy_positions)
        n_bins = len(rp_bins) - 1

        if verbose:
            print(f"Computing ΔΣ for {n_galaxies} galaxies...")

        # Initialize sum
        delta_sigma_sum = np.zeros(n_bins)

        # Sum contributions from all galaxies
        for i, gal_pos in enumerate(galaxy_positions):
            if verbose and (i + 1) % max(1, n_galaxies // 10) == 0:
                print(f"  Progress: {i+1}/{n_galaxies} ({100*(i+1)/n_galaxies:.1f}%)")

            # Interpolate ΔΣ at this galaxy position
            delta_sigma_i = self.interpolate_deltasigma_at_position(gal_pos)

            # Add to sum
            delta_sigma_sum += delta_sigma_i

        # Average over all galaxies (galaxy-galaxy lensing is an average, not a sum)
        delta_sigma_avg = delta_sigma_sum / n_galaxies

        # Compute bin centers
        rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

        if verbose:
            print(f"Done! Averaged ΔΣ computed at {n_bins} radial bins")

        return rp_centers, delta_sigma_avg

    def compute_deltasigma_averaged(
        self,
        galaxy_positions: np.ndarray,
        rp_bins: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Compute bin-averaged ΔΣ (same output format as legacy calculator).

        This is a convenience method that matches the output format of the
        original DeltaSigmaCalculator.compute_deltasigma_averaged() method.

        Parameters
        ----------
        galaxy_positions : np.ndarray, shape (N_galaxies, 3)
            Positions of galaxies [Mpc/h]
        rp_bins : np.ndarray, optional
            Projected separation bin edges [Mpc/h]
        verbose : bool, default=False
            Print progress information

        Returns
        -------
        delta_sigma_averaged : np.ndarray
            Bin-averaged surface mass density contrast [Msun h/pc²]

        Examples
        --------
        >>> # Drop-in replacement for legacy calculator
        >>> ds = calc.compute_deltasigma_averaged(halo.positions_gal, rp_bins)

        Notes
        -----
        This method provides the same interface as the legacy calculator for
        easy integration into existing code.
        """
        _, delta_sigma = self.compute_deltasigma_for_galaxies(
            galaxy_positions, rp_bins, verbose
        )
        return delta_sigma

    def __repr__(self):
        return (
            f"FastDeltaSigmaCalculator("
            f"n_positions={len(self.positions)}, "
            f"n_bins={len(self.rp_bins)-1}, "
            f"method='{self.interpolation_method}', "
            f"k={self.k_neighbors})"
        )
