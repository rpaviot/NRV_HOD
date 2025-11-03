"""
Improved spatial interpolation methods for noisy ΔΣ(r) data.

This module provides robust interpolation methods specifically designed
for handling noisy surface density profiles from downsampled simulations.

References
----------
.. [1] Rasmussen & Williams (2006), Gaussian Processes for Machine Learning
.. [2] Shepard (1968), ACM National Conference
.. [3] Nadaraya (1964), Theory of Probability & Its Applications
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional, Callable
import warnings


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Gaussian kernel for kernel smoothing.

    K(d) = exp(-0.5 * (d/h)²)

    Parameters
    ----------
    distances : np.ndarray
        Distances from query point
    bandwidth : float
        Kernel bandwidth (smoothing parameter)

    Returns
    -------
    weights : np.ndarray
        Kernel weights
    """
    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def epanechnikov_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Epanechnikov kernel (more compact support than Gaussian).

    K(d) = max(0, 1 - (d/h)²)
    """
    u = distances / bandwidth
    return np.maximum(0, 1 - u ** 2)


class ImprovedDeltaSigmaInterpolator:
    """
    Advanced interpolation methods for noisy ΔΣ data.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Pre-computed positions [Mpc/h]
    deltasigma : np.ndarray, shape (N, M)
        Pre-computed ΔΣ values [Msun h/pc²]
    Lbox : float, optional
        Box size for periodic boundaries [Mpc/h]
    method : str, default='kernel_smooth'
        Interpolation method:
        - 'kernel_smooth': Nadaraya-Watson kernel regression (best for noisy data)
        - 'shepard': Modified Shepard's method with larger neighborhood
        - 'rbf_smooth': Smoothed RBF interpolation
        - 'gp': Gaussian Process interpolation (slowest, best quality)
    k_neighbors : int, default=32
        Number of neighbors (increased from 8 for better stability)
    bandwidth : float, optional
        Kernel bandwidth for kernel smoothing (auto-computed if None)
    kernel : str, default='gaussian'
        Kernel type for kernel smoothing: 'gaussian' or 'epanechnikov'

    Examples
    --------
    >>> # Kernel smoothing (recommended for noisy data)
    >>> interp = ImprovedDeltaSigmaInterpolator(
    ...     positions, deltasigma, Lbox=1000.0,
    ...     method='kernel_smooth', k_neighbors=64, bandwidth=5.0
    ... )
    >>> ds = interp.interpolate_at_position(galaxy_pos)

    >>> # Shepard's method with larger k
    >>> interp = ImprovedDeltaSigmaInterpolator(
    ...     positions, deltasigma, method='shepard', k_neighbors=32
    ... )

    >>> # Gaussian Process (slow but highest quality)
    >>> interp = ImprovedDeltaSigmaInterpolator(
    ...     positions, deltasigma, method='gp', k_neighbors=100
    ... )
    """

    def __init__(
        self,
        positions: np.ndarray,
        deltasigma: np.ndarray,
        Lbox: Optional[float] = None,
        method: str = 'kernel_smooth',
        k_neighbors: int = 32,
        bandwidth: Optional[float] = None,
        kernel: str = 'gaussian'
    ):
        self.positions = positions
        self.deltasigma = deltasigma
        self.Lbox = Lbox
        self.method = method
        self.k_neighbors = k_neighbors
        self.kernel_type = kernel

        # Build KD-tree
        if Lbox is not None:
            self.kdtree = cKDTree(positions, boxsize=Lbox)
        else:
            self.kdtree = cKDTree(positions)

        # Auto-compute bandwidth if not provided
        if bandwidth is None and method == 'kernel_smooth':
            # Use median distance to k-th neighbor as bandwidth estimate
            sample_size = min(1000, len(positions))
            sample_idx = np.random.choice(len(positions), sample_size, replace=False)
            distances, _ = self.kdtree.query(
                positions[sample_idx], k=k_neighbors+1
            )
            # Median of k-th nearest neighbor distance
            self.bandwidth = np.median(distances[:, -1]) * 1.5  # Factor 1.5 for smoothing
            print(f"Auto-computed bandwidth: {self.bandwidth:.3f} Mpc/h")
        else:
            self.bandwidth = bandwidth

        # Select kernel function
        if kernel == 'gaussian':
            self.kernel_func = gaussian_kernel
        elif kernel == 'epanechnikov':
            self.kernel_func = epanechnikov_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        print(f"ImprovedDeltaSigmaInterpolator initialized:")
        print(f"  Method: {method}")
        print(f"  K-neighbors: {k_neighbors}")
        if method == 'kernel_smooth':
            print(f"  Bandwidth: {self.bandwidth:.3f} Mpc/h")
            print(f"  Kernel: {kernel}")

    def interpolate_at_position(
        self,
        position: np.ndarray,
        power: float = 3.0
    ) -> np.ndarray:
        """
        Interpolate ΔΣ(rp) at a single position.

        Parameters
        ----------
        position : np.ndarray, shape (3,)
            Query position [Mpc/h]
        power : float, default=3.0
            Power for Shepard's method (higher = less smooth, more local)

        Returns
        -------
        delta_sigma_interp : np.ndarray
            Interpolated ΔΣ values
        """
        # Query k-nearest neighbors
        distances, indices = self.kdtree.query(position, k=self.k_neighbors)

        # Handle exact match
        if distances[0] < 1e-10:
            return self.deltasigma[indices[0]]

        # Remove any points with zero distance (shouldn't happen, but safety)
        mask = distances > 1e-10
        distances = distances[mask]
        indices = indices[mask]

        if len(distances) == 0:
            warnings.warn("No valid neighbors found!")
            return np.zeros(self.deltasigma.shape[1])

        # Select interpolation method
        if self.method == 'kernel_smooth':
            return self._kernel_smoothing(distances, indices)

        elif self.method == 'shepard':
            return self._shepard_interpolation(distances, indices, power)

        elif self.method == 'rbf_smooth':
            return self._rbf_smoothed(distances, indices)

        elif self.method == 'gp':
            return self._gaussian_process(distances, indices)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _kernel_smoothing(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Nadaraya-Watson kernel regression.

        f(x) = Σ K(d_i) * y_i / Σ K(d_i)

        This is optimal for noisy data with smooth underlying signal.
        """
        # Compute kernel weights
        weights = self.kernel_func(distances, self.bandwidth)

        # Normalize weights
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            # Fallback to uniform weights if all kernel values too small
            weights = np.ones_like(weights)
            weight_sum = len(weights)

        weights /= weight_sum

        # Weighted average
        delta_sigma_interp = np.sum(
            weights[:, np.newaxis] * self.deltasigma[indices],
            axis=0
        )

        return delta_sigma_interp

    def _shepard_interpolation(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        power: float = 3.0
    ) -> np.ndarray:
        """
        Modified Shepard's method with higher power for locality.

        w_i = 1 / d_i^p

        Higher k and power give more stable results for noisy data.
        """
        # Inverse distance weighting with higher power
        weights = 1.0 / (distances ** power)
        weights /= weights.sum()

        delta_sigma_interp = np.sum(
            weights[:, np.newaxis] * self.deltasigma[indices],
            axis=0
        )

        return delta_sigma_interp

    def _rbf_smoothed(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Smoothed RBF interpolation with regularization.

        Uses thin-plate spline RBF with smoothing parameter to handle noise.
        """
        neighbor_positions = self.positions[indices]
        neighbor_deltasigma = self.deltasigma[indices]

        n_bins = self.deltasigma.shape[1]
        delta_sigma_interp = np.zeros(n_bins)

        # Compute typical length scale
        epsilon = np.median(distances)

        for i in range(n_bins):
            try:
                # Use thin plate spline with smoothing
                rbf = RBFInterpolator(
                    neighbor_positions,
                    neighbor_deltasigma[:, i],
                    kernel='thin_plate_spline',
                    smoothing=0.1,  # Regularization for noise
                    epsilon=epsilon
                )
                delta_sigma_interp[i] = rbf(position.reshape(1, -1))[0]
            except Exception as e:
                # Fallback to simple weighted average
                weights = 1.0 / (distances ** 2)
                weights /= weights.sum()
                delta_sigma_interp[i] = np.sum(
                    weights * neighbor_deltasigma[:, i]
                )

        return delta_sigma_interp

    def _gaussian_process(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Gaussian Process interpolation (highest quality, slowest).

        Uses squared exponential kernel with noise term.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            warnings.warn("sklearn not available, falling back to kernel smoothing")
            return self._kernel_smoothing(distances, indices)

        neighbor_positions = self.positions[indices]
        neighbor_deltasigma = self.deltasigma[indices]

        n_bins = self.deltasigma.shape[1]
        delta_sigma_interp = np.zeros(n_bins)

        # Define kernel: RBF + noise
        length_scale = np.median(distances)
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.1)

        for i in range(n_bins):
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.1,  # Additional noise regularization
                    n_restarts_optimizer=0  # Speed up
                )
                gp.fit(neighbor_positions, neighbor_deltasigma[:, i])
                delta_sigma_interp[i] = gp.predict(position.reshape(1, -1))[0]
            except Exception as e:
                # Fallback
                weights = 1.0 / (distances ** 2)
                weights /= weights.sum()
                delta_sigma_interp[i] = np.sum(
                    weights * neighbor_deltasigma[:, i]
                )

        return delta_sigma_interp


class AdaptiveDeltaSigmaInterpolator(ImprovedDeltaSigmaInterpolator):
    """
    Adaptive interpolation that adjusts parameters based on local density.

    In dense regions: use smaller k and bandwidth (less smoothing needed)
    In sparse regions: use larger k and bandwidth (more smoothing needed)

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Pre-computed positions [Mpc/h]
    deltasigma : np.ndarray, shape (N, M)
        Pre-computed ΔΣ values [Msun h/pc²]
    Lbox : float, optional
        Box size for periodic boundaries [Mpc/h]
    k_min : int, default=16
        Minimum number of neighbors in dense regions
    k_max : int, default=64
        Maximum number of neighbors in sparse regions
    bandwidth_min : float, optional
        Minimum bandwidth in dense regions
    bandwidth_max : float, optional
        Maximum bandwidth in sparse regions

    Examples
    --------
    >>> # Adaptive interpolation automatically adjusts to local conditions
    >>> interp = AdaptiveDeltaSigmaInterpolator(
    ...     positions, deltasigma, Lbox=1000.0,
    ...     k_min=16, k_max=64
    ... )
    >>> ds = interp.interpolate_at_position(galaxy_pos)
    """

    def __init__(
        self,
        positions: np.ndarray,
        deltasigma: np.ndarray,
        Lbox: Optional[float] = None,
        k_min: int = 16,
        k_max: int = 64,
        bandwidth_min: Optional[float] = None,
        bandwidth_max: Optional[float] = None,
        kernel: str = 'gaussian'
    ):
        # Initialize with maximum k
        super().__init__(
            positions, deltasigma, Lbox,
            method='kernel_smooth',
            k_neighbors=k_max,
            bandwidth=bandwidth_max,
            kernel=kernel
        )

        self.k_min = k_min
        self.k_max = k_max

        # Compute local densities for adaptation
        print("Computing local density field for adaptive interpolation...")
        self._compute_local_densities()

        # Set bandwidth range
        if bandwidth_min is None:
            self.bandwidth_min = self.bandwidth * 0.5
        else:
            self.bandwidth_min = bandwidth_min

        if bandwidth_max is None:
            self.bandwidth_max = self.bandwidth * 2.0
        else:
            self.bandwidth_max = bandwidth_max

        print(f"Adaptive bandwidth range: [{self.bandwidth_min:.3f}, {self.bandwidth_max:.3f}] Mpc/h")

    def _compute_local_densities(self):
        """Compute local density at each pre-computed position."""
        # Sample positions to estimate density field
        sample_size = min(5000, len(self.positions))
        sample_idx = np.random.choice(len(self.positions), sample_size, replace=False)

        # Query k-nearest neighbors for each sample
        distances, _ = self.kdtree.query(
            self.positions[sample_idx], k=self.k_max
        )

        # Local density ∝ k / V_k where V_k ∝ r_k^3
        self.local_density_scale = np.median(distances[:, -1])

    def interpolate_at_position(self, position: np.ndarray) -> np.ndarray:
        """
        Interpolate with adaptive parameters based on local density.
        """
        # Query neighbors to assess local density
        distances, indices = self.kdtree.query(position, k=self.k_max)

        # Handle exact match
        if distances[0] < 1e-10:
            return self.deltasigma[indices[0]]

        # Estimate local density from k-th nearest neighbor distance
        # Sparse region: large distance → use more neighbors, larger bandwidth
        # Dense region: small distance → use fewer neighbors, smaller bandwidth
        local_scale = distances[self.k_max // 2]  # Median distance
        density_ratio = local_scale / self.local_density_scale

        # Adapt k: more neighbors in sparse regions
        k_adaptive = int(np.clip(
            self.k_min + (self.k_max - self.k_min) * density_ratio,
            self.k_min, self.k_max
        ))

        # Adapt bandwidth: larger in sparse regions
        bandwidth_adaptive = np.clip(
            self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * density_ratio,
            self.bandwidth_min, self.bandwidth_max
        )

        # Use adapted parameters
        distances_adapt = distances[:k_adaptive]
        indices_adapt = indices[:k_adaptive]

        # Kernel smoothing with adaptive bandwidth
        weights = self.kernel_func(distances_adapt, bandwidth_adaptive)
        weight_sum = weights.sum()

        if weight_sum < 1e-10:
            weights = np.ones_like(weights)
            weight_sum = len(weights)

        weights /= weight_sum

        delta_sigma_interp = np.sum(
            weights[:, np.newaxis] * self.deltasigma[indices_adapt],
            axis=0
        )

        return delta_sigma_interp
