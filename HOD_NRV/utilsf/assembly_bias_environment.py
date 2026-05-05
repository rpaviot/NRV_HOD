"""
Assembly Bias Environmental Properties Calculator

This module provides functions to compute density and shear fields from halo and particle
catalogues, following the assembly bias approach described in Paviot et al. (2024).

Based on equations 15-16 from the paper:
log M^mod_min = log M_min + A_cent f_A + B_cent f_B  (Eq. 15)
log M^mod_1 = log M_1 + A_sat f_A + B_sat f_B        (Eq. 16)

Where f_A and f_B are normalized internal and environmental halo properties.
"""

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
from pysco.mesh import TSC, invTSC
from pysco.utils import periodic_wrap, argsort_par,injection_with_indices
from pysco.fourier import fft_3D_real, ifft_3D_real
from pysco import morton
import gc
from numpy import linalg as LA
from scipy import stats
from typing import Tuple, Optional, Union


def fourier_axes(N: int, Lbox: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return broadcastable Fourier wavevectors in float32.
    
    Parameters
    ----------
    N : int
        Number of grid cells per dimension
    Lbox : float
        Box size in Mpc/h
        
    Returns
    -------
    tuple of ndarray
        kx, ky, kz wavevector grids
    """
    kx = (np.fft.fftfreq(N, d=Lbox/N) * 2*np.pi).astype(np.float32)[:, None, None]
    ky = (np.fft.fftfreq(N, d=Lbox/N) * 2*np.pi).astype(np.float32)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=Lbox/N) * 2*np.pi).astype(np.float32)[None, None, :]
    return kx, ky, kz


def grid_positions(ncells_1d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate grid positions for density field calculation.
    
    Parameters
    ----------
    ncells_1d : int
        Number of grid cells per dimension
        
    Returns
    -------
    tuple of ndarray
        x, y, z grid position arrays
    """
    delta = 1.0 / ncells_1d
    x = delta * (np.arange(ncells_1d) + 0.5)[:, None, None]
    y = delta * (np.arange(ncells_1d) + 0.5)[None, :, None]
    z = delta * (np.arange(ncells_1d) + 0.5)[None, None, :]
    return x, y, z


def reorder_positions(positions: np.ndarray) -> np.ndarray:
    """
    Reorder positions using Morton ordering for better memory access.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions array of shape (N, 3)
        
    Returns
    -------
    ndarray
        Reordered positions array
    """
    index = morton.positions_to_keys(positions)
    nthreads = numba.get_num_threads()
    
    if nthreads > 1:
        arg = argsort_par(index, nthreads)
    else:
        arg = np.argsort(index)
    
    index = 0  # Free memory
    positions = injection_with_indices(arg, positions)
    return positions


@njit(
    ["void(c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def compensate_mas(x: np.ndarray, p: int) -> None:
    """
    Inplace deconvolution of Fourier-space field by the MAS window (Jing 2005).
    
    Parameters
    ----------
    x : ndarray, complex64
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP=1, CIC=2, TSC=3)
    """
    ncells_1d = len(x)
    h = np.float32(1.0 / ncells_1d)
    middle = ncells_1d // 2

    for i in prange(ncells_1d):
        if i >= middle:
            kx = np.float32(i - ncells_1d)
        else:
            kx = np.float32(i)
        w_x = np.sinc(kx * h)

        for j in prange(ncells_1d):
            if j >= middle:
                ky = np.float32(j - ncells_1d)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * h)

            for k in prange(middle + 1):
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * h)
                if w_xyz != 0.0:
                    x[i, j, k] /= w_xyz**p


@njit(parallel=True, fastmath=True, cache=True)
def apply_interlacing(fourier1: np.ndarray, fourier2: np.ndarray) -> None:
    """
    Combine two Fourier fields via Jing-2005 interlacing to cancel aliases.

    Stores ``0.5 * (fourier1 + e^{i k·Δx} fourier2)`` in-place into ``fourier1``,
    where ``Δx = (H/2, H/2, H/2)`` and ``H = Lbox / N``. ``fourier2`` must be
    the FFT of the density field painted from positions shifted by ``H/2`` in
    every dimension.

    Parameters
    ----------
    fourier1 : ndarray, complex64
        FFT of the unshifted density field. Modified in place.
    fourier2 : ndarray, complex64
        FFT of the half-cell-shifted density field.
    """
    N = fourier1.shape[0]
    middle = N // 2

    for i in prange(N):
        if i >= middle:
            kx = i - N
        else:
            kx = i

        for j in range(N):
            if j >= middle:
                ky = j - N
            else:
                ky = j

            for k in range(middle + 1):
                kz = k
                phase = np.exp(1j * np.pi * (kx + ky + kz) / N)
                fourier1[i, j, k] = 0.5 * (fourier1[i, j, k]
                                           + phase * fourier2[i, j, k])


def compute_Tij(i: int, j: int, deltak: np.ndarray, phase_axes: Tuple[np.ndarray, ...], 
                halo_positions: np.ndarray, k2: np.ndarray, threads: int = 32) -> np.ndarray:
    """
    Compute tidal tensor component T_ij.
    
    Parameters
    ----------
    i, j : int
        Tensor indices (0, 1, 2 for x, y, z)
    deltak : ndarray
        Fourier-space density field
    phase_axes : tuple
        Fourier wavevector grids (kx, ky, kz)
    halo_positions : ndarray
        Halo positions for interpolation
    k2 : ndarray
        k² field with k²[0,0,0] = inf to avoid division by zero
    threads : int, optional
        Number of threads for FFT
        
    Returns
    -------
    ndarray
        Tidal tensor component at halo positions
    """
    Tij_k = (phase_axes[i] * phase_axes[j] / k2) * deltak
    Tij_x = ifft_3D_real(Tij_k, threads=threads)
    Tij_h = invTSC(Tij_x, halo_positions)
    return Tij_h


def normalize_distribution(p: np.ndarray) -> np.ndarray:
    """
    Normalize distribution to range [-1, 1] using percentiles.
    
    Parameters
    ----------
    p : ndarray
        Input distribution
        
    Returns
    -------
    ndarray
        Normalized distribution in range [-1, 1]
    """
    p = np.log(p)
    low_percentile = np.percentile(p, 2)
    high_percentile = np.percentile(p, 98)
    median = np.median(p)
    norm_p = (p - median) / (high_percentile - low_percentile)
    norm_p[norm_p < -1.] = -1.
    norm_p[norm_p > 1.] = 1.
    return norm_p


@njit(parallel=True, fastmath=True, cache=True)
def compute_shear_eigenvalues(T11: np.ndarray, T12: np.ndarray, T13: np.ndarray,
                             T22: np.ndarray, T23: np.ndarray, T33: np.ndarray) -> np.ndarray:
    """
    Compute shear q_R² from tidal tensor eigenvalues.
    
    Parameters
    ----------
    T11, T12, T13, T22, T23, T33 : ndarray
        Tidal tensor components
        
    Returns
    -------
    ndarray
        Shear q_R² values
    """
    size = len(T11)
    qr2 = np.zeros(size)
    
    for i in prange(size):
        Tij = np.zeros((3, 3))
        Tij[0, 0] = T11[i]
        Tij[0, 1] = T12[i]
        Tij[1, 0] = T12[i]
        Tij[0, 2] = T13[i]
        Tij[2, 0] = T13[i]
        Tij[1, 2] = T23[i]
        Tij[2, 1] = T23[i]
        Tij[1, 1] = T22[i]
        Tij[2, 2] = T33[i]
        
        eigenvalues, _ = LA.eig(Tij)
        L1, L2, L3 = np.sort(eigenvalues)
        qr2[i] = (1./2.) * ((L2 - L1)**2 + (L3 - L1)**2 + (L3 - L2)**2)
    
    return qr2


class AssemblyBiasEnvironment:
    """
    Class to compute environmental properties for assembly bias modeling.
    
    This implements the methodology from Paviot et al. (2024) for computing
    density and shear fields that are used in the assembly bias HOD framework.
    """
    
    def __init__(self, Nmesh: int = 1024, Lbox: float = 681, 
                 smoothing_radius: float = 1.0, threads: int = 32):
        """
        Initialize the assembly bias environment calculator.
        
        Parameters
        ----------
        Nmesh : int, optional
            Number of mesh cells per dimension (default: 1024)
        Lbox : float, optional
            Box size in Mpc/h (default: 681)
        smoothing_radius : float, optional
            Gaussian smoothing radius in Mpc/h (default: 1.0)
        threads : int, optional
            Number of threads for FFT operations (default: 32)
        """
        self.Nmesh = Nmesh
        self.Lbox = Lbox
        self.smoothing_radius = smoothing_radius
        self.threads = threads
        self.step = Lbox / Nmesh
        
        # Pre-compute Fourier axes and k²
        self.phase_axes = fourier_axes(Nmesh, Lbox)
        self.k2 = sum(ki**2 for ki in self.phase_axes)
        self.k2[0, 0, 0] = np.inf  # Avoid division by zero
        
        # Gaussian smoothing kernel
        self.W = np.exp(-0.5 * self.k2 * (smoothing_radius**2))
    
    def _compute_fourier_density(self, particle_positions: np.ndarray,
                                normalize: bool = True) -> np.ndarray:
        """
        Compute MAS-compensated Fourier density field without Gaussian smoothing.

        Parameters
        ----------
        particle_positions : ndarray
            Particle positions array of shape (N, 3) in units of Lbox
        normalize : bool, optional
            Whether to normalize positions and apply periodic wrapping

        Returns
        -------
        ndarray
            Fourier-space density field (unsmoothed, MAS-compensated)
        """
        if normalize:
            positions = particle_positions.copy().astype(np.float32)
            positions = positions / self.Lbox
            periodic_wrap(positions)
            positions = reorder_positions(positions)
        else:
            positions = particle_positions.astype(np.float32)

        delta1 = TSC(positions, ncells_1d=self.Nmesh)
        m1 = np.mean(delta1)
        delta1 = (delta1 - m1) / m1
        deltak1 = fft_3D_real(delta1, threads=self.threads)
        del delta1
        gc.collect()

        # Half-cell-shifted paint for Jing-2005 interlacing.
        shift = np.float32(0.5 / self.Nmesh)
        positions_shifted = positions + shift
        periodic_wrap(positions_shifted)
        del positions
        delta2 = TSC(positions_shifted, ncells_1d=self.Nmesh)
        del positions_shifted
        m2 = np.mean(delta2)
        delta2 = (delta2 - m2) / m2
        deltak2 = fft_3D_real(delta2, threads=self.threads)
        del delta2
        gc.collect()

        apply_interlacing(deltak1, deltak2)
        del deltak2
        compensate_mas(deltak1, p=3)  # TSC compensation

        return deltak1

    def compute_density_field(self, particle_positions: np.ndarray,
                              normalize: bool = True) -> np.ndarray:
        """
        Compute density field from particle positions.

        Parameters
        ----------
        particle_positions : ndarray
            Particle positions array of shape (N, 3) in units of Lbox
        normalize : bool, optional
            Whether to normalize positions and apply periodic wrapping

        Returns
        -------
        ndarray
            Real-space density field δ(x)
        """
        deltak_base = self._compute_fourier_density(particle_positions, normalize)

        # Apply Gaussian smoothing at the single global scale
        deltak = deltak_base * self.W
        del deltak_base
        gc.collect()
        delta = ifft_3D_real(deltak, threads=self.threads)

        return delta, deltak
    
    def compute_shear_field(self, deltak: np.ndarray, 
                           halo_positions: np.ndarray) -> np.ndarray:
        """
        Compute shear field q_R² at halo positions from density field.
        
        Parameters
        ----------
        deltak : ndarray
            Fourier-space density field
        halo_positions : ndarray
            Halo positions for interpolation (in units of Lbox)
            
        Returns
        -------
        ndarray
            Shear q_R² values at halo positions
        """
        # Compute all tidal tensor components
        T11 = compute_Tij(0, 0, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T22 = compute_Tij(1, 1, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T33 = compute_Tij(2, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T12 = compute_Tij(0, 1, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T13 = compute_Tij(0, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T23 = compute_Tij(1, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        
        # Compute shear from eigenvalues
        qr2 = compute_shear_eigenvalues(T11, T12, T13, T22, T23, T33)
        del T11, T22, T33, T12, T13, T23
        gc.collect()

        return qr2
    
    def compute_multiscale_properties(self, deltak_base: np.ndarray,
                                      pos: np.ndarray,
                                      halo_rvir: np.ndarray,
                                      compute_shear: bool = True,
                                      r_min: float = 0.5,
                                      r_max: float = 6.0,
                                      dr: float = 0.25,
                                      rvir_factor: float = 2.25) -> dict:
        """
        Compute density (and optionally shear) at a per-halo smoothing scale R = rvir_factor * r_vir.

        The density field is computed at each smoothing radius in [r_min, r_max] (step dr).
        The per-halo value is then obtained by linear interpolation at R = rvir_factor * r_vir,
        clamped to [r_min, r_max].

        Parameters
        ----------
        deltak_base : ndarray
            Unsmoothed MAS-compensated Fourier density field from _compute_fourier_density()
        pos : ndarray
            Halo positions in [0, 1] units, shape (N_halos, 3)
        halo_rvir : ndarray
            Halo virial radii in Mpc/h, shape (N_halos,)
        compute_shear : bool, optional
            Whether to compute per-halo shear q_R² (default: True)
        r_min : float, optional
            Minimum smoothing radius in Mpc/h (default: 0.5)
        r_max : float, optional
            Maximum smoothing radius in Mpc/h (default: 6.0)
        dr : float, optional
            Step size for smoothing radius grid in Mpc/h (default: 0.25)
        rvir_factor : float, optional
            Smoothing scale multiplier: R_h = rvir_factor * r_vir[h] (default: 2.25)

        Returns
        -------
        dict
            - 'delta_h': per-halo density at R = rvir_factor * r_vir
            - 'qr2': per-halo shear q_R² at R = rvir_factor * r_vir (if compute_shear=True)
        """
        R_arr = np.arange(r_min, r_max + dr / 2, dr).astype(np.float32)
        N_R = len(R_arr)
        N_halos = len(pos)

        delta_grid = np.zeros((N_R, N_halos), dtype=np.float32)
        if compute_shear:
            qr2_grid = np.zeros((N_R, N_halos), dtype=np.float32)

        for i, R in enumerate(R_arr):
            print(f"  Scale {i + 1}/{N_R}: R = {R:.2f} Mpc/h")
            W_R = np.exp(-0.5 * self.k2 * float(R) ** 2)
            deltak_R = deltak_base * W_R
            delta_R = ifft_3D_real(deltak_R, threads=self.threads)
            delta_grid[i] = invTSC(delta_R, pos)
            del delta_R
            if compute_shear:
                qr2_grid[i] = self.compute_shear_field(deltak_R, pos)
            del deltak_R

        # Vectorized per-halo linear interpolation along the R axis
        R_halo = np.clip(rvir_factor * halo_rvir, R_arr[0], R_arr[-1]).astype(np.float32)
        idx = np.clip(np.searchsorted(R_arr, R_halo) - 1, 0, N_R - 2)
        t = (R_halo - R_arr[idx]) / (R_arr[idx + 1] - R_arr[idx])
        halo_idx = np.arange(N_halos)
        delta_rvir = (1 - t) * delta_grid[idx, halo_idx] + t * delta_grid[idx + 1, halo_idx]
        del delta_grid

        results = {'delta_h': delta_rvir}
        if compute_shear:
            qr2_rvir = (1 - t) * qr2_grid[idx, halo_idx] + t * qr2_grid[idx + 1, halo_idx]
            del qr2_grid
            results['qr2'] = qr2_rvir
        gc.collect()

        return results

    def compute_environmental_properties(self, particle_positions: np.ndarray,
                                       halo_positions: np.ndarray,
                                       halo_masses: Optional[np.ndarray] = None,
                                       halo_rvir: Optional[np.ndarray] = None,
                                       compute_shear: bool = True,
                                       mass_bins: Union[int, np.ndarray] = 30,
                                       normalize_positions: bool = True,
                                       r_min: float = 0.5,
                                       r_max: float = 6.0,
                                       dr: float = 0.25,
                                       rvir_factor: float = 2.25) -> dict:
        """
        Compute density and shear environmental properties for assembly bias.

        Parameters
        ----------
        particle_positions : ndarray
            Particle positions array of shape (N, 3)
        halo_positions : ndarray
            Halo positions array of shape (M, 3)
        halo_masses : ndarray, optional
            Halo masses for mass-dependent normalization
        halo_rvir : ndarray, optional
            Halo virial radii in Mpc/h, shape (M,). If provided, the smoothing scale
            is set per-halo to rvir_factor * r_vir[h], interpolated from a grid of
            smoothing radii [r_min, r_max] with step dr. If None, the global
            smoothing_radius set at construction time is used.
        compute_shear : bool, optional
            Whether to compute shear field (default: True)
        mass_bins : int or ndarray, optional
            Number of mass bins for normalization (default: 30), or a pre-defined
            array of bin edges in log10(M) units.
        normalize_positions : bool, optional
            Whether to normalize positions (default: True)
        r_min : float, optional
            Minimum smoothing radius in Mpc/h for rvir mode (default: 0.5)
        r_max : float, optional
            Maximum smoothing radius in Mpc/h for rvir mode (default: 6.0)
        dr : float, optional
            Step size for smoothing radius grid in Mpc/h (default: 0.25)
        rvir_factor : float, optional
            Smoothing scale multiplier: R_h = rvir_factor * r_vir[h] (default: 2.25)

        Returns
        -------
        dict
            Dictionary containing:
            - 'delta_h': density field at halo positions
            - 'delta_norm': normalized density field (if halo_masses provided)
            - 'qr2': shear field at halo positions (if compute_shear=True)
            - 'fs_norm': normalized shear field (if halo_masses provided and compute_shear=True)
        """
        # Normalize halo positions
        if normalize_positions:
            pos = halo_positions.copy().astype(np.float32)
            pos = pos / self.Lbox
            periodic_wrap(pos)
        else:
            pos = halo_positions.astype(np.float32)

        if halo_rvir is not None:
            # --- Per-halo virial radius mode ---
            print("Computing multi-scale density and shear fields (per-halo Rvir)...")
            deltak_base = self._compute_fourier_density(particle_positions, normalize_positions)
            ms = self.compute_multiscale_properties(
                deltak_base, pos, halo_rvir, compute_shear,
                r_min, r_max, dr, rvir_factor
            )
            del deltak_base
            gc.collect()
            deltah = ms['delta_h']
            results = {'delta_h': deltah}
            if compute_shear:
                results['qr2'] = ms['qr2']
            del ms
            gc.collect()
        else:
            # --- Single global smoothing radius mode ---
            print("Computing density field...")
            delta, deltak = self.compute_density_field(particle_positions, normalize_positions)
            deltah = invTSC(delta, pos)
            del delta
            gc.collect()
            results = {'delta_h': deltah}
            if compute_shear:
                print("Computing shear field...")
                results['qr2'] = self.compute_shear_field(deltak, pos)
            del deltak
            gc.collect()

        # Mass-dependent normalization (same for both modes)
        if halo_masses is not None:
            print("Applying mass-dependent normalization...")
            
            # Create mass bins
            log_masses = np.log10(halo_masses)
            if isinstance(mass_bins, np.ndarray):
                bins_mass = mass_bins
            else:
                low_m = log_masses.min()
                high_m = log_masses.max()
                bins_mass = np.linspace(low_m, high_m, mass_bins + 1)
            
            # Assign halos to mass bins
            _, _, bin_number = stats.binned_statistic(log_masses, log_masses, 
                                                    statistic='count', bins=bins_mass)
            
            # Normalize density and shear within each mass bin
            delta_norm = np.zeros_like(deltah)
            if compute_shear:
                fs_norm = np.zeros_like(results['qr2'])
            
            for n in range(len(bins_mass) - 1):
                cond = bin_number == n + 1
                if np.sum(cond) == 0:
                    continue
                    
                subsample = deltah[cond]
                if len(subsample) != 0:
                    fdi = normalize_distribution(1 + subsample)
                    delta_norm[cond] = fdi
                    
                if compute_shear:
                    qri = results['qr2'][cond]
                    if len(qri) != 0:
                        fsi = normalize_distribution(1 + qri)
                        fs_norm[cond] = fsi
            
            results['delta_norm'] = delta_norm
            if compute_shear:
                results['fs_norm'] = fs_norm
            del log_masses, bin_number, bins_mass
            gc.collect()

        return results


def compute_assembly_bias_properties(halo_catalogue: Union[str, pd.DataFrame],
                                   particle_positions: Union[str, np.ndarray],
                                   compute_shear: bool = True,
                                   Nmesh: int = 1024,
                                   Lbox: float = 681,
                                   smoothing_radius: float = 1.0,
                                   position_columns: Tuple[str, str, str] = ('x', 'y', 'z'),
                                   mass_column: str = 'mass',
                                   rvir_column: Optional[str] = None,
                                   r_min: float = 0.5,
                                   r_max: float = 6.0,
                                   dr: float = 0.25,
                                   rvir_factor: float = 2.25,
                                   mass_bins: Union[int, np.ndarray] = 30,
                                   threads: int = 32) -> dict:
    """
    High-level function to compute assembly bias environmental properties.

    Parameters
    ----------
    halo_catalogue : str or DataFrame
        Path to halo catalogue parquet file or DataFrame with halo data
    particle_positions : str or ndarray
        Path to particle catalogue parquet file or array of positions
    compute_shear : bool, optional
        Whether to compute shear field (default: True)
    Nmesh : int, optional
        Number of mesh cells per dimension (default: 1024)
    Lbox : float, optional
        Box size in Mpc/h (default: 681)
    smoothing_radius : float, optional
        Gaussian smoothing radius in Mpc/h for the single-scale mode (default: 1.0).
        Ignored when rvir_column is provided.
    position_columns : tuple, optional
        Column names for x, y, z positions (default: ('x', 'y', 'z'))
    mass_column : str, optional
        Column name for halo masses (default: 'mass')
    rvir_column : str, optional
        Column name for halo virial radii in Mpc/h. If provided, per-halo smoothing
        at R = rvir_factor * r_vir is used instead of the single global smoothing_radius.
    r_min : float, optional
        Minimum smoothing radius in Mpc/h for rvir mode (default: 0.5)
    r_max : float, optional
        Maximum smoothing radius in Mpc/h for rvir mode (default: 6.0)
    dr : float, optional
        Step size for smoothing radius grid in Mpc/h (default: 0.25)
    rvir_factor : float, optional
        Smoothing scale multiplier: R_h = rvir_factor * r_vir[h] (default: 2.25)
    mass_bins : int or ndarray, optional
        Number of mass bins for normalization (default: 30), or a pre-defined
        array of bin edges in log10(M) units.
    threads : int, optional
        Number of threads for FFT operations (default: 32)

    Returns
    -------
    dict
        Dictionary with environmental properties as described in compute_environmental_properties
    """
    print("Loading data...")

    # Load halo catalogue
    if isinstance(halo_catalogue, str):
        df_halo = pd.read_parquet(halo_catalogue)
    else:
        df_halo = halo_catalogue

    # Extract halo positions and masses
    halo_positions = np.column_stack([
        df_halo[position_columns[0]].values,
        df_halo[position_columns[1]].values,
        df_halo[position_columns[2]].values
    ])

    halo_masses = df_halo[mass_column].values if mass_column in df_halo.columns else None
    halo_rvir = df_halo[rvir_column].values if rvir_column is not None else None
    if isinstance(halo_catalogue, str):
        del df_halo
        gc.collect()

    # Load particle positions
    if isinstance(particle_positions, str):
        df_part = pd.read_parquet(particle_positions)
        particle_pos = np.column_stack([
            df_part[position_columns[0]].values,
            df_part[position_columns[1]].values,
            df_part[position_columns[2]].values
        ])
        del df_part
        gc.collect()
    else:
        particle_pos = particle_positions

    # Initialize calculator
    calc = AssemblyBiasEnvironment(Nmesh=Nmesh, Lbox=Lbox,
                                   smoothing_radius=smoothing_radius, threads=threads)

    # Compute environmental properties
    results = calc.compute_environmental_properties(
        particle_pos, halo_positions, halo_masses,
        halo_rvir=halo_rvir,
        compute_shear=compute_shear,
        mass_bins=mass_bins,
        r_min=r_min, r_max=r_max, dr=dr, rvir_factor=rvir_factor
    )
    if isinstance(particle_positions, str):
        del particle_pos
    del halo_positions, halo_masses, halo_rvir
    gc.collect()

    print("Assembly bias environmental properties computed successfully!")
    return results


# Example usage
if __name__ == "__main__":
    # Example with the paths from your do_shear.py
    halo_path = '/feynman/work/dap/lceg/rp269101/stuff/flamingo/snapshots_hydro/NISP_catalogue_flamingo_withSHEAR.parquet'
    particle_path = '/feynman/work/dap/lceg/rp269101/stuff/flamingo/snapshots_hydro/hydro_flamingo_0058_downsampled_0.1percent.parquet'
    
    # Compute assembly bias properties
    results = compute_assembly_bias_properties(
        halo_catalogue=halo_path,
        particle_positions=particle_path,
        compute_shear=True,
        Nmesh=1024,
        Lbox=681,
        smoothing_radius=1.0
    )
    
    print("Available results:")
    for key, value in results.items():
        print(f"  {key}: shape {value.shape if hasattr(value, 'shape') else type(value)}")