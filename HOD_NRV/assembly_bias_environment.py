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
from pysco.utils import periodic_wrap, argsort_par
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
    positions = morton.injection_with_indices(arg, positions)
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
def apply_interlacing(fourier1: np.ndarray) -> None:
    """
    Apply Fourier-space interlacing to reduce aliasing (Jing 2005).
    
    Parameters
    ----------
    fourier1 : ndarray, complex64
        Fourier transform of the unshifted density field
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
                # dot product k · Δx
                phase = np.exp(1j * np.pi * (kx + ky + kz) / N)
                # interlacing: average of unshifted and shifted contribution
                fourier1[i, j, k] = 0.5 * (fourier1[i, j, k] + phase * fourier1[i, j, k])


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
        if normalize:
            positions = particle_positions.copy().astype(np.float32)
            positions = positions / self.Lbox
            periodic_wrap(positions)
            positions = reorder_positions(positions)
        else:
            positions = particle_positions.astype(np.float32)
            
        # Create density field using TSC assignment
        delta = TSC(positions, ncells_1d=self.Nmesh)
        delta = (delta - np.mean(delta)) / np.mean(delta)
        
        # Fourier transform and apply corrections
        deltak = fft_3D_real(delta, threads=self.threads)
        apply_interlacing(deltak)
        compensate_mas(deltak, p=3)  # TSC compensation
        
        # Apply Gaussian smoothing
        deltak = deltak * self.W
        
        # Transform back to real space
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
        
        return qr2
    
    def compute_environmental_properties(self, particle_positions: np.ndarray,
                                       halo_positions: np.ndarray,
                                       halo_masses: Optional[np.ndarray] = None,
                                       compute_shear: bool = True,
                                       mass_bins: int = 30,
                                       normalize_positions: bool = True) -> dict:
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
        compute_shear : bool, optional
            Whether to compute shear field (default: True)
        mass_bins : int, optional
            Number of mass bins for normalization (default: 30)
        normalize_positions : bool, optional
            Whether to normalize positions (default: True)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'delta_h': density field at halo positions
            - 'delta_norm': normalized density field (if halo_masses provided)
            - 'qr2': shear field at halo positions (if compute_shear=True)
            - 'fs_norm': normalized shear field (if halo_masses provided and compute_shear=True)
        """
        print("Computing density field...")
        
        # Normalize halo positions
        if normalize_positions:
            pos = halo_positions.copy().astype(np.float32)
            pos = pos / self.Lbox
            periodic_wrap(pos)
        else:
            pos = halo_positions.astype(np.float32)
        
        # Compute density field
        delta, deltak = self.compute_density_field(particle_positions, normalize_positions)
        
        # Interpolate density at halo positions
        deltah = invTSC(delta, pos)
        
        results = {'delta_h': deltah}
        
        # Compute shear if requested
        if compute_shear:
            print("Computing shear field...")
            qr2 = self.compute_shear_field(deltak, pos)
            results['qr2'] = qr2
        
        # Mass-dependent normalization
        if halo_masses is not None:
            print("Applying mass-dependent normalization...")
            
            # Create mass bins
            log_masses = np.log10(halo_masses)
            low_m = log_masses.min()
            high_m = log_masses.max()
            bins_mass = np.linspace(low_m, high_m, mass_bins + 1)
            
            # Assign halos to mass bins
            _, _, bin_number = stats.binned_statistic(log_masses, log_masses, 
                                                    statistic='count', bins=bins_mass)
            
            # Normalize density and shear within each mass bin
            delta_norm = np.zeros_like(deltah)
            if compute_shear:
                fs_norm = np.zeros_like(qr2)
            
            for n in range(len(bins_mass) - 1):
                cond = bin_number == n + 1
                if np.sum(cond) == 0:
                    continue
                    
                subsample = deltah[cond]
                if len(subsample) != 0:
                    fdi = normalize_distribution(1 + subsample)
                    delta_norm[cond] = fdi
                    
                if compute_shear:
                    qri = qr2[cond]
                    if len(qri) != 0:
                        fsi = normalize_distribution(1 + qri)
                        fs_norm[cond] = fsi
            
            results['delta_norm'] = delta_norm
            if compute_shear:
                results['fs_norm'] = fs_norm
        
        return results


def compute_assembly_bias_properties(halo_catalogue: Union[str, pd.DataFrame],
                                   particle_positions: Union[str, np.ndarray],
                                   compute_shear: bool = True,
                                   Nmesh: int = 1024,
                                   Lbox: float = 681,
                                   smoothing_radius: float = 1.0,
                                   position_columns: Tuple[str, str, str] = ('x', 'y', 'z'),
                                   mass_column: str = 'mass',
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
        Gaussian smoothing radius in Mpc/h (default: 1.0)
    position_columns : tuple, optional
        Column names for x, y, z positions (default: ('x', 'y', 'z'))
    mass_column : str, optional
        Column name for halo masses (default: 'mass')
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
        particle_pos, halo_positions, halo_masses, compute_shear=compute_shear
    )
    
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