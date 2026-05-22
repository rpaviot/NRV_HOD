"""
Field & mesh utilities: TSC mass-assignment + interlacing + MAS compensation,
P(k) estimation, and halo environmental properties for assembly bias.

Combines what used to live in `measure_pk.py` (P(k) estimator) and
`assembly_bias_environment.py` (environment ranking). Both modules built on
the same pysco primitives; merging them removes the cross-module private
import and gives "anything that lays a field on a mesh" one home.

Conventions
-----------
- Box size Lbox in Mpc/h; wavenumbers k in h/Mpc; P(k) in (Mpc/h)^3.
- Forward FFT is unnormalized (numpy.rfftn / FFTW FORWARD); P(k) is
  normalized by ``(Lbox / Nmesh**2)**3`` to match pysco's convention.

Public API
----------
- ``PowerSpectrumEstimator``, ``log_kbins`` — periodic-box P(k) estimator.
- ``AssemblyBiasEnvironment``, ``compute_assembly_bias_properties`` — density
  and shear environmental properties at halo positions, with mass-binned
  rank normalization for assembly bias HOD modelling
  (Paviot et al. 2024, Eq. 15-16).
"""

import gc
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
from numpy import linalg as LA
from scipy import stats

from pysco.mesh import TSC, invTSC
from pysco.utils import periodic_wrap, argsort_par, injection_with_indices
from pysco.fourier import fft_3D_real, ifft_3D_real
from pysco import morton


# ---------------------------------------------------------------------------
# Shared FFT / MAS helpers
# ---------------------------------------------------------------------------


def fourier_axes(N: int, Lbox: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcastable Fourier wavevectors (kx, ky, kz) in float32."""
    kx = (np.fft.fftfreq(N, d=Lbox / N) * 2 * np.pi).astype(np.float32)[:, None, None]
    ky = (np.fft.fftfreq(N, d=Lbox / N) * 2 * np.pi).astype(np.float32)[None, :, None]
    kz = (np.fft.rfftfreq(N, d=Lbox / N) * 2 * np.pi).astype(np.float32)[None, None, :]
    return kx, ky, kz


def grid_positions(ncells_1d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centred grid positions in [0, 1] for a cubic mesh."""
    delta = 1.0 / ncells_1d
    x = delta * (np.arange(ncells_1d) + 0.5)[:, None, None]
    y = delta * (np.arange(ncells_1d) + 0.5)[None, :, None]
    z = delta * (np.arange(ncells_1d) + 0.5)[None, None, :]
    return x, y, z


def reorder_positions(positions: np.ndarray) -> np.ndarray:
    """Reorder positions by Morton key for better cache locality during TSC."""
    index = morton.positions_to_keys(positions)
    nthreads = numba.get_num_threads()

    if nthreads > 1:
        arg = argsort_par(index, nthreads)
    else:
        arg = np.argsort(index)

    index = 0  # free memory
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
    """In-place deconvolution by the MAS window (Jing 2005). p=3 for TSC."""
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
    """Jing-2005 interlacing: store 0.5 (F1 + e^{ik·Δx} F2) into F1 in-place."""
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
    """Tidal tensor component T_ij evaluated at halo positions."""
    Tij_k = (phase_axes[i] * phase_axes[j] / k2) * deltak
    Tij_x = ifft_3D_real(Tij_k, threads=threads)
    Tij_h = invTSC(Tij_x, halo_positions)
    return Tij_h


def normalize_distribution(p: np.ndarray) -> np.ndarray:
    """Map ln(p) to [-1, 1] via (median, 2nd, 98th percentiles), then clip."""
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
    """q_R² = ½ Σ_{i<j} (L_i − L_j)² of the tidal tensor eigenvalues."""
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
        qr2[i] = (1. / 2.) * ((L2 - L1)**2 + (L3 - L1)**2 + (L3 - L2)**2)

    return qr2


# ---------------------------------------------------------------------------
# PowerSpectrumEstimator
# ---------------------------------------------------------------------------


class PowerSpectrumEstimator:
    """Periodic-box P(k) / cross-P(k) estimator.

    TSC + Jing-2005 interlacing + MAS compensation (p=3). Same recipe as
    ``AssemblyBiasEnvironment._compute_fourier_density``, so density fields
    produced here are bit-compatible with that pipeline.
    """

    def __init__(self, Nmesh: int, Lbox: float, threads: int = 32):
        self.Nmesh = Nmesh
        self.Lbox = Lbox
        self.threads = threads
        self.phase_axes = fourier_axes(Nmesh, Lbox)
        k2 = sum(ki ** 2 for ki in self.phase_axes)
        self.k_mag = np.sqrt(k2).astype(np.float32)
        # Normalization for an unnormalized forward FFT (numpy.rfftn / FFTW
        # FORWARD), matching pysco's `Pk *= (L/N**2)**3` in solver.py:138.
        self.pk_norm = float(Lbox / (Nmesh ** 2)) ** 3
        # Orszag 2/3 cut: anti-aliasing limit used by pysco.fourier_grid_to_Pk.
        self.k_orszag = (2.0 / 3.0) * np.pi * Nmesh / Lbox

    def delta_k(self, positions: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Density-contrast field in Fourier space, complex64 (Nmesh, Nmesh, Nmesh//2+1)."""
        if normalize:
            pos = positions.astype(np.float32, copy=True) / self.Lbox
            periodic_wrap(pos)
            pos = reorder_positions(pos)
        else:
            pos = positions.astype(np.float32, copy=False)

        delta1 = TSC(pos, ncells_1d=self.Nmesh)
        m1 = np.mean(delta1)
        delta1 = (delta1 - m1) / m1
        deltak1 = fft_3D_real(delta1, threads=self.threads)
        del delta1
        gc.collect()

        # Half-cell-shifted paint for Jing-2005 interlacing.
        shift = np.float32(0.5 / self.Nmesh)
        pos_shifted = pos + shift
        periodic_wrap(pos_shifted)
        del pos
        delta2 = TSC(pos_shifted, ncells_1d=self.Nmesh)
        del pos_shifted
        m2 = np.mean(delta2)
        delta2 = (delta2 - m2) / m2
        deltak2 = fft_3D_real(delta2, threads=self.threads)
        del delta2
        gc.collect()

        apply_interlacing(deltak1, deltak2)
        del deltak2
        compensate_mas(deltak1, p=3)
        return deltak1

    def auto_pk(
        self,
        deltak: np.ndarray,
        kbins: np.ndarray,
        n_tracer: int,
        subtract_shot_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Auto P(k) with optional Poisson shot-noise subtraction (Lbox^3 / n_tracer)."""
        field = np.real(deltak * np.conj(deltak)).astype(np.float64)
        k_centers, P_k, n_modes = self._bin_field(field, kbins)
        P_k *= self.pk_norm
        if subtract_shot_noise and n_tracer > 0:
            P_k -= float(self.Lbox ** 3) / float(n_tracer)
        return k_centers, P_k, n_modes

    def cross_pk(
        self,
        deltak1: np.ndarray,
        deltak2: np.ndarray,
        kbins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross P(k) between two independent tracers (no shot-noise subtraction)."""
        field = np.real(deltak1 * np.conj(deltak2)).astype(np.float64)
        k_centers, P_k, n_modes = self._bin_field(field, kbins)
        P_k *= self.pk_norm
        return k_centers, P_k, n_modes

    def _bin_field(
        self,
        field: np.ndarray,
        kbins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Hermitian-symmetric weights: kz=0 and kz=Nyquist planes count once,
        # all other kz count twice (matches a full-grid sum).
        N = self.Nmesh
        weight = np.ones(N // 2 + 1, dtype=np.float64) * 2.0
        weight[0] = 1.0
        if N % 2 == 0:
            weight[-1] = 1.0
        w3d = np.broadcast_to(weight[None, None, :], field.shape)

        k_flat = self.k_mag.ravel()
        f_flat = (field * w3d).ravel()
        w_flat = w3d.ravel()

        idx = np.digitize(k_flat, kbins) - 1
        valid = (idx >= 0) & (idx < len(kbins) - 1)
        idx = idx[valid]
        f_flat = f_flat[valid]
        w_flat = w_flat[valid]
        k_flat = k_flat[valid]

        nbins = len(kbins) - 1
        n_modes = np.bincount(idx, weights=w_flat, minlength=nbins)
        sum_P = np.bincount(idx, weights=f_flat, minlength=nbins)
        sum_k = np.bincount(idx, weights=k_flat * w_flat, minlength=nbins)

        with np.errstate(invalid="ignore", divide="ignore"):
            P_k = np.where(n_modes > 0, sum_P / n_modes, np.nan)
            k_centers = np.where(n_modes > 0, sum_k / n_modes, np.nan)
        return k_centers, P_k, n_modes


def log_kbins(
    Lbox: float,
    Nmesh: int,
    n_bins: int = 20,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
) -> np.ndarray:
    """Log-spaced k-bin edges between the box fundamental and the Orszag 2/3 cutoff."""
    k_fund = 2.0 * np.pi / Lbox
    k_orszag = (2.0 / 3.0) * np.pi * Nmesh / Lbox  # pysco default cutoff
    k_min = k_min if k_min is not None else k_fund
    k_max = k_max if k_max is not None else k_orszag
    return np.geomspace(k_min, k_max, n_bins + 1)


# ---------------------------------------------------------------------------
# AssemblyBiasEnvironment
# ---------------------------------------------------------------------------


class AssemblyBiasEnvironment:
    """Density and shear environmental properties at halo positions.

    Implements Paviot et al. (2024) Eq. 15-16: paint a smoothed density
    contrast field, evaluate δ(x) and the tidal tensor shear q_R² at halo
    positions, then mass-bin rank-normalize each to [-1, 1].
    """

    def __init__(self, Nmesh: int = 1024, Lbox: float = 681,
                 smoothing_radius: float = 1.0, threads: int = 32):
        self.Nmesh = Nmesh
        self.Lbox = Lbox
        self.smoothing_radius = smoothing_radius
        self.threads = threads
        self.step = Lbox / Nmesh

        self.phase_axes = fourier_axes(Nmesh, Lbox)
        self.k2 = sum(ki**2 for ki in self.phase_axes)
        self.k2[0, 0, 0] = np.inf  # avoid /0

        self.W = np.exp(-0.5 * self.k2 * (smoothing_radius**2))

    def _compute_fourier_density(self, particle_positions: np.ndarray,
                                 normalize: bool = True) -> np.ndarray:
        """MAS-compensated Fourier density field (unsmoothed)."""
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
        compensate_mas(deltak1, p=3)

        return deltak1

    def compute_density_field(self, particle_positions: np.ndarray,
                              normalize: bool = True) -> np.ndarray:
        """Real-space δ(x) at the global smoothing scale, plus its Fourier field."""
        deltak_base = self._compute_fourier_density(particle_positions, normalize)

        deltak = deltak_base * self.W
        del deltak_base
        gc.collect()
        delta = ifft_3D_real(deltak, threads=self.threads)

        return delta, deltak

    def compute_shear_field(self, deltak: np.ndarray,
                            halo_positions: np.ndarray) -> np.ndarray:
        """Shear q_R² at halo positions from a smoothed Fourier density field."""
        T11 = compute_Tij(0, 0, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T22 = compute_Tij(1, 1, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T33 = compute_Tij(2, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T12 = compute_Tij(0, 1, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T13 = compute_Tij(0, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)
        T23 = compute_Tij(1, 2, deltak, self.phase_axes, halo_positions, self.k2, self.threads)

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
        """Per-halo δ (and shear) at R = rvir_factor * r_vir via interpolation on a grid of R."""
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
        """δ, q_R², and mass-binned rank-normalized (f_A, f_B) at halo positions."""
        if normalize_positions:
            pos = halo_positions.copy().astype(np.float32)
            pos = pos / self.Lbox
            periodic_wrap(pos)
        else:
            pos = halo_positions.astype(np.float32)

        if halo_rvir is not None:
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

        if halo_masses is not None:
            print("Applying mass-dependent normalization...")

            log_masses = np.log10(halo_masses)
            if isinstance(mass_bins, np.ndarray):
                bins_mass = mass_bins
            else:
                low_m = log_masses.min()
                high_m = log_masses.max()
                bins_mass = np.linspace(low_m, high_m, mass_bins + 1)

            _, _, bin_number = stats.binned_statistic(log_masses, log_masses,
                                                     statistic='count', bins=bins_mass)

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
    """High-level wrapper: load halos/particles, compute (δ, q_R²) at halo positions, normalize."""
    print("Loading data...")

    if isinstance(halo_catalogue, str):
        df_halo = pd.read_parquet(halo_catalogue)
    else:
        df_halo = halo_catalogue

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

    calc = AssemblyBiasEnvironment(Nmesh=Nmesh, Lbox=Lbox,
                                   smoothing_radius=smoothing_radius, threads=threads)

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
