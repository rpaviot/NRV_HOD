"""
P(k) and cross-P(k) estimator for tracer fields in a periodic box.

Built from the same pysco primitives used in `assembly_bias_environment.py`
(TSC + interlacing + MAS compensation), so its delta_k() is bit-compatible
with the existing density-field pipeline.

All wavenumbers and box sizes are in h/Mpc and Mpc/h respectively. Power
spectra are returned in (Mpc/h)^3.
"""

from typing import Optional, Tuple

import gc
import numpy as np

from pysco.mesh import TSC
from pysco.fourier import fft_3D_real
from pysco.utils import periodic_wrap

from .assembly_bias_environment import (
    fourier_axes,
    apply_interlacing,
    compensate_mas,
    reorder_positions,
)


class PowerSpectrumEstimator:
    """
    Periodic-box P(k) / cross-P(k) estimator.

    Parameters
    ----------
    Nmesh : int
        Mesh cells per dimension.
    Lbox : float
        Box size in Mpc/h.
    threads : int
        Threads for FFT.
    """

    def __init__(self, Nmesh: int, Lbox: float, threads: int = 32):
        self.Nmesh = Nmesh
        self.Lbox = Lbox
        self.threads = threads
        self.phase_axes = fourier_axes(Nmesh, Lbox)
        k2 = sum(ki ** 2 for ki in self.phase_axes)
        self.k_mag = np.sqrt(k2).astype(np.float32)
        # Normalization for an unnormalized forward FFT (numpy.rfftn / FFTW
        # FORWARD), matching pysco's convention `Pk *= (L/N**2)**3` in
        # solver.py:138.
        self.pk_norm = float(Lbox / (Nmesh ** 2)) ** 3
        # Orszag 2/3 cut: anti-aliasing limit used by pysco.fourier_grid_to_Pk.
        self.k_orszag = (2.0 / 3.0) * np.pi * Nmesh / Lbox

    def delta_k(self, positions: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Density-contrast field in Fourier space.

        Returns complex64 array shaped (Nmesh, Nmesh, Nmesh//2 + 1).
        TSC + interlacing + MAS compensation (p=3) — same recipe as
        `AssemblyBiasEnvironment._compute_fourier_density`.
        """
        if normalize:
            pos = positions.astype(np.float32, copy=True) / self.Lbox
            periodic_wrap(pos)
            pos = reorder_positions(pos)
        else:
            pos = positions.astype(np.float32, copy=False)

        delta = TSC(pos, ncells_1d=self.Nmesh)
        del pos
        mean = np.mean(delta)
        delta = (delta - mean) / mean

        deltak = fft_3D_real(delta, threads=self.threads)
        del delta
        gc.collect()
        apply_interlacing(deltak)
        compensate_mas(deltak, p=3)
        return deltak

    def auto_pk(
        self,
        deltak: np.ndarray,
        kbins: np.ndarray,
        n_tracer: int,
        subtract_shot_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Auto power spectrum P(k) of a single tracer.

        Parameters
        ----------
        deltak : ndarray
            From `delta_k()`.
        kbins : ndarray
            k-bin edges in h/Mpc (length nbins+1).
        n_tracer : int
            Number of tracer points (for shot-noise subtraction).
        subtract_shot_noise : bool
            If True, subtract Lbox^3 / n_tracer.

        Returns
        -------
        k_centers, P_k, n_modes
        """
        field = np.real(deltak * np.conj(deltak)).astype(np.float64)
        k_centers, P_k, n_modes = self._bin_field(field, kbins)
        P_k *= self.pk_norm
        if subtract_shot_noise and n_tracer > 0:
            # Shot noise of a Poisson tracer field measured on a unit-mean
            # density grid: V_box / N_tracer, in the same (Mpc/h)^3 units.
            P_k -= float(self.Lbox ** 3) / float(n_tracer)
        return k_centers, P_k, n_modes

    def cross_pk(
        self,
        deltak1: np.ndarray,
        deltak2: np.ndarray,
        kbins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cross power spectrum between two independent tracers (no shot-noise
        subtraction).
        """
        field = np.real(deltak1 * np.conj(deltak2)).astype(np.float64)
        k_centers, P_k, n_modes = self._bin_field(field, kbins)
        P_k *= self.pk_norm
        return k_centers, P_k, n_modes

    def _bin_field(
        self,
        field: np.ndarray,
        kbins: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spherical-shell average of a 3D Fourier-space scalar field on the
        rfftn grid. Hermitian-symmetric weights: kz=0 and kz=Nyquist planes
        count once, all other kz count twice (matches a full-grid sum).
        """
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
    """
    Convenience: log-spaced k-bin edges between the box fundamental and the
    Nyquist frequency.
    """
    k_fund = 2.0 * np.pi / Lbox
    k_orszag = (2.0 / 3.0) * np.pi * Nmesh / Lbox  # pysco default cutoff
    k_min = k_min if k_min is not None else k_fund
    k_max = k_max if k_max is not None else k_orszag
    return np.geomspace(k_min, k_max, n_bins + 1)
