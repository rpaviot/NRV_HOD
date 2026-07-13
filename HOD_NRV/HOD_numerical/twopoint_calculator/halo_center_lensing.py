"""
Optimized DeltaSigma Calculator using Precomputed Halo-Center Profiles.

This module provides a fast galaxy-galaxy lensing calculation by precomputing
DeltaSigma profiles at halo centers. Since central galaxies are exactly at
halo centers, their lensing contribution can be looked up instantly rather
than computed at runtime.

The approach splits the calculation:
- **Centrals**: Direct lookup from precomputed halo-center profiles (instant)
- **Satellites**: Runtime computation via pycorr (smaller population)

This provides ~4-10x speedup depending on satellite fraction, without the
interpolation errors of spatial interpolation approaches.

Key Insight
-----------
Central galaxies are positioned exactly at halo centers:
    cent_positions = positions[is_cent]  # from population_engine.py:118

Therefore, precomputing DeltaSigma at each halo center allows instant lookup
by halo index - no interpolation needed.

The precomputation uses KD-tree + histogram to compute xi_gm per halo,
then feeds into the standard DeltaSigmaCalculator pipeline.

References
----------
.. [1] Yuan et al. (2021), MNRAS, arXiv:2110.11412 (AbacusHOD paper)
"""

import os
import time
import multiprocessing as mp
import numpy as np
import h5py
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration
from .standard_two_point_calculator import (
    compute_corr, DeltaSigmaCalculator, binavg_2D
)

def periodic_distance(pos1: np.ndarray, pos2: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Compute distance with periodic boundary conditions.

    Parameters
    ----------
    pos1 : np.ndarray, shape (N, 3) or (3,)
        First set of positions [Mpc/h]
    pos2 : np.ndarray, shape (3,)
        Second position (typically a center) [Mpc/h]
    boxsize : float
        Simulation box size [Mpc/h]

    Returns
    -------
    distances : np.ndarray
        Distances accounting for periodic boundaries [Mpc/h]
    """
    delta = pos1 - pos2
    delta -= boxsize * np.round(delta / boxsize)
    return np.sqrt(np.einsum('ij,ij->i', delta, delta))


def compute_xigm_at_position(
    galaxy_pos: np.ndarray,
    nearby_particles: np.ndarray,
    r_bins: np.ndarray,
    boxsize: float,
    n_particles_total: int,
    volume_total: float
) -> np.ndarray:
    """
    Compute galaxy-matter correlation function xi_gm(r) at a single position.

    Counts particles in spherical shells and compares to expected density.

    Parameters
    ----------
    galaxy_pos : np.ndarray, shape (3,)
        Galaxy position [Mpc/h]
    nearby_particles : np.ndarray, shape (N_nearby, 3)
        Positions of nearby particles [Mpc/h]
    r_bins : np.ndarray
        Radial bin edges [Mpc/h]
    boxsize : float
        Simulation box size for periodic boundaries [Mpc/h]
    n_particles_total : int
        Total number of particles in full catalog
    volume_total : float
        Total volume of simulation [Mpc/h]^3

    Returns
    -------
    xi_gm : np.ndarray, shape (len(r_bins)-1,)
        Galaxy-matter correlation function
    """
    r = periodic_distance(nearby_particles, galaxy_pos, boxsize)
    counts, _ = np.histogram(r, bins=r_bins)
    shell_volumes = (4.0/3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    n_observed = counts / shell_volumes
    n_mean = n_particles_total / volume_total
    xi_gm = (n_observed / n_mean) - 1.0
    return xi_gm


# ---------------------------------------------------------------------------
# Multiprocessing shared state + worker functions
# ---------------------------------------------------------------------------

_shared = {}


def _process_batch_prequeried(halo_indices):
    """Worker: compute xi_gm + DeltaSigma for halos with pre-queried indices."""
    s = _shared
    particle_positions = s['particle_positions']
    halo_positions = s['halo_positions']
    bins_comp = s['bins_comp']
    shell_volumes = s['shell_volumes']
    n_mean = s['n_mean']
    Lbox = s['Lbox']
    r_centers = s['r_centers']
    rp_bins = s['rp_bins']
    RHO_M = s['RHO_M']
    chi_max = s['chi_max']
    n_rp_bins = s['n_rp_bins']
    idx_lists = s['idx_lists']
    n_comp_bins = len(bins_comp) - 1

    halo_bin_index = s.get('halo_bin_index')
    tabulate = halo_bin_index is not None
    if tabulate:
        xi_sum = np.zeros((s['n_bins'], n_comp_bins))
        xi_count = np.zeros(s['n_bins'])

    results = np.empty((len(halo_indices), n_rp_bins))

    for k, i in enumerate(halo_indices):
        idx = idx_lists[i]
        if len(idx) == 0:
            xi_gm = -np.ones(n_comp_bins)
        else:
            delta = particle_positions[idx] - halo_positions[i]
            delta -= Lbox * np.round(delta / Lbox)
            r = np.sqrt(np.einsum('ij,ij->i', delta, delta))
            counts, _ = np.histogram(r, bins=bins_comp)
            xi_gm = (counts / shell_volumes / n_mean) - 1.0

        if tabulate:
            xi_sum[halo_bin_index[i]] += xi_gm
            xi_count[halo_bin_index[i]] += 1

        calc = DeltaSigmaCalculator(r_centers, xi_gm, RHO_M, chi_max=chi_max)
        results[k] = calc.compute_deltasigma_averaged(rp_bins)

    if tabulate:
        return results, xi_sum, xi_count
    return results


def _process_batch_with_query(halo_indices):
    """Worker: query KD-tree + compute xi_gm + DeltaSigma for halos."""
    s = _shared
    particle_positions = s['particle_positions']
    halo_positions = s['halo_positions']
    bins_comp = s['bins_comp']
    shell_volumes = s['shell_volumes']
    n_mean = s['n_mean']
    Lbox = s['Lbox']
    r_centers = s['r_centers']
    rp_bins = s['rp_bins']
    RHO_M = s['RHO_M']
    chi_max = s['chi_max']
    n_rp_bins = s['n_rp_bins']
    search_radius = s['search_radius']
    kdtree = s['kdtree']
    n_comp_bins = len(bins_comp) - 1

    halo_bin_index = s.get('halo_bin_index')
    tabulate = halo_bin_index is not None
    if tabulate:
        xi_sum = np.zeros((s['n_bins'], n_comp_bins))
        xi_count = np.zeros(s['n_bins'])

    results = np.empty((len(halo_indices), n_rp_bins))
    sub_chunk_size = 10
    ptr = 0

    for start in range(0, len(halo_indices), sub_chunk_size):
        sub_indices = halo_indices[start:start + sub_chunk_size]
        sub_positions = halo_positions[sub_indices]
        sub_idx_lists = kdtree.query_ball_point(
            sub_positions, r=search_radius, workers=1, return_sorted=False
        )

        for j, i in enumerate(sub_indices):
            idx = sub_idx_lists[j]
            if len(idx) == 0:
                xi_gm = -np.ones(n_comp_bins)
            else:
                delta = particle_positions[idx] - halo_positions[i]
                delta -= Lbox * np.round(delta / Lbox)
                r = np.sqrt(np.einsum('ij,ij->i', delta, delta))
                counts, _ = np.histogram(r, bins=bins_comp)
                xi_gm = (counts / shell_volumes / n_mean) - 1.0

            if tabulate:
                xi_sum[halo_bin_index[i]] += xi_gm
                xi_count[halo_bin_index[i]] += 1

            calc = DeltaSigmaCalculator(r_centers, xi_gm, RHO_M, chi_max=chi_max)
            results[ptr] = calc.compute_deltasigma_averaged(rp_bins)
            ptr += 1

    if tabulate:
        return results, xi_sum, xi_count
    return results


class HaloCenterLensingCache:
    """
    Cache for precomputed DeltaSigma profiles at halo centers.

    This class stores and manages precomputed lensing profiles that can be
    looked up by halo index. Since central galaxies are exactly at halo
    centers, this enables instant DeltaSigma retrieval without interpolation.

    Parameters
    ----------
    positions : np.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    deltasigma : np.ndarray, shape (N_halos, N_rp_bins)
        Precomputed DeltaSigma profiles [Msun h/pc^2]
    rp_bins : np.ndarray, shape (N_rp_bins+1,)
        Projected separation bin edges [Mpc/h]
    metadata : dict, optional
        Metadata (cosmology, Lbox, RHO_M, etc.)

    Examples
    --------
    >>> cache = precompute_halo_center_lensing(
    ...     halo_positions, particle_positions, Lbox, rsd_axis, RHO_M, rp_bins
    ... )
    >>> cache.save('halo_lensing_cache.h5')
    >>>
    >>> cache = HaloCenterLensingCache.load('halo_lensing_cache.h5')
    >>> rp, ds = halo.compute_galaxy_lensing_optimized(rp_bins, cache)
    """

    def __init__(
        self,
        positions: np.ndarray,
        deltasigma: np.ndarray,
        rp_bins: np.ndarray,
        metadata: Optional[Dict] = None,
        xi_gm_bins: Optional[np.ndarray] = None,
        bin_counts: Optional[np.ndarray] = None,
        bins_comp: Optional[np.ndarray] = None,
        bin_logM_edges: Optional[np.ndarray] = None,
        bin_fI_edges: Optional[np.ndarray] = None,
    ):
        self.positions = np.asarray(positions)
        self.deltasigma = np.asarray(deltasigma)
        self.rp_bins = np.asarray(rp_bins)
        self.rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        self.metadata = metadata or {}

        # Optional TabCorr-style tabulation: mean xi_gm(r) per (logM [, fI]) bin
        self.xi_gm_bins = None if xi_gm_bins is None else np.asarray(xi_gm_bins)
        self.bin_counts = None if bin_counts is None else np.asarray(bin_counts)
        self.bins_comp = None if bins_comp is None else np.asarray(bins_comp)
        self.bin_logM_edges = None if bin_logM_edges is None else np.asarray(bin_logM_edges)
        self.bin_fI_edges = None if bin_fI_edges is None else np.asarray(bin_fI_edges)

        # Validate shapes
        n_halos = len(positions)
        n_rp_bins = len(rp_bins) - 1
        if deltasigma.shape != (n_halos, n_rp_bins):
            raise ValueError(
                f"deltasigma shape {deltasigma.shape} doesn't match "
                f"expected ({n_halos}, {n_rp_bins})"
            )

    @property
    def has_tabulation(self) -> bool:
        return self.xi_gm_bins is not None

    def save(self, output_path: str) -> None:
        """
        Save cache to HDF5 file.

        Parameters
        ----------
        output_path : str
            Path to output HDF5 file
        """
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('positions', data=self.positions, compression='gzip')
            f.create_dataset('deltasigma', data=self.deltasigma, compression='gzip')
            f.create_dataset('rp_bins', data=self.rp_bins)

            if self.has_tabulation:
                tab_grp = f.create_group('tabulation')
                tab_grp.create_dataset('xi_gm_bins', data=self.xi_gm_bins,
                                       compression='gzip')
                tab_grp.create_dataset('bin_counts', data=self.bin_counts)
                tab_grp.create_dataset('bins_comp', data=self.bins_comp)
                tab_grp.create_dataset('bin_logM_edges', data=self.bin_logM_edges)
                if self.bin_fI_edges is not None:
                    tab_grp.create_dataset('bin_fI_edges', data=self.bin_fI_edges)

            if self.metadata:
                meta_grp = f.create_group('metadata')
                for key, value in self.metadata.items():
                    if isinstance(value, (int, float, str)):
                        meta_grp.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        meta_grp.create_dataset(key, data=value)

        print(f"Saved HaloCenterLensingCache to {output_path}")
        print(f"  {len(self.positions)} halos")
        print(f"  {len(self.rp_bins)-1} radial bins")

    @classmethod
    def load(cls, input_path: str) -> 'HaloCenterLensingCache':
        """
        Load cache from HDF5 file.

        Parameters
        ----------
        input_path : str
            Path to input HDF5 file

        Returns
        -------
        HaloCenterLensingCache
            Loaded cache object
        """
        with h5py.File(input_path, 'r') as f:
            positions = f['positions'][:]
            deltasigma = f['deltasigma'][:]
            rp_bins = f['rp_bins'][:]

            tab = {}
            if 'tabulation' in f:
                tab_grp = f['tabulation']
                tab['xi_gm_bins'] = tab_grp['xi_gm_bins'][:]
                tab['bin_counts'] = tab_grp['bin_counts'][:]
                tab['bins_comp'] = tab_grp['bins_comp'][:]
                tab['bin_logM_edges'] = tab_grp['bin_logM_edges'][:]
                if 'bin_fI_edges' in tab_grp:
                    tab['bin_fI_edges'] = tab_grp['bin_fI_edges'][:]

            metadata = {}
            if 'metadata' in f:
                meta_grp = f['metadata']
                for key in meta_grp.attrs:
                    metadata[key] = meta_grp.attrs[key]
                for key in meta_grp.keys():
                    metadata[key] = meta_grp[key][:]

        print(f"Loaded HaloCenterLensingCache from {input_path}")
        print(f"  {len(positions)} halos")
        print(f"  {len(rp_bins)-1} radial bins")
        if tab:
            print(f"  xi_gm tabulation: {tab['xi_gm_bins'].shape[0]} bins")

        return cls(positions, deltasigma, rp_bins, metadata, **tab)

    def __repr__(self) -> str:
        return (
            f"HaloCenterLensingCache("
            f"n_halos={len(self.positions)}, "
            f"n_rp_bins={len(self.rp_bins)-1}, "
            f"rp_range=[{self.rp_bins[0]:.2f}, {self.rp_bins[-1]:.2f}] Mpc/h)"
        )


def satellite_radial_nodes(Rvir, conc, f_exp, tau, lambda_NFW,
                           u_nodes, u_w, x_norm):
    """
    Weighted 3D radial satellite-offset nodes (r_j, w_j).

    Mirrors NFW_jax sampling exactly: NFW component with Rs/lambda_NFW and
    c*lambda_NFW truncated at Rvir (inverse CDF on the same normalized
    radial grid as the sampler); exponential component
    dN/dr ~ exp(-r/(tau*Rs)) truncated at 3*Rvir.

    Rvir in Mpc/h. u_nodes/u_w are Gauss-Legendre nodes/weights on [0, 1];
    x_norm is the normalized radial grid.
    """
    Rs = Rvir / conc
    comps = []
    if f_exp < 1.0:
        rbins = x_norm * Rvir
        x = rbins / (Rs / lambda_NFW)
        cdf = np.log(1 + x) - x / (1 + x)
        cdf = cdf / cdf[-1]
        comps.append((np.interp(u_nodes, cdf, rbins), (1.0 - f_exp) * u_w))
    if f_exp > 0.0:
        u_max = 1.0 - np.exp(-3.0 * Rvir / (tau * Rs))
        comps.append((-tau * Rs * np.log(1.0 - u_nodes * u_max), f_exp * u_w))

    r_all = np.concatenate([c[0] for c in comps])
    w_r = np.concatenate([c[1] for c in comps])
    return r_all, w_r / w_r.sum()


def satellite_offset_nodes(Rvir, conc, f_exp, tau, lambda_NFW,
                           u_nodes, u_w, mu_nodes, mu_w, x_norm):
    """
    Weighted projected (transverse) satellite-offset nodes (rho_j, w_j).

    Projected offset rho = r * sqrt(1 - mu^2) with mu = |cos theta|
    uniform in [0, 1]; radial nodes from satellite_radial_nodes.
    """
    r_all, w_r = satellite_radial_nodes(
        Rvir, conc, f_exp, tau, lambda_NFW, u_nodes, u_w, x_norm)
    sin_th = np.sqrt(1.0 - mu_nodes ** 2)
    rho = (r_all[:, None] * sin_th[None, :]).ravel()
    w_rho = (w_r[:, None] * mu_w[None, :]).ravel()
    return rho, w_rho / w_rho.sum()


def build_tabulation_bins(
    logM: np.ndarray,
    fI: Optional[np.ndarray] = None,
    n_logM_bins: int = 40,
    n_fI_bins: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Assign each halo to a (logM [, fI]) tabulation bin.

    logM bins are equal-width over the catalog range; fI bins (optional,
    for assembly bias) are equal-count quantile bins so every fI bin is
    well populated.

    Returns
    -------
    bin_index : np.ndarray, shape (N_halos,)
        Flat bin index i_logM * n_fI_bins + i_fI (or i_logM if fI is None)
    logM_edges : np.ndarray, shape (n_logM_bins+1,)
    fI_edges : np.ndarray or None, shape (n_fI_bins+1,)
    """
    logM = np.asarray(logM)
    eps = 1e-6
    logM_edges = np.linspace(logM.min() - eps, logM.max() + eps, n_logM_bins + 1)
    i_m = np.clip(np.digitize(logM, logM_edges) - 1, 0, n_logM_bins - 1)

    if fI is None:
        return i_m.astype(np.int64), logM_edges, None

    fI = np.asarray(fI)
    fI_edges = np.quantile(fI, np.linspace(0, 1, n_fI_bins + 1))
    fI_edges[0] -= eps
    fI_edges[-1] += eps
    i_f = np.clip(np.digitize(fI, fI_edges) - 1, 0, n_fI_bins - 1)

    return (i_m * n_fI_bins + i_f).astype(np.int64), logM_edges, fI_edges


def precompute_halo_center_lensing(
    halo_positions: np.ndarray,
    particle_positions: np.ndarray,
    Lbox: float,
    rsd_axis: str,
    RHO_M: float,
    rp_bins: np.ndarray,
    bins_comp: Optional[np.ndarray] = None,
    verbose: bool = True,
    n_workers: int = -1,
    prequery_all: bool = False,
    chi_max: float = 100.0,
    halo_logM: Optional[np.ndarray] = None,
    halo_fI: Optional[np.ndarray] = None,
    n_logM_bins: int = 40,
    n_fI_bins: int = 8,
) -> HaloCenterLensingCache:
    """
    Precompute DeltaSigma profiles at all halo centers using KD-tree + multiprocessing.

    Phase 1: Build KD-tree from particle positions.
    Phase 2: Set shared state for fork-based COW sharing.
    Phase 3: Parallel xi_gm + DeltaSigma via multiprocessing.Pool.

    Parameters
    ----------
    halo_positions : np.ndarray, shape (N_halos, 3)
        Halo center positions [Mpc/h]
    particle_positions : np.ndarray, shape (N_particles, 3)
        Matter tracer positions [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    rsd_axis : str
        Line-of-sight axis ('x', 'y', or 'z')
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    bins_comp : np.ndarray, optional
        Computation bins for xi_gm. Default: geomspace(5e-3, 120, 201)
    verbose : bool, default=True
        Print progress information
    n_workers : int, default=-1
        Number of worker processes (-1 for cpu_count).
    prequery_all : bool, default=True
        If True (Mode 1), query ALL halos at once with workers=-1 (maximally
        efficient KD-tree query), then distribute pre-queried indices to workers
        via fork COW. Uses more RAM (~18 GB for downsampled case).
        If False (Mode 2), each worker queries the shared KD-tree independently
        for its halos. Lower RAM, slightly less efficient KD-tree queries.
    chi_max : float, default=100.0
        Maximum line-of-sight distance for the Sigma integral [Mpc/h]
    halo_logM : np.ndarray, optional, shape (N_halos,)
        Halo log10 masses. If given, the mean xi_gm(r) is additionally
        tabulated in (logM [, fI]) bins and stored in the cache — this is
        the TabCorr-style table used by TabulatedDeltaSigma for the
        satellite term. Costs nothing extra (xi_gm per halo is already
        computed for the central profiles).
    halo_fI : np.ndarray, optional, shape (N_halos,)
        Assembly-bias property (e.g. fs_norm). Adds a second, quantile-based
        binning dimension to the xi_gm tabulation.
    n_logM_bins, n_fI_bins : int
        Tabulation bin counts (fI bins only used when halo_fI is given).

    Returns
    -------
    HaloCenterLensingCache
        Cache containing precomputed DeltaSigma profiles

    Examples
    --------
    >>> cache = precompute_halo_center_lensing(
    ...     halo_positions=halo.positions,
    ...     particle_positions=halo.positions_part,
    ...     Lbox=halo.Lbox,
    ...     rsd_axis=halo.rsd_axis,
    ...     RHO_M=halo.RHO_M,
    ...     rp_bins=np.logspace(-1, np.log10(50), 16)
    ... )
    >>> cache.save('halo_center_lensing.h5')
    """
    if bins_comp is None:
        bins_comp = np.geomspace(5e-3, 120, 201)

    n_halos = len(halo_positions)
    n_rp_bins = len(rp_bins) - 1
    search_radius = bins_comp[-1]
    r_centers = np.sqrt(bins_comp[:-1] * bins_comp[1:])
    t_total = time.time()

    if verbose:
        print(f"Precomputing DeltaSigma at {n_halos} halo centers...")
        print(f"  rp_bins: {n_rp_bins} bins from {rp_bins[0]:.3f} to {rp_bins[-1]:.1f} Mpc/h")
        print(f"  Particles: {len(particle_positions)}")
        print(f"  Mode: {'prequery_all' if prequery_all else 'per-worker query'}")

    # ── Phase 1: KD-tree build ──
    if verbose:
        print(f"  Building KD-tree (search_radius={search_radius:.1f} Mpc/h)...")
    t0 = time.time()
    kdtree = cKDTree(particle_positions, boxsize=Lbox)
    t_tree = time.time() - t0
    if verbose:
        print(f"  Building KD-tree... done ({t_tree:.1f}s)")

    # ── Phase 2: Set shared state ──
    volume_total = Lbox**3
    n_particles_total = len(particle_positions)
    shell_volumes = (4.0/3.0) * np.pi * (bins_comp[1:]**3 - bins_comp[:-1]**3)
    n_mean = n_particles_total / volume_total

    tabulate = halo_logM is not None
    if tabulate:
        bin_index, logM_edges, fI_edges = build_tabulation_bins(
            halo_logM, halo_fI, n_logM_bins, n_fI_bins
        )
        n_bins = n_logM_bins * (n_fI_bins if halo_fI is not None else 1)
        if verbose:
            print(f"  Tabulating xi_gm in {n_bins} bins "
                  f"({n_logM_bins} logM x {n_fI_bins if halo_fI is not None else 1} fI)")

    _shared.update({
        'particle_positions': particle_positions,
        'halo_positions': halo_positions,
        'bins_comp': bins_comp,
        'shell_volumes': shell_volumes,
        'n_mean': n_mean,
        'Lbox': Lbox,
        'r_centers': r_centers,
        'rp_bins': rp_bins,
        'RHO_M': RHO_M,
        'chi_max': chi_max,
        'n_rp_bins': n_rp_bins,
        'search_radius': search_radius,
        'halo_bin_index': bin_index if tabulate else None,
        'n_bins': n_bins if tabulate else None,
    })

    if prequery_all:
        if verbose:
            print(f"  Querying KD-tree for all {n_halos} halos (workers=-1)...")
        t0 = time.time()
        _shared['idx_lists'] = kdtree.query_ball_point(
            halo_positions, r=search_radius, workers=-1, return_sorted=False
        )
        t_query = time.time() - t0
        if verbose:
            print(f"  KD-tree query done ({t_query:.1f}s)")
        worker_fn = _process_batch_prequeried
    else:
        _shared['kdtree'] = kdtree
        worker_fn = _process_batch_with_query

    # ── Phase 3: Parallel computation ──
    if n_workers == -1:
        n_workers = os.cpu_count() or 1

    if verbose:
        print(f"  Computing xi_gm + DeltaSigma ({n_halos} halos, n_workers={n_workers})...")
    t0 = time.time()

    all_indices = np.arange(n_halos)
    batches = [arr.tolist() for arr in np.array_split(all_indices, n_workers)]

    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        results = pool.map(worker_fn, batches)

    if tabulate:
        all_deltasigma = np.vstack([r[0] for r in results])
        xi_sum = np.sum([r[1] for r in results], axis=0)
        xi_count = np.sum([r[2] for r in results], axis=0)
        xi_gm_bins = np.where(xi_count[:, None] > 0,
                              xi_sum / np.maximum(xi_count, 1)[:, None], 0.0)
    else:
        all_deltasigma = np.vstack(results)

    t_phase3 = time.time() - t0
    if verbose:
        print(f"  xi_gm + DeltaSigma complete ({t_phase3:.1f}s)")

    # ── Phase 4: Cleanup + return ──
    _shared.clear()

    t_elapsed = time.time() - t_total
    if verbose:
        print(f"Done! Total: {t_elapsed:.1f}s")
        print(f"  Mean DeltaSigma at rp={np.sqrt(rp_bins[0]*rp_bins[1]):.2f} Mpc/h: "
              f"{np.mean(all_deltasigma[:, 0]):.2e} Msun h/pc^2")

    metadata = {
        'Lbox': Lbox,
        'RHO_M': RHO_M,
        'rsd_axis': rsd_axis,
        'n_particles': len(particle_positions),
        'chi_max': chi_max,
    }

    if tabulate:
        return HaloCenterLensingCache(
            halo_positions, all_deltasigma, rp_bins, metadata,
            xi_gm_bins=xi_gm_bins, bin_counts=xi_count, bins_comp=bins_comp,
            bin_logM_edges=logM_edges, bin_fI_edges=fI_edges,
        )
    return HaloCenterLensingCache(halo_positions, all_deltasigma, rp_bins, metadata)


class OptimizedDeltaSigmaCalculator:
    """
    Calculator that combines precomputed central profiles with runtime satellite computation.

    Uses precomputed halo-center profiles for centrals (instant lookup) and
    computes satellites at runtime via pycorr (single bulk cross-correlation).

    Parameters
    ----------
    cache : HaloCenterLensingCache
        Precomputed halo-center lensing profiles
    particle_positions : np.ndarray, shape (N_particles, 3)
        Matter tracer positions [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    rsd_axis : str
        Line-of-sight axis ('x', 'y', or 'z')
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]

    Examples
    --------
    >>> cache = HaloCenterLensingCache.load('halo_center_lensing.h5')
    >>> calc = OptimizedDeltaSigmaCalculator(
    ...     cache, positions_part, Lbox, 'z', RHO_M
    ... )
    >>> rp, ds = calc.compute_deltasigma(
    ...     cent_halo_indices, sat_positions, satellite_fraction, rp_bins
    ... )
    """

    def __init__(
        self,
        cache: HaloCenterLensingCache,
        particle_positions: np.ndarray,
        Lbox: float,
        rsd_axis: str,
        RHO_M: float
    ):
        self.cache = cache
        self.particle_positions = particle_positions
        self.Lbox = Lbox
        self.rsd_axis = rsd_axis
        self.RHO_M = RHO_M

    def compute_deltasigma(
        self,
        cent_halo_indices: np.ndarray,
        sat_positions: np.ndarray,
        satellite_fraction: float,
        rp_bins: np.ndarray,
        bins_comp: Optional[np.ndarray] = None,
        galaxy_subsample_fraction: float = 1.0,
        galaxy_subsample_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DeltaSigma using optimized approach.

        Parameters
        ----------
        cent_halo_indices : np.ndarray, shape (N_centrals,)
            Indices of halos hosting central galaxies
        sat_positions : np.ndarray, shape (N_satellites, 3)
            Satellite galaxy positions [Mpc/h]
        satellite_fraction : float
            Fraction of galaxies that are satellites
        rp_bins : np.ndarray
            Projected separation bin edges [Mpc/h]
        bins_comp : np.ndarray, optional
            Computation bins for satellite xi_gm
        galaxy_subsample_fraction : float, default=1.0
            Fraction of centrals and satellites to keep (0 < f <= 1).
            Both populations are subsampled at the same rate to preserve
            the satellite fraction. Weights f_cen/f_sat use original counts.
        galaxy_subsample_seed : int, default=42
            Random seed for galaxy subsampling. Centrals use this seed,
            satellites use seed + 1.

        Returns
        -------
        rp_centers : np.ndarray
            Projected bin centers [Mpc/h]
        delta_sigma : np.ndarray
            Surface mass density contrast [Msun h/pc^2]
        """
        if bins_comp is None:
            bins_comp = np.geomspace(5e-3, 120, 201)

        rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        n_rp_bins = len(rp_bins) - 1

        # === CENTRAL CONTRIBUTION (instant lookup) ===
        if len(cent_halo_indices) > 0:
            cent_indices = np.asarray(cent_halo_indices)
            if galaxy_subsample_fraction < 1.0:
                n_cen_keep = int(len(cent_indices) * galaxy_subsample_fraction)
                rng_cen = np.random.RandomState(galaxy_subsample_seed)
                idx_cen = np.sort(rng_cen.choice(len(cent_indices), n_cen_keep, replace=False))
                cent_indices = cent_indices[idx_cen]

            ds_cen_profiles = self.cache.deltasigma[cent_indices]
            ds_cen = np.mean(ds_cen_profiles, axis=0)
        else:
            ds_cen = np.zeros(n_rp_bins)

        # === SATELLITE CONTRIBUTION (runtime pycorr — single bulk call) ===
        if len(sat_positions) > 0:
            if galaxy_subsample_fraction < 1.0:
                n_sat_keep = int(len(sat_positions) * galaxy_subsample_fraction)
                rng_sat = np.random.RandomState(galaxy_subsample_seed + 1)
                idx_sat = np.sort(rng_sat.choice(len(sat_positions), n_sat_keep, replace=False))
                sat_positions = sat_positions[idx_sat]
            rr, xi_gm = compute_corr(
                's', sat_positions, bins_comp,
                catalog2=self.particle_positions,
                boxsize=self.Lbox, los=self.rsd_axis, output='auto'
            )
            ds_calc = DeltaSigmaCalculator(rr, xi_gm, self.RHO_M)
            ds_sat = ds_calc.compute_deltasigma_averaged(rp_bins)
        else:
            ds_sat = np.zeros(n_rp_bins)

        # === COMBINE ===
        f_cen = 1.0 - satellite_fraction
        f_sat = satellite_fraction
        ds_total = f_cen * ds_cen + f_sat * ds_sat

        return rp_centers, ds_total

    def __repr__(self) -> str:
        return (
            f"OptimizedDeltaSigmaCalculator("
            f"cache={self.cache!r}, "
            f"Lbox={self.Lbox})"
        )


class TabulatedDeltaSigma:
    """
    TabCorr-style tabulated DeltaSigma predictor (Zheng & Guo 2016;
    Lange et al. 2019, 2025 arXiv:2512.15962 Sect. 3.2).

    DeltaSigma is linear in the halo occupation, so the Monte-Carlo
    population step can be replaced by occupation-weighted sums over
    precomputed per-halo / per-bin lensing tables:

    - **Centrals** (exact, per halo): weighted average of the per-halo
      cache profiles with weights <N_cen>(logM_h, fI_h). Assembly bias
      is handled natively since weights are evaluated per halo.
    - **Satellites** (per (logM, fI) bin): the tabulated mean xi_gm(r)
      per bin gives Sigma(R) around halo centers; the satellite signal
      is the miscentering convolution of Sigma with the projected
      satellite offset distribution. The radial profile (truncated NFW
      rescaled by lambda_NFW + exponential tail with f_exp, tau,
      truncated at 3 Rvir) is analytic, so arbitrary profile parameters
      are exact — no interpolation over profile parameters is needed
      (unlike TabCorr's spline over eta). The inverse-CDF quadrature
      mirrors NFW_jax sampling exactly, so the prediction is the
      expectation value of the Monte-Carlo pipeline.

    Predictions are noise-free (no realization scatter) and cost ~0.1 s,
    so a sampler can call this directly — no LHS grid or NN emulator.

    Not supported: triaxial satellite profiles, subhalo satellite
    placement (both break the isotropic-offset convolution).

    Parameters
    ----------
    cache : HaloCenterLensingCache
        Cache with xi_gm tabulation (precompute with halo_logM given).
    halo : HaloOccupation
        Configured halo model (set_halo_model already called). Supplies
        per-halo logM, radius, concentration, fI and the Occupation object.

    Examples
    --------
    >>> cache = precompute_halo_center_lensing(..., halo_logM=halo.logM,
    ...                                        halo_fI=halo.fI)
    >>> tab = TabulatedDeltaSigma(cache, halo)
    >>> rp, ds, info = tab.predict(dict_params)
    """

    def __init__(
        self,
        cache: HaloCenterLensingCache,
        halo,
        n_sigma_grid: int = 96,
        n_out_grid: int = 64,
        n_gl_u: int = 32,
        n_gl_mu: int = 16,
        n_gl_phi: int = 12,
    ):
        if not cache.has_tabulation:
            raise ValueError(
                "Cache has no xi_gm tabulation. Regenerate it with "
                "precompute_halo_center_lensing(..., halo_logM=..., halo_fI=...)."
            )
        if not hasattr(halo, 'HOD'):
            raise ValueError("halo has no HOD model — call set_halo_model() first.")
        if len(halo.logM) != len(cache.positions):
            raise ValueError(
                f"Halo catalog ({len(halo.logM)}) and cache "
                f"({len(cache.positions)}) sizes differ — same catalog required."
            )

        self.cache = cache
        self.halo = halo
        self.RHO_M = halo.RHO_M
        chi_max = float(cache.metadata.get('chi_max', 100.0))

        logM = np.asarray(halo.logM)
        logM_edges = cache.bin_logM_edges
        fI_edges = cache.bin_fI_edges
        self.n_m = len(logM_edges) - 1
        self.n_f = 1 if fI_edges is None else len(fI_edges) - 1

        i_m = np.clip(np.digitize(logM, logM_edges) - 1, 0, self.n_m - 1)
        if fI_edges is None:
            self.bin_index = i_m.astype(np.int64)
        else:
            # AB property: whichever of fI/fE the halo model carries — must be
            # the same array the cache tabulation was built with
            ab_prop = halo.fI if getattr(halo, 'fI', None) is not None else halo.fE
            if ab_prop is None:
                raise ValueError("Cache has fI tabulation but halo has no fI/fE.")
            fI = np.asarray(ab_prop)
            i_f = np.clip(np.digitize(fI, fI_edges) - 1, 0, self.n_f - 1)
            self.bin_index = (i_m * self.n_f + i_f).astype(np.int64)

        # Mean Rvir [Mpc/h] and concentration per logM bin (halo.radius is kpc/h)
        counts_m = np.maximum(np.bincount(i_m, minlength=self.n_m), 1)
        self.Rvir_m = np.bincount(
            i_m, weights=np.asarray(halo.radius) / 1e3, minlength=self.n_m) / counts_m
        self.conc_m = np.bincount(
            i_m, weights=np.asarray(halo.concentration), minlength=self.n_m) / counts_m
        self.Rvir_m[self.Rvir_m == 0] = 1e-3   # empty bins (weights will be 0)
        self.conc_m[self.conc_m == 0] = 5.0

        # ── Tabulate Sigma(R) per bin from mean xi_gm (rho_m units) ──
        r_centers = np.sqrt(cache.bins_comp[:-1] * cache.bins_comp[1:])
        xi = cache.xi_gm_bins
        R_max = cache.rp_bins[-1] + 3.5 * self.Rvir_m.max()
        self.R_sigma = np.geomspace(5e-3, min(R_max, chi_max), n_sigma_grid)

        t_chi, w_chi = np.polynomial.legendre.leggauss(200)
        chi = 0.5 * (t_chi + 1) * chi_max
        w_chi = 0.5 * chi_max * w_chi
        rr = np.sqrt(self.R_sigma[:, None] ** 2 + chi[None, :] ** 2)

        n_bins = xi.shape[0]
        self.Sigma_bins = np.empty((n_bins, n_sigma_grid))
        for b in range(n_bins):
            spl = interp1d(r_centers, xi[b], kind='cubic', bounds_error=False,
                           fill_value=(xi[b, 0], 0.0))
            self.Sigma_bins[b] = 2.0 * (spl(rr) @ w_chi)
        self.Sigma_bins = self.Sigma_bins.reshape(self.n_m, self.n_f, n_sigma_grid)

        # Output grid for the satellite Sigma / DeltaSigma pipeline
        self.R_out = np.geomspace(1e-2, cache.rp_bins[-1] * 1.05, n_out_grid)

        # Quadrature nodes (unit intervals; scaled at predict time)
        self._u_nodes, self._u_w = self._gl_unit(n_gl_u)          # CDF u in [0,1]
        self._mu_nodes, self._mu_w = self._gl_unit(n_gl_mu)       # |cos theta| in [0,1]
        t_phi, w_phi = np.polynomial.legendre.leggauss(n_gl_phi)  # phi in [0,pi]
        self._cos_phi = np.cos(0.5 * (t_phi + 1) * np.pi)
        self._phi_w = 0.5 * w_phi                                 # weights of (1/pi) dphi

        # Normalized radial grid mirroring NFW_jax._X_NORM_GRID for inverse CDF
        self._x_norm = np.geomspace(1e-4, 1.0, 1000)

    @staticmethod
    def _gl_unit(n):
        t, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (t + 1), 0.5 * w

    def _offset_nodes(self, m, f_exp, tau, lambda_NFW):
        """Weighted projected-offset nodes (rho_j, w_j) for mass bin m."""
        return satellite_offset_nodes(
            self.Rvir_m[m], self.conc_m[m], f_exp, tau, lambda_NFW,
            self._u_nodes, self._u_w, self._mu_nodes, self._mu_w, self._x_norm,
        )

    def predict(
        self,
        dict_params: Dict,
        rp_bins: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict DeltaSigma for an HOD parameter set (no population step).

        Parameters
        ----------
        dict_params : dict
            HOD parameters (same keys as populate_haloes), optionally
            including profile parameters f_exp, tau, lambda_NFW and
            assembly-bias parameters.
        rp_bins : np.ndarray, optional
            Projected bin edges. Defaults to (and must match) cache.rp_bins,
            on which the per-halo central profiles were bin-averaged.

        Returns
        -------
        rp_centers : np.ndarray
        delta_sigma : np.ndarray [Msun h/pc^2]
        info : dict with ngal, fsat, ds_cen, ds_sat
        """
        if rp_bins is None:
            rp_bins = self.cache.rp_bins
        elif not np.allclose(rp_bins, self.cache.rp_bins):
            raise ValueError("rp_bins must match cache.rp_bins (central profiles "
                             "are pre-averaged on that binning).")
        rp_centers = self.cache.rp_centers

        f_exp = float(dict_params.get('f_exp', 0.0))
        tau = float(dict_params.get('tau', 6.0))
        lambda_NFW = float(dict_params.get('lambda_NFW', 1.0))

        probC, probS = self.halo.HOD.compute_HOD_occupation(
            np.asarray(self.halo.logM), dict_params
        )
        # Bernoulli sampling in populate_centrals clips probC at 1 implicitly
        probC = np.minimum(np.asarray(probC, dtype=np.float64), 1.0)
        probS = np.asarray(probS, dtype=np.float64)

        sum_C, sum_S = probC.sum(), probS.sum()
        ngal = (sum_C + sum_S) / self.halo.Lbox ** 3
        fsat = sum_S / (sum_C + sum_S)

        # ── Centrals: exact per-halo weighted average ──
        ds_cen = (probC @ self.cache.deltasigma) / sum_C

        # ── Satellites: per-bin Sigma + analytic offset convolution ──
        w = np.bincount(self.bin_index, weights=probS,
                        minlength=self.n_m * self.n_f).reshape(self.n_m, self.n_f)
        Sigma_M = np.einsum('mf,mfr->mr', w, self.Sigma_bins)  # unnormalized
        w_M = w.sum(axis=1)

        Sigma_sat = np.zeros_like(self.R_out)
        for m in np.nonzero(w_M > 0)[0]:
            rho, w_rho = self._offset_nodes(m, f_exp, tau, lambda_NFW)
            # dist(R_out, rho, phi): azimuthal + offset average of Sigma_M[m]
            d2 = (self.R_out[:, None, None] ** 2 + rho[None, :, None] ** 2
                  + 2.0 * self.R_out[:, None, None] * rho[None, :, None]
                  * self._cos_phi[None, None, :])
            dist = np.sqrt(np.maximum(d2, 0.0))
            Sig = np.interp(dist, self.R_sigma, Sigma_M[m],
                            left=Sigma_M[m, 0], right=0.0)
            Sigma_sat += (Sig @ self._phi_w) @ w_rho
        Sigma_sat /= w_M.sum()

        # DeltaSigma from Sigma (same quadrature as DeltaSigmaCalculator)
        spl_S = interp1d(self.R_out, Sigma_sat, kind='cubic', bounds_error=False,
                         fill_value=(Sigma_sat[0], 0.0))
        Sigma_mean = 2.0 * gauss_legendre_integration(
            lambda r: spl_S(r) * r, 0, self.R_out) / self.R_out ** 2
        ds_sat_grid = (Sigma_mean - Sigma_sat) * self.RHO_M / 1e12
        spl_ds = interp1d(self.R_out, ds_sat_grid, kind='cubic', bounds_error=False,
                          fill_value=(ds_sat_grid[0], ds_sat_grid[-1]))
        ds_sat = binavg_2D(spl_ds, rp_bins)

        ds_total = (1.0 - fsat) * ds_cen + fsat * ds_sat
        info = {'ngal': ngal, 'fsat': fsat, 'ds_cen': ds_cen, 'ds_sat': ds_sat}
        return rp_centers, ds_total, info

    def make_predict_jax(self, n_sub_logM: int = 16, n_sub_fI: int = 4):
        """Build a pure-JAX twin of :meth:`predict` for jit/vmap sampling.

        Returns ``predict_fn(params) -> (ds_total, ngal, fsat)`` where
        ``params`` is a dict of (traced) scalars. The satellite pipeline is
        the exact jnp translation of :meth:`predict` (the scipy cubic-spline
        stages are linear in their inputs and are precomputed as matrices by
        probing the NumPy code with unit vectors). The centrals — exact
        per-halo sums in :meth:`predict` — are tabulated on a fine
        (n_m*n_sub_logM, n_f*n_sub_fI) occupation grid nested inside the
        cache bins: profile sums are exact, only the occupation weight is
        evaluated at the per-cell mean (logM, fI) instead of per halo.

        The ngal/fsat catalog sums use the same fine grid. AB is supported
        for ``ab_method`` 'mass' or 'direct' with a single AB property
        (fI or fE); 'direct' uses exact per-cell means of sign(prop).
        """
        import jax
        import jax.numpy as jnp
        from ..HOD_models import build_occupation_fn_jax

        occ = self.halo.HOD
        if occ.assembly_bias and occ.ab_method not in ("mass", "direct"):
            raise NotImplementedError(
                "make_predict_jax supports ab_method 'mass' or 'direct'.")

        # ── AB property and the coefficient names that multiply it ──
        prop = None
        cen_coef = sat_coef = None
        if occ.assembly_bias:
            if occ.fI is not None and occ.fE is not None:
                raise NotImplementedError(
                    "make_predict_jax supports a single AB property (fI or fE).")
            if occ.fI is not None:
                prop, cen_coef, sat_coef = np.asarray(occ.fI), "A_cent", "A_sat"
            else:
                prop, cen_coef, sat_coef = np.asarray(occ.fE), "B_cent", "B_sat"

        # ── Fine occupation grid nested in the cache tabulation bins ──
        logM = np.asarray(self.halo.logM)
        edges_m = np.asarray(self.cache.bin_logM_edges)
        n_mc = self.n_m * n_sub_logM
        fine_m_edges = np.concatenate(
            [np.linspace(edges_m[i], edges_m[i + 1], n_sub_logM + 1)[:-1]
             for i in range(self.n_m)] + [edges_m[-1:]])
        i_mc = np.clip(np.digitize(logM, fine_m_edges) - 1, 0, n_mc - 1)

        if self.n_f > 1:
            fI_edges = np.asarray(self.cache.bin_fI_edges)
            i_f = np.clip(np.digitize(prop, fI_edges) - 1, 0, self.n_f - 1)
            n_fc = self.n_f * n_sub_fI
            i_fc = np.empty(len(prop), dtype=np.int64)
            eps = 1e-9
            for f in range(self.n_f):
                sel = i_f == f
                q = np.quantile(prop[sel], np.linspace(0, 1, n_sub_fI + 1))
                q[0] -= eps
                q[-1] += eps
                i_fc[sel] = f * n_sub_fI + np.clip(
                    np.digitize(prop[sel], q) - 1, 0, n_sub_fI - 1)
        else:
            n_fc, n_sub_fI = 1, 1
            i_fc = np.zeros(len(logM), dtype=np.int64)

        cell = i_mc * n_fc + i_fc
        n_cells = n_mc * n_fc
        N_cell = np.bincount(cell, minlength=n_cells).astype(np.float64)
        safe_N = np.maximum(N_cell, 1.0)
        logM_cell = np.bincount(cell, weights=logM, minlength=n_cells) / safe_N
        # empty cells: use the fine-bin midpoint (weight is 0 anyway)
        mid_m = 0.5 * (fine_m_edges[:-1] + fine_m_edges[1:])
        empty = N_cell == 0
        logM_cell[empty] = np.repeat(mid_m, n_fc)[empty]
        if prop is not None:
            prop_cell = np.bincount(cell, weights=prop, minlength=n_cells) / safe_N
            # 'direct' AB uses sign(prop): exact per-cell mean of the signs
            sign_cell = np.bincount(
                cell, weights=np.sign(prop), minlength=n_cells) / safe_N
        else:
            prop_cell = np.zeros(n_cells)
            sign_cell = np.zeros(n_cells)

        n_rp = len(self.cache.rp_centers)
        D_cen = np.empty((n_cells, n_rp))
        dsig = np.asarray(self.cache.deltasigma)
        for j in range(n_rp):
            D_cen[:, j] = np.bincount(cell, weights=dsig[:, j], minlength=n_cells)

        # ── Linear-map matrices probing the exact NumPy spline stages ──
        n_out = len(self.R_out)
        M_sig = np.empty((n_out, n_out))
        M_avg = np.empty((n_rp, n_out))
        rp_bins = self.cache.rp_bins
        for j in range(n_out):
            e = np.zeros(n_out)
            e[j] = 1.0
            spl_S = interp1d(self.R_out, e, kind='cubic', bounds_error=False,
                             fill_value=(e[0], 0.0))
            M_sig[:, j] = 2.0 * gauss_legendre_integration(
                lambda r: spl_S(r) * r, 0, self.R_out) / self.R_out ** 2
            spl_d = interp1d(self.R_out, e, kind='cubic', bounds_error=False,
                             fill_value=(e[0], e[-1]))
            M_avg[:, j] = binavg_2D(spl_d, rp_bins)

        # ── Constants as jnp arrays ──
        j_logM_cell = jnp.asarray(logM_cell.reshape(n_mc, n_fc))
        j_prop_cell = jnp.asarray(prop_cell.reshape(n_mc, n_fc))
        j_sign_cell = jnp.asarray(sign_cell.reshape(n_mc, n_fc))
        j_N_cell = jnp.asarray(N_cell.reshape(n_mc, n_fc))
        j_D_cen = jnp.asarray(D_cen.reshape(n_mc, n_fc, n_rp))
        j_Sigma_bins = jnp.asarray(self.Sigma_bins)
        j_Rvir_m = jnp.asarray(self.Rvir_m)
        j_conc_m = jnp.asarray(self.conc_m)
        j_R_sigma = jnp.asarray(self.R_sigma)
        j_R_out = jnp.asarray(self.R_out)
        j_x_norm = jnp.asarray(self._x_norm)
        j_u_nodes, j_u_w = jnp.asarray(self._u_nodes), jnp.asarray(self._u_w)
        j_mu_w = jnp.asarray(self._mu_w)
        j_sin_th = jnp.asarray(np.sqrt(1.0 - self._mu_nodes ** 2))
        j_cos_phi, j_phi_w = jnp.asarray(self._cos_phi), jnp.asarray(self._phi_w)
        j_M_sig, j_M_avg = jnp.asarray(M_sig), jnp.asarray(M_avg)
        occ_fn = build_occupation_fn_jax(occ)
        n_m, n_f = self.n_m, self.n_f
        Lbox3 = self.halo.Lbox ** 3
        rho_fac = self.RHO_M / 1e12
        has_ab = occ.assembly_bias
        ab_method = occ.ab_method

        def predict_fn(params):
            cshift = sshift = 0.0
            if has_ab and ab_method == "mass":
                cshift = params.get(cen_coef, 0.0) * j_prop_cell
                sshift = params.get(sat_coef, 0.0) * j_prop_cell
            probC, probS = occ_fn(j_logM_cell, params, cshift, sshift)
            if has_ab and ab_method == "direct":
                ab_c = params.get(cen_coef, 0.0) * j_sign_cell
                ab_s = params.get(sat_coef, 0.0) * j_sign_cell
                probC = probC + ab_c * jnp.minimum(probC, 1.0 - probC)
                probS = probS * (1.0 + ab_s)
            probC = jnp.minimum(probC, 1.0)

            sum_C = jnp.sum(probC * j_N_cell)
            sum_S = jnp.sum(probS * j_N_cell)
            ngal = (sum_C + sum_S) / Lbox3
            fsat = sum_S / (sum_C + sum_S)

            # Centrals: binned twin of probC @ cache.deltasigma / sum_C
            ds_cen = jnp.einsum('mf,mfr->r', probC, j_D_cen) / sum_C

            # Satellites: per-(logM, fI)-bin Sigma + offset convolution
            w = (probS * j_N_cell).reshape(
                n_m, n_sub_logM, n_f, n_sub_fI).sum(axis=(1, 3))
            Sigma_M = jnp.einsum('mf,mfr->mr', w, j_Sigma_bins)

            f_exp = params.get('f_exp', 0.0)
            tau = params.get('tau', 6.0)
            lam = params.get('lambda_NFW', 1.0)

            def per_m(Rvir, conc, SigM_row):
                Rs = Rvir / conc
                # radial nodes: NFW comp (weight 1-f_exp) + exp comp (weight f_exp)
                x = j_x_norm * conc * lam
                cdf = jnp.log1p(x) - x / (1.0 + x)
                cdf = cdf / cdf[-1]
                r_nfw = jnp.interp(j_u_nodes, cdf, j_x_norm * Rvir)
                u_max = 1.0 - jnp.exp(-3.0 * Rvir / (tau * Rs))
                r_exp = -tau * Rs * jnp.log1p(-j_u_nodes * u_max)
                r_all = jnp.concatenate([r_nfw, r_exp])
                w_r = jnp.concatenate([(1.0 - f_exp) * j_u_w, f_exp * j_u_w])
                w_r = w_r / jnp.sum(w_r)
                rho = (r_all[:, None] * j_sin_th[None, :]).ravel()
                w_rho = (w_r[:, None] * j_mu_w[None, :]).ravel()
                w_rho = w_rho / jnp.sum(w_rho)
                d2 = (j_R_out[:, None, None] ** 2 + rho[None, :, None] ** 2
                      + 2.0 * j_R_out[:, None, None] * rho[None, :, None]
                      * j_cos_phi[None, None, :])
                dist = jnp.sqrt(jnp.maximum(d2, 0.0))
                Sig = jnp.interp(dist, j_R_sigma, SigM_row,
                                 left=SigM_row[0], right=0.0)
                return (Sig @ j_phi_w) @ w_rho

            Sigma_sat = jax.vmap(per_m)(j_Rvir_m, j_conc_m, Sigma_M).sum(axis=0)
            Sigma_sat = Sigma_sat / jnp.sum(w)

            Sigma_mean = j_M_sig @ Sigma_sat
            ds_sat = j_M_avg @ ((Sigma_mean - Sigma_sat) * rho_fac)

            ds_total = (1.0 - fsat) * ds_cen + fsat * ds_sat
            return ds_total, ngal, fsat

        return predict_fn

    def __repr__(self) -> str:
        return (f"TabulatedDeltaSigma(n_logM_bins={self.n_m}, "
                f"n_fI_bins={self.n_f}, n_halos={len(self.bin_index)})")
