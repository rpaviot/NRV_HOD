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

from .standard_two_point_calculator import (
    compute_corr, DeltaSigmaCalculator
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

        calc = DeltaSigmaCalculator(r_centers, xi_gm, RHO_M, chi_max=chi_max)
        results[k] = calc.compute_deltasigma_averaged(rp_bins)

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

            calc = DeltaSigmaCalculator(r_centers, xi_gm, RHO_M, chi_max=chi_max)
            results[ptr] = calc.compute_deltasigma_averaged(rp_bins)
            ptr += 1

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
        metadata: Optional[Dict] = None
    ):
        self.positions = np.asarray(positions)
        self.deltasigma = np.asarray(deltasigma)
        self.rp_bins = np.asarray(rp_bins)
        self.rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
        self.metadata = metadata or {}

        # Validate shapes
        n_halos = len(positions)
        n_rp_bins = len(rp_bins) - 1
        if deltasigma.shape != (n_halos, n_rp_bins):
            raise ValueError(
                f"deltasigma shape {deltasigma.shape} doesn't match "
                f"expected ({n_halos}, {n_rp_bins})"
            )

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

        return cls(positions, deltasigma, rp_bins, metadata)

    def __repr__(self) -> str:
        return (
            f"HaloCenterLensingCache("
            f"n_halos={len(self.positions)}, "
            f"n_rp_bins={len(self.rp_bins)-1}, "
            f"rp_range=[{self.rp_bins[0]:.2f}, {self.rp_bins[-1]:.2f}] Mpc/h)"
        )


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
    prequery_all: bool = False
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
    }

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
