"""
Parallel pre-computation for galaxy-galaxy lensing (NumPy-based).

This module provides a parallelized version of the ΔΣ computation functions
using joblib for multi-core CPU parallelization. Pure NumPy implementation
avoids JAX fork() conflicts.

Key features:
- Multi-core CPU parallelization using joblib
- Reuses proven functions from precompute_deltasigma.py
- No JAX dependencies - works on any system
- ~10-50× speedup on multi-core systems

References
----------
.. [1] Yuan et al. (2021), MNRAS, arXiv:2110.11412 (AbacusHOD paper)
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Optional
from joblib import Parallel, delayed
import warnings
import os
import tempfile

# Import core functions from serial version
from HOD_NRV.twopoint_calculator.precompute_deltasigma import (
    compute_deltasigma_spherical,
    compute_deltasigma_at_position,
    save_precomputed_lensing,
    load_precomputed_lensing
)


def _create_memmap_array(array: np.ndarray, name: str = "array") -> Tuple[np.ndarray, str]:
    """
    Create a memory-mapped version of an array in a temporary file.

    This allows joblib workers to share read-only data without copying,
    drastically reducing memory usage for large arrays.

    Parameters
    ----------
    array : np.ndarray
        Array to convert to memmap
    name : str
        Name for temporary file (for identification)

    Returns
    -------
    memmap_array : np.ndarray
        Memory-mapped view of the array
    temp_path : str
        Path to temporary file (for cleanup)
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"joblib_memmap_{name}_{os.getpid()}.mmap")

    # Create memmap file
    memmap_array = np.memmap(
        temp_path,
        dtype=array.dtype,
        mode='w+',
        shape=array.shape
    )

    # Copy data
    memmap_array[:] = array[:]

    # Flush to disk
    memmap_array.flush()

    # Reopen in read-only mode
    memmap_array = np.memmap(
        temp_path,
        dtype=array.dtype,
        mode='r',
        shape=array.shape
    )

    return memmap_array, temp_path


def process_particle_batch(
    particle_batch: np.ndarray,
    kdtree: cKDTree,
    particle_positions_full: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    particle_mass: float,
    Lbox: float,
    search_radius_computation: float,
    n_particles_total: int,
    method: str = 'spherical',
    los_axis: str = 'z'
) -> Tuple[list, list]:
    """
    Process a batch of particles: compute ΔΣ at each particle position.

    This function is designed to be called in parallel for different particle batches.

    Parameters
    ----------
    particle_batch : np.ndarray, shape (N_batch, 3)
        Batch of particle positions to process [Mpc/h]
    kdtree : cKDTree
        KD-tree for full particle catalog (for neighbor queries)
    particle_positions_full : np.ndarray
        Full particle position array [Mpc/h]
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    particle_mass : float
        Particle mass [Msun/h]
    Lbox : float
        Box size [Mpc/h]
    search_radius_computation : float
        Search radius [Mpc/h] for querying particles when computing ΔΣ
    n_particles_total : int
        Total number of particles in full catalog
    method : str, default='spherical'
        Method to compute ΔΣ: 'spherical' or 'cylindrical'
    los_axis : str, default='z'
        Line-of-sight axis (only for cylindrical method)

    Returns
    -------
    positions_list : list
        List of positions where ΔΣ was computed
    deltasigma_list : list
        List of ΔΣ arrays
    """
    positions_list = []
    deltasigma_list = []

    for particle_pos in particle_batch:
        # Find particles near this particle position
        local_indices = kdtree.query_ball_point(particle_pos, r=search_radius_computation)

        if len(local_indices) < 50:  # Need more particles for spherical method
            continue

        local_particles = particle_positions_full[local_indices]

        try:
            # Compute ΔΣ using selected method
            if method == 'spherical':
                delta_sigma = compute_deltasigma_spherical(
                    particle_pos,
                    local_particles,
                    RHO_M,
                    rp_bins,
                    particle_mass=particle_mass,
                    n_particles_total=n_particles_total,
                    Lbox=Lbox
                )
            elif method == 'cylindrical':
                delta_sigma = compute_deltasigma_at_position(
                    particle_pos,
                    local_particles,
                    RHO_M,
                    rp_bins,
                    particle_mass=particle_mass,
                    n_particles_total=n_particles_total,
                    Lbox=Lbox,
                    los_axis=los_axis
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Check for NaN/Inf
            if np.all(np.isfinite(delta_sigma)):
                positions_list.append(particle_pos.copy())
                deltasigma_list.append(delta_sigma)

        except Exception as e:
            # Silently skip failures (can log if verbose)
            continue

    return positions_list, deltasigma_list


def precompute_lensing_grid_parallel(
    selected_particle_positions: np.ndarray,
    particle_positions_full: np.ndarray,
    RHO_M: float,
    rp_bins: np.ndarray,
    Lbox: float,
    particle_mass: Optional[float] = None,
    method: str = 'spherical',
    los_axis: str = 'z',
    search_radius_computation: Optional[float] = None,
    n_jobs: int = -1,
    batch_size: int = 1000,
    backend: str = 'loky',
    use_memmap: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute ΔΣ grid using parallel processing (pure NumPy).

    This function processes pre-selected particles and computes ΔΣ at each
    position using parallel batch processing. Matches the workflow of the
    serial version (test_fast_dsigma.py).

    Parameters
    ----------
    selected_particle_positions : np.ndarray, shape (N_selected, 3)
        Pre-selected particle positions where ΔΣ should be computed [Mpc/h]
        (typically particles within r_factor × R_vir of halos)
    particle_positions_full : np.ndarray, shape (N_total, 3)
        Full particle catalog for neighbor queries [Mpc/h]
    RHO_M : float
        Mean matter density [Msun/h / (Mpc/h)^3]
    rp_bins : np.ndarray
        Projected separation bin edges [Mpc/h]
    Lbox : float
        Simulation box size [Mpc/h]
    particle_mass : float, optional
        Mass of individual particles [Msun/h]. If not provided, computes
        from RHO_M × V_box / N_particles_full (automatic upweighting).
    method : str, default='spherical'
        Method to compute ΔΣ: 'spherical' (Abel transform, recommended) or
        'cylindrical' (direct projection, legacy)
    los_axis : str, default='z'
        Line-of-sight axis for projection ('x', 'y', or 'z')
        Only used if method='cylindrical'
    search_radius_computation : float, optional
        Search radius [Mpc/h] for querying particles when computing ΔΣ.
        If not provided, defaults to rp_bins[-1] * 1.5 (recommended).
        Should be large enough to capture sufficient neighbors (~50+ particles).
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all CPUs)
    batch_size : int, default=1000
        Number of particles to process per batch (tune for memory)
    backend : str, default='loky'
        Joblib backend: 'loky' (process-based, recommended),
        'multiprocessing' (fork-based, avoid on macOS), or 'threading'
    use_memmap : bool, default=True
        Use memory-mapped arrays to avoid data copying across workers.
        Recommended for large particle catalogs (>1M particles) with loky backend.
        Reduces memory usage from N_jobs × array_size to ~1× array_size.
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    positions : np.ndarray, shape (N_success, 3)
        Positions where ΔΣ was successfully computed [Mpc/h]
    deltasigma_values : np.ndarray, shape (N_success, len(rp_bins)-1)
        Pre-computed ΔΣ(rp) at each position [Msun h/pc²]

    Examples
    --------
    >>> # Pre-select particles near halos (3×Rvir)
    >>> selected_indices = []
    >>> for halo_pos, rvir in zip(halo_positions, halo_rvir):
    ...     nearby = kdtree.query_ball_point(halo_pos, r=3*rvir)
    ...     selected_indices.extend(nearby)
    >>> selected_positions = particle_positions[selected_indices]
    >>>
    >>> # Parallel pre-computation
    >>> positions, deltasigma = precompute_lensing_grid_parallel(
    ...     selected_positions, particle_positions,
    ...     RHO_M=8.6e10, rp_bins=np.logspace(-1, 1.5, 15),
    ...     Lbox=1000.0, n_jobs=-1
    ... )

    Notes
    -----
    Expected performance:
    - Single-threaded: ~1-10 particles/second
    - 16-core CPU: ~10-50× speedup vs single-threaded

    Memory usage (without memmap): ~1-3 GB per worker × N_jobs
    Memory usage (with memmap): ~1-3 GB total (shared read-only)

    Backend selection:
    - 'loky': Best for most cases, especially macOS (no fork issues)
    - 'multiprocessing': Faster but can have fork issues on macOS
    - 'threading': Shared memory but limited by Python GIL

    Memory-mapping (use_memmap=True):
    - Creates temporary memory-mapped files for large arrays
    - Workers share read-only access (no data copying!)
    - Automatically cleaned up after computation
    - Essential for large particle catalogs with zero downsampling
    """
    # Total number of particles in full catalog
    n_particles_total = len(particle_positions_full)

    # Track memmap files for cleanup
    memmap_files = []

    # Compute particle mass if not provided
    if particle_mass is None:
        V_box = Lbox**3
        particle_mass = RHO_M * V_box / n_particles_total
        if verbose:
            print(f"Auto-computed particle mass: {particle_mass:.6e} Msun/h")

    # Set search radius for ΔΣ computation if not provided
    if search_radius_computation is None:
        search_radius_computation = rp_bins[-1] * 1.5  # Default: extend beyond max rp
        if verbose:
            print(f"Auto-set search_radius_computation: {search_radius_computation:.2f} Mpc/h")

    if verbose:
        print(f"Building KD-tree for {n_particles_total:,} particles...")

    kdtree = cKDTree(particle_positions_full, boxsize=Lbox)
    n_selected = len(selected_particle_positions)

    # ========================================================================
    # Memory-mapping for large arrays (reduces memory usage)
    # ========================================================================
    if use_memmap and backend == 'loky':
        if verbose:
            print(f"\nCreating memory-mapped arrays (backend={backend})...")
            print(f"  This enables zero-copy shared memory across workers")
            print(f"  Particle array size: {particle_positions_full.nbytes / 1e6:.1f} MB")

        try:
            # Create memmap for full particle positions
            particle_positions_mmap, particle_mmap_file = _create_memmap_array(
                particle_positions_full, name="particles"
            )
            memmap_files.append(particle_mmap_file)

            # Create memmap for selected positions
            selected_positions_mmap, selected_mmap_file = _create_memmap_array(
                selected_particle_positions, name="selected"
            )
            memmap_files.append(selected_mmap_file)

            # Use memmapped arrays for computation
            particle_positions_to_use = particle_positions_mmap
            selected_positions_to_use = selected_positions_mmap

            if verbose:
                print(f"  ✓ Memory-mapped arrays created successfully")
                print(f"  Expected memory saving: {(n_jobs - 1) * particle_positions_full.nbytes / 1e9:.2f} GB")

        except Exception as e:
            if verbose:
                print(f"  ⚠ Memory-mapping failed: {e}")
                print(f"  Falling back to regular arrays (higher memory usage)")
            particle_positions_to_use = particle_positions_full
            selected_positions_to_use = selected_particle_positions
            memmap_files = []  # Clear any partial memmaps

    else:
        # No memmap: use original arrays directly
        particle_positions_to_use = particle_positions_full
        selected_positions_to_use = selected_particle_positions

        if verbose and not use_memmap:
            print(f"\nMemory-mapping disabled (use_memmap=False)")
            print(f"  Warning: Each worker will copy {particle_positions_full.nbytes / 1e6:.1f} MB")
            print(f"  Total memory usage: ~{n_jobs * particle_positions_full.nbytes / 1e9:.2f} GB")

    if verbose:
        print(f"\nProcessing {n_selected:,} selected particles with {n_jobs} parallel jobs...")
        print(f"Backend: {backend}")
        print(f"Method: {method}")
        print(f"Batch size: {batch_size} particles/batch")
        print(f"Memory-mapping: {'enabled' if (use_memmap and backend == 'loky' and memmap_files) else 'disabled'}")

    # Split selected particles into batches
    n_batches = (n_selected + batch_size - 1) // batch_size
    particle_batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_selected)
        particle_batches.append(selected_positions_to_use[start_idx:end_idx])

    # Parallel processing across particle batches
    try:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10 if verbose else 0)(
            delayed(process_particle_batch)(
                batch, kdtree, particle_positions_to_use,
                RHO_M, rp_bins, particle_mass, Lbox,
                search_radius_computation, n_particles_total,
                method, los_axis
            )
            for batch in particle_batches
        )

        # Collect results from all workers
        all_positions = []
        all_deltasigma = []

        for pos_list, ds_list in results:
            all_positions.extend(pos_list)
            all_deltasigma.extend(ds_list)

        if len(all_positions) == 0:
            warnings.warn("No valid ΔΣ values computed! Check input parameters.")
            return np.array([]), np.array([])

        positions = np.array(all_positions)
        deltasigma_values = np.array(all_deltasigma)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Computation complete!")
            print(f"  Selected particles: {n_selected:,}")
            print(f"  Successful computations: {len(all_positions):,}")
            print(f"  Success rate: {len(all_positions)/n_selected*100:.1f}%")
            print(f"  ΔΣ shape: {deltasigma_values.shape}")
            print(f"  Memory size: {deltasigma_values.nbytes / 1e6:.1f} MB")
            print(f"{'='*60}")

        return positions, deltasigma_values

    finally:
        # ====================================================================
        # Cleanup: Remove memory-mapped files
        # ====================================================================
        if memmap_files:
            if verbose:
                print(f"\nCleaning up {len(memmap_files)} memory-mapped files...")

            for mmap_file in memmap_files:
                try:
                    if os.path.exists(mmap_file):
                        os.remove(mmap_file)
                        if verbose:
                            print(f"  ✓ Removed {os.path.basename(mmap_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Failed to remove {os.path.basename(mmap_file)}: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_memory_usage(
    n_halos: int,
    n_particles: int,
    n_rp_bins: int,
    n_jobs: int,
    avg_particles_per_halo: int = 1000
) -> Dict[str, float]:
    """
    Estimate memory usage for parallel pre-computation.

    Parameters
    ----------
    n_halos : int
        Number of halos to process
    n_particles : int
        Total number of particles
    n_rp_bins : int
        Number of projected radial bins
    n_jobs : int
        Number of parallel jobs
    avg_particles_per_halo : int, default=1000
        Average number of particles processed per halo

    Returns
    -------
    memory_dict : dict
        Dictionary with memory estimates in GB
    """
    # Particle positions array: shared, read-only
    particle_array_gb = (n_particles * 3 * 8) / 1e9  # float64

    # Per-worker memory (KDTree copy + working arrays)
    kdtree_gb = particle_array_gb * 0.2  # KDTree is ~20% of array size
    working_gb = (avg_particles_per_halo * 3 * 8) / 1e9
    per_worker_gb = kdtree_gb + working_gb

    # Output arrays (positions + deltasigma)
    expected_output_positions = n_halos * avg_particles_per_halo
    output_gb = (expected_output_positions * (3 + n_rp_bins) * 8) / 1e9

    total_gb = particle_array_gb + (per_worker_gb * n_jobs) + output_gb

    return {
        'particle_array_gb': particle_array_gb,
        'per_worker_gb': per_worker_gb,
        'output_gb': output_gb,
        'total_gb': total_gb,
        'n_jobs': n_jobs
    }


if __name__ == "__main__":
    print("NumPy-based Parallel ΔΣ Pre-computation Module")
    print("Pure NumPy implementation - no JAX dependencies")
    print("Uses joblib for multi-core parallelization")
