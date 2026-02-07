"""
Two-Point Calculator Module
===========================

Two-point statistics and galaxy-galaxy lensing calculations.

Standard Calculators
--------------------
compute_corr : Compute two-point correlation functions
DeltaSigmaCalculator : Standard galaxy-galaxy lensing calculator
compute_galaxy_clustering : Convenience wrapper for clustering
compute_galaxy_lensing : Convenience wrapper for lensing

Optimized Lensing (Halo-Center Precomputation)
----------------------------------------------
HaloCenterLensingCache : Cache for precomputed halo-center DeltaSigma profiles
precompute_halo_center_lensing : One-time precomputation function
OptimizedDeltaSigmaCalculator : Runtime calculator using precomputed profiles

The optimized approach provides ~4-10x speedup by:
- Precomputing DeltaSigma at each halo center (one-time cost)
- Looking up central contributions instantly by halo index
- Computing satellite contributions at runtime (smaller population)
"""

from .standard_two_point_calculator import (
    compute_corr,
    DeltaSigmaCalculator,
    compute_galaxy_clustering,
    compute_galaxy_lensing
)

from .halo_center_lensing import (
    HaloCenterLensingCache,
    precompute_halo_center_lensing,
    OptimizedDeltaSigmaCalculator
)

__all__ = [
    # Standard calculators
    'compute_corr',
    'DeltaSigmaCalculator',
    'compute_galaxy_clustering',
    'compute_galaxy_lensing',
    # Optimized halo-center approach
    'HaloCenterLensingCache',
    'precompute_halo_center_lensing',
    'OptimizedDeltaSigmaCalculator',
]
