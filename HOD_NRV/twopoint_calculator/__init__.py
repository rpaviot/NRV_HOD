"""
Fast Two-Point Statistics Calculator for NRVpy HOD Code

This module implements an optimized galaxy-galaxy lensing calculator based on
the method described in Yuan et al. (2021) - AbacusHOD paper (arXiv:2110.11412).

The key optimization is pre-computing ΔΣ at particle positions near halos, then
using spatial interpolation for galaxy realizations. This provides ~100-1000x
speedup compared to computing correlations for each HOD realization.

Workflow:
---------
1. Pre-computation (done once per simulation snapshot):
   - For each halo, find particles within 3×R_vir
   - Compute ΔΣ(rp) at each particle position
   - Save to disk

2. Fast evaluation (per HOD realization):
   - For each galaxy position, interpolate ΔΣ from nearby pre-computed values
   - Sum contributions from all galaxies

Modules:
--------
- precompute_deltasigma: Pre-computation utilities with KD-tree
- fast_two_point: Fast lensing calculator with interpolation
- two_point_legacy: Legacy implementation (for reference/null testing)

Author: NRVpy Development Team
"""

# Fast calculator (recommended for HOD parameter sampling)
from .precompute_deltasigma import (
    build_particle_kdtree,
    compute_deltasigma_at_position,
    precompute_lensing_grid,
    save_precomputed_lensing,
    load_precomputed_lensing
)

from .fast_two_point import FastDeltaSigmaCalculator

# Legacy functions (for backward compatibility and null testing)
from .standard_two_point_calculator import (
    compute_corr,
    DeltaSigmaCalculator,
    compute_galaxy_clustering,
    compute_galaxy_lensing
)

__all__ = [
    # Fast calculator functions
    'build_particle_kdtree',
    'compute_deltasigma_at_position',
    'precompute_lensing_grid',
    'save_precomputed_lensing',
    'load_precomputed_lensing',
    'FastDeltaSigmaCalculator',
    # Legacy functions (for backward compatibility)
    'compute_corr',
    'DeltaSigmaCalculator',
    'compute_galaxy_clustering',
    'compute_galaxy_lensing',
]

__version__ = '1.0.0'
