"""
NRVpy - Halo Occupation Distribution (HOD) Code
================================================

A Python-based HOD code for modeling galaxy-halo connections in cosmological simulations.
Performs halo occupation fitting to projected clustering and galaxy-galaxy lensing 2-point functions.

Main Classes
------------
- HaloOccupation: Main user interface for halo occupation modeling
- Occupation: HOD model implementations
- DeltaSigmaCalculator: Galaxy-galaxy lensing calculator
- FastDeltaSigmaCalculator: Optimized lensing calculator with pre-computation

Main Functions
--------------
- compute_galaxy_clustering: Measure galaxy clustering correlation functions
- compute_galaxy_lensing: Measure galaxy-galaxy lensing signals
"""

# Core HOD classes and population functions
from .HOD_numerical.HOD.HOD_catalogue import HaloOccupation
from .HOD_models import Occupation
from .HOD_numerical.HOD.population_engine import (
    populate_centrals,
    populate_satellites,
    populate_haloes_full,
    combine_galaxy_populations,
    apply_rsd_to_galaxies
)

# NFW profile implementations
from .HOD_numerical.satellites import NFW_jax, NFW

# Utility functions
from .utilsf import utils_functions, data_reader, emulator_utils, assembly_bias_environment

# Two-point statistics
from .HOD_numerical.twopoint_calculator.standard_two_point_calculator import (
    compute_corr,
    DeltaSigmaCalculator,
    compute_galaxy_clustering,
    compute_galaxy_lensing
)

# Fast two-point calculator
from .HOD_numerical.twopoint_calculator.fast_two_point import FastDeltaSigmaCalculator
from .HOD_numerical.twopoint_calculator.precompute_deltasigma import (
    build_particle_kdtree,
    precompute_lensing_grid,
    save_precomputed_lensing,
    load_precomputed_lensing
)

__all__ = [
    # Main classes
    "HaloOccupation",
    "Occupation",

    # Population functions
    "populate_centrals",
    "populate_satellites",
    "populate_haloes_full",
    "combine_galaxy_populations",
    "apply_rsd_to_galaxies",

    # NFW modules
    "NFW_jax",
    "NFW",

    # Utility modules
    "utils_functions",
    "data_reader",
    "emulator_utils",
    "assembly_bias_environment",

    # Standard two-point functions
    "compute_corr",
    "DeltaSigmaCalculator",
    "compute_galaxy_clustering",
    "compute_galaxy_lensing",

    # Fast two-point calculator
    "FastDeltaSigmaCalculator",
    "build_particle_kdtree",
    "precompute_lensing_grid",
    "save_precomputed_lensing",
    "load_precomputed_lensing",
]

__version__ = '0.1.0'
__author__ = 'NRVpy Development Team'
__license__ = 'MIT'