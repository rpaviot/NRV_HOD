import jax
jax.config.update("jax_enable_x64", True)

"""
HOD Numerical Module
====================

This module contains the numerical implementation of HOD modeling, including:
- Core HOD catalogue and population engine
- NFW profile implementations for satellite positioning
- Utility functions for data reading and cosmology setup
- Two-point correlation functions and lensing calculations
- Test modules for validation

Submodules
----------
- HOD: Main HOD catalogue and population engine
- satellites: NFW profile implementations
- utils: Data reading, cosmology, and utility functions
- twopoint_calculator: Two-point statistics and galaxy-galaxy lensing
- test: Test and validation modules
"""

__all__ = ['HOD', 'satellites', 'utils', 'twopoint_calculator', 'test']
