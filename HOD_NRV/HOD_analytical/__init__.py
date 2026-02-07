"""
HOD Analytical Framework
========================

Semi-analytical halo model implementation for computing galaxy power spectra
using pyccl and JAX optimization.

This module provides:
- Standard Zheng et al. HOD models
- Conditional Stellar Mass Function (CSMF) HOD models
- Fast multi-redshift power spectrum calculations with JAX
- Efficient MCMC sampling support

Main Classes
------------
MultiRedshiftHaloModel : Primary interface for power spectrum calculations
StandardHOD : Standard Zheng et al. HOD parameterization
CSMF_HOD : Conditional Stellar Mass Function HOD
HODType : Enum for HOD model types

Example Usage
-------------
>>> from HOD_NRV.HOD_analytical import MultiRedshiftHaloModel, HODType
>>> import pyccl as ccl
>>>
>>> # Setup cosmology
>>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
>>>
>>> # Define HOD parameters
>>> hod_params = {
>>>     'log10Mmin': 12.0,
>>>     'siglnM': 0.4,
>>>     'log10M0': 11.5,
>>>     'log10M1': 13.3,
>>>     'alpha': 1.0
>>> }
>>>
>>> # Create model
>>> model = MultiRedshiftHaloModel(
>>>     cosmo,
>>>     hod_type='standard',
>>>     hod_params=hod_params,
>>>     z_array=[0.3, 0.5, 0.7],
>>>     units_per_h=True
>>> )
>>>
>>> # Compute power spectra
>>> Pk_gg, Pk_gm, n_gal = model.compute_both_all_z()

Notes
-----
This framework is separate from the numerical HOD implementation in the parent
HOD_NRV package. It uses pyccl for halo model calculations and JAX for optimization,
making it suitable for MCMC sampling and parameter inference.
"""

from .hod_analytical import (
    # HOD classes and factory
    AnalyticalHOD,
    StandardHOD,
    CSMF_HOD,
    create_hod,

    # Parameter definitions
    HOD_PARAM_DEFINITIONS,
    CSMF_HOD_PARAMS,

    # Validation
    validate_hod_params,
)

from .halo_model import (
    # Main class
    HaloModel,
)

__all__ = [
    # Main classes
    'HaloModel',
    'AnalyticalHOD',
    'StandardHOD',
    'CSMF_HOD',

    # Factory and convenience functions
    'create_hod',

    # Parameter definitions
    'HOD_PARAM_DEFINITIONS',
    'CSMF_HOD_PARAMS',

    # Validation
    'validate_hod_params',
]

__version__ = '0.1.0'
__author__ = 'NRVpy Development Team'
__description__ = 'Semi-analytical halo model for galaxy clustering and lensing'
