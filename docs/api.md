# API

The main entry points, from their docstrings. HOD names are
case-insensitive in both models.

## Analytical model

```python
from HOD_NRV.HOD_analytical import (
    HaloModel, create_hod, rescale_Ac_to_target_ngal,
    AnalyticalHODFitter, CSMFFitter, ParameterPrior)
```

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_analytical.HaloModel
   :members: set_hod_params, ngal, satellite_fraction, effective_halo_mass, Pgg, Pgm, DeltaSigma, wgg

.. autofunction:: HOD_NRV.HOD_analytical.rescale_Ac_to_target_ngal

.. autofunction:: HOD_NRV.HOD_analytical.create_hod
```

### Fitting

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_analytical.AnalyticalHODFitter
   :members: log_likelihood, minimize, run

.. autoclass:: HOD_NRV.HOD_analytical.CSMFFitter
   :members: load_data, set_priors, minimize, minimize_de, run, get_best_fit, save_results

.. autoclass:: HOD_NRV.HOD_analytical.ParameterPrior
```

## Numerical model

```python
from HOD_NRV.HOD_numerical.HOD import HaloOccupation
from HOD_NRV.HOD_numerical.HOD_models import rescale_Ac_to_target_ngal
```

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_numerical.HOD.HaloOccupation
   :members: set_halo_model, populate_haloes, compute_galaxy_clustering, compute_galaxy_lensing, compute_galaxy_lensing_optimized

.. autofunction:: HOD_NRV.HOD_numerical.HOD_models.rescale_Ac_to_target_ngal
```

### Tabulated estimators

```python
from HOD_NRV.HOD_numerical.twopoint_calculator import (
    precompute_halo_center_lensing, HaloCenterLensingCache, TabulatedDeltaSigma,
    precompute_wgg_tabulation, WggTabulation, TabulatedWgg)
```

```{eval-rst}
.. autofunction:: HOD_NRV.HOD_numerical.twopoint_calculator.precompute_halo_center_lensing

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.HaloCenterLensingCache
   :members: save, load

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.TabulatedDeltaSigma
   :members: predict

.. autofunction:: HOD_NRV.HOD_numerical.twopoint_calculator.precompute_wgg_tabulation

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.WggTabulation
   :members: save, load

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.TabulatedWgg
   :members: predict
```

### Fitting

```python
from HOD_NRV.utilsf.numerical_sampler import TabulatedFitter
```

```{eval-rst}
.. autoclass:: HOD_NRV.utilsf.numerical_sampler.TabulatedFitter
   :members: run, log_likelihood, build_batched_loglike
```
