# API

The main entry points, from their docstrings. HOD names are
case-insensitive in both models.

## Analytical model

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_analytical.halo_model.HaloModel
   :members: set_hod_params, ngal, satellite_fraction, effective_halo_mass, Pgg, Pgm, DeltaSigma, wgg

.. autofunction:: HOD_NRV.HOD_analytical.analytical_sampler.rescale_Ac_to_target_ngal

.. autofunction:: HOD_NRV.HOD_analytical.hod_analytical.create_hod
```

### Fitting

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_analytical.analytical_sampler.AnalyticalHODFitter
   :members: log_likelihood, minimize, run

.. autoclass:: HOD_NRV.HOD_analytical.sampler.CSMFFitter
   :members: load_lrg_data, load_bgs_data, set_priors, minimize, minimize_de, run, get_best_fit, save_results

.. autoclass:: HOD_NRV.HOD_analytical.sampler.ParameterPrior
```

## Numerical model

```{eval-rst}
.. autoclass:: HOD_NRV.HOD_numerical.HOD.HOD_catalogue.HaloOccupation
   :members: set_halo_model, populate_haloes, compute_galaxy_clustering, compute_galaxy_lensing, compute_galaxy_lensing_optimized

.. autofunction:: HOD_NRV.HOD_numerical.HOD_models.rescale_Ac_to_target_ngal
```

### Tabulated estimators

```{eval-rst}
.. autofunction:: HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing.precompute_halo_center_lensing

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing.HaloCenterLensingCache
   :members: save, load

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing.TabulatedDeltaSigma
   :members: predict

.. autofunction:: HOD_NRV.HOD_numerical.twopoint_calculator.tabulated_wgg.precompute_wgg_tabulation

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.tabulated_wgg.WggTabulation
   :members: save, load

.. autoclass:: HOD_NRV.HOD_numerical.twopoint_calculator.tabulated_wgg.TabulatedWgg
   :members: predict
```

### Fitting

```{eval-rst}
.. autoclass:: HOD_NRV.utilsf.numerical_sampler.TabulatedFitter
   :members: run, log_likelihood, build_batched_loglike
```
