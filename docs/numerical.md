# Numerical model

`HOD_NRV.HOD_numerical` puts galaxies in the halos of a simulation box and
measures their two-point functions. Where the analytical model assumes the
halo mass function, bias and profiles, here they are whatever the simulation
contains — including assembly bias, halo exclusion and the matter beyond the
virial radius.

## Set up a box

```python
import pandas as pd
from HOD_NRV.HOD_numerical.HOD import HaloOccupation

column_mapping = {"x": "x", "y": "y", "z": "z",          # Mpc/h
                  "vx": "vx", "vy": "vy", "vz": "vz",    # km/s
                  "mass": "mass", "radius": "rvir",      # Msun/h, kpc/h
                  "c": "c", "vrms": "vrms"}

halo = HaloOccupation(
    cosmology={"h": 0.681, "Omc": 0.2566, "Omb": 0.0486, "Omnu": 1.39e-3,
               "A_s": 2.099e-9, "n_s": 0.967},   # same keys as the analytical model
    zeff=1.0, Lbox=681.0,
    column_mapping=column_mapping,
    mass_definition="MassDef200m",
    DataFrame=pd.read_parquet("halos.parquet"),
    DataFrame_part=pd.read_parquet("particles.parquet"),  # only for lensing
    apply_rsd=True, rsd_axis="z",
    do_test=False,
)
halo.set_halo_model("ELG_mHMQ")      # LRG, ELG_GHOD, ELG_SFR, ELG_mHMQ
```

`column_mapping` maps the names NRVpy expects to your catalogue's columns,
so any halo finder works. Options on top of the default spherical NFW
satellites:

- **Assembly bias** — `assembly_bias=True` and an `fI` and/or `fE` column
  (any halo property: concentration, formation time, local density, tidal
  shear). `set_halo_model(..., ab_method=...)` chooses how it acts: `"mass"`
  shifts $\log M_{\min}$ and $\log M_1$ by $A f_I + B f_E$ in dex (Yuan et
  al. 2022); `"variant"` modulates the occupation directly (Hadzhiyska et al.
  2023); `"direct"` is the step of Hearin et al. (2016). `ab_rank=True` uses
  the property's rank at fixed mass rather than its value.
- **Conformity** — `set_halo_model(..., conformity=True)` makes the satellite
  occupation depend on whether a central was actually drawn, through
  `kappa_EE` (AbacusHOD, Yuan et al. 2022).
- **Elliptical satellites** — `triaxial_NFW=True` with the halo shape
  columns (`a_x…`, `b_x…`, `b_to_a`, `c_to_a`).
- **Extended profile** — the HOD parameters `f_exp`, `tau` and `lambda_NFW`
  add an exponential tail beyond $R_{\rm vir}$ and rescale the NFW
  concentration (Rocher et al. 2023).

## Populate

```python
hod = {"Ac": 0.02, "Mmin": 11.7, "sig_M": 0.5, "gamma": 4.0,
       "As": 0.05, "M1": 13.0, "alpha": 0.7, "kappa": 1.0}
halo.populate_haloes(hod, random_seed=1)

halo.positions_gal, halo.satellite_fraction
```

Each call is one Monte-Carlo realisation. As in the analytical model, the
two-point functions depend on $A_c/A_s$ only;
`HOD_numerical.HOD_models.rescale_Ac_to_target_ngal` fixes $n_{\rm gal}$.

## Direct estimator

Pair counts on the populated catalogue, with pycorr:

```python
import numpy as np
rp_bins = np.geomspace(0.1, 50, 16)

rp, wp = halo.compute_galaxy_clustering("rppi", rp_bins,
                                        bins2=np.linspace(-80, 80, 161),
                                        output="wp")
rp, ds = halo.compute_galaxy_lensing(rp_bins)   # galaxy x particle
```

$\Delta\Sigma$ comes from the galaxy–matter correlation function
$\xi_{gm}(r)$, projected and averaged over each $r_p$ bin. It needs the
particles, and it is noisy: average several realisations. The cost scales
with the number of particles and galaxies; `particle_fraction` (in
`HaloOccupation`) and galaxy subsampling trade accuracy for speed.
`example_scripts/benchmark_dsigma_convergence.py` measures how far each can
go for your box.

A faster variant, `compute_galaxy_lensing_optimized`, reads the central
term from a per-halo cache (below) and only pair-counts the satellites.

## Tabulated estimators

Both two-point functions are linear in the halo occupation. So instead of
populating the box, NRVpy tabulates the halo–matter and halo–halo
correlations once, in bins of halo mass (and of the assembly-bias property),
and predicts the two-point functions as occupation-weighted sums — the
TabCorr method of Zheng & Guo (2016) and Lange et al. (2019, 2025). The
prediction is the expectation value of the Monte-Carlo pipeline, without its
noise, in well under a second: fast enough to call inside a nested sampler.

```python
from HOD_NRV.HOD_numerical.twopoint_calculator.halo_center_lensing import (
    precompute_halo_center_lensing, HaloCenterLensingCache, TabulatedDeltaSigma)

cache = precompute_halo_center_lensing(          # once per box (hours)
    halo_positions=np.asarray(halo.positions),
    particle_positions=np.asarray(halo.positions_part),
    Lbox=halo.Lbox, rsd_axis=halo.rsd_axis, RHO_M=halo.RHO_M,
    rp_bins=rp_bins, halo_logM=np.asarray(halo.logM))
cache.save("ds_cache.h5")

tab = TabulatedDeltaSigma(HaloCenterLensingCache.load("ds_cache.h5"), halo)
rp, ds, info = tab.predict(hod)
```

- **Centrals** are exact: each halo's own $\Delta\Sigma$ profile, weighted
  by its $\langle N_{\rm cen}\rangle$, so assembly bias is carried halo by
  halo.
- **Satellites** use the mean profile of their (mass, property) bin,
  convolved with the satellite offset distribution. The radial profile is
  analytic, so $f_{\rm exp}$, $\tau$ and $\lambda_{\rm NFW}$ can vary freely
  with no retabulation.

`TabulatedWgg` (`twopoint_calculator.tabulated_wgg`) does the same for
$w_{gg}$ from halo–halo pair counts between bins
(`precompute_wgg_tabulation`).

Not supported by the tabulation: elliptical satellite profiles (they break
the isotropic-offset convolution). `example_scripts/cross_check_tabulated.py`
checks both estimators against the direct Monte-Carlo calculation.

## Fitting

`TabulatedFitter` (`HOD_NRV.utilsf.numerical_sampler`) runs nautilus on the
tabulated likelihood: $\Delta\Sigma$ and optionally $w_{gg}$, with separate
scale cuts, $(A_c, A_s)$ rescaled to a target $n_{\rm gal}$ at every call,
and a JAX `jit`/`vmap` batched likelihood (`vectorized=True`). With assembly
bias free, anchor the rescaling on the catalogue (`ngal_anchor="catalogue"`,
the default): only the halos themselves know how the property reweights them
at fixed mass.
