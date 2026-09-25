# NRVpy

**N**extgen **R**ealisations of **V**irialized structures — a halo occupation
distribution (HOD) code that fits projected galaxy clustering w_gg(r_p) and
galaxy–galaxy lensing ΔΣ(r_p).

Documentation: https://nrvpy.readthedocs.io

## Two models

**Analytical** (`HOD_NRV.HOD_analytical`) — a JAX halo model on top of
pyccl: P_gg, P_gm, w_gg and ΔΣ from an HOD (LRG, ELG, or a conditional
stellar mass function) and a halo mass function, with optional non-linear
halo bias from the Dark Emulator. Includes iminuit / nautilus fitters.

**Numerical** (`HOD_NRV.HOD_numerical`) — populates the halos of an N-body or
hydro simulation (parquet catalogues, any halo finder via a column mapping)
and measures w_gg and ΔΣ from pair counts. Supports assembly bias along any
halo property, conformity, and spherical, elliptical or extended satellite
profiles. Tabulated (TabCorr-style) estimators give noise-free predictions
fast enough for nested sampling.

## Install

```
git clone https://github.com/rpaviot/NRV_HOD.git && cd NRV_HOD
pip install -e .
```

Optional: `pycorr` (pair counts), `dark_emulator` + `interpax` (non-linear
bias), `nautilus-sampler`, `iminuit`.

## Quick start

```python
import numpy as np
from HOD_NRV.HOD_analytical import HaloModel

cosmo = {"h": 0.681, "Omc": 0.2566, "Omb": 0.0486, "A_s": 2.099e-9, "n_s": 0.967}
model = HaloModel(cosmo, z=1.0, hod_type="LRG", units_per_h=True,
                  mass_definition="MassDef200m")
hod = {"Ac": 1.0, "log10Mmin": 13.0, "sig_M": 0.3,
       "As": 0.5, "log10M1": 14.0, "alpha": 1.0, "kappa": 1.0}

rp_bins = np.geomspace(0.1, 50, 16)
rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])
rp, ds = model.DeltaSigma(rp, rp_bins=rp_bins, hod_params=hod)
```

```python
import pandas as pd
from HOD_NRV.HOD_numerical.HOD import HaloOccupation

halo = HaloOccupation(cosmology=cosmo, zeff=1.0, Lbox=681.0,
                      column_mapping=column_mapping,  # your catalogue's names
                      mass_definition="MassDef200m",
                      DataFrame=pd.read_parquet("halos.parquet"),
                      DataFrame_part=pd.read_parquet("particles.parquet"))
halo.set_halo_model("LRG")
halo.populate_haloes({"Ac": 1.0, "Mmin": 13.0, "sig_M": 0.3,
                      "As": 0.5, "M1": 14.0, "alpha": 1.0, "kappa": 1.0})
rp, ds = halo.compute_galaxy_lensing(rp_bins)
```

## Validation

`example_scripts/` holds the tests against independent references (pyccl,
the direct Monte-Carlo pipeline, analytical vs numerical); see the
[validation page](https://nrvpy.readthedocs.io/en/latest/validation.html).

## Authors

Romain Paviot, with Claude (Anthropic).
