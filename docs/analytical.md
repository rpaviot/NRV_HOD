# Analytical model

`HOD_NRV.HOD_analytical` is a halo model written in JAX on top of pyccl.
pyccl provides the linear power spectrum, halo mass function, halo bias and
concentration; NRVpy adds the HOD, the 1- and 2-halo power spectra and their
projection to $w_{gg}$ and $\Delta\Sigma$.

## Set up a model

```python
import numpy as np
from HOD_NRV.HOD_analytical import HaloModel

cosmo = {"h": 0.681, "Omc": 0.2566, "Omb": 0.0486, "Omnu": 1.39e-3,
         "A_s": 2.099e-9, "n_s": 0.967}          # or "s8" instead of "A_s"

model = HaloModel(
    cosmo_params=cosmo,
    z=1.0,                        # one redshift or a list
    hod_type="ELG_MHMQ",          # LRG, ELG_GHOD, ELG_SFR, ELG_MHMQ, CSMF
    units_per_h=True,             # k in h/Mpc, masses in Msun/h
    mass_definition="MassDef200m",
    mass_function="Tinker08",     # any pyccl MassFunc*, default Tinker10
)
```

The halo-model ingredients are pyccl names: `mass_function`, `halo_bias`
(default Tinker10), `concentration` (default Duffy08) and `mass_definition`
(default `MassDef200c`). Everything that does not depend on the HOD — mass
function, bias, NFW profiles, linear $P(k)$ — is computed once here, so a new
set of HOD parameters costs only the mass integrals.

## Predictions

```python
hod = {"Ac": 0.1, "log10Mmin": 11.8, "sig_M": 0.4, "gamma": 4.0,
       "As": 0.03, "log10M1": 13.0, "alpha": 0.8, "kappa": 1.0}

model.ngal(hod)                    # (Mpc/h)^-3
model.satellite_fraction(hod)
model.effective_halo_mass(hod)
model.Pgg(hod), model.Pgm(hod)     # on model.get_k()

rp_bins = np.geomspace(0.1, 50, 16)
rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])
rp, ds  = model.DeltaSigma(rp, rp_bins=rp_bins, hod_params=hod)
rp, wgg = model.wgg(rp, rp_bins=rp_bins, hod_params=hod)
```

With `rp_bins` the outputs are averaged over each bin, as a measurement is.

### Non-linear halo bias

`include_beta_nl=True` adds the $\beta^{\rm NL}(k, M_1, M_2)$ correction of
Mead & Verde (2021) to the 2-halo term, interpolated from the Dark Emulator
(Nishimichi et al. 2019). It matters in the 1-halo to 2-halo transition,
$r_p \sim 1$–$10$ Mpc/$h$. The emulator is defined for $M_{200m}$ halos, so
use `mass_definition="MassDef200m"` with it.

```python
model = HaloModel(cosmo, z=1.0, hod_type="ELG_MHMQ", units_per_h=True,
                  mass_definition="MassDef200m", include_beta_nl=True,
                  beta_nl_kwargs={"log_M_min": 11.5, "log_M_max": 15.0})
```

## Fixing the number density (optional)

$w_{gg}$ and $\Delta\Sigma$ are normalised by the galaxy number density, so
they depend on the central and satellite amplitudes only through their
ratio $A_c/A_s$. Their absolute value sets only $n_{\rm gal}$. Two ways to
handle it:

- **Rescale.** Sample the other parameters and derive $(A_c, A_s)$ at every
  step so that $n_{\rm gal}$ equals a target, keeping their ratio:

  ```python
  from HOD_NRV.HOD_analytical import rescale_Ac_to_target_ngal
  Ac, As = rescale_Ac_to_target_ngal(model, hod, target_ngal=2e-4)
  ```

  This removes one parameter that the two-point functions cannot
  constrain. `AnalyticalHODFitter` does this at every likelihood call.
- **Fit $n_{\rm gal}$.** Keep both amplitudes free and add the measured
  number density to the likelihood. This uses the abundance as information,
  but then the predicted $n_{\rm gal}$ carries any error of the halo mass
  function (for Flamingo at $z=1$, Tinker10 is ~25% high and Tinker08
  ~8% high in the ELG mass range).

## Fitting

Two fitters wrap the model.

**`AnalyticalHODFitter`** (`HOD_analytical.analytical_sampler`) fits
$\Delta\Sigma$ and optionally $w_{gg}$ with the LRG / ELG HODs at fixed
$n_{\rm gal}$, with separate scale cuts for each probe, and runs iminuit
(`minimize`) or nautilus (`run`).

**`CSMFFitter`** (`HOD_analytical.sampler`) fits a conditional stellar mass
function HOD (Yang et al. 2008; Dvornik et al. 2023) to
$\Delta\Sigma$ in several stellar-mass bins at once. This is how the
UNIONS × DESI analysis used the package:

```python
from HOD_NRV.HOD_analytical.sampler import (
    CSMFFitter, DEFAULT_CSMF_PRIORS, DEFAULT_COSMO_PARAMS, ParameterPrior)

fitter = CSMFFitter(
    cosmo_params=dict(DEFAULT_COSMO_PARAMS),
    observables=["DeltaSigma"],          # + "ngal" to anchor the abundance
    rp_min=0.1, rp_max=30.0,
    include_beta_nl=True,
    units_per_h=True,
    halo_model_kwargs={"mass_definition": "MassDef200m"},
)
fitter.load_lrg_data("data/", mass_bins=[0, 1, 2, 3])        # one file per bin
fitter.load_bgs_data("data/", mass_bins=[0, 1, 2], selection="SFR")

priors = dict(DEFAULT_CSMF_PRIORS)
priors["gamma1"] = ParameterPrior(name="gamma1", prior_type="gaussian",
                                  mean=7.0, std=2.0)
fitter.set_priors(priors)     # fixed_params={...} pins any of them

fitter.minimize_de(maxiter=500, workers=8)    # global search, or
fitter.minimize(run_hesse=True)               # iminuit + Hesse errors, or
fitter.run(n_live=2000, n_eff=10000)          # nautilus posterior
fitter.save_results("csmf_fit.npz")
```

`load_lrg_data` also applies the lens-magnification correction. The
priors cover the CSMF parameters and the concentration normalisations
$f_h$, $f_s$ of the matter and satellite profiles.
