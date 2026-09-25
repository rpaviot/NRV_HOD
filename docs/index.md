# NRVpy

**N**extgen **R**ealisations of **V**irialized structures — a halo occupation
distribution (HOD) code that fits projected galaxy clustering $w_{gg}(r_p)$
and galaxy–galaxy lensing $\Delta\Sigma(r_p)$.

NRVpy models the galaxy–halo connection two ways, one page each:

1. [Analytical model](analytical.md) — a JAX halo model built on pyccl:
   $P_{gg}$, $P_{gm}$, $w_{gg}$ and $\Delta\Sigma$ from an HOD and a halo mass
   function, with optional non-linear halo bias ($\beta^{\rm NL}$, Dark
   Emulator). Fast enough to sample directly; used for the UNIONS × DESI
   lensing fits.
2. [Numerical model](numerical.md) — populates the halos of an N-body or
   hydro simulation with galaxies and measures $w_{gg}$ and $\Delta\Sigma$
   from pair counts. Supports assembly bias along any halo property,
   conformity, and spherical, elliptical or extended satellite profiles. The
   tabulated estimators turn it into a likelihood fast enough for nested
   sampling.

[HOD models](hod_models.md) gives the occupation functions both use.
[Validation](validation.md) lists the tests that check each part against an
independent reference.

## Install

```
git clone https://github.com/rpaviot/NRV_HOD.git && cd NRV_HOD
pip install -e .
```

Optional: `pycorr` (pair counts, numerical model), `dark_emulator` +
`interpax` ($\beta^{\rm NL}$), `nautilus-sampler` and `iminuit` (fitting).
pyccl needs CAMB (installed with pyccl from conda-forge; with pip,
`pip install camb`).

## Units

Positions and radii in Mpc/$h$, masses in $M_\odot/h$ with HOD mass
parameters in $\log_{10}$, $\Delta\Sigma$ in $h\,M_\odot\,{\rm pc}^{-2}$,
$w_{gg}$ in Mpc/$h$. The analytical model follows these conventions when
`units_per_h=True` (then $k$ is in $h$/Mpc).

```{toctree}
:hidden:
:maxdepth: 1

hod_models
analytical
numerical
validation
```
