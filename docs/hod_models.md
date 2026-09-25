# HOD models

An HOD gives the mean number of central and satellite galaxies in a halo of
mass $M$. Centrals are Bernoulli draws, satellites Poisson draws. Masses are
in $M_\odot/h$; mass parameters are $\log_{10}$ values, written
`log10Mmin` / `log10M1` in the analytical model and `Mmin` / `M1` in the
numerical one. Below, $\mu = \log_{10} M$.

| Model | Centrals | Satellites | Analytical | Numerical |
|-------|----------|------------|:---:|:---:|
| `LRG` | error function | power law | ✓ | ✓ |
| `ELG_GHOD` | Gaussian | power law | ✓ | ✓ |
| `ELG_SFR` | Gaussian + power-law tail | power law | ✓ | ✓ |
| `ELG_MHMQ` / `ELG_mHMQ` | skewed Gaussian | power law | ✓ | ✓ |
| `CSMF` | conditional stellar mass function | idem | ✓ | |

## Centrals

**LRG** (Zheng et al. 2007) — a smoothed step, rising to $A_c$ above
$M_{\min}$:

$$\langle N_{\rm cen}\rangle = \frac{A_c}{2}\left[1 + {\rm erf}\left(\frac{\mu - \mu_{\min}}{\sigma_M}\right)\right]$$

:::{note}
The numerical model divides by $\sqrt{2}\,\sigma_M$ instead of $\sigma_M$, so
the same `sig_M` gives a step $\sqrt{2}$ wider there. The ELG forms below are
identical in the two models.
:::

**ELG_GHOD** (Gaussian HOD; Avila et al. 2020) — star-forming centrals live
in a narrow mass range:

$$\langle N_{\rm cen}\rangle = \frac{A_c}{\sqrt{2\pi}\,\sigma_M}\exp\left[-\frac{(\mu - \mu_{\min})^2}{2\sigma_M^2}\right]$$

**ELG_SFR** — the Gaussian below $M_{\min}$, a power law in $\mu$ above it
(continuous at $M_{\min}$):

$$\langle N_{\rm cen}\rangle = \frac{A_c}{\sqrt{2\pi}\,\sigma_M}\left(\frac{\mu}{\mu_{\min}}\right)^{\gamma}, \qquad \mu \ge \mu_{\min}$$

**ELG_mHMQ** (modified high-mass quenched; Alam et al. 2020, Rocher et al.
2023) — the Gaussian times an asymmetric factor; $\gamma > 0$ keeps a tail
at high mass:

$$\langle N_{\rm cen}\rangle = N_{\rm GHOD}(\mu)\left[1 + {\rm erf}\left(\frac{\gamma(\mu - \mu_{\min})}{\sqrt{2}\,\sigma_M}\right)\right]$$

## Satellites

All four share one power law, switched on above $\kappa M_{\min}$:

$$\langle N_{\rm sat}\rangle = A_s\left(\frac{M - \kappa M_{\min}}{M_1}\right)^{\alpha}, \qquad M > \kappa M_{\min}$$

The numerical model has two alternatives:

- **ELG cut-off** — `set_halo_model(..., elg_satellite=True)`, with
  parameters `Mcut` and `Mmax`:
  $\langle N_{\rm sat}\rangle = A_s (M/M_1)^{\alpha}\, e^{-M_{\rm cut}/M}\, e^{-M/M_{\max}}$.
- **Conformity** — `set_halo_model(..., conformity=True)`: in halos where a
  central was drawn, $M_1$ becomes $\kappa_{EE} M_1$ (AbacusHOD; Yuan et al.
  2022). $\kappa_{EE} < 1$ puts more satellites around existing centrals.

## Normalisation

$w_{gg}$ and $\Delta\Sigma$ depend on $A_c$ and $A_s$ only through their
ratio; the absolute value sets $n_{\rm gal}$. Both models can fix it by
rescaling $(A_c, A_s)$ together to a target density (see
[Analytical model](analytical.md)).

## Assembly bias (numerical model)

With `assembly_bias=True`, the occupation also depends on a secondary halo
property $f_I$ (internal, e.g. concentration) and/or $f_E$ (environment,
e.g. tidal shear), with amplitudes `A_cent`, `B_cent`, `A_sat`, `B_sat`.
`set_halo_model(..., ab_method=...)` picks the form:

| `ab_method` | Centrals | Satellites |
|---|---|---|
| `"mass"` (default) | $\mu_{\min} \to \mu_{\min} + A f_I + B f_E$ | $\mu_1 \to \mu_1 + A f_I + B f_E$ |
| `"variant"` (Hadzhiyska et al. 2023) | $N_{\rm cen}[1 + (A f_I + B f_E)(1 - N_{\rm cen})]$ | $N_{\rm sat}[1 + A f_I + B f_E]$ |
| `"direct"` (Hearin et al. 2016) | $N_{\rm cen} + \delta \min(N_{\rm cen}, 1 - N_{\rm cen})$ | $N_{\rm sat}(1 + \delta)$ |

In `"mass"`, $A$ and $B$ are in dex and $B > 0$ lowers the occupation of
halos with high $f_E$. In `"direct"`, $\delta = A\,{\rm sign}(f_I) + B\,{\rm sign}(f_E)$: only
which side of the median a halo falls on matters. `ab_rank=True` replaces $f_I$, $f_E$ by their rank within
0.1 dex mass bins, uniform in $[-1, 1]$.

## CSMF (analytical model)

The conditional stellar mass function (Yang et al. 2008; Dvornik et al.
2023) predicts galaxies in a stellar-mass bin
$[M_*^{\min}, M_*^{\max}]$, so one set of parameters describes several
lens bins at once. Parameters: `M0`, `M1` ($\log_{10}$), `gamma1`, `gamma2`,
`sigma_c`, `alpha_s`, `b0`, `b1`.

- **Centrals** — a log-normal of width $\sigma_c$ around the
  stellar-to-halo mass relation
  $M_*^c(M) = M_0\,\dfrac{(M/M_1)^{\gamma_1}}{(1 + M/M_1)^{\gamma_1 - \gamma_2}}$,
  integrated over the bin:
  $\langle N_{\rm cen}\rangle = \tfrac12\left[{\rm erf}(x_{\max}) - {\rm erf}(x_{\min})\right]$,
  $x = \log_{10}(M_*/M_*^c)/(\sqrt2\,\sigma_c)$.
- **Satellites** — a modified Schechter function integrated over the bin,
  $\Phi_s(M_*) = \dfrac{\phi_s}{M_*^s}\left(\dfrac{M_*}{M_*^s}\right)^{\alpha_s} e^{-(M_*/M_*^s)^2}$,
  with $M_*^s = 0.56\,M_*^c$ and
  $\log_{10}\phi_s = b_0 + b_1 \log_{10}(M/10^{13})$.

`HaloModel(..., hod_type="CSMF")` needs `Mstar_min` and `Mstar_max` (one
value per redshift); `CSMFFitter` sets them from the data files.
