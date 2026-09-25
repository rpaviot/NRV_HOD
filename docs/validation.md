# Validation

Each test in `example_scripts/` compares one part of NRVpy with an
independent calculation.

| Script | Compares | Status |
|--------|----------|--------|
| `null_test_analytical.py` | Analytical $P_{gg}$, $P_{gm}$ vs pyccl's own halo model | < 0.4%, natural and $h$ units |
| `cross_check_tabulated.py` | Tabulated $\Delta\Sigma$ / $w_{gg}$ vs the direct Monte-Carlo pipeline | 0.6% on $\Delta\Sigma$ |
| `cross_check_analytical_numerical.py` | Analytical vs numerical $\Delta\Sigma$, same HOD and cosmology | < 2.5% below 0.45 Mpc/$h$ and above 3 Mpc/$h$ with $\beta^{\rm NL}$; +10–20% at 0.6–2 Mpc/$h$ (halo model 1-halo/2-halo transition) |
| `benchmark_dsigma_convergence.py` | Direct $\Delta\Sigma$ vs particle / galaxy downsampling and number of realisations | reports the cheapest setting within a tolerance |

`cross_check_analytical_numerical.py` needs the numerical baseline written by
`numerical_dsigma_example.py`, which stores the HOD it populated so both
sides evaluate the same occupation:

```
python example_scripts/numerical_dsigma_example.py \
    --halo_path halos.parquet --particle_path particles.parquet \
    --cache baseline.npz
python example_scripts/cross_check_analytical_numerical.py --cache baseline.npz
```

The two sides also differ in the halo mass function: the analytical side
uses a fitting function (Tinker10 by default), the numerical side the halos
in the box. For Flamingo at $z=1$ this shifts $n_{\rm gal}$ by ~25%; evaluated
on the same halos, the two agree to 0.3%.
