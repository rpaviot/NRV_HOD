"""
Cross-check: Analytical vs Numerical DeltaSigma
=================================================

Compares DeltaSigma predictions from the numerical (pair-counting) and
analytical (halo model + Hankel transform) pipelines using the same HOD
parameters and cosmology.

Numerical baseline is loaded from baseline_dsigma_cache.npz (produced by
numerical_dsigma_example.py). If the cache is missing, an error is raised.

Data: Flamingo L1000N1800 for the numerical side.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from HOD_NRV.HOD_analytical.halo_model import HaloModel

# ============================================================================
# Shared configuration
# ============================================================================

BASELINE_CACHE = "baseline_dsigma_cache.npz"

Lbox = 681.0   # Mpc/h
zeff = 1.0

# Cosmology -- numerical uses H0/Om0/Ob0, analytical uses h/Omc/Omb

dict_cosmo = {
        'h': 0.681,
        'Omc': 0.306-0.0486-1.39e-3,
        'Omb': 0.0486,
        'A_s': 2.099e-9,
        'n_s': 0.967,
        'Omnu':1.39e-3
    }


# Base HOD parameters (numerical naming: Mmin, M1 in log10)
base_hod_params_numerical = {
    "As": 0.3,
    "Mmin": 13.0,
    "sig_M": 0.3,
    "M1": 13.0,
    "alpha": 0.80,
    "kappa": 0.80,
}

target_ngal = 2e-4  # (Mpc/h)^-3

# Output options
SAVE_TABLE = True
SAVE_PLOT  = True
OUTPUT_DIR = "."
TABLE_FILE = "cross_check_results.txt"
PLOT_FILE  = "cross_check_dsigma.png"

# Non-linear bias options (passed to HaloModel as beta_nl_kwargs)
beta_nl_opts = {
    'n_k': 50,  'n_mass': 30,
    'k_min': 1e-2,  'k_max': 30.0,
    'log_M_min': 11.5,  'log_M_max': 15.0,
    'method': 'linear',
    'bias_method': 'halo-halo',
    'force_to_zero': 'additive',
    'k_lin': 0.02,
    'constant_low': False,
    'verbose': True,
}

# ============================================================================
# Part 1: Load numerical baseline from cache
# ============================================================================

print("=" * 70)
print("Cross-check: Numerical vs Analytical DeltaSigma")
print("=" * 70)

print("\n--- NUMERICAL PIPELINE (from cache) ---")

if not os.path.exists(BASELINE_CACHE):
    raise FileNotFoundError(
        f"Baseline cache not found: {BASELINE_CACHE}\n"
        f"Run numerical_dsigma_example.py first to generate it."
    )

cache = np.load(BASELINE_CACHE)
ds_all = cache["ds_all"]
rp_num = cache["rp"]
ngal_arr = cache["ngal"]
sat_frac_arr = cache["sat_frac"]
Ac_rescaled = np.array([cache["Ac"]])
As_rescaled = np.array([cache["As"]])

ds_numerical = ds_all.mean(axis=0)
ngal_numerical = ngal_arr.mean()
sat_frac_numerical = sat_frac_arr.mean()

print(f"  Loaded {len(ds_all)} realizations from {BASELINE_CACHE}")
print(f"  Ac = {Ac_rescaled[0]:.6f}, As = {As_rescaled[0]:.6f}")
print(f"  ngal (mean)     = {ngal_numerical:.6e}")
print(f"  sat_frac (mean) = {sat_frac_numerical:.4f}")

# Shared lensing bins (derived from cache)
rp_bins = np.geomspace(0.1, 50.0, 16)
rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])

# ============================================================================
# Part 2: Analytical pipeline
# ============================================================================

print("\n--- ANALYTICAL PIPELINE ---")

# Convert numerical HOD parameter names to analytical names:
#   numerical Mmin  -> analytical log10Mmin
#   numerical M1    -> analytical log10M1
hod_params_analytical = {
    "Ac": Ac_rescaled,
    "log10Mmin": base_hod_params_numerical["Mmin"],       # already log10
    "sig_M": base_hod_params_numerical["sig_M"],
    "As": As_rescaled,
    "log10M1": base_hod_params_numerical["M1"],            # already log10
    "alpha": base_hod_params_numerical["alpha"],
    "kappa": base_hod_params_numerical["kappa"],
}

# The analytical module uses pyccl mass definitions via string.
# For virial mass, use "MassDefVir".
print("Initializing HaloModel ...")
model = HaloModel(
    cosmo_params=dict_cosmo,
    z=zeff,
    hod_type="ELG_GHOD",
    units_per_h=True,
    mass_definition="MassDef200m",
    verbose=True,
)

model.set_hod_params(hod_params_analytical)

ngal_analytical = float(model.ngal())
print(f"  ngal (analytical) = {ngal_analytical:.6e}")

print("Computing analytical DeltaSigma ...")
rp_ana, ds_analytical = model.DeltaSigma(
    rp_centers, rp_bins=rp_bins, method="direct",
    hod_params=hod_params_analytical, include_stellar=False,
)

# ============================================================================
# Part 2b: Analytical pipeline -- non-linear bias (beta^NL)
# ============================================================================

print("\n--- ANALYTICAL PIPELINE (beta^NL) ---")
print("Initializing HaloModel with include_beta_nl=True ...")
model_nl = HaloModel(
    cosmo_params=dict_cosmo,
    z=zeff,
    hod_type="ELG_GHOD",
    units_per_h=True,
    mass_definition="MassDef200m",
    include_beta_nl=True,
    beta_nl_kwargs=beta_nl_opts,
    verbose=True,
)

# Guard: warn if beta^NL caches are empty (missing interpax / dark_emulator)
if len(model_nl._beta_nl_gl_cache) == 0:
    print("  WARNING: beta^NL caches are empty -- results will be identical to linear.")

model_nl.set_hod_params(hod_params_analytical)

print("Computing analytical DeltaSigma (beta^NL) ...")
rp_ana_nl, ds_analytical_nl = model_nl.DeltaSigma(
    rp_centers, rp_bins=rp_bins, method="direct",
    hod_params=hod_params_analytical, include_stellar=False,
)

# ============================================================================
# Part 3: Comparison
# ============================================================================

print("\n--- COMPARISON ---")
print(f"\n  ngal numerical  = {ngal_numerical:.6e}  (Mpc/h)^-3")
print(f"  ngal analytical = {ngal_analytical:.6e}  (Mpc/h)^-3")

header = (f"{'rp [Mpc/h]':>12s}  {'DS_num':>14s}  {'DS_ana_L':>14s}  "
          f"{'DS_ana_NL':>14s}  {'num/L':>10s}  {'NL/L':>10s}")
print(f"\n{header}")
print("-" * 82)

table_lines = []
for i in range(len(rp_num)):
    r = rp_num[i]
    ds_n = ds_numerical[i]
    ds_l = ds_analytical[i] if i < len(ds_analytical) else np.nan
    ds_nl = ds_analytical_nl[i] if i < len(ds_analytical_nl) else np.nan
    ratio_num_l = ds_n / ds_l if ds_l != 0 else np.nan
    ratio_nl_l = ds_n / ds_l if ds_l != 0 else np.nan
    line = (f"  {r:10.4f}    {ds_n:12.6e}    {ds_l:12.6e}    "
            f"{ds_nl:12.6e}    {ratio_num_l:8.4f}    {ratio_num_l:8.4f}")
    print(line)
    table_lines.append(line)

if SAVE_TABLE:
    table_path = os.path.join(OUTPUT_DIR, TABLE_FILE)
    with open(table_path, "w") as f:
        f.write(f"# Cross-check: Analytical vs Numerical DeltaSigma\n")
        f.write(f"# z = {zeff},  target_ngal = {target_ngal:.2e} (Mpc/h)^-3\n")
        f.write(f"# ngal numerical  = {ngal_numerical:.6e} (Mpc/h)^-3\n")
        f.write(f"# ngal analytical = {ngal_analytical:.6e} (Mpc/h)^-3\n")
        f.write(f"# Units: rp in Mpc/h, DeltaSigma in h*Msun/pc^2\n")
        f.write(f"#\n")
        f.write(f"# {header}\n")
        f.write(f"# {'-' * 80}\n")
        for tl in table_lines:
            f.write(tl + "\n")
    print(f"\nTable saved to {table_path}")

# ============================================================================
# Part 4: Plot
# ============================================================================

print("\n--- PLOTTING ---")

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 7),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

# Top panel: rp * DeltaSigma
ax_top.semilogx(
    rp_num, rp_num * ds_numerical,
    "ko", ms=5, label="Numerical",
)
ax_top.semilogx(
    rp_ana, rp_ana * ds_analytical,
    "bs--", ms=4, lw=1.2, label="Analytical (linear)",
)
ax_top.semilogx(
    rp_ana_nl, rp_ana_nl * ds_analytical_nl,
    "r^--", ms=4, lw=1.2, label=r"Analytical ($\beta^{\rm NL}$)",
)
ax_top.set_ylabel(r"$r_p \; \Delta\Sigma$ [Mpc/h $\cdot$ $h\,M_\odot/\mathrm{pc}^2$]")
ax_top.legend(fontsize=9)
ax_top.set_title("DeltaSigma cross-check: numerical vs analytical")

# Bottom panel: fractional difference (analytical / numerical - 1)
frac_l = ds_analytical / ds_numerical - 1.0
frac_nl = ds_analytical_nl / ds_numerical - 1.0

ax_bot.axhline(0, color="k", lw=0.8)
ax_bot.axhspan(-0.10, 0.10, color="gray", alpha=0.15, label=r"$\pm 10\%$")
ax_bot.semilogx(rp_num, frac_l, "bs--", ms=4, lw=1.2, label="Linear")
ax_bot.semilogx(rp_num, frac_nl, "r^--", ms=4, lw=1.2, label=r"$\beta^{\rm NL}$")
ax_bot.set_xlabel(r"$r_p$ [Mpc/$h$]")
ax_bot.set_ylabel("ana / num $-$ 1")
ax_bot.legend(fontsize=8, ncol=3)
ax_bot.set_ylim(-0.5, 0.5)

if SAVE_PLOT:
    plot_path = os.path.join(OUTPUT_DIR, PLOT_FILE)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
else:
    plt.show()

print("\nDone.")
