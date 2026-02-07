"""
NULL TEST: pyccl Halo Model vs Optimized JAX Implementation
============================================================

This script validates the MultiRedshiftHaloModel against pyccl's built-in
halo model implementation.

Unit conversion rules (CCL natural -> h-units):
    k [h/Mpc]                = k [1/Mpc] / h
    P(k) [(Mpc/h)³]          = P(k) [Mpc³] × h³
    M_halo [Msun/h]          = M [Msun] × h
    M_stellar [Msun/h²]      = M* [Msun] × h²
    dn/dlogM [(Mpc/h)⁻³]     = dn/dlogM [Mpc⁻³] / h²  (NOT h³!)
    ρ_m [(Msun/h)/(Mpc/h)³]  = ρ_m [Msun/Mpc³] / h²
    n_gal [(Mpc/h)⁻³]        = n_gal [Mpc⁻³] / h³

Author: Null test implementation
"""

import numpy as np
import jax
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import pyccl as ccl
import time

# Import our halo model
from halo_model import HaloModel, StandardHOD, CSMF_HOD
from HOD_NRV.utilsf.hankel_transforms import Pk_to_DeltaSigma_direct

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

print("="*70)
print("NULL TEST: pyccl vs Optimized JAX Halo Model")
print("WITH CORRECT UNIT CONVERSIONS")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")


# ============================================================================
# Cosmology Setup
# ============================================================================

dict_cosmo = {
    'h': 0.6766,
    'Omc': 0.11933/(0.6766)**2,
    'Omb': 0.02242/(0.6766)**2,
    's8': 0.8102,
    'A_s': 2.105209331337507e-09,
    'n_s': 0.9665,
    'Omnu': 0.0014034
}

Omc = dict_cosmo['Omc']
Omb = dict_cosmo['Omb']
mnu = 0.06
As = dict_cosmo['A_s']
ns = dict_cosmo['n_s']
h = dict_cosmo['h']

cosmo = ccl.Cosmology(
    Omega_c=Omc, 
    Omega_b=Omb,
    m_nu=mnu, 
    h=h, 
    A_s=As, 
    n_s=ns,
    transfer_function='boltzmann_camb',
    matter_power_spectrum='camb',
    mass_split='normal',
    extra_parameters={"camb": {"halofit_version": "mead2020"}}
)

print(f"\nCosmology: Ωc={Omc:.4f}, Ωb={Omb:.4f}, h={h}, σ8={dict_cosmo['s8']}")


# ============================================================================
# HOD Parameters
# ============================================================================

# Natural units parameters (Msun) - this is what pyccl uses internally
hod_params_natural = {
    'log10Mmin': 12.0,   # log10(Mmin / [Msun])
    'siglnM': 0.4,
    'log10M0': 11.5,     # log10(M0 / [Msun])
    'log10M1': 13.3,     # log10(M1 / [Msun])
    'alpha': 1.0
}

# h-units parameters (Msun/h) - convert from natural by subtracting log10(h)
# M [Msun/h] = M [Msun] / h  =>  log10(M [Msun/h]) = log10(M [Msun]) - log10(h)
log10h = np.log10(h)
hod_params_h_units = {
    'log10Mmin': 12.0 + log10h,   # log10(Mmin / [Msun/h])
    'siglnM': 0.4,
    'log10M0': 11.5 + log10h,
    'log10M1': 13.3 + log10h,
    'alpha': 1.0
}

print("\nStandard Zheng HOD Parameters:")
print("  In natural units (Msun) - pyccl internal:")
for key, val in hod_params_natural.items():
    print(f"    {key:15s} = {val}")
print(f"\n  In h-units (Msun/h) - for units_per_h=True mode:")
for key, val in hod_params_h_units.items():
    if isinstance(val, float) and 'log10' in key:
        print(f"    {key:15s} = {val:.4f}")
    else:
        print(f"    {key:15s} = {val}")


# ============================================================================
# Test Configuration
# ============================================================================

z_array = np.array([0.1, 0.3, 0.4, 0.5])
k_min = 1e-3  # 1/Mpc (natural)
k_max = 100   # 1/Mpc (natural)
n_k = 512
M_min = 1e9   # Msun (natural)
M_max = 1e16  # Msun (natural)
n_M = 256

print(f"\nTest configuration:")
print(f"  Redshifts: {z_array}")
print(f"  k range: [{k_min}, {k_max}] 1/Mpc (natural units)")
print(f"  M range: [{M_min:.0e}, {M_max:.0e}] Msun (natural units)")


# ============================================================================
# PYCCL Reference Implementation
# ============================================================================

def compute_pyccl_power_spectra(cosmo, hod_params_h, z_array, 
                                 k_min=1e-3, k_max=100, n_k=256,
                                 M_min=1e9, M_max=1e16, n_M=256,
                                 units_per_h=False):
    """
    Compute P_gg and P_gm using pyccl's built-in halo model.
    
    Parameters
    ----------
    cosmo : ccl.Cosmology
    hod_params_h : dict
        HOD parameters. pyccl HaloProfileHOD expects h-units internally.
    z_array : array
        Redshifts
    k_min, k_max : float
        k range in NATURAL units (1/Mpc)
    M_min, M_max : float
        Mass range in NATURAL units (Msun)
    units_per_h : bool
        If True, return results in h-units
        If False, return in natural units
    
    Returns
    -------
    k_out : array
        k in h/Mpc if units_per_h else 1/Mpc
    Pk_gg, Pk_gm : arrays
        Power spectra
    n_gal : array
        Galaxy density
    """
    h_val = cosmo['h']
    
    # k and M arrays in natural units for CCL internal computation
    k_arr_natural = np.geomspace(k_min, k_max, n_k)  # 1/Mpc
    M_arr_natural = np.geomspace(M_min, M_max, n_M)  # Msun
    
    # CCL halo model setup
    mass_def = ccl.halos.MassDef200c
    concentration = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    mass_func = ccl.halos.MassFuncTinker10(mass_def=mass_def)
    halo_bias = ccl.halos.HaloBiasTinker10(mass_def=mass_def)

    hod_params_h2 = hod_params_h.copy()
    if units_per_h==True:
        hod_params_h2['log10Mmin'] = hod_params_h['log10Mmin'] - np.log10(h_val)
        hod_params_h2['log10M0'] = hod_params_h['log10M0'] - np.log10(h_val)
        hod_params_h2['log10M1'] = hod_params_h['log10M1'] - np.log10(h_val)

    
    # IMPORTANT: pyccl HaloProfileHOD expects mass parameters in h-units!
    prof_hod = ccl.halos.HaloProfileHOD(
        mass_def=mass_def,
        concentration=concentration,
        log10Mmin_0=hod_params_h2['log10Mmin'],
        siglnM_0=hod_params_h2['siglnM'],
        log10M0_0=hod_params_h2['log10M0'],
        log10M1_0=hod_params_h2['log10M1'],
        alpha_0=hod_params_h2['alpha'],
        fc_0=1.0,
        ns_independent=True
    )
    
    prof_nfw = ccl.halos.HaloProfileNFW(
        mass_def=mass_def,
        concentration=concentration,
        fourier_analytic=True
    )
    
    hmc = ccl.halos.HMCalculator(
        mass_function=mass_func,
        halo_bias=halo_bias,
        mass_def=mass_def
    )
    
    prof_2pt = ccl.halos.Profile2ptHOD()
    
    n_z = len(z_array)
    
    # CCL computes everything in natural units internally
    Pk_gg = np.zeros((n_z, n_k))
    Pk_gm = np.zeros((n_z, n_k))
    n_gal = np.zeros(n_z)
    
    for iz, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        
        # Power spectra in Mpc³ (natural)
        Pk_gg[iz] = ccl.halos.halomod_power_spectrum(
            cosmo, hmc, k_arr_natural, a,
            prof_hod, prof_2pt=prof_2pt
        )
        
        Pk_gm[iz] = ccl.halos.halomod_power_spectrum(
            cosmo, hmc, k_arr_natural, a,
            prof_hod, prof2=prof_nfw
        )
        
        # Galaxy number density in Mpc⁻³ (natural)
        n_M_arr = mass_func(cosmo, M_arr_natural, a)  # Mpc⁻³
        N_g = np.array([prof_hod._Nc(M, a) + prof_hod._Ns(M, a) for M in M_arr_natural])
        n_gal[iz] = simpson(N_g * n_M_arr, x=np.log10(M_arr_natural))
    
    # Unit conversion if requested
    if units_per_h:
        k_out = k_arr_natural / h_val          # 1/Mpc -> h/Mpc
        Pk_gg = Pk_gg * h_val**3               # Mpc³ -> (Mpc/h)³
        Pk_gm = Pk_gm * h_val**3               # Mpc³ -> (Mpc/h)³
        n_gal = n_gal / h_val**3               # Mpc⁻³ -> (Mpc/h)⁻³
    else:
        k_out = k_arr_natural
    

    return k_out, Pk_gg, Pk_gm, n_gal


# ============================================================================
# Run Null Test
# ============================================================================

def run_null_test(units_per_h: bool):
    """Run null test with specified unit convention"""
    
    unit_label = "h-units" if units_per_h else "natural units"
    k_unit = "h/Mpc" if units_per_h else "1/Mpc"
    P_unit = "(Mpc/h)³" if units_per_h else "Mpc³"
    n_unit = "(h/Mpc)³" if units_per_h else "Mpc⁻³"
    
    print("\n" + "="*70)
    print(f"TESTING WITH {unit_label.upper()}")
    print("="*70)
    
    # ====================================================================
    # PYCCL Reference
    # ====================================================================
    
    print("\n" + "-"*50)
    print("COMPUTING PYCCL REFERENCE")
    print("-"*50)


    if units_per_h:
        input_params = hod_params_h_units
    else:
        input_params = hod_params_natural
    
    t0 = time.time()
    k_pyccl, Pk_gg_pyccl, Pk_gm_pyccl, ngal_pyccl = compute_pyccl_power_spectra(
        cosmo, input_params, z_array,
        k_min=k_min, k_max=k_max, n_k=n_k,
        M_min=M_min, M_max=M_max, n_M=n_M,
        units_per_h=units_per_h
    )
    t_pyccl = time.time() - t0
    print(f"Time: {t_pyccl:.2f}s")
    
    print(f"\nGalaxy densities [{n_unit}]:")
    for iz, z in enumerate(z_array):
        print(f"  z={z:.1f}: n_gal = {ngal_pyccl[iz]:.6e}")
    
    # ====================================================================
    # Optimized JAX Implementation
    # ====================================================================

    print("\n" + "-"*50)
    print("COMPUTING OPTIMIZED JAX (HaloModel)")
    print("-"*50)

    # Choose parameters based on unit convention
    # units_per_h=True: params in h-units, model converts internally
    # units_per_h=False: params already in natural units
    if units_per_h:
        input_params = hod_params_h_units
    else:
        input_params = hod_params_natural

    # When units_per_h=True, HaloModel expects k_array in h/Mpc
    # Convert from natural units (1/Mpc) so both models evaluate at the same physical k
    if units_per_h:
        k_array_input = np.geomspace(k_min / h, k_max / h, n_k)  # h/Mpc
    else:
        k_array_input = np.geomspace(k_min, k_max, n_k)  # 1/Mpc

    t0 = time.time()
    model = HaloModel(
        cosmo_params=dict_cosmo,
        z=z_array,
        hod_type='standard',
        k_array=k_array_input,
        M_min=M_min,
        M_max=M_max,
        units_per_h=units_per_h,
        verbose=True
    )
    model.set_hod_params(input_params)
    t_init = time.time() - t0
    print(f"Init time: {t_init:.2f}s")

    t0 = time.time()
    Pk_gg_jax = model.Pgg()
    Pk_gm_jax = model.Pgm()
    ngal_jax = model.ngal()
    t_compute = time.time() - t0
    print(f"Compute time: {t_compute*1000:.1f}ms")

    k_jax = model.get_k()
    
    # ====================================================================
    # Comparison
    # ====================================================================
    
    print("\n" + "-"*50)
    print("COMPARISON")
    print("-"*50)
    
    print(f"\nn_gal differences:")
    for iz, z in enumerate(z_array):
        diff_pct = (ngal_jax[iz] - ngal_pyccl[iz]) / ngal_pyccl[iz] * 100
        print(f"  z={z:.1f}: JAX={ngal_jax[iz]:.6e}, pyccl={ngal_pyccl[iz]:.6e}, diff={diff_pct:+.4f}%")
    
    print(f"\nMax |ΔP/P| (k > 0.01 {k_unit}):")
    for iz, z in enumerate(z_array):
        k_mask = k_jax > (0.01 if units_per_h else 0.01 * h)
        frac_diff_gg = np.abs((Pk_gg_jax[iz] - Pk_gg_pyccl[iz]) / Pk_gg_pyccl[iz])[k_mask]
        frac_diff_gm = np.abs((Pk_gm_jax[iz] - Pk_gm_pyccl[iz]) / Pk_gm_pyccl[iz])[k_mask]
        print(f"  z={z:.1f}: P_gg={np.max(frac_diff_gg)*100:.4f}%, P_gm={np.max(frac_diff_gm)*100:.4f}%")
    
    return {
        'k_jax': k_jax,
        'k_pyccl': k_pyccl,
        'Pk_gg_jax': Pk_gg_jax,
        'Pk_gg_pyccl': Pk_gg_pyccl,
        'Pk_gm_jax': Pk_gm_jax,
        'Pk_gm_pyccl': Pk_gm_pyccl,
        'ngal_jax': ngal_jax,
        'ngal_pyccl': ngal_pyccl
    }


# ============================================================================
# Test CSMF HOD
# ============================================================================

def test_csmf_hod():
    """Test the CSMF HOD implementation"""

    print("\n" + "="*70)
    print("TESTING CSMF HOD")
    print("="*70)

    # CSMF parameters (example values)
    csmf_params = {
        'M0': 10.5,        # log10(M0/Msun) - characteristic stellar mass
        'M1': 12.0,        # log10(M1/Msun) - halo mass for SHMR
        'gamma1': 3.0,     # low-mass slope
        'gamma2': 0.5,     # high-mass slope
        'sigma_c': 0.2,    # scatter
        'alpha_s': -1.0,   # satellite slope
        'b0': 0.5,         # normalization intercept
        'b1': 1.0          # normalization slope
    }

    # Stellar mass bin (in Msun for natural units)
    Mstar_min = 10**10.0
    Mstar_max = 10**11.0

    print(f"\nCSMF Parameters:")
    for key, val in csmf_params.items():
        print(f"  {key}: {val}")
    print(f"\nStellar mass bin: [{Mstar_min:.2e}, {Mstar_max:.2e}] Msun")

    # Create model
    model = HaloModel(
        cosmo_params=dict_cosmo,
        z=z_array,
        hod_type='csmf',
        k_array=np.geomspace(k_min, k_max, n_k),
        M_min=M_min,
        M_max=M_max,
        Mstar_min=Mstar_min,
        Mstar_max=Mstar_max,
        units_per_h=False,  # Natural units
        masses_are_log10=True,
        verbose=True
    )
    model.set_hod_params(csmf_params)

    # Compute power spectra
    t0 = time.time()
    Pk_gg = model.Pgg()
    Pk_gm = model.Pgm()
    n_gal = model.ngal()
    print(f"\nCompute time: {(time.time()-t0)*1000:.1f}ms")

    print(f"\nCSMF Results:")
    for iz, z in enumerate(z_array):
        print(f"  z={z:.1f}: n_gal={n_gal[iz]:.4e} Mpc⁻³, P_gg(k=0.1)={Pk_gg[iz, 100]:.2e} Mpc³")

    return model, Pk_gg, Pk_gm, n_gal


# ============================================================================
# Generate Comparison Plot
# ============================================================================

def generate_comparison_plot(results_h, save_path='null_test_comparison.png'):
    """Generate comparison plot with h-units"""
    
    k_jax = results_h['k_jax']
    k_pyccl = results_h['k_pyccl']
    Pk_gg_jax = results_h['Pk_gg_jax']
    Pk_gg_pyccl = results_h['Pk_gg_pyccl']
    Pk_gm_jax = results_h['Pk_gm_jax']
    Pk_gm_pyccl = results_h['Pk_gm_pyccl']
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    colors = ['C0', 'C1', 'C2', 'C3']
    
    for iz, z in enumerate(z_array):
        # P_gg
        axes[0, iz].loglog(k_pyccl, Pk_gg_pyccl[iz], 'k-', lw=2, label='pyccl')
        axes[0, iz].loglog(k_jax, Pk_gg_jax[iz], '--', color=colors[iz], lw=2, label='JAX')
        axes[0, iz].set_xlabel(r'$k$ [$h$/Mpc]')
        axes[0, iz].set_ylabel(r'$P_{gg}(k)$ [(Mpc/$h$)$^3$]')
        axes[0, iz].set_title(f'z = {z:.1f}')
        axes[0, iz].legend()
        axes[0, iz].grid(True, alpha=0.3)
        
        # P_gm
        axes[1, iz].loglog(k_pyccl, Pk_gm_pyccl[iz], 'k-', lw=2, label='pyccl')
        axes[1, iz].loglog(k_jax, Pk_gm_jax[iz], '--', color=colors[iz], lw=2, label='JAX')
        axes[1, iz].set_xlabel(r'$k$ [$h$/Mpc]')
        axes[1, iz].set_ylabel(r'$P_{gm}(k)$ [(Mpc/$h$)$^3$]')
        axes[1, iz].legend()
        axes[1, iz].grid(True, alpha=0.3)
        
        # Fractional difference
        frac_diff_gg = (Pk_gg_jax[iz] - Pk_gg_pyccl[iz]) / Pk_gg_pyccl[iz] * 100
        frac_diff_gm = (Pk_gm_jax[iz] - Pk_gm_pyccl[iz]) / Pk_gm_pyccl[iz] * 100
        
        axes[2, iz].semilogx(k_jax, frac_diff_gg, 'b-', lw=1.5, label=r'$P_{gg}$')
        axes[2, iz].semilogx(k_jax, frac_diff_gm, 'r-', lw=1.5, label=r'$P_{gm}$')
        axes[2, iz].axhline(0, color='k', ls=':', alpha=0.5)
        axes[2, iz].fill_between(k_jax, -1, 1, color='gray', alpha=0.2)
        axes[2, iz].set_xlabel(r'$k$ [$h$/Mpc]')
        axes[2, iz].set_ylabel('Diff [%]')
        axes[2, iz].set_ylim([-5, 5])
        axes[2, iz].legend()
        axes[2, iz].grid(True, alpha=0.3)
    
    plt.suptitle('NULL TEST: pyccl vs JAX Halo Model (h-units)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig


# ============================================================================
# DeltaSigma Null Test (h-units only)
# ============================================================================

def test_delta_sigma():
    """
    Compare DeltaSigma between pyccl reference and JAX HaloModel.

    Both use the same Pk_to_DeltaSigma_direct Hankel transform,
    so this tests whether the Pgm -> DeltaSigma pipeline is consistent.
    """
    print("\n" + "="*70)
    print("DELTA SIGMA NULL TEST (h-units)")
    print("="*70)

    rp_bins = np.logspace(-1, 1.7, 20)  # Mpc/h
    rp = np.sqrt(rp_bins[:-1] * rp_bins[1:])  # geometric bin centres

    # ------------------------------------------------------------------
    # JAX HaloModel
    # ------------------------------------------------------------------
    print("\n" + "-"*50)
    print("COMPUTING JAX DeltaSigma")
    print("-"*50)

    k_array_h = np.geomspace(k_min / h, k_max / h, n_k)  # h/Mpc

    model = HaloModel(
        cosmo_params=dict_cosmo,
        z=z_array,
        hod_type='standard',
        k_array=k_array_h,
        M_min=M_min,
        M_max=M_max,
        units_per_h=True,
        verbose=True
    )
    model.set_hod_params(hod_params_h_units)

    rp_jax, ds_jax = model.DeltaSigma(rp, rp_bins=rp_bins, include_stellar=False)
    if ds_jax.ndim == 1:
        ds_jax = ds_jax[np.newaxis, :]

    print(f"  rp range: [{rp_jax[0]:.2f}, {rp_jax[-1]:.2f}] Mpc/h")
    for iz, z in enumerate(z_array):
        print(f"  z={z:.1f}: DS(rp=1) ~ {ds_jax[iz, np.argmin(np.abs(rp_jax - 1.0))]:.4e} h*Msun/pc^2")

    # ------------------------------------------------------------------
    # pyccl reference: compute Pk_gm then same Hankel transform
    # ------------------------------------------------------------------
    print("\n" + "-"*50)
    print("COMPUTING PYCCL DeltaSigma")
    print("-"*50)

    k_pyccl_h, _, Pk_gm_pyccl, _ = compute_pyccl_power_spectra(
        cosmo, hod_params_h_units, z_array,
        k_min=k_min, k_max=k_max, n_k=n_k,
        M_min=M_min, M_max=M_max, n_M=n_M,
        units_per_h=True
    )

    # rho_m in h-units (same as model.RHO_M)
    rho_m_h = ccl.rho_x(cosmo, 1.0, 'matter', is_comoving=True) / h**2

    ds_pyccl = np.zeros((len(z_array), len(rp)))
    for iz in range(len(z_array)):
        rp_out, ds_iz = Pk_to_DeltaSigma_direct(
            k_pyccl_h, Pk_gm_pyccl[iz], rho_m_h, rp, rp_bins=rp_bins
        )
        ds_pyccl[iz] = ds_iz

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "-"*50)
    print("COMPARISON")
    print("-"*50)

    rp_mask = (rp_jax > 0.3) & (rp_jax < 30.0)
    print(f"\nMax |ΔDS/DS| (0.3 < rp < 30 Mpc/h):")
    for iz, z in enumerate(z_array):
        frac = np.abs((ds_jax[iz] - ds_pyccl[iz]) / ds_pyccl[iz])[rp_mask]
        print(f"  z={z:.1f}: {np.max(frac)*100:.4f}%")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, len(z_array), figsize=(5*len(z_array), 8))
    colors = ['C0', 'C1', 'C2', 'C3']

    for iz, z in enumerate(z_array):
        # DeltaSigma
        axes[0, iz].semilogx(rp_jax, rp*ds_jax[iz], '--', color=colors[iz], lw=2, label='JAX')
        axes[0, iz].semilogx(rp_jax, rp*ds_pyccl[iz], 'k-', lw=2, label='pyccl')
        axes[0, iz].set_xlabel(r'$r_p$ [Mpc/$h$]')
        axes[0, iz].set_ylabel(r'$\Delta\Sigma$ [$h\,M_\odot/\mathrm{pc}^2$]')
        axes[0, iz].set_title(f'z = {z:.1f}')
        axes[0, iz].legend()
        axes[0, iz].grid(True, alpha=0.3)

        # Fractional difference
        frac_diff = (ds_jax[iz] - ds_pyccl[iz]) / ds_pyccl[iz] * 100
        axes[1, iz].semilogx(rp_jax, frac_diff, '-', color=colors[iz], lw=1.5)
        axes[1, iz].axhline(0, color='k', ls=':', alpha=0.5)
        axes[1, iz].fill_between(rp_jax, -1, 1, color='gray', alpha=0.2)
        axes[1, iz].set_xlabel(r'$r_p$ [Mpc/$h$]')
        axes[1, iz].set_ylabel('Diff [%]')
        axes[1, iz].set_ylim([-5, 5])
        axes[1, iz].grid(True, alpha=0.3)

    plt.suptitle(r'NULL TEST: $\Delta\Sigma$ — pyccl vs JAX (h-units)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = 'null_test_DeltaSigma.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    return {
        'rp': rp_jax,
        'ds_jax': ds_jax,
        'ds_pyccl': ds_pyccl,
    }


# ============================================================================
# MCMC Usage Example
# ============================================================================

def mcmc_usage_example():
    """Demonstrate efficient MCMC usage pattern"""

    print("\n" + "="*70)
    print("MCMC USAGE EXAMPLE")
    print("="*70)

    # Initial parameters (StandardHOD format)
    initial_params = {
        'log10Mmin': 12.0,
        'siglnM': 0.4,
        'log10M0': 11.5,
        'log10M1': 13.3,
        'alpha': 1.0
    }

    # Create model once (expensive: pre-computes CCL quantities)
    print("\nCreating model (one-time cost)...")
    t0 = time.time()
    model = HaloModel(
        cosmo_params=dict_cosmo,
        z=z_array,
        hod_type='standard',
        units_per_h=True,
        verbose=True
    )
    model.set_hod_params(initial_params)
    print(f"Init time: {time.time()-t0:.2f}s")

    # Simulate MCMC iterations
    n_iterations = 100
    print(f"\nSimulating {n_iterations} MCMC iterations...")

    times = []
    for i in range(n_iterations):
        # Vary parameters (as would happen in MCMC)
        new_params = {
            'log10Mmin': 12.0 + 0.1 * np.random.randn(),
            'siglnM': 0.4 + 0.05 * np.random.randn(),
            'log10M0': 11.5 + 0.1 * np.random.randn(),
            'log10M1': 13.3 + 0.1 * np.random.randn(),
            'alpha': 1.0 + 0.1 * np.random.randn()
        }

        t0 = time.time()
        model.set_hod_params(new_params)
        Pk_gg = model.Pgg()
        Pk_gm = model.Pgm()
        n_gal = model.ngal()
        times.append(time.time() - t0)

    times = np.array(times)
    print(f"\nPer-iteration statistics:")
    print(f"  Mean time: {np.mean(times)*1000:.2f} ms")
    print(f"  Std time:  {np.std(times)*1000:.2f} ms")
    print(f"  Min time:  {np.min(times)*1000:.2f} ms")
    print(f"  Max time:  {np.max(times)*1000:.2f} ms")
    print(f"\nEffective throughput: {1.0/np.mean(times):.1f} iterations/sec")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    
    # Run null tests for both unit conventions
    print("\n\n" + "#"*70)
    print("# RUNNING NULL TESTS")
    print("#"*70)
    
    results_natural = run_null_test(units_per_h=False)
    results_h = run_null_test(units_per_h=True)
    
    # Test CSMF HOD
    print("\n\n" + "#"*70)
    print("# TESTING CSMF HOD")
    print("#"*70)
    
    csmf_model, Pk_gg_csmf, Pk_gm_csmf, n_gal_csmf = test_csmf_hod()
    
    # Generate comparison plot
    print("\n\n" + "#"*70)
    print("# GENERATING COMPARISON PLOT")
    print("#"*70)
    
    fig = generate_comparison_plot(results_h)

    # DeltaSigma null test (h-units)
    print("\n\n" + "#"*70)
    print("# DELTA SIGMA NULL TEST")
    print("#"*70)

    ds_results = test_delta_sigma()

    # MCMC usage example
    print("\n\n" + "#"*70)
    print("# MCMC USAGE DEMONSTRATION")
    print("#"*70)
    
    mcmc_usage_example()

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The MultiRedshiftHaloModel class provides:

1. CORRECT UNIT CONVERSIONS:
   - Input: HOD masses in h-units (Msun/h) when units_per_h=True
   - Internal: All computations in natural units via pyccl
   - Output: k, P(k), n_gal in h-units when units_per_h=True
   
2. SUPPORT FOR BOTH HOD TYPES:
   - StandardHOD: Classic Zheng et al. parameterization
   - CSMF_HOD: Conditional Stellar Mass Function (Dvornik+2022)
   
3. OPTIMIZED FOR MCMC:
   - Pre-computes CCL quantities once during initialization
   - update_hod_params() allows efficient parameter updates
   - Full vmap over all k and z for fast computation
   
4. DICT-BASED PARAMETERS:
   - Parameters passed as dictionaries
   - Easy integration with samplers like nautilus
   
Usage for MCMC:
    model = MultiRedshiftHaloModel(cosmo, 'standard', params, z_array, units_per_h=True)
    for params in sampler:
        model.update_hod_params(params)
        Pk_gg, Pk_gm, n_gal = model.compute_both_all_z(verbose=False)
        # compute likelihood...
    """)