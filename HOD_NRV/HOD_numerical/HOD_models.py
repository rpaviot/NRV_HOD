import warnings
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jax import jit, vmap
from scipy.interpolate import CubicSpline as CS
from HOD_NRV.utilsf.utils_functions import gauss_legendre_integration

sqrtpi = jnp.sqrt(2 * jnp.pi)

@jit
def func_erf(x):
    return jsp.erf(x)

@jit
def LRG_Zheng07(logM, Ac, Mmin, sig_M):
    Ncen = Ac / 2. * (1 + func_erf((logM - Mmin)/(jnp.sqrt(2)*sig_M)))
    return Ncen

@jit
def ELG_GHOD(logM, Ac, Mmin ,sig_M):
    Ncen = Ac / (sqrtpi*sig_M) * jnp.exp(-((logM - Mmin) ** 2) / (2 * sig_M ** 2))
    return Ncen

@jit 
def ELG_mHMQ(logM, Ac,Mmin,sig_M,gamma):
    exp_part =ELG_GHOD(logM, Ac, Mmin ,sig_M)
    quench_fraction = 1 + func_erf((gamma*(logM - Mmin))/(jnp.sqrt(2)*sig_M))
    return exp_part*quench_fraction

@jit
def ELG_SFR(logM, Ac, Mmin, sig_M, gamma):
    exp_part =ELG_GHOD(logM, Ac, Mmin ,sig_M)
    power_part = Ac/(sqrtpi*sig_M) *jnp.power(logM/Mmin, gamma)
    Ncen = jnp.where(logM < Mmin, exp_part, power_part)
    return Ncen

@jit
def HOD_satellite(logM, As, Mmin, M1, alpha, kappa):
    Nsat = As * jnp.power((10**logM - kappa*10**Mmin) / (10**M1),alpha)
    Nsat = jnp.where(logM > Mmin + jnp.log10(kappa),Nsat,0)
    return Nsat

@jit
def HOD_satellite_conformity(logM, As, Mmin, M1, alpha, kappa, kappa_EE, has_central):
    """
    AbacusHOD-style conformity: satellite occupation depends on central presence

    Parameters
    ----------
    logM : halo mass
    As : satellite amplitude
    Mmin : minimum mass for centrals
    M1 : standard satellite mass scale
    alpha : satellite power law index
    kappa : satellite cutoff parameter
    kappa_EE : scaling factor for M1 when central present (M1_EE = kappa_EE * M1)
    has_central : boolean array indicating central presence

    Returns
    -------
    Satellite occupation probability

    Notes
    -----
    The effective satellite mass scale is M1_EE = kappa_EE * M1 when a central
    galaxy is present. This parametrization allows M1 to be fixed while still
    varying the conformity strength via kappa_EE.
    """
    # Compute M1_EE from scaling: M1_EE = kappa_EE * M1
    M1_EE = M1 + jnp.log10(kappa_EE)  # log10(M1_EE) = log10(kappa_EE * M1)

    # Use different M1 based on central presence
    M1_eff = jnp.where(has_central, M1_EE, M1)

    Nsat = As * jnp.power((10**logM - kappa*10**Mmin) / (10**M1_eff), alpha)
    Nsat = jnp.where(logM > Mmin + jnp.log10(kappa), Nsat, 0)
    return Nsat


@jit
def ELG_satellite_cutoff(logM, As, M1, alpha, Mcut, Mmax):
    M = 10**logM
    Nsat = As * jnp.power(M / 10**M1, alpha) * jnp.exp(-10**Mcut / M) * jnp.exp(-M / 10**Mmax)
    return Nsat


@jit
def ELG_satellite_conformity_cutoff(logM, As, M1, alpha, Mcut, Mmax, kappa_EE, has_central):
    M1_EE = M1 + jnp.log10(kappa_EE)          # log10(kappa_EE * M1)
    M1_eff = jnp.where(has_central, M1_EE, M1)
    M = 10**logM
    Nsat = As * jnp.power(M / 10**M1_eff, alpha) * jnp.exp(-10**Mcut / M) * jnp.exp(-M / 10**Mmax)
    return Nsat


def compute_ngal_(logM,mass_function,probC,probS):
    integrand = mass_function*(probC + probS)
    func_intg = CS(logM,integrand)
    ngal = gauss_legendre_integration(func_intg,logM.min(),logM.max())
    return ngal
    
def compute_fsat_(logM,mass_function,probC,probS):
    ngal = compute_ngal_(logM,mass_function,probC,probS)
    integrand = mass_function*probS
    func_intg = CS(logM,integrand)
    nsat = gauss_legendre_integration(func_intg,logM.min(),logM.max())
    return nsat/ngal

def compute_Meff_(logM,mass_function,probC,probS):
    ngal = compute_ngal_(logM,mass_function,probC,probS)
    integrand = mass_function*probC*10**logM
    func_intg = CS(logM,integrand)
    Mnum = gauss_legendre_integration(func_intg,logM.min(),logM.max())
    return Mnum/ngal

@jit
def assembly_bias_direct_cen(Ncen, AB_value):
    """Ncen_mod = Ncen + AB * min(Ncen, 1 - Ncen)  (Hearin et al. Eq. 8)

    fI/fE are step-function values (±1, above/below median), so with |A| ≤ 1
    the result stays in [0, 1] by construction.
    """
    return Ncen + AB_value * jnp.minimum(Ncen, 1.0 - Ncen)


@jit
def assembly_bias_direct_sat(Nsat, AB_value):
    """Nsat_mod = Nsat * (1 + AB)  (Hearin et al. Eq. 9)"""
    return Nsat * (1.0 + AB_value)


def assembly_bias_variant_cen(Ncen, A_cent, fI, B_cent, fE):
    """N_hat_cen = [1 + (A_cent·fI + B_cent·fE) · (1 − N_cen)] · N_cen  (paper Eq. 12)

    fI, fE are continuous values in [-1, 1] (log+percentile normalization).
    Physical constraint: |A_cent| + |B_cent| ≤ 1.
    """
    return Ncen * (1.0 + _ab_combine(A_cent, fI, B_cent, fE) * (1.0 - Ncen))


def assembly_bias_variant_sat(Nsat, A_sat, fI, B_sat, fE):
    """N_hat_sat = [1 + (A_sat·fI + B_sat·fE)] · N_sat  (paper Eq. 13)"""
    return Nsat * (1.0 + _ab_combine(A_sat, fI, B_sat, fE))


def _ab_combine(A, fI, B, fE):
    """Compute assembly bias amplitude A*fI + B*fE, skipping None fields."""
    return (A * fI if fI is not None else 0.0) + (B * fE if fE is not None else 0.0)


def assembly_bias_mass(logM, A, B, fI, fE):
    """Shift mass threshold by A*fI + B*fE."""
    return logM + _ab_combine(A, fI, B, fE)


def compute_ngal_with_fiducial_Ac(hod_model, params, Ac_fiducial=1.0):
    """Compute galaxy number density with a fiducial Ac value.

    If hod_model is a HaloOccupation (has a .HOD Occupation), routes through
    HOD.compute_ngal with Lbox. Otherwise calls compute_ngal directly.
    """
    params_with_Ac = {**params, 'Ac': Ac_fiducial}
    if hasattr(hod_model, 'HOD'):
        return hod_model.HOD.compute_ngal(params_with_Ac, Lbox=hod_model.Lbox)
    return hod_model.compute_ngal(params_with_Ac)


def rescale_Ac_to_target_ngal(hod_model, params, target_ngal, Ac_fiducial=1.0):
    """Rescale Ac and As by the same factor to hit target_ngal, preserving Ac/As ratio."""
    ngal_fiducial = compute_ngal_with_fiducial_Ac(hod_model, params, Ac_fiducial)
    rescale_factor = target_ngal / ngal_fiducial
    return Ac_fiducial * rescale_factor, params['As'] * rescale_factor


class Occupation:
    central_funcs = {
        "LRG": (LRG_Zheng07, ["Ac", "Mmin", "sig_M"]),
        "ELG_GHOD": (ELG_GHOD, ["Ac", "Mmin", "sig_M"]),
        "ELG_SFR": (ELG_SFR, ["Ac", "Mmin", "sig_M", "gamma"]),
        "ELG_mHMQ": (ELG_mHMQ, ["Ac", "Mmin", "sig_M", "gamma"]),
    }

    satellite_params=["As", "Mmin", "M1", "alpha", "kappa"]
    satellite_conformity_params=["As", "Mmin", "M1", "alpha", "kappa", "kappa_EE"]
    satellite_elg_params=["As", "M1", "alpha", "Mcut", "Mmax"]
    satellite_elg_conformity_params=["As", "M1", "alpha", "Mcut", "Mmax", "kappa_EE"]
    assembly_bias_params=['A_cent','B_cent','A_sat','B_sat']

    def __init__(self, hod_type, logM_bins, mass_function, assembly_bias=False,
                 conformity=False, elg_satellite=False, fI=None, fE=None,
                 ab_method="mass", logM_halos=None):

        if hod_type not in self.central_funcs:
            raise AttributeError(f"Unknown HOD type: {hod_type}")
        if ab_method not in ("direct", "variant", "mass"):
            raise ValueError(f"Unknown ab_method '{ab_method}'. Choose 'direct', 'variant', or 'mass'.")

        self.logM_bins = logM_bins
        self.mass_function=mass_function
        self.hod_type = hod_type
        self.HOD_central, self.central_params = self.central_funcs[hod_type]
        self.assembly_bias = assembly_bias
        self.conformity = conformity
        self.elg_satellite = elg_satellite
        self.ab_method = ab_method
        self.fI = fI
        self.fE = fE

        # Set satellite function and parameters based on conformity / elg_satellite
        if self.conformity and self.elg_satellite:
            self.HOD_satellite = ELG_satellite_conformity_cutoff
            self._sat_param_names = self.satellite_elg_conformity_params
        elif self.conformity:
            self.HOD_satellite = HOD_satellite_conformity
            self._sat_param_names = self.satellite_conformity_params
        elif self.elg_satellite:
            self.HOD_satellite = ELG_satellite_cutoff
            self._sat_param_names = self.satellite_elg_params
        else:
            self.HOD_satellite = HOD_satellite
            self._sat_param_names = self.satellite_params

        if self.assembly_bias:
            if fI is None and fE is None:
                raise ValueError(
                    "Assembly bias requires at least one of fI or fE."
                )
            ab_params = (["A_cent", "A_sat"] if fI is not None else []) + \
                        (["B_cent", "B_sat"] if fE is not None else [])
        else:
            ab_params = []
        self._ab_param_names = ab_params
        self.key = set(self.central_params + self._sat_param_names + ab_params)

        if assembly_bias and logM_halos is None:
            raise ValueError("assembly_bias=True requires logM_halos (per-halo logM array).")
        self.logM_halos = logM_halos

    def set_params(self, dict_params):
        required_keys = set(self.central_params + self._sat_param_names)
        try:
            [dict_params[key] for key in required_keys]
        except KeyError as e:
            raise KeyError(f"Missing an argument: {e.args[0]}")

        sat_params = self._sat_param_names

        self.central_args, self.satellite_args = (
            [dict_params[key] for key in self.central_params],
            [dict_params[key] for key in sat_params])
        
        self.params = dict_params

        if self.assembly_bias:
            A_cent = dict_params.get("A_cent", 0.0)
            B_cent = dict_params.get("B_cent", 0.0)
            A_sat  = dict_params.get("A_sat",  0.0)
            B_sat  = dict_params.get("B_sat",  0.0)

            if self.ab_method == "direct":
                # Binarize to step functions: +1 above median, -1 below
                # sign(0) = 0 for halos exactly at the median → no AB effect
                fI = jnp.sign(self.fI) if self.fI is not None else None
                fE = jnp.sign(self.fE) if self.fE is not None else None
                # Pre-compute per-halo AB amplitude (±A_cent ± B_cent per halo)
                self._ab_cent_val = _ab_combine(A_cent, fI, B_cent, fE)
                self._ab_sat_val  = _ab_combine(A_sat,  fI, B_sat,  fE)
            elif self.ab_method == "mass":
                # Shift mass thresholds: Mmin += A_cent*fI + B_cent*fE, M1 += A_sat*fI + B_sat*fE
                if "Mmin" in self.central_params:
                    id_cent = self.central_params.index("Mmin")
                    self.central_args[id_cent] = assembly_bias_mass(
                        dict_params["Mmin"], A_cent, B_cent, self.fI, self.fE
                    )
                if "M1" in self._sat_param_names:
                    id_sat = self._sat_param_names.index("M1")
                    self.satellite_args[id_sat] = assembly_bias_mass(
                        dict_params["M1"], A_sat, B_sat, self.fI, self.fE
                    )
            else:  # variant: continuous fI/fE
                self._A_cent = A_cent
                self._B_cent = B_cent
                self._A_sat  = A_sat
                self._B_sat  = B_sat


    def _apply_direct_ab(self, probC, probS):
        """Apply direct assembly bias to occupation numbers (ab_method='direct' only)."""
        probC = assembly_bias_direct_cen(probC, self._ab_cent_val)
        probS = assembly_bias_direct_sat(probS, self._ab_sat_val)
        return probC, probS

    def _apply_variant_ab(self, probC, probS):
        """Apply variant assembly bias (paper Eq. 12–13)."""
        probC = assembly_bias_variant_cen(probC, self._A_cent, self.fI, self._B_cent, self.fE)
        probS = assembly_bias_variant_sat(probS, self._A_sat,  self.fI, self._B_sat,  self.fE)
        return probC, probS

    def _compute_probS(self, logM, probC, has_central=None):
        """Compute satellite occupation, handling conformity.

        When conformity=True and has_central is None, falls back to the
        probabilistic expectation: E[Nsat] = probC * Nsat(has=1) + (1-probC) * Nsat(has=0).
        """
        if self.conformity:
            if has_central is not None:
                return self.HOD_satellite(logM, *self.satellite_args, has_central)
            ones  = jnp.ones_like(logM, dtype=bool)
            zeros = jnp.zeros_like(logM, dtype=bool)
            return (probC * self.HOD_satellite(logM, *self.satellite_args, ones)
                    + (1.0 - probC) * self.HOD_satellite(logM, *self.satellite_args, zeros))
        return self.HOD_satellite(logM, *self.satellite_args)

    def _compute_prob_bins(self, dict_params):
        """Return (probC, probS) evaluated on logM_bins with scalar params (no AB shift).

        Used by compute_ngal / compute_fsat / compute_Meff, where AB is zero-mean
        and must not expand scalar params into per-halo arrays.
        """
        self._set_core_params(dict_params)
        probC = self.HOD_central(self.logM_bins, *self.central_args)
        probS = self._compute_probS(self.logM_bins, probC)
        return probC, probS

    def compute_HOD_occupation(self, logM, dict_params, has_central=None):
        self.set_params(dict_params)
        probC = self.HOD_central(logM, *self.central_args)
        probS = self._compute_probS(logM, probC, has_central)
        if self.assembly_bias and self.ab_method == "direct":
            probC, probS = self._apply_direct_ab(probC, probS)
        elif self.assembly_bias and self.ab_method == "variant":
            probC, probS = self._apply_variant_ab(probC, probS)
        return probC, probS

    def compute_central_occupation(self, logM, dict_params):
        self.set_params(dict_params)
        probC = self.HOD_central(logM, *self.central_args)
        if self.assembly_bias and self.ab_method == "direct":
            probC = assembly_bias_direct_cen(probC, self._ab_cent_val)
        elif self.assembly_bias and self.ab_method == "variant":
            probC = assembly_bias_variant_cen(probC, self._A_cent, self.fI, self._B_cent, self.fE)
        return probC

    def compute_satellite_occupation(self, logM, dict_params, has_central=None):
        self.set_params(dict_params)
        if self.conformity and has_central is not None:
            probS = self.HOD_satellite(logM, *self.satellite_args, has_central)
        else:
            probS = self.HOD_satellite(logM, *self.satellite_args)
        if self.assembly_bias and self.ab_method == "direct":
            probS = assembly_bias_direct_sat(probS, self._ab_sat_val)
        elif self.assembly_bias and self.ab_method == "variant":
            probS = assembly_bias_variant_sat(probS, self._A_sat, self.fI, self._B_sat, self.fE)
        return probS
    
    def _set_core_params(self, dict_params):
        """Set central/satellite args from scalar params, skipping all AB shifts.
        Used by ngal/fsat/Meff integrals where AB is zero-mean and must not expand scalars to halo arrays."""
        self.central_args = [dict_params[key] for key in self.central_params]
        self.satellite_args = [dict_params[key] for key in self._sat_param_names]
        self.params = dict_params

    def compute_ngal(self, dict_params, Lbox=None):
        if self.logM_halos is not None:
            if Lbox is None:
                raise ValueError("Lbox required for direct per-halo ngal sum.")
            probC, probS = self._probC_probS_direct(self.logM_halos, dict_params)
            return float(jnp.sum(probC + probS)) / Lbox**3
        probC, probS = self._compute_prob_bins(dict_params)
        return compute_ngal_(self.logM_bins, self.mass_function, probC, probS)

    def compute_fsat(self, dict_params):
        if self.logM_halos is not None:
            probC, probS = self._probC_probS_direct(self.logM_halos, dict_params)
            return float(jnp.sum(probS) / jnp.sum(probC + probS))
        probC, probS = self._compute_prob_bins(dict_params)
        return compute_fsat_(self.logM_bins, self.mass_function, probC, probS)

    def compute_fsat_batched(self, params_dict_of_arrays, Ac_fiducial):
        """Vectorized f_sat over N HOD parameter rows.

        Evaluates probC/probS on `self.logM_bins` for all rows in one fused JAX
        kernel (vmap over params), then integrates with the trapezoidal rule.
        Intended for the `max_fsat` rejection filter in
        `generate_hod_parameter_grid` — orders of magnitude faster than the
        scalar `compute_fsat` loop.

        For models with conformity or assembly bias, the no-extension f_sat
        baseline is used (conformity / AB are second-order corrections to the
        integrated satellite fraction, and `max_fsat` is a soft user threshold).
        A warning is issued in those cases. The full HOD with all extensions
        is still used downstream during grid evaluation.

        Parameters
        ----------
        params_dict_of_arrays : dict[str, ndarray]
            Mapping from free param name to (N,) array of values.
        Ac_fiducial : float
            Fiducial Ac value injected into the batch (f_sat is invariant
            under (Ac, As) joint rescaling, so this matches the convention
            used in the per-row path).

        Returns
        -------
        np.ndarray, shape (N,)
            f_sat for each row.
        """
        if self.conformity or self.assembly_bias:
            warnings.warn(
                "compute_fsat_batched ignores conformity/assembly-bias "
                "corrections; valid only for the max_fsat rejection filter.",
                stacklevel=2,
            )

        if self.elg_satellite:
            sat_fn = ELG_satellite_cutoff
            sat_names = ["As", "M1", "alpha", "Mcut", "Mmax"]
        else:
            sat_fn = HOD_satellite
            sat_names = ["As", "Mmin", "M1", "alpha", "kappa"]

        n = len(next(iter(params_dict_of_arrays.values())))

        def _col(name):
            if name == "Ac":
                return jnp.full((n,), Ac_fiducial)
            if name in params_dict_of_arrays:
                return jnp.asarray(params_dict_of_arrays[name])
            raise KeyError(
                f"compute_fsat_batched: missing required column '{name}' "
                f"in params_dict_of_arrays"
            )

        central_args = [_col(k) for k in self.central_params]
        sat_args = [_col(k) for k in sat_names]

        in_axes_cen = (None,) + (0,) * len(central_args)
        in_axes_sat = (None,) + (0,) * len(sat_args)
        probC = vmap(self.HOD_central, in_axes=in_axes_cen)(self.logM_bins, *central_args)
        probS = vmap(sat_fn, in_axes=in_axes_sat)(self.logM_bins, *sat_args)

        mf = self.mass_function[None, :]
        num = jnp.trapezoid(mf * probS, self.logM_bins, axis=-1)
        den = jnp.trapezoid(mf * (probC + probS), self.logM_bins, axis=-1)
        return np.asarray(num / den)

    def compute_Meff(self, dict_params):
        if self.logM_halos is not None:
            probC, probS = self._probC_probS_direct(self.logM_halos, dict_params)
            return float(jnp.sum(probC * 10**self.logM_halos) / jnp.sum(probC + probS))
        probC, probS = self._compute_prob_bins(dict_params)
        return compute_Meff_(self.logM_bins, self.mass_function, probC, probS)

    def _probC_probS_direct(self, logM_halo, dict_params):
        """Compute per-halo (probC, probS) with all AB effects applied."""
        self.set_params(dict_params)
        probC = self.HOD_central(logM_halo, *self.central_args)
        probS = self._compute_probS(logM_halo, probC)
        if self.assembly_bias and self.ab_method == "direct":
            probC, probS = self._apply_direct_ab(probC, probS)
        elif self.assembly_bias and self.ab_method == "variant":
            probC, probS = self._apply_variant_ab(probC, probS)
        # 'mass' method: AB already baked into central_args/satellite_args by set_params
        return probC, probS


