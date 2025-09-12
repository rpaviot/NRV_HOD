 
import jax.random as jrandom
from typing import Dict, Optional, Union, Any

from .HOD_models import Occupation
from . import NFW_jax as NFWj
from .data_reader import (
    read_halo_catalog, setup_cosmology, setup_data_arrays,
    setup_particle_data_arrays, setup_triaxial_shapes, 
    setup_assembly_bias_data, apply_rsd_preprocessing, validate_rsd_axis
)
from .population_engine import populate_haloes_full
from . import test

class HaloOccupation:
    """
    Main class for populating dark matter halos with galaxies using HOD models.
    
    This class provides a complete interface for Halo Occupation Distribution (HOD)
    modeling of galaxy populations in cosmological simulations. It handles data
    loading, cosmological setup, galaxy population, and two-point function
    calculations.
    
    Parameters
    ----------
    cosmology : dict
        Cosmological parameters dictionary containing:
        - 'Om0': Matter density parameter at z=0
        - 'Ob0': Baryon density parameter at z=0
    zeff : float
        Effective redshift for calculations
    Lbox : float
        Simulation box size [Mpc/h]
    column_mapping : dict
        Mapping from standard column names to DataFrame column names
    mass_definition : str
        Halo mass definition (e.g., "M200c", "Mvir")
    DataFrame : pd.DataFrame, optional
        Pre-loaded halo catalog DataFrame
    halo_path : str, optional
        Path to halo catalog parquet file
    DataFrame_part : pd.DataFrame, optional
        Particle catalog DataFrame for lensing calculations
    assembly_bias : bool, default=False
        Whether to enable assembly bias effects
    NFW_scaled : bool, default=True
        Whether to use scaled NFW profiles
    outerprofile : bool, default=True
        Whether to use outer profile extensions
    apply_rsd : bool, default=True
        Whether to apply redshift space distortions
    triaxial_NFW : str, default='False'
        Whether to use triaxial NFW profiles
    rsd_axis : str, default='z'
        Axis along which to apply RSD ('x', 'y', or 'z')
    do_test : bool, default=True
        Whether to run validation tests on initialization
        
    Attributes
    ----------
    positions : jnp.ndarray, shape (N_halos, 3)
        Halo positions [Mpc/h]
    velocities : jnp.ndarray, shape (N_halos, 3) 
        Halo velocities [km/s]
    mass : jnp.ndarray, shape (N_halos,)
        Halo masses [Msun/h]
    logM : jnp.ndarray, shape (N_halos,)
        Log10 halo masses
    HOD : HOD_models.Occupation
        HOD model instance
    positions_gal : jnp.ndarray, shape (N_galaxies, 3)
        Galaxy positions after population [Mpc/h]
    velocities_gal : jnp.ndarray, shape (N_galaxies, 3)
        Galaxy velocities after population [km/s]
        
    Examples
    --------
    >>> # Basic usage
    >>> halo = HaloOccupation(
    ...     cosmology={'Om0': 0.3, 'Ob0': 0.049},
    ...     zeff=1.0, Lbox=1000, column_mapping=cm,
    ...     mass_definition="Mvir", DataFrame=df
    ... )
    >>> 
    >>> # Set HOD model and populate
    >>> halo.set_halo_model("LRG")
    >>> halo.populate_haloes({"Ac": 1.0, "Mmin": 13.0, "sig_M": 0.5, ...})
    >>> 
    >>> # Compute clustering
    >>> r, xi = halo.compute_galaxy_clustering('s', bins)
    
    Notes
    -----
    This class integrates data reading, cosmological calculations, galaxy
    population algorithms, and statistical measurements into a unified
    interface for HOD modeling workflows.
    
    References
    ----------
    .. [1] Berlind & Weinberg (2002), ApJ 575, 587
    .. [2] Zheng et al. (2005), ApJ 633, 791
    .. [3] Wechsler & Tinker (2018), ARA&A 56, 435
    """

    def __init__(self, cosmology: Dict[str, float], 
                 zeff: float, 
                 Lbox: float, 
                 column_mapping: Dict[str, str], 
                 mass_definition: str,
                 DataFrame: Optional[Any] = None, 
                 halo_path: Optional[str] = None, 
                 DataFrame_part: Optional[Any] = None, 
                 assembly_bias: bool = False,
                 NFW_scaled: bool = True, 
 
                 outerprofile: bool = True,
                 apply_rsd: bool = True,
                 triaxial_NFW: Union[bool, str] = False,
                 rsd_axis: str = 'z',
                 do_test: bool = True):
        
        # Store configuration parameters
        self.dict_cosmology = cosmology
        self.zeff = zeff
        self.Lbox = Lbox
        self.assembly_bias = assembly_bias
        self.mass_definition = mass_definition
        self.apply_rsd = apply_rsd
        self.triaxial_NFW = bool(triaxial_NFW) if isinstance(triaxial_NFW, str) else triaxial_NFW
        self.NFW_scaled = NFW_scaled
        self.outerprofile = outerprofile
        

        # Validate and store RSD axis
        self.rsd_axis = rsd_axis
        self.rsd_axis_index = validate_rsd_axis(rsd_axis)

        # === DATA LOADING ===
        # Read halo catalog
        self.DataFrame = read_halo_catalog(halo_path=halo_path, DataFrame=DataFrame)
        
        # Store particle data if provided
        if DataFrame_part is not None:
            self.DataFrame_part = DataFrame_part

        # === COSMOLOGICAL SETUP ===
        (self.cosmology, self.RHO_M, self.rsd_factor, 
         self.logM_bins, self.mass_function) = setup_cosmology(
            self.dict_cosmology, self.zeff, self.mass_definition
        )

        # === HALO DATA ARRAYS ===
        (self.positions, self.velocities, self.mass, self.radius,
         self.concentration, self.vrms, self.logM) = setup_data_arrays(
            self.DataFrame, column_mapping
        )
        
        # === PARTICLE DATA ARRAYS ===
        if DataFrame_part is not None:
            self.positions_part, self.velocities_part = setup_particle_data_arrays(
                self.DataFrame_part, column_mapping
            )
            
            # Apply RSD preprocessing to particles if requested
            if self.apply_rsd:
                self.positions_part = apply_rsd_preprocessing(
                    self.positions_part, self.velocities_part, self.Lbox,
                    self.rsd_factor, self.rsd_axis_index
                )
            else:
                self.positions_part = (self.positions_part + self.Lbox) % self.Lbox

        # === TRIAXIAL SHAPES ===
        if self.triaxial_NFW:
            self.shapes, self.ratios = setup_triaxial_shapes(
                self.DataFrame, column_mapping
            )
        else:
            self.shapes = None
            self.ratios = None

        # === ASSEMBLY BIAS DATA ===
        if assembly_bias:
            self.fA, self.fB = setup_assembly_bias_data(
                self.DataFrame, column_mapping
            )
        else:
            self.fA = None
            self.fB = None
        
        # Legacy attribute names for backward compatibility
        self.fI = self.fA
        self.fE = self.fB

        # Store number of halos
        self.n_halos = len(self.DataFrame)

        # Pre-generate sphere points for NFW sampling
        key = jrandom.key(0)
        self.SpherePoints = NFWj.create_point_on_unit_sphere(key)
        
        # Run validation tests if requested
        if do_test:
            test.run_all_tests()



    def set_halo_model(self, hod_type: str, conformity: bool = False):
        """
        Configure the Halo Occupation Distribution model.
        
        Parameters
        ----------
        hod_type : str
            HOD model type. Options:
            - "LRG": Error function (erf) model for Luminous Red Galaxies
            - "ELG_GHOD": Gaussian HOD model for Emission Line Galaxies  
            - "ELG_SFR": Star Formation Rate based ELG model
        conformity : bool, default=False
            Whether to use AbacusHOD-style conformity for satellites.
            When True, satellite occupation depends on actual central
            galaxy realization rather than just central probability.
            
        Examples
        --------
        >>> halo.set_halo_model("LRG")  # Standard LRG model
        >>> halo.set_halo_model("ELG_GHOD", conformity=True)  # ELG with conformity
        
        Notes
        -----
        This method must be called before `populate_haloes()` to define
        the HOD model parameters and occupation functions.
        
        The conformity parameter enables satellite occupation to depend
        on whether a central galaxy is actually realized (not just the
        probability of central occupation). This follows the AbacusHOD
        implementation with separate M1_EE parameter.
        """
        self.HOD = Occupation(
            hod_type, self.logM_bins, self.mass_function,
            assembly_bias=self.assembly_bias, conformity=conformity,
            fI=self.fI, fE=self.fE
        )






    def populate_haloes(self, dict_params: Dict[str, float], 
                       random_seed: Optional[int] = None) -> None:
        """
        Populate halos with galaxies using the configured HOD model.
        
        This method orchestrates the complete galaxy population workflow:
        1. Compute central and satellite occupation probabilities
        2. Populate central galaxies at halo centers
        3. Populate satellite galaxies using NFW density profiles  
        4. Apply redshift space distortions if requested
        5. Store results in self.positions_gal and self.velocities_gal
        
        Parameters
        ----------
        dict_params : dict
            HOD parameter dictionary. Required keys depend on the HOD model:
            
            For LRG models:
            - "Ac": Central amplitude
            - "Mmin": Minimum mass threshold [log10(Msun/h)]
            - "sig_M": Mass scatter parameter
            - "As": Satellite amplitude  
            - "M1": Satellite mass scale [log10(Msun/h)]
            - "alpha": Satellite power law index
            - "kappa": Satellite cutoff parameter
            
            Additional parameters for conformity models:
            - "M1_EE": Satellite mass scale when central present
            
            Additional parameters for assembly bias:
            - "A_cent", "B_cent": Central assembly bias coefficients
            - "A_sat", "B_sat": Satellite assembly bias coefficients
            
            Additional parameters for extended NFW profiles:
            - "f_exp": Exponential component fraction [0, 1] (default: 0.0)
            - "tau": Exponential decay scale in units of Rs (default: 6.0)
            - "lambda_NFW": NFW profile rescaling factor (default: 1.0)
        
        random_seed : int, optional
            Random seed for reproducible galaxy populations.
            If None, uses a random seed.
            
        Examples
        --------
        >>> # LRG population
        >>> params = {
        ...     "Ac": 1.0, "Mmin": 13.0, "sig_M": 0.5,
        ...     "As": 1.0, "M1": 14.0, "alpha": 1.0, "kappa": 1.0
        ... }
        >>> halo.populate_haloes(params)
        >>> 
        >>> # Access populated galaxies
        >>> galaxy_positions = halo.positions_gal
        >>> galaxy_velocities = halo.velocities_gal
        
        Notes
        -----
        After calling this method, galaxy positions and velocities are
        available in `self.positions_gal` and `self.velocities_gal`.
        
        For conformity models, satellite occupation depends on actual
        central galaxy realization rather than just central probability.
        
        Extended NFW profiles are used automatically when f_exp > 0
        or lambda_NFW ≠ 1 in the parameter dictionary, providing more
        flexible satellite positioning.
        """
        if not hasattr(self, 'HOD'):
            raise RuntimeError("HOD model not set. Call set_halo_model() first.")
            
        # Extract extended NFW parameters from dict_params (with defaults)
        f_exp = dict_params.get('f_exp', 0.0)
        tau = dict_params.get('tau', 6.0) 
        lambda_NFW = dict_params.get('lambda_NFW', 1.0)
        
        # Use the population engine for the complete workflow
        self.positions_gal, self.velocities_gal = populate_haloes_full(
            positions=self.positions,
            velocities=self.velocities, 
            mass=self.mass,
            radius=self.radius,
            concentration=self.concentration,
            vrms=self.vrms,
            logM=self.logM,
            hod_model=self.HOD,
            dict_params=dict_params,
            SpherePoints=self.SpherePoints,
            triaxial_NFW=self.triaxial_NFW,
            shapes=self.shapes,
            ratios=self.ratios,
            f_exp=f_exp,
            tau=tau,
            lambda_NFW=lambda_NFW,
            apply_rsd=self.apply_rsd,
            rsd_factor=self.rsd_factor,
            rsd_axis_index=self.rsd_axis_index,
            Lbox=self.Lbox,
            random_seed=random_seed
        )



















        


