import pandas as pd
import numpy as np 
import jax.numpy as jnp
import jax.random as jrandom
#import jax.numpy as jnp
import gc
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from .HOD_models import Occupation
from .import NFW_jax as NFWj
from .utils import *
from .two_point import *
from . import test

class HaloOccupation:
    
    """ 
    This is the main class that populate dark halos of a dark matter simulation with galaxies
    """

    def __init__(self, cosmology, zeff, Lbox, column_mapping, mass_definition,
                 DataFrame=None, halo_path=None, DataFrame_part=None, assembly_bias=False,\
                    NFW_scaled=True, conformity=True, outerprofile=True,apply_rsd=True,triaxial_NFW='False',rsd_axis='z',do_test=True,
                    f_exp=0.0, tau=6.0, lambda_NFW=1.0):

        self.dict_cosmology = cosmology
        self.zeff = zeff
        self.Lbox = Lbox
        self.assembly_bias = assembly_bias
        self.mass_definition = mass_definition  # e.g., "M200c", "Mvir"
        self.apply_rsd=apply_rsd
        self.set_cosmology(self.dict_cosmology)
        self.triaxial_NFW = triaxial_NFW
        self.NFW_scaled = NFW_scaled
        self.outerprofile = outerprofile
        
        # Extended NFW parameters
        self.f_exp = f_exp
        self.tau = tau
        self.lambda_NFW = lambda_NFW

        self.rsd_axis=rsd_axis
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if self.rsd_axis not in axis_map:
            raise ValueError("rsd_axis must be one of 'x', 'y', or 'z'")
        self.rsd_axis_index = axis_map[self.rsd_axis]


        if halo_path:
            self.DataFrame = pd.read_parquet(halo_path) 
        elif DataFrame is not None:
            self.DataFrame = DataFrame
        else:
            raise ValueError("You must provide either a DataFrame or a halo_path.")

        if DataFrame_part is not None:
            self.DataFrame_part = DataFrame_part


        df = self.DataFrame
        df_part = self.DataFrame_part
        cm = column_mapping  # shorthand

        # === Required Fields ===
        self.positions = jnp.stack([
            jnp.array(df[cm['x']]),
            jnp.array(df[cm['y']]),
            jnp.array(df[cm['z']])
        ], axis=-1)

        self.positions_part = jnp.stack([
            jnp.array(df_part[cm['x']]),
            jnp.array(df_part[cm['y']]),
            jnp.array(df_part[cm['z']])
        ], axis=-1)


        self.velocities = jnp.stack([
            jnp.array(df[cm['vx']]),
            jnp.array(df[cm['vy']]),
            jnp.array(df[cm['vz']])
        ], axis=-1)

        self.velocities_part = jnp.stack([
            jnp.array(df_part[cm['vx']]),
            jnp.array(df_part[cm['vy']]),
            jnp.array(df_part[cm['vz']])
        ], axis=-1)
        

        if self.apply_rsd is True:
            self.positions_part = self.apply_rsd_to_array(self.positions_part,self.velocities_part)
        else:
            self.positions_part = (self.positions_part + self.Lbox) % self.Lbox


        if self.triaxial_NFW is True:
            a = jnp.stack([jnp.array(df[cm['a_x']]),
                           jnp.array(df[cm['a_y']]),
                           jnp.array(df[cm['a_z']])
                           ],axis=-1)
            
            b = jnp.stack([jnp.array(df[cm['b_x']]),
                           jnp.array(df[cm['b_y']]),
                           jnp.array(df[cm['b_z']])
                           ],axis=-1)
            
            a = a/ jnp.linalg.norm(a,axis=-1)[:,None]
            b = b/ jnp.linalg.norm(b,axis=-1)[:,None]
            c = jnp.cross(a,b)

            self.shapes = jnp.stack([a,b,c],axis=-1)
            self.ratios = jnp.stack([jnp.array(df[cm['b_to_a']]),
                                     jnp.array(df[cm['c_to_a']]),
                                     ],axis=-1)

        self.mass = jnp.array(df[cm['mass']])
        self.radius = jnp.array(df[cm['radius']])
        self.concentration = jnp.array(df[cm['c']])
        self.vrms = jnp.array(df[cm['vrms']])
        self.logM = jnp.log10(self.mass)

        self.fI = None
        self.fE = None

        # === Optional fields for assembly bias ===

        if assembly_bias:
            if 'fI' not in cm and 'fE' not in cm:
                raise AttributeError("Assembly bias is enabled but 'fI' AND 'fE' are missing in column mapping.")
            self.fA = jnp.array(df[cm['fI']]) if 'fI' in cm else None
            self.fB = jnp.array(df[cm['fE']]) if 'fE' in cm else None

        self.n_halos = len(df)

        key = jrandom.key(0)
        self.SpherePoints = NFWj.create_point_on_unit_sphere(key)
        if do_test is True:
            test.run_all_tests()


    def set_cosmology(self,dict_cosmology):

        self.cosmology = FlatLambdaCDM(Om0=self.dict_cosmology['Om0'],Ob0=self.dict_cosmology['Ob0'],H0=100)
        self.RHO_M = ((self.cosmology.critical_density0*self.cosmology.Om0).to(u.Msun/u.Mpc**3)/self.cosmology.h**2).value # Msun/h/(Mpc/h)**3
        Hz = self.cosmology.H(self.zeff)

        a = 1. / (1 + self.zeff)
        rsd_factor = 1. / (Hz * a).value 
        self.rsd_factor = rsd_factor

        self.logM_bins = np.log10(np.logspace(11, 15, 1024))
        self.mass_function = set_mass_function(self.dict_cosmology,self.logM_bins, z=self.zeff,mass_definition=self.mass_definition)

    def set_halo_model(self,hod_type,conformity=False):

        """
        Parameters
        ----------
        hod_type: Has to be either LRG (erf), ELG_GHOD, or ELG_SFR HOD central occupation
        conformity: bool, whether to use AbacusHOD-style conformity for satellites

        Returns:
        ----------
        Set the HOD_models class.
        """

        self.HOD = Occupation(hod_type,self.logM_bins,self.mass_function,assembly_bias=self.assembly_bias,conformity=conformity,fI=self.fI,fE=self.fE)

    def apply_rsd_to_array(self, positions, velocities):
        """
        Apply RSD along self.rsd_axis_index to positions and velocities arrays.

        Parameters
        ----------
        positions : jnp.ndarray of shape (N, 3)
        velocities : jnp.ndarray of shape (N, 3)

        Returns
        -------
        rsd_positions : jnp.ndarray of shape (N, 3)
        """
        axis = self.rsd_axis_index
        # Shift along RSD axis
        shift = velocities[:, axis] * self.rsd_factor
        rsd_positions = positions.at[:, axis].add(shift)

        # Wrap to box
        rsd_positions = (rsd_positions + self.Lbox) % self.Lbox
        return rsd_positions


    def populate_centrals(self,key,probC):

        rand_uniform = random_uniform_jax(key,self.n_halos)
        is_cent = probC > rand_uniform
        self.is_cent = is_cent  # Store for conformity use
        return self.positions[is_cent],self.velocities[is_cent]


    def populate_satellites(self,key_s, key_s_pos, key_s_vel, probS):

        pre_cond = probS > 0
        probS = probS[pre_cond]

        N_s = random_poisson_jax(key_s,probS)
        has_sat = N_s > 0 

        cent_positions = self.positions[pre_cond][has_sat]
        cent_velocities = self.velocities[pre_cond][has_sat]
        vrms = self.vrms[pre_cond][has_sat]
        radius = self.radius[pre_cond][has_sat]
        concentration = self.concentration[pre_cond][has_sat]
        shapes_for_sat = self.shapes[pre_cond][has_sat]
        ratios_for_sat = self.ratios[pre_cond][has_sat]

        N_s = N_s[has_sat]
        N_s_tot = int(jnp.sum(N_s))

        # Check if extended NFW profile should be used
        use_extended_NFW = (self.f_exp > 0.0 or self.lambda_NFW != 1.0)
        
        if use_extended_NFW:
            if self.triaxial_NFW is True:
                sat_positions = NFWj.extended_elliptical_NFW_satellites_positions(key_s_pos,self.SpherePoints,cent_positions,radius,concentration,shapes_for_sat,ratios_for_sat,N_s,N_s_tot,f_exp=self.f_exp,tau=self.tau,lambda_NFW=self.lambda_NFW)
            else:
                sat_positions = NFWj.extended_NFW_satellites_positions(key_s_pos,self.SpherePoints,cent_positions,radius,concentration,N_s,N_s_tot,f_exp=self.f_exp,tau=self.tau,lambda_NFW=self.lambda_NFW)
        else:
            # Use standard NFW profiles
            if self.triaxial_NFW is True:
                sat_positions = NFWj.elliptical_NFW_satellites_positions(key_s_pos,self.SpherePoints,cent_positions,radius,concentration,shapes_for_sat,ratios_for_sat,N_s,N_s_tot)
            else:
                sat_positions = NFWj.spherical_NFW_satellites_positions(key_s_pos,self.SpherePoints,cent_positions,radius,concentration,N_s,N_s_tot)

        sat_velocities = NFWj.dispersion_velocities_satellites(key_s_vel,cent_velocities,vrms,N_s,N_s_tot)

        return sat_positions,sat_velocities

    def populate_haloes(self,dict_params):
        
        key = jrandom.key(np.random.randint(low=1,high=int(1e18)))
        key_c, key_s,key_s_pos,key_s_vel, = jrandom.split(key,num=4)
        
        # Compute central occupation probability and populate centrals
        probC = self.HOD.compute_central_occupation(self.logM,dict_params)
        self.cent_positions,self.cent_velocities = self.populate_centrals(key_c,probC)
        
        # For conformity, compute satellite occupation using actual central realization
        if self.HOD.conformity:
            probS = self.HOD.compute_satellite_occupation(self.logM,dict_params,has_central=self.is_cent)
        else:
            probS = self.HOD.compute_satellite_occupation(self.logM,dict_params)

        self.sat_positions, self.sat_velocities = self.populate_satellites(key_s,key_s_pos,key_s_vel,probS)

        self.positions_gal = jnp.vstack([self.cent_positions,self.sat_positions])
        self.velocities_gal = jnp.vstack([self.cent_velocities,self.sat_velocities])

        if self.apply_rsd is True:
            self.positions_gal = self.apply_rsd_to_array(self.positions_gal,self.velocities_gal)
        else:
            self.positions_gal = (self.positions_gal + self.Lbox) % self.Lbox


    def compute_galaxy_clustering(self,mode,bins1,catalog2=None, bins2=None,output='auto'):
        r, xi = compute_corr(mode,self.positions_gal,bins1,catalog2=catalog2, bins2=bins2,boxsize=self.Lbox,los=self.rsd_axis,output=output)
        return r,xi

    def compute_galaxy_lensing(self,bins1,output='xi',bins2=None,bins_comp=np.geomspace(5e-3,100,81)):

        rp_centers =  np.sqrt(bins1[:-1] * bins1[1:])
        rr, xi_gm = compute_corr('s',self.positions_gal,bins_comp,catalog2=self.positions_part, bins2=bins2,boxsize=self.Lbox,los=self.rsd_axis,output=output)
        DeltaSigma_pipeline =  DeltaSigmaCalculator(rr,xi_gm,self.RHO_M)
        DeltaSigma = DeltaSigma_pipeline.compute_deltasigma_averaged(bins1)
        return rp_centers,DeltaSigma
















        


