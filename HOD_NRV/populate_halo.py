import pandas as pd
import numpy as np 
import jax.numpy as jnp
import jax.random as jrandom
#import jax.numpy as jnp
from .HOD_models import Occupation
from .NFW_jax import create_point_on_unit_sphere
from .utils import *
import gc

class halo_occupation:
    
    """ 
    This is the main class that populate dark halos of a dark matter simulation with galaxies
    """

    def __init__(self, halo_path, cosmology, zeff,Lbox):
        
        df = pd.read_parquet(halo_path)
        self.halo = JaxDataSet(df)

        self.dict_cosmology = cosmology
        self.Lbox = Lbox
        self.zeff = zeff

        key = jrandom.key(0)
        self.SpherePoints = create_point_on_unit_sphere(key)
        self.n_halos = len(self.halo_catalogue)
        self.mass_function = set_mass_function(self.dict_cosmology,self.logM_bins,z=zeff)

        del df
        gc.collect()
        

    def set_halo_model(self,hod_type):

        """
        Parameters
        ----------
        hod_type: Has to be either LRG (erf), ELG_GHOD, or ELG_SFR HOD central occupation

        Returns:
        ----------
        Set the HOD_models class.
        """

        self.HOD = Occupation(hod_type)


    def populate_galaxies(self,dict_params):
        
        key = jrandom.key(np.random.uniform(0,int(1e32)))
        key_c, key_s = jrandom.split(key=2)
        probC,probS = self.HOD.compute_HOD_occupation(self.halo_catalogue.logM,dict_params)

        rand_uniform = random_uniform_jax(key_c,self.n_halos)
        rand_poisson = random_poisson_jax(key_s,probS)
        
        halo_cent = self.halo[probC > rand_uniform]
        halo_sat = self.halo[rand_poisson > 0]















        


