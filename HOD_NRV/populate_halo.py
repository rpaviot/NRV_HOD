import pandas as pd
import numpy as np 
import jax.numpy as jnp
import jax.random as jrandom
#import jax.numpy as jnp
from .HOD_models import Occupation
from .NFW_jax import create_point_on_unit_sphere
from .utils import random_uniform_jax,random_poisson_jax


class halo_occupation:
    
    """ 
    This is the main class that populate dark halos of a dark matter simulation with galaxies
    """

    def __init__(self, halo_path, cosmology, zeff):

        self.halo = pd.read_parquet(halo_path)
        self.dict_cosmology = cosmology
        self.Lbox = jnp.int(jnp.round(self.halo.x.values.max() - self.halo.y.values.min()),2)
        self.zeff = zeff
        self.SpherePoints = create_point_on_unit_sphere()

    def set_halo_model(self,hod_type):

        """
        Parameters
        ----------
        hod_type: Has to be either LRG (erf), ELG_GHOD, or ELG_SFR HOD central occupation

        Returns:
        ----------
        Set the HOD_models class.
        """

        self.HOD = Occupation(hod_type,self.dict_cosmology,self.zeff)


    #def central_galaxies(self):







        


