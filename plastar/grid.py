import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R
import healpy as hp
from healpy.rotator import rotateVector

class StellarGrid:
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the StellarGrid class with the provided arguments.
        
        Parameters:
        - *args: Positional arguments.
        - **kwargs: Keyword arguments.
        """
        self.star_dict = kwargs.pop('star_dict')
        include_planet = kwargs.pop('include_planet', False)
        self.include_planet = include_planet
        include_spot = kwargs.pop('include_spot', False)
        self.include_spot = include_spot
        
        self.star_inc = self.star_dict['inc'] * np.pi / 180.0
        self.inc_rot_mat = R.from_rotvec([self.star_inc + np.pi/2, 0, 0]).as_matrix()
        
        
    def construct_grid(self):
        
        NSIDE = self.star_dict['nside']
        
        NPIX = hp.nside2npix(NSIDE)
        print(NPIX)
        # m = np.arange(NPIX) ## Values of the pixels
        # hp.mollview(m, title="Mollview image RING")
        # hp.graticule()
        # plt.show()
        
        m = np.zeros(NPIX) ## Values of the pixels
        #### Create the limb darkened map:
        for ipix in range(NPIX):
            theta, phi = hp.pix2ang(NSIDE, ipix)
            # print(theta, phi)
            m[ipix] = 1. - np.cos(phi)
        
        ### Mark some spots 
        vec1 = hp.ang2vec(np.pi / 2, np.pi * 3 / 4) ### theta, phi
        vec2 = hp.ang2vec(np.pi/3, np.pi * 3 / 4) ### theta, phi
        
        ## Rotate them according to the inclination of the star 
        vec1 = rotateVector(self.inc_rot_mat, vec1)
        vec2 = rotateVector(self.inc_rot_mat, vec2)
        
        ipix_disc1 = hp.query_disc(nside=NSIDE, vec=vec1, radius=np.radians(10))
        m[ipix_disc1] = m.max()
        ipix_disc2 = hp.query_disc(nside=NSIDE, vec=vec2, radius=np.radians(5))
        m[ipix_disc2] = m.min()
        
        # hp.mollview(m, title="Mollview image RING")
        hp.orthview(m, title="Orthview image RING")
        hp.graticule()
        plt.show()
        