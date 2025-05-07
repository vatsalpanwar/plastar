import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R
import healpy as hp
import jax
from ldtk import LDPSetCreator, BoxcarFilter
from spotter import Star, show, core
from spotter.doppler import spectrum
from astropy import units as un
import jax.numpy as jnp
import copy
# from healpy.rotator import rotateVector

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
        include_spots_and_faculae = kwargs.pop('include_spots_and_faculae', False)
        self.include_spots_and_faculae = include_spots_and_faculae
        
        self.star_inc = self.star_dict['inc'] * np.pi / 180.0
        self.nside = self.star_dict['nside']
        self.wavsoln = kwargs.pop('wavsoln')
        self.rstar = self.star_dict['Rs'] ## in solar radius 
        self.teff = np.float32(self.star_dict['teff'])  # in Kelvin.
        self.v_eq = self.star_dict['v_eq'] # Equatorial velocity, in km/s 
        self.log_g = np.float32(self.star_dict['log_g']) # Equatorial velocity, in km/s 
        self.met = np.float32(self.star_dict['met']) # Equatorial velocity, in km/s 
        
        
        # self.inc_rot_mat = R.from_rotvec([self.star_inc + np.pi/2, 0, 0]).as_matrix()
        # self.inst_dict = kwargs.pop('inst_dict')
        
        if self.star_dict['LD'] == 'quadratic_phoenix':
            filters = [BoxcarFilter('a', self.wavsoln[0], self.wavsoln[-1])]  # In nm

            sc_star = LDPSetCreator(teff=(self.teff,   50),    # Define your star, and the code
                            logg=( self.log_g, 0.20),    # downloads the uncached stellar
                                z=(self.met, 0.05),    # spectra from the Husser et al.
                                filters=filters)    # FTP server automatically.

            ps = sc_star.create_profiles()                # Create the limb darkening profiles
            self.cq_star,self.eq_star = ps.coeffs_qd(do_mc=False)         # Estimate quadratic law coefficients
            self.LD_coeffs_star = np.array( [self.cq_star[0][0], self.cq_star[0][1]] )
        
        self.star_period = ( (2 * np.pi * self.rstar * un.Rsun) / (self.v_eq * un.km / un.s) ).to(un.day)
        
        if include_spots_and_faculae:
            self.spots_and_faculae_dict = kwargs.pop('spots_and_faculae_dict')
            # lat: float, lon: float, radius: float, sharpness: float = 20
            self.N_het = len(self.spots_and_faculae_dict['teff']) ## Number of stellar heterogeneity features.
            self.spot_lat_array = np.array(self.spots_and_faculae_dict['lat']) * (np.pi/180.) ## in radians
            self.spot_lon_array = np.array(self.spots_and_faculae_dict['lon']) * (np.pi/180.) ## in radians
            self.spot_radius_array = np.array(self.spots_and_faculae_dict['rad']) * (np.pi/180.) ## in radians
            self.spot_sharpness_array = np.array(self.spots_and_faculae_dict['sharpness']) ## in radians
            
            self.spot_teff_array = np.float32(np.array(self.spots_and_faculae_dict['teff'])) ## in Kelvin
            self.spot_cb_sup_array = np.array(self.spots_and_faculae_dict['cb_sup']) ## Relative suppression in the convective blueshift of the spot as compared to the photosphere. In m/s.
            self.spot_log_g_array = np.float32(np.array(self.spots_and_faculae_dict['log_g'])) ## in cgs
            self.spot_met_array = np.float32(np.array(self.spots_and_faculae_dict['met'])) ## in cgs

            # self.sp_contrast = self.spot_dict['sp_contrast']
            # self.sp_contrast_for_plot = self.spot_dict['sp_contrast_for_plot']
            # self.sp_long0 = self.spot_dict['sp_long0']* (np.pi/180.) ## in radians
            # self.sp_lat = self.spot_dict['sp_lat'] * (np.pi/180.)
            # self.sp_rad = self.spot_dict['sp_rad']
                        
            # self.sp_teff = self.spot_dict['sp_teff']
            # self.sp_cb_sup = self.spot_dict['sp_cb_sup'] ## Relative suppression in the convective blueshift of the spot as compared to the photosphere. In m/s.
            
            # ####### Fix spot log_g and metallicity to the stellar values for now, so the only difference in the limb darkening comes from the difference in Teff 
            # self.sp_log_g = self.log_g
            # self.sp_met = self.met
            self.LD_coeffs_spot = []
            
            for ihet in range(self.N_het):
                if self.spots_and_faculae_dict['LD'] == 'quadratic_phoenix':
                    filters = [BoxcarFilter('a', self.wavsoln[0], self.wavsoln[-1])]  # In nm
                    sc_spot = LDPSetCreator(teff=(self.spot_teff_array[ihet],   50),    # Define your star, and the code
                        logg=(self.spot_log_g_array[ihet], 0.20),    # downloads the uncached stellar
                            z=(self.spot_met_array[ihet], 0.05),    # spectra from the Husser et al.
                            filters=filters)    # FTP server automatically.

                    ps = sc_spot.create_profiles()                # Create the limb darkening profiles
                    self.cq_spot,self.eq_spot = ps.coeffs_qd(do_mc=False)         # Estimate quadratic law coefficients
                    self.LD_coeffs_spot.append( np.array( [self.cq_spot[0][0], self.cq_spot[0][1]] ) )
                elif self.spots_and_faculae_dict['LD'] == 'quadratic_self':
                    self.cq_spot = [[ self.spots_and_faculae_dict['u1'][ihet], self.spots_and_faculae_dict['u2'][ihet] ]]
                    self.LD_coeffs_spot.append( np.array( [self.cq_spot[0], self.cq_spot[1]] ) )
        
        

        self.star = Star.from_sides(self.nside, 
                               inc = self.star_inc,
                               period = self.star_period.value,
                               u=(self.LD_coeffs_star[0], self.LD_coeffs_star[1]) )
        
        
    def plot_star_grid(self, star_grid = None):
        """
        Plot the stellar grid based on the input parameters.""" 
        plt.figure(figsize=(3, 3))
        show(star_grid)
        plt.show()
            
    def set_spectral_values(self, stellar_spectrum = None, spot_spectra = None, include_spots_and_faculae = False):
        
        # if includes_spots_and_faculae:
        #     assert(spot_spectra.shape == (self.N_het, len(self.wavsoln)))
            
        #     spots = []
        #     base_star = self.star
        #     for ihet in range(self.N_het):
        #         spot = core.spot(self.star.sides, self.spot_lat_array[ihet], self.spot_lon_array[ihet], self.spot_radius_array[ihet], sharpness = 20 ) 
        #         spots.append(spot)

            
        #     stellar_spectrum = jnp.array(stellar_spectrum)
        #     spot_spectra = jnp.array(spot_spectra)
        #     ##### Set the spectrum of the unspotted part of the stellar spectrum 
        #     spectra = (base_star.y - spots[0]) * stellar_spectrum[:, None]
        #     spectra = spectra + spots[0][None,:] * spot_spectra[0][:, None]
            
        #     # for ihet in range(self.N_het):
        #     #     spectra = spectra + spots[ihet][None,:] * spot_spectra[ihet][:, None]
            
        #     star = base_star.set(y = spectra, wv = 1e-9 * self.wavsoln)
        # return star
        if include_spots_and_faculae:
            assert(spot_spectra.shape == (self.N_het, len(self.wavsoln)))
            
            spots = []
            base_star = self.star
            for ihet in range(self.N_het):
                spot = core.spot(self.star.sides, self.spot_lat_array[ihet], self.spot_lon_array[ihet], self.spot_radius_array[ihet], sharpness = 20)#  sharpness = self.spot_sharpness_array[ihet] ) 
                spots.append(spot)

            
            
            stellar_spectrum = jnp.array(stellar_spectrum)
            spot_spectra = jnp.array(spot_spectra)
            
            # for isp, sp in enumerate(spots):
                # base_star.set(y = base_star.y - sp)
            
            # # import pdb; pdb.set_trace()
            
            # ##### Set the spectrum of the unspotted part of the stellar spectrum 
            # base_star_y_copy = copy.deepcopy(base_star.y)
            # for isp, sp in enumerate(spots):
            #     base_star_y_copy = base_star_y_copy - sp
            # import pdb; pdb.set_trace()
            
            # base_star.set(y = base_star_y_copy)
            
            spectra = base_star.y * stellar_spectrum[:, None]
            # star = base_star.set(y = spectra, wv = 1e-9 * self.wavsoln)
            
            for isp, sp in enumerate(spots):
                spectra = spectra + sp[None,:] * (spot_spectra[isp][:, None] - stellar_spectrum[:, None])
            
            # # # for ihet in range(self.N_het):
            # # #     spectra = spectra + spots[ihet][None,:] * spot_spectra[ihet][:, None]
            
            star = base_star.set(y = spectra, wv = 1e-9 * self.wavsoln)
        else:
            base_star = self.star
            spectra = base_star.y * stellar_spectrum[:, None]
            star = base_star.set(y = spectra, wv = 1e-9 * self.wavsoln)
        return star
    
    def get_spectral_time_series(self, time=None, stellar_spectrum = None, spot_spectra = None, include_spots_and_faculae = False):
        """
        Generate a spectral time series for the star.
        
        Parameters:
        - star: The star object.
        - t: Time array.
        - include_spots_and_faculae: Boolean indicating whether to include spots and faculae.
        
        Returns:
        - spectrum: The spectral time series.
        """
        star = self.set_spectral_values(stellar_spectrum = stellar_spectrum, spot_spectra = spot_spectra, include_spots_and_faculae = include_spots_and_faculae)
        return star, spectrum(star, time)