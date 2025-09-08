import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R
import healpy as hp
import jax
from ldtk import LDPSetCreator, BoxcarFilter
from spotter import Star, show, core
from spotter.doppler import spectrum, transit_spectrum
from astropy import units as un
import jax.numpy as jnp
import copy

from jaxoplanet.orbits.keplerian import Central, Body, System
from jaxoplanet.orbits import TransitOrbit
from spotter.light_curves import transit_design_matrix, transit_light_curve
from spotter.star import Star, transited_star
from plastar import utils
from tqdm import tqdm
# from plastar.planet import Planet
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
        
        # self.star_period = ( (2 * np.pi * self.rstar * un.Rsun) / (self.v_eq * un.km / un.s) ).to(un.day)
        self.star_period = ( (2 * np.pi * self.rstar * 6.957e+8) / (self.v_eq * 1000.) ) / (24.*3600.)
        print(self.star_period)
        if include_spots_and_faculae:
            self.spots_and_faculae_dict = kwargs.pop('spots_and_faculae_dict')
            
            if self.spots_and_faculae_dict['parametrization'] == 'explicit':
                self.N_het = len(self.spots_and_faculae_dict['delta_teff']) ## Number of stellar heterogeneity features.
                self.spot_lat_array = np.array(self.spots_and_faculae_dict['lat']) * (np.pi/180.) ## in radians
                self.spot_lon_array = np.array(self.spots_and_faculae_dict['lon']) * (np.pi/180.) ## in radians
                if self.spots_and_faculae_dict['radunit'] == 'degree':
                    self.spot_radius_array = np.array(self.spots_and_faculae_dict['rad']) * (np.pi/180.) ## Convert to radians
                elif self.spots_and_faculae_dict['radunit'] == 'stellar_radius':
                    self.spot_radius_array = np.array(self.spots_and_faculae_dict['rad']) ## in radians radius will be R_spot/Rstar, but since R_spot is in Rstar units, it is just R_spot.
                    
                self.spot_sharpness_array = np.array(self.spots_and_faculae_dict['sharpness']) ## in radians
                self.spot_teff_array = np.float32(self.teff + np.array(self.spots_and_faculae_dict['delta_teff'])) ## in Kelvin
                self.spot_cb_sup_array = np.array(self.spots_and_faculae_dict['cb_sup']) ## Relative suppression in the convective blueshift of the spot as compared to the photosphere. In m/s.
                self.spot_log_g_array = np.float32(np.array(self.spots_and_faculae_dict['log_g'])) ## in cgs
                self.spot_met_array = np.float32(np.array(self.spots_and_faculae_dict['met'])) ## in cgs


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
        
        
        if self.include_planet:
            self.planet_dict = kwargs.pop('planet_dict')
            self.planet_orb_per = self.planet_dict['orbper']
            self.planet_impact_param = self.planet_dict['impact_param']
            self.planet_radius = self.planet_dict['Rp_by_Rs'] * self.star_dict['Rs'] ## convert the planet Radius in terms of solar radius.
            
            self.body = Body(time_transit=0.0, period=self.planet_orb_per, radius=self.planet_radius,
                        impact_param=self.planet_impact_param)
            self.system = System().add_body(self.body)
            
        self.star = Star.from_sides(self.nside, 
                                radius = self.rstar,
                               inc = self.star_inc,
                               period = self.star_period,
                               u=(self.LD_coeffs_star[0], self.LD_coeffs_star[1]) )
        
            
    def planet_coords(self, time):
        """Calculate the XYZ position of the planet coordinates for each time stamp.

        :param time: _description_
        :type time: _type_
        :return: _description_
        :rtype: _type_
        """
        xos, yos, zos = self.system.relative_position(time)
        x = xos[0] / self.system.central.radius
        y = yos[0] / self.system.central.radius
        z = zos[0] / self.system.central.radius
        return x, y, z
    
    def flux_model(self, star, time):
        
        
        coords = self.planet_coords(time)
        
        # flux = jax.vmap(
        #     lambda coords, time: transit_light_curve(star, *coords, r=self.planet_radius, 
        #                                              time=time, normalize = False)
        # )(coords, time).T[0]
        flux = jax.vmap(
            lambda coords, time: transit_light_curve(star, *coords, r=self.planet_radius, 
                                                     time=time, normalize = False)
        )(coords, time).T
        
        return flux
    
    
    def plot_star_grid(self, star_grid = None):
        """
        Plot the stellar grid based on the input parameters.""" 
        plt.figure(figsize=(3, 3))
        show(star_grid)
        plt.show()
            
    def set_spectral_values(self, stellar_spectrum = None, spot_spectra = None, wavelength_solution = None, include_spots_and_faculae = False):
        
        if include_spots_and_faculae:
            # assert(spot_spectra.shape == (self.N_het, len(self.wavsoln)))
            
            # Define all the spot objects first and create a list of them 
            spots = []
            base_star = self.star
            for ihet in range(self.N_het):
                spot = core.spot(self.star.sides, self.spot_lat_array[ihet], self.spot_lon_array[ihet], self.spot_radius_array[ihet], sharpness = 20)#  sharpness = self.spot_sharpness_array[ihet] ) 
                spots.append(spot)

            
            stellar_spectrum = jnp.array(stellar_spectrum)
            spot_spectra = jnp.array(spot_spectra)
            
            # Set the base_star with the base stellar spectrum
            spectra = base_star.y * stellar_spectrum[:, None]
                        
            # Compute the spectrum to be assigned for the points where there are spots. Subtract the stellar spectrum for these points as that has already been assigned to these points before. 
            for isp, sp in enumerate(spots):
                spectra = spectra + sp[None,:] * (spot_spectra[isp][:, None] - stellar_spectrum[:, None])
            
            
            star = base_star.set(y = spectra, wv = 1e-9 * wavelength_solution)
            
        else:
            base_star = self.star
            spectra = base_star.y * stellar_spectrum[:, None]
            star = base_star.set(y = spectra, wv = 1e-9 * wavelength_solution)
        
        return star
    
    def get_spectral_time_series(self, time=None, stellar_spectrum = None, spot_spectra = None, include_spots_and_faculae = False, wavelength_chunk_length = None, wavelength_overlap_length = None):
        """
        Generate a spectral time series for the star.
        
        Parameters:
        - star: The star object.
        - t: Time array.
        - include_spots_and_faculae: Boolean indicating whether to include spots and faculae.
        
        Returns:
        - spectrum: The spectral time series.
        """
        
        
        if wavelength_chunk_length is None: # Could also consider chunking in time.
            star = self.set_spectral_values(stellar_spectrum = stellar_spectrum, spot_spectra = spot_spectra, 
                                            wavelength_solution = self.wavsoln,
                                            include_spots_and_faculae = include_spots_and_faculae)
            if self.include_planet:
                x, y, z = self.planet_coords(time)
                return star, transit_spectrum(star, time, x, y, z, self.planet_radius, normalize = False), self.wavsoln
            else:
                return star, spectrum(star, time, normalize = False), self.wavsoln
        else:
            chunked_stellar_spectrum = utils.create_overlapping_chunks(stellar_spectrum, wavelength_chunk_length, wavelength_overlap_length)
            chunked_wavelength_solution = utils.create_overlapping_chunks(self.wavsoln, wavelength_chunk_length, wavelength_overlap_length)
            
            chunked_spot_spectra = []
            for isp in range(spot_spectra.shape[0]):
                chunked_spot_spectrum = utils.create_overlapping_chunks(spot_spectra[isp,:], wavelength_chunk_length, wavelength_overlap_length)
                chunked_spot_spectra.append(chunked_spot_spectrum)
            chunked_spot_spectra = np.array(chunked_spot_spectra)

            ### Calculate the length of the merged integrated spectrum, may not be the same as the original wavelength solution because of chunks being all the same length.
            step = wavelength_chunk_length - wavelength_overlap_length
            merged_length = (chunked_stellar_spectrum.shape[0] - 1) * step + wavelength_chunk_length
            
            
            if self.include_planet:
                x, y, z = self.planet_coords(time)
                integrated_spectrum = np.zeros((len(time), chunked_stellar_spectrum.shape[0], chunked_stellar_spectrum.shape[1]))
                merged_integrated_spectrum = np.zeros((len(time), merged_length))
                
            for ichunk in tqdm(range(chunked_stellar_spectrum.shape[0])):
                star = self.set_spectral_values(stellar_spectrum = chunked_stellar_spectrum[ichunk,:], 
                                                spot_spectra = chunked_spot_spectra[:,ichunk,:], 
                                                wavelength_solution = chunked_wavelength_solution[ichunk,:],
                                                include_spots_and_faculae = include_spots_and_faculae)
                if self.include_planet:
                    integrated_spectrum[:,ichunk,:] = transit_spectrum(star, time, x, y, z, self.planet_radius, normalize = False)
                else:
                    integrated_spectrum[ichunk,:] = spectrum(star, time, normalize = False)

            ## Merge the chunks back, accounting for the overlap
            merged_wavelength_solution = utils.merge_chunks_back(chunked_wavelength_solution, wavelength_chunk_length, wavelength_overlap_length)
            
            if self.include_planet:
                for itime in range(len(time)):
                    merged_integrated_spectrum[itime,:] = utils.merge_chunks_back(integrated_spectrum[itime,:,:], wavelength_chunk_length, wavelength_overlap_length)
            else:
                merged_integrated_spectrum = utils.merge_chunks_back(integrated_spectrum, wavelength_chunk_length, wavelength_overlap_length)

            return star, merged_integrated_spectrum, merged_wavelength_solution
            

                
                
