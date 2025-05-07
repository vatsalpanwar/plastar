import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import yaml
import sys
import jax.numpy as jnp
sys.path.append('/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/plastar/')
from plastar import grid
from astropy.io import fits
from spotter import show

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

################################################################
################################################################
now = datetime.datetime.now()
# Format the date and time
d1 = now.strftime("%d-%m-%YT%H-%M-%S")
print('Date tag for this run which will be used to save the results: ', d1)

################################################################
################################################################
"""Read in the config file."""
################################################################
################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=False)

args = vars(parser.parse_args())

config_file_path = args['config_file_path']
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)
infostring = config_dd['infostring'] + d1
savedir = config_dd['simulations_savedir'] + infostring + '/'

star_dict = config_dd['star']
spots_and_faculae_dict = config_dd['spots_and_faculae']

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

################################################################
################################################################
"""Read in the PHOENIX model and splice and convolve to instrument resolution."""
################################################################
################################################################

R = config_dd['model_resolution_working']
lam_min, lam_max = config_dd['instrument']['wavelength_min'], config_dd['instrument']['wavelength_max']
if config_dd['instrument']['wavelength_unit'] == 'Angstrom':
    lam_min, lam_max = lam_min*1.0e-10, lam_max*1.0e-10
elif config_dd['instrument']['wavelength_unit'] == 'nm':
    lam_min, lam_max = lam_min*1.0e-9, lam_max*1.0e-9
    
num_vals = R*(np.log(lam_max) - np.log(lam_min))
lam = np.logspace(np.log(lam_min),np.log(lam_max),num = int(num_vals)+1,endpoint=True,base=np.e) # constant resolution, so need evenly spaced in ln(lambda)

lam_nm = lam*1e9

"""Load the PHOENIX models for the star."""
star_teff, star_log_g, star_met = star_dict['teff'], star_dict['log_g'], star_dict['met']
flux_model_star_path = config_dd['phoenix_model_dir'] + f'lte0{star_teff}-{star_log_g}-{star_met}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
wavsoln_model_star_path = config_dd['phoenix_model_dir'] + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

wavsoln_model_star = fits.getdata(wavsoln_model_star_path)/10.
flux_model_star = fits.getdata(flux_model_star_path)* 1e-7 * 1e4 * 1e2/np.pi

"""Load the PHOENIX models for the spots and faculae."""
wavsoln_model_spot_path = config_dd['phoenix_model_dir'] + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
flux_model_spot_path_list = []
for isp in range(len(spots_and_faculae_dict['teff'])):
    spf_teff, spf_log_g, spf_met = spots_and_faculae_dict['teff'][isp], spots_and_faculae_dict['log_g'][isp], spots_and_faculae_dict['met'][isp]
    flux_model_spot_path = config_dd['phoenix_model_dir'] + f'lte0{spf_teff}-{spf_log_g}-{spf_met}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    flux_model_spot_path_list.append(flux_model_spot_path)

wavsoln_model_spot = fits.getdata(wavsoln_model_spot_path)/10.
flux_model_spot = []
for isp in range(len(config_dd['spots_and_faculae']['teff'])):
    flux_model_spot.append(fits.getdata(flux_model_spot_path_list[isp])* 1e-7 * 1e4 * 1e2/np.pi)
flux_model_spot = np.array(flux_model_spot)
 
wav_inds_star = [ np.argmin(abs(wavsoln_model_star-lam_nm[0])) , np.argmin(abs(wavsoln_model_star-lam_nm[-1])) ]
wav_inds_spot = [ np.argmin(abs(wavsoln_model_spot-lam_nm[0])) , np.argmin(abs(wavsoln_model_spot-lam_nm[-1])) ]

"""Slice the model to this range, a bit wider to allow interpolation and Doppler shifts"""
flux_model_star = flux_model_star[wav_inds_star[0]:wav_inds_star[1]]
wavsoln_model_star = wavsoln_model_star[wav_inds_star[0]:wav_inds_star[1]]

flux_model_spot = flux_model_spot[:,wav_inds_spot[0]:wav_inds_spot[1]]
wavsoln_model_spot = wavsoln_model_spot[wav_inds_spot[0]:wav_inds_spot[1]]


star_grid = grid.StellarGrid(star_dict = star_dict, spots_and_faculae_dict = spots_and_faculae_dict,
                        wavsoln = wavsoln_model_star,
                        include_spots_and_faculae = True)


# star = star_grid.set_spectral_values(stellar_spectrum = flux_model_star,
#                                      spot_spectra = flux_model_spot,
#                                     # spot_spectra = np.array([flux_slice_spot]),
#                                      include_spots_and_faculae = True)
# star_grid.plot_star_grid(star)
n = 3
fig, axes = plt.subplots(2, n, figsize=(8, 2.5))
phases = jnp.linspace(jnp.pi / 2, -jnp.pi / 2, n)
time_0 = phases[0] * star_grid.star.period / (2 * jnp.pi)
_, nonspotted_spectrum = star_grid.get_spectral_time_series(time=time_0, stellar_spectrum = flux_model_star, 
                                                         spot_spectra = flux_model_spot, include_spots_and_faculae = False)


for i, phase in enumerate(phases):
    ax = axes[1, i]
    time = phase * star_grid.star.period / (2 * jnp.pi)
    
    star, spotty_spectrum = star_grid.get_spectral_time_series(time=time, stellar_spectrum = flux_model_star, 
                                                         spot_spectra = flux_model_spot, include_spots_and_faculae = True)
    
    ax.plot(spotty_spectrum, c="k", lw=1, label="spotted")
    ax.plot(nonspotted_spectrum, "-", c="r", lw=1, label="non-spotted")
    ax.axis("off")

    ax = axes[0, i]
    show(star, phase, ax=ax)

    if i == n - 1:
        plt.legend()
plt.show()