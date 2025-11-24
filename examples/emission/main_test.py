import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import yaml
import jax.numpy as jnp
from plastar import grid
from plastar import utils
from plastar import planet
from plastar import ccf
from astropy.io import fits
from spotter import show, viz, core
# from jax.numpy import interp
import healpy as hp

from tqdm import tqdm
import imageio.v2 as imageio # Use imageio.v2 for the current API
from shutil import copyfile
from scipy import interpolate

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
# print('Date tag for this run which will be used to save the results: ', d1)

################################################################
"""Read in the config files."""
################################################################
# parser = argparse.ArgumentParser(description='Read the user inputs.')
# parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
#                     type=str, required=False)

# args = vars(parser.parse_args())

# config_file_path = args['config_file_path']

config_file_path = '/home/astro/phsprd/code/plastar/examples/emission/config/test/'

with open(config_file_path+'star.yaml') as f:
    config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
with open(config_file_path+'planet.yaml') as f:
    config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
with open(config_file_path+'telluric.yaml') as f:
    config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
with open(config_file_path+'simulation.yaml') as f:
    config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)

# dTspot--1100_spot_size-0.2_long-0_vsini-10_CRIRES_K-band_2280-2330-nm_
dTspot = str(config_dd_star['spots_and_faculae']['delta_teff'][0])
spot_size = str(config_dd_star['spots_and_faculae']['rad'][0])
long0 = str(config_dd_star['spots_and_faculae']['lon'][0])
vsini = str(config_dd_star['star']['v_eq'])
waverange = [str(config_dd_simulation['instrument']['wavelength_common_min']), 
            str(config_dd_simulation['instrument']['wavelength_common_max'])]
instrument = config_dd_simulation['instrument']['name']
addinfo = f'dTspot-{dTspot}_spot_size-{spot_size}_long-{long0}_vsini-{vsini}_{instrument}_{waverange[0]}-{waverange[1]}-nm'
infostring = config_dd_simulation['infostring'] + addinfo
savedir = config_dd_simulation['simulations_savedir'] + infostring + '/'

star_dict = config_dd_star['star']
spots_and_faculae_dict = config_dd_star['spots_and_faculae']
planet_dict = config_dd_planet
simulation_dict = config_dd_simulation

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir
    
copyfile(config_file_path+'star.yaml', savedir + 'star.yaml')
copyfile(config_file_path+'planet.yaml', savedir + 'planet.yaml')
copyfile(config_file_path+'telluric.yaml', savedir + 'telluric.yaml')
copyfile(config_file_path+'simulation.yaml', savedir + 'simulation.yaml')

################################################################
"""Compute the planetary flux first."""
################################################################
# import pdb; pdb.set_trace()
planet_atmosphere = planet.PlanetAtmosphere(planet_dict = planet_dict,
                                            simulation_dict = simulation_dict,
                                            star_dict = star_dict,
                                            wavelength_solution = None)

wavsoln, F_planet = planet_atmosphere.get_Fp_or_Rp()
# import pdb; pdb.set_trace()
################################################################

################################################################
"""Read in the PHOENIX models for star and spots and splice and convolve to instrument resolution."""
################################################################
wavsoln_model_star, flux_model_star = utils.get_stellar_spectral_models_phoenix(config_file_path = config_file_path)
wavsoln_model_spot, flux_model_spot = utils.get_spot_spectral_models_phoenix(config_file_path = config_file_path)

################################################################
"""Get the phases for the star and the planet, and the time stamps of the observation."""
################################################################
# phases_planet, phases_star, time_stamps = utils.get_star_planet_phases(config_file_path = config_file_path)
phases_planet, time_stamps = utils.get_star_planet_phases(config_file_path = config_file_path)

################################################################
################################################################
"""Create the instance for the StellarGrid and compute the spectral time series."""
################################################################
################################################################
print('Starting stellar flux calculation...')
star_grid = grid.StellarGrid(star_dict = star_dict, spots_and_faculae_dict = spots_and_faculae_dict,
                             planet_dict = planet_dict, 
                        wavsoln = wavsoln_model_star,
                        include_spots_and_faculae = True, include_planet = False)

star_quiet, F_star_quiet_, wavsoln_star = star_grid.get_spectral_time_series(time=time_stamps, 
                                                            stellar_spectrum = flux_model_star, 
                                                        spot_spectra = flux_model_spot, 
                                                        include_spots_and_faculae = False,
                                                        wavelength_chunk_length = config_dd_simulation['wavelength_chunk_length'], 
                                                        wavelength_overlap_length = config_dd_simulation['wavelength_overlap_length']
                                                        )

star, F_star_, wavsoln_star = star_grid.get_spectral_time_series(time=time_stamps, 
                                                            stellar_spectrum = flux_model_star, 
                                                        spot_spectra = flux_model_spot, 
                                                        include_spots_and_faculae = True,
                                                        wavelength_chunk_length = config_dd_simulation['wavelength_chunk_length'], 
                                                        wavelength_overlap_length = config_dd_simulation['wavelength_overlap_length']
                                                        )

print('Stellar flux calculation done!')

# N, n_n = core._N_or_Y_to_N_n(star.y[0])
# spot_cen_pix = hp.ang2pix(N, np.pi/2, 0)

# plt.figure()
# # plt.plot(wavsoln_star, F_star_[0,:,spot_cen_pix], label = 'active')
# # plt.plot(wavsoln_star, F_star_quiet_[0,:,spot_cen_pix], label = 'quiet')
# plt.plot(wavsoln_star, F_star_[0,:,spot_cen_pix], label = 'active')
# plt.plot(wavsoln_star, F_star_quiet_[0,:,spot_cen_pix], label = 'quiet')
# plt.legend()
# plt.savefig(savedir + 'design_matrix_test.png', dpi = 300, format = 'png')

# plt.figure()
# plt.plot(F_star_[0,0,97450:97622], label = 'active')
# plt.plot(F_star_quiet_[0,0,97450:97622], label = 'quiet')
# plt.plot(F_star_[2,0,97450:97622], linestyle = 'dashed', label = 'active')
# plt.plot(F_star_quiet_[2,0,97450:97622], linestyle = 'dashed', label = 'quiet')
# plt.legend()
# plt.savefig(savedir + 'projected_area_at_equator.png', dpi = 300, format = 'png')

# import pdb; pdb.set_trace()
################################################################
#### Interpolate F_star to wavsoln_planet
F_star = np.zeros((len(phases_planet), len(wavsoln)))
F_star_quiet = np.zeros((len(phases_planet), len(wavsoln)))

for ip in range(len(phases_planet)):
    model_spl_star = interpolate.make_interp_spline(wavsoln_star, 
                                F_star_[ip,:], bc_type='natural')
    model_spl_star_quiet = interpolate.make_interp_spline(wavsoln_star, 
                                F_star_quiet_[ip,:], bc_type='natural')
    F_star[ip,:] = model_spl_star(wavsoln)
    F_star_quiet[ip,:] = model_spl_star_quiet(wavsoln)

################################################################
################################################################
"""Create the observing sequence."""
################################################################
################################################################
## Compute the planet RV for each phase and Doppler shift and stack F_planet
berv = np.zeros((len(phases_planet),))
RV_array = ccf.compute_RV(Kp = planet_dict['Kp'], 
                          Vsys = star_dict['Vsys'], 
                          phases = phases_planet,
                          berv = berv)


###### Get the range of common wavelength vector that avoids extrapolations
wav_com_min, wav_com_max = simulation_dict["instrument"]["wavelength_common_min"], simulation_dict["instrument"]["wavelength_common_max"]
wav_com_min_ind = np.argmin(abs(wavsoln - wav_com_min)) 
wav_com_max_ind = np.argmin(abs(wavsoln - wav_com_max))

## Doppler shift and stack F_planet 
F_planet_shifted = np.ones( (len(phases_planet), len(wavsoln[wav_com_min_ind:wav_com_max_ind]) ) )
F_planet_spl = interpolate.make_interp_spline(wavsoln, F_planet)
for ip in range(len(phases_planet)):
    # wavsoln_shifted = ccf.doppler_shift_wavsoln(RV_array[ip], wavsoln[wav_com_min_ind:wav_com_max_ind])
    wavsoln_shifted = ccf.doppler_shift_wavsoln(-RV_array[ip], wavsoln[wav_com_min_ind:wav_com_max_ind]) ## When injecting as well, doppler shift the wavelength solution by -RV to shift the model by +RV effectively. 
    F_planet_shifted[ip,:] = F_planet_spl(wavsoln_shifted)

## Doppler shift F_star by Vsys and berv 
F_star_shifted = np.ones((len(phases_planet), len(wavsoln[wav_com_min_ind:wav_com_max_ind]) ))
for ip in range(len(phases_planet)):
    F_star_spl = interpolate.make_interp_spline(wavsoln, F_star[ip,:])
    RV_star = star_dict['Vsys'] + berv[ip]
    # wavsoln_shifted = ccf.doppler_shift_wavsoln(RV_star, wavsoln[wav_com_min_ind:wav_com_max_ind])
    wavsoln_shifted = ccf.doppler_shift_wavsoln(-RV_star, wavsoln[wav_com_min_ind:wav_com_max_ind])

    F_star_shifted[ip,:] = F_star_spl(wavsoln_shifted)
    
F_star_quiet_shifted = np.ones((len(phases_planet), len(wavsoln[wav_com_min_ind:wav_com_max_ind]) ))
for ip in range(len(phases_planet)):
    F_star_quiet_spl = interpolate.make_interp_spline(wavsoln, F_star_quiet[ip,:])
    RV_star = star_dict['Vsys'] + berv[ip]
    wavsoln_shifted = ccf.doppler_shift_wavsoln(RV_star, wavsoln[wav_com_min_ind:wav_com_max_ind])
    F_star_quiet_shifted[ip,:] = F_star_quiet_spl(wavsoln_shifted)

### Slice it to common wavelength range to avoid extrapolations
wavsoln_orig = wavsoln
wavsoln = wavsoln[wav_com_min_ind:wav_com_max_ind]
# F_planet = F_planet[wav_com_min_ind:wav_com_max_ind]
# F_star = F_star[:,wav_com_min_ind:wav_com_max_ind]
# F_star_quiet = F_star[:,wav_com_min_ind:wav_com_max_ind]
# F_planet_shifted = F_planet_shifted[:,wav_com_min_ind:wav_com_max_ind]
# F_star_shifted = F_star_shifted[:,wav_com_min_ind:wav_com_max_ind]
# F_star_quiet_shifted = F_star_quiet_shifted[:,wav_com_min_ind:wav_com_max_ind]

### Plot and check 
plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, F_planet_shifted)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Fp')
plt.savefig(savedir + 'Fp_doppler_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, F_star_shifted)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Fs')
plt.savefig(savedir + 'Fs_doppler_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.plot(wavsoln_orig, F_star[0,:], label = 'First exposure, Active', color = 'k')
plt.plot(wavsoln_orig, F_star[-1,:], label = 'Last exposure, Active', color = 'r')
plt.plot(wavsoln_orig, F_star_quiet[0,:], label = 'First exposure, Quiet', color = 'k', linestyle = 'dashed')
plt.plot(wavsoln_orig, F_star_quiet[-1,:], label = 'Last exposure, Quiet', color = 'r', linestyle = 'dashed')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fs')
plt.title('Fs')
plt.legend()
plt.savefig(savedir + 'Fs_1D_orig_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.plot(wavsoln_orig, F_star[0,:]/F_star_quiet[0,:], label = 'First exposure, Active/Quiet', color = 'k')
plt.plot(wavsoln_orig, F_star[-1,:]/F_star_quiet[-1,:], label = 'Last exposure, Active/Quiet', color = 'r')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fs')
plt.title('Fs')
plt.legend()
plt.savefig(savedir + 'Fs_1D_orig_Active_by_Quiet_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
Fp_by_Fs = F_planet_shifted/F_star_shifted
plt.pcolormesh(wavsoln, phases_planet, Fp_by_Fs)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Fp/Fs')
plt.savefig(savedir + 'Fp_by_Fs_doppler_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
Fp_by_Fs_quiet = F_planet_shifted/F_star_quiet_shifted
plt.pcolormesh(wavsoln, phases_planet, Fp_by_Fs_quiet )
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Fp/Fs quiet')
plt.savefig(savedir + 'Fp_by_Fs_quiet_doppler_shifted.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, Fp_by_Fs - Fp_by_Fs_quiet)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Fp/Fs - Fp/Fs (quiet)')
plt.savefig(savedir + 'Fp_by_Fs_minus_Fp_by_Fs_quiet_doppler_shifted.png', format = 'png', dpi = 300)

##### Plot the middle 1D Fp/Fs 
plt.figure(figsize = (18,10))
plt.plot(wavsoln, Fp_by_Fs[int(len(Fp_by_Fs)/2),:], label = 'active' )
plt.plot(wavsoln, Fp_by_Fs_quiet[int(len(Fp_by_Fs_quiet)/2),:], label = 'quiet' )
plt.savefig(savedir + 'Fp_by_Fs_1D_comparison.png', format = 'png', dpi = 300)

##### Save the outputs
spdd = {}
spdd['datacube'] = F_star_shifted*(1. + Fp_by_Fs)
spdd['datacube_quiet'] = F_star_quiet_shifted*(1. + Fp_by_Fs_quiet)

spdd['F_star_orig'] = F_star
spdd['F_star_quiet_orig'] = F_star_quiet
spdd['F_planet_orig'] = F_planet

spdd['F_star'] = F_star_shifted
spdd['F_star_quiet'] = F_star_quiet_shifted
spdd['F_planet'] = F_planet_shifted

spdd['berv'] = berv
spdd['phases'] = phases_planet
spdd['wavsoln_orig'] = wavsoln_orig
spdd['wav_com_min_ind'] = wav_com_min_ind
spdd['wav_com_max_ind'] = wav_com_max_ind
spdd['wavsoln'] = wavsoln
spdd['Fp_by_Fs'] = Fp_by_Fs
spdd['Fp_by_Fs_quiet'] = Fp_by_Fs_quiet
np.save(savedir + 'spdd.npy', spdd)

##### Plot the datacubes as well
plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, spdd['datacube'])
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('datacube')
plt.savefig(savedir + 'datacube.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, spdd['datacube_quiet'])
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('datacube (quiet)')
plt.savefig(savedir + 'datacube_quiet.png', format = 'png', dpi = 300)

plt.figure(figsize = (18,10))
plt.pcolormesh(wavsoln, phases_planet, spdd['datacube'] - spdd['datacube_quiet'])
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('datacube - datacube (quiet)')
plt.savefig(savedir + 'datacube_minus_datacube_quiet.png', format = 'png', dpi = 300)

exit()
# plt.figure()
# plt.plot(wavsoln, Fp/flux[0], label = 'Fp/Fs')
# plt.savefig(savedir + 'Fp_by_Fs.png', format = 'png', dpi = 300)

# plt.figure()
# plt.plot(wavsoln, Fp/np.max(Fp), label = 'Fp norm')
# plt.plot(wavsoln, flux[0]/np.max(flux[0]), label = 'Fs norm')
# plt.legend()
# plt.savefig(savedir + 'Fp_and_Fs.png', format = 'png', dpi = 300)

# import pdb
# pdb.set_trace()

################################################################
"""Get the planetary spectrum Fp using genesis"""
################################################################



"""Make the video"""
output_video_path = savedir + 'output_spectrum.mp4'
fps = 1 # Frames per second for the output video
dpi = 300
num_frames = len(phases_planet)

images_in_memory = []
for ip, phase_star in enumerate(phases_planet):
    print(phase_star)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(25, 15))
    
    ax = axes[0]
    ### Show the star first for this phase  
    # show(star, phase = phase_star, ax=ax, rv = True, radius = star.radius, period = star.period)
    viz.show(
    star.y[0],
    inc=star.inc,
    obl=star.obl if star.obl is not None else 0.0,
    u=star.u[0],
    xsize=800,
    phase=phase_star,
    ax=ax,
    radius=star.radius,
    period=star.period,
    rv=False)
    
    # viz.show(star.y, u=star.u[0])
    circle = plt.Circle((xp[ip], yp[ip]), star_grid.planet_radius, color="0.1", zorder=10)
    ax.add_artist(circle)
    
    ax = axes[1]
    
    ### Plot the spotty and non-spotty spectrum 
    # ax.plot(wavsoln_model_star, spotty_spectrum[ip,:], c="k", lw=1, label="spotted")
    # ax.plot(wavsoln_model_star, nonspotted_spectrum, "-", c="r", lw=1, label="non-spotted")
    print( np.mean(flux[0,:]), np.mean(flux[ip,:]) )
    # ax.plot(wavsoln_model_star, flux[ip,:]-flux[0,:], 
    #         c="k", lw=1, label='Flux')
    ax.plot(wavsoln, flux[ip,:], 
        c="k", lw=1, label='Flux')
    
    # ax.axis("off")
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(' F(t=0) - F(t)')
    # ax.set_ylim(0.95, 0.99)
    # plt.savefig(savedir + 'output_spectrum_phase_'+str(ip)+'.png', dpi = 300, format = 'png')
    
    # Important: Draw the canvas before getting the pixel data
    # This renders the figure to an internal buffer.
    fig.canvas.draw()
    
    # Get the raw RGBA pixels from the figure's canvas as a NumPy array
    # (width, height, 4 channels: R, G, B, Alpha)
    image_from_plot = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Append the image array to our list
    images_in_memory.append(image_from_plot)
    
    # Close the figure to free up memory immediately after processing
    plt.close(fig)

print(f"Finished generating {len(images_in_memory)} frames in memory.")

# --- 4. Create the video from the in-memory images ---
print(f"Creating video '{output_video_path}'...")
try:
    imageio.mimsave(output_video_path, images_in_memory, fps=fps)
    print("Video created successfully!")
except Exception as e:
    print(f"Error creating video: {e}")
    print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
    print("You might also need to install imageio with the ffmpeg plugin: 'pip install imageio[ffmpeg]'")


