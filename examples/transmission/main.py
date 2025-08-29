import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import yaml
import jax.numpy as jnp
from plastar import grid
from plastar import utils
from astropy.io import fits
from spotter import show, viz
from tqdm import tqdm
import imageio.v2 as imageio # Use imageio.v2 for the current API
from shutil import copyfile

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
"""Read in the config files."""
################################################################
# parser = argparse.ArgumentParser(description='Read the user inputs.')
# parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
#                     type=str, required=False)

# args = vars(parser.parse_args())

# config_file_path = args['config_file_path']

with open('./config/star.yaml') as f:
    config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
with open('./config/planet.yaml') as f:
    config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
with open('./config/telluric.yaml') as f:
    config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
with open('./config/simulation.yaml') as f:
    config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)


infostring = config_dd_simulation['infostring'] + d1
savedir = config_dd_simulation['simulations_savedir'] + infostring + '/'

star_dict = config_dd_star['star']
spots_and_faculae_dict = config_dd_star['spots_and_faculae']
planet_dict = config_dd_planet

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir
    
copyfile('./config/star.yaml', savedir + 'star.yaml')
copyfile('./config/planet.yaml', savedir + 'planet.yaml')
copyfile('./config/telluric.yaml', savedir + 'telluric.yaml')
copyfile('./config/simulation.yaml', savedir + 'simulation.yaml')

################################################################
"""Read in the PHOENIX models for star and spots and splice and convolve to instrument resolution."""
################################################################
wavsoln_model_star, flux_model_star = utils.get_stellar_spectral_models_phoenix(config_file_path = './config/')
wavsoln_model_spot, flux_model_spot = utils.get_spot_spectral_models_phoenix(config_file_path = './config/')

################################################################
"""Get the phases for the star and the planet, and the time stamps of the observation."""
################################################################
phases_planet, phases_star, time_stamps = utils.get_star_planet_phases(config_file_path = './config/')

################################################################
################################################################
"""Create the instance for the StellarGrid and compute the spectral time series."""
################################################################
################################################################

star_grid = grid.StellarGrid(star_dict = star_dict, spots_and_faculae_dict = spots_and_faculae_dict,
                             planet_dict = planet_dict, 
                        wavsoln = wavsoln_model_star,
                        include_spots_and_faculae = True, include_planet = True)

star, flux, wavsoln = star_grid.get_spectral_time_series(time=time_stamps, 
                                                            stellar_spectrum = flux_model_star, 
                                                        spot_spectra = flux_model_spot, 
                                                        include_spots_and_faculae = True,
                                                        wavelength_chunk_length = config_dd_simulation['wavelength_chunk_length'], 
                                                        wavelength_overlap_length = config_dd_simulation['wavelength_overlap_length']
                                                        )
xp, yp, zp = star_grid.planet_coords(time_stamps)

plt.figure()
plt.plot(phases_planet, np.sum(flux, axis = 1)/np.max(np.sum(flux, axis = 1)) )
# plt.plot(phases_planet, np.sum(flux, axis = 1) )
plt.savefig(savedir + 'output_light_curve.png', dpi = 300, format = 'png')


"""Make the video"""
output_video_path = savedir + 'output_spectrum.mp4'
fps = 1 # Frames per second for the output video
dpi = 300
num_frames = len(phases_star)

images_in_memory = []
for ip, phase_star in enumerate(phases_star):
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


