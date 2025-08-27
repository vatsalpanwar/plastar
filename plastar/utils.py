import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import yaml
import sys
import jax.numpy as jnp
from plastar import grid
from astropy.io import fits
from spotter import show, viz
import astropy.constants as const
import astropy.units as un
import time
from tqdm import tqdm
import imageio.v2 as imageio # Use imageio.v2 for the current API
from shutil import copyfile
from expecto import get_spectrum


def get_stellar_spectral_models_phoenix(config_file_path = None):
    
    # Load the config files.
    with open(config_file_path + 'star.yaml') as f:
        config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'planet.yaml') as f:
        config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'telluric.yaml') as f:
        config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'simulation.yaml') as f:
        config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)
    
    # Get the star dictionary
    star_dict = config_dd_star['star']

    # Get the working resolution of the model    
    R = config_dd_planet['model_resolution_working']
    
    # Get the minimum and maximum wavelength range 
    lam_min, lam_max = config_dd_simulation['instrument']['wavelength_min'], config_dd_simulation['instrument']['wavelength_max']
    
    if config_dd_simulation['instrument']['wavelength_unit'] == 'Angstrom':
        lam_min, lam_max = lam_min*1.0e-10, lam_max*1.0e-10
    elif config_dd_simulation['instrument']['wavelength_unit'] == 'nm':
        lam_min, lam_max = lam_min*1.0e-9, lam_max*1.0e-9
    
    # Generate a constant wavelength resolution grid (be default in nm) within the instrument wavelength bounds
    num_vals = R*(jnp.log(lam_max) - jnp.log(lam_min))
    lam = jnp.logspace(jnp.log(lam_min),jnp.log(lam_max),num = int(num_vals)+1,endpoint=True,base=np.e) # constant resolution, so need evenly spaced in ln(lambda)
    lam_nm = lam*1e9
    
    # Load the PHOENIX model from the FTP server
    R_solar_squared = 6.957e8*6.957e8 ## in m2
    star_teff, star_log_g, star_met = star_dict['teff'], star_dict['log_g'], star_dict['met']
    
    spectrum = get_spectrum(T_eff=float(star_teff), log_g=float(star_log_g), Z = float(star_met),
                            cache=True)
    # Convert the spectrum to desired units 
    wavsoln_model = jnp.array(spectrum.wavelength.value/10.) # Convert wavelengths to nm 
    flux_model = jnp.array(spectrum.flux.value * 1e-7 * 1e4 * 1e2) * R_solar_squared # Convert to SI units, and scale by R_sun^2 (because the stellar radius is defined in solar radius.)
    
    # Get the indices corresponding to the desired walvength grid 
    wav_inds = [ jnp.argmin(abs(wavsoln_model-lam_nm[0])) , jnp.argmin(abs(wavsoln_model-lam_nm[-1])) ]
    
    # Slice the model to this range, a bit wider to allow interpolation and Doppler shifts
    flux_model = flux_model[wav_inds[0]:wav_inds[1]]
    wavsoln_model = wavsoln_model[wav_inds[0]:wav_inds[1]]
    
    return wavsoln_model, flux_model
    

def get_spot_spectral_models_phoenix(config_file_path = None):
    """
    Load the spectra for all the spot regions.
    Can be for either explicitly defined N_spot, deltaTs, delta_log_g, and delta_met for each spot region, 
    or sample them from a distribution. 
    Do it for explicitly defined case for now.
    """
    
    # Load the config files.
    with open(config_file_path + 'star.yaml') as f:
        config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'planet.yaml') as f:
        config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'telluric.yaml') as f:
        config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'simulation.yaml') as f:
        config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)
    
    # Get the spots and faculae dictionary
    sp_fac_dict = config_dd_star['spots_and_faculae']

    # Get the working resolution of the model    
    R = config_dd_planet['model_resolution_working']
    
    # Get the minimum and maximum wavelength range 
    lam_min, lam_max = config_dd_simulation['instrument']['wavelength_min'], config_dd_simulation['instrument']['wavelength_max']
    
    if config_dd_simulation['instrument']['wavelength_unit'] == 'Angstrom':
        lam_min, lam_max = lam_min*1.0e-10, lam_max*1.0e-10
    elif config_dd_simulation['instrument']['wavelength_unit'] == 'nm':
        lam_min, lam_max = lam_min*1.0e-9, lam_max*1.0e-9
    
    # Generate a constant wavelength resolution grid (be default in nm) within the instrument wavelength bounds
    num_vals = R*(jnp.log(lam_max) - jnp.log(lam_min))
    lam = jnp.logspace(jnp.log(lam_min),jnp.log(lam_max),num = int(num_vals)+1,endpoint=True,base=np.e) # constant resolution, so need evenly spaced in ln(lambda)
    lam_nm = lam*1e9
    
    # Load the PHOENIX model from the FTP server
    R_solar_squared = 6.957e8*6.957e8 ## in m2
    
    flux_model_all = []
    for isp in range(len(sp_fac_dict['delta_teff'])):
        sp_fac_teff, sp_fac_log_g, sp_fac_met = sp_fac_dict['delta_teff'][isp] + config_dd_star['star']['teff'], sp_fac_dict['log_g'][isp], sp_fac_dict['met'][isp]
        
        spectrum = get_spectrum(T_eff=float(sp_fac_teff), log_g=float(sp_fac_log_g), Z = float(sp_fac_met),
                                cache=True)
        
        # Convert the spectrum to desired units 
        wavsoln_model = jnp.array(spectrum.wavelength.value/10.) # Convert wavelengths to nm 
        flux_model = jnp.array(spectrum.flux.value * 1e-7 * 1e4 * 1e2) * R_solar_squared # Convert to SI units, and scale by R_sun^2 (because the stellar radius is defined in solar radius.)
        
        # Get the indices corresponding to the desired walvength grid 
        wav_inds = [ jnp.argmin(abs(wavsoln_model-lam_nm[0])) , jnp.argmin(abs(wavsoln_model-lam_nm[-1])) ]
        
        # Slice the model to this range, a bit wider to allow interpolation and Doppler shifts
        flux_model = flux_model[wav_inds[0]:wav_inds[1]]

        wavsoln_model = wavsoln_model[wav_inds[0]:wav_inds[1]]
        flux_model_all.append(flux_model)
    
    wavsoln_model_all = wavsoln_model # Take the value from the last iteration of the loop 
    flux_model_all = jnp.array(flux_model_all)
    
    return wavsoln_model_all, flux_model_all
    
def get_star_planet_phases(config_file_path = None):
    # Load the config files.
    with open(config_file_path + 'star.yaml') as f:
        config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'planet.yaml') as f:
        config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'telluric.yaml') as f:
        config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
    with open(config_file_path + 'simulation.yaml') as f:
        config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)
        
    """Setup the time stamps and phases for both the star and the planet."""
    ## Planet 
    phase_range = config_dd_simulation['phase_range']
    time_range = config_dd_planet['orbper'] * np.array(phase_range) ## In days

    ## Calculate the time stamps for the simulation, using the start and end time based on the planetary orbital phase.
    time_stamps = np.arange(time_range[0], time_range[1], config_dd_simulation['time_step']/(3600.*24)) ## In days
    print("Number of time stamps: ", len(time_stamps))
    phases_planet = time_stamps/config_dd_planet['orbper']

    ## Star
    ## Get the zeroth time stamp for the star, for the simulation 
    time_stamp0_star = config_dd_simulation['planet_phase_at_star_phase0']*config_dd_planet['orbper']
    star_period = ( (2 * np.pi * config_dd_star['star']['Rs'] * un.Rsun) / (config_dd_star['star']['v_eq'] * un.km / un.s) ).to(un.day)
    phases_star = (time_stamps - time_stamp0_star)/star_period.value
    
    return phases_planet, phases_star, time_stamps


def create_overlapping_chunks(arr, chunk_length, overlap_length):
    """
    Creates overlapping chunks from a 1D NumPy array.

    Args:
        arr (np.ndarray): The input 1D array.
        chunk_length (int): The length of each chunk.
        overlap (int): The number of overlapping elements.

    Returns:
        np.ndarray: A 2D array of overlapping chunks.
    """
    step = chunk_length - overlap_length
    shape = (arr.size - overlap_length) // step, chunk_length
    
    if (arr.size - chunk_length) % step != 0:
        print("Warning: Array size does not perfectly align with chunking parameters.")
    
    strides = (step * arr.itemsize, arr.itemsize)
    chunked = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return chunked

def merge_chunks_back(chunks, chunk_length, overlap_length):
    """
    Merges overlapping chunks back into a 1D array.

    Args:
        chunks (np.ndarray): The 2D array of overlapping chunks.
        chunk_length (int): The length of each chunk.
        overlap (int): The number of overlapping elements.

    Returns:
        np.ndarray: The reconstructed 1D array.
    """
    step = chunk_length - overlap_length
    reconstructed_length = (chunks.shape[0] - 1) * step + chunk_length
    reconstructed_arr = np.zeros(reconstructed_length, dtype=chunks.dtype)

    for i in range(chunks.shape[0]):
        start_index = i * step
        end_index = start_index + chunk_length
        reconstructed_arr[start_index:end_index] = chunks[i]
    
    return reconstructed_arr

# # Example usage
# arr = np.arange(10)
# chunks = create_overlapping_chunks(arr, chunk_length=4, overlap=2)
# print("Original Array:", arr)
# print("Overlapping Chunks:\n", chunks)

# Example usage
# merged_arr = merge_chunks_back(chunks, chunk_length=4, overlap=2)
# print("\nMerged Array:", merged_arr)