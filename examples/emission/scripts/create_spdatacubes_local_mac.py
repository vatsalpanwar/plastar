import numpy as np
import skycalc_ipy
from plastar import utils
import yaml 
import matplotlib.pyplot as plt 
from scipy import interpolate
import os
from plastar import ccf
import jax.numpy as jnp
from shutil import copyfile

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#### Load the plastar simulated models ####
savedir = '/Users/v.panwar@bham.ac.uk/source/code/plastar/examples/emission/results/datacubes/'
results_root = '/Users/v.panwar@bham.ac.uk/source/code/plastar/examples/emission/results/'


### Specify the model directory name 
# dir_name = 'MAIN_emission_eq-chem-ER_planet_phases_rev_dTspot--1100.0_spot_size-0.2_long-0.0_vsini-20.0_crires_2280.0-2330.0-nm'
# dir_name = 'MAIN_emission_eq-chem-ER_planet_phases_rev_dTspot--1100.0_spot_size-0.2_long-0.0_vsini-3.5_crires_2280.0-2330.0-nm'

dir_name = 'MAIN_emission_eq-chem-ER_planet_phases_rev_dTspot--1100.0_spot_size-0.2_long-0.0_vsini-20.0_crires_1890.0-2560.0-nm'
# dir_name = 'MAIN_emission_eq-chem-ER_planet_phases_rev_dTspot--1100.0_spot_size-0.2_long-0.0_vsini-3.5_crires_1890.0-2560.0-nm'

# SNR = 500
# SNR = 500
SNR = 500
RES = '200k'
# RES = '100k'
scale_factor = 1

addinfo = '_only_tell_modulated_noise_SNR-' + str(SNR) + f'res-{RES}'
# addinfo = '_only_tell_modulated_noise_SNR-' + str(SNR) +'res-{RES}'

print(dir_name)

results_path = results_root + dir_name + '/'
savedir = savedir + dir_name + '/'
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

### Load the plastar model dictionary 
spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()

data_wavsoln = spdd['wavsoln_sliced']
star_model_wavsoln = spdd['wavsoln_orig']

# wav_sim_data_min, wav_sim_data_max = 1950., 2450.
# wav_star_min, wav_star_max = 1900., 2500.
wav_sim_data_min, wav_sim_data_max = 2000., 2400.
wav_star_min, wav_star_max = 1900., 2500.

wav_sim_data_min_ind = np.argmin(abs(data_wavsoln - wav_sim_data_min)) 
wav_sim_data_max_ind = np.argmin(abs(data_wavsoln - wav_sim_data_max)) 

wav_star_min_ind = np.argmin(abs(star_model_wavsoln - wav_star_min)) 
wav_star_max_ind = np.argmin(abs(star_model_wavsoln - wav_star_max)) 

data_wavsoln = data_wavsoln[wav_sim_data_min_ind:wav_sim_data_max_ind]
star_model_wavsoln = star_model_wavsoln[wav_star_min_ind:wav_star_max_ind]

phases = spdd['phases']
berv = spdd['berv']

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
###### ###### ###### DEFINE THE DATACUBE ###### ###### ###### ###### ###### 
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 

###### Do you want the data to be active or quiet? 
F_star_active_for_datacube = spdd['F_star_active_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind]
F_star_quiet_for_datacube = spdd['F_star_quiet_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind]
F_planet_for_datacube = spdd['F_planet_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind]

datacube_active = F_star_active_for_datacube + F_planet_for_datacube
datacube_quiet = F_star_quiet_for_datacube + F_planet_for_datacube

datacube_active_only_star = F_star_active_for_datacube
datacube_quiet_only_star = F_star_quiet_for_datacube

# datacube_active = spdd['F_star_active_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind] + spdd['F_planet_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind]
# datacube_quiet = spdd['F_star_quiet_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind] + spdd['F_planet_shifted'][:,wav_sim_data_min_ind:wav_sim_data_max_ind]
    
### Copy and save the config files in the simulated datafile
with open(results_path + 'star.yaml') as f:
    config_dd_star = yaml.load(f,Loader=yaml.FullLoader)
with open(results_path + 'planet.yaml') as f:
    config_dd_planet = yaml.load(f,Loader=yaml.FullLoader)
with open(results_path + 'telluric.yaml') as f:
    config_dd_telluric = yaml.load(f,Loader=yaml.FullLoader)
with open(results_path + 'simulation.yaml') as f:
    config_dd_simulation = yaml.load(f,Loader=yaml.FullLoader)
copyfile(results_path + 'star.yaml', savedir + 'star.yaml')
copyfile(results_path + 'planet.yaml', savedir + 'planet.yaml')
copyfile(results_path + 'telluric.yaml', savedir + 'telluric.yaml')
copyfile(results_path + 'simulation.yaml', savedir + 'simulation.yaml')

### Compute a vector of PWV (at par with mean Paranal values), and compute a time series of skycalc transmission spectrum.
pwv_vector = utils.get_random_pwv(tExp = config_dd_simulation['time_step'], size = datacube_active.shape[0])

telluric_transmission = []
eso_sky_calc_pwv_grid = [-1.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
for pwv_val in pwv_vector:
    skycalc = skycalc_ipy.SkyCalc()
    _ = skycalc.get_sky_spectrum()
    skycalc["wres"] = 200000 # config_dd_simulation["instrument"]["resolution"]
    skycalc["wmin"], skycalc["wmax"] = min(data_wavsoln), max(data_wavsoln)
    pwv_ind = np.argmin(abs(eso_sky_calc_pwv_grid - pwv_val))
    skycalc["pwv"] = eso_sky_calc_pwv_grid[pwv_ind]
    wave, transmission, flux = skycalc.get_sky_spectrum(return_type="array")
    
    model_spl = interpolate.make_interp_spline(wave, transmission, bc_type = "natural")
    transmission = model_spl(data_wavsoln)
    telluric_transmission.append(transmission)
    
# telluric_wave = wave
telluric_wave = data_wavsoln
telluric_transmission = np.array(telluric_transmission)

## Plot the telluric transmission
plt.figure(figsize = (18,10))
for iexp in range(5):
    plt.plot(telluric_wave, telluric_transmission[iexp,:], label = 'Exposure number: ' + str(iexp))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Telluric transmission')
plt.savefig(savedir + 'telluric_sequence_1D.png', format = 'png', dpi = 300)
plt.close()

plt.figure(figsize = (18,10))
plt.pcolormesh(telluric_wave, phases, telluric_transmission)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Telluric Transmission ESO SkyCalc')
plt.savefig(savedir + 'telluric_sequence.png', format = 'png', dpi = 300)
plt.close()

## Convolve the simulated data to instrument resolution and add random noise to it.
datacube_quiet_noiseless, datacube_active_noiseless = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)
datacube_quiet_noiseless_tell_mod, datacube_active_noiseless_tell_mod = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)
datacube_quiet_only_star_noiseless_tell_mod, datacube_active_only_star_noiseless_tell_mod = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)

datacube_quiet_tell_mod, datacube_active_tell_mod = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)
datacube_quiet_only_star_tell_mod, datacube_active_only_star_tell_mod = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)

noise_active_all, noise_quiet_all = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)
noise_active_all_tell_mod, noise_quiet_all_tell_mod = np.zeros(datacube_active.shape), np.zeros(datacube_active.shape)
for ip in range(datacube_active.shape[0]):

    if RES == '100k':
        instrument_resolution = 100000
    elif RES == '200k':
        instrument_resolution = None
    # datacube_active[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = instrument_resolution, 
    #                                                            model_resolution = 200000, 
    #                                                            model_spec_orig = datacube_active[ip,:])
    # datacube_quiet[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = instrument_resolution, 
    #                                                            model_resolution = 200000, 
    #                                                            model_spec_orig = datacube_quiet[ip,:])

    datacube_active_noiseless[ip,:] = datacube_active[ip,:]/scale_factor
    datacube_quiet_noiseless[ip,:] = datacube_quiet[ip,:]/scale_factor

    datacube_active_noiseless_tell_mod[ip,:] = datacube_active[ip,:]* telluric_transmission[ip,:]/scale_factor
    datacube_quiet_noiseless_tell_mod[ip,:] = datacube_quiet[ip,:]* telluric_transmission[ip,:]/scale_factor

    datacube_active_only_star_noiseless_tell_mod[ip,:] = datacube_active_only_star[ip,:]* telluric_transmission[ip,:]/scale_factor
    datacube_quiet_only_star_noiseless_tell_mod[ip,:] = datacube_quiet_only_star[ip,:]* telluric_transmission[ip,:]/scale_factor
    ## Also do the planet and star spectra separately 
    # F_star_active_for_datacube[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = instrument_resolution, 
    #                                                            model_resolution = 200000, 
    #                                                            model_spec_orig = F_star_active_for_datacube[ip,:])

    # F_star_quiet_for_datacube[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = instrument_resolution, 
    #                                                            model_resolution = 200000, 
    #                                                            model_spec_orig = F_star_quiet_for_datacube[ip,:])

    # F_planet_for_datacube[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = instrument_resolution, 
    #                                                            model_resolution = 200000, 
    #                                                            model_spec_orig = F_planet_for_datacube[ip,:])

    # noise_active = np.random.normal(0., abs(datacube_active[ip,:] * telluric_transmission[ip,:]/SNR )/scale_factor )
    # noise_quiet = np.random.normal(0., abs(datacube_quiet[ip,:] * telluric_transmission[ip,:]/SNR )/scale_factor )
    
    loc_active = datacube_active[ip,:] * telluric_transmission[ip,:]/scale_factor
    sigma_active = abs(loc_active / SNR)

    loc_quiet = datacube_quiet[ip,:] * telluric_transmission[ip,:]/scale_factor
    sigma_quiet = abs(loc_quiet / SNR) 

    noise_active = np.random.normal(0., sigma_active)
    noise_quiet = np.random.normal(0., sigma_quiet )
    
    datacube_active_tell_mod[ip,:] = datacube_active_noiseless_tell_mod[ip,:]+ noise_active
    datacube_quiet_tell_mod[ip,:] = datacube_quiet_noiseless_tell_mod[ip,:]+ noise_quiet
    
    datacube_active[ip,:] = datacube_active_noiseless[ip,:]+ noise_active
    datacube_quiet[ip,:] = datacube_quiet_noiseless[ip,:]+ noise_quiet

    #### Only star 
    loc_active_only_star = datacube_active_only_star[ip,:] * telluric_transmission[ip,:]/scale_factor
    sigma_active_only_star = abs(loc_active_only_star / SNR)

    loc_quiet_only_star = datacube_quiet_only_star[ip,:] * telluric_transmission[ip,:]/scale_factor
    sigma_quiet_only_star = abs(loc_quiet_only_star / SNR) 

    noise_active_only_star = np.random.normal(0., sigma_active_only_star)
    noise_quiet_only_star = np.random.normal(0., sigma_quiet_only_star )

    datacube_active_only_star_tell_mod[ip,:] = datacube_active_only_star_noiseless_tell_mod[ip,:]+ noise_active_only_star
    datacube_quiet_only_star_tell_mod[ip,:] = datacube_quiet_only_star_noiseless_tell_mod[ip,:]+ noise_quiet_only_star

    ######## Save the noise values
    noise_active_all[ip,:], noise_quiet_all[ip,:] = noise_active, noise_quiet
    noise_active_all_tell_mod[ip,:], noise_quiet_all_tell_mod[ip,:] = noise_active_only_star, noise_quiet_only_star
    # noise = np.random.normal(0., abs(datacube[ip,:] * telluric_transmission[ip,:]/SNR ) )
    # datacube[ip,:] = datacube[ip,:]*telluric_transmission[ip,:] + noise
    
spdd_save_active = {}
spdd_save_active['spdatacube'] = datacube_active[np.newaxis,:,:]
spdd_save_active['F_planet_conv'] = F_planet_for_datacube
spdd_save_active['F_star_conv'] = F_star_active_for_datacube
spdd_save_active['spdatacube_noiseless'] = datacube_active_noiseless[np.newaxis,:,:]
spdd_save_active['SNR'] = SNR 
spdd_save_active['phases'] = phases
spdd_save_active['time'] = phases
spdd_save_active['bary_RV'] = berv
spdd_save_active['wavsoln'] = data_wavsoln[np.newaxis,:]
spdd_save_active['file_name'] = None
np.save(savedir + 'spdd' + addinfo + '_active.npy', spdd_save_active)

spdd_save_quiet = {}
spdd_save_quiet['spdatacube'] = datacube_quiet[np.newaxis,:,:]
spdd_save_quiet['F_planet_conv'] = F_planet_for_datacube
spdd_save_quiet['F_star_conv'] = F_star_quiet_for_datacube
spdd_save_quiet['spdatacube_noiseless'] = datacube_quiet_noiseless[np.newaxis,:,:]
spdd_save_quiet['SNR'] = SNR 
spdd_save_quiet['phases'] = phases
spdd_save_quiet['time'] = phases
spdd_save_quiet['bary_RV'] = berv
spdd_save_quiet['wavsoln'] = data_wavsoln[np.newaxis,:]
spdd_save_quiet['file_name'] = None
np.save(savedir + 'spdd' + addinfo + '_quiet.npy', spdd_save_quiet)

####### Plot the noisy data in the savedir for both active and quiet cases
plt.figure(figsize = (18,10))
plt.pcolormesh(spdd_save_active['wavsoln'], spdd_save_active['phases'], 
               spdd_save_active['spdatacube'][0,:,:]) ## Only one order 
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Datacube with telluric modulated noise: Active')
plt.savefig(savedir + 'datacube_with_noise'+addinfo+'_active.png', format = 'png', dpi = 300)
plt.close()

plt.figure(figsize = (18,10))
plt.pcolormesh(spdd_save_quiet['wavsoln'], spdd_save_quiet['phases'], 
               spdd_save_quiet['spdatacube'][0,:,:]) ## Only one order 
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Datacube with telluric modulated noise: Quiet')
plt.savefig(savedir + 'datacube_with_noise'+addinfo+'_quiet.png', format = 'png', dpi = 300)
plt.close()

plt.figure(figsize = (18,10))
plt.plot(spdd_save_active['wavsoln'][0,:], spdd_save_active['spdatacube'][0,0,:], label = 'active')
plt.plot(spdd_save_quiet['wavsoln'][0,:], spdd_save_quiet['spdatacube'][0,0,:], label = 'quiet')
plt.legend()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fp+Fs (with noise)')
plt.savefig(savedir + 'datacube_1exp_with_noise'+addinfo+'_quiet-and-active.png', format = 'png', dpi = 300)
plt.close()

####### Also save the stellar model separately for use during retrievals
star_planet_model = {}
star_planet_model['F_star_active_orig'] = spdd['F_star_active_orig'][:,wav_star_min_ind:wav_star_max_ind] ## Not Doppler shifted or convolved
star_planet_model['F_star_quiet_orig'] = spdd['F_star_quiet_orig'][:,wav_star_min_ind:wav_star_max_ind] ## Not Doppler shifted or convolved
star_planet_model['F_planet_orig'] =  spdd['F_planet_orig'][wav_star_min_ind:wav_star_max_ind] ## Not Doppler shifted or convolved
star_planet_model['wavsoln'] = star_model_wavsoln
np.save(savedir + 'star_planet_model.npy', star_planet_model)

####### Plot a stellar model for active and quiet cases for an arbitrary time point 
fig, ax = plt.subplots(2,1,figsize = (10,18))
ax[0].plot(star_planet_model['wavsoln'], star_planet_model['F_star_active_orig'][10,:], label = 'Active')
ax[0].plot(star_planet_model['wavsoln'], star_planet_model['F_star_quiet_orig'][10,:], label = 'Quiet')
ax[0].legend()
ax[1].plot(star_planet_model['wavsoln'], star_planet_model['F_star_active_orig'][10,:] / star_planet_model['F_star_quiet_orig'][10,:],
           label = 'Fs Active/Quiet')
ax[1].legend()
ax[0].set_xlabel('Wavelength [nm]')
ax[0].set_ylabel('Flux')
ax[1].set_ylabel('Flux')
plt.savefig(savedir + 'Fstar_active_and_quiet.png', format = 'png', dpi = 300)
plt.close()

###### Also compute a quick Kp-Vsys map with the injected model and save it 
Vsys_range = jnp.linspace(-50., 0., 50)
Kp_range = jnp.linspace(120, 180, 60)
model_wavsoln = jnp.array(star_model_wavsoln)
data_wavsoln = jnp.array(data_wavsoln)
phases = jnp.array(spdd_save_active['phases'])
berv = jnp.array(spdd_save_active['bary_RV'])
modelcube_Fp = jnp.array(spdd['F_planet_orig'][np.newaxis,wav_star_min_ind:wav_star_max_ind] * np.ones((len(phases), len(model_wavsoln) )) )
modelcube_Fs_active = jnp.array(star_planet_model['F_star_active_orig'])
modelcube_Fs_quiet = jnp.array(star_planet_model['F_star_quiet_orig'])

###### Do telluric and stellar correction with N_PCA = 1 and without PCA
N_PCA = 1
datacube_active_det = ccf.get_PCA_detrended_datacube(datacube = datacube_active, nc = N_PCA)
datacube_quiet_det = ccf.get_PCA_detrended_datacube(datacube = datacube_quiet, nc = N_PCA)


datacube_active_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_active, correction_cube = spdd_save_active['F_star_conv']/scale_factor)
datacube_quiet_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_quiet, correction_cube = spdd_save_quiet['F_star_conv']/scale_factor)

datacube_active_tell_mod_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_active_tell_mod, correction_cube = telluric_transmission * spdd_save_active['F_star_conv']/scale_factor)
datacube_quiet_tell_mod_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_quiet_tell_mod, correction_cube = telluric_transmission * spdd_save_quiet['F_star_conv']/scale_factor)

datacube_active_only_star_tell_mod_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_active_only_star_tell_mod, correction_cube = telluric_transmission * spdd_save_active['F_star_conv']/scale_factor)
datacube_quiet_only_star_tell_mod_det_no_pca = ccf.get_perfect_detrended_datacube(datacube = datacube_quiet_only_star_tell_mod, correction_cube = telluric_transmission * spdd_save_quiet['F_star_conv']/scale_factor)


fig, axx = plt.subplots(2, 1, figsize=(12, 5*2))

### Plot the 2D cube of telluric transmission
plt.figure()
im = plt.pcolormesh(spdd_save_active['wavsoln'], spdd_save_active['phases'], 
              telluric_transmission) ## Only one order 
plt.show()

fig, axx = plt.subplots(4, 1, figsize=(12, 5*4), sharex = True)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_noiseless[0,:], label = 'active noiseless', alpha = 0.7)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_noiseless_tell_mod[0,:], label = 'active noiseless tell mod', alpha = 0.7)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active[0,:], 'o', label = 'active, orig', alpha = 0.7)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_tell_mod[0,:], 'o',label = 'active, tell mod, orig', alpha = 0.7)
axx[0].legend()
# axx[0].set_ylim(0,7)
axx[1].plot(spdd_save_active['wavsoln'][0], noise_active_all[0,:], 'o-', label = 'noise, active, orig', alpha = 0.7)
axx[1].legend()
# axx[1].set_ylim(-0.05,0.05)


axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_noiseless[0,:], label = 'quiet noiseless', alpha = 0.7)
axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_noiseless_tell_mod[0,:], label = 'quiet noiseless tell mod', alpha = 0.7)
axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet[0,:], 'o', label = 'quiet, orig', alpha = 0.7)
axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_tell_mod[0,:], 'o',label = 'quiet, tell mod, orig', alpha = 0.7)
axx[2].legend()
# axx[2].set_ylim(0,7)
axx[3].plot(spdd_save_quiet['wavsoln'][0], noise_quiet_all[0,:], 'o-', label = 'noise, quiet, orig', alpha = 0.7)
axx[3].legend()
# axx[3].set_ylim(-0.05,0.05)

axx[0].set_ylabel('Flux')
axx[1].set_ylabel('Flux')
axx[2].set_ylabel('Flux')
axx[3].set_ylabel('Flux')
axx[3].set_xlabel('Wavelength [nm]')
plt.show()

##### compare detrended and noise 
fig, axx = plt.subplots(4, 1, figsize=(12, 5*4), sharex = True)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_det_no_pca[0,:], label = 'active, det', alpha = 0.7)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_tell_mod_det_no_pca[0,:], label = 'active, tell mod, det', alpha = 0.7)
axx[0].plot(spdd_save_active['wavsoln'][0], datacube_active_only_star_tell_mod_det_no_pca[0,:], label = 'active, only star, tell mod, det', alpha = 0.7)
axx[0].legend(frameon = False)
# axx[0].set_ylim(-0.008,0.008)
axx[1].plot(spdd_save_active['wavsoln'][0], noise_active_all[0,:], label = 'noise, active, orig', alpha = 0.7)
# axx[1].set_ylim(-0.05,0.05)
axx[1].legend(frameon = False)

axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_det_no_pca[0,:], label = 'quiet, det', alpha = 0.7)
axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_tell_mod_det_no_pca[0,:], label = 'quiet, tell mod, det', alpha = 0.7)
axx[2].plot(spdd_save_quiet['wavsoln'][0], datacube_quiet_only_star_tell_mod_det_no_pca[0,:], label = 'quiet, tell mod, det', alpha = 0.7)
axx[2].legend(frameon = False)
# axx[2].set_ylim(-0.008,0.008)
axx[3].plot(spdd_save_quiet['wavsoln'][0], noise_quiet_all[0,:], label = 'noise, quiet, orig', alpha = 0.7)
axx[3].legend(frameon = False)
# axx[3].set_ylim(-0.05,0.05)

axx[0].set_ylabel('Flux')
axx[1].set_ylabel('Flux')
axx[2].set_ylabel('Flux')
axx[3].set_ylabel('Flux')
axx[3].set_xlabel('Wavelength [nm]')
plt.show()



#### Plot the detrended datacubes

## Active
fig, axx = plt.subplots(2, 1, figsize=(12, 5*2))
plt.subplots_adjust(hspace=0.8)

im = axx[0].pcolormesh(spdd_save_active['wavsoln'], spdd_save_active['phases'], 
               datacube_active_det) ## Only one order 
fig.colorbar(im, ax=axx[0])
im = axx[1].pcolormesh(spdd_save_active['wavsoln'], spdd_save_active['phases'], 
datacube_active_det_no_pca) ## Only one order 
fig.colorbar(im, ax=axx[1])

axx[1].set_xlabel('Wavelength [nm]')
axx[0].set_ylabel('Phases')
axx[1].set_ylabel('Phases')

axx[0].set_title(f'Datacube (Active) detrended with N_PCA = {str(N_PCA)}')
axx[1].set_title(f'Perfect detrending')

plt.savefig(savedir + 'datacube_active_detrended'+ addinfo + '_N_PCA-' + str(N_PCA) +'.png', format = 'png', dpi = 300)


## Quiet
fig, axx = plt.subplots(2, 1, figsize=(12, 5*2))
plt.subplots_adjust(hspace=0.8)

im = axx[0].pcolormesh(spdd_save_quiet['wavsoln'], spdd_save_quiet['phases'], 
               datacube_quiet_det) ## Only one order 
fig.colorbar(im, ax=axx[0])
im = axx[1].pcolormesh(spdd_save_quiet['wavsoln'], spdd_save_quiet['phases'], 
datacube_quiet_det_no_pca) ## Only one order 
fig.colorbar(im, ax=axx[1])

axx[1].set_xlabel('Wavelength [nm]')
axx[0].set_ylabel('Phases')
axx[1].set_ylabel('Phases')

axx[0].set_title(f'Datacube (Quiet) detrended with N_PCA = {str(N_PCA)}')
axx[1].set_title(f'Perfect detrending')

plt.savefig(savedir + 'datacube_quiet_detrended'+ addinfo + '_N_PCA-' + str(N_PCA) +'.png', format = 'png', dpi = 300)



print('Done, exiting without Kp-Vsys calculation...')
exit()

##### Compute the CCF maps 
### For PCA detrended maps 
ccf_DA_MA = ccf.compute_logL_map_per_order(datacube_active_det, modelcube_Fp, 
                                                  modelcube_Fs_active,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DA_MQ = ccf.compute_logL_map_per_order(datacube_active_det, modelcube_Fp, 
                                                  modelcube_Fs_quiet,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DQ_MQ = ccf.compute_logL_map_per_order(datacube_quiet_det, modelcube_Fp, 
                                                  modelcube_Fs_quiet,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DA_MA = np.array(ccf_DA_MA)
ccf_DA_MQ = np.array(ccf_DA_MQ)
ccf_DQ_MQ = np.array(ccf_DQ_MQ)

### For No PCA detrended maps 
ccf_DA_MA_no_pca = ccf.compute_logL_map_per_order(datacube_active_det_no_pca, modelcube_Fp, 
                                                  modelcube_Fs_active,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DA_MQ_no_pca = ccf.compute_logL_map_per_order(datacube_active_det_no_pca, modelcube_Fp, 
                                                  modelcube_Fs_quiet,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DQ_MQ_no_pca = ccf.compute_logL_map_per_order(datacube_quiet_det_no_pca, modelcube_Fp, 
                                                  modelcube_Fs_quiet,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_DA_MA_no_pca = np.array(ccf_DA_MA_no_pca)
ccf_DA_MQ_no_pca = np.array(ccf_DA_MQ_no_pca)
ccf_DQ_MQ_no_pca = np.array(ccf_DQ_MQ_no_pca)

print("Computed Kp-Vsys map, plotting it now...")
titles = ['data-active_model-active',
          'data-active_model-quiet',
          'data-quiet_model-quiet'
          ]
ccf_list = [
    [ccf_DA_MA, ccf_DA_MA_no_pca],
    [ccf_DA_MQ, ccf_DA_MQ_no_pca],
    [ccf_DQ_MQ, ccf_DQ_MQ_no_pca]
]
for ii in range(3):
    fig, axx = plt.subplots(2, 1, figsize=(12, 5*2))
    plt.subplots_adjust(hspace=0.8)
    
    im = axx[0].pcolormesh(Vsys_range, Kp_range, ccf_list[ii][0])
    fig.colorbar(im, ax=axx[0])

    im = axx[1].pcolormesh(Vsys_range, Kp_range, ccf_list[ii][1])
    fig.colorbar(im, ax=axx[1])

    axx[0].axhline(y = 154, color = 'w')
    axx[0].axvline(x=-15., color = 'w')
    axx[1].axhline(y = 154, color = 'w')
    axx[1].axvline(x=-15., color = 'w')

    axx[0].set_xlabel('Vsys')
    axx[0].set_ylabel('Kp')
    axx[1].set_xlabel('Vsys')
    axx[1].set_ylabel('Kp')

    axx[0].set_title(titles[ii] + ' PCA')
    axx[1].set_title(titles[ii] + ' no PCA')

    plt.savefig(savedir + f'KpVsys_{titles[ii]}'+addinfo+'_N_PCA-' + str(N_PCA) +'_and_no_PCA.png', format = 'png', dpi = 300)














# plt.figure(figsize = (15,15))
# plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_active_model - ccf_active_data_quiet_model)
# plt.axhline(y = 154, color = 'w')
# plt.axvline(x=-15., color = 'w')
# plt.colorbar()
# plt.xlabel('Vsys')
# plt.ylabel('Kp')
# plt.title('Active:Data, Active:Model \n - Active:Data, Quiet:Model')
# plt.savefig(savedir + 'Delta_KpVsys_active_active_minus_active_quiet_'+addinfo+ '_N_PCA-' + str(N_PCA) +'.png', format = 'png', dpi = 300)


# dTspot = '-1100.0'
# spot_size = '0.2'
# long0 = '0.0'
# vsini = '3.5'
# instrument = 'crires'
# waverange = ['2280.0', '2330.0']

# dTspot = '-1100.0'
# spot_size = '0.2'
# long0 = '0.0'
# vsini = '20.0'
# instrument = 'crires'
# waverange = ['2280.0', '2330.0']

# addinfo = '_tell_with_tell_modulated_noise_SNR-' + str(SNR)
# dir_name = 'MAIN_emission_eq-chem-ER_' + f'dTspot-{dTspot}_spot_size-{spot_size}_long-{long0}_vsini-{vsini}_{instrument}_{waverange[0]}-{waverange[1]}-nm'
# dir_name = 'MAIN_emission_eq-chem-ER_' + f'dTspot-{dTspot}_spot_size-{spot_size}_long-{long0}_vsini-{vsini}_{instrument}_{waverange[0]}-{waverange[1]}-nm'
