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

savedir = '/home/astro/phsprd/stellar_activity_retrievals/data/emission/'
### Load the simulated data 
results_root = '/home/astro/phsprd/code/plastar/examples/emission/results/'

# spot_size = '0.2'
# long = '45'
# vsini = '20'
# date = '06-11-2025T01-43-40'

# spot_size = '0.2'
# long = '0'
# vsini = '3'
# date = '06-11-2025T14-21-45'

spot_size = '0.2'
long = '0'
vsini = '10'
date = '13-11-2025T15-23-41'

# spot_size = '0.2'
# long = '0'
# vsini = '20'
# date = '06-11-2025T14-25-34'

# spot_size = '0.2'
# long = '45'
# vsini = '3'
# date = '06-11-2025T01-47-33'

SNR = 1000
addinfo = '_only_tell_modulated_noise_SNR-' + str(SNR)
# addinfo = '_tell_with_tell_modulated_noise_SNR-' + str(SNR)
dir_name = f'MAIN_emission_eq-chem-ER_dTspot--1100_spot_size-{spot_size}_long-{long}_vsini-{vsini}_CRIRES_K-band_2280-2330-nm_{date}'
results_path = results_root + dir_name + '/'

savedir = savedir + dir_name + '/'
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()
data_wavsoln = spdd['wavsoln']
phases = spdd['phases']
berv = spdd['berv']
# datacube = spdd['datacube']
datacube = spdd['F_star'] + spdd['F_planet']
### Add noise to the datacube and convolve to instrumental resolution 


### Compute a vector of PWV (at par with mean Paranal values), and compute a time series of skycalc transmission spectrum.
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

pwv_vector = utils.get_random_pwv(tExp = config_dd_simulation['time_step'], size = datacube.shape[0])

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
    
telluric_wave = wave
telluric_transmission = np.array(telluric_transmission)


plt.figure(figsize = (18,10))
for iexp in range(5):
    plt.plot(telluric_wave, telluric_transmission[iexp,:], label = 'Exposure number: ' + str(iexp))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Telluric transmission')
plt.savefig(savedir + 'telluric_sequence_1D.png', format = 'png', dpi = 300)


plt.figure(figsize = (18,10))
plt.pcolormesh(telluric_wave.value, phases, telluric_transmission)
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Telluric Transmission ESO SkyCalc')
plt.savefig(savedir + 'telluric_sequence.png', format = 'png', dpi = 300)


for ip in range(datacube.shape[0]):
    datacube[ip,:] = utils.convolve_spectra_to_instrument_resolution(instrument_resolution = 100000, 
                                                               model_resolution = 200000, 
                                                               model_spec_orig = datacube[ip,:])

    noise = np.random.normal(0., abs(datacube[ip,:] * telluric_transmission[ip,:]/SNR ) )
    datacube[ip,:] = datacube[ip,:] + noise

    # noise = np.random.normal(0., abs(datacube[ip,:] * telluric_transmission[ip,:]/SNR ) )
    # datacube[ip,:] = datacube[ip,:]*telluric_transmission[ip,:] + noise
    
spdd_save = {}
spdd_save['spdatacube'] = datacube[np.newaxis,:,:]
spdd_save['SNR'] = SNR 
spdd_save['phases'] = phases
spdd_save['time'] = phases
spdd_save['bary_RV'] = berv
spdd_save['wavsoln'] = data_wavsoln[np.newaxis,:]
spdd_save['file_name'] = None
np.save(savedir + 'spdd' + addinfo + '.npy', spdd_save)

####### Plot the noisy data in the savedir 
plt.figure(figsize = (18,10))
plt.pcolormesh(spdd_save['wavsoln'], spdd_save['phases'], 
               spdd_save['spdatacube'][0,:,:]) ## Only one order 
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Datacube with telluric modulated noise')
plt.savefig(savedir + 'datacube_with_noise'+addinfo+'.png', format = 'png', dpi = 300)

####### Also save the stellar model 
star_planet_model = {}
star_planet_model['F_star_active_orig'] = spdd['F_star_orig'] ## Not Doppler shifted 
star_planet_model['F_star_quiet_orig'] = spdd['F_star_quiet_orig'] ## Not Doppler shifted 
star_planet_model['F_planet_orig'] =  spdd['F_planet_orig'] ## Not Doppler shifted 
star_planet_model['wavsoln'] = spdd['wavsoln_orig']
np.save(savedir + 'star_planet_model.npy', star_planet_model)

fig, ax = plt.subplots(2,1,figsize = (10,18))
ax[0].plot(star_planet_model['wavsoln'], star_planet_model['F_star_active_orig'][10,:], label = 'Active')
ax[0].plot(star_planet_model['wavsoln'], star_planet_model['F_star_quiet_orig'][10,:], label = 'Quiet')
ax[1].plot(star_planet_model['wavsoln'], star_planet_model['F_star_active_orig'][10,:] / star_planet_model['F_star_quiet_orig'][10,:])

ax[0].set_xlabel('Wavelength [nm]')
ax[0].set_ylabel('Flux')
ax[1].set_ylabel('Flux')

plt.savefig(savedir + 'Fstar_active_and_quiet.png', format = 'png', dpi = 300)



###### Also compute a quick Kp-Vsys map with the injected model and save it 
Vsys_range = jnp.linspace(-50., 0., 50)
Kp_range = jnp.linspace(120, 180, 60)
model_wavsoln = jnp.array(spdd['wavsoln_orig'])
data_wavsoln = jnp.array(spdd_save['wavsoln'][0,:])
phases = jnp.array(spdd_save['phases'])
berv = jnp.array(spdd_save['bary_RV'])
modelcube_Fp = jnp.array(spdd['F_planet_orig'][np.newaxis,:] * np.ones((len(phases), len(model_wavsoln) )) )
modelcube_Fs = jnp.array(spdd['F_star_orig'])
modelcube_Fs_quiet = jnp.array(spdd['F_star_quiet_orig'])
### Do telluric and stellar correction with one PCA component
datacube = ccf.get_PCA_detrended_datacube(datacube = datacube, nc = 1)

plt.figure(figsize = (18,10))
plt.pcolormesh(spdd_save['wavsoln'], spdd_save['phases'], 
               datacube) ## Only one order 
plt.colorbar()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Phases')
plt.title('Datacube detrended with N_PCA = 1')
plt.savefig(savedir + 'datacube_detrended'+addinfo+'.png', format = 'png', dpi = 300)

ccf_active_data_active_model = ccf.compute_logL_map_per_order(datacube, modelcube_Fp, modelcube_Fs,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_active_data_quiet_model = ccf.compute_logL_map_per_order(datacube, modelcube_Fp, modelcube_Fs_quiet,
                                                              Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)
ccf_active_data_active_model = np.array(ccf_active_data_active_model)
ccf_active_data_quiet_model = np.array(ccf_active_data_quiet_model)


print("Computed Kp-Vsys map, plotting it now...")
plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_active_model)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-15., color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.title('Data: Fs_active * (1 + Fp / Fs_active) \n Model: Fp / Fs_active')
plt.savefig(savedir + 'KpVsys_active_data_active_model'+addinfo+'.png', format = 'png', dpi = 300)

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_active_model - ccf_active_data_quiet_model)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-15., color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.title('Active:Data, Active:Model \n - Active:Data, Quiet:Model')
plt.savefig(savedir + 'Delta_KpVsys_active_active_minus_active_quiet_'+addinfo+'.png', format = 'png', dpi = 300)