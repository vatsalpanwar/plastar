import numpy as np

def get_gaussian_kernel(size = None, sigma = None):
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    # x = np.linspace(-int(span/2), int(span/2)+1, num = 200)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize the kernel

def convolve_spectra_to_instrument_resolution(instrument_resolution = None, model_resolution = None, model_spec_orig = None):
    
    delwav_by_wav = 1/instrument_resolution # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
    delwav_by_wav_model = 1./model_resolution   ### np.diff(model_wav)/model_wav[1:]
    
    FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )
    gauss_kernel = get_gaussian_kernel(size = sig*10, sigma = sig)
    # model_spec = np.convolve(model_spec_orig)           
    model_spec = np.convolve(model_spec_orig, gauss_kernel, mode = 'same')
    return model_spec
    

savedir = '/home/astro/phsprd/stellar_activity_retrievals/data/emission/'
### Load the simulated data 
results_root = '/home/astro/phsprd/code/plastar/examples/emission/results/'
#### J band 
# results_path = '/home/astro/phsprd/code/plastar/examples/emission/results/TEST_emission_dTspot--1100_spot_size-0.2_long-J_band_1120-1125-nm_07-10-2025T14-07-06/'
#### K band 
dir_name = 'TEST_rev_emission_dTspot--1100_spot_size-0.2_long-K_band_2440-2445-nm_21-10-2025T12-09-10' 
results_path = results_root + dir_name + '/'

spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()
data_wavsoln = spdd['wavsoln']
phases = spdd['phases']
berv = spdd['berv']
# datacube = spdd['datacube']
datacube = spdd['F_star'] + spdd['F_planet']
### Add noise to the datacube and convolve to instrumental resolution 
SNR = 100
for ip in range(datacube.shape[0]):
    
    datacube[ip,:] = convolve_spectra_to_instrument_resolution(instrument_resolution = 100000, 
                                                               model_resolution = 200000, 
                                                               model_spec_orig = datacube[ip,:])
    noise = np.random.normal(0., abs(datacube[ip,:]/SNR ))
    datacube[ip,:] = datacube[ip,:] + noise
    
spdd_save = {}
spdd_save['spdatacube'] = datacube[np.newaxis,:,:]
spdd_save['phases'] = phases
spdd_save['time'] = phases
spdd_save['bary_RV'] = berv
spdd_save['wavsoln'] = data_wavsoln[np.newaxis,:]
spdd_save['file_name'] = None
np.save(savedir + 'spdd_' + dir_name + '.npy', spdd_save)

star_planet_model = {}
star_planet_model['F_star_orig'] = spdd['F_star_orig']
star_planet_model['F_star_quiet_orig'] = spdd['F_star_quiet_orig']
star_planet_model['wavsoln'] = spdd['wavsoln']

np.save(savedir + 'star_planet_model_' + dir_name + '.npy', star_planet_model)
