import numpy as np
from numpy import interp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate

def get_R(data, model):
    """
    """
    R = (1. / len(data)) * np.dot(data, model)  ## R in Brogi and Line
    return R

def get_C(data, model):
    """
    """
    R = get_R(data, model)
    C = R / np.sqrt(np.var(data) * np.var(model))  ## C in Brogi and Line
    return C

def get_logL(data, model):
    """
    """
    R = get_R(data, model)
    logL = (-len(data)/2) * np.log(np.var(data) + np.var(model) - 2.*R)
    return logL

def doppler_shift_wavsoln(velocity, wavsoln):
    """
    This function applies a Doppler shift to a 1D array of wavelengths.
    wav_obs = wav_orig (1. + velocity/c) where if velocity is positive it corresponds to a redshift
    (i.e. source moving away from you, so wavelength solution gets shifted towards positive direction) and vice versa
    for a negative velocity corresponding to blueshift i.e. source moving towards you.

    :param wavsoln: 1D array if wavelengths, ideally in nanometers.
    :type wavsoln: array_like
    

    :param velocity: Float value of the velocity of the source, in km/s. Note that the astropy value of speed of light (c) is
    in m/s.
    :type velocity: float64

    :return: Doppler shifted wavelength solution.
    :rtype: array_like
    """
    wavsoln_doppler = wavsoln * (1. + (1000. * velocity) / 299792458.0)
    return wavsoln_doppler

def compute_RV(Kp: float, Vsys: float, phases, berv):
    return Kp * np.sin(2. * np.pi * phases) + Vsys + berv


def doppler_shift_modelcube(modelcube, RV, model_wavsoln, data_wavsoln):
    
    def doppler_shift_model1D(model_1D, RV_val, model_wavsoln, data_wavsoln):
        model_spl = interpolate.make_interp_spline(model_wavsoln, model_1D)
        data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
        model_shifted = model_spl(data_wavsoln_shifted)
        # model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_1D)
        return model_shifted
    modelcube_shifted = np.ones( (modelcube.shape[0], len(data_wavsoln)) )
    for i in range(modelcube.shape[0]):
        modelcube_shifted[i,:] = doppler_shift_model1D(modelcube[i,:], RV[i], model_wavsoln, data_wavsoln)
    return modelcube_shifted

def doppler_shift_modelcube_fft(modelcube, RV, model_wavsoln, data_wavsoln):
    
    def doppler_shift_model1D(model_1D, RV_val, model_wavsoln, data_wavsoln):
        N_wavelength = len(model_1D)        
        model_ft = np.fft.fft(model_1D, axis = 1)
        k = np.fft.fftfreq(N_wavelength).reshape(1, -1)
        c = 299792458.0
        w_shift = RV_val/c
        dw = model_wavsoln[1] - model_wavsoln[0]
        shift = w_shift[:,None] * model_wavsoln/dw
        phase_shift = np.exp(-2j * np.pi * k * shift)
        model_shifted = np.fft.ifft(model_ft * phase_shift)
        model_shifted = np.real(model_shifted)
        
        ### Interpolate it back to data wavelength solution grid 
        model_spl = interpolate.make_interp_spline(model_wavsoln, model_shifted)
        model_shifted_fin = model_spl(data_wavsoln)
        return model_shifted_fin
    
        # model_spl = interpolate.make_interp_spline(model_wavsoln, model_1D)
        # data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
        # model_shifted = model_spl(data_wavsoln_shifted)
        # # model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_1D)
        # return model_shifted
    
    
    modelcube_shifted = np.ones( (modelcube.shape[0], len(data_wavsoln)) )
    for i in range(modelcube.shape[0]):
        modelcube_shifted[i,:] = doppler_shift_model1D(modelcube[i,:], RV[i], model_wavsoln, data_wavsoln)
    return modelcube_shifted

def logL_per_KpVsys(Kp, Vsys, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
    RV_p = compute_RV(Kp, Vsys, phases, berv)
    RV_s = compute_RV(0, Vsys, phases, berv) ## Can add Ks here with a 0.5 offset to the phase 
    
    modelcube_shifted_Fp = doppler_shift_modelcube(modelcube_Fp, RV_p, model_wavsoln, data_wavsoln)
    modelcube_shifted_Fs = doppler_shift_modelcube(modelcube_Fs, RV_s, model_wavsoln, data_wavsoln)
    modelcube_shifted = modelcube_shifted_Fp/modelcube_shifted_Fs
    
    ########## plot to check 
    # fig, ax = plt.subplots(4,1, figsize = (8,32))
    # im = ax[0].pcolormesh(data_wavsoln, phases, modelcube_shifted_Fp)
    # ax[0].set_title('shifted planet modelcube')
    # cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    
    # im = ax[1].pcolormesh(data_wavsoln, phases, modelcube_shifted_Fs)
    # ax[1].set_title('shifted stellar modelcube')
    # cbar = fig.colorbar(im, ax=ax[1], orientation='vertical')
    
    # im = ax[2].pcolormesh(data_wavsoln, phases, modelcube_shifted)
    # ax[2].set_title('shifted Fp/Fs modelcube')
    # cbar = fig.colorbar(im, ax=ax[2], orientation='vertical')
    
    # im = ax[3].pcolormesh(data_wavsoln, phases, datacube)
    # ax[3].set_title('datacube')
    # cbar = fig.colorbar(im, ax=ax[3], orientation='vertical')
    
    # # plt.colorbar()
    # ax[2].set_xlabel('Wavelength [nm]')
    # ax[0].set_ylabel('Phases')
    # ax[1].set_ylabel('Phases')
    # ax[2].set_ylabel('Phases')
    # ax[3].set_ylabel('Phases')
    
    # plt.suptitle('KpVsys' + str(Kp) + ' ' + str(Vsys))
    # plt.savefig(f'/home/astro/phsprd/code/plastar/examples/emission/results/TEST_rev_emission_dTspot--1100_spot_size-0.2_long-K_band_2440-2445-nm_15-10-2025T02-36-01/check_shifted_data_model_cubes.png',
    #                     dpi = 300)
    # import pdb; pdb.set_trace()
    ######################## 
    
    logL_values = np.zeros((len(phases),))
    for i in range(len(phases)):
        modelcube_shifted[i,:] = modelcube_shifted[i,:] - np.mean(modelcube_shifted[i,:])
        # if Kp > 140. and Kp < 180.:
        #     fig, ax = plt.subplots(2,1,figsize = (12,10))
        #     ax[0].plot(data_wavsoln, datacube[i,:], label = 'data')
        #     ax[1].plot(data_wavsoln, modelcube_shifted[i,:], label = 'model')
        #     plt.legend()
        #     plt.suptitle('Kp = ' + str(Kp))
        #     plt.savefig(f'/home/astro/phsprd/code/plastar/examples/emission/results/TEST_rev_emission_dTspot--1100_spot_size-0.2_long-K_band_2440-2445-nm_15-10-2025T02-36-01/check_data_model_{str(Kp)}.png',
        #                 dpi = 300)
        #     plt.close()
        #     import pdb; pdb.set_trace()
        datacube[i,:] = datacube[i,:] - np.mean(datacube[i,:])
        logL_values[i] = get_C(datacube[i,:], modelcube_shifted[i,:])
    return np.sum(logL_values)

def compute_logL_map_per_order(datacube, modelcube_Fp, modelcube_Fs, Kp_range,
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv):
    
    logL_map = np.zeros( ( len(Kp_range), len(Vsys_range) ) )
    Kp_grid, Vsys_grid = np.meshgrid(Kp_range, Vsys_range,indexing='ij')
    for iKp in tqdm(range(len(Kp_range))):
        for iVsys in range(len(Vsys_range)):
            logL_map[iKp, iVsys] = logL_per_KpVsys(Kp_grid[iKp,iVsys], Vsys_grid[iKp,iVsys], 
                                                   datacube, modelcube_Fp, modelcube_Fs,
                                                   model_wavsoln, data_wavsoln, phases, berv)
            
    return logL_map