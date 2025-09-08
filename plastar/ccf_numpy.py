import numpy as np
from numpy import interp
from tqdm import tqdm
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
        data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
        model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_1D)
        return model_shifted
    modelcube_shifted = np.ones(modelcube.shape)
    for i in range(modelcube.shape[0]):
        modelcube_shifted[i,:] = doppler_shift_model1D(modelcube[i,:], RV[i], model_wavsoln, data_wavsoln)
    return modelcube_shifted

def logL_per_KpVsys(Kp, Vsys, datacube, modelcube, model_wavsoln, data_wavsoln, phases, berv):
    RV = compute_RV(Kp, Vsys, phases, berv)
    modelcube_shifted = doppler_shift_modelcube(modelcube, RV, model_wavsoln, data_wavsoln)
    logL_values = np.zeros((len(phases),))
    for i in range(len(phases)):
        logL_values[i] = get_R(datacube[i,:], modelcube_shifted[i,:])
    return np.sum(logL_values)

def compute_logL_map_per_order(datacube, modelcube, Kp_range,
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv):
    
    logL_map = np.zeros( ( len(Kp_range), len(Vsys_range) ) )
    Kp_grid, Vsys_grid = np.meshgrid(Kp_range, Vsys_range,indexing='ij')
    for iKp in tqdm(range(len(Kp_range))):
        for iVsys in range(len(Vsys_range)):
            logL_map[iKp, iVsys] = logL_per_KpVsys(Kp_grid[iKp,iVsys], Vsys_grid[iKp,iVsys], datacube, modelcube, model_wavsoln, data_wavsoln, phases, berv)
            
    return logL_map