import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from functools import partial 
# from splinex import BSpline
from jax.numpy import interp
import jax
import numpy as np

@jit
def get_R(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    # breakpoint()
    R = (1. / len(data)) * jnp.dot(data, model)  ## R in Brogi and Line
    return R

@jit
def get_C(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    data = data - jnp.mean(data)
    model = model - jnp.mean(model)
    R = get_R(data, model)
    C = R / jnp.sqrt(jnp.var(data) * jnp.var(model))  ## C in Brogi and Line
    return C

@jit
def get_logL(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    data = data - jnp.mean(data)
    model = model = jnp.mean(model)
    R = get_R(data, model)
    logL = (-len(data)/2) * jnp.log(jnp.var(data) + jnp.var(model) - 2.*R)
    return logL

@jit
def doppler_shift_wavsoln(velocity: float, wavsoln: ArrayLike) -> ArrayLike:
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

@jit
def compute_RV(Kp: float, Vsys: float, phases: ArrayLike, berv: ArrayLike) -> ArrayLike:
    return Kp * jnp.sin(2. * jnp.pi * phases) + Vsys + berv


@jit
def doppler_shift_modelcube(modelcube: ArrayLike, RV: ArrayLike, model_wavsoln: ArrayLike, data_wavsoln: ArrayLike) -> ArrayLike:
    def doppler_shift_model1D(model_1D, RV_val, model_wavsoln, data_wavsoln):
        data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
        model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_1D)
        return model_shifted
    return jax.vmap(doppler_shift_model1D, in_axes = (0,0,None,None))(modelcube, RV, model_wavsoln, data_wavsoln)

# @jit
# def logL_per_KpVsys(Kp, Vsys, datacube, modelcube, model_wavsoln, data_wavsoln, phases, berv):
#     RV = compute_RV(Kp, Vsys, phases, berv)
#     modelcube_shifted = doppler_shift_modelcube(modelcube, RV, model_wavsoln, data_wavsoln)
#     return jnp.sum(jax.vmap(get_logL, in_axes=(0, 0))(datacube, modelcube_shifted))

@jit
def logL_per_KpVsys(Kp, Vsys, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
    RV_p = compute_RV(Kp, Vsys, phases, berv)
    RV_s = compute_RV(0, Vsys, phases, berv)
    
    modelcube_shifted_Fp = doppler_shift_modelcube(modelcube_Fp, RV_p, model_wavsoln, data_wavsoln)
    modelcube_shifted_Fs = doppler_shift_modelcube(modelcube_Fs, RV_s, model_wavsoln, data_wavsoln)
    modelcube_shifted = modelcube_shifted_Fp/modelcube_shifted_Fs
    
    # return jnp.sum(jax.vmap(get_logL, in_axes=(0, 0))(datacube, modelcube_shifted))
    return jnp.sum(jax.vmap(get_C, in_axes=(0, 0))(datacube, modelcube_shifted))

# jax.config.update("jax_disable_jit", True)


# jax.config.update("jax_disable_jit", True)

@jit
def compute_logL_map_per_order(datacube: ArrayLike, modelcube_Fp: ArrayLike, modelcube_Fs: ArrayLike,
                               Kp_range: ArrayLike, 
                           model_wavsoln: ArrayLike, data_wavsoln: ArrayLike,
                           Vsys_range: ArrayLike, phases: ArrayLike, berv: ArrayLike) -> ArrayLike:
    
    def vectorize_1D_row(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
        # jax.debug.print("Value of Kp_row: {Kp_row}", Kp_row = Kp_row)
        # jax.debug.print("Value of Vsys_row: {Vsys_row}", Vsys_row = Vsys_row)
        return jax.vmap(logL_per_KpVsys, in_axes=(0, 0, None, None, None, None, None, None, None))(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)

    Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range, indexing='ij')
    # breakpoint()
    
    vectorized_grid_func = jax.vmap(vectorize_1D_row, in_axes=(0, 0, None, None,None, None, None, None, None))

    return vectorized_grid_func(Kp_grid, Vsys_grid, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)

####### Detrending functions
def standardise_data(datacube=None):
    """
    Standardise datacube before running the PCA on it.
    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already; dimensions should be [time,wavelength].

    :return: Standardised array of datacube, in the same format as the original datacube.
    """
    nf, nx = datacube.shape
    fStd = datacube.copy()
    for i in range(nx):
        fStd[:,i] -= np.mean(fStd[:,i])
        # This is the biased stdev (normalised by nx rather than nx-1)
        # It needs changing to match CORRELATE.pro
        fStd[:,i] /= np.std(fStd[:,i]) + 1e-100
    fStd = np.nan_to_num(fStd,0.) ## This is in case a whole spectral channel was set to zero pre-standardisation, which can lead to some spectral channels being nans. 
    return fStd

def get_eigenvectors(datacube=None,nc=None):
    nf, nx = datacube.shape # nf : number of frames, nx : number of wavelength channels
    xMat = np.ones((nf,nc+1)) # The second dimension is nc+1 because besides the nc eigenvectors
                              # you want to have the first component to be 1 (required for the multi-linear regression).
    u, s, vh = np.linalg.svd(datacube, full_matrices=False) # u is the matrix of eigenvectors (shape : (nf, nx) ),
                                                       # s is a vector of eigenvalues. vh is the Unitary array.
    xMat[:,1:] = u[:,0:nc] # Take only nc eigenvectors.
    return xMat

def linear_regression(X=None,Y=None):
    """
    Calculate the multi-variate linear regression fit between the matrix of
    eigenvectors X [nf, nc] and the observed spectral datacube Y [nf, nx].
    :param X: array_like
     Matrix of eigenvectors, shape [nf, nc] i.e. [time, number of components]

    :param Y: array_like
    Datacube, shape [nf, nx] i.e. [time, wavelength]

    :return: Calculated PCA fit to the datacube using the input eigenvector matrix.
    """

    XT = X.T
    term1 = np.linalg.inv(jnp.dot(XT,X))
    term2 = np.dot(term1,XT)
    beta = np.dot(term2,Y)
    return np.dot(X,beta)

def get_PCA_detrended_datacube(datacube = None, nc = None):
    nspec, nwav = datacube.shape[0], datacube.shape[1]
    
    datacube_standard = standardise_data(datacube)
    
    fStd = datacube_standard.copy()
    pca_eigenvectors = get_eigenvectors(fStd, nc=nc)
    
    datacube_fit = linear_regression(X=pca_eigenvectors, Y=datacube)
    datacube_detrended = datacube/(datacube_fit+1e-100) - 1.
    
    return datacube_detrended

def get_perfect_detrended_datacube(datacube = None, correction_cube = None):

    datacube_detrended = datacube/(correction_cube) - 1.
    
    return datacube_detrended
    

# @jit
# def compute_logL_map_per_order(datacube: ArrayLike, modelcube_Fp: ArrayLike, modelcube_Fs: ArrayLike,
#                                Kp_range: ArrayLike, 
#                            model_wavsoln: ArrayLike, data_wavsoln: ArrayLike,
#                            Vsys_range: ArrayLike, phases: ArrayLike, berv: ArrayLike) -> ArrayLike:
    
#     def vectorize_1D_row(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
#         # jax.debug.print("Value of Kp_row: {Kp_row}", Kp_row = Kp_row)
#         # jax.debug.print("Value of Vsys_row: {Vsys_row}", Vsys_row = Vsys_row)
#         return jax.vmap(logL_per_KpVsys, in_axes=(0, 0, None, None, None, None, None, None, None))(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)

#     Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range, indexing='ij')
#     breakpoint()
#     # Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range)
#     vectorized_grid_func = jax.vmap(vectorize_1D_row, in_axes=(0, 0, None, None,None, None, None, None, None))
#     # jax.debug.print("Value of Kp_grid: {Kp_grid}", Kp_grid = Kp_grid)
#     # jax.debug.print("Value of Vsys_grid: {Vsys_grid}", Vsys_grid = Vsys_grid)
#     return vectorized_grid_func(Kp_grid, Vsys_grid, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)


    