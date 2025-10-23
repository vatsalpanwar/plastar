import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from functools import partial 
# from splinex import BSpline
from jax.numpy import interp
import jax

@jit
def get_R(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    data = data - jnp.mean(data)
    model = model - jnp.mean(model)
    # breakpoint()
    R = (1. / len(data)) * jnp.dot(data, model)  ## R in Brogi and Line
    return R

@jit
def get_C(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    R = get_R(data, model)
    C = R / jnp.sqrt(jnp.var(data) * jnp.var(model))  ## C in Brogi and Line
    return C

@jit
def get_logL(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """

    R = get_R(data, model)
    data = data - jnp.median(data)
    model = model = jnp.median(model)
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


    