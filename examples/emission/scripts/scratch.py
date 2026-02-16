from plastar import ccf
from plastar import ccf_numpy
import jax.numpy as jnp

datacube = jnp.ones((40, 2048))
modelcube = jnp.ones((40, 2048))
Kp_range = jnp.linspace(0, 50, 50)
model_wavsoln = jnp.linspace(0,2048,2048)
data_wavsoln = jnp.linspace(0,2048,2048)
Vsys_range = jnp.linspace(0, 50, 50)
phases = jnp.linspace(0, 40, 40)
berv = jnp.linspace(0, 40, 40)

import time

start = time.time()
out_jax = ccf.compute_logL_map_per_order(datacube, modelcube, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)
print(out_jax.shape)
end = time.time()
print("Time taken for jax:",(end-start), "seconds")

start = time.time()

import numpy as np
datacube = np.ones((40, 2048))
modelcube = np.ones((40, 2048))
Kp_range = np.linspace(0, 50, 50)
model_wavsoln = np.linspace(0,2048,2048)
data_wavsoln = np.linspace(0,2048,2048)
Vsys_range = np.linspace(0, 50, 50)
phases = np.linspace(0, 40, 40)
berv = np.linspace(0, 40, 40)

out_numpy = ccf_numpy.compute_logL_map_per_order(datacube, modelcube, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)
print(out_numpy.shape)
end = time.time()
print("Time taken for numpy:",(end-start), "seconds")
import pdb
pdb.set_trace()
