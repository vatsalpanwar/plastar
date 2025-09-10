from plastar import ccf
from plastar import ccf_numpy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# datacube = jnp.ones((40, 2048))
# modelcube = jnp.ones((40, 2048))
# Kp_range = jnp.linspace(0, 50, 50)
# model_wavsoln = jnp.linspace(0,2048,2048)
# data_wavsoln = jnp.linspace(0,2048,2048)
# Vsys_range = jnp.linspace(0, 50, 50)
# phases = jnp.linspace(0, 40, 40)
# berv = jnp.linspace(0, 40, 40)

### Load the simulated data 
results_path = '/home/astro/phsprd/code/plastar/examples/emission/results/TEST_emission_dTspot--1100_spot_size-0.2_long-0_09-09-2025T16-58-54/'
spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()

Vsys_range = jnp.linspace(-50, 50, 100)
Kp_range = jnp.linspace(150, 250, 100)
modelcube = jnp.array(spdd['Fp_by_Fs'])
model_wavsoln = jnp.array(spdd['wavsoln'])
data_wavsoln = jnp.array(spdd['wavsoln'])
phases = jnp.array(spdd['phases'])
berv = jnp.array(spdd['berv'])
datacube = spdd['datacube']

## Mean subtract each row of datacube 


import time

start = time.time()
out_jax = ccf.compute_logL_map_per_order(datacube, modelcube, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)
print(out_jax.shape)
end = time.time()
print("Time taken for jax:",(end-start), "seconds")

import pdb; pdb.set_trace()

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, out_jax)
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.savefig(results_path + 'KpVsys.png', format = 'png', dpi = 300)
