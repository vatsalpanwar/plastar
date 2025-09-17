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
results_path = '/home/astro/phsprd/code/plastar/examples/emission/results/TEST_emission_dTspot--1100_spot_size-0.2_long-0_16-09-2025T12-07-41/'
spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()

Vsys_range = jnp.linspace(-25, 25, 100)
Kp_range = jnp.linspace(100, 200, 100)
modelcube = jnp.array(spdd['Fp_by_Fs'])
modelcube_quiet = jnp.array(spdd['Fp_by_Fs_quiet'])
model_wavsoln = jnp.array(spdd['wavsoln'])
data_wavsoln = jnp.array(spdd['wavsoln'])
phases = jnp.array(spdd['phases'])
berv = jnp.array(spdd['berv'])
datacube = spdd['datacube']

## Divide out the first exposure as a proxy for stellar correction 
## May not work for emission as the first exposure also includes planet
# norm_factor = datacube[0,:].copy()
# norm_factor = np.median(datacube[0,:])
norm
for ip in range(datacube.shape[0]):
    datacube[ip,:] = datacube[ip,:] / norm_factor

## Mean subtract each row of datacube 
for ip in range(datacube.shape[0]):
    datacube[ip,:] = datacube[ip,:] - np.median(datacube[ip,:])
datacube = jnp.array(datacube)

import time

start = time.time()
out_jax = ccf.compute_logL_map_per_order(datacube, modelcube, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)
out_jax_modelcube_quiet = ccf.compute_logL_map_per_order(datacube, modelcube_quiet, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

print(out_jax.shape)
end = time.time()
print("Time taken for jax:",(end-start), "seconds")


plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, out_jax)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-2.2, color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.savefig(results_path + 'KpVsys_data-active_model-active.png', format = 'png', dpi = 300)

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, out_jax_modelcube_quiet)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-2.2, color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.savefig(results_path + 'KpVsys_data-active_model-quiet.png', format = 'png', dpi = 300)

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, out_jax - out_jax_modelcube_quiet)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-2.2, color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.savefig(results_path + 'Delta_KpVsys_active_minus_quiet.png', format = 'png', dpi = 300)
