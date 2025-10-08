from plastar import ccf
from plastar import ccf_numpy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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

# datacube = jnp.ones((40, 2048))
# modelcube = jnp.ones((40, 2048))
# Kp_range = jnp.linspace(0, 50, 50)
# model_wavsoln = jnp.linspace(0,2048,2048)
# data_wavsoln = jnp.linspace(0,2048,2048)
# Vsys_range = jnp.linspace(0, 50, 50)
# phases = jnp.linspace(0, 40, 40)
# berv = jnp.linspace(0, 40, 40)

### Load the simulated data 
#### J band 
# results_path = '/home/astro/phsprd/code/plastar/examples/emission/results/TEST_emission_dTspot--1100_spot_size-0.2_long-J_band_1120-1125-nm_07-10-2025T14-07-06/'
#### K band 
results_path = '/home/astro/phsprd/code/plastar/examples/emission/results/TEST_emission_dTspot--1100_spot_size-0.2_long-K_band_2440-2445-nm_07-10-2025T14-40-18/'
spdd = np.load(results_path + 'spdd.npy', allow_pickle = True).item()

Vsys_range = jnp.linspace(-50, 0, 100)
Kp_range = jnp.linspace(100, 200, 100)
modelcube_active = jnp.array(spdd['Fp_by_Fs'])
modelcube_quiet = jnp.array(spdd['Fp_by_Fs_quiet'])
model_wavsoln = jnp.array(spdd['wavsoln'])
data_wavsoln = jnp.array(spdd['wavsoln'])
phases = jnp.array(spdd['phases'])
berv = jnp.array(spdd['berv'])
datacube_only_star_active = spdd['F_star']
datacube_only_star_quiet = spdd['F_star_quiet']
datacube = spdd['datacube']


## Divide out the median of all exposures as a proxy for stellar correction 
norm_factor_active = np.median(spdd['F_star'], axis = 0)
datacube_active = datacube.copy()
for ip in range(datacube.shape[0]):
    datacube_active[ip,:] = (datacube_active[ip,:] / norm_factor_active) - 1.
datacube_active = jnp.array(datacube_active)

import time
start = time.time()
ccf_active_data_active_model = ccf.compute_logL_map_per_order(datacube_active, modelcube_active, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

ccf_active_data_quiet_model = ccf.compute_logL_map_per_order(datacube_active, modelcube_quiet, Kp_range, 
                           model_wavsoln, data_wavsoln,
                           Vsys_range, phases, berv)

end = time.time()
print("Time taken:",(end-start), "seconds")
ccf_active_data_active_model = np.array(ccf_active_data_active_model)
ccf_active_data_quiet_model = np.array(ccf_active_data_quiet_model)

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_active_model)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-15., color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.title('Data: Fs_active * (1 + Fp / Fs_active) \n Model: Fp / Fs_active')
plt.savefig(results_path + 'KpVsys_active_data_active_model.png', format = 'png', dpi = 300)

plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_quiet_model)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-15., color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.title('Data: Fs_active * (1 + Fp / Fs_active) \n Model: Fp / Fs_quiet')
plt.savefig(results_path + 'KpVsys_active_data_quiet_model.png', format = 'png', dpi = 300)


plt.figure(figsize = (15,15))
plt.pcolormesh(Vsys_range, Kp_range, ccf_active_data_active_model - ccf_active_data_quiet_model)
plt.axhline(y = 154, color = 'w')
plt.axvline(x=-15., color = 'w')
plt.colorbar()
plt.xlabel('Vsys')
plt.ylabel('Kp')
plt.title('Active:Data, Active:Model \n - Active:Data, Quiet:Model')
plt.savefig(results_path + 'Delta_KpVsys_active_active_minus_active_quiet.png', format = 'png', dpi = 300)