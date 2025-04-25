import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import yaml
import sys
sys.path.append('/Users/vatsalpanwar/sterster/sterster/')
from plastar import grid

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

################################################################
################################################################
now = datetime.datetime.now()
# Format the date and time
d1 = now.strftime("%d-%m-%YT%H-%M-%S")
print('Date tag for this run which will be used to save the results: ', d1)

################################################################
################################################################
"""Read in the config file."""
################################################################
################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=False)

args = vars(parser.parse_args())

config_file_path = args['config_file_path']
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)
infostring = config_dd['infostring'] + d1
savedir = config_dd['simulations_savedir'] + infostring + '/'

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

star_dict = config_dd['star']

star = grid.StellarGrid(star_dict = star_dict, include_planet = False, include_spot = False)
star.construct_grid()