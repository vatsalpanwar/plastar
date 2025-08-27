"""
Compute the spectrum from the planet alone, without the contribution from the star. 
Defines the Planet class, which takes in the planet properties from an input config file,
and generates either the transmission radius (Rp(wavelength)) or the emission flux (Fp(wavelength)). 
Both of these outputs can be used by StellarGrid object to compute the 
observed transmission spectrum (Rp/Rs)**2 (wavelength), or emission spectrum Fp/Fs (wavelength). 
 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
import copy
import genesis

class PlanetAtmosphere():
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the PlanetAtmosphere class with the provided arguments, 
        to compute the spectrum of the planet (in transmission: Rp, in emission: Fp).
        
        Parameters:
        - *args: Positional arguments.
        - **kwargs: Keyword arguments.
        """
        self.planet_dict = kwargs.pop('planet_dict')