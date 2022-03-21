#!/usr/bin/env python3
"""module processes observation/simulation ground magnetometer data
"""
import os
import sys
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import swmfpy
from spacepy import coordinates as coord
from spacepy import time as spacetime
from spacepy import pybats as bats



def stationXYZgsm(Lon, Lat, timeinfo, **kwargs):
    """Function converts lon, lat and timing info into gsm coordinates
    """
    pass

if __name__ == "__main__":
    testfile = './mag_grid/mag_grid_e20161110-020100.out'
    from IPython import embed; embed()
    #find files for station locations
    #               simulation data
    #               supermag   data
    #organize data based on time (different options, default 2day event)
    #for each time segment
        #process station locations to XYZgsm based on timing of dipole
        #process data in any way that is desired (please keep simple)
        #initiate tecplot session in either connected or batch mode
        #create earth zone with visual wrapper with the right positioning
    #
    pass
