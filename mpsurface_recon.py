#!/usr/bin/env python3
"""Script for turning 3D magnetopause data into a smooth connected zone
"""

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plot
import scipy
from scipy import interpolate as interp
import pandas as pd
import tecplot as tp

START = time.time()

def yz_slicer(zone,x_min, x_max, n_slice, n_theta):
    """Function loops through x position to create 2D closed curves in YZ
    Inputs
        x_min
        x_max
        n_slice- must be >= 2
        n_theta
    Outputs
        mesh- mesh of X,Y,Z points in a pandas DataFrame object
    """
    dx = (x_max-x_min)/(2*(n_slice-1))
    for i in np.linspace(x_min, x_max, n_slice):
        zone_temp = zone[zone['X [R]'] < i+dx]
        zone_temp = zone_temp[zone_temp['X [R]'] > i-dx]
        #tck, u = interp.splprep([zone_temp['Y [R]'],
        #                         zone_temp['Z [R]']],
        #                         s=2, per=True)
        #y_new, z_new = interp.splev(u, tck)
        plot.scatter(zone_temp['Y [R]'],zone_temp['Z [R]'])
        plot.show()
        #print(zone_temp,'\n')



#main program
if __name__ == "__main__":
    #Read in data values and sort by X position
    ZONE = pd.read_csv('mp_points.csv')
    ZONE = ZONE.drop(columns=['Unnamed: 3'])
    ZONE = ZONE.sort_values(by=['X [R]'])
    yz_slicer(ZONE,-20, 10, 50, 50)

    print("___ %s seconds ___" % (time.time() - START))
