#!/usr/bin/env python3
"""Tools for handling magnetic mapping in tecplot
"""
#import logging as log
import os
import sys
import time
import glob
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#interpackage modules
from global_energetics.makevideo import get_time, time_sort
from global_energetics.write_disp import write_to_hdf, display_progress
from global_energetics.extract.tec_tools import (integrate_tecplot,mag2gsm,
                                                    create_stream_zone,
                                                    calc_terminator_zone,
                                                    calc_ocflb_zone,
                                                    get_global_variables)
from global_energetics.extract.equations import (get_dipole_field)
from global_energetics.extract import line_tools
from global_energetics.extract import surface_tools

def reversed_mapping(gmzone,state_var,**kwargs):
    # Convert theta/phi mapping variable into cartesian footpoint values
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    # Pull some data from tecplot-> numpy
    state = gmzone.values(state_var).as_numpy_array()
    theta_1 = gmzone.values('theta_1 *').as_numpy_array()
    phi_1 = gmzone.values('phi_1 *').as_numpy_array()
    x = gmzone.values('X *').as_numpy_array()
    # Make a new set of variables
    if 'daynight' not in gmzone.dataset.variable_names:
        gmzone.dataset.add_variable('daynight')
    if kwargs.get('debug',False):
        if 'mapID' not in gmzone.dataset.variable_names:
            gmzone.dataset.add_variable('mapID')
        mapID = gmzone.values('mapID').as_numpy_array()
    daynight = gmzone.values('daynight').as_numpy_array()
    theta_bins = np.linspace(0,90,10)
    phi_bins = np.linspace(0,360,37)
    k=0
    for i,th in enumerate(theta_bins[1::]):
        for j,ph in enumerate(phi_bins[1::]):
            dayside,nightside,split = 0,0,False
            inbins = ((state==1)&
                      (theta_1<th)&
                      (theta_1>theta_bins[i-1])&
                      (phi_1<ph)&
                      (phi_1>phi_bins[j-1]))
            if any(inbins):
                k+=1
                if kwargs.get('debug',False):
                    mapID[inbins] = k
                if x[inbins].mean()>0:
                    dayside = 1
                if x[inbins].mean()<0:
                    nightside = 1
                if dayside*nightside>0:
                    split = True
                if not split:
                    if dayside:
                        daynight[inbins] = 1
                    elif nightside:
                        daynight[inbins] = -1
                else:
                    daynight[inbins] = -999
                if kwargs.get('verbose',False):
                    print(k,theta_bins[i-1],th,
                          phi_bins[j-1],ph,
                          x[inbins].min(),x[inbins].max(),
                          '\tday:',dayside,'\tnight:',nightside)
    gmzone.values('daynight')[::] = daynight
    if kwargs.get('debug',False):
        gmzone.values('mapID')[::] = mapID



    #TODO
    #   A new section where we
    #       combine GM and IE
    #       Rotate IE to GSM
    #       using inputs from IEzone+singleGMzone
    #           for k,ie_point:
    #               if ie_point not matched w GMzone:
    #                   skip
    #               else
    #                   gm_index = where[th/phi in close]
    #                   xl,xu = X_gm[gm_index].minmax
    #                   '' for y and z too
    #               set {daymapped} for all index (in GM)
    #               set {xyzl,xyzu} for ie[k]
    #               derive {daymapped} for ie from above
