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

def reversed_mapping(gmzone,state_var):
    # Convert theta/phi mapping variable into cartesian footpoint values
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    if 'xfoot_1' not in gmzone.dataset.variable_names:
        eq('{xfoot_1}=sin(-{theta_1 [deg]}*pi/180+pi/2)*'+
                      'cos({phi_1 [deg]}*pi/180)')
        eq('{yfoot_1}=sin(-{theta_1 [deg]}*pi/180+pi/2)*'+
                     'sin({phi_1 [deg]}*pi/180)')
        eq('{zfoot_1}=cos(-{theta_1 [deg]}*pi/180+pi/2)')
        #eq('{xfoot_2}=sin(-{theta_2 [deg]}*pi/180+pi/2)*'+
        #             'cos({phi_2 [deg]}*pi/180)')
        #eq('{yfoot_2}=sin(-{theta_2 [deg]}*pi/180+pi/2)*'+
        #             'sin({phi_2 [deg]}*pi/180)')
        #eq('{zfoot_2} = 1*cos(-{theta_2 [deg]}*pi/180+pi/2)')
    # Pull some data from tecplot-> numpy
    state = gmzone.values(state_var).as_numpy_array()
    xfoot_1 = gmzone.values('xfoot_1').as_numpy_array()[state==1]
    yfoot_1 = gmzone.values('yfoot_1').as_numpy_array()[state==1]
    zfoot_1 = gmzone.values('zfoot_1').as_numpy_array()[state==1]
    #xfoot_2 = gmzone.values('xfoot_2').as_numpy_array()[state==1]
    #yfoot_2 = gmzone.values('yfoot_2').as_numpy_array()[state==1]
    #zfoot_2 = gmzone.values('zfoot_2').as_numpy_array()[state==1]
    # Make a new set of variables
    gmzone.dataset.add_variable('x_extent_1')
    gmzone.dataset.add_variable('y_extent_1')
    gmzone.dataset.add_variable('z_extent_1')
    x_extent = gmzone.values('x_extend_1').as_numpy_array()[state==1]
    y_extent = gmzone.values('y_extend_1').as_numpy_array()[state==1]
    z_extent = gmzone.values('z_extend_1').as_numpy_array()[state==1]
    theta_bins = np.linspace(-180,180,11)
    phi_bins = np.linspace(0,360,11)
    from IPython import embed; embed()


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
