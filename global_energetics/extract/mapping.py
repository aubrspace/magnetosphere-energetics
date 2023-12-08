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
from numpy import abs, pi, cos, sin, sqrt,sign
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

def check_bin(x,theta_1,phi_1,inbin,state):
    """Function checks 4 quadrants of bin to determine contestation (daynight)
    Inputs
        x (arr[float])
        theta_1 (arr[float])
        phi_1 (arr[float])
        inbin (arr[float])
        state (arr[float])
    Return
        qs (list[list[bools]]) - quadrants of the original bin
        contested (bool) - if the quads agree about daynight
    """
    thHigh = theta_1[inbin].max()
    thLow = theta_1[inbin].min()
    phHigh = phi_1[inbin].max()
    phLow = phi_1[inbin].min()
    thMid = (thHigh+thLow)/2
    phMid = (phHigh+phLow)/2
    q1 = ((state==1)&
          (theta_1<thHigh)&(theta_1>thMid)&
          (phi_1<phMid)&(phi_1>phLow))
    q2 = ((state==1)&
          (theta_1<thMid)&(theta_1>thLow)&
          (phi_1<phMid)&(phi_1>phLow))
    q3 = ((state==1)&
          (theta_1<thMid)&(theta_1>thLow)&
          (phi_1<phHigh)&(phi_1>phMid))
    q4 = ((state==1)&
          (theta_1<thHigh)&(theta_1>thMid)&
          (phi_1<phHigh)&(phi_1>phMid))
    quadbins = []
    for q in [q1,q2,q3,q4]:
        if any(q):
            quadbins.append(q)
    signs = [sign(x[q].mean()) for q in quadbins]
    if (len(signs)==0):
        contested = False
    elif (len(signs)==signs.count(signs[0])):
        contested = False
    else:
        contested = True
    return quadbins, contested

def reversed_mapping(gmzone,state_var,**kwargs):
    # Convert theta/phi mapping variable into cartesian footpoint values
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    # Pull some data from tecplot-> numpy
    if state_var==1:
        state = 1
    else:
        state = gmzone.values(state_var).as_numpy_array()
    theta_1 = gmzone.values('theta_1 *').as_numpy_array()
    phi_1 = gmzone.values('phi_1 *').as_numpy_array()
    volume = gmzone.values('dvol *').as_numpy_array()
    x = gmzone.values('X *').as_numpy_array()*volume #NOTE volume weighted
    # Make a new set of variables
    if 'daynight' not in gmzone.dataset.variable_names:
        gmzone.dataset.add_variable('daynight')
    if kwargs.get('debug',False):
        if 'mapID' not in gmzone.dataset.variable_names:
            gmzone.dataset.add_variable('mapID')
        mapID = gmzone.values('mapID').as_numpy_array()
    daynight = gmzone.values('daynight').as_numpy_array()
    # Create an initial set of coarse bins
    theta_bins = np.linspace(0,90,10)
    phi_bins = np.linspace(0,360,37)
    k=0
    # Iterate through each bin
    for i,thHigh in enumerate(theta_bins[1::]):
        for j,phHigh in enumerate(phi_bins[1::]):
            thLow = theta_bins[i-1]
            phLow = phi_bins[j-1]
            inbins = ((state==1)&
                      (theta_1<thHigh)&
                      (theta_1>thLow)&
                      (phi_1<phHigh)&
                      (phi_1>phLow))
            if any(inbins):
                # Subdivide bin until 4 subquadrants agree
                finished_bins = []
                contested_bins = [inbins]
                i=0
                while len(contested_bins)>0:
                    i+=1
                    old_contested_bins = contested_bins
                    contested_bins = []
                    for b in old_contested_bins:
                        qs, contested = check_bin(x,theta_1,phi_1,b,state)
                        if not contested:
                            finished_bins.append(b)
                        else:
                            for q in qs:
                                contested_bins.append(q)
                    if i>1 and kwargs.get('verbose',False):
                        print(i)
                    if i>5:
                        for q in qs:
                            finished_bins.append(q)
                        contested_bins = []
                # Now actually set the values using the finished_bin list
                for inbin in finished_bins:
                    dayside,nightside,split = 0,0,False
                    k+=1
                    if kwargs.get('debug',False):
                        mapID[inbins] = k
                    if x[inbin].mean()>0:
                        dayside = 1
                    if x[inbin].mean()<0:
                        nightside = 1
                    if dayside*nightside>0:
                        split = True
                    if not split:
                        if dayside:
                            daynight[inbin] = 1
                        elif nightside:
                            daynight[inbin] = -1
                    else:
                        daynight[inbins] = -999
                    if kwargs.get('verbose',False):
                        print(k,thLow,thHigh,
                          phLow,phHigh,
                          x[inbins].min(),x[inbins].max(),
                          '\tday:',dayside,'\tnight:',nightside)
    # Set the values in Tecplot from our numpy array
    gmzone.values('daynight')[::] = daynight
    if kwargs.get('debug',False):
        gmzone.values('mapID')[::] = mapID

def port_mapping_to_ie(ocflb,gm,**kwargs):
    """Function that does some stuff
    Inputs
        ocflb (Zone)-
        gm (Zone) -
    Returns
        None
    """
    # Pull in some Tecplot objects
    if 'North' in ocflb.name:
        ie_theta = ocflb.values('theta_1 *').as_numpy_array()
        ie_phi = ocflb.values('phi_1 *').as_numpy_array()
        gm_theta = gm.values('theta_1 *').as_numpy_array()
        gm_phi = gm.values('phi_1 *').as_numpy_array()
    elif 'South' in ocflb.name:
        ie_theta = ocflb.values('theta_2 *').as_numpy_array()
        ie_phi = ocflb.values('phi_2 *').as_numpy_array()
        gm_theta = gm.values('theta_2 *').as_numpy_array()
        gm_phi = gm.values('phi_2 *').as_numpy_array()
    gm_status = gm.values('Status').as_numpy_array()
    gm_mapping = gm.values('daynight').as_numpy_array()
    ie_mapping = ocflb.values('daynight').as_numpy_array()
    for i,(th,ph) in enumerate(zip(ie_theta,ie_phi)):
        if kwargs.get('verbose',False):
            print(i,'/',len(ie_theta))
        # find a neighborhood in GM around theta/phi and Status==3
        neighborhood = ((gm_status==3)&
                        (abs(gm_theta-th)<1)&
                        (abs(gm_phi-ph)<1))
        if not any(neighborhood):
            neighborhood = ((gm_status==3)&
                            (abs(gm_theta-th)<2)&
                            (abs(gm_phi-ph)<2))
        theta_neighbors = gm_theta[neighborhood]
        phi_neighbors = gm_phi[neighborhood]
        # calculate the distance matrix given theta/phi matricies
        distances = np.sqrt(2-2*(
                    sin(theta_neighbors)*sin(th)*cos(phi_neighbors-ph)+
                    cos(theta_neighbors)*cos(th)))
        # take the daynight (+1,-1) of the closest point
        closest = distances==distances.min()
        mapping = gm_mapping[neighborhood][closest][0]
        # Save this value in an array
        ie_mapping[i] = mapping
    # Update the Tecplot state
    ocflb.values('daynight')[::] = ie_mapping
