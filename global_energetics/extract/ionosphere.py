#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#interpackage modules
'''
from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import volume_tools
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (calc_dayside_mp,
                                                    calc_tail_mp,
                                                    dump_to_pandas,
                                                    create_cylinder,
                                                    load_cylinder,
                                                    abs_to_timestamp,
                                                    write_to_timelog)
'''
def get_ionosphere_zone(eventdt, datapath):
    """Function to find the correct ionosphere datafile and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of event time
        datapath- str path to ionosphere data
    """
    eventstring = str(eventdt)
    #parse event string (will update later)
    yr = eventstring.split('-')[0][-2::]
    mn = eventstring.split('-')[1]
    dy = eventstring.split('-')[2].split(' ')[0]
    hr = eventstring.split(' ')[1].split(':')[0]
    minute = eventstring.split(':')[1]
    datafile = 'it'+yr+mn+dy+'_'+hr+minute+'00_000.tec'
    #load file matching description
    if os.path.exists(datapath+datafile):
        ie_data = tp.data.load_tecplot(datapath+datafile)
        north_iezone = ie_data.zone('IonN *')
        south_iezone = ie_data.zone('IonS *')
        return north_iezone, south_iezone
    else:
        print('no ionosphere data found!')
        return None, None

def get_ionosphere(field_data, ndatafile, sdatafile, *, show=False,
                   comp1=True, comp2=True, comp3=True,
                   local_integrate=False):
    """Function that finds, plots and calculates energetics on the
        ionosphere surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        ndatafile, sdatafile- ionosphere data, assumes .csv files
        show- boolean for display
        comp1, comp2- boolean for which comp to include
        local_integrate- boolean for integration in python or pass to tec
    """
    #get date and time info from datafiles name

    #Read .csv files into dataframe objects

    #Call calc_components for each hemisphere

    if show:
        #display component values
        pass

    if local_integrate:
        #integrate in python using pandas
        comp1var, comp2var, comp3var = [],[],[]
        compvars = [comp1var, comp2var, comp3var]
        compstrings = ['comp1', 'comp2', 'comp3']
        for comp in enumerate([comp1, comp2, comp3]):
            if comp[1]:
                compvar[comp[0]] = (northdf[compstrings[comp[0]]].sum()+
                                   southdf[compstrings[comp[0]]].sum())[0]
        [comp1var, comp2var, comp3var] = compvars
    else:
        #pass data to tecplot
        load_ionosphere_tecplot(northdf, southdf)



# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()
    tp.new_layout()
    get_ionosphere(sys.argv[1],sys.argv[2])
