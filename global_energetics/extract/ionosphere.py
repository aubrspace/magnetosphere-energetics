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
def load_ionosphere_tecplot(field_data, ndf, sdf, *, integrate=True):
    """Function loads ionosphere shell data into tecplot
    Input
        field_data- tecplot DataSet object with 3D field data
        ndf, sdf- north and south data in pandas DataFrame
    Output
        izone- tecplot zone object
    """
    if integrate:
        from global_energetics.extract.surface_tools import surface_analysis

def calc_components(df, *, comp1=True, comp2=True, comp3=True):
    """Function takes hemipheric IE data and appends component values
    Inputs
        df- pandas datframe object of ionosphere shell
        comp1, comp2- boolean for which comp to include
    Returns
        df_mod- pandas dataframe with component values appended
    """
    if comp1:
        comp1df = df

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
