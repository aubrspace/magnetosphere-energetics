#!/usr/bin/env python3
#proc_temporal.py
"""Functions for handling and processing time varying magnetopause surface
    data that is spatially averaged, reduced, etc
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
from progress.bar import Bar
from progress.spinner import Spinner
import spacepy
#from spacepy import coordinates as coord
from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *

def handle_datasets_temporal(data_path_integrated, data_path_stats):
    #find out which mp datasets are available based on whats passed
    #validate that data at data_path is okay, random sample:
        #if multiple files in path, check that they have the same parameters
        #check that time period is monatomically increasing
    pass

def compile_temporal(listofdfs):
    pass

def process_temporal_mp(*, data_path_integrated=None, data_path_stats=None,
                                                            make_fig=True):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path_integrated/stats- path to the data, default will skip
        make_fig- bool for figures
    """
    if data_path_integrated == None and data_path_stats == None:
        print('Nothing to do, no data_paths were given!')
    else:
        #call handle_datasets_temporal, return column names
        #for each file:
            #read dataset as dataframe and append to list of df's
        #call compile_temporal, return 1 dataframe with everything
        #if make_fig:
            #call make_fig1 based on some idea on how to compare figs
        #if make_fig2:
            #call make_fig2 based on some idea on how to compare figs
        pass
    pass
