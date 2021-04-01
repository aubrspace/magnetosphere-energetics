#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import logging as log
import os
import sys
import glob
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import scipy as sp
from scipy import integrate
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import Bar
from progress.spinner import Spinner
import spacepy
#from spacepy import coordinates as coord
from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
#Interpackage imports
if os.path.exists('../supermag-data'):
    os.system('ln -s ../supermag-data/supermag.py')
    print('soft link to supermag.py created')
    SUPERMAGDATAPATH = '../supermag-data/data'
else:
    try:
        import supermag
    except ModuleNotFoundError:
        print('Cant find supermag.py!')
try:
    import supermag
except ModuleNotFoundError:
    print('supermag.py module not linked!')
    EXIT=True
else:
    EXIT=False
#from global_energetics.mpdynamics_analysis.proc_temporal import (
#                                                        plot_all_runs_1Qty)
def plot_dst(axis, dflist, dflabels, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in enumerate(dflist):
        if dflabels[df[0]] == 'supermag':
            qtkey = 'SM-R [nT]'
        elif dflabels[df[0]] == 'swmf_log':
            qtkey = 'dst'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[1][timekey],data[1][qtkey],
                      label=qtkey+'_'+dflabels[data[0]],
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def get_supermag_data(start, end, datapath):
    """function gathers supermag data for comparison
        (supermag.py INTERNAL USE ONLY!!!)
    Inputs
        start, end- datatime objects used for supermag.py
    """
    supermag.DIR = SUPERMAGDATAPATH
    print('obtaining supermag data')
    return pd.DataFrame(supermag.supermag(start,end))

def get_swmf_data(datapath):
    """Function reads in swmf geoindex log and log
    Inputs
        datapath- path where log files are
    Outputs
        geoindexdata, swmflogdata- pandas DataFrame with data
    """
    #read files
    geoindex = glob.glob(datapath+'*geo*.log')[0]
    swmflog = glob.glob(datapath+'*log_*.log')[0]
    print('reading: \n\t{}\n\t{}'.format(geoindex,swmflog))
    def datetimeparser(datetimestring):
        #maybe move this somewhere to call a diff parser depending on file
        return dt.datetime.strptime(datetimestring,'%Y %m %d %H %M %S %f')
    geoindexdata = pd.read_csv(geoindex, sep='\s+', skiprows=1,
            parse_dates={'times':['year','mo','dy','hr','mn','sc','msc']},
            date_parser=datetimeparser,
            infer_datetime_format=True, keep_date_col=True)
    swmflogdata = pd.read_csv(swmflog, sep='\s+', skiprows=1,
            parse_dates={'times':['year','mo','dy','hr','mn','sc','msc']},
            date_parser=datetimeparser,
            infer_datetime_format=True, keep_date_col=True)
    return geoindexdata, swmflogdata

def prepare_figures(swmf_index, swmf_log, supermag):
    """Function creates figure and axis objects to fill with plots and
        saves them
    Inputs
        swmf_index, swmf_log, supermag- DataFrame objects to pull from
    """
    #print data
    print('swmf_index:\n{}\nswmf_log:\n{}\nsupermag:\n{}'.format(
                                                              swmf_index,
                                                              swmf_log,
                                                              supermag))
    data_names = ['swmf_index', 'swmf_log', 'supermag']
    for df in enumerate([swmf_index, swmf_log, supermag]):
        if not df.empty():
            dflist.append(df[1])
            dfnames.append(data_names[df[0]])
    ######################################################################
    #SMR index comparison
    if True:
        smr, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True,
                                                          figsize=[18,8])
        #Time
        timekey = 'times'
        ylabel = 'SMR [nT]'
        plot_dst(ax1, dflist, dfnames, ylabel)
        plt.show()
    ######################################################################

def process_indices(data_path, outputpath):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path- path to the data
        outputpaht
    """
    swmf_index, swmf_log = get_swmf_data(data_path)
    supermag = get_supermag_data(swmf_index['times'][0],
                                 swmf_index['times'].iloc[-1],
                                 SUPERMAGDATAPATH)
    prepare_figures(swmf_index, swmf_log, supermag)

if __name__ == "__main__":
    if not EXIT:
        datapath = sys.argv[1]
        print('processing indices output at {}'.format(datapath))
        process_indices(datapath,datapath+'indices/')
