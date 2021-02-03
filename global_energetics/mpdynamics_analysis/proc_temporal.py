#!/usr/bin/env python3
"""Functions for handling and processing time varying magnetopause surface
    data that is spatially averaged, reduced, etc
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

def plot_all_runs_1Qty(axis, dflist, dflabels, timekey, qtkey, ylabel):
    """Function plots cumulative energy over time on axis with lables
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey, qtkey- used to located column with time and the qt to plot
    """
    for data in enumerate(dflist):
        if dflabels[data[0]].find('flow') != -1:
            axis.plot(data[1][timekey],data[1][qtkey],color='orange',
                      label='Flowline')
        if dflabels[data[0]].find('shue') != -1:
            axis.plot(data[1][timekey],data[1][qtkey],color='black',
                      label='Shue98')
        if dflabels[data[0]].find('field') != -1:
            axis.plot(data[1][timekey],data[1][qtkey],color='blue',
                      label='Fieldline')
        if dflabels[data[0]].find('hybrid') != -1:
            axis.plot(data[1][timekey],data[1][qtkey],color='green',
                      label='Hybrid')
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend()

def get_energy_dataframes(dflist, dfnames):
    """Function adds cumulative energy columns based on power columns
        assuming a constant dt
    Inputs
        dflist, dfnames- list and corresponding names of dataframes
    Outputs
        use_dflist, use_dfnames
    """
    use_dflist, use_dflabels = [], []
    for df in enumerate(dflist):
        if not df[1].empty & (dfnames[df[0]].find('stat')==-1):
            ###Add cumulative energy terms
            #Compute cumulative energy In, Out, and Net
            delta_t = (df[1]['Time [UTC]'].loc[1]-
                       df[1]['Time [UTC]'].loc[0]).seconds
            #use pandas cumulative sum method
            cumu_E_in = df[1]['mp K_in [kW]'].cumsum()*delta_t
            cumu_E_out = df[1]['mp K_out [kW]'].cumsum()*delta_t
            cumu_E_net = df[1]['mp K_net [kW]'].cumsum()*delta_t
            #Add column to dataframe
            dflist[df[0]].loc[:,'CumulE_in [kJ]'] = cumu_E_in
            dflist[df[0]].loc[:,'CumulE_out [kJ]'] = cumu_E_out
            dflist[df[0]].loc[:,'CumulE_net [kJ]'] = cumu_E_net
            ###Add modified dataframe to list
            use_dflist.append(dflist[df[0]])
            use_dflabels.append(dfnames[df[0]])
    return use_dflist, use_dflabels

def prepare_figures(dflist, dfnames, outpath):
    """Function calls which figures to be plotted
    Inputs
        dfilist- list object containing dataframes
        dfnames- names of each dataset
    """
    energy_dfs, energy_dfnames= get_energy_dataframes(dflist, dfnames)
    stats_dfs, stats_dfnames = [], []
    for df in enumerate(dflist):
        if dfnames[df[0]].find('stat') != -1:
            stats_dfs.append(df[1])
            stats_dfnames.append(dfnames[df[0]])
    if energy_dfs != []:
        ###################################################################
        #Cumulative Energy, and Power
        if True:
            figname = 'PowerEnergy'
            cumulative_E, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'CumulE_net [kJ]'
            ylabel = 'Cumulative Energy [kJ]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp K_net [kW]'
            ylabel = 'Net Power [kW]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            cumulative_E.savefig(outpath+'{}.png'.format(figname))
            plt.close(cumulative_E)
        ###################################################################
        #Volume and Surface Area
        if True:
            figname = 'VolumeSurfaceArea'
            VolArea, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'mp Area [m^2]'
            ylabel = 'Surface Area [m^2]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp Volume [R^3]'
            ylabel = 'Volume [m^3]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            VolArea.savefig(outpath+'{}.png'.format(figname))
            plt.close(VolArea)
        ###################################################################
        #Other Energies
        if True:
            figname = 'VolumeEnergies'
            Volume_E, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'mp uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            plt.show()
            Volume_E.savefig(outpath+'{}.png'.format(figname))
            plt.close(Volume_E)
        ###################################################################
    else:
        print('Unable to create ___ plot, data missing!')
    pass

def process_temporal_mp(*, data_path_integrated=None,data_path_stats=None,
                           outputpath=None):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path_integrated/stats- path to the data, default will skip
        make_fig- bool for figures
    """
    if data_path_integrated == None and data_path_stats == None:
        print('Nothing to do, no data_paths were given!')
    else:
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid', 'field']
        dflist, dfnames = [], []
        spin = Spinner('finding available temporal data ')
        for path in [data_path_integrated, data_path_stats]:
            if path != None:
                for datafile in glob.glob(path+'/*.h5'):
                    with pd.HDFStore(datafile) as hdf_file:
                        for key in hdf_file.keys():
                            if any([key.find(match)!=-1
                                   for match in approved]):
                                df = pd.read_hdf(hdf_file,key,'r')
                                dflist.append(df)
                                dfnames.append(key)
                            else:
                                print('key {} not understood!'.format(key))
                            spin.next()
        prepare_figures(dflist, dfnames, outputpath)

if __name__ == "__main__":
    #IPATH = ('/Users/ngpdl/Code/swmf-energetics/output/'+
    #                 'mpdynamics/jan27_3surf')
    IPATH = ('/home/aubr/Code/swmf-energetics/output')
    SPATH = ('/home/aubr/Code/swmf-energetics/output/meshdata/stats')
    OPATH = ('/home/aubr/Code/swmf-energetics/output/figures/')
    process_temporal_mp(data_path_integrated=IPATH,
                        data_path_stats=SPATH,
                        outputpath=OPATH)
