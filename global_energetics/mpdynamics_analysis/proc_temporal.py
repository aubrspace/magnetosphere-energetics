#!/usr/bin/env python3
#proc_temporal.py
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

def cleanup_dataframes(dflist, dfnames):
    """Function fixes time column and adds cumulative energy columns based
        on power columns assuming a constant dt
    Inputs
        dflist, dfnames- list and corresponding names of dataframes
    Outputs
        use_dflist, use_dfnames
    """
    use_dflist, use_dflabels = [], []
    for df in enumerate(dflist):
        if not df[1].empty:
            ###Combine yr,mo,da,hr,sec into datetime
            timecol = pd.DataFrame(columns=['Time [UTC]'])
            for index in df[1].index:
                timecol = timecol.append(pd.DataFrame([[dt.datetime(
                                            df[1]['year'].iloc[index],
                                            df[1][' month'].iloc[index],
                                            df[1][' day'].iloc[index],
                                            df[1][' hour'].iloc[index],
                                            df[1][' minute'].iloc[index],
                                            df[1][' second'].iloc[index])]],
                                            columns=['Time [UTC]']),
                                            ignore_index=True)
            dflist[df[0]].loc[:,'Time [UTC]'] = timecol
            dflist[df[0]] = dflist[df[0]].drop(columns=['year',' month',
                                                       ' day',' hour',
                                                      ' minute',' second'])
            ###Add cumulative energy terms
            #Compute cumulative energy In, Out, and Net
            delta_t = (df[1]['Time [UTC]'].loc[1]-
                       df[1]['Time [UTC]'].loc[0]).seconds
            #use pandas cumulative sum method
            cumu_E_in = df[1][' K_in [kW]'].cumsum()*delta_t
            cumu_E_out = df[1][' K_out [kW]'].cumsum()*delta_t
            cumu_E_net = df[1][' K_net [kW]'].cumsum()*delta_t
            #Add column to dataframe
            dflist[df[0]].loc[:,'CumulE_in [kJ]'] = cumu_E_in
            dflist[df[0]].loc[:,'CumulE_out [kJ]'] = cumu_E_out
            dflist[df[0]].loc[:,'CumulE_net [kJ]'] = cumu_E_net
            ###Add modified dataframe to list
            use_dflist.append(dflist[df[0]])
            use_dflabels.append(dfnames[df[0]])
    return use_dflist, use_dflabels

def prepare_figures(statsdf, shue98df, shue97df, flowdf, hybriddf,
                    fielddf, dfnames):
    """Function calls which figures to be plotted
    Inputs
        statsdf..fielddf- list containing all dataframes
        dfnames- names of each dataset
    """
    dflist = [shue98df, shue97df, flowdf, hybriddf, fielddf]
    energy_dfs, energy_dfnames= cleanup_dataframes(dflist, dfnames[1::])
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
            qtykey = ' K_net [kW]'
            ylabel = 'Net Power [kW]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            cumulative_E.savefig('./temp/{}.png'.format(figname))
            plt.close(cumulative_E)
        ###################################################################
        #Volume and Surface Area
        if True:
            figname = 'VolumeSurfaceArea'
            cumulative_E, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = ' Area [m^2]'
            ylabel = 'Surface Area [m^2]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = ' Volume [R^3]'
            ylabel = 'Volume [m^3]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            cumulative_E.savefig('./temp/{}.png'.format(figname))
            plt.close(cumulative_E)
        ###################################################################
        #Other Energies
        if True:
            figname = 'VolumeEnergies'
            cumulative_E, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = ' uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = ' KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = ' KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = ' Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            plt.show()
            cumulative_E.savefig('./temp/{}.png'.format(figname))
            plt.close(cumulative_E)
        ###################################################################
    else:
        print('Unable to create ___ plot, data missing!')
    pass

def process_temporal_mp(*, data_path_integrated=None,data_path_stats=None):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path_integrated/stats- path to the data, default will skip
        make_fig- bool for figures
    """
    if data_path_integrated == None and data_path_stats == None:
        print('Nothing to do, no data_paths were given!')
    else:
        dfnames = ['stats', 'shue98', 'shue97', 'flow', 'hybrid', 'field']
        statsdf, shue98df, shue97df, flowdf, hybriddf, fielddf=[
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame()]
        spin = Spinner('finding available temporal data ')
        for path in [data_path_integrated, data_path_stats]:
            if path != None:
                for datafile in glob.glob(path+'/*.csv'):
                    if datafile.split('/')[-1].find('stats') != -1:
                        statsdf = pd.read_csv(datafile, index_col=0)
                        statsdf = statsdf.sort_values('Time [UTC]')
                    elif datafile.split('/')[-1].find('shue98') != -1:
                        shue98df = pd.read_csv(datafile, index_col=False)
                    elif datafile.split('/')[-1].find('shue97') != -1:
                        shue97df = pd.read_csv(datafile, index_col=False)
                    elif datafile.split('/')[-1].find('shue') != -1:
                        shue98df = pd.read_csv(datafile, index_col=False)
                    elif datafile.split('/')[-1].find('flow') != -1:
                        flowdf = pd.read_csv(datafile, index_col=False)
                    elif datafile.split('/')[-1].find('field') != -1:
                        fielddf = pd.read_csv(datafile, index_col=False)
                    elif datafile.split('/')[-1].find('hybrid') != -1:
                        hybriddf = pd.read_csv(datafile, index_col=False)
                    else:
                        print('filename {} not understood!'.format(
                                                  datafile.split('/')[-1]))
                    spin.next()
        print('Found:')
        for df in enumerate([statsdf, shue98df, shue97df, flowdf, hybriddf,
                             fielddf]):
            if not df[1].empty:
                print(dfnames[df[0]])
                print(df[1])
        prepare_figures(statsdf, shue98df, shue97df, flowdf, hybriddf,
                        fielddf, dfnames)
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
if __name__ == "__main__":
    IPATH = ('/Users/ngpdl/Code/swmf-energetics/output/'+
                     'mpdynamics/jan27_3surf')
    process_temporal_mp(data_path_integrated=IPATH)
