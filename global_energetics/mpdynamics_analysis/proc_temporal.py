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

def plot_all_runs_1Qty(axis, dflist, dflabels, timekey, qtkey, ylabel, *,
                       xlim=None, ylim=None):
    """Function plots cumulative energy over time on axis with lables
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey, qtkey- used to located column with time and the qt to plot
    """
    for data in enumerate(dflist):
        axis.plot(data[1][timekey],data[1][qtkey],
                  label=qtkey)
        '''
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
        '''
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
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
            if len(df[1]) > 1:
                ###Add cumulative energy terms
                #Compute cumulative energy In, Out, and Net
                delta_t = (df[1]['Time [UTC]'].loc[1]-
                        df[1]['Time [UTC]'].loc[0]).seconds
                #use pandas cumulative sum method
                cumu_E_in = df[1]['mp_iso_innersurf K_injection [W]'].cumsum()*delta_t
                cumu_E_out = df[1]['mp_iso_innersurf K_escape [W]'].cumsum()*delta_t
                cumu_E_net = df[1]['mp_iso_innersurf K_net [W]'].cumsum()*delta_t
                #Add column to dataframe
                dflist[df[0]].loc[:,'CumulE_injection [J]'] = cumu_E_in
                dflist[df[0]].loc[:,'CumulE_escape [J]'] = cumu_E_out
                dflist[df[0]].loc[:,'CumulE_net [J]'] = cumu_E_net
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
        for df in enumerate(energy_dfs):
            energy_dfs[df[0]] = df[1][(df[1]['Time [UTC]'] > dt.datetime(2014,2,18,8))]
        ###################################################################
        #Cumulative Energy
        if True:
            figname = 'Energy Accumulation'
            cumulative_E, ax1 = plt.subplots(nrows=1,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'CumulE_injection [J]'
            ylabel = 'Cumulative Energy injection [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel, ylim=[-1e18,1e18])
            timekey = 'Time [UTC]'
            qtykey = 'CumulE_escape [J]'
            ylabel = 'Cumulative Energy escape [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel, ylim=[-1e18,1e18])
            timekey = 'Time [UTC]'
            qtykey = 'CumulE_net [J]'
            ylabel = 'Cumulative Energy net [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel, ylim=[-1e18,1e18])
            cumulative_E.savefig(outpath+'{}.png'.format(figname))
            plt.show()
            plt.close(cumulative_E)
        ###################################################################
        #Power
        if True:
            figname = 'Power'
            power, (ax1) = plt.subplots(nrows=1,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'mp_iso_innersurf K_injection [W]'
            ylabel = 'Power injecting [W]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'mp_iso_innersurf K_escape [W]'
            ylabel = 'Power escaping [W]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'mp_iso_innersurf K_net [W]'
            ylabel = 'Power transfer net [W]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            power.savefig(outpath+'{}.png'.format(figname))
            plt.show()
            plt.close(power)
        ###################################################################
        #Volume and Surface Area
        if True:
            figname = 'VolumeSurfaceArea'
            VolArea, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,
                                               sharex=True, figsize=[18,6])
            timekey = 'Time [UTC]'
            qtykey = 'mp_iso_innersurf Area [Re^2]'
            ylabel = 'Surface Area [Re^2]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp_rho_innerradius Volume [Re^3]'
            ylabel = 'Volume [Re^3]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            VolArea.savefig(outpath+'{}.png'.format(figname))
            plt.show()
            plt.close(VolArea)
        ###################################################################
        #Other Energies
        if True:
            figname = 'VolumeEnergies'
            Volume_E, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[20,10])
            timekey = 'Time [UTC]'
            qtykey = 'mp_rho_innerradius uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'mp_rho_innerradius uE [J]'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax5, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp_rho_innerradius KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp_rho_innerradius KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'mp_rho_innerradius Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            Volume_E.savefig(outpath+'{}.png'.format(figname))
            plt.show()
            plt.close(Volume_E)
        ###################################################################
        #Energy split
        if True:
            data = energy_dfs[0]
            uB = data['mp_rho_innerradius uB [J]']
            uB = uB - uB.min()
            uE = data['mp_rho_innerradius uE [J]']
            Etherm = data['mp_rho_innerradius Etherm [J]']
            KEpar = data['mp_rho_innerradius KEpar [J]']
            KEperp = data['mp_rho_innerradius KEperp [J]']
            total = uE+Etherm+KEpar+KEperp
            energy_dfs[0].loc[:,'total energy'] = total.values
            #energy_dfs[0].loc[:,'uB partition'] = uB/total*100
            energy_dfs[0].loc[:,'uE partition'] = uE/total*100
            energy_dfs[0].loc[:,'Etherm partition'] = Etherm/total*100
            energy_dfs[0].loc[:,'KEpar partition'] = KEpar/total*100
            energy_dfs[0].loc[:,'KEperp partition'] = KEperp/total*100
            figname = 'VolumeEnergyBreakdown'
            Volume_Ebreakdown, (ax1) = plt.subplots(nrows=1,ncols=1,
                                               sharex=True, figsize=[20,10])
            '''
            timekey = 'Time [UTC]'
            qtykey = 'uB partition'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            '''
            timekey = 'Time [UTC]'
            qtykey = 'uE partition'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm partition'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar partition'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp partition'
            ylabel = 'Energy fraction [%]'
            plot_all_runs_1Qty(ax1, energy_dfs, energy_dfnames, timekey,
                               qtykey, ylabel)
            Volume_Ebreakdown.savefig(outpath+'{}.png'.format(figname))
            plt.show()
            plt.close(Volume_E)
        ###################################################################
    else:
        print('Unable to create ___ plot, data missing!')
    pass

def process_temporal_mp(data_path_list, outputpath):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path_list- paths to the data, default will skip
        outputpaht
    """
    if data_path_list == []:
        print('Nothing to do, no data_paths were given!')
    else:
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid', 'field', 'mp_iso_innersurf']
        dflist, dfnames = [], []
        spin = Spinner('finding available temporal data ')
        for path in data_path_list:
            if path != None:
                for datafile in glob.glob(path+'/*.h5'):
                    with pd.HDFStore(datafile) as hdf_file:
                        for key in hdf_file.keys():
                            if any([key.find(match)!=-1
                                   for match in approved]):
                                dflist.append(hdf_file[key])
                                dfnames.append(key)
                            else:
                                print('key {} not understood!'.format(key))
                            spin.next()
        prepare_figures(dflist, dfnames, outputpath)

if __name__ == "__main__":
    datapath = '/Users/ngpdl/Code/swmf-energetics/output/mpdynamics/isoFeb17/'
    process_temporal_mp([datapath],datapath+'figures/')
