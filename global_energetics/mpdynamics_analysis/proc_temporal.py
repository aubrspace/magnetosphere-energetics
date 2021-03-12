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
                       xlim=None, ylim=None, Color=None):
    """Function plots cumulative energy over time on axis with lables
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey, qtkey- used to located column with time and the qt to plot
    """
    for data in enumerate(dflist):
        if Color == None:
            axis.plot(data[1][timekey],data[1][qtkey],
                  label=qtkey+dflabels[data[0]])
            legend_loc = 'lower left'
        else:
            axis.plot(data[1][timekey],data[1][qtkey],
                  label=qtkey+dflabels[data[0]], color=Color)
            legend_loc = 'lower right'
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
    axis.legend(loc=legend_loc)

def get_energy_dataframes(dflist, dfnames):
    """Function adds cumulative energy columns based on power columns
        assuming a constant dt
    Inputs
        dflist, dfnames- list and corresponding names of dataframes
    Outputs
        use_dflist, use_dfnames
    """
    use_dflist, use_dflabels = [], []
    #totalE0 = (dflist[0]['Total [J]'].loc[dflist[0].index[0]]+
    #           dflist[1]['Total [J]'].loc[dflist[1].index[1]])
    '''
    dflist[0].loc[:,'K_net_minus_bound [W]'] = (dflist[0]['K_net [W]']-
                                                dflist[1]['K_net [W]'])
    dflist[1].loc[:,'K_net_minus_bound [W]'] = (dflist[1]['K_net [W]']-
                                                dflist[1]['K_net [W]'])
    dflist[2].loc[:,'K_net_minus_bound [W]'] = (dflist[2]['K_net [W]']-
                                                dflist[1]['K_net [W]'])
    '''
    for df in enumerate(dflist):
        if not df[1].empty & (dfnames[df[0]].find('stat')==-1):
            if len(df[1]) > 1:
                #dflist[df[0]].loc[:,'K_net_minus_bound [W]'] = (
                #    dflist[df[0]]['K_net [W]']-dflist[1]['K_net [W]'])
                ###Add cumulative energy terms
                #Compute cumulative energy In, Out, and Net
                start = df[1].index[0]
                totalE = df[1]['Total [J]']
                delta_t = (df[1]['Time [UTC]'].loc[start+1]-
                        df[1]['Time [UTC]'].loc[start]).seconds
                #use pandas cumulative sum method
                cumu_E_net = df[1]['K_net [W]'].cumsum()*delta_t*-1
                #adjust_cumu_E_net = df[1]['K_net_minus_bound [W]'].cumsum()*delta_t*-1
                cumu_E_in = df[1]['K_injection [W]'].cumsum()*delta_t*-1
                cumu_E_out = df[1]['K_escape [W]'].cumsum()*delta_t*-1
                #readjust to initialize error to 0 at start
                cumu_E_net = (cumu_E_net+totalE.loc[start]-
                              cumu_E_net.loc[start])
                #adjust_cumu_E_net = (adjust_cumu_E_net+totalE.loc[start]-
                #                     adjust_cumu_E_net.loc[start])
                E_net_error = cumu_E_net - totalE
                E_net_rel_error = E_net_error/totalE*100
                #Add column to dataframe
                dflist[df[0]].loc[:,'CumulE_net [J]'] = cumu_E_net
                #dflist[df[0]].loc[:,'Adjusted_CumulE_net [J]'] = adjust_cumu_E_net
                dflist[df[0]].loc[:,'CumulE_injection [J]'] = cumu_E_in
                dflist[df[0]].loc[:,'CumulE_escape [J]'] = cumu_E_out
                dflist[df[0]].loc[:,'Energy_error [J]'] = E_net_error
                dflist[df[0]].loc[:,'RelativeE_error [%]'] =E_net_rel_error
                ###Add derivative power terms
                #Compute derivative of energy total using central diff
                total_behind = totalE.copy()
                total_forward = totalE.copy()
                total_behind.index = total_behind.index-1
                total_forward.index = total_forward.index+1
                derived_Power = (total_behind-total_forward)/(-2*delta_t)
                power_error = abs(derived_Power-df[1]['K_net [W]'])
                dflist[df[0]].loc[:,'Power_derived [W]'] = derived_Power
                dflist[df[0]].loc[:,'Power_error [W]'] = power_error
                ###Add volume/surface area and estimated error
                dflist[df[0]].loc[:,'V/SA [Re]'] = (df[1]['Volume [Re^3]']/
                                                    df[1]['Area [Re^2]'])
                dflist[df[0]].loc[:,'Relative error [%]'] = ((
                        df[1]['X_subsolar [Re]']/3-df[1]['V/SA [Re]'])/
                        (df[1]['X_subsolar [Re]']/3))
                use_dflist.append(dflist[df[0]])
                use_dflabels.append(dfnames[df[0]])
                from IPython import embed; embed()
    return use_dflist, use_dflabels

def prepare_figures(dflist, dfnames, outpath):
    """Function calls which figures to be plotted
    Inputs
        dfilist- list object containing dataframes
        dfnames- names of each dataset
    """
    for df in enumerate(dflist):
        dflist[df[0]] = df[1][(df[1]['Time [UTC]'] > dt.datetime(
                                                          2013,9,19,4,25))]
    energy_dfs, energy_dfnames= get_energy_dataframes(dflist, dfnames)
    mpdfs, mpnames, boxdfs, boxnames, spdfs, spnames = [],[],[],[],[],[]
    for df in enumerate(energy_dfs):
        if energy_dfnames[df[0]].find('mp') != -1:
            mpdfs.append(energy_dfs[df[0]])
            mpnames.append(energy_dfnames[df[0]])
        if energy_dfnames[df[0]].find('box') != -1:
            boxdfs.append(energy_dfs[df[0]])
            boxnames.append(energy_dfnames[df[0]])
        if energy_dfnames[df[0]].find('sphere') != -1:
            spdfs.append(energy_dfs[df[0]])
            spnames.append(energy_dfnames[df[0]])
    #specific assignments
    coredfs, coredfnames = spdfs[::], spnames[::]
    #if mpdfs[0]['Volume [Re^3]'].max()>mpdfs[1]['Volume [Re^3]'].max():
    #    mpshelldfs, mpshellnames = [mpdfs[1]], [mpnames[1]]
    #    mpdfs, mpnames = [mpdfs[0]], [mpnames[0]]
    #else:
    #    mpshelldfs, mpshellnames = [mpdfs[0]], [mpnames[0]]
    #    mpdfs, mpnames = [mpdfs[1]], [mpnames[1]]
    if energy_dfs != []:
        cumulative_E, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
                                           sharex=True, figsize=[18,12])
        ###################################################################
        #Cumulative Energy
        if True:
            figname = 'EnergyAccumulation'
            #figure settings
            #cumulative_E, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
            #                                   sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Energies on Primary Axes
            qtylist = ['Total [J]', 'CumulE_net [J]']
            ylabel = 'Energy [J]'
            for qty in enumerate(qtylist):
                #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
                #                qty[1], ylabel)
                #plot_all_runs_1Qty(ax2, mpshelldfs, mpshellnames, timekey,
                #                qty[1], ylabel)
                plot_all_runs_1Qty(ax1, boxdfs[0:1], boxnames[0:1], timekey,
                                qty[1], ylabel)
                #plot_all_runs_1Qty(ax4, coredfs, coredfnames, timekey,
                #                qty[1], ylabel)
            #Error on twin Axes
            qtylist = ['RelativeE_error [%]']
            ylabel = 'Error [%]'
            for qty in enumerate(qtylist):
                #plot_all_runs_1Qty(ax1.twinx(), mpdfs, mpnames, timekey,
                #                   qty[1], ylabel, Color='green')
                #plot_all_runs_1Qty(ax2.twinx(), mpshelldfs, mpshellnames,
                #                   timekey, qty[1], ylabel, Color='green')
                plot_all_runs_1Qty(ax1.twinx(), boxdfs[0:1], boxnames[0:1], timekey,
                                   qty[1], ylabel, Color='red')
                #plot_all_runs_1Qty(ax4.twinx(), coredfs, coredfnames,
                #                   timekey, qty[1], ylabel, Color='green')
            #cumulative_E.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
        ###################################################################
        #Power
        if True:
            #figname = 'Power'
            #figure settings
            #power, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
            #                                   sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            #qtylist = ['K_injection [W]', 'K_escape [W]', 'K_net [W]']
            qtylist = ['Power_derived [W]', 'K_net [W]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
                #                qtykey, ylabel)
                #plot_all_runs_1Qty(ax2, mpshelldfs, mpshellnames, timekey,
                #                qtykey, ylabel)
                plot_all_runs_1Qty(ax2, boxdfs[0:1], boxnames[0:1], timekey,
                                qtykey, ylabel)
                #plot_all_runs_1Qty(ax4, coredfs, coredfnames, timekey,
                #                qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            #axislist = [ax1, ax2, ax3, ax4]
            #dflist = [mpdfs, mpshelldfs, boxdfs, coredfs]
            axislist = [ax1]
            dflist = [boxdfs]
            '''
            for ax in enumerate(axislist):
                df = dflist[ax[0]][0]
                ax[1].fill_between(df['Time [UTC]'], 0, df['K_net [W]'])
            '''
            '''
            #Error on twin Axes
            qtylist = ['Power_error [W]']
            ylabel = 'Error [W]'
            for qtykey in qtylist:
                plot_all_runs_1Qty(ax1.twinx(), mpdfs, mpnames, timekey,
                                   qtykey, ylabel, Color='red')
                plot_all_runs_1Qty(ax2.twinx(), mpshelldfs, mpshellnames,
                                   timekey, qtykey, ylabel, Color='red')
                plot_all_runs_1Qty(ax3.twinx(), boxdfs, boxnames, timekey,
                                   qtykey, ylabel, Color='red')
                plot_all_runs_1Qty(ax4.twinx(), coredfs, coredfnames,
                                   timekey, qtykey, ylabel, Color='red')
            '''
            #power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(cumulative_E)
        ###################################################################
        #Power- surface compare
        if False:
            figname = 'Power_surface_compare'
            #figure settings
            power_compare, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['K_injection [W]', 'K_escape [W]', 'K_net [W]']
            axislist = [ax1, ax2, ax3]
            ylabel = 'Power [W]'
            for qtykey in enumerate(qtylist):
                axis = axislist[qtykey[0]]
                #plot_all_runs_1Qty(axis, mpdfs, mpnames, timekey,
                #                qtykey[1], ylabel)
                #plot_all_runs_1Qty(axis, mpshelldfs, mpshellnames, timekey,
                #                qtykey[1], ylabel)
                plot_all_runs_1Qty(axis, boxdfs, boxnames, timekey,
                                qtykey[1], ylabel)
                #plot_all_runs_1Qty(axis, coredfs, coredfnames, timekey,
                #                qtykey[1], ylabel)
            power_compare.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #Poynting Power
        if True:
            #figname = 'PoyntingPower'
            #quick manipulation
            dfs = energy_dfs[0:1]
            dfnames = energy_dfnames[0:1]
            #figure settings
            #power, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
            #                                   sharex=True, figsize=[18,6])
            #axes
            timekey = 'Time [UTC]'
            ylabel = 'Power [W]'
            #mp
            qtykey = 'ExB_injection [W]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'ExB_escape [W]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'ExB_net [W]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            #sphere
            qtykey = 'ExB_injection [W]'
            plot_all_runs_1Qty(ax3, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            qtykey = 'ExB_escape [W]'
            plot_all_runs_1Qty(ax3, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            qtykey = 'ExB_net [W]'
            plot_all_runs_1Qty(ax3, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            #core
            qtykey = 'ExB_injection [W]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'ExB_escape [W]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'ExB_net [W]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            #ax1.fill_between(mpdfs[0]['Time [UTC]'], 0,
            #                 mpdfs[0]['ExB_net [W]'])
            #ax2.fill_between(spdfs[0]['Time [UTC]'], 0,
            #                 spdfs[0]['ExB_net [W]'])
            #ax3.fill_between(coredfs[0]['Time [UTC]'], 0,
            #                 coredfs[0]['ExB_net [W]'])
            #power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #Flow Power
        if True:
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['ExB_net [W]', 'P0_net [W]', 'K_net [W]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
                #                qtykey, ylabel)
                #plot_all_runs_1Qty(ax2, mpshelldfs, mpshellnames, timekey,
                #                qtykey, ylabel)
                plot_all_runs_1Qty(ax4, boxdfs[0:1], boxnames[0:1], timekey,
                                qtykey, ylabel)
                #plot_all_runs_1Qty(ax4, coredfs, coredfnames, timekey,
                #                qtykey, ylabel)
        ###################################################################
        #Normalized Power
        if False:
            figname = 'AveragePower'
            #quick manipulation
            dfs = energy_dfs[0:1]
            dfnames = energy_dfnames[0:1]
            #figure settings
            power, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,6])
            #axes
            timekey = 'Time [UTC]'
            ylabel = 'Average Power [W/Re^2]'
            #mp
            qtykey = 'Average K_injection [W/Re^2]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'Average K_escape [W/Re^2]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'Average K_net [W/Re^2]'
            #plot_all_runs_1Qty(ax1, mpdfs, mpnames, timekey,
            #                   qtykey, ylabel)
            #sphere
            qtykey = 'Average K_injection [W/Re^2]'
            plot_all_runs_1Qty(ax2, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Average K_escape [W/Re^2]'
            plot_all_runs_1Qty(ax2, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Average K_net [W/Re^2]'
            plot_all_runs_1Qty(ax2, boxdfs, boxnames, timekey,
                               qtykey, ylabel)
            #core
            qtykey = 'Average K_injection [W/Re^2]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'Average K_escape [W/Re^2]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            qtykey = 'Average K_net [W/Re^2]'
            #plot_all_runs_1Qty(ax3, coredfs, coredfnames, timekey,
            #                   qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            #ax1.fill_between(mpdfs[0]['Time [UTC]'], 0,
            #                 mpdfs[0]['Average K_net [W/Re^2]'])
            #ax2.fill_between(boxdfs[0]['Time [UTC]'], 0,
            #                 boxdfs[0]['Average K_net [W/Re^2]'])
            #ax3.fill_between(coredfs[0]['Time [UTC]'], 0,
            #                 coredfs[0]['Average K_net [W/Re^2]'])
            power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #Volume and Surface Area
        if False:
            figname = 'VolumeSurfaceArea'
            #figure settings
            VolArea, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Volume, Surface Area, Vol/SA on primary axes
            qtylist = ['Volume [Re^3]', 'Area [Re^2]', 'V/SA [Re]']
            axislist = [ax1, ax2, ax3]
            for qtykey in enumerate(qtylist):
                axis = axislist[qtykey[0]]
                #plot_all_runs_1Qty(axis, mpdfs, mpnames, timekey,
                #                qtykey[1], qtykey[1])
            #VolArea.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(VolArea)
        ###################################################################
        '''
        #Other Energies
        if True:
            figname = 'VolumeEnergies'
            dfs = energy_dfs[1::]
            dfnames = energy_dfnames[1::]
            Volume_E, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[20,10])
            timekey = 'Time [UTC]'
            qtykey = 'uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'uE [J]'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax5, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, dfs, dfnames, timekey,
                               qtykey, ylabel)
            Volume_E.savefig(outpath+'{}.png'.format(figname))
        '''
        ###################################################################
        #Other Energies
        if False:
            figname = 'VolumeEnergies'
            #quick manipulations
            dfs = energy_dfs[0:1]
            dfnames = energy_dfnames[0:1]
            #figure settings
            Volume_E, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[20,10])
            #axes
            timekey = 'Time [UTC]'
            qtykey = 'uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'uE [J]'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax5, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, dfs, dfnames, timekey,
                               qtykey, ylabel)
            Volume_E.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(Volume_E)
        ###################################################################
        #Energy split
        if False:
            figname = 'VolumeEnergyBreakdown'
            #quick manipulation
            data = energy_dfs[0]
            uB = data['uB [J]']
            uE = data['uE [J]']
            Etherm = data['Etherm [J]']
            KEpar = data['KEpar [J]']
            KEperp = data['KEperp [J]']
            total = uE+uB+Etherm+KEpar+KEperp
            energy_dfs[0].loc[:,'total energy'] = total.values
            energy_dfs[0].loc[:,'uB partition'] = uB/total*100
            energy_dfs[0].loc[:,'uE partition'] = uE/total*100
            energy_dfs[0].loc[:,'Etherm partition'] = Etherm/total*100
            energy_dfs[0].loc[:,'KEpar partition'] = KEpar/total*100
            energy_dfs[0].loc[:,'KEperp partition'] = KEperp/total*100
            dfs = energy_dfs[0:1]
            dfnames = energy_dfnames[0:1]
            #figure settings
            Volume_Ebreakdown, (ax1) = plt.subplots(nrows=1,ncols=1,
                                               sharex=True, figsize=[20,10])
            #axes
            timekey = 'Time [UTC]'
            qtykey = 'uB partition'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'uE partition'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm partition'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar partition'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp partition'
            ylabel = 'Energy fraction [%]'
            plot_all_runs_1Qty(ax1, dfs, dfnames, timekey,
                               qtykey, ylabel, ylim=[0,5])
            Volume_Ebreakdown.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(Volume_E)
        ###################################################################
        print('saving to {}'.format(outpath))
        cumulative_E.savefig(outpath+'{}.png'.format(figname))
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
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid', 'field', 'mp_', 'box', 'sphere']
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
    datapath = os.getcwd()+'/testout/'
    print(datapath)
    process_temporal_mp([datapath],datapath+'figures/')
