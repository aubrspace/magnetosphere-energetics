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

def plot_all_runs_1Qty(axis, dflist, timekey, qtkey, ylabel, *,
                       xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots cumulative energy over time on axis with lables
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey, qtkey- used to located column with time and the qt to plot
    """
    legend_loc = 'lower right'
    for data in dflist:
        if Color == None:
            axis.plot(data[timekey],data[qtkey],
                  label=qtkey+data['name'].iloc[-1],
                  linewidth=Size, linestyle=ls)
            legend_loc = 'lower left'
        else:
            axis.plot(data[timekey],data[qtkey],
                  label=qtkey+data['name'].iloc[-1], color=Color,
                  linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def get_energy_dataframes(dflist):
    """Function adds cumulative energy columns based on power columns
        assuming a constant dt
    Inputs
        dflist, dfnames- list and corresponding names of dataframes
    Outputs
        use_dflist
    """
    use_dflist = []
    for df in enumerate(dflist):
        if not df[1].empty:
            if len(df[1]) > 1:
                ###Add cumulative energy terms
                #Compute cumulative energy In, Out, and Net
                start = df[1].index[0]
                totalE = df[1]['Total [J]']
                delta_t = (df[1]['Time [UTC]'].loc[start+1]-
                        df[1]['Time [UTC]'].loc[start]).seconds
                #use pandas cumulative sum method
                cumu_E_net = df[1]['K_net [W]'].cumsum()*delta_t*-1
                cumu_E_in = df[1]['K_injection [W]'].cumsum()*delta_t*-1
                cumu_E_out = df[1]['K_escape [W]'].cumsum()*delta_t*-1
                #readjust to initialize error to 0 at start
                cumu_E_net = (cumu_E_net+totalE.loc[start]-
                              cumu_E_net.loc[start])
                E_net_error = cumu_E_net - totalE
                E_net_rel_error = E_net_error/totalE*100
                #Add column to dataframe
                dflist[df[0]].loc[:,'CumulE_net [J]'] = cumu_E_net
                dflist[df[0]].loc[:,'CumulE_injection [J]'] = cumu_E_in
                dflist[df[0]].loc[:,'CumulE_escape [J]'] = cumu_E_out
                dflist[df[0]].loc[:,'Energy_error [J]'] = E_net_error
                dflist[df[0]].loc[:,'RelativeE_error [%]'] =E_net_rel_error
                ###Add derivative power terms
                Power_dens = df[1]['K_net [W]']/df[1]['Volume [Re^3]']
                #Compute derivative of energy total using central diff
                total_behind = totalE.copy()
                total_forward = totalE.copy()
                total_behind.index = total_behind.index-1
                total_forward.index = total_forward.index+1
                derived_Power = (total_behind-total_forward)/(-2*delta_t)
                derived_Power_dens = derived_Power/df[1]['Volume [Re^3]']
                #Estimate error in power term
                rms_Power = abs(df[1]['K_escape [W]'])
                power_error = df[1]['K_net [W]']-derived_Power
                power_error_rel = (df[1]['K_net [W]']-derived_Power)/(
                                   rms_Power/100)
                power_error_dens = power_error/df[1]['Volume [Re^3]']
                dflist[df[0]].loc[:,'Power_density [W/Re^3]'] = Power_dens
                dflist[df[0]].loc[:,'Power_derived [W]'] = derived_Power
                dflist[df[0]].loc[:,'Power_derived_density [W/Re^3]'] = (
                                                        derived_Power_dens)
                dflist[df[0]].loc[:,'Power_error [W]'] = power_error
                dflist[df[0]].loc[:,'Power_error [%]'] = power_error_rel
                dflist[df[0]].loc[:,'Power_error_density [W/Re^3]'] = (
                                                          power_error_dens)
                ###Add 1step energy terms
                predicted_energy = total_behind+df[1]['K_net [W]']*delta_t
                predicted_error = predicted_energy-totalE
                predicted_error_rel = (predicted_energy-totalE)/totalE*100
                dflist[df[0]].loc[:,'1step_Predict_E [J]'] = (
                                                          predicted_energy)
                dflist[df[0]].loc[:,'1step_Predict_E_error [J]'] = (
                                                           predicted_error)
                dflist[df[0]].loc[:,'1step_Predict_E_error [%]'] = (
                                                       predicted_error_rel)
                ###Add volume/surface area and estimated error
                dflist[df[0]].loc[:,'V/SA [Re]'] = (df[1]['Volume [Re^3]']/
                                                    df[1]['Area [Re^2]'])
                dflist[df[0]].loc[:,'V/(SA*X_ss)'] = (df[1]['V/SA [Re]']/
                                                  df[1]['X_subsolar [Re]'])
                dflist[df[0]].loc[:,'nVolume'] = (df[1]['Volume [Re^3]']/
                                              df[1]['Volume [Re^3]'].max())
                dflist[df[0]].loc[:,'nArea'] = (df[1]['Area [Re^2]']/
                                              df[1]['Area [Re^2]'].max())
                dflist[df[0]].loc[:,'nX_ss'] = (df[1]['X_subsolar [Re]']/
                                            df[1]['X_subsolar [Re]'].max())
                dflist[df[0]].loc[:,'Relative error [%]'] = ((
                        df[1]['X_subsolar [Re]']/3-df[1]['V/SA [Re]'])/
                        (df[1]['X_subsolar [Re]']/3))
                use_dflist.append(dflist[df[0]])
    return use_dflist

def prepare_figures(dflist, outpath):
    """Function calls which figures to be plotted
    Inputs
        dfilist- list object containing dataframes
        outpath
    """
    #cut out some data based on time
    for df in enumerate(dflist):
        name = pd.Series({'name':df[1]['name'].iloc[-1]})
        cuttofftime = dt.datetime(2014,2,18,7,0)
        cutdf = df[1][df[1]['Time [UTC]']>cuttofftime].append(name,
                                                 ignore_index=True)
        dflist[df[0]] = cutdf
    #identify magnetopause dfs from box dfs
    energy_dfs = get_energy_dataframes(dflist)
    mpdfs, boxdfs = [],[]
    for df in energy_dfs:
        if df['name'].iloc[-1].find('mp') != -1:
            mpdfs.append(df)
        if df['name'].iloc[-1].find('box') != -1:
            boxdfs.append(df)
    if energy_dfs != []:
        ###################################################################
        #Cumulative Energy
        if True:
            figname = 'EnergyAccumulation'
            #figure settings
            cumulative_E, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,
                                                                ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Energies on Primary Axes
            qtylist = ['Total [J]', '1step_Predict_E [J]']
            ylabel = 'Energy [J]'
            linestyle= '-'
            for qty in enumerate(qtylist):
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                       qty[1], ylabel,ylim=[
                                       0,mpdfs[0]['Total [J]'].max()*1.05])
                if boxdfs !=[]:
                    plot_all_runs_1Qty(ax2, boxdfs[0:1],
                                   timekey, qty[1], ylabel, Color='blue',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax2, boxdfs[1:2],
                                   timekey, qty[1], ylabel, Color='orange',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax3, boxdfs[2:3],
                                   timekey, qty[1], ylabel, Color='blue',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax3, boxdfs[3:4],
                                   timekey, qty[1], ylabel, Color='orange',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax4, boxdfs[4:5],
                                   timekey, qty[1], ylabel, Color='blue',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax4, boxdfs[5:6],
                                   timekey, qty[1], ylabel, Color='orange',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax5, boxdfs[6:7],
                                   timekey, qty[1], ylabel, Color='blue',
                                   ls=linestyle)
                    plot_all_runs_1Qty(ax5, boxdfs[7::],
                                   timekey, qty[1], ylabel, Color='orange',
                                   ls=linestyle)
                    linestyle= '--'
            #Error on twin Axes
            qtylist = ['1step_Predict_E_error [%]']
            ylabel = 'Error [%]'
            errlim = [-1,1]
            for qty in enumerate(qtylist):
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1.twinx(), mpdfs, timekey,
                                   qty[1], ylabel, Color='green',
                                   ylim=errlim)
            cumulative_E.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
        ###################################################################
        #Power
        if True:
            figname = 'Power'
            #figure settings
            power, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[9,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['K_injection [W]', 'K_escape [W]', 'K_net [W]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                qtykey, ylabel)
                if boxdfs != []:
                    plot_all_runs_1Qty(ax2, boxdfs[0:2],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax3, boxdfs[2:4],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax4, boxdfs[4:6],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax5, boxdfs[6::],
                                timekey,
                                qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            axislist = [ax1]
            dflist = [mpdfs]
            for ax in enumerate(axislist):
                if mpdfs != []:
                    ax[1].fill_between(mpdfs[0]['Time [UTC]'],
                                       mpdfs[0]['K_net [W]'])
            power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(cumulative_E)
        ###################################################################
        #Power vs derivedPower
        if True:
            figname = 'Power_vs_derived'
            #figure settings
            dpower, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['Power_density [W/Re^3]',
                       'Power_derived_density [W/Re^3]']
            ylabel = 'Power Density [W/Re^3]'
            for qtykey in qtylist:
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                    qtykey, ylabel)
                if boxdfs != []:
                    plot_all_runs_1Qty(ax2, boxdfs[0:2],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax3, boxdfs[2:4],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax4, boxdfs[4:6],
                                timekey,
                                qtykey, ylabel)
                    plot_all_runs_1Qty(ax5, boxdfs[6::],
                                timekey,
                                qtykey, ylabel)
            dpower.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(cumulative_E)
        ###################################################################
        #Power- type compare
        if True:
            figname = 'Power_type_compare'
            #figure settings
            power_compare1, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            #qtylist = ['ExB_net [W]', 'P0_net [W]', 'K_net [W]',
            qtylist = ['ExB_injection [W]', 'P0_injection [W]',
                       'K_injection [W]',
                       'ExB_escape [W]', 'P0_escape [W]', 'K_escape [W]']
            axislist = [ax1, ax2, ax1, ax2]
            ylabel = 'Power [W]'
            for qtykey in enumerate(qtylist):
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                qtykey[1], ylabel)
                if boxdfs != []:
                    plot_all_runs_1Qty(ax2, boxdfs, timekey,
                                qtykey[1], ylabel)
            power_compare1.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #Power- surface compare
        if True:
            figname = 'Power_surface_compare'
            #figure settings
            power_compare, (ax1,ax2,ax3,ax4) =plt.subplots(nrows=4,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist1 = ['Power_error [%]', 'RelativeE_error [%]']
            qtylist2 = ['Power_error [W]', 'Energy_error [J]']
            axislist1 = [ax1, ax2]
            axislist2 = [ax3, ax4]
            ylabel1 = 'Error [%]'
            ylabel2 = 'Error [W] (surface-volume)'
            ylabel3 = 'Error [J] (surface-volume)'
            errlim = [-20,20]
            for qtykey in enumerate(qtylist1):
                axis = axislist1[qtykey[0]]
                if mpdfs != []:
                    plot_all_runs_1Qty(axis, mpdfs, timekey,
                                qtykey[1], ylabel1, ylim=errlim)
                if boxdfs != []:
                    plot_all_runs_1Qty(axis, boxdfs, timekey,
                                qtykey[1], ylabel1, ylim=errlim)
            ylabel = ylabel2
            for qtykey in enumerate(qtylist2):
                axis = axislist2[qtykey[0]]
                if mpdfs != []:
                    plot_all_runs_1Qty(axis, mpdfs, timekey,
                                qtykey[1], ylabel2)
                if boxdfs != []:
                    plot_all_runs_1Qty(axis, boxdfs[0:1],
                                   timekey, qtykey[1], ylabel)
                    plot_all_runs_1Qty(axis, boxdfs[2:5],
                                   timekey, qtykey[1], ylabel)
                    plot_all_runs_1Qty(axis, boxdfs[6:7],
                                   timekey, qtykey[1], ylabel)
                ylabel = ylabel3
            power_compare.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #ExB Power
        if True:
            figname = 'ExBPower'
            #figure settings
            ExBpower, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['ExB_injection [W]', 'ExB_escape [W]', 'ExB_net [W]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                qtykey, ylabel)
                if boxdfs != []:
                    plot_all_runs_1Qty(ax2, boxdfs, timekey,
                                qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            axislist = [ax1, ax2]
            dflist = []
            if mpdfs != []:
                dflist.append(mpdfs)
            if boxdfs != []:
                dflist.append(boxdfs)
            for ax in enumerate(axislist):
                df = dflist[ax[0]][0]
                ax[1].fill_between(df['Time [UTC]'], 0, df['ExB_net [W]'])
            ExBpower.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(ExBpower)
        ###################################################################
        #Flow Power
        if True:
            figname = 'P0Power'
            #figure settings
            P0power, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['P0_injection [W]', 'P0_escape [W]', 'P0_net [W]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                if mpdfs != []:
                    plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                qtykey, ylabel)
                if boxdfs != []:
                    plot_all_runs_1Qty(ax2, boxdfs, timekey,
                                qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            axislist = [ax1, ax2]
            dflist = []
            if mpdfs != []:
                dflist.append(mpdfs)
            if boxdfs != []:
                dflist.append(boxdfs)
            for ax in enumerate(axislist):
                df = dflist[ax[0]][0]
                ax[1].fill_between(df['Time [UTC]'], 0, df['P0_net [W]'])
            P0power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(P0power)
        ###################################################################
        #Normalized Power
        if True:
            figname = 'AveragePower'
            #figure settings
            power, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Power terms on Primary Axes
            qtylist = ['Average K_injection [W/Re^2]',
                       'Average K_escape [W/Re^2]',
                       'Average K_net [W/Re^2]']
            ylabel = 'Power [W]'
            for qtykey in qtylist:
                plot_all_runs_1Qty(ax1, mpdfs, timekey,
                                qtykey, ylabel)
                plot_all_runs_1Qty(ax2, boxdfs, timekey,
                                qtykey, ylabel)
            #Fill between to distinguish +/- net powers
            axislist = [ax1, ax2]
            dflist = [mpdfs, boxdfs]
            for ax in enumerate(axislist):
                df = dflist[ax[0]][0]
                ax[1].fill_between(df['Time [UTC]'], 0,
                                   df['Average K_net [W/Re^2]'])
            power.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
        #Volume and Surface Area
        if True:
            figname = 'VolumeSurfaceArea'
            #figure settings
            VolArea, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,
                                               sharex=True, figsize=[18,12])
            #Time
            timekey = 'Time [UTC]'
            #Volume, Surface Area, Vol/SA on primary axes
            qtylist = ['Volume [Re^3]', 'Area [Re^2]', 'X_subsolar [Re]']
            axislist = [ax1, ax2, ax3]
            for qtykey in enumerate(qtylist):
                axis = axislist[qtykey[0]]
                plot_all_runs_1Qty(axis, mpdfs, timekey,
                                qtykey[1], qtykey[1])
            qtylist = ['nVolume', 'nArea', 'nX_ss']
            for qtykey in enumerate(qtylist):
                plot_all_runs_1Qty(ax4, mpdfs, timekey,
                                qtykey[1], qtykey[1])
            VolArea.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(VolArea)
        ###################################################################
        #Other Energies
        if False:
            figname = 'VolumeEnergies'
            #quick manipulations
            dfs = energy_dfs[0:1]
            #figure settings
            Volume_E, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,
                                               sharex=True, figsize=[20,10])
            #axes
            timekey = 'Time [UTC]'
            qtykey = 'uB [J]'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'uE [J]'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax5, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar [J]'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax2, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp [J]'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax3, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm [J]'
            ylabel = 'Thermal Energy [J]'
            plot_all_runs_1Qty(ax4, dfs, timekey,
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
            #figure settings
            Volume_Ebreakdown, (ax1) = plt.subplots(nrows=1,ncols=1,
                                               sharex=True, figsize=[20,10])
            #axes
            timekey = 'Time [UTC]'
            qtykey = 'uB partition'
            ylabel = 'Magnetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel)
            timekey = 'Time [UTC]'
            qtykey = 'uE partition'
            ylabel = 'Electric Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'Etherm partition'
            ylabel = 'Parallel Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'KEpar partition'
            ylabel = 'Perpendicular Kinetic Energy [J]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel)
            qtykey = 'KEperp partition'
            ylabel = 'Energy fraction [%]'
            plot_all_runs_1Qty(ax1, dfs, timekey,
                               qtykey, ylabel, ylim=[0,5])
            Volume_Ebreakdown.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(Volume_E)
        ###################################################################
        #Power Error histograms
        if boxdfs != [] and mpdfs != []:
            figname = 'PowerErrorHistograms'
            #figure settings
            hist, axis = plt.subplots(nrows=3,ncols=3,sharex=False,
                                      sharey=False, figsize=[18,18])
            print('axis:')
            print(axis)
            #Power terms on Primary Axes
            qtylist = ['Power_error [W]']
            ylabel = 'P(X)'
            title = 'X=Power Error Density [W/Re^3]'
            histylim = [0, 2e-8]
            count = 0
            for row in range(0,3):
                for col in range(0,3):
                    if count == 8:
                        axis[row][col].hist(mpdfs[0][qtylist[0]]/
                                            mpdfs[0]['Volume [Re^3]'],
                                            bins=100, range=(-3.5e8,3e8),
                                            density=True)
                        name = mpdfs[0]['name'].iloc[-1]
                        axis[row][col].set_xlabel(name)
                    else:
                        axis[row][col].hist(boxdfs[count][qtylist[0]]/
                                        boxdfs[count]['Volume [Re^3]'],
                                        bins=100, range=(-3.5e8,3e8),
                                        density=True)
                        name = boxdfs[count]['name'].iloc[-1]
                        axis[row][col].set_xlabel(name)
                    axis[row][col].axvline(c='green',ls='-')
                    axis[row][col].set_ylim(histylim)
                    axis[row][col].set_ylabel(ylabel)
                    axis[row][col].annotate('mean=',(2,2))
                    count+=1
            hist.suptitle(title,fontsize=18)
            hist.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
            #plt.close(power)
        ###################################################################
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
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid',
                    'field', 'mp_', 'box', 'sphere']
        dflist = []
        spin = Spinner('finding available temporal data ')
        for path in data_path_list:
            print(path)
            if path != None:
                for datafile in glob.glob(path+'/*.h5'):
                    with pd.HDFStore(datafile) as hdf_file:
                        for key in hdf_file.keys():
                            print(key)
                            if any([key.find(match)!=-1
                                   for match in approved]):
                                nametag = pd.Series({'name':key})
                                df = hdf_file[key].append(nametag,
                                                ignore_index=True)
                                print(df['name'])
                                dflist.append(df)
                            else:
                                print('key {} not understood!'.format(key))
                            spin.next()
        prepare_figures(dflist, outputpath)
    print('done!')

if __name__ == "__main__":
    datapath = sys.argv[1]
    print(datapath)
    process_temporal_mp([datapath],datapath+'figures/')
