#!/usr/bin/env python3
"""Functions for handling and processing time varying magnetopause surface
    data that is spatially averaged, reduced, etc
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime

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

def plot_simple(dflist, timekey, outpath):
    """Function plots each column of each dataframe vs time
    Inputs
        dflist
        timekey
    """
    npanels = len(dflist)
    for col in dflist[-1].keys():
        #col = ''.join(col.split('/'))
        print('making basic plot of column: {}'.format(col))
        validname = ''.join(x for x in col if (x.isalnum() or x in "._- "))
        figname = 'simple_'+'_'.join(validname.split(' '))
        #figure settings
        fig, axes = plt.subplots(nrows=npanels, ncols=1, sharex=True,
                                 figsize=[18,4*npanels])
        ylabel = col
        for df in enumerate(dflist):
            if any([key==col and key!='name' for key in df[1].keys()]):
                plot_all_runs_1Qty(axes[df[0]], [df[1]], timekey,
                                   col, ylabel)
                axes[df[0]].set_title(col)
        fig.savefig(outpath+'/simple_plots/{}.png'.format(figname))
        plt.close()
        #plt.show()

def add_derived_variables(dflist):
    """Function adds variables based on existing columns
    Inputs
        dflist - list of dataframes
    Outputs
        dflist
    """
    for df in enumerate(dflist):
        if not df[1].empty:
            if len(df[1]) > 1 and df[1]['name'].iloc[-1].find('fixed')==-1:
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
                dflist[df[0]]['CumulE_net [J]'] = cumu_E_net
                dflist[df[0]]['CumulE_injection [J]'] = cumu_E_in
                dflist[df[0]]['CumulE_escape [J]'] = cumu_E_out
                dflist[df[0]]['Energy_error [J]'] = E_net_error
                dflist[df[0]]['RelativeE_error [%]'] =E_net_rel_error
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
                dflist[df[0]]['Power_density [W/Re^3]'] = Power_dens
                dflist[df[0]]['Power_derived [W]'] = derived_Power
                dflist[df[0]]['Power_derived_density [W/Re^3]'] = (
                                                        derived_Power_dens)
                dflist[df[0]]['Power_error [W]'] = power_error
                dflist[df[0]]['Power_error [%]'] = power_error_rel
                dflist[df[0]]['Power_error_density [W/Re^3]'] = (
                                                          power_error_dens)
                ###Add 1step energy terms
                predicted_energy = total_behind+df[1]['K_net [W]']*delta_t
                predicted_error = predicted_energy-totalE
                predicted_error_rel = (predicted_energy-totalE)/totalE*100
                dflist[df[0]]['1step_Predict_E [J]'] = (
                                                          predicted_energy)
                dflist[df[0]]['1step_Predict_E_error [J]'] = (
                                                           predicted_error)
                dflist[df[0]]['1step_Predict_E_error [%]'] = (
                                                       predicted_error_rel)
                ###Add volume/surface area and estimated error
                dflist[df[0]]['V/SA [Re]'] = (df[1]['Volume [Re^3]']/
                                                    df[1]['Area [Re^2]'])
                dflist[df[0]]['V/(SA*X_ss)'] = (df[1]['V/SA [Re]']/
                                                  df[1]['X_subsolar [Re]'])
                dflist[df[0]]['nVolume'] = (df[1]['Volume [Re^3]']/
                                              df[1]['Volume [Re^3]'].max())
                dflist[df[0]]['nArea'] = (df[1]['Area [Re^2]']/
                                              df[1]['Area [Re^2]'].max())
                dflist[df[0]]['nX_ss'] = (df[1]['X_subsolar [Re]']/
                                            df[1]['X_subsolar [Re]'].max())
    return dflist

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
    dflist = add_derived_variables(dflist)
    plot_simple(dflist, 'Time [UTC]', outpath)
    '''
    mpdfs, boxdfs = [],[]
    for df in dflist:
        if df['name'].iloc[-1].find('mp') != -1:
            mpdfs.append(df)
        if df['name'].iloc[-1].find('box') != -1:
            boxdfs.append(df)
    '''
    if False:
        ###################################################################
        #Cumulative Energy
        if True:
            figname = 'EnergyAccumulation'
            #figure settings
            cumulative_E, (ax1) = plt.subplots(nrows=1,
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
                if False:
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
                                   qty[1], ylabel, Color='green')
                                   #ylim=errlim)
            cumulative_E.savefig(outpath+'{}.png'.format(figname))
            #plt.show()
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

def read_energetics(data_path_list, *, add_variables=True):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path_list- paths to the data
    Outputs
        dflist- list object full of pandas dataframes, 1 per 3Dvolume
    """
    if data_path_list == []:
        print('Nothing to do, no data_paths were given!')
    else:
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid',
                    'field', 'mp_', 'box', 'sphere', 'lcb', 'fixed']
        dflist = []
        for path in data_path_list:
            print(path)
            if path != None:
                for datafile in glob.glob(path+'/*.h5'):
                    with pd.HDFStore(datafile) as hdf_file:
                        include_timetag = False
                        for key in hdf_file.keys():
                            if key.find('Time')!=-1:
                                timetag = pd.Series(
                                        {'Time_UTC':hdf_file[key][0]})
                                include_timetag = True
                        for key in hdf_file.keys():
                            print(key)
                            if any([key.find(match)!=-1
                                   for match in approved]):
                                nametag = pd.Series({'name':key})
                                df = hdf_file[key].append(nametag,
                                                ignore_index=True)
                                if include_timetag:
                                    df = df.append(timetag,
                                            ignore_index=True)
                                print(df['name'])
                                dflist.append(df)
                            else:
                                print('key {} not understood!'.format(key))
        if add_variables:
            if len(dflist) > 0:
                dflist = add_derived_variables(dflist)
    print('done!')
    return dflist

if __name__ == "__main__":
    datapath = sys.argv[1]
    print(datapath)
    process_temporal_mp([datapath],datapath+'figures/')
