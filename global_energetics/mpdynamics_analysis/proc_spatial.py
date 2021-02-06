#!/usr/bin/env python3
"""Functions for handling and processing spatial magnetopause surface data
    that is static in time
"""
import logging as log
import os
import sys
import glob
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, linspace, deg2rad
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from progress.bar import Bar
from progress.spinner import Spinner
import spacepy
import pandas as pd
#from spacepy import coordinates as coord
from spacepy import time as spacetime


def count_files(path):
    """Function returns number of files at the path
    Inputs
        path
    Outputs
        nfile
    """
    nfile=0
    for datafile in glob.glob(path+'/*.h5'):
        nfile += 1
    return nfile

def plot_2Dpositions(axis, data, datalabels, alpha, timestamp):
    """Functions plots all non-empty 2D slices with a and timestamp label
    Inputs
        axis- axis object to be plotted on
        data, datalabels
        alpha, timestamp- for figure display
    """
    for curve in enumerate(data):
        if datalabels[curve[0]].find('flow') != -1:
            axis.plot(curve[1]['X [R]'], curve[1]['height [R]'],
                      color='orange')
        if datalabels[curve[0]].find('hybrid') != -1:
            axis.plot(curve[1]['X [R]'], curve[1]['height [R]'],
                      color='green')
    for curve in enumerate(data):
        if datalabels[curve[0]].find('shue') != -1:
            axis.plot(curve[1]['X [R]'], curve[1]['height [R]'],
                      color='grey', linewidth=8)
    axis.set_xlabel('X [R]')
    axis.set_ylabel('sqrt(Y**2+Z**2) [R]')
    #axis.legend(loc='lower left')

def prep_slices(dflist, dfnames, alpha, timestamp):
    """Function calls which figures to be plotted
    Inputs
        dflist- list containing all dataframes
        dfnames- names of each dataset
        alpha, timestamp- for figure display
    Outputs
        nonempty_list, nonempty_names
        timelabel
    """
    #establish timestamp label
    timelabel = ''
    for snip1 in str(timestamp.values[0]).split(' '):
        for snip2 in snip1.split(':'):
            timelabel = timelabel + snip2
    timelabel = timelabel.split('.')[0]
    #weed out empty datasets
    nonempty_list = []
    nonempty_names = []
    for df in enumerate(dflist):
        if not df[1].empty:
            nonempty_list.append(df[1])
            nonempty_names.append(dfnames[df[0]])
    return nonempty_list, nonempty_names, timelabel

def prepare_figures(dflist, dfnames, alpha, timedf, outpath):
    """Function calls which figures to be plotted
    Inputs
        dflist- list containing all dataframes
        dfnames- names of each dataset
        alpha, timestamp- for figure display
    """
    #nonempty_list, nonempty_names, timelabel = prep_slices(dflist, dfnames,
    #                                                      alpha, timestamp)
    if dflist != []:
        datestring = (str(timedf[0].year)+'-'+str(timedf[0].month)+'-'+
                      str(timedf[0].day)+'-'+str(timedf[0].hour)+'-'+
                      str(timedf[0].minute))
        ###################################################################
        #2D curve plot
        if True:
            curve_plot, ax1 = plt.subplots(nrows=1,ncols=1,sharex=True,
                                                            figsize=[18,6])
            curve_plot.text(0,0,str(timedf[0]))
            ax1.set_xlim([-40,12])
            ax1.set_ylim([0,30])
            plot_2Dpositions(ax1, dflist, dfnames, alpha, str(timedf[0]))
            curve_plot.savefig(outpath+'height_maps_{}.png'.format(datestring))
            plt.close(curve_plot)
        ###################################################################

def get_maxdiff(dflist1, dflist2, xmin, xmax):
    """Function finds the maximum difference between heights of dataframes
    Inputs
        dflist1, dflist2- list containing dataframe objects
    Outputs
        maxdiff- single value with maximum difference
        xloc, aloc- single value with xlocation of max difference
    """
    maxdiff, xloc, aloc = 0, None, None
    #Look in intervals of dx=1
    xbins, dx = linspace(xmin, xmax, int((xmax-xmin)/1), retstep=True)
    for x in xbins:
        dftemp1, dftemp2 = [], []
        for df in enumerate(dflist1):
            dftemp1.append(df[1][((df[1]['X [R]']<x+dx/2) &
                          (df[1]['X [R]']>x-dx/2))])
        for df in enumerate(dflist2):
            dftemp2.append(df[1][((df[1]['X [R]']<x+dx/2) &
                          (df[1]['X [R]']>x-dx/2))])
        #Look at all permutations of two sets of lists
        for df1 in dftemp1:
            for df2 in dftemp2:
                if (not df1.empty) & (not df2.empty):
                    h1 = df1['height [R]'].mean()
                    h2 = df2['height [R]'].mean()
                    diff = abs(h1-h2)
                    #diff = abs(df1['height [R]']-df2['height [R]']).max()
                    if diff > maxdiff:
                        maxdiff = diff
                        xloc = (df1['X [R]'].mean()+df2['X [R]'].mean())/2
                        aloc = df2['alpha(deg)'].mean()
    return maxdiff, xloc, aloc

def get_meandiff(dflist1, dflist2, xmin, xmax):
    """Function finds the mean difference between heights of dataframes
    Inputs
        dflist1, dflist2- list containing dataframe objects
    Outputs
        meandiff- single value with maximum difference
        aloc- single value with xlocation of max difference
    """
    meandiff, aloc = 0, None
    xbins, dx = linspace(xmin, xmax, int((xmax-xmin)/1), retstep=True)
    #Look at each permutation of sets of lists
    for df1 in dflist1:
        for df2 in dflist2:
            #Look in intervals of dx=1
            diffx_sum, num_x = 0, 0
            for x in xbins:
                df1x = df1[((df1['X [R]']<x+dx/2) &
                             (df1['X [R]']>x-dx/2))]
                df2x = df2[((df2['X [R]']<x+dx/2) &
                             (df2['X [R]']>x-dx/2))]
                if (not df1x.empty) & (not df2x.empty):
                    h1x = df1x['height [R]'].mean()
                    h2x = df2x['height [R]'].mean()
                    diffx_sum = diffx_sum+abs(h1x-h2x)
                    num_x += 1
            if diffx_sum != 0:
                diff = diffx_sum/num_x
                if diff > meandiff:
                    meandiff = diff
                    aloc = df2['alpha(deg)'].mean()
    return meandiff, aloc

def expand_azimuth(dflist, dfnames, alpha_points, da):
    """Function splits dataframes in list into number of dataframes
    Inputs
        dflist, dfnames- list of datasets and their names
        alpha_points, da- azimuthal information for splitting
    Outputs
        newlist, newnames- expanded set of df's and names
    """
    newlist, newnames = [], []
    for a in alpha_points:
        #split off only values in +/- da/4 range
        for df in enumerate(dflist):
            if not df[1].empty:
                newlist.append(df[1][((df[1]['alpha(deg)']<a+da/4) &
                              (df[1]['alpha(deg)']>a-da/4))])
                newnames.append(dfnames[df[0]]+'_'+str(a))
    return newlist, newnames

def prepare_stats(timedf, shuedfs, shuenames, nonshue_dfs, nonshue_names,
                  alpha_points, da):
    """Function calls which statistics are to be calculated
    Inputs
        timedf- holds time information
        shuedfs,shuenames- DataFrame objects with XYZ data from shue model
        nonshue_dfs,nonshue_names- rest of DataFrames, non symmetric
        alpha_points, da- used to set the slices for comparison
    Outputs
        stats, statnames- single valued and string list
    """
    stats = []
    statnames = []
    ###################################################################
    #Time
    stats.append(timedf.values[0])
    statnames.append('Time [UTC]')
    ###################################################################
    #Compare flow and shue98
    if False:
        if shue98.empty | flow.empty:
            print('Missing data for shue98-flow compare!')
        else:
            #Expand flow into a list of df at each angle
            flow_expand, flow_names = expand_azimuth([flow],['flow'],
                                                     alpha_points, da)
            maxdiff, maxdiff_xloc, maxdiff_aloc = get_maxdiff([shue98],
                                                        flow_expand,-10,12)
            max_meandiff, max_meandiff_aloc = get_meandiff([shue98],
                                                        flow_expand,-10,12)
            for stat in [maxdiff, maxdiff_xloc, maxdiff_aloc,
                         max_meandiff, max_meandiff_aloc]:
                stats.append(stat)
            for name in ['maxdiff', 'maxdiff_xloc', 'maxdiff_aloc',
                         'max_meandiff', 'max_meandiff_aloc']:
                statnames.append('flow-shue98_'+name)
    ###################################################################
    return stats, statnames


def single_file_proc(dflist, dfnames, alpha_points, da, make_fig,
                     get_stats, outpath):
    """Function takes set of magnetopause surfaces at fixed time and
        calculates statistics/makes figures
    Inputs
        timedf- Series object with a single datetime entry
        shue98..hybrid- DataFrame objects with X [R], Y [R], Z [R] data
        alpha_points, da- used to set the slices for comparison
    Outputs
        stats, statnames
    """
    timedf, nonshue_dfs, nonshue_names, shuedfs, shuenames=(pd.DataFrame(),
                                                            [], [], [], [])
    #identify timedf, and shuedfs 
    for df in enumerate(dflist):
        if dfnames[df[0]].find('time') != -1:
            timedf = df[1]
        elif dfnames[df[0]].find('shue') != -1:
            shuedfs.append(df[1])
            shuenames.append(dfnames[df[0]])
        else:
            nonshue_dfs.append(df[1])
            nonshue_names.append(dfnames[df[0]])
    #check that time exists
    if timedf.empty:
        print('No time given! file not processed')
        return None
    #add alpha and height as a column in the 3D dataset
    for df in enumerate(nonshue_dfs):
        if not df[1].empty:
            alpha = pd.DataFrame(rad2deg(np.arctan2(df[1]['Z [R]'],
                                                    df[1]['Y [R]'])),
                                 columns=['alpha(deg)'])
            height = pd.DataFrame(np.sqrt(df[1]['Z [R]'].values**2+
                                          df[1]['Y [R]'].values**2),
                                  columns=['height [R]'])
            nonshue_dfs[df[0]].loc[:,'alpha(deg)'] = alpha
            nonshue_dfs[df[0]].loc[:,'height [R]'] = height
    for df in enumerate(shuedfs):
        if not df[1].empty:
            alpha = pd.DataFrame(rad2deg(np.arctan2(df[1]['Z [R]'],
                                                    df[1]['Y [R]'])),
                                 columns=['alpha(deg)'])
            height = pd.DataFrame(np.sqrt(df[1]['Z [R]'].values**2+
                                          df[1]['Y [R]'].values**2),
                                  columns=['height [R]'])
            shuedfs[df[0]].loc[:,'alpha(deg)'] = alpha
            shuedfs[df[0]].loc[:,'height [R]'] = height
    if make_fig:
        #expanded list for (shue + nonshue*nalpha)
        expanded_df_list, expanded_df_names = expand_azimuth(nonshue_dfs,
                                                             nonshue_names,
                                                             alpha_points,
                                                             da)
        for df in enumerate(shuedfs):
            expanded_df_list.append(df[1])
            expanded_df_names.append(shuenames[df[0]])
        prepare_figures(expanded_df_list, expanded_df_names, alpha_points,
                        timedf, outpath)
    if get_stats:
        stats, statnames = prepare_stats(timedf, shuedfs, shuenames,
                                         nonshue_dfs, nonshue_names,
                                         alpha_points, da)
    else:
        stats, statnames = None, None
    return stats, statnames

def process_spatial_mp(data_path_list, nalpha, nslice, *, make_fig=True,
                                                        get_stats=True,
                                                        outpath=None):
    """Top level function reads data files in data_path then
        gives figures and/or single valued data statistics for temporal
        processing
    Inputs
        data_path- path to surface spatial datafiles
        nalpha, nslice- dimensions used to bin datapoints for analysis
        make_fig,get_stats- boolean options for saving figures and stats
    """
    alpha_points, da = linspace(-180,180,nalpha, retstep=True)
    if data_path_list == []:
        print('Nothing to do, no data_paths given!')
    else:
        approved = ['stats', 'shue', 'shue98', 'shue97', 'flow', 'hybrid',
                    'field', 'time']
        #initialize objects
        statsdf = pd.DataFrame()
        for path in data_path_list:
            if path != None:
                bar = Bar('processing data at {}'.format(path),
                          max=count_files(path))
                for datafile in glob.glob(path+'/*.h5'):
                    dflist, dfnames = [],[]
                    with pd.HDFStore(datafile) as store:
                        for key in store.keys():
                            if any([key.find(match)!=-1
                                   for match in approved]):
                                dflist.append(store[key])
                                dfnames.append(key)
                            else:
                                print('key {} not understood!'.format(key))
                    bar.next()
                    stats, statnames = single_file_proc(dflist, dfnames,
                                                       alpha_points, da,
                                                       make_fig, get_stats,
                                                       outpath)
        if len(statsdf.keys()) == 0:
            statsdf = pd.DataFrame(columns=statnames)
        else:
            statsdf = statsdf.append(pd.DataFrame([stats],
                                                  columns=statnames))
    statsdf = statsdf.sort_values(by='Time [UTC]')
    os.system('mkdir '+data_path+'/stats')
    with pd.HDFStore(data_path+'/stats/meshstats.h5') as store:
        store['stats'] = statsdf
    print(statsdf)


if __name__ == "__main__":
    PATH = ['output/mpdynamics/jan27_3surf/meshdata']
    OPATH = 'output/mpdynamics/jan27_3surf/temp/'
    NALPHA = 36
    NSLICE = 60
    process_spatial_mp(PATH, NALPHA, NSLICE, make_fig=True, outpath=OPATH)
