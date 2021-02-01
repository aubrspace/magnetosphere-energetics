#!/usr/bin/env python3
#proc_spatial.py
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
#from global_energetics.makevideo import get_time


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

def prepare_figures(dflist, dfnames, alpha, timestamp):
    """Function calls which figures to be plotted
    Inputs
        dflist- list containing all dataframes
        dfnames- names of each dataset
        alpha, timestamp- for figure display
    """
    nonempty_list, nonempty_names, timelabel = prep_slices(dflist, dfnames,
                                                          alpha, timestamp)
    if nonempty_list != []:
        ###################################################################
        #2D curve plot
        if True:
            curve_plot, ax1 = plt.subplots(nrows=1,ncols=1,sharex=True,
                                                            figsize=[18,6])
            ax1.set_xlim([-40,12])
            ax1.set_ylim([0,30])
            plot_2Dpositions(ax1, nonempty_list, nonempty_names, alpha,
                             timestamp)
            curve_plot.savefig('./temp/{}_a{}.png'.format(timelabel,alpha))
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

def prepare_stats(timedf, shue98, shue97, flow, field, hybrid,
                    alpha_points, da):
    """Function calls which statistics are to be calculated
    Inputs
        timestamp- Series object with a single datetime entry
        shue98..hybrid- DataFrame objects with X [R], Y [R], Z [R] data
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
    if True:
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


def single_file_proc(timedf, shue98, shue97, flow, field, hybrid,
                    alpha_points, da, make_fig, get_stats):
    """Function takes set of magnetopause surfaces at fixed time and
        calculates statistics/makes figures
    Inputs
        timedf- Series object with a single datetime entry
        shue98..hybrid- DataFrame objects with X [R], Y [R], Z [R] data
        alpha_points, da- used to set the slices for comparison
    Outputs
        stats, statnames
    """
    dflist = [shue98, shue97, flow, field, hybrid]
    dfnames = ['shue98', 'shue97', 'flow', 'field', 'hybrid']
    #check that time exists
    if timedf.empty:
        print('No time given! file not processed')
        return None
    #add alpha and height as a column in the 3D dataset
    for df in enumerate(dflist):
        if not df[1].empty:
            alpha = pd.DataFrame(rad2deg(np.arctan2(df[1]['Z [R]'],
                                                    df[1]['Y [R]'])),
                                 columns=['alpha(deg)'])
            height = pd.DataFrame(np.sqrt(df[1]['Z [R]'].values**2+
                                          df[1]['Y [R]'].values**2),
                                  columns=['height [R]'])
            dflist[df[0]] = dflist[df[0]].combine(alpha, np.minimum,
                                                  fill_value=1000)
            dflist[df[0]] = dflist[df[0]].combine(height, np.maximum,
                                                  fill_value=-1000)
    shue98, shue97, flow, field, hybrid = [dflist[0], dflist[1], dflist[2],
                                           dflist[3], dflist[4]]
    if make_fig:
        #expanded list for (shue + nonshue*nalpha)
        expanded_df_list, expanded_df_names = expand_azimuth(dflist[0:2],
                                                             dfnames[0:2],
                                                             alpha_points,
                                                             da)
        prepare_figures(expanded_df_list, expanded_df_names, a, timedf)
    if get_stats:
        stats, statnames = prepare_stats(timedf, shue98, shue97, flow,
                                         field, hybrid, alpha_points, da)
    else:
        stats, statnames = None, None
    return stats, statnames

def process_spatial_mp(data_path, nalpha, nslice, *, make_fig=True,
                                                        get_stats=True):
    """Top level function reads data files in data_path then
        gives figures and/or single valued data statistics for temporal
        processing
    Inputs
        data_path- path to surface spatial datafiles
        nalpha, nslice- dimensions used to bin datapoints for analysis
        make_fig,get_stats- boolean options for saving figures and stats
    """
    alpha_points, da = linspace(-180,180,nalpha, retstep=True)
    timedf, shue98df, shue97df, flowdf, hybriddf, fielddf=[pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame(),
                                                           pd.DataFrame()]
    #initialize stats objects for each object of each type
    print(count_files(data_path))
    statsdf = pd.DataFrame()
    bar = Bar('processing spatial data ',max=count_files(data_path))
    for datafile in glob.glob(data_path+'/*.h5'):
        with pd.HDFStore(datafile) as hdf_file:
            keysfound = hdf_file.keys()
            for key in hdf_file.keys():
                if key=='/shue98':
                    shue98df = pd.read_hdf(hdf_file,key,'r')
                elif key=='/shue97':
                    shue97df = pd.read_hdf(hdf_file,key,'r')
                elif key=='/shue':
                    shue98df = pd.read_hdf(hdf_file,key,'r')
                elif key=='/flowline':
                    flowdf = pd.read_hdf(hdf_file,key,'r')
                elif key=='/fieldline':
                    fielddf = pd.read_hdf(hdf_file,key,'r')
                elif key=='/hybrid':
                    hybriddf = pd.read_hdf(hdf_file,key,'r')
                elif key=='/time':
                    timedf = pd.read_hdf(hdf_file,key,'r')
                else:
                    print('key {} not understood!'.format(key))
        bar.next()
        stats, statnames = single_file_proc(timedf, shue98df, shue97df,
                                           flowdf, fielddf, hybriddf,
                                           alpha_points, da, make_fig,
                                           get_stats)
        if len(statsdf.keys()) == 0:
            statsdf = pd.DataFrame(columns=statnames)
        else:
            statsdf = statsdf.append(pd.DataFrame([stats],
                                                  columns=statnames))
    statsdf = statsdf.sort_values(by='Time [UTC]')
    statsdf.to_csv(data_path+'/../spatial_stats.csv')
    print(statsdf)


if __name__ == "__main__":
    PATH = ('/Users/ngpdl/Code/swmf-energetics/output/'+
                     'mpdynamics/jan25_3surf/meshdata')
    NALPHA = 36
    NSLICE = 60
    process_spatial_mp(PATH, NALPHA, NSLICE, make_fig=False)
