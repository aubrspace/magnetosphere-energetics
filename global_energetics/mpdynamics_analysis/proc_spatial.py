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
    for datafile in glob.glob(path+'/*.csv'):
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

def prepare_figures(dflist, dfnames, alpha, timestamp):
    """Function calls which figures to be plotted
    Inputs
        dflist- list containing all dataframes
        dfnames- names of each dataset
        alpha, timestamp- for figure display
    """
    timelabel = ''
    for snip1 in str(timestamp.values[0]).split(' '):
        for snip2 in snip1.split(':'):
            timelabel = timelabel + snip2
    timelabel = timelabel.split('.')[0]
    nonempty_list = []
    nonempty_names = []
    for df in enumerate(dflist):
        if not df[1].empty:
            nonempty_list.append(df[1])
            nonempty_names.append(dfnames[df[0]])
    curve_plot, ax1 = plt.subplots(nrows=1,ncols=1,sharex=True,
                                                 figsize=[18,6])
    ax1.set_xlim([-40,12])
    ax1.set_ylim([0,30])
    if nonempty_list != []:
        plot_2Dpositions(ax1, nonempty_list, nonempty_names, alpha, timestamp)
        #plt.show()
        curve_plot.savefig('./temp/{}_a{}.png'.format(timelabel,alpha))
        plt.close(curve_plot)

def get_maxdiff(df1, df1name, df2, df2name):
    """Function finds the maximum difference between heights of dataframes
    Inputs
        df1, df2- list containing dataframe objects
    Outputs
        maxdiff- single value with maximum difference
        maxdiff_loc- single value with xlocation of max difference
    """
    print('inside get_maxdiff')
    from IPython import embed; embed()
    print('resuming in 3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    pass

def single_file_proc(timestamp, shue98, shue97, flow, field, hybrid,
                    alpha_points, da):
    """Function takes set of magnetopause surfaces at fixed time and
        calculates statistics/makes figures
    Inputs
        time- Series object with a single datetime entry
        shue98..hybrid- DataFrame objects with X [R], Y [R], Z [R] data
        alpha_points, da- used to set the slices for comparison
    Outputs
    """
    dflist = [shue98, shue97, flow, field, hybrid]
    dfnames = ['shue98', 'shue97', 'flow', 'field', 'hybrid']
    #check that time exists
    if timestamp.empty:
        print('No time given! no statistics generated')
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
    #expanded list for (shue + nonshue*nalpha)
    expanded_df_list = dflist[0:2]
    expanded_df_names = dfnames[0:2]
    for a in alpha_points:
        #split off only values in +/- da/2 range (not shue tho bc symm)
        for df in enumerate(dflist[2::]):
            if not df[1].empty:
                expanded_df_list.append(df[1][((df[1]['alpha(deg)']<a+da/4)
                                           &(df[1]['alpha(deg)']>a-da/4))])
                expanded_df_names.append(dfnames[df[0]+2]+'_'+str(a))
    #Saveplots
    prepare_figures(expanded_df_list, expanded_df_names, a, timestamp)
    #get maxdiff between hybrid and shue98
    #get_maxdiff(dflist_temp[0], dfnames[0], dflist_temp[4], dfnames[4])



    '''
    print('inside process_spatial_mp')
    from IPython import embed; embed()
    print('resuming in 3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    '''

def make_figure():
    pass

def make_video():
    pass

def compile_stats():
    pass

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
    spinner = Spinner('reading hdf files ')
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
        spinner.next()
        single_file_proc(timedf, shue98df, shue97df, flowdf, fielddf,
                            hybriddf, alpha_points, da)
    print('keys found: {}'.format(keysfound))
    #for each file:
        #for each alpha bin:
            #call extract_2D_curves, return axis object a=a_i, x vs h
            #if get_stats:
                #call get_curve_stats, return stats object with various data
    #if make_fig:
        #call make_fig1 based on some idea on how to compare figs
    #if make_fig2:
        #call make_fig2 based on some idea on how to compare figs
    #if make_vid:
        #call make_vid based on some idea on how to compare figs
    #if get_stats:
        #call compile_stats with set of gathered data
    #print status report (what figures and files saved where)
    print('inside process_spatial_mp')
    from IPython import embed; embed()


    pass

if __name__ == "__main__":
    PATH = ('/Users/ngpdl/Code/swmf-energetics/output/'+
                     'mpdynamics/jan25_3surf/meshdata')
    NALPHA = 36
    NSLICE = 60
    process_spatial_mp(PATH, NALPHA, NSLICE)
