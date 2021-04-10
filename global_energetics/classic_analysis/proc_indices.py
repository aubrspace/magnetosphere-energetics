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
import swmfpy
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *


def df_coord_transform(df, timekey, keylist, sys_pair, to_sys_pair):
    """Function converts coordinates from given columns from dataframe
    Inputs
        df- pandas dataframe
        timekey- key to access time data, assuming datetime format
        keylist- list of keys to transform, !!MUST BY XYZ equiv IN ORDER!!
        sys_pair- tuple ('CoordinateSys', 'Type') eg: ('GSM','car')
        to_sys_pair- tuple for what to transform into
    Outputs
        df- dataframe containing transformed keylist vars+[yr,mo,dy...]
    """
    #break datetime into components (weird way to getaround a type issue)
    df['yr'] = [entry.year for entry in df[timekey]]
    df['mn'] = [entry.month for entry in df[timekey]]
    df['dy'] = [entry.day for entry in df[timekey]]
    df['hr'] = [entry.hour for entry in df[timekey]]
    df['min'] = [entry.minute for entry in df[timekey]]
    df['sec'] = [entry.second for entry in df[timekey]]
    #reconstruct list of datetimes from various df columns
    datetimes = []
    for index in df.index:
        if not np.isnan(df['yr'][index]):
            datetimes.append(dt.datetime(int(df['yr'][index]),
                                         int(df['mn'][index]),
                                         int(df['dy'][index]),
                                         int(df['hr'][index]),
                                         int(df['min'][index]),
                                         int(df['sec'][index])))
    #get dataframe with just keylist variables
    coord_df = pd.DataFrame()
    for key in keylist:
        coord_df[key] = df[key][0:-1]
    #Make coordinates object, then add ticks, then convert
    points = coord.Coords(coord_df.values,sys_pair[0],sys_pair[1])
    points.ticks = spacetime.Ticktock(datetimes, 'UTC')
    trans_points = points.convert(to_sys_pair[0], to_sys_pair[1])
    for dim in enumerate([trans_points.x, trans_points.y, trans_points.z]):
        df[keylist[dim[0]]+to_sys_pair[0]] = np.append(dim[1], np.nan)
    #return df with fancy new columns
    return df

def plot_dst(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower left'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SMR (nT)'
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
        elif name == 'omni':
            qtkey = 'sym_h'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_AL(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots AL (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SML (nT)'
        elif name == 'swmf_index':
            qtkey = 'AL'
        elif name == 'omni':
            qtkey = 'al'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_cpcp(axis, dflist, timekey, ylabel, *,south=False,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'swmf_log':
            if south:
                qtkey = 'cpcps'
            else:
                qtkey = 'cpcpn'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_newell(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Newell CF (Wb/s)'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swdensity(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Density (#/cm^3)'
        elif name == 'swmf_sw':
            qtkey = 'dens'
        elif name == 'omni':
            qtkey = 'density'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swclock(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Clock Angle GSM (deg.)'
        elif name == 'swmf_sw':
            data = df_coord_transform(data, 'times', ['bx','by','bz'],
                                      ('GSE','car'), ('GSM','car'))
            data['clock'] = np.rad2deg(np.arctan2(data['byGSM'],
                                                  data['bzGSM']))
            qtkey = 'clock'
        elif name == 'omni':
            data['clock'] = np.rad2deg(np.arctan2(data['by'],
                                                  data['bz']))
            qtkey = 'clock'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swbybz(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey1 = 'ByGSM (nT)'
            qtkey2 = 'BzGSM (nT)'
        elif name == 'swmf_sw':
            data = df_coord_transform(data, 'times', ['bx','by','bz'],
                                      ('GSE','car'), ('GSM','car'))
            qtkey1 = 'byGSM'
            qtkey2 = 'bzGSM'
        elif name == 'omni':
            qtkey1 = 'by'
            qtkey2 = 'bz'
        else:
            qtkey1 = None
            qtkey2 = None
        if qtkey1 != None or qtkey2 !=None:
            axis.plot(data[timekey],data[qtkey1],
                      label=qtkey1+'_'+name,
                      linewidth=Size, linestyle=ls)
            axis.plot(data[timekey],data[qtkey2],
                      label=qtkey2+'_'+name,
                      linewidth=Size, linestyle='--')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swpdyn(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solarwind dynamic pressure w given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dflist:
        qtkey_alt = None
        name = data['name'].iloc[-1]
        if name == 'supermag':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            qtkey = 'Dyn. Pres. (nPa)'
            data['v'] = np.sqrt(data['VxGSE (nT)']**2+data['VyGSE (nT)']**2+
                                data['VzGSM (nT)']**2)
            data['Pdyn'] = data['Density (#/cm^3)']*data['v']**2*convert
            qtkey_alt = 'Pdyn'
        elif name == 'swmf_sw':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            qtkey = 'Pdyn'
        elif name == 'omni':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['Pdyn'] = data['density']*data['v']**2*convert
            qtkey = 'Pdyn'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls)
        if qtkey_alt != None:
            axis.plot(data[timekey],data[qtkey_alt],
                      label=qtkey_alt+'_'+name,
                      linewidth=Size, linestyle=ls)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swvxvyvz(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solarwind dynamic pressure w given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dflist:
        qtkey1 = None
        qtkey2 = None
        qtkey3 = None
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey1 = 'VxGSE (nT)'
            qtkey2 = 'VyGSE (nT)'
            qtkey3 = 'VzGSM (nT)'
        elif name == 'swmf_sw':
            qtkey1 = 'vx'
            qtkey2 = 'vy'
            data = df_coord_transform(data, 'times', ['vx','vy','vz'],
                                      ('GSE','car'), ('GSM','car'))
            qtkey3 = 'vzGSM'
        elif name == 'omni':
            qtkey1 = 'vx_gse'
            qtkey2 = 'vy_gse'
            qtkey3 = 'vz_gse'
        if False:
            axis.plot(data[timekey],data[qtkey1],
                      label=qtkey1+'_'+name,
                      linewidth=Size, linestyle=ls)
        if qtkey2 != None:
            axis.plot(data[timekey],data[qtkey2],
                      label=qtkey2+'_'+name,
                      linewidth=Size, linestyle='-')
        if False:
            axis.plot(data[timekey],data[qtkey3],
                      label=qtkey3+'_'+name,
                      linewidth=Size, linestyle='-.')
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
    return pd.DataFrame(supermag.supermag(start,end)).append(
            pd.Series({'name':'supermag'}),ignore_index=True)

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
    solarwind = glob.glob(datapath+'*IMF.dat*')[0]
    #get dataset names
    geoindexname = geoindex.split('/')[-1].split('.log')[0]
    swmflogname = swmflog.split('/')[-1].split('.log')[0]
    solarwindname = solarwind.split('/')[-1].split('.dat')[0]
    print('reading: \n\t{}\n\t{}\n\t{}'.format(geoindex,swmflog,solarwind))
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
    swdata = pd.read_csv(solarwind, sep='\s+', skiprows=[1,2,3,4],
            parse_dates={'times':['yr','mn','dy','hr','min','sec','msec']},
            date_parser=datetimeparser,
            infer_datetime_format=True, keep_date_col=True)
    swdata = swdata[swdata['times']<geoindexdata['times'].iloc[-1]]
    swdata = swdata[swdata['times']>geoindexdata['times'].iloc[0]]
    #attach names to each dataset
    geoindexdata = geoindexdata.append(pd.Series({'name':'swmf_index'}),
                                       ignore_index=True)
    swmflogdata = swmflogdata.append(pd.Series({'name':'swmf_log'}),
                                       ignore_index=True)
    swdata = swdata.append(pd.Series({'name':'swmf_sw'}),
                                       ignore_index=True)
    return geoindexdata, swmflogdata, swdata

def prepare_figures(swmf_index, swmf_log, swmf_sw, supermag, omni,
                    outputpath):
    """Function creates figure and axis objects to fill with plots and
        saves them
    Inputs
        swmf_index, swmf_log, supermag- DataFrame objects to pull from
        outputpath
    """
    #make list of all data and print keys in available datasets
    print('Data available:')
    dflist = []
    for df in enumerate([swmf_index, swmf_log, swmf_sw, supermag, omni]):
        if not df[1].empty:
            dflist.append(df[1])
            print(df[1]['name'].iloc[-1]+':\n{}'.format(df[1].keys()))
    ######################################################################
    #SMR and SML index comparison
    if True:
        figname = 'SMR_SML'
        smr, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                                          figsize=[18,8])
        #Time
        timekey = 'times'
        ylabel = 'SMR [nT]'
        plot_dst(ax1, dflist, timekey, ylabel)
        ylabel = 'SML [nT]'
        plot_AL(ax2, dflist, timekey, ylabel)
        smr.savefig(outputpath+'{}.png'.format(figname))
    ######################################################################
    #CPCP
    if False:
        figname = 'CPCP'
        cpcp, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                                          figsize=[18,8])
        #Time
        timekey = 'times'
        ylabel = 'CPCP [V]'
        plot_cpcp(ax1, dflist, timekey, ylabel)
        plot_cpcp(ax2, dflist, timekey, ylabel, south=True)
        cpcp.savefig(outputpath+'{}.png'.format(figname))
    ######################################################################
    #Newell
    if True:
        figname = 'Newell'
        newell, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True,
                                                          figsize=[18,8])
        #Time
        timekey = 'times'
        ylabel = 'Newell [Wb/s]'
        plot_newell(ax1, dflist, timekey, ylabel)
        newell.savefig(outputpath+'{}.png'.format(figname))
    ######################################################################
    #SolarWind
    if True:
        figname = 'SolarWind'
        sw, (ax1, ax2, ax3, ax4,ax5) = plt.subplots(nrows=5, ncols=1, sharex=True,
                                                          figsize=[18,20])
        #Time
        timekey = 'times'
        ylabel = 'SW Density [#/cm^3]'
        plot_swdensity(ax1, dflist, timekey, ylabel)
        ylabel = 'SW By,Bz [nT]'
        plot_swbybz(ax2, dflist, timekey, ylabel)
        ylabel = 'SW Clock Angle [deg]'
        plot_swclock(ax3, dflist, timekey, ylabel)
        ylabel = 'SW Dynamic Pressure [nPa]'
        plot_swpdyn(ax4, dflist, timekey, ylabel)
        ylabel = 'SW Velocity [km/s]'
        plot_swvxvyvz(ax5, dflist, timekey, ylabel)
        sw.savefig(outputpath+'{}.png'.format(figname))
    ######################################################################

def process_indices(data_path, outputpath):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path- path to the data
        outputpaht
    """
    swmf_index, swmf_log, swmf_sw = get_swmf_data(data_path)
    if False:
        #cuttoff data past a certain time
        swmf_datalist = [swmf_index, swmf_log, swmf_sw]
        cuttofftime = dt.datetime(2014,2,19,0,0)
        for df in enumerate(swmf_datalist):
            name = pd.Series({'name':df[1]['name'].iloc[-1]})
            cutdf = df[1][df[1]['times']<cuttofftime].append(name,
                                                ignore_index=True)
            swmf_datalist[df[0]] = cutdf
        [swmf_index, swmf_log, swmf_sw] = swmf_datalist
    #find new start/end times
    start = swmf_index['times'][0]
    end = swmf_index['times'].iloc[-2]
    #get supermag and omni
    supermag = get_supermag_data(start, end, SUPERMAGDATAPATH)
    omni = pd.DataFrame(swmfpy.web.get_omni_data(start, end)).append(
            pd.Series({'name':'omni'}), ignore_index=True)
    #make plots
    prepare_figures(swmf_index, swmf_log, swmf_sw, supermag, omni,
                    outputpath)

if __name__ == "__main__":
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
    if not EXIT:
        #enable latex format
        plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
        datapath = sys.argv[1]
        print('processing indices output at {}'.format(datapath))
        process_indices(datapath,datapath+'indices/')
