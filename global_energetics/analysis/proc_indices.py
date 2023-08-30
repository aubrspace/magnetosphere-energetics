#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
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
import swmfpy
from global_energetics.analysis.plot_tools import get_omni_cdas
from global_energetics.extract.shue import r0_alpha_1998
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                    general_plot_settings)

def datetimeparser(datetimestring):
    #maybe move this somewhere to call a diff parser depending on file
    return dt.datetime.strptime(datetimestring,'%Y %m %d %H %M %S %f')
def datetimeparser2(datetimestring):
    #maybe move this somewhere to call a diff parser depending on file
    return dt.datetime.strptime(datetimestring,'%d-%m-%Y %H:%M:%S.%f')
def datetimeparser3(datetimestring):
    #maybe move this somewhere to call a diff parser depending on file
    return dt.datetime.strptime(datetimestring,'%Y-%m-%dT%H:%M:%SZ')
def datetimeparser4(datetimestring):
    #maybe move this somewhere to call a diff parser depending on file
    decHour = float(datetimestring.split(' ')[-1])
    hour = str(int(decHour))
    if len(hour)==1: hour = '0'+hour
    minute = str(int((decHour-int(decHour))*60))
    if len(minute)==1: minute = '0'+minute
    sec = str(int(((decHour-int(decHour))*60 - int((decHour-int(decHour))
                  *60))*60))
    if len(sec)==1: sec = '0'+sec
    newstring = ' '.join([datetimestring.split(' ')[0],hour,minute,sec])
    return dt.datetime.strptime(newstring, '%Y%m%d %H %M %S')

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
    import spacepy
    from spacepy import coordinates as coord
    from spacepy import time as spacetime
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
    coord_df = df[keylist].copy(deep=True)
    #Make coordinates object, then add ticks, then convert
    points = coord.Coords(coord_df.values,sys_pair[0],sys_pair[1])
    points.ticks = spacetime.Ticktock(datetimes, 'UTC')
    points_newCoord = points.convert(to_sys_pair[0],to_sys_pair[1])
    for i, key in enumerate(keylist):
        df[key+'_'+to_sys_pair[0]] = points_newCoord.data[:,i]
    #return df with fancy new columns
    return df

def plot_dst(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,**kwargs):
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
        if 'name' in kwargs:
            name = kwargs.get('name')
        else:
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
    legend_loc = 'lower left'
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

def plot_swdensity(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    if 'density' in df.keys():
        densitykey = 'density'
    elif 'rho' in df.keys():
        densitykey = 'rho'
    else:
        print('Cant find density in ',label,df.keys())
        return
    axis.plot(df[timekey],df[densitykey],label=label)
    return

def plot_swIMF(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    plotBx,plotBy,plotBz = True, True, True
    # Find the right keys for Bx, By, Bz if they exist
    # X
    if 'bx' in df.keys():
        bxkey = 'bx'
    else:
        print('Cant find bx in ',label,df.keys())
        plotBx = False
    # Y
    if 'by' in df.keys():
        bykey = 'by'
    else:
        print('Cant find by in ',label,df.keys())
        plotBy = False
    # Z
    if 'bz' in df.keys():
        bzkey = 'bz'
    else:
        print('Cant find bz in ',label,df.keys())
        plotBz = False
    # Plot
    if plotBx:
        axis.plot(df[timekey],df[bxkey],label=label+'_bx')
    if plotBy:
        axis.plot(df[timekey],df[bykey],label=label+'_by')
    if plotBz:
        axis.plot(df[timekey],df[bzkey],label=label+'_bz')
    return

def plot_swV(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    plotVx,plotVy,plotVz = True, True, True
    # Find the right keys for Vx, Vy, Vz if they exist
    # X
    if 'vx' in df.keys():
        vxkey = 'vx'
    else:
        print('Cant find vx in ',label,df.keys())
        plotVx = False
    # Y
    if 'vy' in df.keys():
        vykey = 'vy'
    else:
        print('Cant find by in ',label,df.keys())
        plotVy = False
    # Z
    if 'vz' in df.keys():
        vzkey = 'vz'
    else:
        print('Cant find vz in ',label,df.keys())
        plotVz = False
    # Plot
    if plotVx:
        axis.plot(df[timekey],df[vxkey],label=label+'_vx')
    if plotVy:
        axis.plot(df[timekey],df[vykey],label=label+'_vy')
    if plotVz:
        axis.plot(df[timekey],df[vzkey],label=label+'_vz')
    return

def plot_swpressure(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    if 'pressure' in df.keys():
        pressurekey = 'pressure'
    elif 'p' in df.keys():
        pressurekey = 'p'
    elif 'P' in df.keys():
        pressurekey = 'P'
    else:
        print('Cant find pressure in ',label,df.keys())
        return
    axis.plot(df[timekey],df[pressurekey],label=label)
    return

def plot_symh(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    if 'sym_h' in df.keys():
        symhkey = 'sym_h'
    elif 'dst_sm' in df.keys():
        symhkey = 'dst_sm'
    elif 'dstflx_R=3.0' in df.keys():
        symhkey = 'dstflx_R=3.0'
    else:
        print('Cant find Sym-H in ',label,df.keys())
        return
    axis.plot(df[timekey],df[symhkey],label=label)
    return

def plot_al(axis, df, timekey, label, **kwargs):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict (dict{str:DataFrame}- datasets
        kwargs:
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    if 'al' in df.keys():
        alkey = 'al'
    elif 'AL' in df.keys():
        alkey = 'AL'
    else:
        print('Cant find AL in ',label,df.keys())
        return
    axis.plot(df[timekey],df[alkey],label=label)
    return

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
            #data = df_coord_transform(data, 'times', ['bx','by','bz'],
            #                          ('GSE','car'), ('GSM','car'))
            data['clock'] = np.rad2deg(np.arctan2(data['by'],
                                                  data['bz']))
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

def plot_swpdyn(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,**kwargs):
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
        if 'name' in kwargs:
            name = kwargs.get('name')
        else:
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
                      label=qtkey+'_'+name,Color=Color,
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

def get_supermag_data(start, end, datapath):
    """function gathers supermag data for comparison
        (supermag.py INTERNAL USE ONLY!!!)
    Inputs
        start, end- datatime objects used for supermag.py
    """
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
        supermag.DIR = SUPERMAGDATAPATH
        print('obtaining supermag data')
        return pd.DataFrame(supermag.supermag(start,end)).append(
                pd.Series({'name':'supermag'}),ignore_index=True)
    else:
        return pd.DataFrame().append(
                pd.Series({'name':'supermag'}),ignore_index=True)

def get_swmf_data(datapath,**kwargs):
    """Function reads in swmf geoindex log and log
    Inputs
        datapath- path where log files are
    Outputs
        geoindexdata, swmflogdata- pandas DataFrame with data
    """
    #read files
    geopath = os.path.join(datapath,kwargs.get('prefix','')+'geo*.log')
    logpath = os.path.join(datapath,kwargs.get('prefix','')+'log_*.log')
    swpath = os.path.join(datapath,kwargs.get('prefix','')+'*IMF.dat')
    geoindex = glob.glob(geopath)[0]
    swmflog = glob.glob(logpath)[0]
    solarwind = glob.glob(swpath)[0]
    #get dataset names
    geoindexname = geoindex.split('/')[-1].split('.log')[0]
    swmflogname = swmflog.split('/')[-1].split('.log')[0]
    solarwindname = solarwind.split('/')[-1].split('.dat')[0]
    print('reading: \n\t{}\n\t{}\n\t{}'.format(geoindex,swmflog,solarwind))
    ##SIMULATION INDICES
    geoindexdata = pd.read_csv(geoindex, sep='\s+', skiprows=1,
        parse_dates={'Time [UTC]':['year','mo','dy','hr','mn','sc','msc']},
        date_parser=datetimeparser,
        infer_datetime_format=True, keep_date_col=True)
    geoindexdata['Time [UTC]']+=dt.timedelta(minutes=kwargs.get('tshift',0))
    geoindexdata.index = geoindexdata['Time [UTC]']
    geoindexdata.drop(columns=['Time [UTC]'],inplace=True)
    ##SIMULATION LOG
    swmflogdata = pd.read_csv(swmflog, sep='\s+', skiprows=1,
        parse_dates={'Time [UTC]':['year','mo','dy','hr','mn','sc','msc']},
        date_parser=datetimeparser,
        infer_datetime_format=True, keep_date_col=True)
    swmflogdata['Time [UTC]']+=dt.timedelta(minutes=kwargs.get('tshift',0))
    swmflogdata.index = swmflogdata['Time [UTC]']
    swmflogdata.drop(columns=['Time [UTC]'],inplace=True)
    ##SIMULATION SOLARWIND
    coordsys = pd.read_csv(solarwind).loc[2][0]#NOTE come back to this
    swdata = pd.read_csv(solarwind, sep='\s+', skiprows=[0,1,2,4,5,6,7],
        parse_dates={'Time [UTC]':['year','month','day',
                                   'hour','min','sec','msec']},
        date_parser=datetimeparser,
        infer_datetime_format=True, keep_date_col=True)
        #infer_datetime_format=True, keep_date_col=True).drop(index=[0,1,2])
    swdata['Time [UTC]']+=dt.timedelta(minutes=kwargs.get('tshift',0))
    if coordsys=='GSE':
        print('GSE solarwind data found!!! Converting to GSM...')
        swdata = df_coord_transform(swdata, 'Time [UTC]', ['vx','vy','vz'],
                                  ('GSE','car'), ('GSM','car'))
        swdata = df_coord_transform(swdata, 'Time [UTC]', ['bx','by','bz'],
                                  ('GSE','car'), ('GSM','car'))
        for key in ['bx','by','bz','vx','vy','vz']:
            swdata[key] = swdata[key+'_GSM']
            swdata.drop(columns=[key+'_GSM'])
    swdata.index = swdata['Time [UTC]']
    swdata.drop(columns=['Time [UTC]'],inplace=True)
    #swdata=swdata[swdata.index < geoindexdata.index[-1]]
    #swdata =swdata[swdata.index > geoindexdata.index[0]]
    #solar wind dynamic pressure
    convert = 1.6726e-27*1e6*(1e3)**2*1e9
    swdata['v']=np.sqrt(swdata['vx']**2+swdata['vy']**2+swdata['vz']**2)
    swdata['pdyn'] = swdata['density']*swdata['v']**2*convert
    if kwargs.get('doBetaMa',True):
        #solar wind plasma beta,beta*,Valfven,Ma
        swdata['P']=swdata['density']*1.3807e-23*swdata['temperature']*1e15
        swdata['B']=np.sqrt(swdata['bx']**2+swdata['by']**2+swdata['bz']**2)
        swdata['Beta'] = swdata['P']/(swdata['B']**2/(4*np.pi*1e-7*1e9))
        swdata['Beta*'] = (swdata['P']+swdata['pdyn'])/(swdata['B']**2/
                                                    (4*np.pi*1e-7*1e9))
        swdata['Va'] = np.sqrt(swdata['B']**2/(4*np.pi)/
                           swdata['density']/1.67)*1e5
        swdata['Ma'] = swdata['v']*1e3/swdata['Va']
    if kwargs.get('doStandoff',True):
        #Shue standoff distance
        swdata['r_shue98'], swdata['alpha'] = (
                                r0_alpha_1998(swdata['bz'],swdata['pdyn']))
    if kwargs.get('doCoupls',True):
        #coupling functions
        Cmp = 1000 #Followed
        #https://supermag.jhuapl.edu/info/data.php?page=swdata
        #term comes from Cai and Clauer[2013].
        #Note that SuperMAG lists the term as 100,
        #however from the paper: "From our work,
        #                       α is estimated to be on order of 10^3"
        swdata['clock'] = np.arctan2(swdata['by'],swdata['bz'])
        swdata['Newell']=Cmp*((swdata['v']*1e3)**(4/3)*np.sqrt(
                    (swdata['by']*1e-9)**2+(swdata['bz']*1e-9)**2)**(2/3)*
                                   abs(np.sin(swdata['clock']/2))**(8/3))
        l = 7*6371*1000
        swdata['eps'] = (swdata['B']**2*swdata['v']*
                                    np.sin(swdata['clock']/2)**4*l**2*
                                              1e3*1e-9**2 / (4*np.pi*1e-7))
    #times Time [UTC]
    geoindexdata['times'] = geoindexdata.index
    swmflogdata['times'] = swmflogdata.index
    swdata['times'] = swdata.index
    swdata['dens'] = swdata['density']

    #trim according to kwargs passed
    geoindexdata = geoindexdata.truncate(before=kwargs.get('start'),
                                         after=kwargs.get('end'))
    swmflogdata = swmflogdata.truncate(before=kwargs.get('start'),
                                       after=kwargs.get('end'))
    swdata = swdata.truncate(before=kwargs.get('start'),
                             after=kwargs.get('end'))

    return geoindexdata, swmflogdata, swdata

def prepare_figures(data, path, **kwargs):
    """Function creates figure and axis objects to fill with plots and
        saves them
    Inputs
        data (dict{str:DataFrame})- data available to plot
        outputpath (str)-
        kwargs:
            doSave (bool)- default True
    """
    #make list of all data and print keys in available datasets
    print('Data available:',data.keys())
    timekeys = {'omni':'times',
                'swmf_index':'times',
                'swmf_log':'times',
                'swmf_sw':'times'}
    ######################################################################
    # Sym-H, AL
    if True:
        figname = 'symh_al'
        symh, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                                          figsize=[18,8])
        # Sym-H
        if 'swmf_log' in data.keys():
            plot_symh(ax1, data['swmf_log'], timekeys['swmf_log'],'swmf')
        if 'omni' in data.keys():
            plot_symh(ax1, data['omni'], timekeys['omni'],'omni')
        # AL
        if 'swmf_sw' in data.keys():
            plot_al(ax2, data['swmf_index'], timekeys['swmf_sw'],'swmf')
        if 'omni' in data.keys():
            plot_al(ax2, data['omni'], timekeys['omni'],'omni')
        # Panel decorations
        general_plot_settings(ax1,do_xlabel=False,legend=True,
                              ylabel=r'$Sym-H \left[nT\right]$',
                              legend_loc='lower left')
        general_plot_settings(ax2,do_xlabel=False,legend=True,
                              ylabel=r'$AL \left[nT\right]$',
                              legend_loc='lower left')
        #save
        symh.tight_layout(pad=1)
        figurename = path+'/Symh_AL.png'
        symh.savefig(figurename)
        plt.close(symh)
        print('\033[92m Created\033[00m',figurename)
    ######################################################################
    #SolarWind
    if True:
        figname = 'SolarWind'
        sw, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,sharex=True,
                                                              figsize=[18,20])
        # IMF
        if 'swmf_sw' in data.keys():
            plot_swIMF(ax1, data['swmf_sw'], timekeys['swmf_sw'],'swmf')
        # Density
        if 'swmf_sw' in data.keys():
            plot_swdensity(ax2, data['swmf_sw'], timekeys['swmf_sw'],'swmf')
        # Velocity
        if 'swmf_sw' in data.keys():
            plot_swV(ax3, data['swmf_sw'], timekeys['swmf_sw'],'swmf')
        # Pressure
        if 'swmf_sw' in data.keys():
            plot_swpressure(ax4, data['swmf_sw'], timekeys['swmf_sw'],'swmf')

        # Panel decorations
        general_plot_settings(ax1,do_xlabel=False,legend=True,
                              ylabel=r'$B \left[nT\right]$',
                              legend_loc='lower left')
        general_plot_settings(ax2,do_xlabel=False,legend=True,
                              ylabel=r'$n \left[\#/cm^3\right]$',
                              legend_loc='lower left')
        general_plot_settings(ax3,do_xlabel=False,legend=True,
                              ylabel=r'$V \left[km/s\right]$',
                              legend_loc='lower left')
        general_plot_settings(ax4,do_xlabel=True,legend=True,
                              ylabel=r'$P \left[nPa\right]$',
                              legend_loc='lower left')
        #save
        sw.tight_layout(pad=1)
        figurename = path+'/SolarWind.png'
        sw.savefig(figurename)
        plt.close(sw)
        print('\033[92m Created\033[00m',figurename)
    ######################################################################

def get_expanded_sw(start, end, data_path):
    """Function gets only supermag and omni solar wind data for period
    Inputs
        start, end
    Outputs
        supermag
        omni
    """
    #get supermag and omni
    supermag = get_supermag_data(start, end, data_path)
    supermag['Time [UTC]'] = supermag['times']
    omni = pd.DataFrame(swmfpy.web.get_omni_data(start, end)).append(
            pd.Series({'name':'omni'}), ignore_index=True)
    omni['Time [UTC]'] = omni['times']
    return supermag, omni

def read_indices(data_path, **kwargs):
    """Top level function handles time varying magnetopause data and
        generates figures according to settings set by inputs
    Inputs
        data_path- path to the data
        kwargs:
            read_swmf=True
            read_supermag=False,
            read_omni=True,
            start=dt.datetime(2014,2,18,6,0),
            end=dt.datetime(2014,2,20,0,0)):
    Returns
        data (dict{DataFrame})- swmf_indices, swmf_log, swmf_sw, supermag, omni
    """
    data = {}
    if kwargs.get('read_swmf',True):
        swmf_index, swmf_log, swmf_sw = get_swmf_data(data_path,**kwargs)
        data.update({'swmf_index':swmf_index})
        data.update({'swmf_log':swmf_log})
        data.update({'swmf_sw':swmf_sw})
        #find new start/end times, will have been trimmed already if needed
        kwargs.update({'start':swmf_index.index[0]})
        kwargs.update({'end':swmf_index.index[-1]})
    #get supermag and omni
    if kwargs.get('read_supermag',True):
        import supermag
        #Get start and duration
        start = [kwargs.get('start').year,
                 kwargs.get('start').month,
                 kwargs.get('start').day,
                 kwargs.get('start').hour,
                 kwargs.get('start').minute,
                 kwargs.get('start').second]
        duration = ((kwargs.get('end')-kwargs.get('start')).days*86400+
                    (kwargs.get('end')-kwargs.get('start')).seconds)
        #Read supermag data from the python API call
        (status,smdata) = supermag.SuperMAGGetIndices('aubr',start,
                                                      duration,'all')
        #(status,smdata) = supermag.SuperMAGGetIndices('aubr',[2022,2,2,0,0,0],
        #                                            3600*24*3,'all')
        #Set a timestamp as the column index
        smdata.index = [supermag.sm_DateToYMDHMS(t) for
                        t in smdata['tval'].values]
        data.update({'supermag':smdata})

        # Now get the virtual equivalent by reading the '.mag' file
        if 'magStationFile' in kwargs:
            from global_energetics.extract.magnetometer import read_virtual_SML
            data.update({'vsupermag':read_virtual_SML(
                                               kwargs.get('magStationFile'))})

        '''
        supermag =get_supermag_data(kwargs.get('start'),kwargs.get('end'),
                                    data_path)
        supermag['Time [UTC]'] = supermag['times']
        supermag.index = supermag['times']
        data.update({'supermag':supermag})
        '''
    if kwargs.get('read_omni',True):
        print(kwargs.get('start'),kwargs.get('end'))
        omni = pd.DataFrame(swmfpy.web.get_omni_data(
                           kwargs.get('start',dt.datetime(2014,2,18,6,0)),
                           kwargs.get('end',dt.datetime(2014,2,20,0,0))))
        omni.index = omni['times']
        omni['Time [UTC]'] = omni['times']
        if all(omni['sym_h'].isna()):#look CDAS if event too new for omni
            omni2 = get_omni_cdas(
                           kwargs.get('start',dt.datetime(2014,2,18,6,0)),
                           kwargs.get('end',dt.datetime(2014,2,20,0,0)))
            omni['sym_h']=omni2['sym_h']
        data.update({'omni':omni})
    return data

if __name__ == "__main__":
    # Setup data paths
    datapath = sys.argv[1]

    # Set default parameters
    plt.rcParams.update(pyplotsetup(mode='digital_presentation'))

    # Read files for data
    data = {}
    data = read_indices(datapath,read_supermag=False,
                        start=dt.datetime(2000,6,24,4),
                        end=dt.datetime(2000,6,24,6))

    # Plot some figures
    prepare_figures(data,os.path.join(datapath,'figures'))
