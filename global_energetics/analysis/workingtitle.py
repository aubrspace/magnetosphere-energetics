#!/usr/bin/env python3
"""Analyze and plot data for the paper "Comprehensive Energy Analysis of
    a Simulated Magnetosphere"(working title)
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
from scipy.stats import linregress
from scipy import integrate
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
from matplotlib import ticker, colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.analysis import analyze_bow_shock
from global_energetics.analysis.plot_tools import (pyplotsetup,safelabel,
                                                   general_plot_settings,
                                                   plot_stack_distr,
                                                   plot_pearson_r,
                                                   plot_stack_contrib)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_hdf import (load_hdf_sort,
                                                 group_subzones,
                                                 get_subzone_contrib)
from global_energetics.analysis.analyze_energetics import plot_power
from global_energetics.analysis.proc_energy_spatial import reformat_lshell
from global_energetics.analysis.proc_timing import (peak2peak,
                                                    pearson_r_shifts)
import seaborn as sns

def combine_closed_rc(data_dict):
    """Function recombines closed and ring current results into one region
    Inputs
        data_dict (dict{str:Dataframe})- analysis data results
    Returns
        closed_new (DataFrame)
    """
    #make deep copies of the existing objects so we don't change anything
    closed_new = data_dict['closed'].copy(deep=True)
    rc_old = data_dict['rc'].copy(deep=True)
    #Change specific columns in our copy to include the rc contribution
    surfacelist = [key.replace('PSB','') for key in closed_new.keys()
                if 'PSB' in key]
    volumelist = [k for k in closed_new.keys() if '[J]' in k]
    for term in surfacelist:
        base, unit = term.split(' ')
        closed_new[base+' '+unit] = (closed_new[base+' '+unit]-
                                     closed_new[base+'L7 '+unit]+
                                     rc_old[base+'LowLat '+unit]+
                                     rc_old[base+'Dayside_inner '+unit])
        closed_new[base+'PSB '+unit] = (closed_new[base+'PSB '+unit]+
                                        rc_old[base+'PSB '+unit])
        closed_new[base+'Dayside '+unit] = (
                                   closed_new[base+'Dayside_reg '+unit]+
                                   rc_old[base+'Dayside_inner '+unit])
        closed_new[base+'Inner '+unit] = (
                                  closed_new[base+'MidLat '+unit]+
                                   rc_old[base+'LowLat '+unit])
        closed_new[base+'PSB '+unit]=(closed_new[base+' '+unit]-
                                      closed_new[base+'Dayside '+unit]-
                                      closed_new[base+'Inner '+unit]-
                                      closed_new[base+'Tail_close '+unit])
        #NOTE theres quite a bit of numerical error getting added here
        #   using pd.DataFrame().describe() it shows mean~8%, std 106%...
    for term in surfacelist:
        closed_new[term] += rc_old[term]
    closed_new['Volume [Re^3]'] = (closed_new['Volume [Re^3]']+
                                   rc_old['Volume [Re^3]'])
    return closed_new

def compile_polar_cap(sphere_input,terminator_input_n,
                      terminator_input_s,**kwargs):
    """Function calculates dayside and nightside rxn rates from components
    Inputs
        sphere_input (pandas DataFrame)
        terminator_input_n (pandas DataFrame)
        terminator_input_s (pandas DataFrame)
        kwargs:
            dotshift (bool)- default False
    Returns
        polar_caps (pandas DataFrame)
    """
    #make a copy
    polar_caps = sphere_input.copy(deep=True)
    #Calculate new terms
    day_dphidt_N=central_diff(abs(polar_caps['Bf_netPolesDayN [Wb]']),120)
    day_dphidt_S=central_diff(abs(polar_caps['Bf_netPolesDayS [Wb]']),120)
    night_dphidt_N=central_diff(abs(polar_caps['Bf_netPolesNightN [Wb]']),120)
    night_dphidt_S=central_diff(abs(polar_caps['Bf_netPolesNightS [Wb]']),120)
    Terminator_N = terminator_input_n['dPhidt_net [Wb/s]']
    Terminator_S = terminator_input_s['dPhidt_net [Wb/s]']
    DayRxn_N = -day_dphidt_N + Terminator_N
    DayRxn_S = -day_dphidt_S + Terminator_S
    NightRxn_N = -night_dphidt_N - Terminator_N
    NightRxn_S = -night_dphidt_S - Terminator_S
    #load terms into copy
    polar_caps['day_dphidt_N'] = day_dphidt_N
    polar_caps['day_dphidt_S'] = day_dphidt_S
    polar_caps['night_dphidt_N'] = night_dphidt_N
    polar_caps['night_dphidt_S'] = night_dphidt_S
    polar_caps['terminator_N'] = Terminator_N
    polar_caps['terminator_S'] = Terminator_S
    polar_caps['DayRxn_N'] = DayRxn_N
    polar_caps['DayRxn_S'] = DayRxn_S
    polar_caps['NightRxn_N'] = NightRxn_N
    polar_caps['NightRxn_S'] = NightRxn_S
    for term in ['day_dphidt','night_dphidt','terminator',
                 'DayRxn','NightRxn']:
        polar_caps[term] = polar_caps[term+'_N']+polar_caps[term+'_S']
    polar_caps['minusdphidt'] = (polar_caps['DayRxn']+
                                 polar_caps['NightRxn'])
    #ExpandingContractingPolarCap ECPC
    polar_caps['ECPC']=['expanding']*len(polar_caps['DayRxn'])
    polar_caps.loc[polar_caps['minusdphidt']>0,'ECPC']='contracting'
    if kwargs.get('dotshift',False):
        #Timshift the dayside values backwards so it maximizes corr
        #   between day and night
        tshift = kwargs.get('tshift',dt.timedelta(minutes=-16))
        for qty in ['DayRxn','ECPC']:
            copy = polar_caps[qty].copy(deep=True)
            copy.index +=tshift
            tstart = polar_caps.index[0]
            polar_caps[qty] = copy.loc[copy.index>=tstart]
            polar_caps[qty].fillna(method='ffill',inplace=True)
    return polar_caps

def prep_for_correlations(data_input, solarwind_input,**kwargs):
    """Function constructs some useful new columns and flags for representation in joint distribution plots and correlation comparisons
    Inputs
        data_input (DataFrame)- input analysis data for a single df
        solarwind_input (DataFrame)- same for some sw variables
        kwargs:
            keyset (str)- tells which strings to get for sheath-region etc
    Return
        data_output (DataFrame)
    """
    ##Gather data
    moments = locate_phase(data_input.index)
    #Analysis
    #Make a deepcopy of dataframe
    data_output = data_input.copy(deep=True)
    dEdt = central_diff(data_output['Utot [J]'],60)
    K_motional = (-1*dEdt-data_output['K_net [W]'])/1e12
    K_static = data_output['K_net [W]']/1e12
    ytime = [float(n) for n in data_output.index.to_numpy()]
    #Logdata bz, pdyn, clock
    solarwind_output = solarwind_input.copy(deep=True)
    tshift = dt.timedelta(seconds=630)
    xtime=[float(n) for n in (solarwind_output.index+tshift).to_numpy()]
    #Shorthand interfaces
    data_output['volume'] = data_output['Volume [Re^3]']
    data_output['energy'] = data_output['Utot [J]']/1e15
    data_output['static'] = data_output['K_net [W]']/1e12
    data_output['minusdEdt'] = -dEdt
    data_output['motion'] = -dEdt/1e12 - data_output['static']
    if kwargs.get('keyset','lobes')== 'lobes':
        data_output['sheath'] = data_output['K_netFlank [W]']/1e12
        data_output['closed'] = data_output['K_netPSB [W]']/1e12
        data_output['passed'] = abs(data_output['sheath']+data_output['closed'])
    elif kwargs.get('keyset','lobes')== 'closed_rc':
        data_output['sheath'] = data_output['K_netDayside [W]']/1e12
        data_output['lobes'] = data_output['K_netPSB [W]']/1e12
        data_output['passed'] = abs(data_output['sheath']+data_output['lobes'])
    for i,xvariable in enumerate(['bz','pdyn','Newell']):
        #Then interpolate other datasource and add it to the copied df
        data_output[xvariable] = np.interp(ytime, xtime,
                                           solarwind_output[xvariable])
    #Add conditional variables to use as hue ID's
    #phase
    data_output['phase'] = ['pre']*len(data_output['static'])
    data_output.loc[(data_output.index>moments['impact']) &
                    (data_output.index<moments['peak2']),'phase'] = 'main'
    data_output.loc[data_output.index>moments['peak2'],'phase'] = 'rec'
    #North/south
    data_output['signBz']=['northIMF']*len(data_output['static'])
    data_output.loc[data_output['bz']<0,'signBz'] = 'southIMF'
    #Expand/contract (motional)
    data_output['expandContract'] = ['expanding']*len(
                                                data_output['static'])
    data_output.loc[data_output['motion']>0,'expandContract']='contracting'
    #inflow/outflow (static)
    data_output['inoutflow']=['inflow']*len(data_output['sheath'])
    data_output.loc[data_output['sheath']>0,'inoutflow']='outflow'
    #Gathering/releasing (-dEdt)
    data_output['gatherRelease']=['gathering']*len(data_output['static'])
    data_output.loc[data_output['minusdEdt']>0,'gatherRelease']='releasing'
    return data_output

def hotfix_cdiff(mpdict,msdict):
    lobes_copy = msdict['lobes'].copy(deep=True)
    closed_copy = msdict['closed'].copy(deep=True)
    mp_copy = mpdict['ms_full'].copy(deep=True)
    for region in [lobes_copy,closed_copy,mp_copy]:
        for target in [k for k in region.keys() if 'Utot_net' in k]:
            ogindex = region[target].copy(deep=True).index
            fdiff = region[target].reset_index(drop=True)
            back = fdiff.copy(deep=True)
            back.index = back.index-1
            back.drop(index=[-1],inplace=True)
            cdiff = (fdiff+back)/2
            region[target] = cdiff.values
    mpdict['ms_full'] = mp_copy
    msdict['lobes'] = lobes_copy
    msdict['closed'] = closed_copy
    return mpdict,msdict

def hotfix_interfSharing(mpdict,msdict,inner):
    """
    """
    # If poles section is missing from the lobes
    if all(msdict['lobes']['K_netPoles [W]'].dropna()==0):
        inner_copy = inner.copy(deep=True)
        lobes_copy = msdict['lobes'].copy(deep=True)
        targets=[inner_copy[k] for k in inner_copy.keys() if 'Poles'in k]
        for target in targets:
            lobes_copy[target.name] = target.values
        #Reassemble polar values, calculated as: (Dnor/Dsou/Nnor/Nsou)
        for whole_piece in [k for k in lobes_copy.keys() if 'Poles 'in k]:
            lobes_copy[whole_piece] = (
                          lobes_copy[whole_piece.replace(' ','DayN ')]+
                          lobes_copy[whole_piece.replace(' ','DayS ')]+
                          lobes_copy[whole_piece.replace(' ','NightN ')]+
                          lobes_copy[whole_piece.replace(' ','NightS ')])
        msdict['lobes'] = lobes_copy
    # If Dayside flux is missing from the closed field
    mp_copy = mpdict['ms_full'].copy(deep=True)
    closed_copy = msdict['closed'].copy(deep=True)
    day = mp_copy[[k for k in mp_copy.keys() if'Dayside_reg'in k]]
    #Fix closed dayside
    if closed_copy[[k for k in closed_copy.keys() if 'Dayside_reg' in k]].empty:
        for k in day.keys(): closed_copy[k]=day_copy[k]
    msdict['closed'] = closed_copy
    return msdict

def hotfix_psb(msdict):
    """Temporary stop gap until we figure out why plasma sheet boundary
        isn't being captured at run time
    Inputs
        msdict
    Return
        msdict (modified)
    """
    lobes = msdict['lobes']
    closed = msdict['closed']
    rc = msdict['rc']
    for df in [lobes, closed, rc]:
        whole_keys=[k for k in df.keys()if ('_injection 'in k
                                        or  '_escape 'in k
                                        or  '_net 'in k
                                         and 'Test'not in k
                                         and 'u' not in k
                                         and 'U' not in k)]
        for k in whole_keys:
            units = '['+k.split('[')[1].split(']')[0]+']'
            psb = k.split(' [')[0]+'PSB '+units
            fl = k.split(' [')[0]+'Flank '+units
            pl = k.split(' [')[0]+'Poles '+units
            tl_l = k.split(' [')[0]+'Tail_lobe '+units
            tl_c = k.split(' [')[0]+'Tail_close '+units
            dy = k.split(' [')[0]+'Dayside_reg '+units
            dyi = k.split(' [')[0]+'Dayside_inner '+units
            l_7 = k.split(' [')[0]+'L7 '+units
            ml = k.split(' [')[0]+'MidLat '+units
            ll = k.split(' [')[0]+'LowLat '+units
            if tl_l in df.keys():
                df[psb] = (df[k]-df[fl]-df[pl]-df[tl_l])
            elif tl_c in df.keys():
                df[psb] = (df[k]-df[dy]-df[l_7]-df[ml]-df[tl_c])
            elif ll in df.keys():
                df[psb] = (df[k]-df[ll]-df[l_7]-df[dyi])

    #msdict['lobes'] = lobes
    #msdict['closed'] = closed
    #msdict['closed'] = closed
    return msdict

def central_diff(dataframe,dt,**kwargs):
    """Takes central difference of the columns of a dataframe
    Inputs
        df (DataFrame)- data
        dt (int)- spacing used for denominator
        kwargs:
            fill (float)- fill value for ends of diff
    Returns
        cdiff (DataFrame)
    """
    working_dataframe = dataframe.copy(deep=True)#I hate having to do this
    ogindex = dataframe.index
    working_dataframe.reset_index(drop=True,inplace=True)
    working_dataframe_fwd = working_dataframe.copy()
    working_dataframe_fwd.index = working_dataframe.index-1
    working_dataframe_bck = working_dataframe.copy()
    working_dataframe_bck.index = working_dataframe.index+1
    cdiff = (working_dataframe_fwd-working_dataframe_bck)/(2*dt)
    cdiff.drop(index=[-1,cdiff.index[-1]],inplace=True)
    cdiff.index = ogindex
    cdiff.fillna(value=kwargs.get('fill',0),inplace=True)
    return cdiff

def get_interfaces(sz):
    """Gets list of interfaces given a subzone region
    Inputs
        sz (DataFrame)- dataset with keys which contain various interfaces
    Return
        interfaces
    """
    interfaces = [k.split('net')[-1].split(' [')[0]
                    for k in sz.keys() if 'K_net' in k]
    return interfaces

def locate_phase(times,**kwargs):
    """Function handpicks phase times for a number of events, returns the
        relevant event markers
    Inputs
        times (Series(datetime))- pandas series of datetime objects
    Returns
        start,impact,peak1,peak2,inter_start,inter_end (datetime)- times
    """
    #Hand picked times
    start = dt.timedelta(minutes=kwargs.get('startshift',60))
    #August2018
    aug2018_impact = dt.datetime(2018,8,25,17,30)
    aug2018_endMain1 = dt.datetime(2018,8,26,5,37)
    aug2018_endMain2 = dt.datetime(2018,8,26,5,37)
    aug2018_inter_start = aug2018_impact
    aug2018_inter_end = aug2018_endMain1
    #August2018
    jun2015_impact = dt.datetime(2015,6,22,19,0)
    jun2015_endMain1 = dt.datetime(2015,6,23,4,30)
    jun2015_endMain2 = dt.datetime(2015,6,23,4,30)
    jun2015_inter_start = jun2015_impact
    jun2015_inter_end = jun2015_endMain1
    #Feb
    #feb2014_impact = dt.datetime(2014,2,18,16,15)
    feb2014_impact = dt.datetime(2014,2,18,17,57)
    #feb2014_endMain1 = dt.datetime(2014,2,19,4,0)
    #feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    feb2014_endMain1 = dt.datetime(2014,2,19,6,45)
    feb2014_endMain2 = dt.datetime(2014,2,19,6,45)
    feb2014_inter_start = dt.datetime(2014,2,18,15,0)
    feb2014_inter_end = dt.datetime(2014,2,18,17,30)
    #Starlink
    #TODO
    starlink_impact = dt.datetime(2022,2,2,23,58)
    starlink_endMain1 = dt.datetime(2022,2,3,11,54)
    #starlink_endMain1 = dt.datetime(2022,2,4,13,10)
    starlink_endMain2 = dt.datetime(2022,2,3,11,54)
    starlink_inter_start = (starlink_endMain1+
                            dt.timedelta(hours=-1,minutes=-30))
    starlink_inter_end = (starlink_endMain1+
                            dt.timedelta(hours=1,minutes=30))
    #May2019
    may2019_impact = dt.datetime(2019,5,14,4,11)
    #may2019_impact = dt.datetime(2019,5,13,19,35)
    may2019_endMain1 = dt.datetime(2019,5,14,7,24)
    may2019_endMain2 = dt.datetime(2019,5,14,7,24)
    #may2019_inter_start = (may2019_endMain1+
    #                       dt.timedelta(hours=-1,minutes=-30))
    #may2019_inter_end = (may2019_endMain1+
    #                     dt.timedelta(hours=1,minutes=30))
    may2019_inter_start = (dt.datetime(2019,5,14,4,0))
    may2019_inter_end = (may2019_endMain1)
    #Aug2019
    aug2019_impact = dt.datetime(2019,8,30,20,56)
    aug2019_endMain1 = dt.datetime(2019,8,31,18,0)
    aug2019_endMain2 = dt.datetime(2019,8,31,18,0)
    aug2019_inter_start = dt.datetime(2019,8,30,19,41)
    aug2019_inter_end = dt.datetime(2019,8,30,22,11)

    #Determine where dividers are based on specific event
    #if abs(times-feb2014_impact).min() < dt.timedelta(minutes=15):
    if any([abs(t-feb2014_impact)<dt.timedelta(days=1) for t in times]):
        impact = feb2014_impact
        peak1 = feb2014_endMain1
        peak2 = feb2014_endMain2
        inter_start = feb2014_inter_start
        inter_end = feb2014_inter_end
    elif any([abs(t-starlink_impact)<dt.timedelta(days=1) for t in times]):
    #elif abs(times-starlink_impact).min() < dt.timedelta(minutes=15):
        impact = starlink_impact
        peak1 = starlink_endMain1
        peak2 = starlink_endMain2
        inter_start = starlink_inter_start
        inter_end = starlink_inter_end
    elif any([abs(t-may2019_impact)<dt.timedelta(days=1) for t in times]):
    #elif abs(times-may2019_impact).min() < dt.timedelta(minutes=15):
        impact = may2019_impact
        peak1 = may2019_endMain1
        peak2 = may2019_endMain2
        inter_start = may2019_inter_start
        inter_end = may2019_inter_end
    elif any([abs(t-aug2018_impact)<dt.timedelta(days=1) for t in times]):
    #elif abs(times-aug2019_impact).min() < dt.timedelta(minutes=15):
        impact = aug2018_impact
        peak1 = aug2018_endMain1
        peak2 = aug2018_endMain2
        inter_start = aug2018_inter_start
        inter_end = aug2018_inter_end
    elif any([abs(t-jun2015_impact)<dt.timedelta(days=1) for t in times]):
    #elif abs(times-aug2019_impact).min() < dt.timedelta(minutes=15):
        impact = jun2015_impact
        peak1 = jun2015_endMain1
        peak2 = jun2015_endMain2
        inter_start = jun2015_inter_start
        inter_end = jun2015_inter_end
    elif any([abs(t-aug2019_impact)<dt.timedelta(days=1) for t in times]):
    #elif abs(times-aug2019_impact).min() < dt.timedelta(minutes=15):
        impact = aug2019_impact
        peak1 = aug2019_endMain1
        peak2 = aug2019_endMain2
        inter_start = aug2019_inter_start
        inter_end = aug2019_inter_end
    else:
        impact = times[0]
        peak1 = times[round(len(times)/2)]
        peak2 = times[round(len(times)/2)]
        #peak2 = times[-1]
        inter_start = peak1-dt.timedelta(minutes=75)
        inter_end = peak1+dt.timedelta(minutes=75)
    #return start, impact, peak1, peak2, inter_start, inter_end
    moments={'start':start,
             'impact':impact,
             'peak1':peak1,
             'peak2':peak2,
             'inter_start':inter_start,
             'inter_end':inter_end}
    return moments

def parse_phase(indata,phasekey,**kwargs):
    """Function returns subset of given data based on a phasekey
    Inputs
        indata (DataFrame, Series, or dict{df/s})- data to be subset
        phasekey (str)- 'main', 'recovery', etc.
    Returns
        phase (same datatype given)
        rel_time (series object of times starting from 0)
    """
    #assert((type(indata)==dict or type(indata)==pd.core.frame.DataFrame or
    #         type(indata)==pd.core.series.Series),
    #        'Data type only excepts dict, DataFrame, or Series')

    #Get time information based on given data type
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        if indata.empty:
            return indata, indata
        else:
            times = indata.index
    elif type(indata) == dict:
        times = [df for df in indata.values() if not df.empty][0].index

    #Find the phase marks based on the event itself
    moments = locate_phase(times, **kwargs)

    #Set condition based on dividers and phase requested
    if 'qt' in phasekey:
        cond=(times>times[0]+moments['start'])&(times<moments['impact'])
    elif 'main' in phasekey:
        if '2' in phasekey:
            cond = (times>moments['peak1']) & (times<moments['peak2'])
        else:
            cond = (times>moments['impact']) & (times<moments['peak1'])
    elif 'rec' in phasekey:
        cond = times>moments['peak1']#NOTE
    elif 'interv' in phasekey:
        #cond=(times>moments['inter_start'])&(times<moments['inter_end'])
        cond = (times>moments['impact']) & (times<moments['peak1'])
    elif 'lineup' in phasekey:
        cond = times>times[0]+moments['start']

    #Reload data filtered by the condition
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        if 'lineup' not in phasekey and 'interv' not in phasekey:
            rel_time = [dt.datetime(2000,1,1)+r for r in
                        indata[cond].index-indata[cond].index[0]]
        else:
            #rel_time = [dt.datetime(2000,1,1)+r for r in
            #            indata[cond].index-peak1]
            rel_time = indata[cond].index-moments['peak1']
        return indata[cond], rel_time
    elif type(indata) == dict:
        phase = indata.copy()
        for key in [k for k in indata.keys() if not indata[k].empty]:
            df = indata[key]
            phase.update({key:df[cond]})
            if 'lineup' not in phasekey and 'interv' not in phasekey:
                rel_time = [dt.datetime(2000,1,1)+r for r in
                            df[cond].index-df[cond].index[0]]
            else:
                #rel_time = [dt.datetime(2000,1,1)+r for r in
                #            df[cond].index-peak1]
                rel_time = df[cond].index-moments['peak1']
        return phase, rel_time

def plot_contour(fig,ax,df,lower,upper,xkey,ykey,zkey,**kwargs):
    """Function sets up and plots contour
    Inputs
        ax (Axis)- axis to plot on
        df (DataFrame)- pandas dataframe to plot from
        lowerlim,upperlim (float)- values for contour range
        xkey,ykey,zkey (str)- keys used to access xyz data
        kwargs:
            doLog (bool)- default False
            cmap (str)- which colormap to use
            will pass along to general_plot_settings (labels, legends, etc)
    Returns
        None
    """
    contour_kw = {}
    cbar_kw = {}
    if kwargs.get('doLog',False):
        lev_exp = np.linspace(lower, upper, kwargs.get('nlev',21))
        levs = np.power(10,lev_exp)
        contour_kw['norm'] = colors.LogNorm()
        cbar_kw['ticks'] = ticker.LogLocator()
    else:
        levs = np.linspace(lower, upper, kwargs.get('nlev',21))
    X_u = np.sort(df[xkey].unique())
    Y_u = np.sort(df[ykey].unique())
    X,Y = np.meshgrid(X_u,Y_u)
    Z = df.pivot_table(index=xkey,columns=ykey,values=zkey).T.values
    #TODO try:
    '''
        df[xkey] = df.index
        tab = df.pivot(index=xkey,columns=ykey,values=zkey)
        #iterate through levels
            tab[lev] = np.NaN
        #reorder the values so the table is in ascending level order
        tab.resample('30s').asfreq().interpolate()
        tab.interpolate(axis=1)
        #Isolate values that have a certain threshold
    '''
    from IPython import embed; embed()
    time.sleep(3)
    conplot = ax.contourf(X,Y,Z, cmap=kwargs.get('cmap','RdBu_r'),
                          levels=levs,extend='both',**contour_kw)
    cbarconplot = fig.colorbar(conplot,ax=ax,label=safelabel(zkey),
                                  **cbar_kw)
    ax.scatter(df[abs(df[zkey]-1e14)<9e13][xkey],
               df[abs(df[zkey]-1e14)<9e13][ykey])
    #for i,path in enumerate(conplot.collections[11].get_paths()):
    #    x,y = path.vertices[:,0], path.vertices[:,1]
    #    ax.plot(x[0:int(len(x)/2)],y[0:int(len(x)/2)])
    general_plot_settings(ax, **kwargs)
    ax.grid()

def plot_quiver(ax,df,lower,upper,xkey,ykey,ukey,vkey,**kwargs):
    """Function sets up and plots contour
    Inputs
        ax (Axis)- axis to plot on
        df (DataFrame)- pandas dataframe to plot from
        lowerlim,upperlim (float)- values for contour range
        xkey,ykey,ukey,vkey (str)- keys used to access xyz data
        kwargs:
            doLog (bool)- default False
            cmap (str)- which colormap to use
            will pass along to general_plot_settings (labels, legends, etc)
    Returns
        None
    """
    #contour_kw = {}
    #cbar_kw = {}
    if kwargs.get('doLog',False):
        pass
        #lev_exp = np.linspace(lower, upper, kwargs.get('nlev',21))
        #levs = np.power(10,lev_exp)
        #contour_kw['norm'] = colors.LogNorm()
        #cbar_kw['ticks'] = ticker.LogLocator()
    else:
        pass
        #levs = np.linspace(lower, upper, kwargs.get('nlev',21))
    X_u = np.sort(df[::5][xkey].unique())
    Y_u = np.sort(df[::5][ykey].unique())
    X,Y = np.meshgrid(X_u,Y_u)
    df_copy = df.copy()
    U = df[::5].pivot_table(index=xkey,columns=ykey,values=ukey).T.values
    V = df_copy[::5].pivot_table(index=xkey,columns=ykey,values=vkey).T.values
    scale = (Y.max()-Y.min())/(X.max()-X.min())
    V = V*scale
    for key in [k for k in df.keys() if k!='L' and k!='t']:
        B = df.pivot_table(index=xkey,columns=ykey,values=key).T.values
        print(key,B.shape)
        #TODO: solved before but currently lost,
        #       why do d/dt derivatives have the wrong shape?!
    from IPython import embed; embed()
    time.sleep(3)
    norm = np.linalg.norm(np.array((U, V)), axis=0)
    quiver = ax.quiver(X,Y,U/norm,V/norm,angles='xy',scale=None)
            #cmap=kwargs.get('cmap','RdBu_r'),
            #              levels=levs,extend='both',**contour_kw)
    #cbarconplot = lshell.colorbar(conplot,ax=ax,label=safelabel(zkey),
    #                              **cbar_kw)
    general_plot_settings(ax, **kwargs)
    #ax.grid()

def stack_energy_region_fig(ds,ph,path,hatches,**kwargs):
    """Stack plot Energy by region
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
        hatches(list[str])- hatches to put on stacks to denote events
    Returns
        None
    """
    #plot
    for i,ev in enumerate(ds.keys()):
        moments = locate_phase(ds[ev]['time'])
        if kwargs.get('tabulate',False):
            Elobes = ds[ev]['msdict'+ph]['lobes']['Utot [J]']
            Eclosed = ds[ev]['msdict'+ph]['closed']['Utot [J]']
            Erc = ds[ev]['msdict'+ph]['rc']['Utot [J]']
            Etotal = Elobes+Eclosed+Erc
            ElobesPercent = (Elobes/Etotal)*100
            lobes = ds[ev]['msdict'+ph]['lobes']
            #Tabulate
            print('\nPhase: '+ph+'\nEvent: '+ev+'\n')
            print('{:<25}{:<20}'.format('****','****'))
            print('{:<25}{:<.3}({:<.3})'.format('Emax_lobes',
                                                Elobes.max()/1e15,
                                                ElobesPercent.max()))
            print('{:<25}{:<.3}({:<.3})'.format('Emin_lobes',
                                                Elobes.min()/1e15,
                                                ElobesPercent.min()))
            print('{:<25}{:<20}'.format('****','****'))
            dEdt = central_diff(lobes['Utot [J]'],60)/1e12
            K1 = lobes['K_netFlank [W]']/1e12
            K2 = lobes['K_netPSB [W]']/1e12
            motion = -dEdt - lobes['K_net [W]']/1e12
            inner = lobes['K_netPoles [W]']/1e12
            tail = lobes['K_netTail_lobe [W]']/1e12
            for flux,label in [(dEdt,'dEdt'),(K1,'K1'),(K2,'K2'),
                               (motion,'motion'),(inner,'inner'),
                               (tail,'tail')]:
                fluxMain = flux[(lobes.index>moments['impact'])&
                                (lobes.index<moments['peak2'])]
                fluxRec = flux[(lobes.index>moments['peak2'])]
                print('{:<15}{:<15}{:<20}'.format('****','****','****'))
                print('{:<15}{:<15}{:<.3}/{:<.3}/{:<.3}'.format(label,
                                                                'main',
                                                        fluxMain.min(),
                                                        fluxMain.max(),
                                                      fluxMain.mean()))
                print('{:<15}{:<15}{:<.3}/{:<.3}/{:<.3}'.format(label,
                                                                'rec',
                                                        fluxRec.min(),
                                                        fluxRec.max(),
                                                      fluxRec.mean()))
        #setup figure
        contr,ax=plt.subplots(1,1,figsize=[18,8])
        if 'lineup' in ph or 'interv' in ph:
            dotimedelta=True
        else: dotimedelta=False
        if not ds[ev]['mp'+ph].empty:
            times=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
            rax = ax.twinx()
            plot_stack_contrib(ax,times,ds[ev]['mp'+ph],
                               ds[ev]['msdict'+ph], legend=(i==0),
                               value_key='Utot [J]',label=ev,ylim=[0,50],
                               factor=1e15,
                               ylabel=r'Energy $\left[ PJ\right]$',
                               legend_loc='upper right', hatch=hatches[i],
                               do_xlabel=(i==len(ds.keys())-1),
                               timedelta=dotimedelta)
        rax.plot(times,ds[ev]['msdict'+ph]['lobes']['Utot [J]']/1e15,
                   color='Navy',linestyle=None)
        rax.set_ylim([0,50])
        #NOTE mark impact w vertical line here
        dtime_impact = (moments['impact']-
                        moments['peak2']).total_seconds()*1e9
        ax.axvline(dtime_impact,color='black',ls='--')
        ax.axvline(0,color='black',ls='--')
        general_plot_settings(rax,
                              do_xlabel=False, legend=False,
                              timedelta=dotimedelta)
        ax.set_xlabel(r'Time $\left[hr:min\right]$')
        #save
        contr.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
        contr.tight_layout(pad=0.8)
        figname = path+'/contr_energy'+ph+'_'+ev+'.png'
        contr.savefig(figname)
        plt.close(contr)
        print('\033[92m Created\033[00m',figname)

def stack_energy_type_fig(ds,ph,path):
    """Stack plot Energy by type (hydro,magnetic) for each region
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
    Returns
        None
    """
    for sz in ['ms_full','lobes','closed','rc']:
        #setup figures
        distr, ax = plt.subplots(len(ds.keys()),1,sharey=True,
                                 sharex=True,figsize=[14,4*len(ds.keys())])
        if len(ds.keys())==1:
            ax = [ax]

        #plot
        for i,ev in enumerate(ds.keys()):
            if 'lineup' in ph or 'interv' in ph:
                dotimedelta=True
            else: dotimedelta=False
            if not ds[ev]['mp'+ph].empty:
                times=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
                moments = locate_phase(times)
                ax[i].fill_between(times,
                               ds[ev]['msdict'+ph]['lobes']['Utot [J]'],
                                   color='tab:blue')
                general_plot_settings(ax[i],
                                ylabel=r'Lobe Energy $\left[ J\right]$',
                                  do_xlabel=True,
                                  legend=False,timedelta=True)
                '''
                plot_stack_distr(ax[i],times,ds[ev]['mp'+ph],
                                 ds[ev]['msdict'+ph], value_set='Energy2',
                                 doBios=False, label=ev,
                                 ylabel=r'Energy $\left[ J\right]$',
                                 legend_loc='upper left',subzone=sz,
                                 timedelta=dotimedelta)
                '''
        #save
        distr.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
        distr.tight_layout(pad=1)
        figname = path+'/distr_energy'+ph+sz+'.png'
        distr.savefig(figname)
        plt.close(distr)
        print('\033[92m Created\033[00m',figname)

def tail_cap_fig(ds,ph,path):
    """Line plot of the tail cap areas
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
    Returns
        None
    """
    #setup figure
    tail,ax=plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                        figsize=[14,4*len(ds.keys())])
    if len(ds.keys())==1:
        ax = [ax]
    #plot
    for i,ev in enumerate(ds.keys()):
        lobe = ds[ev]['msdict'+ph]['lobes']
        inner = ds[ev]['inner_mp'+ph]
        obs = ds[ev]['obs']['swmf_sw'+ph]
        obstime = ds[ev]['swmf_sw_otime'+ph]
        moments = locate_phase(obstime)
        if not lobe.empty:
            #Polar cap areas compared with Newell Coupling function
            fobstime = [float(n) for n in obstime.to_numpy()]#bad hack
            ax[i].fill_between(fobstime, obs['Newell'],label=ev+'By',
                               fc='grey')
            ax[i].spines['left'].set_color('grey')
            ax[i].tick_params(axis='y',colors='grey')
            #ax[i].set_ylabel(r'$B_y \left[ nT\right]$')
            ax[i].set_ylabel(r'Newell $\left[ Wb/s\right]$')
            rax = ax[i].twinx()
            rax.plot(ds[ev]['time'+ph],lobe.get('TestAreaTail_lobe [Re^2]',
                                                   np.zeros(len(lobe))),
                       label=ev+'S')
            rax.plot(ds[ev]['time'+ph],lobe.get('ExB_injectionTail_lobe [W]',
                                                np.zeros(len(lobe)))/-1e9,
                       label=r'$S_{inj}\left[ GW\right]$')
            rax.plot(ds[ev]['time'+ph],lobe.get('ExB_escapeTail_lobe [W]',
                                                np.zeros(len(lobe)))/1e9,
                       label=r'$S_{esc}\left[ GW\right]$')
            general_plot_settings(rax,ylabel=r'Area $\left[ Re^2\right]$'
                                  ,do_xlabel=(i==len(ds.keys())-1),
                                  legend=True,timedelta=('lineup' in ph))
    #save
    tail.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
    tail.tight_layout(pad=1)
    figname = path+'/tail_cap_lobe'+ph+'.png'
    tail.savefig(figname)
    plt.close(tail)
    print('\033[92m Created\033[00m',figname)

def polar_cap_area_fig(ds,ph,path):
    """Line plot of the polar cap areas (projected to inner boundary)
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
    Returns
        None
    """
    #setup figure
    pca,ax=plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                        figsize=[14,4*len(ds.keys())])
    if len(ds.keys())==1:
        ax = [ax]
    #plot
    for i,ev in enumerate(ds.keys()):
        filltime = [float(n) for n in ds[ev]['time'+ph].to_numpy()]
        moments = locate_phase(filltime)
        if 'termdict' in ds[ev].keys():
            #filltime = ds[ev]['time'+ph].astype('float64')
            lobe = ds[ev]['termdict'+ph]['sphere2.65_surface']
            nterm = ds[ev]['termdict'+ph]['terminator2.65north']
            sterm = ds[ev]['termdict'+ph]['terminator2.65south']
        else:
            lobe = ds[ev]['msdict'+ph]['lobes']
        if 'lineup' in ph or 'interv' in ph:
            dotimedelta=True
        else: dotimedelta=False
        if not lobe.empty and 'Bf_netPolesDayN [Wb]' in lobe.keys():
            ax[i].fill_between(filltime,abs(lobe['Bf_netPolesDayN [Wb]']),
                               label=ev+'DayN')
            base = abs(lobe['Bf_netPolesDayN [Wb]'])
            ax[i].fill_between(filltime,base,
                               base+abs(lobe['Bf_netPolesNightN [Wb]']),
                               label=ev+'NightN')
            general_plot_settings(ax[i],ylabel=r'$\Phi\left[ Wb\right]$'
                                  ,do_xlabel=(i==len(ds.keys())-1),
                                  timedelta=('lineup' in ph),legend=True)
    #save
    pca.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
    pca.tight_layout(pad=1)
    figname = path+'/polar_cap_area'+ph+'.png'
    pca.savefig(figname)
    plt.close(pca)
    print('\033[92m Created\033[00m',figname)

def polar_cap_flux_stats(dataset,path,**kwargs):
    for i,event in enumerate(dataset.keys()):
        #Gather data
        lobes = dataset[event]['msdict']['lobes']
        solarwind = dataset[event]['obs']['swmf_sw']
        working_lobes = prep_for_correlations(lobes,solarwind)
        working_polar_cap = compile_polar_cap(
                 dataset[event]['termdict']['sphere2.65_surface'],
                 dataset[event]['termdict']['terminator2.65north'],
                 dataset[event]['termdict']['terminator2.65south'])
        for term in ['day_dphidt','night_dphidt','terminator',
                     'DayRxn','NightRxn','minusdphidt','ECPC']:
            working_lobes[term] = working_polar_cap[term]
        #Intervals
        main = working_lobes['phase'] =='main'
        recovery = working_lobes['phase'] =='rec'
        expanding = working_lobes['ECPC']=='expanding'
        contracting = working_lobes['ECPC']=='contracting'
        #Dayside Max[kV]
        dayside_max_main = working_lobes['DayRxn'][main].min()
        dayside_max_rec = working_lobes['DayRxn'][recovery].min()
        #Nightside Max[kV]
        nightside_max_main = working_lobes['NightRxn'][main].max()
        nightside_max_rec = working_lobes['NightRxn'][recovery].max()
        #Dayside Mean [kV]
        dayside_mean_main_exp = working_lobes['DayRxn'][
                                                   main][expanding].mean()
        dayside_mean_rec_exp = working_lobes['DayRxn'][
                                               recovery][expanding].mean()
        dayside_mean_main_con = working_lobes['DayRxn'][
                                                 main][contracting].mean()
        dayside_mean_rec_con = working_lobes['DayRxn'][
                                             recovery][contracting].mean()
        #Nightside Mean [kV]
        nightside_mean_main_exp = working_lobes['NightRxn'][
                                                   main][expanding].mean()
        nightside_mean_rec_exp = working_lobes['NightRxn'][
                                               recovery][expanding].mean()
        nightside_mean_main_con = working_lobes['NightRxn'][
                                                 main][contracting].mean()
        nightside_mean_rec_con = working_lobes['NightRxn'][
                                             recovery][contracting].mean()
        #Net Min [kV]
        net_min_main_exp = working_lobes['minusdphidt'][
                                                    main][expanding].min()
        net_min_rec_exp = working_lobes['minusdphidt'][
                                                recovery][expanding].min()
        net_min_main_con = working_lobes['minusdphidt'][
                                                  main][contracting].min()
        net_min_rec_con = working_lobes['minusdphidt'][
                                              recovery][contracting].min()
        #Net Max [kV]
        net_max_main_exp = working_lobes['minusdphidt'][
                                                    main][expanding].max()
        net_max_rec_exp = working_lobes['minusdphidt'][
                                                recovery][expanding].max()
        net_max_main_con = working_lobes['minusdphidt'][
                                                  main][contracting].max()
        net_max_rec_con = working_lobes['minusdphidt'][
                                              recovery][contracting].max()
        #Net Mean [kV]
        net_mean_main_exp = working_lobes['minusdphidt'][
                                                   main][expanding].mean()
        net_mean_rec_exp = working_lobes['minusdphidt'][
                                               recovery][expanding].mean()
        net_mean_main_con = working_lobes['minusdphidt'][
                                                 main][contracting].mean()
        net_mean_rec_con = working_lobes['minusdphidt'][
                                             recovery][contracting].mean()

        #Tabulate
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Dayside Max','Main','Both',dayside_max_main/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Both',dayside_max_rec/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Nightside Max','Main','Both',nightside_max_main/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Both',nightside_max_rec/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Main','Exp',dayside_mean_main_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Dayside Mean','','Con',dayside_mean_main_con/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Exp',dayside_mean_rec_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'','Con',dayside_mean_rec_con/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Main','Exp',nightside_mean_main_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Nightside Mean','','Con',nightside_mean_main_con/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Exp',nightside_mean_rec_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'','Con',nightside_mean_rec_con/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Main','Exp',net_max_main_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Net Max','','Con',net_max_main_con/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Exp',net_max_rec_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'','Con',net_max_rec_con/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Main','Exp',net_min_main_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Net Min','','Con',net_min_main_con/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Exp',net_min_rec_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'','Con',net_min_rec_con/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Main','Exp',net_mean_main_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              'Net Mean','','Con',net_mean_main_con/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'Rec','Exp',net_mean_rec_exp/1000))
        print('{:<14}{:<5}{:<5}{:<.3}'.format(
              ''           ,'','Con',net_mean_rec_con/1000))
        print('{:<14}{:<5}{:<5}{:<20}'.format('****','****','****','****'))
        #Net Mean [kV]
        if kwargs.get('save',False):
            pass
def polar_cap_flux_fig(dataset,phase,path):
    """Line plot of the polar cap areas (projected to inner boundary)
    Inputs
        dataset (DataFrame)- Main data object which contains data
        phase (str)- phase ('main','rec', etc)
        path (str)- where to save figure
    Returns
        None
    """
    #plot
    for i,event in enumerate(dataset.keys()):
        '''
        if 'lineup' in phase or 'interv' in phase:
            dotimedelta=True
        else: dotimedelta=False
        '''
        dotimedelta = True
        #setup figures
        polar_cap_figure1,axis=plt.subplots(1,1,figsize=[16,8])
        #Gather data
        working_polar_cap = compile_polar_cap(
                 dataset[event]['termdict'+phase]['sphere2.65_surface'],
                 dataset[event]['termdict'+phase]['terminator2.65north'],
                 dataset[event]['termdict'+phase]['terminator2.65south'])
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])

        axis.plot(times,working_polar_cap['DayRxn']/1000,label='Day')
        axis.plot(times,working_polar_cap['NightRxn']/1000,label='Night')
        axis.fill_between(times,(working_polar_cap['DayRxn']+
                                 working_polar_cap['NightRxn'])/1000,
                          label=r'$\triangle PC$',
                          fc='grey',ec='black')
        right_axis = axis.twinx()
        right_axis.plot(times,
                       (abs(working_polar_cap['Bf_netPolesDayN [Wb]'])+
                        abs(working_polar_cap['Bf_netPolesDayS [Wb]'])+
                        abs(working_polar_cap['Bf_netPolesNightN [Wb]'])+
                        abs(working_polar_cap['Bf_netPolesNightS [Wb]']))
                        /1e9,
                              color='tab:blue',ls='--')
        right_axis.spines['right'].set_color('tab:blue')
        right_axis.tick_params(axis='y',colors='tab:blue')
        right_axis.set_ylabel(r'Magnetic Flux $\left[ GWb \right]$',
                              color='tab:blue')
        right_axis.set_ylim(0,2)
        general_plot_settings(axis,do_xlabel=True,
                              ylabel=r'$d\Phi/dt\left[ kWb/s\right]$',
                              legend=True,timedelta=dotimedelta)
        #save
        polar_cap_figure1.suptitle('t0='+str(moments['peak1']),
                                   ha='left',x=0.01,y=0.99)
        polar_cap_figure1.tight_layout(pad=1)
        figname = path+'/polar_cap_flux1'+phase+'_'+event+'.png'
        polar_cap_figure1.savefig(figname)
        plt.close(polar_cap_figure1)
        print('\033[92m Created\033[00m',figname)

def stack_volume_fig(ds,ph,path,hatches):
    """Stack plot Volume by region
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
        hatches(list[str])- hatches to put on stacks to denote events
    Returns
        None
    """
    #setup figure
    contr,ax=plt.subplots(len(ds.keys()),1,sharey=True,
                          sharex=True,figsize=[14,4*len(ds.keys())])
    if len(ds.keys())==1:
        ax = [ax]
    #plot
    for i,ev in enumerate(ds.keys()):
        if 'lineup' in ph or 'interv' in ph:
            dotimedelta=True
        else: dotimedelta=False
        if not ds[ev]['mp'+ph].empty:
            #NOTE get lobe only 
            lobes = ds[ev]['msdict'+ph]['lobes']
            times=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
            moments = locate_phase(times)
            plot_stack_contrib(ax[i],times,ds[ev]['mp'+ph],
                               {'lobes':lobes},
                               #ds[ev]['msdict'+ph],
                            value_key='Volume [Re^3]',label=ev,
                            legend=(i==0),
                            ylabel=r'Volume $\left[R_e^3\right]$',
                            legend_loc='upper right', hatch=hatches[i],
                            do_xlabel=(i==len(ds.keys())-1),
                            timedelta=dotimedelta)
            #Calculate quiet time average value and plot as a h line
            rc = ds[ev]['msdict_qt']['rc']['Volume [Re^3]'].mean()
            close=ds[ev]['msdict_qt']['closed']['Volume [Re^3]'].mean()
            lobe=ds[ev]['msdict_qt']['lobes']['Volume [Re^3]'].mean()
            ax[i].axhline(rc,color='grey')
            ax[i].axhline(rc+close,color='grey')
            ax[i].axhline(rc+close+lobe,color='grey')
        #save
        contr.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
        contr.tight_layout(pad=1)
        figname = path+'/contr_volume'+ph+'.png'
        contr.savefig(figname)
        plt.close(contr)
        print('\033[92m Created\033[00m',figname)

def interf_power_fig(ds,ph,path,hatches):
    """Lineplots for event comparison 1 axis per interface
    Inputs
        ds (DataFrame)- Main data object which contains data
        ph (str)- phase ('main','rec', etc)
        path (str)- where to save figure
    Returns
        None
    """
    #Decide on ylimits for interfaces
    #interface_list = ['Dayside_reg','Tail_close','L7','PSB','MidLat',
    #                  'Flank','Tail_lobe','Poles','LowLat']
    '''
    interface_list = ['Dayside_reg','Flank','PSB']
    ylims = [[-8,8],[-5,5],[-10,10]]
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    incr = 7.5
    h_ratios=[3,3,1,1,1,1,4,4,1,1,3,3,1,1,1,1,1,1]
                         figsize=[9,2*len(interface_list)*len(ds.keys())])
    '''
    interface_list = ['Dayside_reg','Flank','PSB']
    ylims = [[-10,10],[-20,10],[-20,10]]
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    incr = 7.5
    h_ratios=[3,3,1,1,1,1,4,4,1,1,3,3,1,1,1,1,1,1]
    interf_fig, ax = plt.subplots(len(ds.keys())*len(interface_list),
                                      sharex=True,
                         figsize=[21,15])
                         #figsize=[18,4*len(interface_list)*len(ds.keys())])
                            #gridspec_kw={'height_ratios':h_ratios})
    if len(ds.keys())==1 and len(interface_list)==1:
        ax = [ax]
    for i,ev in enumerate(ds.keys()):
        dic = ds[ev]['msdict'+ph]
        for j,interf in enumerate(interface_list):
            #if j==0:
            if False:
                filltime=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
                ax[0].fill_between(filltime,
                               ds[ev]['msdict'+ph]['lobes']['Utot [J]'],
                                   color='tab:blue')
                general_plot_settings(ax[0],
                                ylabel=r'Lobe Energy $\left[ J\right]$',
                                  do_xlabel=True,
                                  legend=False,timedelta=True)
            else:
                if interf in closed:
                    sz = 'closed'
                elif interf in lobes:
                    sz = 'lobes'
                elif interf in ring_c:
                    sz = 'rc'
                if not dic[sz].empty and('K_net'+interf+' [W]' in dic[sz]):
                    if 'lineup' in ph or 'interv' in ph:
                        dotimedelta=True
                    else: dotimedelta=False
                    #plot
                    times=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
                    moments = locate_phase(times)
                    labels = [r'Closed - Sheath',
                              r'Lobes - Sheath',
                              r'Lobes - Closed']
                    plot_power(ax[len(ds.keys())*j+i],dic[sz],
                            times,legend=(j==1),
                            inj='K_injection'+interf+' [W]',
                            esc='K_escape'+interf+' [W]',
                            net='K_net'+interf+' [W]',
                            #ylabel=safelabel(interf.split('_reg')[0]),
                            ylabel=labels[j],
                            hatch=hatches[i],ylim=ylims[j],
                        do_xlabel=(j==len(ds.keys())*len(interface_list)-1),
                            legend_loc='lower right',
                            timedelta=dotimedelta)
                    if ph=='_rec':
                        ax[len(ds.keys())*j+i].yaxis.tick_right()
                        ax[len(ds.keys())*j+i].yaxis.set_label_position(
                                                                  'right')
            ax[2].set_xlabel(r'Time $\left[hr:min\right]$')
    #save
    interf_fig.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
    interf_fig.tight_layout(pad=1.2)
    figname = path+'/interf'+ph+'.png'
    interf_fig.savefig(figname)
    plt.close(interf_fig)
    print('\033[92m Created\033[00m',figname)


def region_interface_averages(ds):
    """Bar chart with 2 panels showing avarages by region and interfaces
    Inputs
        ds (DataFrame)- Main data object which contains data
    Returns
        None
    """
    #setup figure
    qt_bar, [ax_top,ax_bot] = plt.subplots(2,1,sharey=False,sharex=False,
                                           figsize=[4*len(ds.keys()),8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Plabel = r'Energy $\left[ PJ\right]$'
    Wlabel = r'Power $\left[ TW\right]$'

    #plot bars for each region
    shifts = np.linspace(-0.1*len(ds.keys()),0.1*len(ds.keys()),
                         len(ds.keys()))
    hatches = ['','*','x','o']
    for i,ev in enumerate(ds.keys()):
        bar_labels = ds[ev]['msdict_qt'].keys()
        bar_ticks = np.arange(len(bar_labels))
        v_keys = [k for k in ['rc','lobes','closed']
                    if not ds[ev]['msdict'][k].empty]
        ax_top.bar(bar_ticks+shifts[i],
         [ds[ev]['msdict_qt'][k]['Utot2 [J]'].mean()/1e15 for k in v_keys],
                      0.4,label=ev,ec='black',fc='silver',hatch=hatches[i])
    ax_top.set_xticklabels(['']+list(ds[ev]['msdict_qt'].keys())+[''])
    general_plot_settings(ax_top,ylabel=Plabel,do_xlabel=False,
                          iscontour=True,legend_loc='upper left')

    #plot bars for each interface
    interface_list = ['Dayside_reg','Tail_close','L7','PSB','MidLat',
                      'Flank','Tail_lobe','Poles','LowLat']
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    for i,ev in enumerate(ds.keys()):
        dic = ds[ev]['msdict_qt']
        clo_inj,lob_inj,rc_inj,bar_labels = [],[],[],[]
        clo_esc,lob_esc,rc_esc  = [],[],[]
        clo_net,lob_net,rc_net  = [],[],[]
        for interf in interface_list:
            bar_labels += [safelabel(interf.split('_reg')[0])]
            #bar_labels += ['']
            z0 = pd.DataFrame({'dummy':[0,0]})
            if interf in closed:
                clo_inj+=[dic['closed'].get(
                 ['K_injection'+interf+' [W]'],z0).mean().values[0]/1e12]
                clo_esc+=[dic['closed'].get(
                 ['K_escape'+interf+' [W]'],z0).mean().values[0]/1e12]
                clo_net += [dic['closed'].get(
                 ['K_net'+interf+' [W]'],z0).mean().values[0]/1e12]
            elif interf in lobes:
                lob_inj+=[dic['lobes'].get(
                 ['K_injection'+interf+' [W]'],z0).mean().values[0]/1e12]
                lob_esc+=[dic['lobes'].get(
                 ['K_escape'+interf+' [W]'],z0).mean().values[0]/1e12]
                lob_net += [dic['lobes'].get(
                 ['K_net'+interf+' [W]'],z0).mean().values[0]/1e12]
            elif interf in ring_c:
                rc_inj += [dic['rc'].get(
                 ['K_injection'+interf+' [W]'],z0).mean().values[0]/1e12]
                rc_esc += [dic['rc'].get(
                 ['K_escape'+interf+' [W]'],z0).mean().values[0]/1e12]
                rc_net += [dic['rc'].get(
                 ['K_net'+interf+' [W]'],z0).mean().values[0]/1e12]
        bar_ticks = np.arange(len(interface_list))+shifts[i]
        ax_bot.bar(bar_ticks,clo_inj+lob_inj+rc_inj,0.4,label=ev+'Inj',
                ec='mediumvioletred',fc='palevioletred',hatch=hatches[i])
        ax_bot.bar(bar_ticks,clo_esc+lob_esc+rc_esc,0.4,label=ev+'Esc',
                ec='peru',fc='peachpuff',hatch=hatches[i])
        ax_bot.bar(bar_ticks,clo_net+lob_net+rc_net,0.4,label=ev+'Net',
                ec='black',fc='silver',hatch=hatches[i])
    ax_bot.set_xticks(bar_ticks)
    ax_bot.set_xticklabels(bar_labels,rotation=15,fontsize=12)
    general_plot_settings(ax_bot,ylabel=Wlabel,do_xlabel=False,
                          legend=False, iscontour=True,
                          timedelta=('lineup' in ph))
    #save
    qt_bar.tight_layout(pad=1)
    figname = outQT+'/quiet_bar_energy.png'
    qt_bar.savefig(figname)
    plt.close(qt_bar)
    print('\033[92m Created\033[00m',figname)

def lshell_contour_figure(ds):
    ##Lshell plot
    for qty in ['Utot2','u_db','uHydro','Utot']:
        for ph,path in [('_main',outMN1),('_rec',outRec)]:
            #Plot contours of quantitiy as well as dLdt of quantity
            for z in [qty+' [J]']:
            #for z in ['dLdt_'+qty]:
                u = 'd'+qty+'dt'
                v = 'd'+qty+' [J]'
                #setup figure
                lshell,ax=plt.subplots(len(ds.keys()),1,sharey=False,
                                 sharex=True,figsize=[14,4*len(ds.keys())])
                if len(ds.keys())==1:
                    ax = [ax]
                for i,ev in enumerate(ds.keys()):
                    closed = ds[ev]['msdict'+ph]['closed']
                    if 'X_subsolar [Re]' in ds[ev]['mp'+ph]:
                        x_sub = ds[ev]['mp'+ph]['X_subsolar [Re]']
                        mptimes = ((x_sub.index-x_sub.index[0]).days*1440+
                                (x_sub.index-x_sub.index[0]).seconds/60)/60
                    if any([qty+'_l' in k for k in closed.keys()]):
                        qt_l=reformat_lshell(ds[ev]['msdict_qt']['closed'],
                                             '_l')
                        l_data = reformat_lshell(closed, '_l')
                        #Set day and night conditions
                        day = l_data['L']>0
                        night = l_data['L']<0
                        qt_dayNight = [qt_l['L']>0, qt_l['L']<0]
                        for j,cond in enumerate([day,night]):
                            #Get quiet time average
                            if 'dLdt_' in z:
                                qt_copy = qt_l[z][qt_dayNight[j]].copy()
                                qt_copy.dropna(inplace=True)
                                qt_ave=qt_copy[abs(qt_copy)!=np.inf].mean()
                                l_copy = l_data[z][cond].copy()
                                l_copy.dropna(inplace=True)
                                l_copy = l_copy[abs(l_copy)!=np.inf]
                                lower = l_copy.min()
                                upper = l_copy.max()
                                low = -8
                                up = 8
                            else:
                                qt_ave=np.log10(
                                            qt_l[z][qt_dayNight[j]].mean())
                                quants = l_data[z][cond].quantile(
                                               [0.1,0.2,0.3,0.4,.5]).values
                                lower =np.floor(np.log10([q for q in quants
                                                               if q>0][0]))
                                upper=np.ceil(np.log10(
                                                l_data[z][cond].max()))
                                dist = max(upper-qt_ave,qt_ave-lower)
                                low = qt_ave-dist
                                up = qt_ave+dist
                            settings ={'legend':False,
                                  'ylabel':r'LShell $\left[R_e\right]$'+ev,
                                       'iscontour':True,
                                       'do_xlabel':(j==1),
                                 'timedelta':('lineup' in ph)}
                            #plot_quiver(ax[i+j],l_data[cond],
                            #            low,up,
                            #            't','L',u,v,
                            #            **settings)
                            plot_contour(lshell,ax[i+j],l_data[cond],
                                         low,up,
                                         't','L',z,
                                         doLog=True,#doLog=('dLdt_' not in z),
                                         **settings)
                            if j==0 and(
                                    'X_subsolar [Re]'in ds[ev]['mp'+ph]):
                                ax[i+j].plot(mptimes,x_sub,color='white')
                #save
                lshell.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
                lshell.tight_layout(pad=0.2)
                figname = path+'/'+z.split(' ')[0]+'_lshell'+ph+'.png'
                lshell.savefig(figname)
                plt.close(lshell)
                print('\033[92m Created\033[00m',figname)

def power_correlations2(dataset, phase, path,optimize_tshift=False):
    """Plots scatter and pearson r comparisons of power terms vs other
       variables to see what may or may not be related
    Inputs
        dataset
        phase
        path
    """
    for i,event in enumerate(dataset.keys()):
        #setup figure
        #correlation_figure1,axis = plt.subplots(1,1,figsize=[16,16])
        correlation_figure1 = plt.figure(figsize=(16,16),
                                         layout="constrained")
        spec = correlation_figure1.add_gridspec(4,4)
        xaxis = correlation_figure1.add_subplot(spec[0,0:3])
        xaxis.xaxis.tick_top()
        yaxis = correlation_figure1.add_subplot(spec[1:4,3])
        yaxis.yaxis.tick_right()
        axis = correlation_figure1.add_subplot(spec[1:4,0:3])
        #tempfig,(xaxis,yaxis) = plt.subplots(2,1,figsize=[16,4])
        #Prepare data
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed_rc']
        solarwind = dataset[event]['obs']['swmf_sw'+phase]
        working_lobes = prep_for_correlations(lobes,solarwind)
        working_polar_cap = compile_polar_cap(
                 dataset[event]['termdict'+phase]['sphere2.65_surface'],
                 dataset[event]['termdict'+phase]['terminator2.65north'],
                 dataset[event]['termdict'+phase]['terminator2.65south'])
        for term in ['day_dphidt','night_dphidt','terminator',
                     'DayRxn','NightRxn','minusdphidt','ECPC']:
            working_lobes[term] = working_polar_cap[term]

        working_lobes['reversedDay'] = -1*working_lobes['DayRxn']

        ## Correlation plot
        # histogram of each sub subset
        number_bins = 50
        for setname,subset,color in[
                   ('main',working_lobes['phase']=='main','magenta'),
                   ('recovery',working_lobes['phase']=='rec','tab:blue')]:
            day1 = working_lobes['DayRxn'][subset][
                                 working_lobes['ECPC']=='expanding']/1000
            day2 = working_lobes['DayRxn'][subset][
                                working_lobes['ECPC']=='contracting']/1000
            night1 = working_lobes['NightRxn'][subset][
                                 working_lobes['ECPC']=='expanding']/1000
            night2 = working_lobes['NightRxn'][subset][
                                 working_lobes['ECPC']=='contracting']/1000
            counts1,bins1 = np.histogram(day1, bins=number_bins,
                                         range=(-400,300),density=True)
            counts2,bins2 = np.histogram(day2, bins=number_bins,
                                         range=(-400,300),density=True)
            xaxis.stairs(counts1+counts2,bins2,fill=False,
                         edgecolor=color)
            xaxis.stairs(counts1,bins1,fill=True,facecolor=color,
                         alpha=0.6)

            counts1,bins1 = np.histogram(night1, bins=number_bins,
                                         range=(-250,450),density=True)
            counts2,bins2 = np.histogram(night2, bins=number_bins,
                                         range=(-250,450),density=True)
            yaxis.stairs(counts1+counts2,bins2,fill=False,
                         edgecolor=color, orientation='horizontal')
            yaxis.stairs(counts1,bins1,fill=True,facecolor=color,
                         orientation='horizontal',alpha=0.6)
        xaxis.set_ylabel('Prob. Density')
        xaxis.grid()
        yaxis.set_xlabel('Prob. Density')
        yaxis.grid()
        for setname,subset,color in[
                   ('main',working_lobes['phase']=='main','magenta'),
                   ('recovery',working_lobes['phase']=='rec','tab:blue')]:
            x1 = working_lobes['DayRxn'][subset][
                                 working_lobes['ECPC']=='expanding']/1000
            y1 = working_lobes['NightRxn'][subset][
                                 working_lobes['ECPC']=='expanding']/1000
            x2 = working_lobes['DayRxn'][subset][
                               working_lobes['ECPC']=='contracting']/1000
            y2 = working_lobes['NightRxn'][subset][
                               working_lobes['ECPC']=='contracting']/1000
            slope1,intercept1,r1,p1,stderr1 = linregress(x1,y1)
            slope2,intercept2,r2,p2,stderr2 = linregress(x2,y2)
            linelabel1=(f'slope:{slope1:.2f}x, r={r1:.2f}')
            linelabel2=(f'slope:{slope2:.2f}x, r={r2:.2f}')
            axis.scatter(x1,y1,label=setname+' Expanding',alpha=0.7,
                         color=color,marker='s')
            axis.scatter(x2,y2,label=setname+' Contracting',alpha=0.7,
                         color=color,
                   marker=mpl.markers.MarkerStyle('s',fillstyle='none'))
            axis.plot(x1[::5],(intercept1+x1*slope1)[::5],
                      label=linelabel1,color=color)
            axis.plot(x2[::5],(intercept2+x2*slope2)[::5],
                      label=linelabel2,color=color,
                      linestyle=':')
        axis.plot(working_lobes['DayRxn']/1000,
                  -working_lobes['DayRxn']/1000,
                  color='black')
        xaxis.set_xlim(-400,300)
        axis.set_xlim(-400,300)
        yaxis.set_ylim(-250,450)
        axis.set_ylim(-250,450)
        axis.legend(loc='upper right',bbox_to_anchor=(1.52,1.53),
                    framealpha=1)
        axis.grid()
        axis.set_xlabel(r'DayRxn $\left[ kV\right]$')
        axis.set_ylabel(r'NightRxn $\left[ kV\right]$')
        correlation_figure1.tight_layout()
        #TODO: plot regression line using r correlation
        #save
        correlation_figure1.suptitle('Event: '+event,ha='left',x=0.01,y=0.99)
        figurename = (path+'/polar_cap_correlation'+phase+'_'+event+'.png')
        correlation_figure1.savefig(figurename)
        print('\033[92m Created\033[00m',figurename)


def quantify_timings(dataset, phase, path,**kwargs):
    """Function plots timing measurements between two quantities
    Inputs
    Returns
        None
    """
    for i,event in enumerate(dataset.keys()):
        ##Gather data
        #   motion, static, sheath,lobe/closed,minusdEdt, passed,
        #   volume, energy, +tags
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed_rc']
        solarwind = dataset[event]['obs']['swmf_sw'+phase]
        working_lobes = prep_for_correlations(lobes,solarwind)
        working_closed = prep_for_correlations(closed,solarwind,
                                               keyset='closed_rc')
        working_polar_cap = compile_polar_cap(
                 dataset[event]['termdict'+phase]['sphere2.65_surface'],
                 dataset[event]['termdict'+phase]['terminator2.65north'],
                 dataset[event]['termdict'+phase]['terminator2.65south'])
        '''
        #NOTE Main phase has a bit of a non-normal dist so let's look at
        #       just expansion times when the distribution is best behaved
        cond = working_lobes['inoutflow']=='inflow'
        working_lobes = working_lobes[cond]
        working_polar_cap = working_polar_cap[cond]
        '''
        r_values = pd.DataFrame()

        #Lobes-Sheath -> Lobes-Closed
        sheath2closed_figure,axis = plt.subplots(1,1,figsize=[16,8])
        time_shifts, r_values['K1-K2']=pearson_r_shifts(
                                                  working_lobes['sheath'],
                                               -1*working_lobes['closed'])
        time_shifts, r_values['K2-K3']=pearson_r_shifts(
                                                 working_closed['lobes'],
                                              -1*working_closed['sheath'])
        time_shifts, r_values['P1-P2']=pearson_r_shifts(
                                             working_polar_cap['DayRxn'],
                                        -1*working_polar_cap['NightRxn'])
        time_shifts, r_values['K1-P1']=pearson_r_shifts(
                                                  working_lobes['sheath'],
                                           -1*working_polar_cap['DayRxn'])
        time_shifts, r_values['K2-P2']=pearson_r_shifts(
                                                  working_lobes['closed'],
                                         -1*working_polar_cap['NightRxn'])
        time_shifts, r_values['P1_motion']=pearson_r_shifts(
                                           -1*working_polar_cap['DayRxn'],
                                                  working_lobes['motion'])
        time_shifts, r_values['P2_motion']=pearson_r_shifts(
                                         -1*working_polar_cap['NightRxn'],
                                                  working_lobes['motion'])
        axis.plot(time_shifts/60,r_values['K1-K2'],
                  label='K1-K2')
        axis.plot(time_shifts/60,r_values['K2-K3'],
                  label='K2-K3')
        axis.plot(time_shifts/60,r_values['P1-P2'],
                  label='P1-P2')

        axis.legend()
        axis.set_xlabel(r'Timeshift $\left[min\right]$')
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.set_ylabel(r'Pearson R')
        axis.grid()
        #save
        sheath2closed_figure.suptitle('Event: '+event,ha='left',x=0.01,y=0.99)
        sheath2closed_figure.tight_layout(pad=1)
        figurename = path+'/internal_timing_rcor'+phase+'_'+event+'.png'
        sheath2closed_figure.savefig(figurename)
        plt.close(sheath2closed_figure)
        print('\033[92m Created\033[00m',figurename)

        #External timings
        solarwind2system_figure1,axis1 = plt.subplots(1,1,figsize=[16,8])
        solarwind2system_figure2,axis2 = plt.subplots(1,1,figsize=[16,8])
        time_shifts, r_values['bz-K1']=pearson_r_shifts(
                                                  -1*solarwind['bz'],
                                               -1*working_lobes['sheath'])
        time_shifts, r_values['newell-K1']=pearson_r_shifts(
                                                 solarwind['Newell'],
                                               -1*working_lobes['sheath'])
        time_shifts, r_values['epsilon-K1']=pearson_r_shifts(
                                                 solarwind['eps'],
                                               -1*working_lobes['sheath'])
        time_shifts, r_values['bz-P1']=pearson_r_shifts(
                                                  -1*solarwind['bz'],
                                           -1*working_polar_cap['DayRxn'])
        time_shifts, r_values['newell-P1']=pearson_r_shifts(
                                                 solarwind['Newell'],
                                           -1*working_polar_cap['DayRxn'])
        time_shifts, r_values['epsilon-P1']=pearson_r_shifts(
                                                 solarwind['eps'],
                                           -1*working_polar_cap['DayRxn'])
        if kwargs.get('tabulate',True):
            r_values.index = time_shifts
            print('\nPhase: '+phase+'\nEvent: '+event+'\n')
            print('{:<14}{:<20}{:<20}'.format('****','****','****'))
            for label, rcurve in r_values.items():
                tshift =rcurve[abs(rcurve)==
                               abs(rcurve).max()].index.values[0]
                extrema = rcurve.loc[tshift]
                print('{:<14}{:<.2}({:<.2})'.format(label,tshift/60,
                                                    extrema))
            print('{:<14}{:<20}{:<20}'.format('****','****','****'))
        axis1.plot(time_shifts/60,r_values['bz-K1'],
                  label='bz-K1')
        axis1.plot(time_shifts/60,r_values['newell-K1'],
                  label='newell-K1')
        axis1.plot(time_shifts/60,r_values['epsilon-K1'],
                  label='epsilon-K1')
        axis2.plot(time_shifts/60,r_values['bz-P1'],
                  label='bz-P1')
        axis2.plot(time_shifts/60,r_values['newell-P1'],
                  label='newell-P1')
        axis2.plot(time_shifts/60,r_values['epsilon-P1'],
                  label='epsilon-P1')
        for axis in [axis1,axis2]:
            axis.legend()
            axis.set_xlabel(r'Timeshift $\left[min\right]$')
            axis.xaxis.set_minor_locator(AutoMinorLocator(5))
            axis.set_ylabel(r'Pearson R')
            axis.grid()
        #save
        solarwind2system_figure1.suptitle('Event: '+event,
                                          ha='left',x=0.01,y=0.99)
        solarwind2system_figure1.tight_layout(pad=1)
        figurename = path+'/solarwind_timing_rcor'+phase+'_'+event+'1.png'
        solarwind2system_figure1.savefig(figurename)
        plt.close(solarwind2system_figure1)
        print('\033[92m Created\033[00m',figurename)
        #save
        solarwind2system_figure1.suptitle('Event: '+event,
                                          ha='left',x=0.01,y=0.99)
        solarwind2system_figure2.tight_layout(pad=1)
        figurename = path+'/solarwind_timing_rcor'+phase+'_'+event+'2.png'
        solarwind2system_figure2.savefig(figurename)
        plt.close(solarwind2system_figure2)
        print('\033[92m Created\033[00m',figurename)

def power_correlations(dataset, phase, path,optimize_tshift=False):
    """Plots scatter and pearson r comparisons of power terms vs other
       variables to see what may or may not be related
    Inputs
        dataset
        phase
        path
    """
    #if len(dataset.keys())==1:
    #    axis = [axis]
    #plot
    for i,event in enumerate(dataset.keys()):
        #setup figure
        power_correlations_figure,axis = plt.subplots(3,1,
                                             sharey=True, sharex=False,
                                          figsize=[8,24*len(ds.keys())])
        ##Gather data
        #Analysis
        lobes = dataset[event]['msdict'+phase]['lobes']
        dEdt = central_diff(lobes['Utot [J]'],60)
        K_motional = -1*dEdt-lobes['K_net [W]']
        K_static = lobes['K_net [W]']
        ytime = [float(n) for n in lobes.index.to_numpy()]
        #Logdata bz,pdyn, clock
        solarwind = ds[ev]['obs']['swmf_sw'+phase]
        if not optimize_tshift:
            #Timeshift by 35Re/(300km/s) = 743s
            tshift = dt.timedelta(seconds=743)
            xtime=[float(n) for n in (solarwind.index+tshift).to_numpy()]
        else:
            #Sweep from 0 to +30min in 2min increments
            #rmax = 0
            smax = 0
            tshift_best = dt.timedelta(seconds=0)
            for tshift_seconds in np.linspace(0,30*60,61):
                tshift = dt.timedelta(seconds=tshift_seconds)
                xtime=[float(n) for n in (solarwind.index+tshift).to_numpy()]
                s = plot_pearson_r(axis[i],xtime,ytime,
                                        solarwind['bz'],K_static/1e12,
                                        skipplot=True)
                if abs(s)>abs(smax):
                    smax = s
                    tshift_best = tshift
            print('shift, s: ',tshift,s,'best: ',tshift_best,smax)
            xtime=[float(n) for n in
                   (solarwind.index+tshift_best).to_numpy()]
        for i,xvariable in enumerate(['pdyn','bz','Newell']):
            r_dEdt = plot_pearson_r(axis[i],xtime,ytime,
                                    solarwind[xvariable],dEdt/1e12,
                                    xlabel=xvariable,
                                    ylabel='Power [TW]')
            r_static = plot_pearson_r(axis[i],xtime,ytime,
                                      solarwind[xvariable],K_static/1e12,
                                      xlabel=xvariable,
                                      ylabel='Power [TW]')
            r_motion = plot_pearson_r(axis[i],xtime,ytime,
                                    solarwind[xvariable],K_motional/1e12,
                                      xlabel=xvariable,
                                      ylabel='Power [TW]')
        #save
        power_correlations_figure.suptitle('Event: '+event,ha='left',x=0.01,y=0.99)
        power_correlations_figure.tight_layout(pad=1)
        figurename = path+'/lobe_power_correlation'+phase+'_'+event+'.png'
        power_correlations_figure.savefig(figurename)
        plt.close(power_correlations_figure)
        print('\033[92m Created\033[00m',figurename)

def lobe_power_histograms(dataset, phase, path,doratios=False):
    """Plots histograms of power for a given phase, intent is to show how
       much energy is gathering or releasing during a given phase
    Inputs
        dataset
        phase
        path
    """
    #setup figure
    power_histograms_figure,axis = plt.subplots(len(dataset.keys()),1,
                                             sharey=True, sharex=True,
                                          figsize=[24,8*len(ds.keys())])
    if len(dataset.keys())==1:
        axis = [axis]
    #plot
    for i,event in enumerate(dataset.keys()):
        #Gather data
        lobes = dataset[event]['msdict'+phase]['lobes']
        dEdt = central_diff(lobes['Utot [J]'],60)
        K_motional = -1*dEdt-lobes['K_net [W]']
        K_static = lobes['K_net [W]']
        if doratios:
            dEdt = dEdt/lobes['Utot [J]']*100*3600
            K_motional = K_motional/lobes['Utot [J]']*100*3600
            K_static = K_static/lobes['Utot [J]']*100*3600
        #Get limits
        K_max = max(K_motional.quantile(0.95),
                    K_static.quantile(0.95),
                    dEdt.quantile(0.95))
        K_min = min(K_motional.quantile(0.05),
                    K_static.quantile(0.05),
                    dEdt.quantile(0.05))
        #Get means
        for datasource,name in [(-dEdt,'-dEdt'),
                                (K_motional,'motional'),
                                (K_static,'static')]:
            #number_bins = int(len(datasource)/5)
            number_bins = 45
            counts,bins = np.histogram(datasource, bins=number_bins,
                                       range=(K_min,K_max))
            #print(name,counts,datasource.mean(),'\n')
            histogram = axis[i].stairs(counts,bins,fill=(name=='-dEdt'),
                                       label=name)
            #Get the color just used to coordinate with a verticle mean
            if name=='-dEdt':
                mean_color = histogram.get_facecolor()
            else:
                mean_color = histogram.get_edgecolor()
            axis[i].axvline(datasource.mean(), ls='--', color=mean_color)
        axis[i].axvline(0, color='black')
        axis[i].legend()
        axis[i].set_xlim(-6e12,6e12)
    #save
    power_histograms_figure.tight_layout(pad=1)
    if doratios:
        figurename = path+'/lobe_power_histograms_ratio'+phase+'.png'
    else:
        figurename = path+'/lobe_power_histograms'+phase+'.png'
    power_histograms_figure.savefig(figurename)
    plt.close(power_histograms_figure)
    print('\033[92m Created\033[00m',figurename)


def lobe_balance_fig(dataset,phase,path):
    """Plot the energy balance on the lobes
    """
#NOTE goal is to show which effect is dominating the regional
#     energy transfer, static flux transfer at the boundary or
#     motion of the boundary of itself.
#     It should be true that when there is flux imbalance:
#       Volume expands and therefore Utot_net is negative
#     But, sometimes it looks like the motional part is resisting
#     which would be like pouring water into a cup but the cup is
#     melting at the bottom so the net amount inside is not increasing
#     as much as it otherwise would be. In this case:
#       Both the energy flux and volume change are acting together
#       to maximize the energy transfer to the closed region so:
#           We should see the peak energy flux into the closed region
#           during these times
    #plot
    for i,event in enumerate(dataset.keys()):
        #setup figure
        total_balance_figure,axis = plt.subplots(1,1,figsize=[16,8])
        if 'lineup' in phase or 'interv' in phase:
            dotimedelta=True
        else: dotimedelta=False
        #NOTE get lobe only 
        lobes = dataset[event]['msdict'+phase]['lobes']
        mp = dataset[event]['mp'+phase]
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        dEdt_cdiff = central_diff(lobes['Utot [J]'],60)
        '''
        dEdt_summed = (#static
                  mp['K_netK1 [W]']+
                  lobes['K_netK2a [W]']+
                  lobes['K_netK2b [W]']+
                  lobes['K_netK3 [W]']+
                  lobes['K_netK4 [W]']+
                  #+motion
                  mp['Utot_netK1 [W]']+
                  lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]']+
                  lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]']+
                  lobes['Utot_netK3 [W]']+
                  lobes['Utot_netK4 [W]'])
        axis.fill_between(times,dEdt_summed/1e12
                         ,label='-dEdt_summed', fc='grey')
        axis.plot(times,-1*dEdt_cdiff/1e12
                         ,label='-dEdt_cdiff', ls='--')
        axis.plot(times,(dEdt_summed+dEdt_cdiff)/1e12,label='error')
        '''
        #Static + Motion terms
        axis.plot(times,(mp['K_netK1 [W]']+
                         mp['Utot_netK1 [W]'])/1e12
                         , label='K+M1')
        axis.plot(times,(lobes['K_netK2a [W]']+
                         lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]'])/1e12
                         , label='K+M2a')
        axis.plot(times,(lobes['K_netK2b [W]']+
                         lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]'])/1e12
                         , label='K+M2b')
        axis.plot(times,(lobes['K_netK3 [W]']+
                         lobes['Utot_netK3 [W]'])/1e12
                         , label='K+M3')
        axis.plot(times,(lobes['K_netK4 [W]']+
                         lobes['Utot_netK4 [W]'])/1e12
                         , label='K+M4')
        #axis.plot(times,-1*dEdt-lobes['K_net [W]'], label='motion')
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        #NOTE mark impact w vertical line here
        #dtime_impact = (dt.datetime(2019,5,14,4,0)-
        #                dt.datetime(2019,5,14,7,45)).total_seconds()*1e9
        #axis.axvline(dtime_impact,color='black',ls='--')
        #axis.axvline(0,color='black',ls='--')
        #axis.axhline(0,color='black')
        '''
        rax = axis.twinx()
        rax.plot(times,
                 dataset[event]['msdict'+phase]['lobes']['Utot [J]']/1e15,
                  color='tab:blue',linestyle='--')
        rax.set_ylim(0,23)
        rax.set_ylabel(r'Energy $\left[ PJ \right]$',color='tab:blue')
        rax.spines['right'].set_color('tab:blue')
        rax.tick_params(axis='y',colors='tab:blue')
        '''
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta,
                              #ylim=[-10,10]
                              )
        #save
        total_balance_figure.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        total_balance_figure.tight_layout(pad=1)
        figurename = path+'/lobe_balance_total'+phase+'_'+event+'.png'
        total_balance_figure.savefig(figurename)
        plt.close(total_balance_figure)
        print('\033[92m Created\033[00m',figurename)

    '''
    #plot
    for i,event in enumerate(dataset.keys()):
        #setup figure
        total_acc_figure,axis = plt.subplots(1,1,figsize=[16,8])
        dotimedelta=True
        lobes = dataset[event]['msdict'+phase]['lobes']
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        dEdt = central_diff(lobes['Utot [J]'],60)
        axis.fill_between(times,(-1*dEdt).cumsum()*60,
                          label='-dEdt', fc='grey')
        axis.plot(times,(lobes['K_net [W]']).cumsum()*60,
                  label='Static')
        axis.plot(times,(-1*dEdt-lobes['K_net [W]']).cumsum()*60,
                  label='motion')
        axis.plot(times,-1*(lobes['Utot [J]']-lobes['Utot [J]'][0]),
                  color='tab:blue',linestyle='--')
        general_plot_settings(axis,do_xlabel=True,legend=True,
                           ylabel=r'Accumulated Energy $\left[ J\right]$',
                              timedelta=dotimedelta)
        #save
        total_acc_figure.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        total_acc_figure.tight_layout(pad=1)
        figurename = path+'/lobe_acc_total'+phase+'_'+event+'.png'
        total_acc_figure.savefig(figurename)
        plt.close(total_acc_figure)
        print('\033[92m Created\033[00m',figurename)

    '''
    #plot
    for i,event in enumerate(dataset.keys()):
        #setup figure
        detailed_balance_figure,(axis,axis2) =plt.subplots(2,1,sharey=True,
                                                        figsize=[28,16])
        if 'lineup' in phase or 'interv' in phase:
            dotimedelta=True
        else: dotimedelta=False
        #NOTE get lobe only 
        mp = dataset[event]['mp'+phase]
        lobes = dataset[event]['msdict'+phase]['lobes']
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        dEdt = central_diff(lobes['Utot [J]'],60)
        #axis.fill_between(times,-1*dEdt,label='-dEdt', fc='grey')
        #axis2.fill_between(times,-1*dEdt,label='-dEdt', fc='grey')
        axis.fill_between(times,
                  mp['K_netK1 [W]']+
                  lobes['K_netK2a [W]']+
                  lobes['K_netK2b [W]']+
                  lobes['K_netK3 [W]']+
                  lobes['K_netK4 [W]']
                         ,label='-dEdt_static', fc='grey')
        axis2.fill_between(times,
                   mp['Utot_netK1 [W]']+
                   lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]']+
                   lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]']+
                   lobes['Utot_netK3 [W]']+
                   lobes['Utot_netK4 [W]']
                         ,label='-dEdt_motion', fc='grey')
        #Static terms
        axis.plot(times,mp['K_netK1 [W]'], label='K1')
        axis.plot(times,lobes['K_netK2a [W]'], label='K2a')
        axis.plot(times,lobes['K_netK2b [W]'], label='K2b')
        axis.plot(times,lobes['K_netK3 [W]'], label='K3')
        axis.plot(times,lobes['K_netK4 [W]'], label='K4')
        #Motion terms
        axis2.plot(times,mp['Utot_netK1 [W]'], label='M1')
        axis2.plot(times,lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]'], label='M2a')
        axis2.plot(times,lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]'], label='M2b')
        axis2.plot(times,lobes['Utot_netK3 [W]'], label='M3')
        axis2.plot(times,lobes['Utot_netK4 [W]'], label='M4')
        #axis.plot(times,-1*dEdt-lobes['K_net [W]'], label='motion')
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ W\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis2,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ W\right]$',
                              ylim=(-1e13,1.2e13),
                              timedelta=dotimedelta)
        #save
        detailed_balance_figure.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        detailed_balance_figure.tight_layout(pad=1)
        figurename = path+'/lobe_balance_detail'+phase+'_'+event+'.png'
        detailed_balance_figure.savefig(figurename)
        plt.close(detailed_balance_figure)
        print('\033[92m Created\033[00m',figurename)

    #plot
    for i,event in enumerate(dataset.keys()):
        #setup figure
        detailed_acc_figure,(axis,axis2) =plt.subplots(2,1,sharey=True,
                                                        figsize=[28,16])
        dotimedelta=True
        lobes = dataset[event]['msdict'+phase]['lobes']
        mp = dataset[event]['mp'+phase]
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        dEdt = central_diff(lobes['Utot [J]'],60)
        axis.fill_between(times,
                  (mp['K_netK1 [W]']+
                   lobes['K_netK2a [W]']+
                   lobes['K_netK2b [W]']+
                   lobes['K_netK3 [W]']+
                   lobes['K_netK4 [W]']).cumsum()*60
                         ,label='-dEdt_static', fc='grey')
        axis2.fill_between(times,
                   (mp['Utot_netK1 [W]']+
                    lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]']+
                    lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]']+
                    lobes['Utot_netK3 [W]']+
                    lobes['Utot_netK4 [W]']).cumsum()*60
                         ,label='-dEdt_motion', fc='grey')
        #Static terms
        axis.plot(times,mp['K_netK1 [W]'].cumsum()*60, label='K1')
        axis.plot(times,lobes['K_netK2a [W]'].cumsum()*60, label='K2a')
        axis.plot(times,lobes['K_netK2b [W]'].cumsum()*60, label='K2b')
        axis.plot(times,lobes['K_netK3 [W]'].cumsum()*60, label='K3')
        axis.plot(times,lobes['K_netK4 [W]'].cumsum()*60, label='K4')
        #Motion terms
        axis2.plot(times,mp['Utot_netK1 [W]'].cumsum()*60, label='M1')
        axis2.plot(times,(lobes['Utot_netK1day&K2a [W]']
                          -mp['Utot_netK1day [W]']).cumsum()*60, label='M2a')
        axis2.plot(times,(lobes['Utot_netK1night&K2b [W]']
                          -mp['Utot_netK1night [W]']).cumsum()*60, label='M2b')
        axis2.plot(times,lobes['Utot_netK3 [W]'].cumsum()*60, label='M3')
        axis2.plot(times,lobes['Utot_netK4 [W]'].cumsum()*60, label='M4')
        #axis.plot(times,-1*dEdt-lobes['K_net [W]'], label='motion')
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ W\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis2,do_xlabel=True,legend=True,
                              ylabel=r'Net Power $\left[ W\right]$',
                              #ylim=(-1e13,1.2e13),
                              timedelta=dotimedelta)
        #save
        detailed_acc_figure.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        detailed_acc_figure.tight_layout(pad=1)
        figurename = path+'/lobe_acc_detail'+phase+'_'+event+'.png'
        detailed_acc_figure.savefig(figurename)
        plt.close(detailed_acc_figure)
        print('\033[92m Created\033[00m',figurename)

def imf_figure(ds,ph,path,hatches):
    imf, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    if len(ds.keys())==1:
        ax = [ax]
    for i,ev in enumerate(ds.keys()):
        filltime = [float(n) for n in ds[ev]['time'+ph].to_numpy()]
        orange = dt.timedelta(minutes=0)
        obs = ds[ev]['obs']['swmf_sw'+ph]
        obstime = ds[ev]['swmf_sw_otime'+ph]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        ax[i].fill_between(ot,obs['B'], ec='dimgrey',fc='thistle',
                               hatch=hatches[i], label=r'$|B|$')
        ax[i].plot(ot,obs['bx'],label=r'$B_x$',c='maroon')
        ax[i].plot(ot,obs['by'],label=r'$B_y$',c='magenta')
        ax[i].plot(ot,obs['bz'],label=r'$B_z$',c='tab:blue')
        if 'lineup' in ph or 'interv' in ph:
            dotimedelta=True
        else: dotimedelta=False
        general_plot_settings(ax[i],ylabel=r'$B\left[nT\right]$'+ev,
                              do_xlabel=(i==len(ds.keys())-1),
                              legend=True,timedelta=dotimedelta)
    #save
    imf.tight_layout(pad=0.8)
    figname = path+'/imf'+ph+'.png'
    imf.savefig(figname)
    plt.close(imf)
    print('\033[92m Created\033[00m',figname)

def swPlasma_figure(ds,ph,path,hatches):
    plasma1, ax1 = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                figsize=[14,4*len(ds.keys())])
    if len(ds.keys())==1:
        ax1 = [ax1]
    for i,ev in enumerate(ds.keys()):
        orange = dt.timedelta(minutes=0)
        obs = ds[ev]['obs']['swmf_sw'+ph]
        obstime = ds[ev]['swmf_sw_otime'+ph]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        ax1[i].fill_between(ot,obs['B'], ec='dimgrey',fc='thistle',
                               hatch=hatches[i], label=r'$|B|$')
        ax1[i].plot(ot,obs['bx'],label=r'$B_x$',c='maroon')
        ax1[i].plot(ot,obs['by'],label=r'$B_y$',c='magenta')
        ax1[i].plot(ot,obs['bz'],label=r'$B_z$',c='tab:blue')
        general_plot_settings(ax1[i],ylabel=r'$B\left[nT\right]$'+ev,
                              do_xlabel=(i==len(ds.keys())-1),
                              legend=True,timedelta=('lineup' in ph))
    #save
    plasma1.tight_layout(pad=0.8)
    figname = path+'/plasma1'+ph+'.png'
    plasma1.savefig(figname)
    plt.close(plasma1)
    print('\033[92m Created\033[00m',figname)

def solarwind_figure(ds,ph,path,hatches,**kwargs):
    """Series of solar wind observatioins/inputs/indices
    """
    if 'lineup' in ph or 'interv' in ph:
        dotimedelta=True
    else: dotimedelta=False
    for i,event in enumerate(ds.keys()):
        dst, ax = plt.subplots(4,1,sharey=False,sharex=False,
                               figsize=[18,4*4])
        #filltime = [float(n) for n in ds[event]['time'+ph].to_numpy()]
        filltime = [float(n) for n in
                    ds[event]['obs']['swmf_sw'+ph].index.to_numpy()]
        sw = ds[event]['obs']['swmf_sw'+ph]
        swtime = ds[event]['swmf_sw_otime'+ph]
        swt = [float(n) for n in swtime.to_numpy()]#bad hack
        sim = ds[event]['obs']['swmf_log'+ph]
        simtime = ds[event]['swmf_log_otime'+ph]
        simt = [float(n) for n in simtime.to_numpy()]#bad hack
        index = ds[event]['obs']['swmf_index'+ph]
        indextime = ds[event]['swmf_index_otime'+ph]
        indext = [float(n) for n in indextime.to_numpy()]#bad hack
        obs = ds[event]['obs']['omni'+ph]
        obstime = ds[event]['omni_otime'+ph]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        #sup = ds[event]['obs']['supermag'+ph]
        #suptime = ds[event]['supermag_otime'+ph]
        #supt = [float(n) for n in suptime.to_numpy()]#bad hack
        if kwargs.get('tabulate',False):
            #start,impact,peak1,peak2,inter_start,inter_end=locate_phase(
            #                                                    sw.index)
            moments = locate_phase(sw.index)
            main_int_newell=integrate.trapz(
                                sw.loc[(sw.index>moments['impact'])&
                                       (sw.index<moments['peak1']),
                                            'Newell']*60)
            rec_int_newell=integrate.trapz(
                                sw.loc[sw.index>moments['peak2'],
                                            'Newell']*60)
            #Tabulate
            print('\n{:<25}{:<20}'.format('****','****'))
            print('{:<25}{:s}'.format('Event',event))
            print('{:<25}{}'.format('Mainphase length',
                                    moments['peak1']-moments['impact']))
            print('{:<25}{:<.3}({:<.3})'.format('Min Dst',
                                                  sim['dst_sm'].min(),
                                               float(obs['sym_h'].min())))
            if all(obs['al'].isna()):
                al = np.zeros(len(obs['al']))
            else:
                al = obs['al']
            print('{:<25}{:<.3}({:<.3})'.format('Min AL',
                                                 index['AL'].min(),
                                                 al.min()))
                                                 #sup['SML (nT)'].min()))
            print('{:<25}{:<.3}/{:<.3}'.format('Min/Mean Bz',
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'bz'].min(),
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'bz'].mean()))
            print('{:<25}{:<.3}/{:<.3}'.format('Max/Mean Pdyn',
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'pdyn'].max(),
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'pdyn'].mean()))
            print('{:<25}{:<.3}/{:<.3}'.format('Max/Mean abs(By)',
                                abs(sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'by']).max(),
                                abs(sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'by']).mean()))
            print('{:<25}{:<.3}/{:<.3}'.format('Max/Mean M_A',
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'Ma'].max(),
                                    sw.loc[(sw.index>moments['impact'])&
                                           (sw.index<moments['peak1']),
                                           'Ma'].mean()))
            print('{:<25}{:<.3}/{:<.3}'.format('Main/Rec Int Newell',
                                       main_int_newell,
                                       rec_int_newell))
            print('{:<25}{:<20}\n'.format('****','****'))
            pass
        #IMF
        ax[0].fill_between(swt,sw['B'], ec='dimgrey',fc='thistle',
                               hatch=hatches[i], label=r'$|B|$')
        ax[0].plot(swt,sw['bx'],label=r'$B_x$',c='maroon')
        ax[0].plot(swt,sw['by'],label=r'$B_y$',c='magenta')
        ax[0].plot(swt,sw['bz'],label=r'$B_z$',c='tab:blue')
        general_plot_settings(ax[0],ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        #Plasma
        ax[1].fill_between(swt,sw['pdyn'],ec='dimgrey',fc='thistle',
                               hatch=hatches[i],label=r'$P_{dyn}$')
        ax[1].plot(swt,sw['Ma'],label=r'$M_{Alf}$',c='magenta')
        ax[1].plot(swt,sw['Beta'],label=r'$\beta$',c='tab:blue')
        general_plot_settings(ax[1],ylabel=r'$M_{Alf},\beta$'+event,
                              legend=True,do_xlabel=False,ylim=[0,25],
                              timedelta=dotimedelta)
        #Dst index
        ax[2].plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        ax[2].plot(ot,obs['sym_h'],label='Obs',c='maroon')
        general_plot_settings(ax[2],ylabel=r'Sym-H$\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        #AL index
        ax[3].plot(indext,index['AL'],label='Sim',c='tab:blue')
        #ax[3].plot(supt,sup['SML (nT)'],label='Obs',c='maroon')
        ax[3].plot(ot,al,label='Obs',c='maroon')
        #Newell coupling function
        ax[3].fill_between(swt, sw['Newell']/100, label=event+'Newell',
                           fc='grey')
        general_plot_settings(ax[3],ylabel=r'AL$\left[nT\right]$,'+
                            r'Newell$\left[ 10\times kWb/s\right]$',
                              do_xlabel=True, legend=True,
                              timedelta=dotimedelta)
        ax[3].set_xlabel(r'Time $\left[hr:min\right]$')
        for axis in ax:
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
        #save
        dst.suptitle('t0='+str(moments['peak1']),ha='left',x=0.01,y=0.99)
        dst.tight_layout(pad=0.8)
        figname = path+'/dst_'+event+'.png'
        dst.savefig(figname)
        plt.close(dst)
        print('\033[92m Created\033[00m',figname)

def diagram_summary(dataset,phase,path):
    """Function plots a summary plot for each timestamp which can be
        played as a video showing the key interface fluxes
    Inputs
        dataset
        path
        kwargs:
    Returns
        None
    """
    dotimedelta=True
    #Load the base background diagram
    background = mpimg.imread('/Users/ngpdl/Desktop/diagram.png')
    for i,event in enumerate(dataset.keys()):
        #Nickname data that we want to use
        mp = dataset[event]['mpdict'+phase]['ms_full']
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        mpt = [float(n) for n in mp.index.to_numpy()]#bad hack
        #working_lobes = prep_for_correlations(lobes,solarwind)
        #working_closed = prep_for_correlations(closed,solarwind,
        #                                       keyset='closed')
        moments = locate_phase(lobes.index)
        sw = dataset[event]['obs']['swmf_sw'+phase]
        swt = [float(n) for n in sw.index.to_numpy()]#bad hack
        sim = dataset[event]['obs']['swmf_log'+phase]
        simt = [float(n) for n in sim.index.to_numpy()]#bad hack
        index = dataset[event]['obs']['swmf_index'+phase]
        indext = [float(n) for n in index.index.to_numpy()]#bad hack
        obs = dataset[event]['obs']['omni'+phase]
        ot = [float(n) for n in obs.index.to_numpy()]#bad hack

        #Set the path for all the image files
        os.makedirs(os.path.join(path,'diagram_summaries_'+event),
                    exist_ok=True)
        #Create a new image for each timestamp
        #for i,now in enumerate(simt[int(len(simt)/2):int(len(simt)/2)+1]):
        ##Setup figure layout
        figure = plt.figure(figsize=(12,12))
        spec = figure.add_gridspec(4,4)
        top = figure.add_subplot(spec[0,0:4])
        main = figure.add_subplot(spec[1:4,0:4])
        # Top dst or something with vertical bar for current time
        #Dst index
        top.plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        top.plot(ot,obs['sym_h'],label='Obs',c='maroon')
        timeline = top.axvline(mpt[0],color='black')
        general_plot_settings(top,ylabel=r'Sym-H$\left[nT\right]$',
                                  do_xlabel=True, legend=True,
                                  timedelta=dotimedelta)
        # Main image with diagram background
        main.imshow(background)
        # Base vectors for each item
        #K1
        k1 = main.quiver([1000],[130],[-1500],[3000],color='green')
        k1text = main.text(1120,160,r'$\vec{K}_1$ ')
        #K2a
        k2a = main.quiver([760],[470],[-3000],[-1500],color='green')
        k2atext = main.text(735,440,r'$\vec{K}_{2a}$ ')
        #K2b
        k2b = main.quiver([1060],[375],[1500],[-3000],color='green')
        k2btext = main.text(1060,330,r'$\vec{K}_{2b}$ ')
        #K3
        k3 = main.quiver([805],[670],[1000],[4000],color='green')
        k3text = main.text(660,770,r'$\vec{K}_3$ ')
        #K4
        k4 = main.quiver([1350],[1150],[5000],[0],color='green')
        k4text = main.text(1125,1100,r'$\vec{K}_4$ ')
        #K5
        k5 = main.quiver([560],[510],[-4000],[1000],color='green')
        k5text = main.text(370,575,r'$\vec{K}_5$ ')
        #K6
        k6 = main.quiver([910],[625],[-4000],[1000],color='green')
        k6text = main.text(920,580,r'$\vec{K}_6$ ')
        #K7
        k7 = main.quiver([1350],[600],[5000],[0],color='green')
        k7text = main.text(1130,700,r'$\vec{K}_7$ ')
        for i,(now,xpos)in enumerate([z for z in zip(mp.index,mpt)]
                                     [1336:1337]
                                     ):
            for vector in [
                           {'vector':k1,
                            'text':k1text,
                            'data':mp['K_netK1 [W]']},
                           {'vector':k2a,
                            'text':k2atext,
                            'data':lobes['K_netK2a [W]']},
                           {'vector':k2b,
                            'text':k2btext,
                            'data':lobes['K_netK2b [W]']},
                           {'vector':k3,
                            'text':k3text,
                            'data':lobes['K_netK3 [W]']},
                           {'vector':k4,
                            'text':k4text,
                            'data':lobes['K_netK4 [W]']},
                           {'vector':k5,
                            'text':k5text,
                            'data':mp['K_netK5 [W]']},
                           {'vector':k6,
                            'text':k6text,
                            'data':closed['K_netK6 [W]']},
                           {'vector':k7,
                            'text':k7text,
                            'data':closed['K_netK7 [W]']}
                           ]:
                factor = vector['data'][now]/max(abs(
                                  vector['data'].quantile([0.05,0.95])))
                vector['vector'].U = np.array([int(
                                            vector['vector'].U*factor)])
                vector['vector'].V = np.array([int(
                                            vector['vector'].V*factor)])
                vector['text'].set_text(vector['text'].get_text()+
                              '{:<.2}'.format(vector['data'][now]/1e12))
                #K2a
                #K2b
                #K3
                #K4
                #K5
                #K6
                #K7
            #Vector for M values:
                #M1
                #M2a
                #M2b
                #M3
                #M4
                #M5
                #M6
                #M7
            #Vector for P values:
                #Pa
                #Pb
            #IMF vector w/ outof plane component:
                #Bxz
                #By
                #Pdyn
            #Bar for energy values
                #Lobes
                #Closed
            #Other items:
                #K1 Day/Night Ratio
                #K5 Day/Night Ratio
            #save
            if now<moments['impact']:
                labelcolor='black'
            elif now>moments['impact']:
                labelcolor='red'
            elif now>moments['peak2']:
                labelcolor='blue'
            timeline.set_xdata([xpos])
            title = figure.suptitle('t='+str(now),ha='left',
                                    x=0.01,y=0.99,color=labelcolor)
            if i==0:
                main.set_xticks([])
                main.set_yticks([])
                figure.tight_layout(pad=0.8)
            figname = os.path.join(path,'diagram_summaries_'+event,
                                   'state_'+str(i)+'.png')
            figure.savefig(figname)
            from IPython import embed; embed()
            print('\033[92m Created\033[00m',figname)

def main_rec_figures(dataset):
    ##Main + Recovery phase
    #hatches = ['','*','x','o']
    hatches = ['','','','']
    #for phase,path in [('_main',outMN1),('_rec',outRec)]:
    for phase,path in [('_lineup',outLine)]:
        #stack_energy_type_fig(dataset,phase,path)
        #stack_energy_region_fig(dataset,phase,path,hatches,tabulate=True)
        #stack_volume_fig(dataset,phase,path,hatches)
        #interf_power_fig(dataset,phase,path,hatches)
        #polar_cap_area_fig(dataset,phase,path)
        #tail_cap_fig(dataset,phase,path)
        #static_motional_fig(dataset,phase,path)
        #solarwind_figure(dataset,phase,path,hatches,tabulate=True)
        #lobe_balance_fig(dataset,phase,path)
        #lobe_power_histograms(dataset, phase, path,doratios=False)
        #lobe_power_histograms(dataset, phase, path,doratios=True)
        #power_correlations(dataset,phase,path,optimize_tshift=True)
        #quantify_timings(dataset, phase, path)
        pass
    #quantify_timings(dataset, '', path)
    #power_correlations2(dataset,'',unfiled, optimize_tshift=False)#Whole event
    #polar_cap_flux_stats(dataset,unfiled)
    diagram_summary(dataset,'',unfiled)

def interval_figures(dataset):
    #hatches = ['','*','x','o']
    hatches = ['','','','']
    for phase,path in [('_interv',outInterv)]:
        #stack_energy_type_fig(dataset,phase,path)
        #stack_energy_region_fig(dataset,phase,path,hatches)
        #stack_volume_fig(dataset,phase,path,hatches)
        #interf_power_fig(dataset,phase,path,hatches)
        #polar_cap_area_fig(dataset,phase,path)
        #polar_cap_flux_fig(dataset,phase,path)
        #static_motional_fig(dataset,phase,path)
        #imf_figure(dataset,phase,path,hatches)
        #quantity_timings(dataset, phase, path)
        lobe_balance_fig(dataset,phase,path)
        #lobe_power_histograms(dataset, phase, path)

if __name__ == "__main__":
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inLogs = os.path.join(sys.argv[-1],'data/logs/')
    inAnalysis = os.path.join(sys.argv[-1],'data/analysis/')
    outPath = os.path.join(inBase,'figures')
    outQT = os.path.join(outPath,'quietTime')
    outSSC = os.path.join(outPath,'shockImpact')
    outMN1 = os.path.join(outPath,'mainPhase1')
    outMN2 = os.path.join(outPath,'mainPhase2')
    outRec = os.path.join(outPath,'recovery')
    outLine = os.path.join(outPath,'outLine')
    unfiled = os.path.join(outPath,'unfiled')
    outInterv = os.path.join(outPath,'interval')
    for path in [outPath,outQT,outSSC,outMN1,outMN2,
                 outRec,unfiled,outInterv]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    #HDF data, will be sorted and cleaned
    dataset = {}
    #dataset['may'] = load_hdf_sort(inAnalysis+'may2019_results.h5')
    #dataset['may'] = load_hdf_sort(inAnalysis+'temp/test_may.h5')
    #dataset['feb'] = load_hdf_sort(inAnalysis+'feb2014_results.h5',
    #                               tshift=45)
    #dataset['star'] = load_hdf_sort(inAnalysis+'starlink2_results.h5')
    dataset['star'] = load_hdf_sort(inAnalysis+'temp/test_star2.h5')
    #dataset['aug'] = {}
    #dataset['jun'] = {}

    #Log files and observational indices
    #dataset['may']['obs'] = read_indices(inLogs, prefix='may2019_',
    #                                read_supermag=False)
    #dataset['feb']['obs'] = read_indices(inLogs, prefix='feb2014_',
    #                                read_supermag=False, tshift=45)
    dataset['star']['obs'] = read_indices(inLogs, prefix='starlink_',
                                     read_supermag=False,
                    end=dataset['star']['msdict']['closed'].index[-1])
    #dataset['aug']['obs'] = read_indices(inLogs, prefix='aug2018_',
    #                                     read_supermag=False)
    #dataset['jun']['obs'] = read_indices(inLogs, prefix='jun2015_',
    #                                     read_supermag=False)

    #NOTE hotfix change FD to CD for motional terms
    #       all calculations are performed as (n_1-n_0)/dt,
    #       this will change to  (n_1-n_-1)/2dt
    for event_key in dataset.keys():
        event = dataset[event_key]
        event['mpdict'],event['msdict'] = hotfix_cdiff(event['mpdict'],
                                                       event['msdict'])
    #NOTE hotfix for closed region tail_closed
    #for ev in ds.keys():
    #    for t in[t for t in ds[ev]['msdict']['closed'].keys()
    #                                                if 'Tail_close'in t]:
    #        ds[ev]['msdict']['closed'][t] = ds[ev]['mpdict']['ms_full'][t]

    ##Construct "grouped" set of subzones, then get %contrib for each
    for event in dataset.keys():
        if 'msdict' in dataset[event].keys():
            dataset[event]['msdict'] = {
                'rc':dataset[event]['msdict'].get('rc',pd.DataFrame()),
                'closed':dataset[event]['msdict'].get(
                                              'closed',pd.DataFrame()),
                'lobes':dataset[event]['msdict'].get(
                                               'lobes',pd.DataFrame())}
    ##Parse storm phases
    for event_key in [k for k in dataset.keys() if 'aug' not in k and
                                                   'jun' not in k]:
        event = dataset[event_key]
        msdict = event['msdict']
        '''
        #NOTE delete this!! | 
        #                   V
        msdict = hotfix_interfSharing(event['mpdict'],msdict,
                                      event['inner_mp'])
        msdict = hotfix_psb(msdict)
        #                   ^
        #                   |
        combined_closed_rc = combine_closed_rc(msdict)
        msdict['closed_rc'] = combined_closed_rc
        '''
        for phase in ['_qt','_main','_rec','_interv','_lineup']:
            if 'mpdict' in event.keys():
                event['mp'+phase], event['time'+phase]=parse_phase(
                                         event['mpdict']['ms_full'],phase)
            if 'inner_mp' in event.keys():
                event['inner_mp'+phase], event['time'+phase]=parse_phase(
                                                  event['inner_mp'],phase)
            if 'msdict' in event.keys():
                event['msdict'+phase], event['time'+phase] = parse_phase(
                                                    event['msdict'],phase)
            if 'termdict' in event.keys():
                event['termdict'+phase],event['time'+phase]=parse_phase(
                                                 event['termdict'],phase)
    for event_key in dataset.keys():
        event = dataset[event_key]
        obs_srcs = list(event['obs'].keys())
        for phase in ['_qt','_main','_rec','_interv','_lineup']:
            for src in obs_srcs:
                event['obs'][src+phase],event[src+'_otime'+phase]=(
                                parse_phase(event['obs'][src],phase))

    ######################################################################
    ##Main + Recovery phase
    main_rec_figures(dataset)
    ######################################################################
    ##Short zoomed in interval
    #interval_figures(dataset)
    ######################################################################
