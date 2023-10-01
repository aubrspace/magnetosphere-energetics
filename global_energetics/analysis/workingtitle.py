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
from global_energetics.analysis.proc_satellites import read_satellites
from global_energetics.analysis.proc_hdf import (load_hdf_sort,
                                                 group_subzones,
                                                 get_subzone_contrib)
from global_energetics.analysis.analyze_energetics import plot_power
from global_energetics.analysis.proc_energy_spatial import reformat_lshell
from global_energetics.analysis.proc_timing import (peak2peak,
                                                    pearson_r_shifts)

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
    day_dphidt_N=central_diff(abs(polar_caps['Bf_netPolesDayN [Wb]']))
    day_dphidt_S=central_diff(abs(polar_caps['Bf_netPolesDayS [Wb]']))
    night_dphidt_N=central_diff(abs(polar_caps['Bf_netPolesNightN [Wb]']))
    night_dphidt_S=central_diff(abs(polar_caps['Bf_netPolesNightS [Wb]']))
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
    dEdt = central_diff(data_output['Utot [J]'])
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
        #data_output['K1'] = data_output['K_netK1 [W]']/1e12
        data_output['K2a'] = data_output['K_netK2a [W]']/1e12
        data_output['K2b'] = data_output['K_netK2b [W]']/1e12
        sheath = 'K2a'
        #data_output['passed']=abs(data_output['sheath']+data_output['closed'])
    elif kwargs.get('keyset','lobes')== 'closed':
        #data_output['K5'] = data_output['K_netK5 [W]']/1e12
        data_output['K2a'] = data_output['K_netK2a [W]']/1e12
        data_output['K2b'] = data_output['K_netK2b [W]']/1e12
        sheath = 'K2a'
        #data_output['passed'] = abs(data_output['sheath']+data_output['lobes'])
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
    data_output['inoutflow']=['inflow']*len(data_output[sheath])
    data_output.loc[data_output[sheath]>0,'inoutflow']='outflow'
    #Gathering/releasing (-dEdt)
    data_output['gatherRelease']=['gathering']*len(data_output['static'])
    data_output.loc[data_output['minusdEdt']>0,'gatherRelease']='releasing'
    return data_output

def hotfix_cdiff(mpdict,msdict):
    lobes_copy = msdict['lobes'].copy(deep=True)
    closed_copy = msdict['closed'].copy(deep=True)
    mp_copy = mpdict['ms_full'].copy(deep=True)
    for region in [lobes_copy,closed_copy,mp_copy]:
        for target in [k for k in region.keys() if ('UtotM' in k) or
                                                   ('uHydroM' in k) or
                                                   ('uBM' in k)]:
            print(target)
            #TODO: change this to a higher order central difference
            # -(n-2)+8(n+1)-8(n-1)+(n-2)
            #           12h
            ogindex = region[target].copy(deep=True).index
            fdiff = region[target].reset_index(drop=True)
            back = fdiff.copy(deep=True)
            back.index = back.index+1
            back.drop(index=[back.index.max()],inplace=True)
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

def central_diff(dataframe,**kwargs):
    """Takes central difference of the columns of a dataframe
    Inputs
        df (DataFrame)- data
        dt (int)- spacing used for denominator
        kwargs:
            fill (float)- fill value for ends of diff
    Returns
        cdiff (DataFrame)
    """
    times = dataframe.copy(deep=True).index
    df = dataframe.copy(deep=True)
    df = df.reset_index(drop=True).fillna(method='ffill')
    df_fwd = df.copy(deep=True)
    df_bck = df.copy(deep=True)
    df_fwd.index -= 1
    df_bck.index += 1
    if kwargs.get('forward',False):
        # Calculate dt at each time interval
        dt = times[1::]-times[0:-1]
        cdiff = (df_fwd-df)/(dt.seconds+dt.microseconds/1e6)
        cdiff.drop(index=[-1],inplace=True)
    else:
        # Calculate dt at each time interval
        dt = times[2::]-times[0:-2]
        diff = (df_fwd-df_bck).drop(index=[-1,0,df_bck.index[-1],
                                                df_bck.index[-2]])
        cdiff = diff/(dt.seconds+dt.microseconds/1e6)
        cdiff.loc[0] = 0
        cdiff.loc[len(cdiff)]=0
        cdiff.sort_index(inplace=True)
    cdiff.index = dataframe.index
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

def find_crossings(in_vsat,in_obssat,satname):
    """Function creates crossing dataframe given virtual and real sat data
    Inputs
        vsat,obssat (DataFrame)
    Returns
        crossings
    """
    #Make copies so we don't mess up our dataframes
    vsat = in_vsat.copy(deep=True)
    obssat = in_obssat.copy(deep=True)
    crossings = pd.DataFrame()
    vsat.reset_index(inplace=True)
    #Interpolate data from obssat into vsat times
    obsinterp = pd.DataFrame({'time':vsat['time'].values})
    xtime = [float(t) for t in obssat.index.to_numpy()]
    ytime = [float(t) for t in vsat['time'].to_numpy()]
    for key in obssat.keys():
        obsinterp[key] = np.interp(ytime,xtime,obssat[key].values)
    ## find crossings
    lookahead = vsat['Status'].copy(deep=True)
    lookahead.index -=1
    lookahead.drop(index=[-1],inplace=True)
    lookahead.loc[lookahead.index.max()+1] = lookahead.loc[
                                                      lookahead.index.max()]
    crossing_status = vsat[['Status']][(lookahead-vsat['Status'])!=0]
    k=0
    crossing_status['id'] = 0
    for i, index in enumerate(crossing_status.index):
        forward = False
        backward = False
        # Check for start
        if i!=len(crossing_status)-1:
            #update next index
            next_index = crossing_status.index[i+1]
            if index == (next_index-1):
                forward = True
        # Check for end
        if i!=0:
            if index == (last_index+1):
                backward = True
        # Check for interior
        if forward and backward:
            crossing_status.loc[index,'id'] = k
        elif forward:
            k+=1
            crossing_status.loc[index,'id'] = k
        elif backward:
            crossing_status.loc[index,'id'] = k
        # Else is lone point (id==0)
        else:
            k+=1
            crossing_status.loc[index,'id'] = k
        # update last
        last_index = index
    vsat['crosstype'] = 'none'
    obssat['crosstype'] = 'none'
    for interval_id in range(1,crossing_status['id'].max()+1):
        interval_points = crossing_status[crossing_status['id']==
                                          interval_id].index
        start = interval_points.min()-1
        end = interval_points.max()+1
        tstart = vsat.loc[start,'time']
        tend = vsat.loc[end,'time']
        start_status = vsat.loc[start,'Status']
        end_status = vsat.loc[end,'Status']
        obswindow = obssat[(obsinterp.index>tstart)&(obsinterp.index<tend)].index
        # A dip if starting and ending with the same status
        #TODO: mark both the virtual and obssat 'crosstype'
        #       Add a day/night to the derived values
        #       Add day/night from virtual to obs
        if start_status==end_status:
            #look at the interior min and max compared with the endpoint value
            if start_status==3 and vsat.loc[start-1:end+1,'Status'].min()<1:
                # K5
                vsat.loc[start-1:end+1,'crosstype'] = 'K5'
                obsinterp.loc[obswindow,'crosstype'] = 'K5'
            else:
                # K2
                vsat.loc[start-1:end+1,'crosstype'] = 'K2'
                obsinterp.loc[obswindow,'crosstype'] = 'K2'
            if ((start_status==2 or start_status==1)and
                 vsat.loc[start-1:end+1,'Status'].max()>2):
                # K2
                vsat.loc[start-1:end+1,'crosstype'] = 'K2'
                obsinterp.loc[obswindow,'crosstype'] = 'K2'
            else:
                # K1
                vsat.loc[start-1:end+1,'crosstype'] = 'K1'
                obsinterp.loc[obswindow,'crosstype'] = 'K1'
        # Solar wind to closed K5
        elif ((start_status==3 and end_status==0) or
              (start_status==0 and end_status==3)):
            vsat.loc[start-1:end+1,'crosstype'] = 'K5'
            obsinterp.loc[obswindow,'crosstype'] = 'K5'
        # Solar wind to open K1
        elif (((start_status==1 or start_status==2) and end_status==0) or
              (start_status==0 and (end_status==1 or end_status==2))):
            vsat.loc[start-1:end+1,'crosstype'] = 'K1'
            obsinterp.loc[obswindow,'crosstype'] = 'K1'
        # Closed to open K2
        elif ((start_status==3 and (end_status==1 or end_status==2)) or
              ((start_status==1 or start_status==2) and end_status==3)):
            vsat.loc[start-1:end+1,'crosstype'] = 'K2'
            obsinterp.loc[obswindow,'crosstype'] = 'K2'
    #TODO: find the full crossing window
    #   check for continuity in the crossing index
    #   buffer by 3 min on either side, call this all one crossing type
    #   Look at end point Status values to determine type
    #   If different endpoints, clearly mark
    #   Else take min max within the interval too guess at what kind
    # Figure out how to keep track of all these pieces of time
    #  The only things we need to mark are (maybe?)
    #       crossing ID
    #       type
    #       crossing start,end
    # ID the points where Status changes value
    # look for the points within (window/2-1) of two points where jump occured
    # for each point
    #   set the window crosstime
    #   set the window half range
    #   set the type based on what the status jump was
    #   for each virtual value
    #       get average over window
    #   for each obs value
    #       get average over window
    return crossings

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
    starlink_impact = (dt.datetime(2022,2,2,23,58)+
                       dt.timedelta(hours=3,minutes=55))
    starlink_endMain1 = dt.datetime(2022,2,3,11,54)
    #starlink_endMain1 = dt.datetime(2022,2,4,13,10)
    starlink_endMain2 = dt.datetime(2022,2,3,11,54)
    starlink_inter_start = (starlink_endMain1+
                            dt.timedelta(hours=-4,minutes=-20))
    starlink_inter_end = (starlink_endMain1+
                            dt.timedelta(hours=-4,minutes=0))
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
        #start = dt.timedelta(minutes=390)
        #impact = dt.datetime(2014,4,10,6,30,0)
        start = dt.timedelta(minutes=0)
        impact = times[0]
        peak1 = times[-1]
        peak2 = times[-1]
        inter_start = impact
        inter_end = peak2
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
        cond=(times>moments['inter_start'])&(times<moments['inter_end'])
        #cond = (times>moments['impact']) & (times<moments['peak1'])
                                            #dt.timedelta(minutes=10))
        #cond = ((times>moments['peak1']-dt.timedelta(minutes=120)) &
        #        (times<moments['peak1']+dt.timedelta(minutes=30)))

        #cond = ((times>moments['impact']) &
        #        (times<moments['impact']+dt.timedelta(minutes=120)))
    elif 'lineup' in phasekey:
        cond = times>times[0]+moments['start']

    #Reload data filtered by the condition
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        #if 'lineup' not in phasekey and 'interv' not in phasekey:
        if False:
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
            #if 'lineup' not in phasekey and 'interv' not in phasekey:
            if False:
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
            #Erc = ds[ev]['msdict'+ph]['rc']['Utot [J]']
            Etotal = Elobes+Eclosed
            ElobesPercent = (Elobes/Etotal)*100
            lobes = ds[ev]['msdict'+ph]['lobes']
            mp = dataset[ev]['mp'+phase]
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
            dEdt = central_diff(lobes['Utot [J]'])/1e12
            K1 = mp['K_netK1 [W]']/1e12
            K2a = lobes['K_netK2a [W]']/1e12
            K2b = lobes['K_netK2b [W]']/1e12
            K3 = lobes['K_netK3 [W]']/1e12
            K4 = lobes['K_netK4 [W]']/1e12
            #motion = -dEdt - lobes['K_net [W]']/1e12
            M1 = lobes['Utot_netK1 [W]']/1e12
            M2a = lobes['Utot_netK2a [W]']/1e12
            M2b = lobes['Utot_netK2b [W]']/1e12
            M3 = lobes['Utot_netK3 [W]']/1e12
            M4 = lobes['Utot_netK4 [W]']/1e12
            for flux,label in [(dEdt,'dEdt'),
                               (K1,'K1'),(K2a,'K2a'),(K2b,'K2b'),
                               (K3,'K1'),(K4,'K4'),
                               (M1,'M1'),(M2a,'M2a'),(M2b,'M2b'),
                               (M3,'M1'),(M4,'M4')]:
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
        dtime_impact = (moments['impact']-
                        moments['peak2']).total_seconds()*1e9
        ax.axvspan(dtime_impact,0,facecolor='lightgrey')
        if 'lineup' in ph or 'interv' in ph:
            dotimedelta=True
        else: dotimedelta=False
        if not ds[ev]['mp'+ph].empty:
            times=[float(n) for n in ds[ev]['time'+ph].to_numpy()]#bad hack
            #rax = ax.twinx()
            plot_stack_contrib(ax,times,ds[ev]['mp'+ph],
                               ds[ev]['msdict'+ph], legend=(i==0),
                               value_key='Utot [J]',label=ev,ylim=[0,30],
                               factor=1e15,
                               ylabel=r'Energy $\left[ PJ\right]$',
                               legend_loc='upper right', hatch=hatches[i],
                               do_xlabel=(i==len(ds.keys())-1),
                               timedelta=dotimedelta)
        ax.set_xlim(times[0],times[-1])
        ax.margins(x=0.01)
        #ax.plot(times,ds[ev]['msdict'+ph]['lobes']['Utot [J]']/1e15,
        #           color='Black')
        #rax.set_ylim([0,55])
        #NOTE mark impact w vertical line here
        #general_plot_settings(rax,
        #                      do_xlabel=False, legend=False,
        #                      timedelta=dotimedelta)
        ax.set_xlabel(r'Time $\left[hr\right]$')
        #save
        contr.suptitle(moments['peak1'].strftime("%b %Y, t0=%d-%H:%M:%S"),
                                                 ha='left',x=0.01,y=0.99)
        contr.tight_layout(pad=0.8)
        #figname = path+'/contr_energy'+ph+'_'+ev+'.png'
        figname = path+'/contr_energy'+ph+'_'+ev+'.eps'
        contr.savefig(figname,dpi=300)
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
        closed = dataset[event]['msdict'+phase]['closed']
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


def quantify_timings2(dataset, phase, path,**kwargs):
    """Function plots timing measurements between two quantities
    Inputs
    Returns
        None
    """
    for i,event in enumerate(dataset.keys()):
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        if phase=='':
            mp = dataset[event]['mpdict']['ms_full']
        else:
            mp = dataset[event]['mp'+phase]
        #times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        #moments = locate_phase(dataset[event]['time'])
        # for solar wind
        sw = dataset[event]['obs']['swmf_sw'+phase]
        #swtime = dataset[event]['swmf_sw_otime'+phase]
        #swt = [float(n) for n in swtime.to_numpy()]#bad hack
        #sim = dataset[event]['obs']['swmf_log'+phase]
        #simtime = dataset[event]['swmf_log_otime'+phase]
        #simt = [float(n) for n in simtime.to_numpy()]#bad hack
        #index = dataset[event]['obs']['swmf_index'+phase]
        #indextime = dataset[event]['swmf_index_otime'+phase]
        #indext = [float(n) for n in indextime.to_numpy()]#bad hack
        #obs = dataset[event]['obs']['omni'+phase]
        #obstime = dataset[event]['omni_otime'+phase]
        #ot = [float(n) for n in obstime.to_numpy()]#bad hack

        ## TOTAL
        #K1,5 from mp
        Ks1 = mp['K_netK1 [W]']
        Ks5 = mp['K_netK5 [W]']
        #K2,3,4 from lobes
        Ks2al = lobes['K_netK2a [W]']
        Ks2bl = lobes['K_netK2b [W]']
        Ks3 = lobes['K_netK3 [W]']
        Ks4 = lobes['K_netK4 [W]']
        #K2,6,7 from closed
        Ks2ac = closed['K_netK2a [W]']
        Ks2bc = closed['K_netK2b [W]']
        Ks6 = closed['K_netK6 [W]']
        Ks7 = closed['K_netK7 [W]']

        ## HYDRO
        #H1,5 from mp
        Hs1 = mp['P0_netK1 [W]']
        Hs5 = mp['P0_netK5 [W]']
        #H2,3,4 from lobes
        Hs2al = lobes['P0_netK2a [W]']
        Hs2bl = lobes['P0_netK2b [W]']
        Hs3 = lobes['P0_netK3 [W]']
        Hs4 = lobes['P0_netK4 [W]']
        #H2,6,7 from closed
        Hs2ac = closed['P0_netK2a [W]']
        Hs2bc = closed['P0_netK2b [W]']
        Hs6 = closed['P0_netK6 [W]']
        Hs7 = closed['P0_netK7 [W]']

        ## MAG
        #S1,5 from mp
        Ss1 = mp['ExB_netK1 [W]']
        Ss5 = mp['ExB_netK5 [W]']
        #S2,3,4 from lobes
        Ss2al = lobes['ExB_netK2a [W]']
        Ss2bl = lobes['ExB_netK2b [W]']
        Ss3 = lobes['ExB_netK3 [W]']
        Ss4 = lobes['ExB_netK4 [W]']
        #S2,6,7 from closed
        Ss2ac = closed['ExB_netK2a [W]']
        Ss2bc = closed['ExB_netK2b [W]']
        Ss6 = closed['ExB_netK6 [W]']
        Ss7 = closed['ExB_netK7 [W]']

        ## TOTAL
        #M1,5,total from mp
        M1 = mp['UtotM1 [W]']
        M5 = mp['UtotM5 [W]']
        M = mp['UtotM [W]']
        #M1a,1b,2b,il from lobes
        M1a = lobes['UtotM1a [W]']
        M1b = lobes['UtotM1b [W]']
        M2b = lobes['UtotM2b [W]']
        M2d = lobes['UtotM2d [W]']
        Mil = lobes['UtotMil [W]']
        #M5a,5b,2a,ic from closed
        M5a = closed['UtotM5a [W]']
        M5b = closed['UtotM5b [W]']
        M2a = closed['UtotM2a [W]']
        M2c = closed['UtotM2c [W]']
        Mic = closed['UtotMic [W]']

        M_lobes = M1a+M1b-M2a+M2b-M2c+M2d
        M_closed = M5a+M5b+M2a-M2b+M2c-M2d

        ## HYDRO
        #HM1,5,total from mp
        HM1 = mp['uHydroM1 [W]']
        HM5 = mp['uHydroM5 [W]']
        HM = mp['uHydroM [W]']
        #HM1a,1b,2b,il from lobes
        HM1a = lobes['uHydroM1a [W]']
        HM1b = lobes['uHydroM1b [W]']
        HM2b = lobes['uHydroM2b [W]']
        HM2d = lobes['uHydroM2d [W]']
        HMil = lobes['uHydroMil [W]']
        #HM5a,5b,2a,ic from closed
        HM5a = closed['uHydroM5a [W]']
        HM5b = closed['uHydroM5b [W]']
        HM2a = closed['uHydroM2a [W]']
        HM2c = closed['uHydroM2c [W]']
        HMic = closed['uHydroMic [W]']

        HM_lobes = HM1a+HM1b-HM2a+HM2b-HM2c+HM2d
        HM_closed = HM5a+HM5b+HM2a-HM2b+HM2c-HM2d

        ## MAG
        #SM1,5,total from mp
        SM1 = mp['uBM1 [W]']
        SM5 = mp['uBM5 [W]']
        SM = mp['uBM [W]']
        #HM1a,1b,2b,il from lobes
        SM1a = lobes['uBM1a [W]']
        SM1b = lobes['uBM1b [W]']
        SM2b = lobes['uBM2b [W]']
        SM2d = lobes['uBM2d [W]']
        SMil = lobes['uBMil [W]']
        #SM5a,5b,2a,ic from closed
        SM5a = closed['uBM5a [W]']
        SM5b = closed['uBM5b [W]']
        SM2a = closed['uBM2a [W]']
        SM2c = closed['uBM2c [W]']
        SMic = closed['uBMic [W]']

        SM_lobes = SM1a+SM1b-SM2a+SM2b-SM2c+SM2d
        SM_closed = SM5a+SM5b+SM2a-SM2b+SM2c-SM2d

        # Central difference of partial volume integrals, total change
        # Total
        K_closed = -1*central_diff(closed['Utot [J]'])
        K_lobes = -1*central_diff(lobes['Utot [J]'])
        K_mp = -1*central_diff(mp['Utot [J]'])
        # Hydro
        H_closed = -1*central_diff(closed['uHydro [J]'])
        H_lobes = -1*central_diff(lobes['uHydro [J]'])
        H_mp = -1*central_diff(mp['uHydro [J]'])
        # Mag
        S_closed = -1*central_diff(closed['uB [J]'])
        S_lobes = -1*central_diff(lobes['uB [J]'])
        S_mp = -1*central_diff(mp['uB [J]'])

        r_values1 = pd.DataFrame()
        r_values2 = pd.DataFrame()
        time_shifts, r_values1['clock-S2a']=pearson_r_shifts(
                                   sw[sw['clock']>0]['clock'],(Ss2al-SM2a))
        time_shifts, r_values1['clock-S2b']=pearson_r_shifts(
                                   sw[sw['clock']>0]['clock'],Ss2bl+SM2b)
        time_shifts, r_values1['clock-H5']=pearson_r_shifts(
                                   sw[sw['clock']>0]['clock'],Hs5+HM5)
        time_shifts, r_values1['clock-S1']=pearson_r_shifts(
                                   sw[sw['clock']>0]['clock'],Ss1+SM1)

        time_shifts, r_values2['Bz-S2a']=pearson_r_shifts(
                                            sw[sw['bz']<0]['bz'],(Ss2al-SM2a))
        time_shifts, r_values2['Bz-S2b']=pearson_r_shifts(
                                            sw[sw['bz']<0]['bz'],Ss2bl+SM2b)
        time_shifts, r_values2['Bz-H5']=pearson_r_shifts(
                                            sw[sw['bz']<0]['bz'],Hs5+HM5)
        time_shifts, r_values2['Bz-S1']=pearson_r_shifts(
                                            sw[sw['bz']<0]['bz'],Ss1+SM1)
        '''
        time_shifts, r_values['Newell-S2a']=pearson_r_shifts(
                sw['Newell'].rolling(10).mean(),(Ss2al-SM2a).rolling(10).mean())
        time_shifts, r_values['Newell-S2b']=pearson_r_shifts(sw['Newell'],Ss2bl+SM2b)
        time_shifts, r_values['Newell-H5']=pearson_r_shifts(sw['Newell'],Hs5+HM5)
        time_shifts, r_values['Newell-S1']=pearson_r_shifts(sw['Newell'],Ss1+SM1)
        '''

        #############
        #setup figure
        flux_timings,(axis1,axis2) = plt.subplots(2,1,figsize=[16,12])
        #Plot
        for key in r_values1.keys():
            axis1.plot(time_shifts/60,r_values1[key],label=key)
        for key in r_values2.keys():
            axis2.plot(time_shifts/60,r_values2[key],label=key)
        #Decorations
        axis1.legend()
        axis1.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis1.set_ylabel(r'Pearson R')
        axis1.grid()
        axis2.legend()
        axis2.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis2.set_ylabel(r'Pearson R')
        axis2.grid()
        axis2.set_xlabel(r'Timeshift $\left[min\right]$')
        #save
        #flux_timings.tight_layout(pad=1)
        figurename = path+'/flux_timings'+phase+'_'+event+'.png'
        flux_timings.savefig(figurename)
        plt.close(flux_timings)
        print('\033[92m Created\033[00m',figurename)
        #############


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
        closed = dataset[event]['msdict'+phase]['closed']
        if phase=='':
            mp = dataset[event]['mpdict']['ms_full']
        else:
            mp = dataset[event]['mp'+phase]
        mp = dataset[event]['mpdict'+phase]
        solarwind = dataset[event]['obs']['swmf_sw'+phase]
        working_lobes = prep_for_correlations(lobes,solarwind)
        working_closed = prep_for_correlations(closed,solarwind)
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
        time_shifts, r_values['K1-K5']=pearson_r_shifts(
                                                  mp['K_netK1 [W]'],
                                                  mp['K_netK5 [W]'])
        time_shifts, r_values['K1-K2b']=pearson_r_shifts(
                                                  mp['K_netK1 [W]'],
                                                  working_lobes['K2b'])
        time_shifts, r_values['K2a-K2b']=pearson_r_shifts(
                                                  working_lobes['K2a'],
                                                  working_lobes['K2b'])
        #time_shifts, r_values['K2a-K2b']=pearson_r_shifts(
        #                                         working_closed['K2a'],
        #                                      -1*working_closed['K2b'])
        time_shifts, r_values['K2b-K5']=pearson_r_shifts(
                                                  working_closed['K2b'],
                                                  mp['K_netK5 [W]'])
        time_shifts, r_values['Pa-Pb']=pearson_r_shifts(
                                            working_polar_cap['DayRxn'],
                                            working_polar_cap['NightRxn'])
        time_shifts, r_values['K1K2a-Pa']=pearson_r_shifts(
                                 mp['K_netK1 [W]']+working_lobes['K2a'],
                                              working_polar_cap['DayRxn'])
        time_shifts, r_values['K2b-Pb']=pearson_r_shifts(
                                                   working_lobes['K2b'],
                                            working_polar_cap['NightRxn'])
        '''
        time_shifts, r_values['Pa_motion']=pearson_r_shifts(
                                           working_polar_cap['DayRxn'],
                                                  working_lobes['motion'])
        time_shifts, r_values['Pb_motion']=pearson_r_shifts(
                                         working_polar_cap['NightRxn'],
                                                  working_lobes['motion'])
        '''
        axis.plot(time_shifts/60,r_values['K1-K5'],
                  label='K1-K5')
        axis.plot(time_shifts/60,r_values['K1-K2b'],
                  label='K1-K2b')
        axis.plot(time_shifts/60,r_values['K2a-K2b'],
                  label='K2a-K2b')
        axis.plot(time_shifts/60,r_values['K2b-K5'],
                  label='K2b-K5')
        axis.plot(time_shifts/60,r_values['Pa-Pb'],
                  label='Pa-Pb')
        axis.plot(time_shifts/60,r_values['K1K2a-Pa'],
                  label='K1K2a-Pa')
        axis.plot(time_shifts/60,r_values['K2b-Pb'],
                  label='K2b-Pb')

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
                                                  solarwind['bz'],
                                               mp['K_netK1 [W]'])
        time_shifts, r_values['newell-K1']=pearson_r_shifts(
                                                 solarwind['Newell'],
                                               mp['K_netK1 [W]'])
        time_shifts, r_values['epsilon-K1']=pearson_r_shifts(
                                                 solarwind['eps'],
                                               mp['K_netK1 [W]'])
        time_shifts, r_values['bz-Pa']=pearson_r_shifts(
                                                  solarwind['bz'],
                                           working_polar_cap['DayRxn'])
        time_shifts, r_values['newell-Pa']=pearson_r_shifts(
                                                 solarwind['Newell'],
                                           working_polar_cap['DayRxn'])
        time_shifts, r_values['epsilon-Pa']=pearson_r_shifts(
                                                 solarwind['eps'],
                                           working_polar_cap['DayRxn'])
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
        axis2.plot(time_shifts/60,r_values['bz-Pa'],
                  label='bz-Pa')
        axis2.plot(time_shifts/60,r_values['newell-Pa'],
                  label='newell-Pa')
        axis2.plot(time_shifts/60,r_values['epsilon-Pa'],
                  label='epsilon-Pa')
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
        dEdt = central_diff(lobes['Utot [J]'])
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
        dEdt = central_diff(lobes['Utot [J]'])
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


def oneD_comparison(dataset,phase,path):
    """Plot the energy balance on the lobes
    """
    for i,event in enumerate(dataset.keys()):
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        mp = dataset[event]['mp'+phase]
        inner = dataset[event]['inner_mp'+phase]
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        sim = dataset[event]['obs']['swmf_log'+phase]
        simtime = dataset[event]['swmf_log_otime'+phase]
        simt = [float(n) for n in simtime.to_numpy()]#bad hack
        obs = dataset[event]['obs']['omni'+phase]
        obstime = dataset[event]['omni_otime'+phase]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        moments = locate_phase(dataset[event]['time'])

        #############
        #setup figure
        injection,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(times,(mp['1DK_netK5 [W]']+
                         mp['1DK_netK1 [W]'])/1e12,
                                 label='1DNet_forward')
        axis.plot(times,(
            #mp['K_netK5 [W]']+
                         lobes['K_netK1 [W]'])/1e12,
                                 label='Net_forward')
        rax = axis.twinx()
        rax.plot(times,
            (
            #mp['K_netK5 [W]']+
             lobes['K_netK1 [W]'])/
            (mp['1DK_netK5 [W]']+
             mp['1DK_netK1 [W]'])*100,
                                 color='grey',label='Ratio')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylim=[-100,30],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              timedelta=True)
        rax.set_ylim([-50,50])
        rax.set_ylabel(r'$\%$')
        axis.axvspan((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,0,
                       fc='lightgrey')
        #save
        injection.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        injection.tight_layout(pad=1)
        figurename = path+'/injection'+phase+'_'+event+'.png'
        injection.savefig(figurename)
        plt.close(injection)
        print('\033[92m Created\033[00m',figurename)
        #############

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
    #event='star'
    #if True:
    for i,event in enumerate(dataset.keys()):
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        if 'xslice' in dataset[event]['msdict'+phase].keys():
            xslice = dataset[event]['msdict'+phase]['xslice']
        mp = dataset[event]['mp'+phase]
        inner = dataset[event]['inner_mp'+phase]
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        sim = dataset[event]['obs']['swmf_log'+phase]
        simtime = dataset[event]['swmf_log_otime'+phase]
        simt = [float(n) for n in simtime.to_numpy()]#bad hack
        obs = dataset[event]['obs']['omni'+phase]
        obstime = dataset[event]['omni_otime'+phase]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        #lobes4 = dataset['star4']['msdict'+phase]['lobes']
        #closed4 = dataset['star4']['msdict'+phase]['closed']
        #mp4 = dataset['star4']['mp'+phase]
        #inner4 = dataset['star4']['inner_mp'+phase]
        #times4=[float(n) for n in dataset['star4']['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        #from IPython import embed; embed()
        # for solar wind
        sw = dataset[event]['obs']['swmf_sw'+phase]
        swtime = dataset[event]['swmf_sw_otime'+phase]
        swt = [float(n) for n in swtime.to_numpy()]#bad hack
        '''
        sim = dataset[event]['obs']['swmf_log'+phase]
        simtime = dataset[event]['swmf_log_otime'+phase]
        simt = [float(n) for n in simtime.to_numpy()]#bad hack
        index = dataset[event]['obs']['swmf_index'+phase]
        indextime = dataset[event]['swmf_index_otime'+phase]
        indext = [float(n) for n in indextime.to_numpy()]#bad hack
        obs = dataset[event]['obs']['omni'+phase]
        obstime = dataset[event]['omni_otime'+phase]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        '''

        ## TOTAL
        #K1,5 from mp
        Ks1 = mp['K_netK1 [W]']
        Ks5 = mp['K_netK5 [W]']
        #K2,3,4 from lobes
        Ks2al = lobes['K_netK2a [W]']
        Ks2bl = lobes['K_netK2b [W]']
        Ks3 = lobes['K_netK3 [W]']
        Ks4 = lobes['K_netK4 [W]']
        #K2,6,7 from closed
        Ks2ac = closed['K_netK2a [W]']
        Ks2bc = closed['K_netK2b [W]']
        Ks6 = closed['K_netK6 [W]']
        Ks7 = closed['K_netK7 [W]']
        #DawnDusk from xslice
        #KsDsk = xslice.get('K_netDusk [W]')
        #KsDwn = xslice.get('K_netDawn [W]')

        ## HYDRO
        #H1,5 from mp
        Hs1 = mp['P0_netK1 [W]']
        Hs5 = mp['P0_netK5 [W]']
        #H2,3,4 from lobes
        Hs2al = lobes['P0_netK2a [W]']
        Hs2bl = lobes['P0_netK2b [W]']
        Hs3 = lobes['P0_netK3 [W]']
        Hs4 = lobes['P0_netK4 [W]']
        #H2,6,7 from closed
        Hs2ac = closed['P0_netK2a [W]']
        Hs2bc = closed['P0_netK2b [W]']
        Hs6 = closed['P0_netK6 [W]']
        Hs7 = closed['P0_netK7 [W]']
        #DawnDusk from xslice
        #HsDsk = xslice.get('P0_netDusk [W]')
        #HsDwn = xslice.get('P0_netDawn [W]')

        ## MAG
        #S1,5 from mp
        Ss1 = mp['ExB_netK1 [W]']
        Ss5 = mp['ExB_netK5 [W]']
        #S2,3,4 from lobes
        Ss2al = lobes['ExB_netK2a [W]']
        Ss2bl = lobes['ExB_netK2b [W]']
        Ss3 = lobes['ExB_netK3 [W]']
        Ss4 = lobes['ExB_netK4 [W]']
        #S2,6,7 from closed
        Ss2ac = closed['ExB_netK2a [W]']
        Ss2bc = closed['ExB_netK2b [W]']
        Ss6 = closed['ExB_netK6 [W]']
        Ss7 = closed['ExB_netK7 [W]']
        #DawnDusk from xslice
        #SsDsk = xslice.get('ExB_netDusk [W]')
        #SsDwn = xslice.get('ExB_netDawn [W]')

        ## TOTAL
        #M1,5,total from mp
        M1 = mp['UtotM1 [W]'].fillna(value=0)
        M5 = mp['UtotM5 [W]'].fillna(value=0)
        M = mp['UtotM [W]'].fillna(value=0)
        MM = mp['MM [kg/s]'].fillna(value=0)
        #M1a,1b,2b,il from lobes
        M1a = lobes['UtotM1a [W]'].fillna(value=0)
        M1b = lobes['UtotM1b [W]'].fillna(value=0)
        M2b = lobes['UtotM2b [W]'].fillna(value=0)
        M2d = lobes['UtotM2d [W]'].fillna(value=0)
        Mil = lobes['UtotMil [W]'].fillna(value=0)
        '''
        MM_lobes = (lobes['MM1a [kg/s]']+
                    lobes['MM1b [kg/s]']-
                    closed['MM2a [kg/s]']+
                    lobes['MM2b [kg/s]']-
                    closed['MM2c [kg/s]']+
                    lobes['MM2d [kg/s]'])
        '''
        #M5a,5b,2a,ic from closed
        M5a = closed['UtotM5a [W]'].fillna(value=0)
        M5b = closed['UtotM5b [W]'].fillna(value=0)
        M2a = closed['UtotM2a [W]'].fillna(value=0)
        M2c = closed['UtotM2c [W]'].fillna(value=0)
        Mic = closed['UtotMic [W]'].fillna(value=0)
        '''
        MM_closed = (closed['MM5a [kg/s]']+
                     closed['MM5b [kg/s]']+
                     closed['MM2a [kg/s]']-
                     lobes['MM2b [kg/s]']+
                     closed['MM2c [kg/s]']-
                     lobes['MM2d [kg/s]'])
        '''

        M_lobes = M1a+M1b-M2a+M2b-M2c+M2d
        M_closed = M5a+M5b+M2a-M2b+M2c-M2d

        ## HYDRO
        #HM1,5,total from mp
        HM1 = mp['uHydroM1 [W]'].fillna(value=0)
        HM5 = mp['uHydroM5 [W]'].fillna(value=0)
        HM = mp['uHydroM [W]'].fillna(value=0)
        #HM1a,1b,2b,il from lobes
        HM1a = lobes['uHydroM1a [W]'].fillna(value=0)
        HM1b = lobes['uHydroM1b [W]'].fillna(value=0)
        HM2b = lobes['uHydroM2b [W]'].fillna(value=0)
        HM2d = lobes['uHydroM2d [W]'].fillna(value=0)
        HMil = lobes['uHydroMil [W]'].fillna(value=0)
        #HM5a,5b,2a,ic from closed
        HM5a = closed['uHydroM5a [W]'].fillna(value=0)
        HM5b = closed['uHydroM5b [W]'].fillna(value=0)
        HM2a = closed['uHydroM2a [W]'].fillna(value=0)
        HM2c = closed['uHydroM2c [W]'].fillna(value=0)
        HMic = closed['uHydroMic [W]'].fillna(value=0)

        HM_lobes = HM1a+HM1b-HM2a+HM2b-HM2c+HM2d
        HM_closed = HM5a+HM5b+HM2a-HM2b+HM2c-HM2d

        ## MAG
        #SM1,5,total from mp
        SM1 = mp['uBM1 [W]'].fillna(value=0)
        SM5 = mp['uBM5 [W]'].fillna(value=0)
        SM = mp['uBM [W]'].fillna(value=0)
        #HM1a,1b,2b,il from lobes
        SM1a = lobes['uBM1a [W]'].fillna(value=0)
        SM1b = lobes['uBM1b [W]'].fillna(value=0)
        SM2b = lobes['uBM2b [W]'].fillna(value=0)
        SM2d = lobes['uBM2d [W]'].fillna(value=0)
        SMil = lobes['uBMil [W]'].fillna(value=0)
        #SM5a,5b,2a,ic from closed
        SM5a = closed['uBM5a [W]'].fillna(value=0)
        SM5b = closed['uBM5b [W]'].fillna(value=0)
        SM2a = closed['uBM2a [W]'].fillna(value=0)
        SM2c = closed['uBM2c [W]'].fillna(value=0)
        SMic = closed['uBMic [W]'].fillna(value=0)

        SM_lobes = SM1a+SM1b-SM2a+SM2b-SM2c+SM2d
        SM_closed = SM5a+SM5b+SM2a-SM2b+SM2c-SM2d

        # Central difference of partial volume integrals, total change
        # Total
        K_closed = -1*central_diff(closed['Utot [J]'])
        #K_lobes = -1*central_diff(lobes['Utot [J]'])
        K_mp = -1*central_diff(mp['Utot [J]'])
        #K_mp4 = -1*central_diff(mp4['Utot [J]'])
        # Hydro
        H_closed = -1*central_diff(closed['uHydro [J]'])
        #H_lobes = -1*central_diff(lobes['uHydro [J]'])
        H_mp = -1*central_diff(mp['uHydro [J]'])
        # Mag
        S_closed = -1*central_diff(closed['uB [J]'])
        #S_lobes = -1*central_diff(lobes['uB [J]'])
        S_mp = -1*central_diff(mp['uB [J]'])
        dDstdt_sim = -1*central_diff(sim['dst_sm'])
        dDstdt_obs = -1*central_diff(obs['sym_h'])
        # Mass
        #Mass_closed = -1*central_diff(closed['M [kg]'])
        #Mass_lobes = -1*central_diff(lobes['M [kg]'])
        #Mass_mp = -1*central_diff(mp['M [kg]'])

        Ksum = Ks1+Ks3+Ks4+Ks5+Ks6+Ks7
        '''
        Ksum4Re = (mp4['K_netK1 [W]']+
                   mp4['K_netK5 [W]']+
                   lobes4['K_netK3 [W]']+
                   lobes4['K_netK4 [W]']+
                   closed4['K_netK6 [W]']+
                   closed4['K_netK7 [W]'])
        M4Re = mp4['UtotM [W]']
        '''
        predicted = mp['Utot [J]'].copy(deep=True)
        predicted.reset_index(drop=True,inplace=True)
        predicted.index+=1
        dt = [(t1-t0)/1e9 for t0,t1 in zip(times[0:-1],times[1::])]
        dt.append(dt[-1])
        predicted -= (Ksum*dt).values
        predicted.iloc[0] = 0
        decay_loss_rate = [-(1-np.e**(-t/36000)) for t in dt]
        decay_loss = decay_loss_rate*closed['uHydro [J]']
        #predicted.drop(index=predicted.index[-1],inplace=True)
        '''
        Masssum_mp = (mp['M_net [kg/s]']+lobes['M_netK3 [kg/s]']+
                      closed['M_netK7 [kg/s]'])
        Masssum_lobes = (mp['M_netK1 [kg/s]']+lobes['M_netK2a [kg/s]']+
                         lobes['M_netK2b [kg/s]']+lobes['M_netK3 [kg/s]']+
                         lobes['M_netK4 [kg/s]'])
        Masssum_closed = (mp['M_netK5 [kg/s]']+closed['M_netK2a [kg/s]']+
                          closed['M_netK2b [kg/s]']+closed['M_netK6 [kg/s]']+
                          closed['M_netK7 [kg/s]'])
        '''
        #correction = 1.744e8*(mp['Area [Re^2]']+lobes['TestAreaK3 [Re^2]']+
        #                      closed['TestAreaK7 [Re^2]'])
        #total_area = (mp['Area [Re^2]']+lobes['TestAreaK3 [Re^2]']+
        #                                closed['TestAreaK7 [Re^2]'])
        #correction = 4.580e-4*total_area
        #mcorrection = 2.0e-4*total_area
        #bcorrection = 2.5e11
        #TODO: Try looking at the split in S vs H in the local error to
        #       determine if the error can be attributed to a specific static
        #       flux
        #   What can we say about the error:
        #       value at each time step
        #       cummulative value
        #       split between S and H locally and cummulatively
        #       M versus d/dt of volume to check if its in the motional comp
        #       Theoretical potentail sources: dV/dt, FdotdA for whole min

        '''
        if 'lineup' in phase or 'interv' in phase:
            dotimedelta=True
        else: dotimedelta=False
        '''
        dotimedelta=True

        '''
        #############
        #setup figure
        lobes_balance_total,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.fill_between(times,K_lobes/1e12,label='Total Lobes',fc='grey')
        axis.plot(times,lobes['K_net [W]']/1e12,label='Static Lobes')
        axis.plot(times,M_lobes/1e12,label='Motion Lobes')
        axis.plot(times,(lobes['K_net [W]']+M_lobes)/1e12,label='Summed Lobes')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        lobes_balance_total.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        lobes_balance_total.tight_layout(pad=1)
        figurename = path+'/lobe_balance_total'+phase+'_'+event+'.png'
        lobes_balance_total.savefig(figurename)
        plt.close(lobes_balance_total)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        lobes_balance_detail,(axis,axis2,axis3)=plt.subplots(
                                                    3,1,figsize=[16,16])
        #Plot
        axis.fill_between(times,K_lobes/1e12,
                          label='Total Lobes',fc='grey')
        axis.plot(times,(lobes['K_net [W]']+M_lobes)/1e12,label='Summed Lobes')
        axis.plot(times,(Ks1+M1)/1e12,label='K1')
        #axis.plot(times,(Ks2al-M2a-M2c+Ks2bl+M2b+M2d)/1e12,label='K2')
        axis.plot(times,(Ks2ac+Ks2bc+M2a-M2b+M2c-M2d)/1e12,label='K2')
        axis.plot(times,Ks3/1e12,label='K3')
        axis.plot(times,Ks4/1e12,label='K4')
        axis.plot(times,(Ks5+M5)/1e12,label='K5')
        axis.plot(times,(Ks6)/1e12,label='K6')
        axis.plot(times,(Ks7)/1e12,label='K7')
        #axis.plot(times,(Ks2al-M2a-M2c)/1e12,label='K2a')
        #axis.plot(times,(Ks2bl+M2b+M2d)/1e12,label='K2b')

        axis2.fill_between(times,lobes['K_net [W]']/1e12,
                          label='Total Static',fc='grey')
        axis2.plot(times,(mp['K_netK1 [W]']+lobes['K_netK2a [W]']+
                          lobes['K_netK2b [W]']+lobes['K_netK3 [W]']+
                          lobes['K_netK4 [W]'])/1e12,
                  label='Summed Lobes')
        axis2.plot(times,mp['K_netK1 [W]']/1e12,label=r'$K_s1$')
        axis2.plot(times,lobes['K_netK2a [W]']/1e12,label=r'$K_s2a$')
        axis2.plot(times,lobes['K_netK2b [W]']/1e12,label=r'$K_s2b$')
        axis2.plot(times,lobes['K_netK3 [W]']/1e12,label=r'$K_s3$')
        axis2.plot(times,lobes['K_netK4 [W]']/1e12,label=r'$K_s4$')

        axis3.fill_between(times,M_lobes/1e12,
                          label='Total Lobes',fc='grey')
        axis3.plot(times,(M1-M2a+M2b-M2c+M2d)/1e12,
        #axis3.plot(times,(M_1day+M_1night+M_3+M_4)/1e12,
                   label='Summed Lobes')
        axis3.plot(times,M1/1e12,label='M1')
        axis3.plot(times,-M2a/1e12,label='M2a')
        axis3.plot(times,M2b/1e12,label='M2b')
        axis3.plot(times,-M2c/1e12,label='M2c')
        axis3.plot(times,M2d/1e12,label='M2d')
        #Decorations
        general_plot_settings(axis,do_xlabel=False,legend=True,
                              legend_loc='upper left',
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis2,do_xlabel=False,legend=True,
                              legend_loc='upper left',
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis3,do_xlabel=True,legend=True,
                              legend_loc='upper left',
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        for axis in [axis,axis2,axis3]:
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
        #save
        lobes_balance_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        lobes_balance_detail.tight_layout(pad=0.3)
        figurename = path+'/lobe_balance_detail'+phase+'_'+event+'.png'
        lobes_balance_detail.savefig(figurename)
        plt.close(lobes_balance_detail)
        print('\033[92m Created\033[00m',figurename)
        '''

        #############
        #setup figure
        '''
        mass_balance_regions,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(times,Mass_mp,label='Cdiff-MS',c='maroon',ls='--')
        axis.plot(times,(Masssum_mp+MM),label='Summed-MS',c='maroon')
        axis.plot(times,Mass_lobes,label='Cdiff-Lobes',c='tab:blue',ls='--')
        axis.plot(times,(Masssum_lobes+MM_lobes),label='Summed-Lobes',
                  c='tab:blue')
        axis.plot(times,Mass_closed,label='Cdiff-Lobes',c='magenta',ls='--')
        axis.plot(times,(Masssum_closed+MM_closed),label='Summed-Closed',
                  c='magenta')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylim=[-40,100],
                              ylabel=r'MassFlux $\left[ kg/s\right]$',
                              legend_loc='upper left',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        mass_balance_regions.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        mass_balance_regions.tight_layout(pad=1)
        figurename = path+'/mass_balance_regions'+phase+'_'+event+'.png'
        mass_balance_regions.savefig(figurename)
        plt.close(mass_balance_regions)
        print('\033[92m Created\033[00m',figurename)
        '''

        #############
        #setup figure
        total_balance_total,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.fill_between(times,(Ksum+M-K_mp)/1e12,label='Error',fc='grey')
        axis.plot(times,(Ksum+M-K_mp-decay_loss)/1e12,label='DecayError')
        #axis.plot(times,decay_loss/1e12,label='Decay')
        #axis.plot(times,(K_mp)/1e12,label='Cdiff')
        #axis.fill_between(times,(K_mp)/1e12,label='Cdiff',fc='grey')
        #axis.fill_between(times4,(K_mp4)/1e12,label='Cdiff4Re',fc='tab:blue',
        #                  alpha=0.2)
        #axis.plot(times,(Ksum+M)/1e12,label='Summed')
        #axis.plot(times,(Ks1)/1e12,label='K1')
        #axis.plot(times,(Ks3)/1e12,label='K3')
        #axis.plot(times,(Ks4)/1e12,label='K4')
        #axis.plot(times,(Ks5)/1e12,label='K5')
        #axis.plot(times,(Ks6)/1e12,label='K6')
        #axis.plot(times,(Ksum+M-(Ks3+Ks7))/1e12,label='minus(K3+K7)')
        #axis.plot(times,(M)/1e12,label='M')
        #axis.plot(times4,(Ksum4Re+M4Re)[Ksum4Re.index]/1e12,label='Summed4Re')
        #axis.plot(times,(Ksum+M-K_mp)/1e12,label='Error3Re')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              #ylim=[-20,5],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        total_balance_total.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        total_balance_total.tight_layout(pad=1)
        figurename = path+'/total_balance_total'+phase+'_'+event+'.png'
        total_balance_total.savefig(figurename)
        plt.close(total_balance_total)
        print('\033[92m Created\033[00m',figurename)

        """
        #############
        #setup figure
        closed_balance_total,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.fill_between(times,K_closed/1e12,label='Total Closed',fc='grey')
        axis.plot(times,closed['K_net [W]']/1e12,label='Static Closed')
        axis.plot(times,M_closed/1e12,label='Motion Closed')
        axis.plot(times,(closed['K_net [W]']+M_closed)/1e12,
                  label='Summed Closed')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        closed_balance_total.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        closed_balance_total.tight_layout(pad=1)
        figurename = path+'/closed_balance_total'+phase+'_'+event+'.png'
        closed_balance_total.savefig(figurename)
        plt.close(closed_balance_total)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        closed_balance_detail,(axis,axis2,axis3)=plt.subplots(
                                                    3,1,figsize=[16,24])
        #Plot
        axis.fill_between(times,K_closed/1e12,
                          label='Total Closed',fc='grey')
        axis.plot(times,(closed['K_net [W]']+M_closed)/1e12,
                  label='Summed Closed')
        axis.plot(times,(Ks5+M5)/1e12,label='K5')
        axis.plot(times,(Ks2ac+M2a+M2c)/1e12,label='K2a')
        axis.plot(times,(Ks2bc-M2b-M2d)/1e12,label='K2b')
        axis.plot(times,Ks6/1e12,label='K6')
        axis.plot(times,Ks7/1e12,label='K7')

        axis2.fill_between(times,closed['K_net [W]']/1e12,
                          label='Total closed',fc='grey')
        axis2.plot(times,(mp['K_netK5 [W]']+closed['K_netK2a [W]']+
                          closed['K_netK2b [W]']+closed['K_netK6 [W]']+
                          closed['K_netK7 [W]'])/1e12,
                  label='Summed Closed')
        axis2.plot(times,mp['K_netK5 [W]']/1e12,label=r'$K_s5$')
        axis2.plot(times,closed['K_netK2a [W]']/1e12,label=r'$K_s2a$')
        axis2.plot(times,closed['K_netK2b [W]']/1e12,label=r'$K_s2b$')
        axis2.plot(times,closed['K_netK6 [W]']/1e12,label=r'$K_s6$')
        axis2.plot(times,closed['K_netK7 [W]']/1e12,label=r'$K_s7$')

        axis3.fill_between(times,M_closed/1e12,
                          label='Total Closed',fc='grey')
        axis3.plot(times,(M5+M2a-M2b+M2c-M2d)/1e12,
                  label='Summed Closed')
        axis3.plot(times,M5/1e12,label='M5')
        axis3.plot(times,M2a/1e12,label='M2a')
        axis3.plot(times,-M2b/1e12,label='M2b')
        axis3.plot(times,M2c/1e12,label='M2c')
        axis3.plot(times,-M2d/1e12,label='M2d')
        #Decorations
        general_plot_settings(axis,do_xlabel=False,legend=True,
                              legend_loc='upper left',
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis2,do_xlabel=False,legend=True,
                              legend_loc='upper left',
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        general_plot_settings(axis3,do_xlabel=True,legend=True,
                              legend_loc='upper left',
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        for axis in [axis,axis2,axis3]:
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
        #save
        closed_balance_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        closed_balance_detail.tight_layout(pad=1)
        figurename = path+'/closed_balance_detail'+phase+'_'+event+'.png'
        closed_balance_detail.savefig(figurename)
        plt.close(closed_balance_detail)
        print('\033[92m Created\033[00m',figurename)

        """
        #############
        #setup figure
        total_acc_total,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(times,K_mp.cumsum()*dt/1e15,label='CentralDiff')
        axis.plot(times,(Ksum+M).cumsum()*dt/1e15,
                         label='Summed')
        axis.plot(times,((Ksum+M-decay_loss)).cumsum()*dt/1e15,
                         label='withDecay')
        #axis.plot(times4,K_mp4.cumsum()*60/1e15,label='CentralDiff4Re')
        #axis.plot(times4,(Ksum4Re+M4Re)[Ksum4Re.index].cumsum()*60/1e15,
        #                 label='Summed4Re')
        axis.plot(times,-1*(predicted-mp['Utot [J]'][0])/1e15,
                  label='1minPrediction')
        axis.fill_between(times,
                         -1*(mp['Utot [J]']-mp['Utot [J]'][0])/1e15,
                          label='-1*Energy',fc='grey')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              #ylim=[-40,2],
                        ylabel=r'Accumulated Power $\left[ PJ\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        total_acc_total.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        total_acc_total.tight_layout(pad=1)
        figurename = path+'/total_acc_total'+phase+'_'+event+'.png'
        total_acc_total.savefig(figurename)
        plt.close(total_acc_total)
        print('\033[92m Created\033[00m',figurename)

        """
        #############
        #setup figure
        lobe_acc_total,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(times,K_lobes.cumsum()*60/1e15,label='Total Lobes')
        axis.plot(times,(M_lobes+lobes['K_net [W]']).cumsum()*60/1e15,
                         label='Summed Lobes')
        axis.plot(times,lobes['K_net [W]'].cumsum()*60/1e15,
                  label='Static Lobes')
        axis.plot(times,M_lobes.cumsum()*60/1e15,label='Motion Lobes')
        axis.fill_between(times,
                         -1*(lobes['Utot [J]']-lobes['Utot [J]'][0])/1e15,
                          label='-1*Lobe Energy',fc='grey')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                        ylabel=r'Accumulated Power $\left[ PJ\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        lobe_acc_total.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        lobe_acc_total.tight_layout(pad=1)
        figurename = path+'/lobe_acc_total'+phase+'_'+event+'.png'
        lobe_acc_total.savefig(figurename)
        plt.close(lobe_acc_total)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        lobe_acc_detail,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.fill_between(times,K_lobes.cumsum()*60/1e15,
                          label='Total Lobes',fc='grey')
        axis.plot(times,(M_lobes+lobes['K_net [W]']).cumsum()*60/1e15,
                         label='Summed Lobes')
        axis.plot(times,(Ks1+M1).cumsum()*60/1e15,label='K1')
        axis.plot(times,(Ks2al-M2a-M2c).cumsum()*60/1e15,label='K2a')
        axis.plot(times,(Ks2bl+M2b+M2d).cumsum()*60/1e15,label='K2b')
        axis.plot(times,Ks3.cumsum()*60/1e15,label='K3')
        axis.plot(times,Ks4.cumsum()*60/1e15,label='K4')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                        ylabel=r'Accumulated Power $\left[ PJ\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        lobe_acc_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        lobe_acc_detail.tight_layout(pad=1)
        figurename = path+'/lobe_acc_detail'+phase+'_'+event+'.png'
        lobe_acc_detail.savefig(figurename)
        plt.close(lobe_acc_detail)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        inner_circulation,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(times,(Ks2ac+M2a+M2c)/1e12,label='Cusp')
        axis.plot(times,-(Ks2bc-M2b-M2d)/1e12,label='-1*Tail')
        axis.axhline(0,color='black')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                     ylabel=r'(Closed Vol) Net Power $\left[ TW\right]$',
                              timedelta=dotimedelta)
        axis.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
        axis.axvline(0,ls='--',color='black')
        #save
        inner_circulation.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        inner_circulation.tight_layout(pad=1)
        figurename = path+'/inner_circulation'+phase+'_'+event+'.png'
        inner_circulation.savefig(figurename)
        plt.close(inner_circulation)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        external,(axis,axis2) = plt.subplots(2,1,figsize=[16,16])
        #Plot
        axis.fill_between(times,(mp['K_netK1 [W]']+mp['K_netK5 [W]']+
                         lobes['K_netK4 [W]']+closed['K_netK6 [W]'])/1e12,
                          fc='grey',label='Static')
        axis.plot(times,(mp['K_netK1 [W]'])/1e12,label=r'$K_s1$')
        axis.plot(times,(mp['K_netK5 [W]'])/1e12,label=r'$K_s5$')
        axis.plot(times,lobes['K_netK4 [W]']/1e12,label=r'$K_s4$')
        axis.plot(times,closed['K_netK6 [W]']/1e12,label=r'$K_s6$')

        axis2.fill_between(times,(M1+M5+Ks1+Ks5+Ks4+Ks6)/1e12,
                           label='Total',fc='grey')
        axis2.plot(times,(M1+Ks1)/1e12,label='K1')
        axis2.plot(times,(M5+Ks5)/1e12,label='K5')
        axis2.plot(times,Ks4/1e12,label='K4')
        axis2.plot(times,Ks6/1e12,label='K6')
        #axis2.axhline(0,color='black')
        #Decorations
        general_plot_settings(axis,do_xlabel=False,legend=True,
                     ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              ylim=[-10,10],
                              timedelta=dotimedelta)
        general_plot_settings(axis2,do_xlabel=True,legend=True,
                     ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              ylim=[-10,10],
                              timedelta=dotimedelta)
        for axis in [axis,axis2]:
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
        #save
        external.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        external.tight_layout(pad=1)
        figurename = path+'/external'+phase+'_'+event+'.png'
        external.savefig(figurename)
        plt.close(external)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        #setup figure
        comboVS,(axis,axis2,axis3,axis4) = plt.subplots(4,1,figsize=[16,12])
        #Plot
        #axis.fill_between(swt,sw['B'], ec='dimgrey',fc='thistle',
        #                  label=r'$|B|$')
        #axis.plot(swt,sw['bx'],label=r'$B_x$',c='maroon')
        axis.fill_between(swt,sw['pdyn'], ec='dimgrey',fc='thistle',
                          label=r'$P_{dyn}$')
        axis.plot(swt,sw['by'],label=r'$B_y$',c='magenta')
        axis.plot(swt,sw['bz'],label=r'$B_z$',c='tab:blue')
        general_plot_settings(axis,ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        '''
        axis.fill_between(times,(M1+M5+Ks1+Ks5+Ks4+Ks6)/1e12,
                           label='Total',fc='grey')
        axis.plot(times,(M1+Ks1)/1e12,label='K1')
        axis.plot(times,(M5+Ks5)/1e12,label='K5')
        axis.plot(times,Ks4/1e12,label='K4')
        axis.plot(times,Ks6/1e12,label='K6')
        '''
        axis2.plot(times,(Ks2ac+M2a+M2c)/1e12,label=r'Cusp $K_{2a}$',
                   color='goldenrod')
        axis2.plot(times,(Ks2bc-M2b-M2d)/1e12,label=r'Tail $K_{2b}$',
                   color='tab:blue')
        axis2.fill_between(times,(Ks2ac+Ks2bc+M2a-M2b+M2c-M2d)/1e12,
                           label=r'Net $K_2$',fc='grey')
        axis3.plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        axis3.plot(ot,obs['sym_h'],label='Obs',c='maroon')
        axis4.plot(times,-1*mp['Utot [J]'],c='black',
                 label=r'$-\int{U_{tot}}\left[ J\right]$')
        #rax.spines['right'].set_color('magenta')
        #rax.tick_params(axis='y',colors='magenta')
        #axis2.axhline(0,color='black')
        #Decorations
        general_plot_settings(axis,ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        '''
        general_plot_settings(axis,do_xlabel=False,legend=True,
                     ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              ylim=[-12,12],
                              timedelta=dotimedelta)
        '''
        general_plot_settings(axis2,do_xlabel=False,legend=True,
                     ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              ylim=[-20,10],
                              timedelta=dotimedelta)
        general_plot_settings(axis3,ylabel=r'Sym-H$\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        general_plot_settings(axis4,ylabel=r'-Energy $\left[ J\right]$',
                              do_xlabel=True, legend=False,
                              timedelta=dotimedelta)
        for axis in [axis,axis2,axis3,axis4]:
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
        #save
        comboVS.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        comboVS.tight_layout(pad=0.04)
        figurename = path+'/comboVS'+phase+'_'+event+'.png'
        comboVS.savefig(figurename)
        plt.close(comboVS)
        print('\033[92m Created\033[00m',figurename)
        #############
        dDstdt_sim = -1*central_diff(sim['dst_sm'])
        dDstdt_obs = -1*central_diff(obs['sym_h'])

        #############
        #setup figure
        dst_comp,(axis) = plt.subplots(1,1,figsize=[20,8])
        '''
        # Interpolate from times and obs to simt
        dDstdt_obsINT = np.interp(simt,ot,dDstdt_obs)
        K_mp_INT = np.interp(simt,times,K_mp/8e13)
        # Scatter plot
        axis.scatter(K_mp_INT,dDstdt_sim,label='sim',alpha=0.7,marker='s')
        axis.scatter(K_mp_INT,dDstdt_obsINT,label='obs',alpha=0.7)
        '''
        axis.plot(ot,dDstdt_obs,label='obs')
        axis.plot(simt,dDstdt_sim,label='sim')
        axis.plot(times,K_mp/8e13,label='-dEdt')
        general_plot_settings(axis,do_xlabel=True,legend=True,
                                  ylabel='d{Dst}/dt [nT/s]',
                                  legend_loc='lower left',
                                  timedelta=dotimedelta)
        axis.axvspan((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,0,
                       fc='lightgrey')
        #save
        dst_comp.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        dst_comp.tight_layout(pad=1)
        figurename = path+'/dst_comp'+phase+'_'+event+'.png'
        dst_comp.savefig(figurename)
        plt.close(dst_comp)
        print('\033[92m Created\033[00m',figurename)
        #############
        """

        #############
        #setup figure
        flavors_external,(axis,axis2,axis3) =plt.subplots(3,1,figsize=[20,24])
        #Plot
        axis.plot(times,(HM1+Hs1)/1e12,label='1')
        axis.plot(times,(HM5+Hs5)/1e12,label='5')
        axis.plot(times,Hs4/1e12,label='4')
        axis.plot(times,Hs6/1e12,label='6')
        axis.plot(times,Hs3/1e12,label='3')
        axis.plot(times,Hs7/1e12,label='7')

        axis2.plot(times,(SM1+Ss1)/1e12,label='S1')
        axis2.plot(times,(SM5+Ss5)/1e12,label='S5')
        axis2.plot(times,Ss4/1e12,label='S4')
        axis2.plot(times,Ss6/1e12,label='S6')
        axis2.plot(times,Ss3/1e12,label='S3')
        axis2.plot(times,Ss7/1e12,label='S7')

        axis3.plot(times,(M1+Ks1)/1e12,label='K1')
        axis3.plot(times,(M5+Ks5)/1e12,label='K5')
        axis3.plot(times,Ks4/1e12,label='K4')
        axis3.plot(times,Ks6/1e12,label='K6')
        axis3.plot(times,Ks3/1e12,label='K3')
        axis3.plot(times,Ks7/1e12,label='K7')

        #axis4.plot(times,(mp['1DK_netK5 [W]']+mp['1DK_netK1 [W]'])/1e12,
        #           label='1D Injection')
        #axis4.plot(times,(lobes['1DK_netK4 [W]']+closed['1DK_netK6 [W]'])/1e12,
        #           label='1D Escape')
        #Decorations
        powerlabel=['Hydro.','Poynting','Tot. Energy']
        for i,ax in enumerate([axis,axis2,axis3]):
            general_plot_settings(ax,do_xlabel=(i==2),legend=False,
                                  ylabel='Integrated '+powerlabel[i]+
                                        r' Flux $\left[ TW\right]$',
                                  legend_loc='lower left',
                                  ylim=[-12,7],
                                  timedelta=dotimedelta)
            ax.axvspan((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,0,
                       fc='lightgrey')
            ax.margins(x=0.01)
        axis.fill_between(times,(HM1+HM5+Hs1+Hs5+Hs4+Hs6+Hs3+Hs7)/1e12,
                           label='Total',fc='dimgray')
        axis2.fill_between(times,(SM1+SM5+Ss1+Ss5+Ss4+Ss6+Ss3+Ss7)/1e12,
                           label='Total',fc='dimgray')
        axis3.fill_between(times,(M1+M5+Ks1+Ks5+Ks4+Ks6+Ks3+Ks7)/1e12,
                           label='Total',fc='dimgray')
        #axis4.fill_between(times,(mp['1DK_netK1 [W]']+mp['1DK_netK5 [W]']+
        #                        lobes['1DK_netK4 [W]']+closed['1DK_netK6 [W]']
        #                          )/1e12,
        #                   label='Sum',fc='dimgray')
        #axis4.set_ylim(-120,70)

        axis.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=7, fancybox=True, shadow=True)
        #save
        flavors_external.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        flavors_external.tight_layout(pad=1)
        #figurename = path+'/flavors_external'+phase+'_'+event+'.png'
        figurename = path+'/flavors_external'+phase+'_'+event+'.eps'
        flavors_external.savefig(figurename,dpi=300)
        plt.close(flavors_external)
        print('\033[92m Created\033[00m',figurename)
        #############

        #############
        ##setup figure
        # Figure
        flux_internal = plt.figure(figsize=[20,26])
        # GridSpecs
        twochunk = plt.GridSpec(2,1,hspace=0.1,top=0.95,figure=flux_internal,
                                height_ratios=[4,1])
        topstacks = twochunk[0].subgridspec(3,1,hspace=0.05)
        botstacks = twochunk[1].subgridspec(1,1,hspace=0.05)
        # Axes
        hydro_ax = flux_internal.add_subplot(topstacks[0,:])
        poynting_ax = flux_internal.add_subplot(topstacks[1,:])
        total_ax = flux_internal.add_subplot(topstacks[2,:])
        dawndusk_ax = flux_internal.add_subplot(botstacks[0,:])

        ##Plot
        # Hydro
        hydro_ax.plot(times,(Hs2ac+HM2a+HM2c)/1e12,label=r'Cusp ${2a}$',
                   color='goldenrod')
        hydro_ax.plot(times,(Hs2bc-HM2b-HM2d)/1e12,label=r'Tail ${2b}$',
                   color='tab:blue')
        # Poynting
        poynting_ax.plot(times,(Ss2ac+SM2a+SM2c)/1e12,label=r'Cusp $S_{2a}$',
                   color='goldenrod')
        poynting_ax.plot(times,(Ss2bc-SM2b-SM2d)/1e12,label=r'Tail $S_{2b}$',
                   color='tab:blue')
        # Total
        total_ax.plot(times,(Hs2ac+HM2a+HM2c+Ss2ac+SM2a+SM2c)/1e12,
                      label=r'Cusp $K_{2a}$',color='goldenrod')
        total_ax.plot(times,(Hs2bc-HM2b-HM2d+Ss2bc-SM2b-SM2d)/1e12,
                      label=r'Tail $K_{2b}$',color='tab:blue')
        # DawnDusk
        '''
        dawndusk_ax.plot(times,HsDwn/1e12,label=r'$H$ Dawn',color='indianred')
        dawndusk_ax.plot(times,HsDsk/1e12,label=r'$H$ Dusk',
                         color='red',ls=':')
        dawndusk_ax.plot(times,SsDwn/1e12,label=r'$S$ Dawn',
                         color='cornflowerblue')
        dawndusk_ax.plot(times,SsDsk/1e12,label=r'$S$ Dusk',
                         color='blue',ls=':')
        '''

        ##Decorations
        powerlabel=['Hydro.','Poynting','Tot. Energy','Night->Day']
        for i,ax in enumerate([hydro_ax,poynting_ax,total_ax,dawndusk_ax]):
            general_plot_settings(ax,do_xlabel=(i==3),legend=False,
                                  ylabel='Integrated '+powerlabel[i]+
                                  #ylabel=r'$\int dA$'+powerlabel[i]+
                                        r' Flux $\left[ TW\right]$',
                                  #legend_loc='lower left',
                                  #ylim=[-10,8],
                                  timedelta=dotimedelta)
            ax.axvspan((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,0,
                       fc='lightgrey')
            ax.margins(x=0.01,tight=None)
            ax.yaxis.label.set_fontsize(22)
        hydro_ax.fill_between(times,(Hs2ac+Hs2bc+HM2a-HM2b+HM2c-HM2d)/1e12,
                           label=r'Sum',fc='dimgray')
        poynting_ax.fill_between(times,(Ss2ac+Ss2bc+SM2a-SM2b+SM2c-SM2d)/1e12,
                           label=r'Sum',fc='dimgray')
        total_ax.fill_between(times,(
                          Hs2ac+Hs2bc+HM2a-HM2b+HM2c-HM2d+
                          Ss2ac+Ss2bc+SM2a-SM2b+SM2c-SM2d
                          )/1e12,
                           label=r'Sum',fc='dimgray')
        #dawndusk_ax.fill_between(times,(HsDwn+HsDsk+SsDwn+SsDsk)/1e12,
        #                   label=r'Sum',fc='dimgray')

        hydro_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                        ncol=3, fancybox=True, shadow=True)
        dawndusk_ax.legend(loc='lower left',ncol=5)
        hydro_ax.set_ylim(-2.5,2)
        poynting_ax.set_ylim(-10,8)
        total_ax.set_ylim(-10,8)
        dawndusk_ax.set_ylim(-10,8)
        #save
        flux_internal.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                               ha='left',x=0.01,y=0.99)
        twochunk.tight_layout(pad=0.6,figure=flux_internal)
        #figurename = path+'/flavors_internal'+phase+'_'+event+'.png'
        figurename = path+'/flavors_internal'+phase+'_'+event+'.eps'
        flux_internal.savefig(figurename,dpi=300)
        plt.close(flux_internal)
        print('\033[92m Created\033[00m',figurename)
        #############

        """
        #############
        #setup figure
        terminator,(axis) = plt.subplots(1,1,figsize=[20,8])

        #Plot
        axis.plot(times,HsDwn/1e12,label=r'$H$ Dawn',color='indianred')
        axis.plot(times,HsDsk/1e12,label=r'$H$ Dusk',color='red',ls=':')
        axis.plot(times,SsDwn/1e12,label=r'$S$ Dawn',color='cornflowerblue')
        axis.plot(times,SsDsk/1e12,label=r'$S$ Dusk',color='blue',ls=':')
        #Decorations
        powerlabel=['Energy']
        for i,ax in enumerate([axis]):
            general_plot_settings(ax,do_xlabel=(i==0),legend=(i==0),
                                  ylabel='Integrated '+powerlabel[i]+
                                        r' Flux $\left[ TW\right]$',
                                  legend_loc='upper left',
                                  ylim=[((KsDwn+KsDsk)/1e12).min(),
                                        ((KsDwn+KsDsk)/1e12).max()],
                                  timedelta=dotimedelta)
            ax.axvspan((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,0,
                       fc='lightgrey')
            ax.margins(x=0.01,tight=None)
        axis.fill_between(times,(HsDwn+HsDsk+SsDwn+SsDsk)/1e12,
                           label=r'$K$ Night -> Day',fc='dimgray')
        axis.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=5, fancybox=False, shadow=False)
        #save
        terminator.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        terminator.tight_layout(pad=1)
        figurename = path+'/terminator'+phase+'_'+event+'.png'
        terminator.savefig(figurename)
        plt.close(terminator)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        poynting_type,(axis,axis2,axis3) = plt.subplots(3,1,figsize=[16,24])
        #Plot
        axis.plot(times,(Ss2ac)/1e12,label=r'Cusp $S_{2a,S}$',
                   color='goldenrod')
        axis.plot(times,(Ss2bc)/1e12,label=r'Tail $S_{2b,S}$',
                   color='tab:blue')
        axis.fill_between(times,(Ss2ac+Ss2bc)/1e12,
                           label=r'Net $S_2$',fc='grey')

        axis2.plot(times,(SM2a+SM2c)/1e12,label=r'Cusp $S_{2a,M}$',
                   color='goldenrod')
        axis2.plot(times,(-SM2b-SM2d)/1e12,label=r'Tail $S_{2b,M}$',
                   color='tab:blue')
        axis2.fill_between(times,(SM2a-SM2b+SM2c-SM2d)/1e12,
                           label=r'Net $S_2$',fc='grey')

        axis3.plot(times,(Ss2ac+SM2a+SM2c)/1e12,label=r'Cusp $S_{2a}$',
                   color='goldenrod')
        axis3.plot(times,(Ss2bc-SM2b-SM2d)/1e12,label=r'Tail $S_{2b}$',
                   color='tab:blue')
        axis3.fill_between(times,(Ss2ac+Ss2bc+SM2a-SM2b+SM2c-SM2d)/1e12,
                           label=r'Net $S_2$',fc='grey')
        #Decorations
        for ax in [axis,axis2,axis3]:
            general_plot_settings(ax,do_xlabel=False,legend=True,
                     ylabel=r'Net Power $\left[ TW\right]$',
                              legend_loc='lower left',
                              ylim=[-12,12],
                              timedelta=dotimedelta)
            ax.axvline((moments['impact']-
                      moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            ax.axvline(0,ls='--',color='black')
        #save
        poynting_type.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        poynting_type.tight_layout(pad=1)
        figurename = path+'/poynting_type'+phase+'_'+event+'.png'
        poynting_type.savefig(figurename)
        plt.close(poynting_type)
        print('\033[92m Created\033[00m',figurename)
        #############
        """

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
                               figsize=[24,4*6])
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
        sup = ds[event]['obs']['supermag'+ph]
        suptime = ds[event]['supermag_otime'+ph]
        supt = [float(n) for n in suptime.to_numpy()]#bad hack
        #vsup = ds[event]['obs']['vsupermag'+ph]
        #vsuptime = ds[event]['vsupermag_otime'+ph]
        #vsupt = [float(n) for n in suptime.to_numpy()]#bad hack
        moments = locate_phase(sw.index)
        if kwargs.get('tabulate',False):
            #start,impact,peak1,peak2,inter_start,inter_end=locate_phase(
            #                                                    sw.index)
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
        ax[0].plot(swt,sw['bx'],label=r'$B_x$',c='maroon')
        ax[0].plot(swt,sw['by'],label=r'$B_y$',c='magenta')
        ax[0].plot(swt,sw['bz'],label=r'$B_z$',c='tab:blue')
        general_plot_settings(ax[0],ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        #Plasma
        ax[1].plot(swt,sw['Beta'],label=r'$\beta$',c='tab:blue')
        #ax[1].plot(swt,sw['Ma'],label=r'$M_{Alf}$',c='magenta')
        #rax = ax[1].twinx()
        #rax.plot(swt,sw['Beta'],label=r'$\beta$',c='tab:blue',ylim=[0,10])
        #rax.spines['right'].set_color('tab:blue')
        #rax.tick_params(axis='y',colors='tab:blue')
        general_plot_settings(ax[1],ylabel=r'$P_{dyn},\beta$',
                              legend=True,do_xlabel=False,
                              ylim=[0,14],timedelta=dotimedelta)
        #Dst index
        ax[2].plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        ax[2].plot(ot,obs['sym_h'],label='Obs',c='maroon')
        general_plot_settings(ax[2],ylabel=r'Sym-H$\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=dotimedelta)
        #SML index
        #ax[3].plot(vsupt,vsup['vSML'],label='Sim',c='tab:blue')
        #ax[3].plot(supt,sup['SML'],label='Obs',c='maroon')
        ax[3].plot(indext,index['AL'],label='AL',c='magenta',ls='--')
        general_plot_settings(ax[3],ylabel=r'SML$\left[nT\right]$',
                              do_xlabel=True, legend=True,
                              timedelta=dotimedelta)
        for axis in ax:
            axis.axvspan((moments['impact']-
                               moments['peak2']).total_seconds()*1e9,0,
                               fc='lightgrey')
            axis.margins(x=0.01)
            '''
            axis.axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis.axvline(0,ls='--',color='black')
            '''
        # NOTE Fills must come after for EPS images (no translucency allowed!)
        ax[0].fill_between(swt,sw['B'], ec='dimgrey',fc='thistle',
                               hatch=hatches[i], label=r'$|B|$')
        ax[0].set_ylim([-21,20])
        ax[1].fill_between(swt,sw['pdyn'],ec='dimgrey',fc='thistle',
                               hatch=hatches[i],label=r'$P_{dyn}$')
        #save
        dst.suptitle(moments['peak1'].strftime("%b %Y, t0=%d-%H:%M:%S"),
                                                     ha='left',x=0.01,y=0.99)
        dst.tight_layout(pad=0.3)
        #figname = path+'/dst_'+event+'.png'
        figname = path+'/dst_'+event+'.eps'
        dst.savefig(figname,dpi=300)
        plt.close(dst)
        print('\033[92m Created\033[00m',figname)

def satellite_comparisons(dataset,phase,path):
    """Time series comparison of virtual and observed satellite data
    """
    dotimedelta=True
    for i,event in enumerate(dataset.keys()):
        moments = locate_phase(dataset[event]['time'])
        # List of satellites we want to use
        satlist = ['cluster4','themisa','themisd','themise','mms1']
        #############
        #setup figure
        #bp_compare_detail,axis = plt.subplots(len(satlist)*2,1,
        #                                     figsize=[20,6*len(satlist)])
        bp_compare = plt.figure(figsize=[20,6*len(satlist)])
        twochunk = plt.GridSpec(2,1,hspace=0.1,top=0.90,figure=bp_compare)
        topstacks = twochunk[0].subgridspec(len(satlist),1,hspace=0.2)
        botstacks = twochunk[1].subgridspec(len(satlist),1,hspace=0.2)
        #Plot
        for i,sat in enumerate(satlist):
            j = i+len(satlist)
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            if False:#export only the used data
                virtual.to_csv(sat+'_sim.csv',sep=',')
                obs.to_csv(sat+'_obs.csv',sep=',')
            ##Plot
            # S
            Bz_axis = bp_compare.add_subplot(topstacks[i,:])
            S_axis = Bz_axis.twinx()
            S_axis.plot(otime,np.sqrt(obs['Sx']**2+
                                        obs['Sy']**2+
                                        obs['Sz']**2)/1e9,
                                        label='obs|S| [MW]',c='magenta')
            S_axis.plot(vtime,np.sqrt(virtual['Sx']**2+
                                        virtual['Sy']**2+
                                        virtual['Sz']**2)/1e9,
                                    label='sim|S| [MW]',c='black',lw=1)
            S_axis.set_ylim([0,20])
            S_axis.spines['right'].set_color('magenta')
            S_axis.spines['left'].set_color('tab:blue')
            S_axis.tick_params(axis='y',colors='magenta')
            S_axis.yaxis.set_minor_locator(AutoMinorLocator())
            Bz_axis.plot(otime,obs['bz'],label='obsBz [nT]',c='tab:blue')
            Bz_axis.plot(vtime,virtual['B_z'],label='simBz [nT]',c='dimgrey',
                         lw=1)
            # H
            P_axis = bp_compare.add_subplot(botstacks[i,:])
            H_axis = P_axis.twinx()
            H_axis.plot(otime,np.sqrt(obs['Hx']**2+
                                        obs['Hy']**2+
                                        obs['Hz']**2)/1e9,
                                        label='obs|H| [MW]',c='crimson')
            H_axis.plot(vtime,np.sqrt(virtual['Hx']**2+
                                        virtual['Hy']**2+
                                        virtual['Hz']**2)/1e9,
                                    label='sim|H| [MW]',lw=1,c='black')
            H_axis.set_ylim([0,20])
            H_axis.spines['right'].set_color('crimson')
            H_axis.spines['left'].set_color('tab:olive')
            H_axis.tick_params(axis='y',colors='crimson')
            H_axis.yaxis.set_minor_locator(AutoMinorLocator())
            P_axis.plot(otime,obs['p']+obs['pdyn'],label='obsP [nPa]',
                         c='tab:olive')
            P_axis.plot(vtime,virtual['P']+virtual['pdyn'],label='simP [nPa]',
                         c='dimgray',lw=1)
            if i==0:
                Bz_axis.legend(loc='lower right', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
                S_axis.legend(loc='lower left', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
                P_axis.legend(loc='lower right', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
                H_axis.legend(loc='lower left', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
            ##Decorations
            general_plot_settings(Bz_axis,
                                  legend=False,
                                  do_xlabel=False,
                                  ylabel=sat,
                                  ylim=[-60,100],
                                  timedelta=dotimedelta)
            Bz_axis.margins(x=0)
            Bz_axis.set_xlim([otime[0],otime[-1]])
            Bz_axis.axvline((moments['impact']-
                               moments['peak2']).total_seconds()*1e9,
                               ls='--',color='black')
            Bz_axis.axvline(0,ls='--',color='black')
            Bz_axis.fill_between(vtime,-1e11,1e11,color='mistyrose',
                                 where=((virtual['Status']>2)).values)
            Bz_axis.fill_between(vtime,-1e11,1e11,color='lightblue',
                                 where=((virtual['Status']<2)&
                                        (virtual['Status']>1)).values)
            Bz_axis.fill_between(vtime,-1e11,1e11,color='paleturquoise',
                                 where=((virtual['Status']<1)&
                                        (virtual['Status']>0)).values)
            Bz_axis.fill_between(vtime,-1e11,1e11,color='lightgrey',
                                 where=((virtual['Status']<0)).values)
            Bz_axis.tick_params(axis='y',colors='tab:blue')
            general_plot_settings(P_axis,
                                  legend=False,
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat,
                                  ylim=[0,5],
                                  timedelta=dotimedelta)
            P_axis.margins(x=0)
            P_axis.set_xlim([otime[0],otime[-1]])
            P_axis.axvline((moments['impact']-
                               moments['peak2']).total_seconds()*1e9,
                               ls='--',color='black')
            P_axis.axvline(0,ls='--',color='black')
            P_axis.fill_between(vtime,-1e11,1e11,color='mistyrose',
                                 where=((virtual['Status']>2)).values)
            P_axis.fill_between(vtime,-1e11,1e11,color='lightblue',
                                 where=((virtual['Status']<2)&
                                        (virtual['Status']>1)).values)
            P_axis.fill_between(vtime,-1e11,1e11,color='paleturquoise',
                                 where=((virtual['Status']<1)&
                                        (virtual['Status']>0)).values)
            P_axis.fill_between(vtime,-1e11,1e11,color='lightgrey',
                                 where=((virtual['Status']<0)).values)
            P_axis.tick_params(axis='y',colors='tab:olive')
        #save
        bp_compare.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        twochunk.tight_layout(pad=0.6,figure=bp_compare)
        #figurename = path+'/bp_compare'+phase+'_'+event+'.png'
        figurename = path+'/bp_compare'+phase+'_'+event+'.eps'
        bp_compare.savefig(figurename,dpi=300)
        plt.close(bp_compare)
        print('\033[92m Created\033[00m',figurename)
        #############
        '''
        #setup figure
        #p_compare_detail,axis = plt.subplots(len(satlist),1,
        #                                     figsize=[20,3*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            j = i+len(satlist)
            # Setup quickaccess and time format
            #virtual = dataset[event]['vsat'][sat+phase]
            #virtualtime = dataset[event][sat+'_vtime'+phase]
            #vtime = [float(t) for t in virtualtime.to_numpy()]
            #obs = dataset[event]['obssat'][sat+phase]
            #obstime = dataset[event][sat+'_otime'+phase]
            #otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            # H
            rjaxis = axis[j].twinx()
            rjaxis.plot(otime,np.sqrt(obs['Hx']**2+
                                        obs['Hy']**2+
                                        obs['Hz']**2)/1e9,
                                        label='obs|H| [MW]',c='crimson')
            rjaxis.plot(vtime,np.sqrt(virtual['Hx']**2+
                                        virtual['Hy']**2+
                                        virtual['Hz']**2)/1e9,
                                    label='sim|H| [MW]',lw=1,c='black')
            rjaxis.set_ylim([0,20])
            raxis.spines['right'].set_color('crimson')
            rjaxis.spines['left'].set_color('tab:olive')
            rjaxis.tick_params(axis='y',colors='crimson')
            rjaxis.yaxis.set_minor_locator(AutoMinorLocator())
            #axis[j].plot(otime,obs['bx'],label='obsBx',c='maroon')
            #axis[j].plot(otime,obs['by'],label='obsBy',c='magenta')
            axis[j].plot(otime,obs['p']+obs['pdyn'],label='obsP [nPa]',
                         c='tab:olive')
            #axis[j].plot(vtime,virtual['B_x'],label='simBx',c='maroon',
            #              ls='--')
            #axis[j].plot(vtime,virtual['B_y'],label='simBy',c='magenta',
            #              ls='--')
            axis[j].plot(vtime,virtual['P']+virtual['pdyn'],label='simP [nPa]',
                         c='dimgray',lw=1)
            if i==0:
                rjaxis.legend(loc='lower left', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
                axis[i].legend(loc='lower right', bbox_to_anchor=(0.5, 1.05),
                          ncol=2, fancybox=True, shadow=True)
            #Decorations
            general_plot_settings(axis[j],
                                  #legend=(i==0),
                                  #legend_loc='upper right',
                                  legend=False,
                                  do_xlabel=(j==2*len(satlist)-1),
                                  #ylabel=sat+r' $B\left[ nT\right]$',
                                  ylabel=sat,
                                  ylim=[0,5],
                                  timedelta=dotimedelta)
            axis[j].margins(x=0)
            axis[j].set_xlim([otime[0],otime[-1]])
            axis[j].axvline((moments['impact']-
                               moments['peak2']).total_seconds()*1e9,
                               ls='--',color='black')
            axis[j].axvline(0,ls='--',color='black')
            axis[j].fill_between(vtime,-1e11,1e11,color='red',alpha=0.2,
                                 where=((virtual['Status']>2)).values)
            axis[j].fill_between(vtime,-1e11,1e11,color='blue',alpha=0.2,
                                 where=((virtual['Status']<2)&
                                        (virtual['Status']>1)).values)
            axis[j].fill_between(vtime,-1e11,1e11,color='cyan',alpha=0.2,
                                 where=((virtual['Status']<1)&
                                        (virtual['Status']>0)).values)
            axis[j].fill_between(vtime,-1e11,1e11,color='grey',alpha=0.2,
                                 where=((virtual['Status']<0)).values)
            #axis[j].axvspan((moments['impact']-
            #                 moments['peak2']).total_seconds()*1e9,0,
            #                     color='grey',alpha=0.2)
            axis[j].tick_params(axis='y',colors='tab:olive')
        #save
        bp_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        bp_compare_detail.tight_layout(pad=0.6)
        figurename = path+'/bp_compare_detail'+phase+'_'+event+'.png'
        bp_compare_detail.savefig(figurename)
        plt.close(bp_compare_detail)
        print('\033[92m Created\033[00m',figurename)
        #############
        '''
        '''
        #setup figure
        u_compare_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['U_x'],label='simUx')
            axis[i].plot(vtime,virtual['U_y'],label='simUy')
            axis[i].plot(vtime,virtual['U_z'],label='simUz')
            axis[i].plot(otime,obs['vx'],label='obsUx')
            axis[i].plot(otime,obs['vy'],label='obsUy')
            axis[i].plot(otime,obs['vz'],label='obsUz')
            #Decorations
            general_plot_settings(axis[i],legend=(i==0),
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' $U\left[ km/s\right]$',
                                  ylim=[-200,200],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                             moments['peak2']).total_seconds()*1e9,
                             ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        u_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        u_compare_detail.tight_layout()
        figurename = path+'/u_compare_detail'+phase+'_'+event+'.png'
        u_compare_detail.savefig(figurename)
        plt.close(u_compare_detail)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        n_compare_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['Rho'],label='simN')
            axis[i].plot(otime,obs['n'],label='obsN')
            #Decorations
            general_plot_settings(axis[i],legend=(i==0),
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' $\rho\left[ amu/cm^3\right]$',
                                  ylim=[0,20],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        n_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        n_compare_detail.tight_layout()
        figurename = path+'/n_compare_detail'+phase+'_'+event+'.png'
        n_compare_detail.savefig(figurename)
        plt.close(n_compare_detail)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        status,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['Status'],label='simStatus')
            #axis[i].plot(otime,obs['n'],label='obsN')
            #Decorations
            general_plot_settings(axis[i],legend=(i==0),
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' Status',
                                  #ylim=[0,10],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                             moments['peak2']).total_seconds()*1e9,
                             ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        status.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        status.tight_layout()
        figurename = path+'/status'+phase+'_'+event+'.png'
        status.savefig(figurename)
        plt.close(status)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        p_compare_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['P'],label='simP')
            axis[i].plot(otime,obs['p'],label='obsP')
            #Decorations
            general_plot_settings(axis[i],legend=(i==0),
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' $P\left[ nPa\right]$',
                                  ylim=[0,7],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                          moments['peak2']).total_seconds()*1e9,
                         ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        p_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        p_compare_detail.tight_layout()
        figurename = path+'/p_compare_detail'+phase+'_'+event+'.png'
        p_compare_detail.savefig(figurename)
        plt.close(p_compare_detail)
        print('\033[92m Created\033[00m',figurename)
        '''
        #############
        #setup figure
        k_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,3*len(satlist)])
        h_detail,haxis = plt.subplots(len(satlist),1,
                                             figsize=[16,3*len(satlist)])
        s_detail,saxis = plt.subplots(len(satlist),1,
                                             figsize=[16,3*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            # K
            axis[i].plot(vtime,np.sqrt(virtual['Kx']**2+
                                       virtual['Ky']**2+
                                       virtual['Kz']**2)
                                       ,label='simK')
            axis[i].plot(otime,np.sqrt(obs['Kx']**2+
                                       obs['Ky']**2+
                                       obs['Kz']**2)
                                       ,label='obsK')
            # H
            haxis[i].plot(vtime,np.sqrt(virtual['Hx']**2+
                                        virtual['Hy']**2+
                                        virtual['Hz']**2)
                                       ,label='simH')
            haxis[i].plot(otime,np.sqrt(obs['Hx']**2+
                                        obs['Hy']**2+
                                        obs['Hz']**2)
                                       ,label='obsH')
            # S
            saxis[i].plot(vtime,np.sqrt(virtual['Sx']**2+
                                        virtual['Sy']**2+
                                        virtual['Sz']**2)
                                       ,label='simS')
            saxis[i].plot(otime,np.sqrt(obs['Sx']**2+
                                        obs['Sy']**2+
                                        obs['Sz']**2)
                                       ,label='obsS')
            #Decorations
            # K
            general_plot_settings(axis[i],legend=False,
                                  do_xlabel=(i==len(satlist)-1),
                                  #ylabel=sat+r' $|K|\left[ KW/Re^2\right]$',
                                  ylabel=sat,
                                  ylim=[0,1e11],
                                  timedelta=dotimedelta)
            # H
            general_plot_settings(haxis[i],legend=False,
                                  do_xlabel=(i==len(satlist)-1),
                                  #ylabel=sat+r' $|H|\left[ KW/Re^2\right]$',
                                  ylabel=sat,
                                  ylim=[0,1e11],
                                  timedelta=dotimedelta)
            # S
            general_plot_settings(saxis[i],legend=False,
                                  do_xlabel=(i==len(satlist)-1),
                                  #ylabel=sat+r' $|S|\left[ KW/Re^2\right]$',
                                  ylabel=sat,
                                  ylim=[0,1e11],
                                  timedelta=dotimedelta)
            for ax in [axis[i],haxis[i],saxis[i]]:
                ax.axvline((moments['impact']-
                               moments['peak2']).total_seconds()*1e9,
                               ls='--',color='black')
                ax.axvline(0,ls='--',color='black')
                ax.fill_between(vtime,1e11,color='red',alpha=0.2,
                                 where=((virtual['Status']>2)).values)
                ax.fill_between(vtime,1e11,color='blue',alpha=0.2,
                                 where=((virtual['Status']<2)&
                                        (virtual['Status']>1)).values)
                ax.fill_between(vtime,1e11,color='cyan',alpha=0.2,
                                 where=((virtual['Status']<1)&
                                        (virtual['Status']>0)).values)
                ax.fill_between(vtime,1e11,color='grey',alpha=0.2,
                                 where=((virtual['Status']<0)).values)
        #save
        # K
        k_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        k_detail.tight_layout(pad=0.04)
        figurename = path+'/k_detail'+phase+'_'+event+'.png'
        k_detail.savefig(figurename)
        plt.close(k_detail)
        print('\033[92m Created\033[00m',figurename)
        # H
        h_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        h_detail.tight_layout()
        figurename = path+'/h_detail'+phase+'_'+event+'.png'
        h_detail.savefig(figurename)
        plt.close(h_detail)
        print('\033[92m Created\033[00m',figurename)
        # S
        s_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        s_detail.tight_layout()
        figurename = path+'/s_detail'+phase+'_'+event+'.png'
        s_detail.savefig(figurename)
        plt.close(s_detail)
        print('\033[92m Created\033[00m',figurename)
        '''
        #############
        #setup figure
        ky_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['Ky'],label='simKy')
            axis[i].plot(otime,obs['Ky'],label='obsKy')
            rax = axis[i].twinx()
            rax.plot(vtime,virtual['Status'],label='simStatus',
                     c='black',ls='--')
            #Decorations
            general_plot_settings(axis[i],legend=True,
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' $K_y\left[ KW/Re^2\right]$',
                                  ylim=[-0.2e11,0.8e11],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                             moments['peak2']).total_seconds()*1e9,
                             ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        ky_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        ky_detail.tight_layout()
        figurename = path+'/ky_detail'+phase+'_'+event+'.png'
        ky_detail.savefig(figurename)
        plt.close(ky_detail)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        kz_detail,axis = plt.subplots(len(satlist),1,
                                             figsize=[16,8*len(satlist)])
        #Plot
        for i,sat in enumerate(satlist):
            # Setup quickaccess and time format
            virtual = dataset[event]['vsat'][sat+phase]
            virtualtime = dataset[event][sat+'_vtime'+phase]
            vtime = [float(t) for t in virtualtime.to_numpy()]
            obs = dataset[event]['obssat'][sat+phase]
            obstime = dataset[event][sat+'_otime'+phase]
            otime = [float(t) for t in obstime.to_numpy()]
            # Plot
            axis[i].plot(vtime,virtual['Kz'],label='simKz')
            axis[i].plot(otime,obs['Kz'],label='obsKz')
            rax = axis[i].twinx()
            rax.plot(vtime,virtual['Status'],label='simStatus',
                     c='black',ls='--')
            #Decorations
            general_plot_settings(axis[i],legend=True,
                                  do_xlabel=(i==len(satlist)-1),
                                  ylabel=sat+r' $K_z\left[ KW/Re^2\right]$',
                                  ylim=[-0.8e11,0.8e11],
                                  timedelta=dotimedelta)
            axis[i].axvline((moments['impact']-
                             moments['peak2']).total_seconds()*1e9,
                             ls='--',color='black')
            axis[i].axvline(0,ls='--',color='black')
        #save
        kz_compare_detail.suptitle(moments['peak1'].strftime(
                                                     "%b %Y, t0=%d-%H:%M:%S"),
                                      ha='left',x=0.01,y=0.99)
        kz_detail.tight_layout()
        figurename = path+'/kz_detail'+phase+'_'+event+'.png'
        kz_detail.savefig(figurename)
        plt.close(ky_detail)
        print('\033[92m Created\033[00m',figurename)
        #############
        '''

def time_integrated(dataset,phase,path):
    """Function creates a table of flux values integrated over a phase
    Inputs
        dataset
        path
        kwargs:
    Returns
        None
    """
    # Set the variables
    for i,event in enumerate(dataset.keys()):
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        mp = dataset[event]['mp'+phase]
        ## TOTAL
        #K1,5 from mp
        Ks1 = mp['K_netK1 [W]']
        Ks5 = mp['K_netK5 [W]']
        #K2,3,4 from lobes
        Ks2al = lobes['K_netK2a [W]']
        Ks2bl = lobes['K_netK2b [W]']
        Ks3 = lobes['K_netK3 [W]']
        Ks4 = lobes['K_netK4 [W]']
        #K2,6,7 from closed
        Ks2ac = closed['K_netK2a [W]']
        Ks2bc = closed['K_netK2b [W]']
        Ks6 = closed['K_netK6 [W]']
        Ks7 = closed['K_netK7 [W]']

        ## HYDRO
        #H1,5 from mp
        Hs1 = mp['P0_netK1 [W]']
        Hs5 = mp['P0_netK5 [W]']
        #H2,3,4 from lobes
        Hs2al = lobes['P0_netK2a [W]']
        Hs2bl = lobes['P0_netK2b [W]']
        Hs3 = lobes['P0_netK3 [W]']
        Hs4 = lobes['P0_netK4 [W]']
        #H2,6,7 from closed
        Hs2ac = closed['P0_netK2a [W]']
        Hs2bc = closed['P0_netK2b [W]']
        Hs6 = closed['P0_netK6 [W]']
        Hs7 = closed['P0_netK7 [W]']

        ## MAG
        #S1,5 from mp
        Ss1 = mp['ExB_netK1 [W]']
        Ss5 = mp['ExB_netK5 [W]']
        #S2,3,4 from lobes
        Ss2al = lobes['ExB_netK2a [W]']
        Ss2bl = lobes['ExB_netK2b [W]']
        Ss3 = lobes['ExB_netK3 [W]']
        Ss4 = lobes['ExB_netK4 [W]']
        #S2,6,7 from closed
        Ss2ac = closed['ExB_netK2a [W]']
        Ss2bc = closed['ExB_netK2b [W]']
        Ss6 = closed['ExB_netK6 [W]']
        Ss7 = closed['ExB_netK7 [W]']

        ## TOTAL
        #M1,5,total from mp
        M1 = mp['UtotM1 [W]']
        M5 = mp['UtotM5 [W]']
        M = mp['UtotM [W]']
        #M1a,1b,2b,il from lobes
        M1a = lobes['UtotM1a [W]']
        M1b = lobes['UtotM1b [W]']
        M2b = lobes['UtotM2b [W]']
        M2d = lobes['UtotM2d [W]']
        Mil = lobes['UtotMil [W]']
        #M5a,5b,2a,ic from closed
        M5a = closed['UtotM5a [W]']
        M5b = closed['UtotM5b [W]']
        M2a = closed['UtotM2a [W]']
        M2c = closed['UtotM2c [W]']
        Mic = closed['UtotMic [W]']

        M_lobes = M1a+M1b-M2a+M2b-M2c+M2d
        M_closed = M5a+M5b+M2a-M2b+M2c-M2d

        ## HYDRO
        #HM1,5,total from mp
        HM1 = mp['uHydroM1 [W]']
        HM5 = mp['uHydroM5 [W]']
        HM = mp['uHydroM [W]']
        #HM1a,1b,2b,il from lobes
        HM1a = lobes['uHydroM1a [W]']
        HM1b = lobes['uHydroM1b [W]']
        HM2b = lobes['uHydroM2b [W]']
        HM2d = lobes['uHydroM2d [W]']
        HMil = lobes['uHydroMil [W]']
        #HM5a,5b,2a,ic from closed
        HM5a = closed['uHydroM5a [W]']
        HM5b = closed['uHydroM5b [W]']
        HM2a = closed['uHydroM2a [W]']
        HM2c = closed['uHydroM2c [W]']
        HMic = closed['uHydroMic [W]']

        HM_lobes = HM1a+HM1b-HM2a+HM2b-HM2c+HM2d
        HM_closed = HM5a+HM5b+HM2a-HM2b+HM2c-HM2d

        ## MAG
        #SM1,5,total from mp
        SM1 = mp['uBM1 [W]']
        SM5 = mp['uBM5 [W]']
        SM = mp['uBM [W]']
        #HM1a,1b,2b,il from lobes
        SM1a = lobes['uBM1a [W]']
        SM1b = lobes['uBM1b [W]']
        SM2b = lobes['uBM2b [W]']
        SM2d = lobes['uBM2d [W]']
        SMil = lobes['uBMil [W]']
        #SM5a,5b,2a,ic from closed
        SM5a = closed['uBM5a [W]']
        SM5b = closed['uBM5b [W]']
        SM2a = closed['uBM2a [W]']
        SM2c = closed['uBM2c [W]']
        SMic = closed['uBMic [W]']

        SM_lobes = SM1a+SM1b-SM2a+SM2b-SM2c+SM2d
        SM_closed = SM5a+SM5b+SM2a-SM2b+SM2c-SM2d

        # Central difference of partial volume integrals, total change
        # Total
        K_closed = -1*central_diff(closed['Utot [J]'])
        K_lobes = -1*central_diff(lobes['Utot [J]'])
        K_mp = -1*central_diff(mp['Utot [J]'])
        K_mp.iloc[0] = 0
        # Hydro
        H_closed = -1*central_diff(closed['uHydro [J]'])
        H_lobes = -1*central_diff(lobes['uHydro [J]'])
        H_mp = -1*central_diff(mp['uHydro [J]'])
        H_mp.iloc[0] = 0
        # Mag
        S_closed = -1*central_diff(closed['uB [J]'])
        S_lobes = -1*central_diff(lobes['uB [J]'])
        S_mp = -1*central_diff(mp['uB [J]'])
        S_mp.iloc[0] = 0

        # Load into a dictionary
        flux_dict = {
                     '1':[Hs1+HM1,Ss1+SM1,Ks1+M1],
                     '2a (closed)':[Hs2ac+HM2a,Ss2ac+SM2a,Ks2ac+M2a],
                     '2b (closed)':[Hs2bc-HM2b,Ss2bc-SM2b,Ks2bc-M2b],
                     '3':[Hs3,Ss3,Ks3],
                     '4':[Hs4,Ss4,Ks4],
                     '5':[Hs5+HM5,Ss5+SM5,Ks5+M5],
                     '6':[Hs6,Ss6,Ks6],
                     '7':[Hs7,Ss7,Ks7],
                     'Summed':[
                         Hs1+Hs5+Hs3+Hs4+Hs6+Hs7+HM,
                         Ss1+Ss5+Ss3+Ss4+Ss6+Ss7+SM,
                         Ks1+Ks5+Ks3+Ks4+Ks6+Ks7+M],
                     'Check_mp':[H_mp,S_mp,K_mp]}
        # Print to screen
        print('\n{:<15}{:<20}{:<20}{:<20}'.format('ID','|H|',
                                                     '|S|',
                                                     '|K|'))
        print('{:<15}{:<20}{:<20}{:<20}'.format('*******','*******',
                                                '*******','*******'))
        for num,[hydro,poynting,total] in flux_dict.items():
            Hdt = integrate.trapezoid(hydro.fillna(method='ffill').values,
                                      dx=60)
            Sdt = integrate.trapezoid(poynting.fillna(method='ffill').values,
                                      dx=60)
            Kdt = integrate.trapezoid(total.fillna(method='ffill').values,
                                      dx=60)
            if num=='Summed' or num=='Check_mp':
                ls='--'
            else:
                ls=None
            plt.plot(total*60/1e12,label=num,ls=ls)
            #plt.plot(poynting.cumsum()*60/1e15,label=num,ls=ls)
            plt.legend()
            print('{:<15}{:<+20.2f}{:<+20.2f}{:<+20.2f}'.format(num,
                                                  Hdt/1e15,Sdt/1e15,Kdt/1e15))
        print('{:<15}{:<20}{:<20}{:<20}'.format('*******','*******',
                                                  '*******','*******'))
        #TODO: look at plot of error and see if it has spikes
        #       Check error between cdiff and summed for lobes
        #       Check error between cdiff and summed for closed
        #       Errors with everything combined
        #       If it IS there at every timestep then maybe doing some kind
        #           of error correction?
        #       Look at an M case and see if maybe using average quantities
        #           vs split quantities is better
        #       *Or even ONLY using magnetosphere quantities*

    pass

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
        #mp = dataset[event]['mpdict'+phase]['ms_full']
        mp = dataset[event]['mp'+phase]
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        U_day = mp['UtotK1day [J]']+mp['UtotK5day [J]']
        U_night = mp['UtotK1night [J]']+mp['UtotK5night [J]']
        K_day = -1*central_diff(U_day)
        M_day = K_day - mp['K_netK1day [W]']+mp['K_netK5day [W]']
        M_2alobes = M_day*0
        M_2aclosed = M_day*0
        K_2alobes = M_2alobes + lobes['K_netK2a [W]']
        K_2aclosed = M_2aclosed + closed['K_netK2a [W]']

        K_night = -1*central_diff(U_night)
        M_night = K_night - mp['K_netK1night [W]']+mp['K_netK5night [W]']
        M_2blobes = M_day*0
        M_2bclosed = M_day*0
        K_2blobes = M_2blobes + lobes['K_netK2b [W]']
        K_2bclosed = M_2bclosed + closed['K_netK2b [W]']

        M_1 = (-1*central_diff(lobes['Utot [J]']) -lobes['K_net [W]']
               -M_2alobes-M_2blobes)
        M_5 =(-1*central_diff(closed['Utot [J]']) -closed['K_net [W]']
               -M_2aclosed-M_2bclosed)
        M_1day = (-1*central_diff(mp['UtotK1day [J]']))
                  #-mp['K_netK1day [W]']-lobes['K_netK2a [W]'])
        M_5day = (-1*central_diff(mp['UtotK5day [J]'])
                  -mp['K_netK5day [W]']-closed['K_netK2a [W]'])
        M_1night = (-1*central_diff(mp['UtotK1night [J]']))
                  #-mp['K_netK1night [W]']-lobes['K_netK2b [W]'])
        M_5night = (-1*central_diff(mp['UtotK5night [J]'])
                  -mp['K_netK5night [W]']-closed['K_netK2b [W]'])
        K_1 = M_1 + mp['K_netK1 [W]']
        K_5 = M_5 + mp['K_netK5 [W]']

        # Central difference of partial volume integrals, total change
        # Total
        K_closed = -1*central_diff(closed['Utot [J]'])
        K_lobes = -1*central_diff(lobes['Utot [J]'])
        K_mp = -1*central_diff(mp['Utot [J]'])
        # Partials (spatial)
        #K_1 = -1*central_diff(U_1)
        #K_2a = -1*central_diff(U_2a)
        #K_2b = -1*central_diff(U_2b)
        K_3 = -1*central_diff(lobes['UtotK3 [J]'])
        K_4 = -1*central_diff(lobes['UtotK4 [J]'])
        #K_5 = -1*central_diff(U_5)
        K_6 = -1*central_diff(closed['UtotK6 [J]'])
        K_7 = -1*central_diff(closed['UtotK7 [J]'])

        # Motional components derived from total - static
        # Total
        M_closed = K_closed - closed['K_net [W]']
        M_lobes = K_lobes - lobes['K_net [W]']
        M_mp = K_mp - mp['K_net [W]']
        # Partials
        #M_1 = K_1 - mp['K_netK1 [W]']
        M_2a_lobes = K_2alobes - lobes['K_netK2a [W]']
        M_2a_closed = K_2aclosed - closed['K_netK2a [W]']
        M_2b_lobes = K_2blobes - lobes['K_netK2b [W]']
        M_2b_closed = K_2bclosed - closed['K_netK2b [W]']
        M_3 = K_3 - lobes['K_netK3 [W]']
        M_4 = K_4 - lobes['K_netK4 [W]']
        #M_5 = K_5 - mp['K_netK5 [W]']
        M_6 = K_6 - closed['K_netK6 [W]']
        M_7 = K_7 - closed['K_netK7 [W]']
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
        #IMF
        top.fill_between(swt,sw['B'], ec='dimgrey',fc='thistle',
                               label=r'$|B|$')
        top.plot(swt,sw['bx'],label=r'$B_x$',c='maroon')
        top.plot(swt,sw['by'],label=r'$B_y$',c='magenta')
        top.plot(swt,sw['bz'],label=r'$B_z$',c='tab:blue')
        timeline = top.axvline(mpt[0],color='black')
        general_plot_settings(top,ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              legend_loc='upper left',
                              timedelta=dotimedelta)
        '''
        #Dst index
        top.plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        top.plot(ot,obs['sym_h'],label='Obs',c='maroon')
        timeline = top.axvline(mpt[0],color='black')
        general_plot_settings(top,ylabel=r'Sym-H$\left[nT\right]$',
                                  do_xlabel=True, legend=True,
                                  timedelta=dotimedelta)
        '''
        # Main image with diagram background
        main.imshow(background)
        # Base vectors for each item
        #K1
        k1 = main.quiver([1000],[130],[-1500],[3000],color='green',
                         scale=10,scale_units='dots')
        k1text = main.text(1120,160,r'$\vec{K}_1$')
        #K2a
        k2a = main.quiver([760],[470],[-3000],[-1500],color='green',
                         scale=10,scale_units='dots')
        k2atext = main.text(735,440,r'$\vec{K}_{2a}$')
        #K2b
        k2b = main.quiver([1060],[375],[1500],[-3000],color='green',
                         scale=10,scale_units='dots')
        k2btext = main.text(1060,330,r'$\vec{K}_{2b}$')
        #K3
        k3 = main.quiver([805],[670],[1000],[4000],color='green',
                         scale=10,scale_units='dots')
        k3text = main.text(660,770,r'$\vec{K}_3$')
        #K4
        k4 = main.quiver([1350],[1150],[5000],[0],color='green',
                         scale=10,scale_units='dots')
        k4text = main.text(1125,1100,r'$\vec{K}_4$')
        #K5
        k5 = main.quiver([560],[510],[-4000],[1000],color='green',
                         scale=10,scale_units='dots')
        k5text = main.text(370,575,r'$\vec{K}_5$')
        #K6
        k6 = main.quiver([910],[625],[-4000],[1000],color='green',
                         scale=10,scale_units='dots')
        k6text = main.text(920,580,r'$\vec{K}_6$')
        #K7
        k7 = main.quiver([1350],[600],[5000],[0],color='green',
                         scale=10,scale_units='dots')
        k7text = main.text(1130,700,r'$\vec{K}_7$')
        base_values = {}
        for i,(now,xpos)in enumerate([z for z in zip(mp.index,mpt)]
                                     #[1336:1337]
                                     #[360:362]
                                     [-30:-1]
                                     ):
            for j,vector in enumerate([
                           {'vector':k1,
                            'text':k1text,
                            'data':K_1.fillna(value=0)},
                           {'vector':k2a,
                            'text':k2atext,
                            'data':K_2alobes.fillna(value=0)},
                           {'vector':k2b,
                            'text':k2btext,
                            'data':K_2blobes.fillna(value=0)},
                           {'vector':k3,
                            'text':k3text,
                            'data':K_3.fillna(value=0)},
                           {'vector':k4,
                            'text':k4text,
                            'data':K_4.fillna(value=0)},
                           {'vector':k5,
                            'text':k5text,
                            'data':K_5.fillna(value=0)},
                           {'vector':k6,
                            'text':k6text,
                            'data':K_6.fillna(value=0)},
                           {'vector':k7,
                            'text':k7text,
                            'data':K_7.fillna(value=0)}
                           ]):
                #factor = vector['data'][now]/max(abs(
                #                  vector['data'].quantile([0.05,0.95])))
                if i==0:
                    base_values.update({'baseU'+str(j):vector['vector'].U})
                    base_values.update({'baseV'+str(j):vector['vector'].V})
                    base_values.update({'basetext'+str(j):
                                        vector['text'].get_text()})
                factor = vector['data'][now]/10e12
                vector['vector'].set_UVC(
                    np.array([int(base_values['baseU'+str(j)]*factor)]),
                    np.array([int(base_values['baseV'+str(j)]*factor)]))
                vector['text'].set_text(base_values['basetext'+str(j)]+
                              '{:<.2}'.format(vector['data'][now]/1e12))
            #save
            if now<moments['impact']:
                labelcolor='black'
            elif now>moments['impact']:
                labelcolor='red'
            elif now>moments['peak2']:
                labelcolor='blue'
            timeline.set_xdata([xpos])
            figure.suptitle(now.strftime("%b %Y, t0=%d-%H:%M:%S"),ha='left',
                                    x=0.01,y=0.99,color=labelcolor)
            if i==0:
                main.set_xticks([])
                main.set_yticks([])
                figure.tight_layout(pad=0.8)
            figname = os.path.join(path,'diagram_summaries_'+event,
                                   'state_'+str(i)+'.png')
            figure.savefig(figname)
            print('\033[92m Created\033[00m',figname)

def main_rec_figures(dataset):
    ##Main + Recovery phase
    #hatches = ['','*','x','o']
    hatches = ['','','','']
    #for phase,path in [('_lineup',outLine),('_main',outMN1),('_rec',outRec)]:
    for phase,path in [('_lineup',outLine)]:
        #stack_energy_type_fig(dataset,phase,path)
        stack_energy_region_fig(dataset,phase,path,hatches,tabulate=False)
        #stack_volume_fig(dataset,phase,path,hatches)
        #interf_power_fig(dataset,phase,path,hatches)
        #polar_cap_area_fig(dataset,phase,path)
        #tail_cap_fig(dataset,phase,path)
        #static_motional_fig(dataset,phase,path)
        solarwind_figure(dataset,phase,path,hatches,tabulate=False)
        lobe_balance_fig(dataset,phase,path)
        #lobe_power_histograms(dataset, phase, path,doratios=False)
        #lobe_power_histograms(dataset, phase, path,doratios=True)
        #power_correlations(dataset,phase,path,optimize_tshift=True)
        #quantify_timings2(dataset, phase, path)
        #satellite_comparisons(dataset, phase, path)
        #oneD_comparison(dataset,phase,path)
        pass
    #power_correlations2(dataset,'',unfiled, optimize_tshift=False)#Whole event
    #polar_cap_flux_stats(dataset,unfiled)
    #diagram_summary(dataset,'',unfiled)
    #time_integrated(dataset,'_main',unfiled)

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
        #quantify_timings2(dataset, phase, path)
        lobe_balance_fig(dataset,phase,path)
        #oneD_comparison(dataset,phase,path)
        #diagram_summary(dataset,phase,unfiled)
        #lobe_power_histograms(dataset, phase, path)
    #time_integrated(dataset,'_interv',unfiled)

if __name__ == "__main__":
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inLogs = os.path.join(sys.argv[-1],'data/logs/')
    #inLogs = os.path.join(inBase,'')
    inSats = os.path.join(inBase,'data/sats/')
    inGround = os.path.join(inBase,'data/ground/')
    inAnalysis = os.path.join(sys.argv[-1],'data/analysis/')
    #inAnalysis = os.path.join(inBase,'analysis/GM/')
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

    ## Analysis Data
    dataset = {}
    #dataset['may'] = load_hdf_sort(inAnalysis+'may2019_results.h5')
    #dataset['may'] = load_hdf_sort(inAnalysis+'temp/test_may.h5')
    #dataset['feb'] = load_hdf_sort(inAnalysis+'feb2014_results.h5',
    #                               tshift=45)
    #dataset['star'] = load_hdf_sort(inAnalysis+'starlink2_results4Re.h5')
    #dataset['star4'] = load_hdf_sort(inAnalysis+'starlink2_results4Re.h5')
    #dataset['star'] = {}
    #dataset['aug'] = {}
    #dataset['jun'] = {}
    #dataset['2000'] = load_hdf_sort(inAnalysis+'gm_results.h5')
    dataset['ideal'] = load_hdf_sort(inAnalysis+'GM/gm_results.h5')

    ## Log Data and Indices
    #dataset['may']['obs'] = read_indices(inLogs, prefix='may2019_',
    #                                read_supermag=False)
    #dataset['feb']['obs'] = read_indices(inLogs, prefix='feb2014_',
    #                                read_supermag=False, tshift=45)
    #dataset['star4']['obs'] = read_indices(inLogs, prefix='starlink_',
    #                                 read_supermag=False,
    #                                 end=dataset['star4']['msdict']['closed'].index[-1],
    #             magStationFile=inGround+'magnetometers_e20220202-050000.mag')
    #dataset['2000']['obs'] = read_indices(inLogs, prefix='', read_supermag=False)
    dataset['ideal']['obs'] = read_indices(inLogs, prefix='',
                                           read_supermag=True)
    #dataset['star']['obs'] = {}
    #dataset['star4']['obs'] = {}
    #dataset['aug']['obs'] = read_indices(inLogs, prefix='aug2018_',
    #                                     read_supermag=False)
    #dataset['jun']['obs'] = read_indices(inLogs, prefix='jun2015_',
    #                                     read_supermag=False)

    ## Satellite Data
    #dataset['star']['vsat'],dataset['star']['obssat'] = {},{}
    #dataset['star4']['vsat'],dataset['star4']['obssat'] = {},{}
    #dataset['star4']['vsat'],dataset['star4']['obssat'] = read_satellites(
    #                                                                inSats)
    #dataset['2000']['vsat'],dataset['2000']['obssat'] = {},{}
    dataset['ideal']['vsat'],dataset['ideal']['obssat'] = {},{}

    for event_key in dataset.keys():
        event = dataset[event_key]
        if 'msdict' in event.keys():
            # Total
            event['msdict']['lobes']['K_netK1 [W]'] = event['mpdict'][
                                                  'ms_full']['K_netK1 [W]']
            event['mpdict']['ms_full']['UtotM1 [W]'] = (
                                      event['msdict']['lobes']['UtotM1a [W]']+
                                      event['msdict']['lobes']['UtotM1b [W]'])
            event['mpdict']['ms_full']['UtotM5 [W]'] = (
                                     event['msdict']['closed']['UtotM5a [W]']+
                                     event['msdict']['closed']['UtotM5b [W]'])
            event['mpdict']['ms_full']['UtotM [W]'] = (
                                     event['mpdict']['ms_full']['UtotM1 [W]']+
                                     event['mpdict']['ms_full']['UtotM5 [W]'])
            # Hydro
            event['msdict']['lobes']['P0_netK1 [W]'] = event['mpdict'][
                                                  'ms_full']['P0_netK1 [W]']
            event['mpdict']['ms_full']['uHydroM1 [W]'] = (
                                    event['msdict']['lobes']['uHydroM1a [W]']+
                                    event['msdict']['lobes']['uHydroM1b [W]'])
            event['mpdict']['ms_full']['uHydroM5 [W]'] = (
                                    event['msdict']['closed']['uHydroM5a [W]']+
                                    event['msdict']['closed']['uHydroM5b [W]'])
            event['mpdict']['ms_full']['uHydroM [W]'] = (
                                    event['mpdict']['ms_full']['uHydroM1 [W]']+
                                    event['mpdict']['ms_full']['uHydroM5 [W]'])
            # Mag
            event['msdict']['lobes']['ExB_netK1 [W]'] = event['mpdict'][
                                                  'ms_full']['ExB_netK1 [W]']
            event['mpdict']['ms_full']['uBM1 [W]'] = (
                                      event['msdict']['lobes']['uBM1a [W]']+
                                      event['msdict']['lobes']['uBM1b [W]'])
            event['mpdict']['ms_full']['uBM5 [W]'] = (
                                     event['msdict']['closed']['uBM5a [W]']+
                                     event['msdict']['closed']['uBM5b [W]'])
            event['mpdict']['ms_full']['uBM [W]'] = (
                                     event['mpdict']['ms_full']['uBM1 [W]']+
                                     event['mpdict']['ms_full']['uBM5 [W]'])

    ##Construct "grouped" set of subzones, then get %contrib for each
    for event in dataset.keys():
        if 'msdict' in dataset[event].keys():
            dataset[event]['msdict'] = {
                #'rc':dataset[event]['msdict'].get('rc',pd.DataFrame()),
                #'xslice':dataset[event]['msdict'].get(
                #                               'xslice',pd.DataFrame()),
                'closed':dataset[event]['msdict'].get(
                                              'closed',pd.DataFrame()),
                'lobes':dataset[event]['msdict'].get(
                                               'lobes',pd.DataFrame())}
    ##Parse storm phases
    for event_key in [k for k in dataset.keys() if 'aug' not in k and
                                                   'jun' not in k]:
        event = dataset[event_key]
        keys = event.copy().keys()
        for phase in ['_qt','_main','_rec','_interv','_lineup']:
            for key in keys:
                if key == 'mpdict':
                    event['mp'+phase], event['time'+phase]=parse_phase(
                                         event['mpdict']['ms_full'],phase)
                elif key == 'inner_mp':
                    event['inner_mp'+phase], event['time'+phase]=parse_phase(
                                                  event['inner_mp'],phase)
                elif key == 'msdict':
                    event['msdict'+phase], event['time'+phase] = parse_phase(
                                                    event['msdict'],phase)
                elif key == 'termdict':
                    event['termdict'+phase],event['time'+phase]=parse_phase(
                                                 event['termdict'],phase)
                else:
                    pass
                    #event[key+phase],event['time'+phase] = parse_phase(
                    #                                         event[key],phase)
                    #try:
                    #    event[key+phase],event['time'+phase] = parse_phase(
                    #                                         event[key],phase)
                    #except:
                    #    print('WORK ON THIS')
    for event_key in dataset.keys():
        event = dataset[event_key]
        if 'obs' in event.keys():
            obs_srcs = list(event['obs'].keys())
        else:
            obs_srcs = []
        if 'obssat' in event.keys():
            satlist = list([sat for sat in event['obssat'].keys()
                            if not event['obssat'][sat].empty])
        else:
            satlist = []
        for phase in ['_qt','_main','_rec','_interv','_lineup']:
            for src in obs_srcs:
                event['obs'][src+phase],event[src+'_otime'+phase]=(
                                parse_phase(event['obs'][src],phase))
            for sat in satlist:
                event['vsat'][sat+phase],event[sat+'_vtime'+phase] = (
                                        parse_phase(event['vsat'][sat],phase))
                event['obssat'][sat+phase],event[sat+'_otime'+phase] = (
                                      parse_phase(event['obssat'][sat],phase))
    '''
        for sat in satlist:
            crossings = find_crossings(event['vsat'][sat],
                                       event['obssat'][sat],sat)
        '''
    ######################################################################
    ##Main + Recovery phase
    main_rec_figures(dataset)
    ######################################################################
    ##Short zoomed in interval
    #interval_figures(dataset)
    ######################################################################
    #TODO
    ph = '_lineup'
    phase = '_interv'
    if False:
        event = 'star'
        exterior = dataset[event]['sphere10_surface']
        perfect_exterior = dataset[event]['perfectsphere10_surface']
        sw_exterior = dataset[event]['solarwind10_surface']
        c35_exterior = dataset[event]['centered35-10_surface']
        c375_exterior = dataset[event]['centered375-10_surface']
        c3875_exterior = dataset[event]['centered3875-10_surface']
        c4_exterior = dataset[event]['centered4-10_surface']
        c45_exterior = dataset[event]['centered45-10_surface']
        c5_exterior = dataset[event]['centered5-10_surface']
        # Initialize the quantities of interest and time
        for key,value in dataset[event]['sphere10_volume'].items():
            exterior[key] = value
        for key,value in dataset[event]['solarwind10_volume'].items():
            sw_exterior[key] = value
        for key,value in dataset[event]['centered35-10_volume'].items():
            c35_exterior[key] = value
        for key,value in dataset[event]['centered375-10_volume'].items():
            c375_exterior[key] = value
        for key,value in dataset[event]['centered3875-10_volume'].items():
            c3875_exterior[key] = value
        for key,value in dataset[event]['centered4-10_volume'].items():
            c4_exterior[key] = value
        for key,value in dataset[event]['centered45-10_volume'].items():
            c45_exterior[key] = value
        for key,value in dataset[event]['centered5-10_volume'].items():
            c5_exterior[key] = value
        interior = dataset[event]['sphere10_inner_surface']
        perfect_interior = dataset[event]['perfectsphere10_inner_surface']
        sw_interior = dataset[event]['solarwind10_inner_surface']
        c35_interior = dataset[event]['centered35-10_inner_surface']
        c375_interior = dataset[event]['centered375-10_inner_surface']
        c3875_interior = dataset[event]['centered3875-10_inner_surface']
        c4_interior = dataset[event]['centered4-10_inner_surface']
        c45_interior = dataset[event]['centered45-10_inner_surface']
        c5_interior = dataset[event]['centered5-10_inner_surface']
        times = c4_interior.index
        moments = locate_phase(c4_interior.index)
        # Short hand for fluxes
        interv = times[times>moments['impact']]
        #interv = times
        '''
        Hs1 = 2.5/1.5*exterior.loc[interv,'P0_net [W]']
        Hs3 = 2.5/1.5*interior.loc[interv,'P0_net [W]']
        pHs1 = 2.5/1.5*perfect_exterior.loc[interv,'P0_net [W]']
        pHs3 = 2.5/1.5*perfect_interior.loc[interv,'P0_net [W]']
        Ss1 = exterior.loc[interv,'ExB_net [W]']
        Ss3 = interior.loc[interv,'ExB_net [W]']
        pSs1 = perfect_exterior.loc[interv,'ExB_net [W]']
        pSs3 = perfect_interior.loc[interv,'ExB_net [W]']
        Ks1 = Hs1+Ss1
        Ks3 = Hs3+Ss3
        pKs1 = pHs1+pSs1
        pKs3 = pHs3+pSs3
        '''
        # Fluxes
        # Regular sphere
        Ks1 = exterior.loc[interv,'K_net [W]']
        Ks3 = interior.loc[interv,'K_net [W]']
        Hs1 = exterior.loc[interv,'P0_net [W]']
        Hs3 = interior.loc[interv,'P0_net [W]']
        Ss1 = exterior.loc[interv,'ExB_net [W]']
        Ss3 = interior.loc[interv,'ExB_net [W]']
        # Continuous sphere
        pKs1 = perfect_exterior.loc[interv,'K_net [W]']
        pKs3 = perfect_interior.loc[interv,'K_net [W]']
        pHs1 = perfect_exterior.loc[interv,'P0_net [W]']
        pHs3 = perfect_interior.loc[interv,'P0_net [W]']
        pSs1 = perfect_exterior.loc[interv,'ExB_net [W]']
        pSs3 = perfect_interior.loc[interv,'ExB_net [W]']
        # Solar wind sphere
        swKs1 = sw_exterior.loc[interv,'K_net [W]']
        swKs3 = sw_interior.loc[interv,'K_net [W]']
        swHs1 = sw_exterior.loc[interv,'P0_net [W]']
        swHs3 = sw_interior.loc[interv,'P0_net [W]']
        swSs1 = sw_exterior.loc[interv,'ExB_net [W]']
        swSs3 = sw_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 3.5Re sphere
        c35Ks1 = c35_exterior.loc[interv,'K_net [W]']
        c35Ks3 = c35_interior.loc[interv,'K_net [W]']
        c35Hs1 = c35_exterior.loc[interv,'P0_net [W]']
        c35Hs3 = c35_interior.loc[interv,'P0_net [W]']
        c35Ss1 = c35_exterior.loc[interv,'ExB_net [W]']
        c35Ss3 = c35_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 3.5Re sphere
        c375Ks1 = c375_exterior.loc[interv,'K_net [W]']
        c375Ks3 = c375_interior.loc[interv,'K_net [W]']
        c375Hs1 = c375_exterior.loc[interv,'P0_net [W]']
        c375Hs3 = c375_interior.loc[interv,'P0_net [W]']
        c375Ss1 = c375_exterior.loc[interv,'ExB_net [W]']
        c375Ss3 = c375_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 3.5Re sphere
        c3875Ks1 = c3875_exterior.loc[interv,'K_net [W]']
        c3875Ks3 = c3875_interior.loc[interv,'K_net [W]']
        c3875Hs1 = c3875_exterior.loc[interv,'P0_net [W]']
        c3875Hs3 = c3875_interior.loc[interv,'P0_net [W]']
        c3875Ss1 = c3875_exterior.loc[interv,'ExB_net [W]']
        c3875Ss3 = c3875_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 4Re sphere
        c4Ks1 = c4_exterior.loc[interv,'K_net [W]']
        c4Ks3 = c4_interior.loc[interv,'K_net [W]']
        c4Hs1 = c4_exterior.loc[interv,'P0_net [W]']
        c4Hs3 = c4_interior.loc[interv,'P0_net [W]']
        c4Ss1 = c4_exterior.loc[interv,'ExB_net [W]']
        c4Ss3 = c4_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 4.5Re sphere
        c45Ks1 = c45_exterior.loc[interv,'K_net [W]']
        c45Ks3 = c45_interior.loc[interv,'K_net [W]']
        c45Hs1 = c45_exterior.loc[interv,'P0_net [W]']
        c45Hs3 = c45_interior.loc[interv,'P0_net [W]']
        c45Ss1 = c45_exterior.loc[interv,'ExB_net [W]']
        c45Ss3 = c45_interior.loc[interv,'ExB_net [W]']
        # Innerboundary shifted -> 5Re sphere
        c5Ks1 = c5_exterior.loc[interv,'K_net [W]']
        c5Ks3 = c5_interior.loc[interv,'K_net [W]']
        c5Hs1 = c5_exterior.loc[interv,'P0_net [W]']
        c5Hs3 = c5_interior.loc[interv,'P0_net [W]']
        c5Ss1 = c5_exterior.loc[interv,'ExB_net [W]']
        c5Ss3 = c5_interior.loc[interv,'ExB_net [W]']
        # Volume integrated derivatives
        # Regular
        U = exterior.loc[interv,'Utot [J]']
        uB = exterior.loc[interv,'uB [J]']
        uH = exterior.loc[interv,'uHydro [J]']
        K_sp = -1*central_diff(U)
        K_sp_fwd = -1*central_diff(U,forward=True)
        H_sp = -1*central_diff(uH)
        S_sp = -1*central_diff(uB)
        # Solar wind
        swU = sw_exterior.loc[interv,'Utot [J]']
        swuB = sw_exterior.loc[interv,'uB [J]']
        swuH = sw_exterior.loc[interv,'uHydro [J]']
        swK_sp = -1*central_diff(swU)
        swK_sp_fwd = -1*central_diff(swU,forward=True)
        swH_sp = -1*central_diff(swuH)
        swS_sp = -1*central_diff(swuB)
        # Shifted 3.5Re
        c35U = c35_exterior.loc[interv,'Utot [J]']
        c35uB = c35_exterior.loc[interv,'uB [J]']
        c35uH = c35_exterior.loc[interv,'uHydro [J]']
        c35K_sp = -1*central_diff(c35U)
        c35K_sp_fwd = -1*central_diff(c35U,forward=True)
        c35H_sp = -1*central_diff(c35uH)
        c35S_sp = -1*central_diff(c35uB)
        # Shifted 3.5Re
        c375U = c375_exterior.loc[interv,'Utot [J]']
        c375uB = c375_exterior.loc[interv,'uB [J]']
        c375uH = c375_exterior.loc[interv,'uHydro [J]']
        c375K_sp = -1*central_diff(c375U)
        c375K_sp_fwd = -1*central_diff(c375U,forward=True)
        c375H_sp = -1*central_diff(c375uH)
        c375S_sp = -1*central_diff(c375uB)
        # Shifted 3.5Re
        c3875U = c3875_exterior.loc[interv,'Utot [J]']
        c3875uB = c3875_exterior.loc[interv,'uB [J]']
        c3875uH = c3875_exterior.loc[interv,'uHydro [J]']
        c3875K_sp = -1*central_diff(c3875U)
        c3875K_sp_fwd = -1*central_diff(c3875U,forward=True)
        c3875H_sp = -1*central_diff(c3875uH)
        c3875S_sp = -1*central_diff(c3875uB)
        # Shifted 4Re
        c4U = c4_exterior.loc[interv,'Utot [J]']
        c4uB = c4_exterior.loc[interv,'uB [J]']
        c4uH = c4_exterior.loc[interv,'uHydro [J]']
        c4K_sp = -1*central_diff(c4U)
        c4K_sp_fwd = -1*central_diff(c4U,forward=True)
        c4H_sp = -1*central_diff(c4uH)
        c4S_sp = -1*central_diff(c4uB)
        # Shifted 3.5Re
        c45U = c45_exterior.loc[interv,'Utot [J]']
        c45uB = c45_exterior.loc[interv,'uB [J]']
        c45uH = c45_exterior.loc[interv,'uHydro [J]']
        c45K_sp = -1*central_diff(c45U)
        c45K_sp_fwd = -1*central_diff(c45U,forward=True)
        c45H_sp = -1*central_diff(c45uH)
        c45S_sp = -1*central_diff(c45uB)
        # Shifted 5Re
        c5U = c5_exterior.loc[interv,'Utot [J]']
        c5uB = c5_exterior.loc[interv,'uB [J]']
        c5uH = c5_exterior.loc[interv,'uHydro [J]']
        c5K_sp = -1*central_diff(c5U)
        c5K_sp_fwd = -1*central_diff(c5U,forward=True)
        c5H_sp = -1*central_diff(c5uH)
        c5S_sp = -1*central_diff(c5uB)
        # balance figure
        #############
        #setup figure
        total_balance_test,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.fill_between(interv,swK_sp/1e12,label='CentralDiff_sw10-3',
                          fc='grey')
        axis.plot(interv,(swKs1+swKs3)/1e12,label='SummedFlux_sw10-3')
        axis.plot(interv,(Ks1+Ks3)/1e12,label='SummedFlux_sp10-3')
        axis.plot(interv,(K_sp)/1e12,ls='--',label='CentralDiff_sp10-3',
                  c='grey')
        axis.plot(interv,(c35Ks1+c35Ks3)/1e12,label='SummedFlux_sp10-35')
        axis.plot(interv,(c35K_sp)/1e12,ls='--',label='CentralDiff_sp10-35',
                  c='grey')
        axis.plot(interv,(c375Ks1+c375Ks3)/1e12,label='SummedFlux_sp10-375')
        axis.plot(interv,(c375K_sp)/1e12,ls='--',label='CentralDiff_sp10-375',
                  c='grey')
        axis.plot(interv,(c3875Ks1+c3875Ks3)/1e12,label='SummedFlux_sp10-3875')
        axis.plot(interv,(c3875K_sp)/1e12,ls='--',label='CentralDiff_sp10-3875',
                  c='grey')
        axis.plot(interv,(c4Ks1+c4Ks3)/1e12,label='SummedFlux_sp10-4')
        axis.plot(interv,(c4K_sp)/1e12,ls='--',label='CentralDiff_sp10-4',
                  c='grey')
        axis.plot(interv,(c45Ks1+c45Ks3)/1e12,label='SummedFlux_sp10-45')
        axis.plot(interv,(c45K_sp)/1e12,ls='--',label='CentralDiff_sp10-45',
                  c='grey')
        axis.plot(interv,(c5Ks1+c5Ks3)/1e12,label='SummedFlux_sp10-5')
        axis.plot(interv,(c5K_sp)/1e12,ls='--',label='CentralDiff_sp10-5',
                  c='grey')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              #xlim=[moments['impact'],moments['peak2']],
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=False)
        #axis.axvspan(moments['impact'],moments['peak2'],alpha=0.2,fc='grey')
        #save
        total_balance_test.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        total_balance_test.tight_layout(pad=1)
        figurename = path+'/total_balance_test_'+event+'.png'
        total_balance_test.savefig(figurename)
        plt.close(total_balance_test)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        total_error_innerbound,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(interv,(Ks1+Ks3-K_sp)/1e12,label='Error_10-3')
        axis.plot(interv,(c35Ks1+c35Ks3-c35K_sp)/1e12,label='Error_10-35')
        axis.plot(interv,(c375Ks1+c375Ks3-c375K_sp)/1e12,label='Error_10-375')
        axis.plot(interv,(c3875Ks1+c3875Ks3-c3875K_sp)/1e12,label='Error_10-3875')
        axis.plot(interv,(c4Ks1+c4Ks3-c4K_sp)/1e12,label='Error_10-4')
        axis.plot(interv,(c45Ks1+c45Ks3-c45K_sp)/1e12,label='Error_10-45')
        axis.plot(interv,(c5Ks1+c5Ks3-c5K_sp)/1e12,label='Error_10-5')
        axis.plot(interv,(swKs1+swKs3-swK_sp)/1e12,label='Error_sw10-3')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              #xlim=[moments['impact'],moments['peak2']],
                              #ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=False)
        #axis.axvspan(moments['impact'],moments['peak2'],alpha=0.2,fc='grey')
        #save
        total_error_innerbound.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        total_error_innerbound.tight_layout(pad=1)
        figurename = path+'/total_error_innerbound_'+event+'.png'
        total_error_innerbound.savefig(figurename)
        plt.close(total_error_innerbound)
        print('\033[92m Created\033[00m',figurename)
        #############
        #setup figure
        total_acc_test,axis = plt.subplots(1,1,figsize=[16,8])
        #Plot
        axis.plot(interv,K_sp.cumsum()*60/1e15,label='CentralDiff_sp10-3')
        #axis.plot(interv,(Ks1+Ks3+0.7e12).cumsum()*60/1e15,label='SummedFlux_sp10-3')
        axis.plot(interv,(Ks1+Ks3).cumsum()*60/1e15,label='SummedFlux_sp10-3')
        axis.plot(interv,(pKs1+pKs3).cumsum()*60/1e15,label='SummedFlux_smooth')
        #axis.plot(interv,K_sp_fwd.cumsum()*60/1e15,label='ForwardDiff_sp10-3')
        axis.fill_between(interv,-1*(U-U[0])/1e15,
                          label='-1*Energy',fc='grey')
        #Decorations
        general_plot_settings(axis,do_xlabel=True,legend=True,
                              #xlim=[moments['impact'],moments['peak2']],
                              ylabel=r'Accumulated Power $\left[ PJ\right]$',
                              timedelta=False)
        #save
        total_acc_test.suptitle('t0='+str(moments['peak1']),
                                      ha='left',x=0.01,y=0.99)
        total_acc_test.tight_layout(pad=1)
        figurename = path+'/total_acc_test_'+event+'.png'
        total_acc_test.savefig(figurename)
        plt.close(total_acc_test)
        print('\033[92m Created\033[00m',figurename)
    if False:
    #for event in dataset.keys():
        #Figure setup
        demo_figure = plt.figure(figsize=(30,20),
                                 layout="constrained")
        spec = demo_figure.add_gridspec(4,6)
        A = demo_figure.add_subplot(spec[0,0:3])
        B = demo_figure.add_subplot(spec[1:3,0:3])
        C = demo_figure.add_subplot(spec[3:4,0:3])
        D = demo_figure.add_subplot(spec[0:2,3:6])
        E = demo_figure.add_subplot(spec[2:4,3:6])
        #Data setup
        lobes = dataset[event]['msdict'+phase]['lobes']
        closed = dataset[event]['msdict'+phase]['closed']
        mp = dataset[event]['mp'+phase]
        times=[float(n) for n in dataset[event]['time'+phase].to_numpy()]
        moments = locate_phase(dataset[event]['time'])
        sw = dataset[event]['obs']['swmf_sw'+ph]
        swtime = dataset[event]['swmf_sw_otime'+ph]
        swt = [float(n) for n in swtime.to_numpy()]#bad hack
        sim = dataset[event]['obs']['swmf_log'+ph]
        simtime = dataset[event]['swmf_log_otime'+ph]
        simt = [float(n) for n in simtime.to_numpy()]#bad hack
        index = dataset[event]['obs']['swmf_index'+ph]
        indextime = dataset[event]['swmf_index_otime'+ph]
        indext = [float(n) for n in indextime.to_numpy()]#bad hack
        obs = dataset[event]['obs']['omni'+ph]
        obstime = dataset[event]['omni_otime'+ph]
        ot = [float(n) for n in obstime.to_numpy()]#bad hack
        #K1,5 from mp
        Ks1 = mp['K_netK1 [W]']
        Ks5 = mp['K_netK5 [W]']
        #K2,3,4 from lobes
        Ks2al = lobes['K_netK2a [W]']
        Ks2bl = lobes['K_netK2b [W]']
        Ks3 = lobes['K_netK3 [W]']
        Ks4 = lobes['K_netK4 [W]']
        #K2,6,7 from closed
        Ks2ac = closed['K_netK2a [W]']
        Ks2bc = closed['K_netK2b [W]']
        Ks6 = closed['K_netK6 [W]']
        Ks7 = closed['K_netK7 [W]']

        #M1,5,total from mp
        M1 = mp['UtotM1 [W]']
        M5 = mp['UtotM5 [W]']
        M = mp['UtotM [W]']
        #M1a,1b,2b,il from lobes
        M1a = lobes['UtotM1a [W]']
        M1b = lobes['UtotM1b [W]']
        M2b = lobes['UtotM2b [W]']
        M2d = lobes['UtotM2d [W]']
        Mil = lobes['UtotMil [W]']
        #M5a,5b,2a,ic from closed
        M5a = closed['UtotM5a [W]']
        M5b = closed['UtotM5b [W]']
        M2a = closed['UtotM2a [W]']
        M2c = closed['UtotM2c [W]']
        Mic = closed['UtotMic [W]']

        # Central difference of partial volume integrals, total change
        # Total
        K_closed = -1*central_diff(closed['Utot [J]'])
        K_lobes = -1*central_diff(lobes['Utot [J]'])
        K_mp = -1*central_diff(mp['Utot [J]'])

        ################
        #Dst index
        A.plot(simt,sim['dst_sm'],label='Sim',c='tab:blue')
        A.plot(ot,obs['sym_h'],label='Obs',c='maroon')
        dtime_impact = (moments['impact']-
                        moments['peak2']).total_seconds()*1e9
        A.axvline(dtime_impact,ls='--',color='black')
        A.axvline(0,ls='--',color='black')
        general_plot_settings(A,ylabel=r'Sym-H$\left[nT\right]$',
                              do_xlabel=False, legend=True,
                              timedelta=True)
        ################
        #Power
        B.plot(times,(Ks3+Ks7)/1e12, label='InnerBound')
        B.plot(times,(M5+Ks5)/1e12, label='ClosedSheath')
        B.plot(times,(M1+Ks1)/1e12, label='OpenSheath')
        B.plot(times,(Ks4+Ks6)/1e12, label='TailCuttoff')
        B.plot(times,(Ks1+Ks5+Ks3+Ks4+Ks6+Ks7+M1+M5)/1e12,label='Summed')
        B.fill_between(times,K_mp/1e12,label='Total',fc='grey')
        general_plot_settings(B,do_xlabel=False,legend=True,
                              legend_loc='upper left',
                              ylim=[-10,10],
                              ylabel=r'Net Power $\left[ TW\right]$',
                              timedelta=True)
        ################
        #Energy
        stackt=[float(n) for n in dataset[event]['time'+ph].to_numpy()]
        plot_stack_contrib(C,stackt,dataset[event]['mp'+ph],
                           dataset[event]['msdict'+ph], legend=True,
                           value_key='Utot [J]',label='',ylim=[0,55],
                           factor=1e15,
                           ylabel=r'Energy $\left[ PJ\right]$',
                           legend_loc='upper left',
                           do_xlabel=True,
                           timedelta=True)
        C.axvspan(dtime_impact,0,alpha=0.2,facecolor='grey')
        C.axvline(0,ls='--',color='black')
        ################
        #Dissect
        quartercut = mpimg.imread('/Users/ngpdl/Desktop/3quarter_cut.png')
        D.imshow(quartercut)
        D.set_xticks([])
        D.set_yticks([])
        ################
        #Diagram
        diagram = mpimg.imread('/Users/ngpdl/Desktop/Diagram_extONLY.png')
        E.imshow(diagram)
        E.set_xticks([])
        E.set_yticks([])
        ################
        #save
        #demo_figure.suptitle('t0='+str(moments['peak1']),ha='left',
        #                     x=0.01,y=0.99)
        demo_figure.suptitle('Feb '+str(sw.index[0])+' - '+
                                    str(sw.index[-1]),
                             ha='left', x=0.01,y=0.99)
        demo_figure.tight_layout(pad=0.6)
        figname = unfiled+'/demo_'+event+'.png'
        demo_figure.savefig(figname)
        plt.close(demo_figure)
        print('\033[92m Created\033[00m',figname)
    '''
    3 panels:
        Dst,
        energy flow through magnetopause (in/out/net)
        energy content in lobes/closed plasma sheet
    Other notes
        Make it's own figure down below
        24 sections with:
            A A A D D D
            B B B D D D
            B B B E E E
            C C C E E E
        Time axis with 00 time format?
    '''
