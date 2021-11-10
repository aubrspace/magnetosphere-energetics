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
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import swmfpy
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage imports
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_temporal import read_energetics
from global_energetics.analysis.analyze_energetics import (chopends_time,
                                                           plot_dst)

def plot_stack_contrib(ax, times, mp, msdict, ylabel, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
    """
    #Figure out value_key
    value_key = kwargs.get('value_key','Virial Volume Total [nT]')
    #Optional layers depending on value_key
    starting_value = 0
    if not '%' in value_key:
        ax.plot(times, mp[value_key], color='black')
        if (not 'BioS' in value_key) and ('Volume Total' in value_key):
            #include the contribution from surface terms
            ax.fill_between(times, mp['Virial Surface Total [nT]'],
                            0, color='grey', label='Surface')
            starting_value = mp['Virial Surface Total [nT]']
            ax.axhline(0,color='black')
    stack_value = [m for m in msdict.values()][0][value_key]
    #Plot line plots
    for ms in enumerate([mslist for mslist in msdict.items()][0:-1]):
        mslabel = ms[1][0]
        if ms[0]==0:
            ax.fill_between(times, stack_value, starting_value,
                            label=mslabel)
        else:
            ax.fill_between(times, stack_value, old_value,
                               label=mslabel)
        old_value = stack_value
        stack_value = stack_value+ [m for m in msdict.values()][ms[0]+1][
                                                                 value_key]
    #Optional plot settings
    if ('BioS' in value_key) and ('%' in value_key):
        ax.set_ylim([-100,100])
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(r'Time $\left[ UTC\right]$')
    #General plot settings
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    ax.tick_params(which='major', length=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(ylabel)
    ax.legend(loc=kwargs.get('legend_loc',None))

def plot_contributions(ax, times, mp, msdict, ylabel, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
            value_key(str)- 'Virial Volume' is default
    """
    #Figure out value_key
    value_key = kwargs.get('value_key','Virial Volume Total [nT]')
    #Optional layers depending on value_key
    if not '%' in value_key:
        if 'BioS' in value_key:
            ax.fill_between(times, mp['BioS full [nT]'], color='olive')
            ax.plot(times, mp['Virial_Dst [nT]'], color='black', ls='--',
                    label='Virial')
            try:
                ax.plot(omni['Time [UTC]'],omni['sym_h'],color='cyan',
                        ls='--', label='OMNI obs')
            except NameError:
                print('omni not loaded! No obs comparisons!')
        ax.fill_between(times, mp[value_key], color='lightgrey')
    #Plot line plots
    for ms in msdict.items():
        if value_key in ms[1].keys():
            ax.plot(times, ms[1][value_key], label=ms[0])
    #Optional plot settings
    if ('BioS' in value_key) and ('%' in value_key):
        ax.set_ylim([-100,100])
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(r'Time $\left[ UTC\right]$')
    #General plot settings
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    ax.tick_params(which='major', length=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(ylabel)
    ax.legend(loc=kwargs.get('legend_loc',None))

def get_interzone_stats(mpdict, msdict, **kwargs):
    """Function finds percent contributions and missing amounts
    Inputs
        mpdict(DataFrame)-
        msdict(Dict of DataFrames)-
    Returns
        [MODIFIED] mpdict(Dict of DataFrames)
        [MODIFIED] msdict(Dict of DataFrames)
    """
    #Remove time column and rename biot savart columns
    for m in mpdict.values():
        m.drop(columns=['Time [UTC]'],inplace=True, errors='ignore')
        m.rename(columns={'BioS mp_iso_betastar':'BioS [nT]'},inplace=True)
        m.rename(columns={'BioS full':'BioS full [nT]'},inplace=True)
    for m in msdict.values():
        m.drop(columns=['Time [UTC]'], inplace=True, errors='ignore')
        m.drop(columns=['BioS mp_iso_betastar',
                        'BioS full', 'X_subsolar [Re]'],inplace=True)
        for key in m.keys():
            if all(m[key].isna()):
                m.drop(columns=[key], inplace=True)
            if 'BioS' in key:
                m.rename(columns={key:'BioS [nT]'},inplace=True)
    #Quantify amount missing from sum of all subzones
    missing_volume = pd.DataFrame()
    for key in [m for m in msdict.values()][0].keys():
        if key in [m for m in mpdict.values()][0].keys():
            fullvolume = [m for m in mpdict.values()][0][key]
            added = [m for m in msdict.values()][0][key]
            for sub in [m for m in msdict.values()][1::]:
                added = added+sub[key]
            missing_volume[key] = fullvolume-added
    msdict.update({'missed':missing_volume})
    #Identify percentage contribution from each piece
    for ms in msdict.values():
        for key in ms.keys():
            ms[key.split('[')[0]+'[%]'] = (100*ms[key]/
                                      [m for m in mpdict.values()][0][key])
    return mpdict, msdict

if __name__ == "__main__":
    datapath = sys.argv[-1]
    figureout = datapath+'figures/'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    [swmf_index, swmf_log, swmf_sw,_,omni]= read_indices(datapath,
                                                       read_supermag=False)
    cuttoffstart = dt.datetime(2014,2,18,6,0)
    cuttoffend = dt.datetime(2014,2,20,0,0)
    simdata = [swmf_index, swmf_log, swmf_sw]
    [swmf_index,swmf_log,swmf_sw] = chopends_time(simdata, cuttoffstart,
                                      cuttoffend, 'Time [UTC]', shift=True)
    store = pd.HDFStore(datapath+'energetics.h5')
    times = store[store.keys()[0]]['Time [UTC]']
    #Clean up and gather statistics
    mpdict = {'ms_full':store['/mp_iso_betastar']}
    mp = [m for m in mpdict.values()][0]
    msdict = {'nlobe':store['/ms_nlobe'].drop(columns=['Time [UTC]']),
              'slobe':store['/ms_slobe'].drop(columns=['Time [UTC]']),
              'rc':store['/ms_rc'].drop(columns=['Time [UTC]']),
              'ps':store['/ms_ps'].drop(columns=['Time [UTC]']),
              'qDp':store['/ms_qDp'].drop(columns=['Time [UTC]'])}
    mpdict, msdict = get_interzone_stats(mpdict, msdict)
    store.close()
    #Construct "grouped" set of subzones
    '''
    msdict_3zone = {'rc':msdict['rc'],
                    'closedRegion':msdict['ps']+msdict['qDp'],
                    'lobes':msdict['nlobe']+msdict['slobe'],
                    'missing':msdict['missed']}
    '''
    msdict_3zone = {
                    'lobes':msdict['nlobe']+msdict['slobe'],
                    'closedRegion':msdict['ps']+msdict['qDp'],
                    'rc':msdict['rc'],
                    'missing':msdict['missed']}
    msdict_lobes ={'nlobe':msdict['nlobe'],
                   'slobe':msdict['slobe'],
                   'remaining':msdict['rc']+msdict['ps']+msdict['qDp'],
                   'missing':msdict['missed']}
    #Calculate 2nd order difference for mass omission
    ######################################################################
    #Define terms
    f_n = mp['Mr^2 [kgm^2]']
    f_n2 = f_n.copy(); f_n1 = f_n.copy() #Forward
    f_n_1 = f_n.copy(); f_n_2 = f_n.copy() #Backward
    f_n1.index=f_n1.index-1; f_n2.index=f_n2.index-2
    f_n_1.index=f_n1.index+1; f_n_2.index=f_n2.index+2
    f_n1 = f_n1.drop(index=[-1])
    f_n2 = f_n2.drop(index=[-1,-2])
    #Types of differences
    central_diff = ((f_n1-2*f_n+f_n_1)/60**2) / (-8e13)
    forward_diff = ((f_n-2*f_n1+f_n2)/60**2) / (-8e13)
    central4_diff=((-f_n2+16*f_n1-30*f_n+16*f_n_1-f_n_2)/(12*60**2))/(-8e13)
    '''
    mp['d2dt2 Mr^2 [J]'] = (f_f-2*f_c+f_b)/60**2
    mp['d2dt2 Mr^2 [nT]'] = mp['d2dt2 Mr^2 [J]']/8e13
    '''
    #Begin plots
    #Fist type: line and stacks of various groupings of terms
    ######################################################################
    #Label setup
    y1labels = [r'Virial Dst $\left[ nT\right]$',
                r'Plasma Virial Dst $\left[ nT\right]$',
                r'Magnetic Virial Dst $\left[ nT\right]$',
                r'Biot Savart Dst $\left[ nT\right]$']
    y2label =   r'$\%$ Contribution'
    value_keys=['Virial Volume Total ','Virial 2x Uk ','Virial Ub ','BioS ']
    #Line plots- total virial, plasma, disturbance, and Biot savart
    for combo in {'allPiece':msdict,'3zone':msdict_3zone,
                  'lobes':msdict_lobes}.items():
        fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig2,ax2 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig3,ax3 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig4,ax4 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        for ax in enumerate([ax1, ax2, ax3, ax4]):
            plot_contributions(ax[1][0], times,mp,combo[1],y1labels[ax[0]],
                               value_key=value_keys[ax[0]]+'[nT]')
            plot_contributions(ax[1][1], times, mp, combo[1], y2label,
                          value_key=value_keys[ax[0]]+'[%]',do_xlabel=True)
        for fig in{'total':fig1,'plasma':fig2,'mag_perturb':fig3,
                   'BioS':fig4}.items():
            fig[1].tight_layout(pad=1)
            fig[1].savefig(figureout+'virial_line_'+fig[0]+combo[0]+'.png')
            plt.close(fig[1])
    #Stack plots- total virial, plasma, disturbance, and Biot savart
    for combo in {'allPiece':msdict,'3zone':msdict_3zone,
                  'lobes':msdict_lobes}.items():
        fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig2,ax2 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig3,ax3 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig4,ax4 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        for ax in enumerate([ax1, ax2, ax3, ax4]):
            plot_stack_contrib(ax[1][0], times,mp,combo[1],y1labels[ax[0]],
                          value_key=value_keys[ax[0]]+'[nT]')
            plot_stack_contrib(ax[1][1], times, mp, combo[1], y2label,
                          value_key=value_keys[ax[0]]+'[%]',do_xlabel=True)
        for fig in{'total':fig1,'plasma':fig2,'mag_perturb':fig3,
                   'BioS':fig4}.items():
            fig[1].tight_layout(pad=1)
            fig[1].savefig(figureout+'virial_stack_'+fig[0]+combo[0]+'.png')
            plt.close(fig[1])
    #Second type: specific 
    ######################################################################
