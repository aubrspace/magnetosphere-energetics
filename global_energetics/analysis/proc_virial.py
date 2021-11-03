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
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage imports
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_temporal import read_energetics
from global_energetics.analysis.analyze_energetics import (chopends_time,
                                                           plot_dst)

def plot_stack_virial_contrib(ax, times, mp, msdict, ylabel, *,
                              do_percent=False, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
    """
    #Relative contribution mode
    if not do_percent:
        ax.plot(times, mp['Virial Volume Total [nT]'],
                color='black')
        value_key = 'Virial Volume Total [nT]'
    else:
        value_key = 'Virial Contribution [%]'
    stack_value = [m for m in msdict.values()][0][value_key]
    for ms in enumerate([mslist for mslist in msdict.items()][0:-1]):
        mslabel = ms[1][0]
        if ms[0]==0:
            ax.fill_between(times, stack_value,
                            label=mslabel)
        else:
            ax.fill_between(times, stack_value, old_value,
                               label=mslabel)
        old_value = stack_value
        stack_value = stack_value+ [m for m in msdict.values()][ms[0]+1][
                                                                 value_key]
    if do_percent:
        ax.axhline(100, color='black')
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(r'Time $\left[ UTC\right]$')
    ax.set_ylabel(ylabel)
    ax.legend(loc=kwargs.get('legend_loc',None))

def plot_raw_virial_contributions(ax, times, mp, msdict, ylabel, *,
                                  do_percent=False, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
    """
    #Relative contribution mode
    if not do_percent:
        ax.fill_between(times, mp['Virial Volume Total [nT]'],
                           color='lightgrey')
        value_key = 'Virial Volume Total [nT]'
    else:
        value_key = 'Virial Contribution [%]'
    for ms in msdict.items():
        ax.plot(times, ms[1][value_key],
                   label=ms[0])
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(r'Time $\left[ UTC\right]$')
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
    for mp in mpdict.values():
        mp['Virial Volume Total [J]'] = (mp['Virial Ub [J]']+
                                          mp['Virial 2x Uk [J]'])
        mp['Virial Volume Total [nT]'] = (mp['Virial Ub [nT]']+
                                          mp['Virial 2x Uk [nT]'])
    for ms in msdict.values():
        ms['Virial Volume Total [J]'] = (ms['Virial Ub [J]']+
                                          ms['Virial 2x Uk [J]'])
        ms['Virial Volume Total [nT]'] = (ms['Virial Ub [nT]']+
                                          ms['Virial 2x Uk [nT]'])
    #Quantify amount missing from sum of all subzones
    missing_volume = pd.DataFrame()
    for key in [m for m in msdict.values()][0].keys():
        fullvolume = [m for m in mpdict.values()][0][key]
        added = [m for m in msdict.values()][0][key]
        for sub in [m for m in msdict.values()][1::]:
            added = added+sub[key]
        missing_volume[key] = fullvolume-added
    msdict.update({'missed':missing_volume})
    #Identify percentage contribution from each piece
    for ms in msdict.values():
        ms['Virial Contribution [%]'] = (ms['Virial Volume Total [J]']/
            [m for m in mpdict.values()][0]['Virial Volume Total [J]']*100)
    return mpdict, msdict

if __name__ == "__main__":
    datapath = sys.argv[-1]
    figureout = datapath+'figures/'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    [swmf_index, swmf_log, swmf_sw,_,__]= read_indices(datapath,
                                                       read_supermag=False,
                                                       read_omni=False)
    cuttoffstart = dt.datetime(2014,2,18,6,0)
    cuttoffend = dt.datetime(2014,2,20,0,0)
    simdata = [swmf_index, swmf_log, swmf_sw]
    [swmf_index,swmf_log,swmf_sw] = chopends_time(simdata, cuttoffstart,
                                      cuttoffend, 'Time [UTC]', shift=True)
    store = pd.HDFStore(datapath+'energetics.h5')
    times = store['Times']
    mpdict = {'ms_full':store['/mp_iso_betastar']}
    mp = [m for m in mpdict.values()][0]
    msdict = {'nlobe':store['/ms_nlobe'],
              'slobe':store['/ms_slobe'],
              'rc':store['/ms_rc'],
              'ps':store['/ms_ps'],
              'qDp':store['/ms_qDp']}
    mpdict, msdict = get_interzone_stats(mpdict, msdict)
    #Construct "grouped" set of subzones
    msgrouped1 = {'lobes':store['/ms_nlobe']+store['/ms_slobe'],
                  'rc':store['/ms_rc'],
                  'closedRegion':store['/ms_ps']+store['/ms_qDp']}
    _, msgrouped1 = get_interzone_stats(mpdict, msgrouped1)
    mslobes = {'nlobe':store['/ms_nlobe'],
               'slobe':store['/ms_slobe'],
               'remaining':store['/ms_rc']+store['/ms_ps']+store['/ms_qDp']}
    _, mslobes = get_interzone_stats(mpdict, mslobes)
    store.close()
    #Begin plots
    #Fist grouping, 'all pieces'
    ######################################################################
    #Line plots
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14,8])
    y1label = r'Virial Dst $\left[ nT\right]$'
    y2label = r'$\%$ Contribution'
    plot_raw_virial_contributions(ax[0], times, mp, msdict, y1label)
    plot_raw_virial_contributions(ax[1], times, mp, msdict, y2label,
                                  do_percent=True, do_xlabel=True)
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'viriallineplot_raw_allpieces.png')
    plt.close(fig)
    #Stack plot
    fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14,8])
    y1label = r'Virial Dst $\left[ nT\right]$'
    y2label = r'$\%$ Contribution'
    plot_stack_virial_contrib(ax[0], times, mp, msdict, y1label)
    plot_stack_virial_contrib(ax[1], times, mp, msdict, y2label,
                              do_percent=True, do_xlabel=True)
    fig2.tight_layout(pad=1)
    fig2.savefig(figureout+'viriallineplot_stack_allpieces.png')
    plt.close(fig2)

    #Second grouping, 'lobes, rc, closed'
    ######################################################################
    #Line plots
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14,8])
    y1label = r'Virial Dst $\left[ nT\right]$'
    y2label = r'$\%$ Contribution'
    plot_raw_virial_contributions(ax[0],times,mp,msgrouped1,y1label)
    plot_raw_virial_contributions(ax[1],times,mp,msgrouped1,y2label,
                                  do_percent=True, do_xlabel=True)
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'viriallineplot_raw_3piece.png')
    plt.close(fig)
    #Stack plot
    fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14,8])
    y1label = r'Virial Dst $\left[ nT\right]$'
    y2label = r'$\%$ Contribution'
    plot_stack_virial_contrib(ax[0], times, mp, msgrouped1, y1label)
    plot_stack_virial_contrib(ax[1], times, mp, msgrouped1, y2label,
                              do_percent=True, do_xlabel=True)
    fig2.tight_layout(pad=1)
    fig2.savefig(figureout+'viriallineplot_stack_3piece.png')
    plt.close(fig2)

    #Third grouping, 'lobe v lobe'
    mslobes.pop('remaining')
    mslobes.pop('missed')
    ######################################################################
    #Line plots
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14,8])
    y1label = r'Virial Dst $\left[ nT\right]$'
    y2label = r'$\%$ Contribution'
    plot_raw_virial_contributions(ax[0],times,mp,mslobes,y1label)
    plot_raw_virial_contributions(ax[1],times,mp,mslobes,y2label,
                                  do_percent=True, do_xlabel=True)
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'viriallineplot_raw_Lobes.png')
    plt.close(fig)
