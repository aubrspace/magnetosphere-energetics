#!/usr/bin/env python3
"""Analyze and plot data for the paper "Comprehensive Energy Analysis of a Simulated Magnetosphere"(working title)
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
#interpackage imports
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings,
                                                   plot_stack_distr,
                                                   plot_stack_contrib)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_hdf import (load_hdf_sort,
                                                 group_subzones,
                                                 get_subzone_contrib)

def locate_phase(dfdict,phasekey,**kwargs):
    """
    """
    phase = {}
    #Hand picked times
    start = dt.timedelta(minutes=kwargs.get('startshift',60))
    #Feb
    feb2014_impact = dt.datetime(2014,2,18,16,15)
    feb2014_endMain1 = dt.datetime(2014,2,19,4,0)
    feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    #Starlink
    starlink_impact = dt.datetime(2022,2,3,0,0)
    starlink_endMain1 = dt.datetime(2022,2,3,11,15)
    starlink_endMain2 = dt.datetime(2022,2,4,13,10)

    #Determine where dividers are based on specific event
    times = [df for df in dfdict.values()][0].index
    if feb2014_impact in times:
        impact = feb2014_impact
        peak1 = feb2014_endMain1
        peak2 = feb2014_endMain2
    elif starlink_impact in times:
        impact = starlink_impact
        peak1 = starlink_endMain1
        peak2 = starlink_endMain2

    #Set condition based on dividers and phase requested
    if 'qt' in phasekey:
        cond = (times>times[0]+start) & (times<impact)
    elif 'main' in phasekey:
        if '2' in phasekey:
            cond = (times>peak1) & (times<peak2)
        else:
            cond = (times>impact) & (times<peak2)
    elif 'rec' in phasekey:
        cond = times>peak2

    #Reload new dictionary filtered by the condition
    for key in dfdict.keys():
       df = dfdict[key]
       phase.update({key:df[cond]})

    return phase

if __name__ == "__main__":
    #Need input path, then create output dir's
    inPath = sys.argv[-1]
    outPath = os.path.join(inPath,'figures')
    outQT = os.path.join(outPath,'quietTime')
    outSSC = os.path.join(outPath,'shockImpact')
    outMN1 = os.path.join(outPath,'mainPhase1')
    outMN2 = os.path.join(outPath,'mainPhase2')
    outRec = os.path.join(outPath,'recovery')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,outQT,outSSC,outMN1,outMN2,outRec,unfiled]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='digital_presentation'))
    #Log files and observational indices
    febObs = read_indices(inPath, prefix='feb2014_', read_supermag=False)
    starObs = read_indices(inPath, prefix='starlink_', read_supermag=False)

    #HDF data, will be sorted and cleaned
    febSim = load_hdf_sort(inPath+'feb2014_results.h5')
    starSim = load_hdf_sort(inPath+'starlink_results.h5')

    ##Construct "grouped" set of subzones, then get %contrib for each
    starSim['mpdict'],starSim['msdict'] = get_subzone_contrib(
                                       starSim['mpdict'],starSim['msdict'])
    starSim['msdict'] = group_subzones(starSim['msdict'],mode='3zone')
    febSim['mpdict'],febSim['msdict'] = get_subzone_contrib(
                                       febSim['mpdict'],febSim['msdict'])
    febSim['msdict'] = group_subzones(febSim['msdict'],mode='3zone')

    ######################################################################
    ##Quiet time
    #parse storm phase
    feb_mp_qt = locate_phase(febSim['mpdict'],'qt')['ms_full']
    febtime = [dt.datetime(2000,1,1)+r for r in
               feb_mp_qt.index-feb_mp_qt.index[0]]
    star_mp_qt = locate_phase(starSim['mpdict'],'qt')['ms_full']
    startime = [dt.datetime(2000,1,1)+r for r in
                star_mp_qt.index-star_mp_qt.index[0]]
    #parse storm phase
    feb_mpdict_qt = locate_phase(febSim['mpdict'],'qt')
    feb_mp_qt = feb_mpdict_qt['ms_full']
    feb_msdict_qt = locate_phase(febSim['msdict'],'qt')
    febtime = [dt.datetime(2000,1,1)+r for r in
               feb_mp_qt.index-feb_mp_qt.index[0]]
    star_mpdict_qt = locate_phase(starSim['mpdict'],'qt')
    star_mp_qt = star_mpdict_qt['ms_full']
    star_msdict_qt = locate_phase(starSim['msdict'],'qt')
    startime = [dt.datetime(2000,1,1)+r for r in
                star_mp_qt.index-star_mp_qt.index[0]]

    #setup figure
    qt_energy, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                        figsize=[14,8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot
    ax1.plot(febtime, feb_mp_qt['Utot [J]'],label=feblabel)
    ax2.plot(startime, star_mp_qt['Utot [J]'],label=starlabel)
    general_plot_settings(ax1,ylabel=Elabel,do_xlabel=False,xlabel=Tlabel)
    general_plot_settings(ax2,ylabel=Elabel,do_xlabel=True,xlabel=Tlabel)

    #save
    qt_energy.tight_layout(pad=1)
    qt_energy.savefig(outQT+'/quiet_total_energy.png')
    plt.close(qt_energy)

    ##Stack plot Energy by region
    #setup figure
    qt_contr, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                       figsize=[14,8])

    feb_msdict_qt.pop('missing')
    star_msdict_qt.pop('missing')
    #plot
    plot_stack_contrib(ax1, febtime,feb_mp_qt, feb_msdict_qt,
                         value_key='Utot [J]', label=feblabel,
                         ylabel=Elabel,legend_loc='upper left')
    plot_stack_contrib(ax2, startime,star_mp_qt, star_msdict_qt,
                         value_key='Utot [J]', label=starlabel,
                         ylabel=Elabel,legend_loc='upper left')

    #save
    qt_contr.tight_layout(pad=1)
    qt_contr.savefig(outQT+'/quiet_contr_energy.png')
    plt.close(qt_contr)
    ######################################################################
    ##Main phase
    #parse storm phase
    feb_mpdict_mn1 = locate_phase(febSim['mpdict'],'main1')
    feb_mp_mn1 = feb_mpdict_mn1['ms_full']
    feb_msdict_mn1 = locate_phase(febSim['msdict'],'main1')
    febtime = [dt.datetime(2000,1,1)+r for r in
               feb_mp_mn1.index-feb_mp_mn1.index[0]]
    star_mpdict_mn1 = locate_phase(starSim['mpdict'],'main1')
    star_mp_mn1 = star_mpdict_mn1['ms_full']
    star_msdict_mn1 = locate_phase(starSim['msdict'],'main1')
    startime = [dt.datetime(2000,1,1)+r for r in
                star_mp_mn1.index-star_mp_mn1.index[0]]

    ##Line plot Energy
    #setup figure
    main1_energy, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                           figsize=[14,8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot
    ax1.plot(febtime, feb_mp_mn1['Utot [J]'],label=feblabel)
    ax2.plot(startime, star_mp_mn1['Utot [J]'],label=starlabel)
    general_plot_settings(ax1,ylabel=Elabel,do_xlabel=False,xlabel=Tlabel)
    general_plot_settings(ax2,ylabel=Elabel,do_xlabel=True,xlabel=Tlabel)

    #save
    main1_energy.tight_layout(pad=1)
    main1_energy.savefig(outMN1+'/main1_total_energy.png')
    plt.close(main1_energy)

    ##Stack plot Energy by type (hydro,magnetic) for each region
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Tlabel = r'Time $\left[ hr\right]$'
    for sz in ['ms_full','lobes','closedRegion','rc']:
        #setup figures
        main1_distr, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                              figsize=[14,8])

        #plot
        plot_stack_distr(ax1, febtime,feb_mp_mn1, feb_msdict_mn1,
                         value_set='Energy2', doBios=False, label=feblabel,
                         ylabel=Elabel,legend_loc='upper left',subzone=sz)
        plot_stack_distr(ax2, startime,star_mp_mn1, star_msdict_mn1,
                         value_set='Energy2', doBios=False, label=starlabel,
                         ylabel=Elabel,legend_loc='upper left',subzone=sz)

        #save
        main1_distr.tight_layout(pad=1)
        main1_distr.savefig(outMN1+'/main1_distr_energy'+sz+'.png')
        plt.close(main1_distr)

    ##Stack plot Energy by region
    #setup figure
    main1_contr, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                          figsize=[14,8])

    feb_msdict_mn1.pop('missing')
    star_msdict_mn1.pop('missing')
    #plot
    plot_stack_contrib(ax1, febtime,feb_mp_mn1, feb_msdict_mn1,
                         value_key='Utot [J]', label=feblabel,
                         ylabel=Elabel,legend_loc='upper left')
    plot_stack_contrib(ax2, startime,star_mp_mn1, star_msdict_mn1,
                         value_key='Utot [J]', label=starlabel,
                         ylabel=Elabel,legend_loc='upper left')

    #save
    main1_contr.tight_layout(pad=1)
    main1_contr.savefig(outMN1+'/main1_contr_energy.png')
    plt.close(main1_contr)

    ######################################################################
