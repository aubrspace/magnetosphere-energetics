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
from global_energetics.analysis.plot_tools import (pyplotsetup,safelabel,
                                                   general_plot_settings,
                                                   plot_stack_distr,
                                                   plot_stack_contrib)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_hdf import (load_hdf_sort,
                                                 group_subzones,
                                                 get_subzone_contrib)
from global_energetics.analysis.analyze_energetics import plot_power

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

def locate_phase(indata,phasekey,**kwargs):
    """Function returns subset of given data based on a phasekey
    Inputs
        indata (DataFrame, Series, or dict{df/s})- data to be subset
        phasekey (str)- 'main', 'recovery', etc.
    Returns
        phase (same datatype given)
    """
    assert (type(indata)==dict or type(indata)==pd.core.series.Series,
            'Data type only excepts dict, DataFrame, or Series')
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
    #starlink_endMain2 = dt.datetime(2022,2,4,13,10)
    starlink_endMain2 = dt.datetime(2022,2,4,22,0)
    #Starlink
    ccmc_impact = dt.datetime(2019,5,13,0,0)
    ccmc_endMain1 = dt.datetime(2019,5,14,7,45)
    ccmc_endMain2 = dt.datetime(2019,5,14,7,45)

    #Get time information based on given data type
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        times = indata.index
    elif type(indata) == dict:
        times = [df for df in indata.values()][0].index

    #Determine where dividers are based on specific event
    if abs(times-feb2014_impact).min() < dt.timedelta(minutes=15):
        impact = feb2014_impact
        peak1 = feb2014_endMain1
        peak2 = feb2014_endMain2
    elif abs(times-starlink_impact).min() < dt.timedelta(minutes=15):
        impact = starlink_impact
        peak1 = starlink_endMain1
        peak2 = starlink_endMain2
    elif abs(times-ccmc_impact).min() < dt.timedelta(minutes=15):
        impact = ccmc_impact
        peak1 = ccmc_endMain1
        peak2 = ccmc_endMain2
    else:
        impact = times[0]
        peak1 = times[round(len(times)/2)]
        peak2 = times[round(len(times)/2)]
        #peak2 = times[-1]

    #Set condition based on dividers and phase requested
    if 'qt' in phasekey:
        cond = (times>times[0]+start) & (times<impact)
    elif 'main' in phasekey:
        if '2' in phasekey:
            cond = (times>peak1) & (times<peak2)
        else:
            cond = (times>impact) & (times<peak1)
    elif 'rec' in phasekey:
        cond = times>peak2

    #Reload data filtered by the condition
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        return indata[cond]
    elif type(indata) == dict:
        for key in indata.keys():
            df = indata[key]
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
    plt.rcParams.update(pyplotsetup(mode='print_presentation'))
    #Log files and observational indices
    febObs = read_indices(inPath, prefix='feb2014_', read_supermag=False,
                          tshift=45)
    starObs = read_indices(inPath, prefix='starlink_', read_supermag=False)

    #HDF data, will be sorted and cleaned
    febSim = load_hdf_sort(inPath+'feb2014_results.h5')
    starSim = load_hdf_sort(inPath+'starlink_results.h5')

    ##Construct "grouped" set of subzones, then get %contrib for each
    starSim['mpdict'],starSim['msdict'] = get_subzone_contrib(
                                       starSim['mpdict'],starSim['msdict'])
    febSim['mpdict'],febSim['msdict'] = get_subzone_contrib(
                                       febSim['mpdict'],febSim['msdict'])

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
    ax1.plot(febtime, feb_mp_qt['Utot2 [J]'],label=feblabel)
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

    feb_msdict_qt.pop('missed')
    star_msdict_qt.pop('missed')
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
    #TODO: summary bar plot with space for contour of time vs L vs energy
    ##Bar plot with energy distribution summary
    #setup figure
    qt_bar, [ax_top,ax_bot] = plt.subplots(2,1,sharey=False,sharex=False,
                                           figsize=[8,8])

    #plot bars for each region
    ax_top.bar(feb_msdict_qt.keys(), [feb_msdict_qt[k]['Utot2 [J]'].mean()
                                        for k in feb_msdict_qt.keys()])
    general_plot_settings(ax_top,ylabel=Elabel,do_xlabel=False,
                          iscontour=True)
    #save
    qt_bar.tight_layout(pad=1)
    qt_bar.savefig(outQT+'/quiet_bar_energy.png')
    plt.close(qt_bar)
    print(feb_msdict_qt['closed']['Utot [J]']-
          feb_msdict_qt['closed']['Utot2 [J]'])
    from IPython import embed; embed()
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

    '''
    #FAKE (all times)
    feb_mpdict_mn1 = febSim['mpdict']
    feb_mp_mn1 = feb_mpdict_mn1['ms_full']
    feb_msdict_mn1 = febSim['msdict']
    febtime = [dt.datetime(2000,1,1)+r for r in
               feb_mp_mn1.index-feb_mp_mn1.index[0]]
    star_mpdict_mn1 = starSim['mpdict']
    star_mp_mn1 = star_mpdict_mn1['ms_full']
    star_msdict_mn1 = starSim['msdict']
    startime = [dt.datetime(2000,1,1)+r for r in
                star_mp_mn1.index-star_mp_mn1.index[0]]
    '''

    ##Line plot Energy
    #setup figure
    main1_energy, [ax1,ax2] = plt.subplots(2,1,sharey=False,sharex=True,
                                           figsize=[14,8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot
    ax1.plot(febtime, feb_mp_mn1['Utot2 [J]'],label=feblabel)
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
    for sz in ['ms_full','lobes','closed','rc']:
        #setup figures
        main1_distr, [ax1,ax2] = plt.subplots(2,1,sharey=False,sharex=True,
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
    main1_contr, [ax1,ax2] = plt.subplots(2,1,sharey=False,sharex=True,
                                          figsize=[14,8])

    feb_msdict_mn1.pop('missed')
    star_msdict_mn1.pop('missed')
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


    ##Interface generic multiplot
    for mpdict,msdict,tag in [(feb_mpdict_mn1, feb_msdict_mn1, 'feb'),
                              (star_mpdict_mn1, star_msdict_mn1, 'star')]:
        for szkey in ['ms_full','lobes','closed','rc']:
            if 'full' in szkey: sz = feb_mpdict_mn1[szkey]
            else: sz = star_msdict_mn1[szkey]
            interfaces = get_interfaces(sz)
            #setup figure
            interf1, axes = plt.subplots(len(interfaces),1,sharey=False,
                                          sharex=True,
                                          figsize=[14,4*len(interfaces)])
            for ax in enumerate(axes):
                Plabel = ('Power '+safelabel(interfaces[ax[0]])+
                                                     r'$\left[ TW\right]$')
                terms = ['K_'+direc+interfaces[ax[0]]+' [W]'
                    for direc in ['injection','escape','net']]
                termdict = {}
                for d,t in zip(['inj','esc','net'],terms):termdict.update({d:t})
                plot_power(ax[1], {szkey:sz}, sz.index, **termdict,
                    ylabel=Plabel,do_xlabel=False,legend_loc='upper left')
            general_plot_settings(ax[-1],ylabel=Plabel,do_xlabel=True,
                                  xlabel=Tlabel)
            interf1.tight_layout(pad=1)
            interf1.savefig(outMN1+'/interf1'+szkey+'.png')

    ######################################################################
