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
    #feb2014_endMain1 = dt.datetime(2014,2,19,4,0)
    #feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    feb2014_endMain1 = dt.datetime(2014,2,19,9,45)
    feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    #Starlink
    starlink_impact = dt.datetime(2022,2,3,0,0)
    starlink_endMain1 = dt.datetime(2022,2,3,11,15)
    starlink_endMain1 = dt.datetime(2022,2,4,13,10)
    starlink_endMain2 = dt.datetime(2022,2,4,22,0)
    #May2019
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
        cond = times>peak1#NOTE

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

    #NOTE hotfix for closed region tail_closed
    for t in[t for t in febSim['msdict']['closed'].keys()
                                                    if 'Tail_close'in t]:
        febSim['msdict']['closed'][t] = febSim['mpdict']['ms_full'][t]
        starSim['msdict']['closed'][t] = starSim['mpdict']['ms_full'][t]

    ##Construct "grouped" set of subzones, then get %contrib for each
    starSim['mpdict'],starSim['msdict'] = get_subzone_contrib(
                                       starSim['mpdict'],starSim['msdict'])
    febSim['mpdict'],febSim['msdict'] = get_subzone_contrib(
                                       febSim['mpdict'],febSim['msdict'])
    febSim['msdict'] = {'rc':febSim['msdict']['rc'],
                        'closed':febSim['msdict']['closed'],
                        'lobes':febSim['msdict']['lobes'],
                       'missed':febSim['msdict']['missed']}
    starSim['msdict'] = {'rc':starSim['msdict']['rc'],
                   'closed':starSim['msdict']['closed'],
                   'lobes':starSim['msdict']['lobes'],
                   'missed':starSim['msdict']['missed']}
    ##Parse storm phases
    #quiet time
    feb_mpdict_qt = locate_phase(febSim['mpdict'],'qt')
    feb_mp_qt = feb_mpdict_qt['ms_full']
    feb_msdict_qt = locate_phase(febSim['msdict'],'qt')
    febtime_qt = [dt.datetime(2000,1,1)+r for r in
               feb_mp_qt.index-feb_mp_qt.index[0]]
    star_mpdict_qt = locate_phase(starSim['mpdict'],'qt')
    star_mp_qt = star_mpdict_qt['ms_full']
    star_msdict_qt = locate_phase(starSim['msdict'],'qt')
    startime_qt = [dt.datetime(2000,1,1)+r for r in
                star_mp_qt.index-star_mp_qt.index[0]]
    #Main phase
    feb_mpdict_mn1 = locate_phase(febSim['mpdict'],'main1')
    feb_mp_mn1 = feb_mpdict_mn1['ms_full']
    feb_msdict_mn1 = locate_phase(febSim['msdict'],'main1')
    febtime_mn1 = [dt.datetime(2000,1,1)+r for r in
               feb_mp_mn1.index-feb_mp_mn1.index[0]]
    feb_obs_mn1 = locate_phase(febObs['swmf_sw'],'main1')
    ofebtime_mn1 = [dt.datetime(2000,1,1)+r for r in
                feb_obs_mn1.index-feb_obs_mn1.index[0]]
    star_mpdict_mn1 = locate_phase(starSim['mpdict'],'main1')
    star_mp_mn1 = star_mpdict_mn1['ms_full']
    star_msdict_mn1 = locate_phase(starSim['msdict'],'main1')
    startime_mn1 = [dt.datetime(2000,1,1)+r for r in
                star_mp_mn1.index-star_mp_mn1.index[0]]
    star_obs_mn1 = locate_phase(starObs['swmf_sw'],'main1')
    ostartime_mn1 = [dt.datetime(2000,1,1)+r for r in
                star_obs_mn1.index-star_obs_mn1.index[0]]
    #Recovery
    feb_mpdict_rec = locate_phase(febSim['mpdict'],'rec')
    feb_mp_rec = feb_mpdict_rec['ms_full']
    feb_msdict_rec = locate_phase(febSim['msdict'],'rec')
    febtime_rec = [dt.datetime(2000,1,1)+r for r in
               feb_mp_rec.index-feb_mp_rec.index[0]]
    feb_obs_rec = locate_phase(febObs['swmf_sw'],'rec')
    ofebtime_rec = [dt.datetime(2000,1,1)+r for r in
                feb_obs_rec.index-feb_obs_rec.index[0]]
    star_mpdict_rec = locate_phase(starSim['mpdict'],'rec')
    star_mp_rec = star_mpdict_rec['ms_full']
    star_msdict_rec = locate_phase(starSim['msdict'],'rec')
    startime_rec = [dt.datetime(2000,1,1)+r for r in
                star_mp_rec.index-star_mp_rec.index[0]]
    star_obs_rec = locate_phase(starObs['swmf_sw'],'rec')
    ostartime_rec = [dt.datetime(2000,1,1)+r for r in
                star_obs_rec.index-star_obs_rec.index[0]]


    ######################################################################
    ##Quiet time
    #setup figure
    qt_energy, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                        figsize=[14,8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Plabel = r'Energy $\left[ PJ\right]$'
    Wlabel = r'Power $\left[ TW\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot
    ax1.plot(febtime_qt, feb_mp_qt['Utot2 [J]'],label=feblabel)
    ax2.plot(startime_qt, star_mp_qt['Utot2 [J]'],label=starlabel)
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
    plot_stack_contrib(ax1, febtime_qt,feb_mp_qt, feb_msdict_qt,
                         value_key='Utot2 [J]', label=feblabel,ylim=[0,15],
                         ylabel=Elabel,legend_loc='upper right')
    plot_stack_contrib(ax2, startime_qt,star_mp_qt, star_msdict_qt,
                         value_key='Utot2 [J]', label=starlabel,ylim=[0,15],
                         do_xlabel=True,xlabel=r'Time $\left[Hr\right]$',
                         ylabel=Elabel,legend=False)

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
    bar_labels = feb_msdict_qt.keys()
    bar_ticks = np.arange(len(bar_labels))
    ax_top.bar(bar_ticks-0.2, [feb_msdict_qt[k]['Utot2 [J]'].mean()/1e15
                                for k in ['rc','lobes','closed']],
                                                     0.4,label='MultiCME',
                                ec='black',fc='silver')
    ax_top.bar(bar_ticks+0.2, [star_msdict_qt[k]['Utot2 [J]'].mean()/1e15
                                for k in ['rc','lobes','closed']],
                                                     0.4,label='Starlink',
                                ec='black',fc='silver',hatch='*')
    ax_top.set_xticklabels(['']+list(star_msdict_qt.keys())+[''])
    general_plot_settings(ax_top,ylabel=Plabel,do_xlabel=False,
                          iscontour=True,legend_loc='upper left')
    #plot bars for each region
    interface_list = ['Dayside_reg','Tail_close','L7','PSB','MidLat',
                      'Flank','Tail_lobe','Poles','LowLat']
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    for i, (dic,tag) in enumerate([(feb_msdict_qt,'feb'),
                                  (star_msdict_qt,'star')]):
        clo_inj,lob_inj,rc_inj,bar_labels = [],[],[],[]
        clo_esc,lob_esc,rc_esc  = [],[],[]
        clo_net,lob_net,rc_net  = [],[],[]
        if i==1: hatch='*'
        else: hatch=None
        for interf in interface_list:
            bar_labels += [safelabel(interf.split('_reg')[0])]
            #bar_labels += ['']
            if interf in closed:
                clo_inj+=[dic['closed']
                            ['K_injection'+interf+' [W]'].mean()/1e12]
                clo_esc+=[dic['closed']
                            ['K_escape'+interf+' [W]'].mean()/1e12]
                clo_net += [dic['closed']
                            ['K_net'+interf+' [W]'].mean()/1e12]
            elif interf in lobes:
                lob_inj+=[dic['lobes']
                            ['K_injection'+interf+' [W]'].mean()/1e12]
                lob_esc+=[dic['lobes']
                            ['K_escape'+interf+' [W]'].mean()/1e12]
                lob_net += [dic['lobes']
                            ['K_net'+interf+' [W]'].mean()/1e12]
            elif interf in ring_c:
                rc_inj += [dic['rc']
                            ['K_injection'+interf+' [W]'].mean()/1e12]
                rc_esc += [dic['rc']
                            ['K_escape'+interf+' [W]'].mean()/1e12]
                rc_net += [dic['rc']
                            ['K_net'+interf+' [W]'].mean()/1e12]
        bar_ticks = np.arange(len(interface_list))+(i-0.5)*(0.2/0.5)
        ax_bot.bar(bar_ticks,clo_inj+lob_inj+rc_inj,0.4,label=tag+'Inj',
                ec='mediumvioletred',fc='palevioletred',hatch=hatch)
        ax_bot.bar(bar_ticks,clo_esc+lob_esc+rc_esc,0.4,label=tag+'Esc',
                ec='peru',fc='peachpuff',hatch=hatch)
        ax_bot.bar(bar_ticks,clo_net+lob_net+rc_net,0.4,label=tag+'Net',
                ec='black',fc='silver',hatch=hatch)
    ax_bot.set_xticks(bar_ticks)
    ax_bot.set_xticklabels(bar_labels,rotation=15,fontsize=12)
    general_plot_settings(ax_bot,ylabel=Wlabel,do_xlabel=False,legend=False,
                          iscontour=True)
    #save
    qt_bar.tight_layout(pad=1)
    qt_bar.savefig(outQT+'/quiet_bar_energy.png')
    plt.close(qt_bar)
    ######################################################################
    ##Main phase
    #Decide on ylimits for interfaces
    #interface_list = ['Dayside_reg','Tail_close','L7','PSB','MidLat',
    #                  'Flank','Tail_lobe','Poles','LowLat']
    interface_list = ['Dayside_reg','Flank','PSB']
    ylims = [[-8,8],[-5,5],[-10,10]]
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    incr = 7.5
    '''
    ylims = [[-2*incr,2*incr],
             [-1*incr,1*incr],
             [-1*incr,1*incr],
             [-3.5*incr,3.5*incr],
             [-0.5*incr,0.5*incr],
             [-2*incr,1*incr],
             [-0.5*incr,0.5*incr],
             [-0.5*incr,0.5*incr],
             [-0.5*incr,0.5*incr]]
    '''
    #h_ratios=[6,2,2,8,1,6,1,1,1]
    h_ratios=[3,3,1,1,1,1,4,4,1,1,3,3,1,1,1,1,1,1]

    ##Line plot Energy
    #setup figure
    main1_energy, [ax1,ax2] = plt.subplots(2,1,sharey=False,sharex=True,
                                           figsize=[14,8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot
    ax1.plot(febtime_mn1, feb_mp_mn1['Utot2 [J]'],label=feblabel)
    ax2.plot(startime_mn1, star_mp_mn1['Utot2 [J]'],label=starlabel)
    general_plot_settings(ax1,ylabel=Elabel,do_xlabel=False,xlabel=Tlabel)
    general_plot_settings(ax2,ylabel=Elabel,do_xlabel=True,xlabel=Tlabel)

    #save
    main1_energy.tight_layout(pad=1)
    main1_energy.savefig(outMN1+'/main1_total_energy.png')
    plt.close(main1_energy)

    ##Stack plot Energy by type (hydro,magnetic) for each region
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ PJ\right]$'
    Tlabel = r'Time $\left[ hr\right]$'
    for sz in ['ms_full','lobes','closed','rc']:
        #setup figures
        main1_distr, [ax1,ax2] = plt.subplots(2,1,sharey=False,sharex=True,
                                              figsize=[14,8])

        #plot
        plot_stack_distr(ax1, febtime_mn1,feb_mp_mn1, feb_msdict_mn1,
                         value_set='Energy2', doBios=False, label=feblabel,
                         ylabel=Elabel,legend_loc='upper left',subzone=sz)
        plot_stack_distr(ax2, startime_mn1,star_mp_mn1, star_msdict_mn1,
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

    feb_msdict_mn1.pop('missed')
    star_msdict_mn1.pop('missed')
    #plot
    plot_stack_contrib(ax1, febtime_mn1,feb_mp_mn1, feb_msdict_mn1,
                         value_key='Utot2 [J]', label=feblabel,ylim=[0,15],
                         ylabel=Elabel,legend_loc='upper right')
    plot_stack_contrib(ax2, startime_mn1,star_mp_mn1, star_msdict_mn1,
                         value_key='Utot2 [J]', label=starlabel,hatch='*',
                         ylabel=Elabel,legend=False,ylim=[0,15],
                         do_xlabel=True,xlabel=r'Time $\left[Hr\right]$')

    #save
    main1_contr.tight_layout(pad=1)
    main1_contr.savefig(outMN1+'/main1_contr_energy.png')
    plt.close(main1_contr)

    ##Lineplots for event comparison 1 axis per interface
    main_ylims = []
    main1_interf, ax = plt.subplots(2*len(interface_list),sharex=True,
                                    figsize=[9,6*len(interface_list)])
                          #gridspec_kw={'height_ratios':h_ratios})
    for i, (dic,times,tag) in enumerate([(feb_msdict_mn1,febtime_mn1,'feb'),
                                  (star_msdict_mn1,startime_mn1,'star')]):
        if i==1: hatch='*'
        else: hatch=None
        for j,interf in enumerate(interface_list):
            if interf in closed:
                sz = 'closed'
            elif interf in lobes:
                sz = 'lobes'
            elif interf in ring_c:
                sz = 'rc'
            #plot
            plot_power(ax[2*j+i],dic[sz],times,legend=False,
                       inj='K_injection'+interf+' [W]',
                       esc='K_escape'+interf+' [W]',
                       net='K_net'+interf+' [W]',
                       ylabel=safelabel(interf.split('_reg')[0]),
                       hatch=hatch,ylim=ylims[j],
                       do_xlabel=(j==2*len(interface_list)-1))
    #save
    main1_interf.tight_layout(pad=0.2)
    main1_interf.savefig(outMN1+'/main1_interf.png')
    plt.close(main1_interf)


    ##Interface generic multiplot
    '''
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

    '''
    ######################################################################
    ##Recovery phase
    ##Lineplots for event comparison 1 axis per interface
    reco_interf, ax = plt.subplots(2*len(interface_list),sharex=True,
                                    figsize=[9,6*len(interface_list)])
                          #gridspec_kw={'height_ratios':h_ratios})
    for i, (dic,times,tag) in enumerate([(feb_msdict_rec,febtime_rec,'feb'),
                                  (star_msdict_rec,startime_rec,'star')]):
        if i==1: hatch='*'
        else: hatch=None
        for j,interf in enumerate(interface_list):
            if interf in closed:
                sz = 'closed'
            elif interf in lobes:
                sz = 'lobes'
            elif interf in ring_c:
                sz = 'rc'
            plot_power(ax[2*j+i],dic[sz],times,legend=False,
                       inj='K_injection'+interf+' [W]',
                       esc='K_escape'+interf+' [W]',
                       net='K_net'+interf+' [W]',
                       ylabel=safelabel(interf.split('_reg')[0]),
                       hatch=hatch,ylim=ylims[j],
                       do_xlabel=(j==2*len(interface_list)-1))
            ax[2*j+i].yaxis.tick_right()
            ax[2*j+i].yaxis.set_label_position('right')
    #save
    reco_interf.tight_layout(pad=0.2)
    reco_interf.savefig(outRec+'/reco_interf.png')
    plt.close(reco_interf)

    ##Stack plot Energy by region
    #setup figure
    reco_contr, [ax1,ax2] = plt.subplots(2,1,sharey=True,sharex=True,
                                          figsize=[14,8])

    feb_msdict_rec.pop('missed')
    star_msdict_rec.pop('missed')
    #plot
    plot_stack_contrib(ax1, febtime_rec,feb_mp_rec, feb_msdict_rec,
                         value_key='Utot2 [J]', label=feblabel,ylim=[0,15],
                         ylabel=Elabel,legend_loc='upper right')
    plot_stack_contrib(ax2, startime_rec,star_mp_rec, star_msdict_rec,
                         value_key='Utot2 [J]', label=starlabel,hatch='*',
                         ylabel=Elabel,legend=False,ylim=[0,15],
                         do_xlabel=True,xlabel=r'Time $\left[Hr\right]$')

    #save
    reco_contr.tight_layout(pad=1)
    reco_contr.savefig(outRec+'/reco_contr_energy.png')
    plt.close(reco_contr)

    ######################################################################
    #TODO: 2 panels:
    ##Bonus plot
    #setup figure
    bonus, ax = plt.subplots(2,1,sharey=True,sharex=True,
                                          figsize=[14,8])
    #setup a combo time with main+recovery
    feb_lob_combo = feb_msdict_mn1['lobes'].append(
                                                 feb_msdict_rec['lobes'])
    febtime_combo = [dt.datetime(2000,1,1)+r for r in
            feb_lob_combo.index-feb_lob_combo.index[0]]

    feb_obs_combo = feb_obs_mn1.append(feb_obs_rec)
    ofebtime_combo = [dt.datetime(2000,1,1)+r for r in
            feb_obs_combo.index-feb_obs_combo.index[0]]

    star_lob_combo = star_msdict_mn1['lobes'].append(
                                                  star_msdict_rec['lobes'])
    startime_combo = [dt.datetime(2000,1,1)+r for r in
            star_lob_combo.index-star_lob_combo.index[0]]

    star_obs_combo = star_obs_mn1.append(star_obs_rec)
    ostartime_combo = [dt.datetime(2000,1,1)+r for r in
            star_obs_combo.index-star_obs_combo.index[0]]

    #plot_pearson_r(ax, tx, ty, xseries, yseries, **kwargs):
    for i, (dic,times,obs,ot,tag) in enumerate([
     (feb_lob_combo,febtime_combo,feb_obs_combo,ofebtime_combo,'feb'),
     (star_lob_combo,startime_combo,star_obs_combo,ostartime_combo,'star')]):
        if i==1:hatch='*';ylabel=''
        else:hatch=None;ylabel=r'Lobe Power/Energy'
        #twin axis with fill (grey) of Energy in lobes
        ax[i].fill_between(times,dic['Utot2 [J]']/1e15,fc='thistle',
                         hatch=hatch,ec='dimgrey',lw=0.0,
                         label=r'Energy $\left[PJ\right]$')
        #Net Flank and PSB on regular axis
        ax[i].plot(times, -1*dic['K_netFlank [W]']/1e12,
                   color='darkgreen',label=r'-1*Flank $\left[TW\right]$',
                   lw=2)
        ax[i].plot(times, dic['K_netPSB [W]']/1e12,
                   color='darkgoldenrod',label=r'PBS $\left[TW\right]$',
                   lw=2)
        #3rd axis with Bz, highlighed by sign???
        ax2 = ax[i].twinx()
        ax2.spines['right'].set_color('tab:blue')
        ax2.plot(ot,obs['bz'],lw=2,
                 color='tab:blue',label=r'$B_z$')
        ax2.set_ylabel(r'IMF $\left[B_z\right]$')
        ax2.set_ylim([-20,20])
        general_plot_settings(ax[i],ylabel=r'Lobe Power/Energy',
                    do_xlabel=(i==1),xlabel=r'Time $\left[Hr\right]$',
                    legend=(i==0),ylim=[-6,6])
        ax[1].set_ylabel('')
    #save
    bonus.tight_layout(pad=1)
    bonus.savefig(outRec+'/bonus.png')
    plt.close(bonus)

    #TODO: 4 panels:
    #setup figure
    sw, [ax1,ax2,ax3,ax4] = plt.subplots(4,1,sharey=True,sharex=True,
                                          figsize=[14,8])
    #       Event info, standard "side" format
    #       Overlay the phases (Quiet, main, recovery)
    #save
    sw.tight_layout(pad=1)
    sw.savefig(outRec+'/reco_contr_energy.png')
    plt.close(sw)
