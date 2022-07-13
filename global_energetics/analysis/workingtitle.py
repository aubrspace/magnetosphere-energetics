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
        rel_time (series object of times starting from 0)
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
    may2019_impact = dt.datetime(2019,5,13,0,0)
    may2019_endMain1 = dt.datetime(2019,5,14,7,45)
    may2019_endMain2 = dt.datetime(2019,5,14,7,45)
    #Aug2019
    aug2019_impact = dt.datetime(2019,8,30,11,56)
    aug2019_endMain1 = dt.datetime(2019,8,31,12,0)
    aug2019_endMain2 = dt.datetime(2019,8,31,12,0)

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
    elif abs(times-may2019_impact).min() < dt.timedelta(minutes=15):
        impact = may2019_impact
        peak1 = may2019_endMain1
        peak2 = may2019_endMain2
    elif abs(times-may2019_impact).min() < dt.timedelta(minutes=15):
        impact = aug2019_impact
        peak1 = aug2019_endMain1
        peak2 = aug2019_endMain2
        #TODO find the points and see if aug2019 is being loaded wrong
        from IPython import embed; embed()
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
        rel_time = [dt.datetime(2000,1,1)+r for r in
                    indata[cond].index-indata[cond].index[0]]
        return indata[cond], rel_time
    elif type(indata) == dict:
        for key in indata.keys():
            df = indata[key]
            phase.update({key:df[cond]})
            rel_time = [dt.datetime(2000,1,1)+r for r in
                        df[cond].index-df[cond].index[0]]
        return phase, rel_time


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

    #HDF data, will be sorted and cleaned
    ds = {}
    ds['feb'] = load_hdf_sort(inPath+'feb2014_results.h5')
    ds['star'] = load_hdf_sort(inPath+'starlink_results.h5')
    ds['aug'] = load_hdf_sort(inPath+'aug2019_results.h5')

    #Log files and observational indices
    ds['feb']['obs'] = read_indices(inPath, prefix='feb2014_',
                                    read_supermag=False, tshift=45)
    ds['star']['obs'] = read_indices(inPath, prefix='starlink_',
                                     read_supermag=False)
    ds['aug']['obs'] = read_indices(inPath, prefix='aug2019_',
                                     read_supermag=False)

    #NOTE hotfix for closed region tail_closed
    for t in[t for t in ds['feb']['msdict']['closed'].keys()
                                                    if 'Tail_close'in t]:
        for event in ds.keys():
            ds[event]['msdict']['closed'][t] = ds[event]['mpdict'][
                                                              'ms_full'][t]

    ##Construct "grouped" set of subzones, then get %contrib for each
    for event in ds.keys():
        ds[event]['mpdict'],ds[event]['msdict'] = get_subzone_contrib(
                                                       ds[event]['mpdict'],
                                                       ds[event]['msdict'])
        ds[event]['msdict'] = {'rc':ds[event]['msdict']['rc'],
                               'closed':ds[event]['msdict']['closed'],
                               'lobes':ds[event]['msdict']['lobes']}
                               #'missed':ds[event]['msdict']['missed']}
    #TODO: refactor to add more toplevels to have one master "dataset" mega
    #      dict with entries for each event
    #       - mp->full + ms-> rc, lobe, close
    #           - qt, mn1, rec
    #               - data + time
    ##Parse storm phases
    for event in ds.keys():
        #quiet time
        ds[event]['mp_qt'], ds[event]['time_qt'] = locate_phase(
                                       ds[event]['mpdict']['ms_full'],'qt')
        ds[event]['msdict_qt'], _ = locate_phase(
                                                  ds[event]['msdict'],'qt')
        #Main phase
        ds[event]['mp_mn1'], ds[event]['time_mn1'] = locate_phase(
                                    ds[event]['mpdict']['ms_full'],'main1')
        ds[event]['msdict_mn1'], _ = locate_phase(
                                               ds[event]['msdict'],'main1')
        ds[event]['obs_mn1'], ds[event]['otime_mn1'] = locate_phase(
                                       ds[event]['obs']['swmf_sw'],'main1')
        #Recovery
        ds[event]['mp_rec'], ds[event]['time_rec'] = locate_phase(
                                    ds[event]['mpdict']['ms_full'],'rec')
        ds[event]['msdict_rec'], _ = locate_phase(
                                               ds[event]['msdict'],'rec')
        ds[event]['obs_rec'], ds[event]['otime_rec'] = locate_phase(
                                       ds[event]['obs']['swmf_sw'],'rec')

    ######################################################################
    ##Quiet time
    #TODO: summary bar plot with space for contour of time vs L vs energy

    ##Bar plot with energy distribution summary
    #setup figure
    qt_bar, [ax_top,ax_bot] = plt.subplots(2,1,sharey=False,sharex=False,
                                           figsize=[4*len(ds.keys()),8])
    feblabel = 'Feb 2014'
    starlabel = 'Feb 2022'
    Elabel = r'Energy $\left[ J\right]$'
    Plabel = r'Energy $\left[ PJ\right]$'
    Wlabel = r'Power $\left[ TW\right]$'
    Tlabel = r'Time $\left[ hr\right]$'

    #plot bars for each region
    shifts = np.linspace(-0.1*len(ds.keys()),0.1*len(ds.keys()),
                         len(ds.keys()))
    hatches = ['','*','x','o']
    for i,ev in enumerate(ds.keys()):
        bar_labels = ds[ev]['msdict_qt'].keys()
        bar_ticks = np.arange(len(bar_labels))
        ax_top.bar(bar_ticks+shifts[i],
         [ds[ev]['msdict_qt'][k]['Utot2 [J]'].mean()/1e15
                                        for k in ['rc','lobes','closed']],
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
        bar_ticks = np.arange(len(interface_list))+shifts[i]
        ax_bot.bar(bar_ticks,clo_inj+lob_inj+rc_inj,0.4,label=ev+'Inj',
                ec='mediumvioletred',fc='palevioletred',hatch=hatches[i])
        ax_bot.bar(bar_ticks,clo_esc+lob_esc+rc_esc,0.4,label=ev+'Esc',
                ec='peru',fc='peachpuff',hatch=hatches[i])
        ax_bot.bar(bar_ticks,clo_net+lob_net+rc_net,0.4,label=ev+'Net',
                ec='black',fc='silver',hatch=hatches[i])
    ax_bot.set_xticks(bar_ticks)
    ax_bot.set_xticklabels(bar_labels,rotation=15,fontsize=12)
    general_plot_settings(ax_bot,ylabel=Wlabel,do_xlabel=False,legend=False,
                          iscontour=True)
    #save
    qt_bar.tight_layout(pad=1)
    qt_bar.savefig(outQT+'/quiet_bar_energy.png')
    plt.close(qt_bar)
    ######################################################################
    ##Main + Recovery phase
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

    for ph,path in [('_mn1',outMN1),('_rec',outRec)]:
        ##Stack plot Energy by type (hydro,magnetic) for each region
        for sz in ['ms_full','lobes','closed','rc']:
            #setup figures
            distr, ax = plt.subplots(len(ds.keys()),1,sharey=True,
                                sharex=True,figsize=[14,4*len(ds.keys())])

            #plot
            for i,ev in enumerate(ds.keys()):
                plot_stack_distr(ax[i],ds[ev]['time'+ph],ds[ev]['mp'+ph],
                                       ds[ev]['msdict'+ph],
                            value_set='Energy2', doBios=False, label=ev,
                          ylabel=Elabel,legend_loc='upper left',subzone=sz)
            #save
            distr.tight_layout(pad=1)
            distr.savefig(path+'/distr_energy'+ph+sz+'.png')
            plt.close(distr)

        ##Stack plot Energy by region
        #setup figure
        contr,ax=plt.subplots(len(ds.keys()),1,sharey=True,
                              sharex=True,figsize=[14,4*len(ds.keys())])

        #plot
        for i,ev in enumerate(ds.keys()):
            plot_stack_contrib(ax[i], ds[ev]['time'+ph],ds[ev]['mp'+ph],
                                      ds[ev]['msdict'+ph],
                            value_key='Utot2 [J]', label=ev,ylim=[0,15],
                            legend=(i==0),ylabel=Elabel,
                            legend_loc='upper right', hatch=hatches[i],
                            do_xlabel=(i==len(ds.keys())-1),
                            xlabel=r'Time $\left[Hr\right]$')

        #save
        contr.tight_layout(pad=1)
        contr.savefig(path+'/contr_energy'+ph+'.png')
        plt.close(contr)

        ##Lineplots for event comparison 1 axis per interface
        interf_fig, ax = plt.subplots(2*len(interface_list),sharex=True,
                         figsize=[9,3*len(interface_list)*len(ds.keys())])
                            #gridspec_kw={'height_ratios':h_ratios})
        for i,ev in enumerate(ds.keys()):
            dic = ds[ev]['msdict'+ph]
            for j,interf in enumerate(interface_list):
                if interf in closed:
                    sz = 'closed'
                elif interf in lobes:
                    sz = 'lobes'
                elif interf in ring_c:
                    sz = 'rc'
                #plot
                plot_power(ax[len(ds.keys())*j+i],dic[sz],
                           ds[ev]['time'+ph],legend=False,
                           inj='K_injection'+interf+' [W]',
                           esc='K_escape'+interf+' [W]',
                           net='K_net'+interf+' [W]',
                           ylabel=safelabel(interf.split('_reg')[0]),
                           hatch=hatches[i],ylim=ylims[j],
                       do_xlabel=(j==len(ds.keys())*len(interface_list)-1))
                if ph=='_rec':
                    ax[len(ds.keys())*j+i].yaxis.tick_right()
                    ax[len(ds.keys())*j+i].yaxis.set_label_position('right')
        #save
        interf_fig.tight_layout(pad=0.2)
        interf_fig.savefig(path+'/interf'+ph+'.png')
        plt.close(interf_fig)

    ######################################################################
    #TODO: 2 panels:
    ##Bonus plot
    #setup figure
    bonus, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    #plot_pearson_r(ax, tx, ty, xseries, yseries, **kwargs):
    for i,ev in enumerate(ds.keys()):
        if i==1:ylabel=''
        else:ylabel=r'Lobe Power/Energy'
        for j,ph in enumerate(['_mn1','_rec']):
            dic = ds[ev]['msdict'+ph]['lobes']
            obs = ds[ev]['obs'+ph]
            if j>0:
                times = [t+(times[-1]-times[0]) for t in ds[ev]['time'+ph]]
                ot    = [t+(ot[-1]-ot[0]) for t in ds[ev]['otime'+ph]]
                h='_'
            else:
                times = ds[ev]['time'+ph]
                ot    = ds[ev]['otime'+ph]
                h=''
            #twin axis with fill (grey) of Energy in lobes
            ax[i].fill_between(times,dic['Utot2 [J]']/1e15,fc='thistle',
                               hatch=hatches[i],ec='dimgrey',lw=0.0,
                               label=h+r'Energy $\left[PJ\right]$')
            #Net Flank and PSB on regular axis
            ax[i].plot(times, -1*dic['K_netFlank [W]']/1e12,
                       color='darkgreen',
                       label=h+r'-1*Flank $\left[TW\right]$', lw=2)
            ax[i].plot(times, dic['K_netPSB [W]']/1e12,
                       color='darkgoldenrod',
                       label=h+r'PBS $\left[TW\right]$',lw=2)
            #3rd axis with Bz, highlighed by sign???
            ax2 = ax[i].twinx()
            ax2.spines['right'].set_color('tab:blue')
            ax2.plot(ot,obs['bz'],lw=2,
                 color='tab:blue',label=h+r'$B_z$')
            ax2.set_ylabel(r'IMF $\left[B_z\right]$')
            ax2.set_ylim([-20,20])
            general_plot_settings(ax[i],ylabel=ylabel,
                    do_xlabel=(i==len(ds.keys())-1),
                    xlabel=r'Time $\left[Hr\right]$',
                    legend=(i==0),ylim=[-6,6])
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
