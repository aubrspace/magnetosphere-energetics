#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys,glob,time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
from scipy.stats import linregress
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker, colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.analysis.plot_tools import (central_diff,
                                                   pyplotsetup,safelabel,
                                                   general_plot_settings,
                                                   plot_stack_distr,
                                                   plot_pearson_r,
                                                   plot_stack_contrib,
                                                   refactor)
from global_energetics.analysis.proc_hdf import (load_hdf_sort)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.analyze_energetics import plot_power
from global_energetics.analysis.proc_timing import (peak2peak,
                                                    pearson_r_shifts)
from global_energetics.analysis.workingtitle import (
                                                     stack_energy_region_fig,
                                                     lobe_balance_fig,
                                                     solarwind_figure)

def interval_average(ev):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       ev['Ks1'].index)
    tave_ev = pd.DataFrame()
    for i,(start,end) in enumerate(interval_list):
        interv = (ev['Ks1'].index>start)&(ev['Ks1'].index<end)
        siminterv = (ev['sim'].index>start)&(ev['sim'].index<end)
        swinterv = ev['sw'].index==(start-(interval_list[0][0]-
                                           dt.datetime(2022,6,6,0,0)))
        for key in ev['sw'].keys():
            tave_ev.loc[i,key] = ev['sw'].loc[swinterv,key].mean()
        banlist = ['lobes','closed','mp','inner',
                   'sim','sw','times','simt','swt','dDstdt_sim']
        keylist = [k for k in ev.keys() if k not in banlist]
        for key in[k for k in keylist if type(ev[k])is not type([])]:
            tave_ev.loc[i,key] = ev[key][interv].mean()
    return tave_ev

def build_interval_list(tstart,tlength,tjump,alltimes):
    interval_list = []
    tend = tstart+tlength
    while tstart<alltimes[-1]:
        interval_list.append([tstart,tend])
        tstart+=tjump
        tend+=tjump
    return interval_list

def series_segments(event,ev,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       ev['Ks1'].index)
    #############
    #setup figure
    series_segK,(Kaxis,Kaxis2,Kaxis3) =plt.subplots(3,1,figsize=[20,24],
                                                 sharex=True)
    series_segH,(Haxis,Haxis2,Haxis3) =plt.subplots(3,1,figsize=[20,24],
                                                 sharex=True)
    series_segS,(Saxis,Saxis2,Saxis3) =plt.subplots(3,1,figsize=[20,24],
                                                 sharex=True)
    #Plot
    Kaxis.plot(ev['times'],ev['U']/1e15,label='Energy',c='grey')
    Haxis.plot(ev['times'],ev['mp']['uHydro [J]']/1e15,label='Energy',c='grey')
    Saxis.plot(ev['times'],ev['mp']['uB [J]']/1e15,label='Energy',c='grey')
    for i,(start,end) in enumerate(interval_list):
        interv = (ev['Ks1'].index>start)&(ev['Ks1'].index<end)
        interv_timedelta = [t-TSTART for t in ev['M1'][interv].index]
        interv_times=[float(n.to_numpy()) for n in interv_timedelta]

        # Total energy
        Kaxis.plot(interv_times,ev['U'][interv]/1e15,label=str(i))
        Kaxis2.plot(interv_times,
                   (ev['mp']['K_injectionK5 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Kaxis2.plot(interv_times,
                   (ev['mp']['K_escapeK5 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Kaxis2.fill_between(interv_times,
                   (ev['mp']['K_netK5 [W]'])[interv]/1e12,
                   label=str(i),fc='grey')
        Kaxis2.fill_between(interv_times,
                   (ev['mp']['K_netK5 [W]'])[interv]/1e12,
             (ev['mp']['K_netK5 [W]']+ev['mp']['UtotM5 [W]'])[interv]/1e12,
                   label=str(i))
        Kaxis3.plot(interv_times,
                   (ev['mp']['K_injectionK1 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Kaxis3.plot(interv_times,
                   (ev['mp']['K_escapeK1 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Kaxis3.fill_between(interv_times,
                           (ev['Ks1'])[interv]/1e12,
                   label=str(i),fc='grey')
        Kaxis3.fill_between(interv_times,
                           (ev['Ks1'])[interv]/1e12,
                           (ev['Ks1']+ev['M1'])[interv]/1e12,
                   label=str(i))
        # Hydro energy
        Haxis.plot(interv_times,ev['mp']['uHydro [J]'][interv]/1e15,
                   label=str(i))
        Haxis2.plot(interv_times,
                   (ev['mp']['P0_injectionK5 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Haxis2.plot(interv_times,
                   (ev['mp']['P0_escapeK5 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Haxis2.fill_between(interv_times,
                   (ev['Hs5'])[interv]/1e12,
                   label=str(i),fc='grey')
        Haxis2.fill_between(interv_times,
                   (ev['Hs5'])[interv]/1e12,
                   (ev['Hs5']+ev['HM1'])[interv]/1e12,
                   label=str(i))
        Haxis3.plot(interv_times,
                   (ev['mp']['P0_injectionK1 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Haxis3.plot(interv_times,
                   (ev['mp']['P0_escapeK1 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Haxis3.fill_between(interv_times,
                           (ev['Hs1'])[interv]/1e12,
                   label=str(i),fc='grey')
        Haxis3.fill_between(interv_times,
                           (ev['Hs1'])[interv]/1e12,
                           (ev['Hs1']+ev['HM1'])[interv]/1e12,
                   label=str(i))
        # Mag energy
        Saxis.plot(interv_times,ev['mp']['uB [J]'][interv]/1e15,label=str(i))
        Saxis2.plot(interv_times,
                   (ev['mp']['ExB_injectionK5 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Saxis2.plot(interv_times,
                   (ev['mp']['ExB_escapeK5 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Saxis2.fill_between(interv_times,
                   (ev['Ss5'])[interv]/1e12,
                   label=str(i),fc='grey')
        Saxis2.fill_between(interv_times,
                   (ev['Ss5'])[interv]/1e12,
                   (ev['Ss5']+ev['SM1'])[interv]/1e12,
                   label=str(i))
        Saxis3.plot(interv_times,
                   (ev['mp']['ExB_injectionK1 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        Saxis3.plot(interv_times,
                   (ev['mp']['ExB_escapeK1 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        Saxis3.fill_between(interv_times,
                           (ev['Ss1'])[interv]/1e12,
                   label=str(i),fc='grey')
        Saxis3.fill_between(interv_times,
                           (ev['Ss1'])[interv]/1e12,
                           (ev['Ss1']+ev['SM1'])[interv]/1e12,
                   label=str(i))


        Kaxis.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Kaxis2.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Kaxis3.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Haxis.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Haxis2.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Haxis3.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Saxis.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Saxis2.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
        Saxis3.axvline(float(pd.Timedelta(start-T0).to_numpy()),c='grey')
    labels = [('U','K'),('uHydro','H'),('uB','S')]
    for i,(axis,axis2,axis3) in enumerate([(Kaxis,Kaxis2,Kaxis3),
                                           (Haxis,Haxis2,Haxis3),
                                           (Saxis,Saxis2,Saxis3)]):
        general_plot_settings(axis,do_xlabel=False,legend=False,
                          ylabel=('Integrated '+labels[i][0]+
                                  r' Energy $\left[PJ\right]$'),
                          timedelta=True)
        general_plot_settings(axis2,do_xlabel=False,legend=False,
                          ylabel=('Integrated '+labels[i][1]+
                                  r'5 Flux $\left[TW\right]$'),
                          timedelta=True)
        general_plot_settings(axis3,do_xlabel=True,legend=False,
                          ylabel=('Integrated '+labels[i][1]+
                                  r'1 Flux $\left[TW\right]$'),
                          timedelta=True)
        axis.margins(x=0.01)
        axis2.margins(x=0.01)
        axis3.margins(x=0.01)
    ##Save
    series_segK.tight_layout(pad=1)
    figurename = path+'/series_seg_K_'+event+'.png'
    series_segK.savefig(figurename)
    plt.close(series_segK)
    print('\033[92m Created\033[00m',figurename)
    series_segH.tight_layout(pad=1)
    figurename = path+'/series_seg_H_'+event+'.png'
    series_segH.savefig(figurename)
    plt.close(series_segH)
    print('\033[92m Created\033[00m',figurename)
    series_segS.tight_layout(pad=1)
    figurename = path+'/series_seg_S_'+event+'.png'
    series_segS.savefig(figurename)
    plt.close(series_segS)
    print('\033[92m Created\033[00m',figurename)
    #############

def segments(event,ev,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       ev['Ks1'].index)
    #############
    #setup figure
    segments,(axis,axis2,axis3) =plt.subplots(3,1,figsize=[20,24],
                                                 sharey=False)
    #from IPython import embed; embed()
    #Plot
    for i,(start,end) in enumerate(interval_list):
        interv = (ev['Ks1'].index>start)&(ev['Ks1'].index<end)
        axis.plot(ev['M1'][interv].index,  ev['U'][interv]/1e15, label=str(i))
        axis2.plot(ev['M1'][interv].index,
                   (ev['mp']['K_injectionK5 [W]'])[interv]/1e12,
                   label=str(i),c='red')
        axis2.plot(ev['M1'][interv].index,
                   (ev['mp']['K_escapeK5 [W]'])[interv]/1e12,
                   label=str(i),c='blue')
        axis2.fill_between(ev['M1'][interv].index,
                   (ev['mp']['K_netK5 [W]'])[interv]/1e12,
                   label=str(i),fc='grey')
        axis2.fill_between(ev['M1'][interv].index,
                   (ev['mp']['K_netK5 [W]'])[interv]/1e12,
             (ev['mp']['K_netK5 [W]']+ev['mp']['UtotM5 [W]'])[interv]/1e12,
                   label=str(i))
        axis3.plot(ev['M1'][interv].index,(ev['Ks1']+ev['M1'])[interv]/1e12,
                   label=str(i))
    labels = [
              r'Integrated U Energy $\left[PJ\right]$',
              r'Integrated K5 Flux $\left[TW\right]$',
              r'Integrated K1 Flux $\left[TW\right]$'
              ]
    for i,ax in enumerate([axis,axis2,axis3]):
        general_plot_settings(ax,do_xlabel=(ax==axis3),legend=False,
                          ylim=[-15,5],
                          ylabel=labels[i],
                          timedelta=False)
        ax.margins(x=0.01)
    axis.set_ylabel(r'Energy $\left[PJ\right]$')
    axis.set_ylim([5,30])
    #Save
    segments.tight_layout(pad=1)
    figurename = path+'/segments'+event+'.png'
    segments.savefig(figurename)
    plt.close(segments)
    print('\033[92m Created\033[00m',figurename)

def common_x_bar(event,ev,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       ev['Ks1'].index)
    #############
    #setup figure
    common_xbar,(axis,axis2,axis3) =plt.subplots(3,1,figsize=[20,24],
                                                 sharey=True)
    #Plot
    for i,(start,end) in enumerate(interval_list):
        interv = (ev['Ks1'].index>start)&(ev['Ks1'].index<end)
        commondelta = [t-start for t in ev['Ks1'][interv].index]
        commont = [float(n.to_numpy()) for n in commondelta]
        axis.bar(i, (ev['Ks1']+ev['M1'])[interv].mean()/1e12, label=i)
        axis2.bar(i,(ev['Ks5']+ev['M5'])[interv].mean()/1e12, label=i)
        axis3.bar(i,(ev['Ks3']+ev['Ks7'])[interv].mean()/1e12, label=i)
    axis.set_ylabel(r'Integrated Tot. Flux K1 $\left[TW\right]$')
    axis2.set_ylabel(r'Integrated Tot. Flux K5 $\left[TW\right]$')
    axis3.set_ylabel(r'Integrated Tot. Flux Kinner $\left[TW\right]$')
    axis3.set_xlabel('Step ID')
    axis3.set_ylim([-15,5])
    axis.margins(x=0.01)
    axis2.margins(x=0.01)
    axis3.margins(x=0.01)
    #Save
    common_xbar.tight_layout(pad=1)
    figurename = path+'/common_xbar_'+event+'.png'
    common_xbar.savefig(figurename)
    plt.close(common_xbar)
    print('\033[92m Created\033[00m',figurename)

def common_x_lineup(event,ev,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       ev['Ks1'].index)
    #############
    #setup figure
    common_xline,(axis,axis2,axis3) =plt.subplots(3,1,figsize=[20,24],
                                                  sharey=True)
    #Plot
    for i,(start,end) in enumerate(interval_list):
        interv = (ev['Ks1'].index>start)&(ev['Ks1'].index<end)
        commondelta = [t-start for t in ev['Ks1'][interv].index]
        commont = [float(n.to_numpy()) for n in commondelta]
        if (i)%4==0:
        #if True:
            axis.plot(commont,  ev['M1'][interv]/1e12, label=str(i))
            axis2.plot(commont, ev['Ks1'][interv]/1e12, label=str(i))
            axis3.plot(commont,(ev['Ks1']+ev['M1'])[interv]/1e12,label=str(i))
            axis.axhline(ev['M1'][interv].mean()/1e12,ls='--')
            axis2.axhline(ev['Ks1'][interv].mean()/1e12,ls='--')
            axis3.axhline((ev['Ks1']+ev['M1'])[interv].mean()/1e12,ls='--')
        for ax in [axis,axis2,axis3]:
            ax.legend()
            ax.set_ylim(-15,5)
            '''
            general_plot_settings(ax,do_xlabel=(ax==axis3),legend=True,
                          ylim=[-10,10],
                          ylabel=r'Integrated Flux $\left[TW\right]$',
                          timedelta=False)
            '''
            ax.margins(x=0.01)
    #Save
    common_xline.tight_layout(pad=1)
    figurename = path+'/common_xline_'+event+'.png'
    common_xline.savefig(figurename)
    plt.close(common_xline)
    print('\033[92m Created\033[00m',figurename)

def test_matrix(event,ev,path):
    #############
    #setup figure
    matrix,(axis) =plt.subplots(1,1,figsize=[20,8],sharex=True)
    axis.plot(ev['swt'],ev['sw']['by'],label='by')
    axis.plot(ev['swt'],ev['sw']['bz'],label='bz')
    axis.fill_between(ev['swt'],np.sqrt(ev['sw']['by']**2+ev['sw']['bz']**2),
                                        fc='grey',label='B')

    general_plot_settings(axis,do_xlabel=True,legend=True,
                          ylabel=r'IMF $\left[nT\right]$',timedelta=True)
    axis.margins(x=0.01)
    matrix.tight_layout(pad=1)
    figurename = path+'/matrix'+event+'.png'
    matrix.savefig(figurename)
    plt.close(matrix)
    print('\033[92m Created\033[00m',figurename)

def indices(dataset,path):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       dataset['HIGHnHIGHu']['time'])
    #############
    #setup figure
    indices,(axis,axis2) =plt.subplots(2,1,figsize=[20,16],sharex=True)
    for event in dataset.keys():
        timedelta = [t-T0 for t in dataset[event]['obs']['swmf_log'].index]
        times = [float(n.to_numpy()) for n in timedelta]
        #supdelta = [t-T0 for t in dataset[event]['obs']['vsupermag'].index]
        #suptimes = [float(n.to_numpy()) for n in supdelta]
        inddelta = [t-T0 for t in dataset[event]['obs']['swmf_index'].index]
        indtimes = [float(n.to_numpy()) for n in inddelta]
        axis.plot(times,
                  dataset[event]['obs']['swmf_log']['dst_sm'],
                  label=event)
        axis2.plot(indtimes,dataset[event]['obs']['swmf_index']['AL'],
                   label=event)
    for interv in interval_list:
        axis.axvline(float(pd.Timedelta(interv[0]-TSTART).to_numpy()),c='grey')
        axis2.axvline(float(pd.Timedelta(interv[0]-TSTART).to_numpy()),
                      c='grey')

    general_plot_settings(axis,do_xlabel=True,legend=True,
                          ylabel=r'Dst $\left[nT\right]$',timedelta=True)
    general_plot_settings(axis2,do_xlabel=True,legend=True,
                          ylabel=r'SML $\left[nT\right]$',timedelta=True)
    axis.margins(x=0.01)
    indices.tight_layout(pad=1)
    figurename = path+'/indices.png'
    indices.savefig(figurename)
    plt.close(indices)
    print('\033[92m Created\033[00m',figurename)

def energies(dataset,path):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       dataset['HIGHnHIGHu']['time'])
    #############
    #setup figure
    energies,(axis) =plt.subplots(1,1,figsize=[20,8],sharex=True)
    for event in dataset.keys():
        timedelta = [t-T0 for t in dataset[event]['time']]
        times = [float(n.to_numpy()) for n in timedelta]
        axis.plot(times,
                  dataset[event]['mpdict']['ms_full']['Utot [J]']/1e15,
                  label=event)
    for interv in interval_list:
        axis.axvline(float(pd.Timedelta(interv[0]-TSTART).to_numpy()),c='grey')

    general_plot_settings(axis,do_xlabel=True,legend=True,
                          ylabel=r'Energy $\left[PJ\right]$',timedelta=True)
    axis.margins(x=0.01)
    energies.tight_layout(pad=1)
    figurename = path+'/energies.png'
    energies.savefig(figurename)
    plt.close(energies)
    print('\033[92m Created\033[00m',figurename)

def couplers(dataset,path):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       dataset['HIGHnHIGHu']['time'])
    #############
    #setup figure
    couplers,(axis) =plt.subplots(1,1,figsize=[20,8],sharex=True)
    for event in dataset.keys():
        swtdelta = [t-T0 for t in dataset[event]['obs']['swmf_sw'].index]
        swtimes=[float(n.to_numpy()) for n in swtdelta]
        E0 = dataset[event]['mpdict']['ms_full']['Utot [J]'].dropna()[0]/1e15
        axis.plot(swtimes,E0+
                  dataset[event]['obs']['swmf_sw']['EinWang'].cumsum()*60/1e15,
                  #axis.plot(swtimes,dataset[event]['obs']['swmf_sw']['bz'],
                  label=event)
    for interv in interval_list:
        axis.axvline(float(pd.Timedelta(interv[0]-TSTART).to_numpy()),c='grey')

    general_plot_settings(axis,do_xlabel=True,legend=True,
                          ylabel=r'Energy $\left[PJ\right]$',timedelta=True)
    axis.margins(x=0.01)
    couplers.tight_layout(pad=1)
    figurename = path+'/couplers.png'
    couplers.savefig(figurename)
    plt.close(couplers)
    print('\033[92m Created\033[00m',figurename)

def explorer(event,ev,path):
    #############
    #setup figure
    explore1,(axis,axis2,axis3) =plt.subplots(3,1,figsize=[20,24],sharex=True)
    axis.plot(ev['swt'],ev['sw']['by'],label='by')
    axis.plot(ev['swt'],ev['sw']['bz'],label='bz')
    axis.fill_between(ev['swt'],np.sqrt(ev['sw']['by']**2+ev['sw']['bz']**2),
                                        fc='grey',label='B')

    axis2.plot(ev['times'],ev['U']/1e15,label='Energy')

    axis3.plot(ev['times'],(ev['Ks1']+ev['M1'])/1e12,label='K1')
    axis3.plot(ev['times'],(ev['Ks5']+ev['M5'])/1e12,label='K5')
    axis3.plot(ev['times'],ev['Ks3']/1e12,label='K3')
    axis3.plot(ev['times'],ev['Ks7']/1e12,label='K7')
    axis3.plot(ev['times'],ev['Ks4']/1e12,label='K4')
    axis3.plot(ev['times'],ev['Ks6']/1e12,label='K6')
    general_plot_settings(axis,do_xlabel=False,legend=True,
                          ylabel=r'IMF $\left[nT\right]$',timedelta=True)
    general_plot_settings(axis2,do_xlabel=False,legend=True,
                          ylabel=r'Energy $\left[PJ\right]$',timedelta=True)
    general_plot_settings(axis3,do_xlabel=True,legend=True,
                          ylim=[-20,18],
                          ylabel=r'Integrated Flux $\left[TW\right]$',
                          timedelta=True)
    axis.margins(x=0.01)
    axis2.margins(x=0.01)
    axis3.margins(x=0.01)
    explore1.tight_layout(pad=1)
    figurename = path+'/explore1'+event+'.png'
    explore1.savefig(figurename)
    plt.close(explore1)
    print('\033[92m Created\033[00m',figurename)


def external_flux_timeseries(event,times,
                             HM1,Hs1,HM5,Hs5,Hs4,Hs6,Hs3,Hs7,
                             SM1,Ss1,SM5,Ss5,Ss4,Ss6,Ss3,Ss7,
                              M1,Ks1, M5,Ks5,Ks4,Ks6,Ks3,Ks7,
                             path):
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

    #Decorations
    powerlabel=['Hydro.','Poynting','Tot. Energy']
    for i,ax in enumerate([axis,axis2,axis3]):
        general_plot_settings(ax,do_xlabel=(i==2),legend=False,
                                ylabel='Integrated '+powerlabel[i]+
                                    r' Flux $\left[ TW\right]$',
                                  legend_loc='lower left',
                                  ylim=[-15,5],
                                  timedelta=True)
        #ax.axvspan((moments['impact']-
        #            moments['peak2']).total_seconds()*1e9,0,
        #            fc='lightgrey')
        ax.margins(x=0.01)
    axis.fill_between(times,(HM1+HM5+Hs1+Hs5+Hs4+Hs6+Hs3+Hs7)/1e12,
                        label='Total',fc='dimgray')
    axis2.fill_between(times,(SM1+SM5+Ss1+Ss5+Ss4+Ss6+Ss3+Ss7)/1e12,
                        label='Total',fc='dimgray')
    axis3.fill_between(times,(M1+M5+Ks1+Ks5+Ks4+Ks6+Ks3+Ks7)/1e12,
                        label='Total',fc='dimgray')
    axis.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=7, fancybox=True, shadow=True)
    #save
    #flavors_external.suptitle(moments['peak1'].strftime(
    #                                                 "%b %Y, t0=%d-%H:%M:%S"),
    #                                  ha='left',x=0.01,y=0.99)
    flavors_external.tight_layout(pad=1)
    figurename = path+'/external_flux_'+event+'.png'
    flavors_external.savefig(figurename)
    plt.close(flavors_external)
    print('\033[92m Created\033[00m',figurename)
    #############

def scatter_Ein_compare(tave):
    #############
    #setup figure
    scatter_Ein,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    all_X = np.array([])
    all_Y = np.array([])
    for case in tave.keys():
        ev = tave[case]
        #axis.scatter(ev['EinWang']/1e12,-(ev['K1'])/1e12,label=case,
        #             s=200)
        #axis.scatter(ev['EinWang']/1e12,
        #             -(ev['K5']+ev['K1']+ev['Ks6']+ev['Ks4'])/1e12,
        #             label=case,s=200)
        axis.scatter(ev['EinWang']/1e12,-(ev['K1'])/1e12,label=case,
                     s=200)
        all_X = np.append(all_X,ev['EinWang'].values/1e12)
        all_Y = np.append(all_Y,-ev['K1'].values/1e12)
    slope,intercept,r,p,stderr = linregress(all_X,all_Y)
    linelabel = (f'-K1 Flux [TW]:{slope:.2f}Ein [TW], r={r:.2f}')
    axis.plot(all_X,(intercept+all_X*slope),
                      label=linelabel,color='grey', ls='--')
    #Decorations
    axis.set_ylim([-5,25])
    axis.set_xlim([-5,25])
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.axline((0,0),(1,1),ls='--',c='grey')
    #axis.set_ylabel(r'-Integrated K1+K5 Flux $\left[ TW\right]$')
    axis.set_ylabel(r'-Integrated LobeSheath K1 Flux $\left[ TW\right]$')
    axis.set_xlabel(r'Wang et. al 2014 $E_{in}\left[TW\right]$')
    axis.legend()
    scatter_Ein.tight_layout(pad=1)
    figurename = path+'/scatter_Ein.png'
    scatter_Ein.savefig(figurename)
    plt.close(scatter_Ein)
    print('\033[92m Created\033[00m',figurename)
    #############

def scatter_Pstorm(tave):
    #############
    #setup figure
    scatterPstorm,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    for case in tave.keys():
        ev = tave[case]
        #axis.scatter(ev['Newell']/1e3,ev['K1']/1e12,label=case,
        #             s=200)
        axis.scatter(-ev['Pstorm']/1e12,-(ev['Ksum'])/1e12,label=case,
                     s=200)
    #Decorations
    axis.set_ylim([-5,25])
    axis.set_xlim([-5,25])
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.axline((0,0),(1,1),ls='--',c='grey')
    #axis.set_ylabel(r'-Integrated K1+K5 Flux $\left[ TW\right]$')
    axis.set_ylabel(r'-Integrated K Flux $\left[ TW\right]$')
    #axis.set_xlabel(r'Newell $\frac{d\phi}{dt}\left[kV\right]$')
    axis.set_xlabel(r'-$P_{storm}\left[TW\right]$')
    axis.legend()
    scatterPstorm.tight_layout(pad=1)
    figurename = path+'/scatter_Pstorm.png'
    scatterPstorm.savefig(figurename)
    plt.close(scatterPstorm)
    print('\033[92m Created\033[00m',figurename)

def scatter_internalFlux(tave):
    #############
    #setup figure
    scatterK2b,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    all_X = np.array([])
    all_Y = np.array([])
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['Ulobes']/1e15,(ev['Ks2bl']+ev['M2b'])/1e12,
                          label=case,
                          marker=shapes[i],
                          s=200)
                          #cmap='twilight',
                          #c=(np.rad2deg(ev['clock'])+360)%360)
        all_X = np.append(all_X,ev['Ulobes'].values/1e15)
        all_Y = np.append(all_Y,(ev['Ks2bl']+ev['M2b']).values/1e12)
    slope,intercept,r,p,stderr = linregress(all_X,all_Y)
    linelabel = (f'K2b Flux [TW]:{slope:.2f}Ulobes [PJ], r={r:.2f}')
    axis.plot(all_X,(intercept+all_X*slope),
                      label=linelabel,color='grey', ls='--')

    #cbar = plt.colorbar(sc)
    #cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    axis.legend()
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.set_xlabel(r'Integrated Lobe Energy $U \left[ PJ\right]$')
    axis.set_ylabel(r'Integrated Tail Flux K2b $\left[ TW\right]$')
    scatterK2b.tight_layout(pad=1)
    figurename = path+'/scatter_K2b.png'
    scatterK2b.savefig(figurename)
    plt.close(scatterK2b)
    print('\033[92m Created\033[00m',figurename)
    #############
    #setup figure
    scatterAL,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    all_X = np.array([])
    all_Y = np.array([])
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['Ulobes']/1e15,(ev['al']),
                          label=case,
                          marker=shapes[i],
                          s=200)
                          #cmap='twilight',
                          #c=(np.rad2deg(ev['clock'])+360)%360)
        all_X = np.append(all_X,ev['Ulobes'].values/1e15)
        all_Y = np.append(all_Y,(ev['al']).values/1e12)
    slope,intercept,r,p,stderr = linregress(all_X,all_Y)
    linelabel = (f'AL [nT]:{slope:.2f}Ulobes [PJ], r={r:.2f}')
    axis.plot(all_X,(intercept+all_X*slope),
                      label=linelabel,color='grey', ls='--')

    #cbar = plt.colorbar(sc)
    #cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    axis.legend()
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.set_xlabel(r'Integrated Lobe Energy $U \left[ PJ\right]$')
    axis.set_ylabel(r'AL $\left[ nT\right]$')
    scatterAL.tight_layout(pad=1)
    figurename = path+'/scatter_AL.png'
    scatterAL.savefig(figurename)
    plt.close(scatterAL)
    print('\033[92m Created\033[00m',figurename)

def scatter_externalFlux(tave):
    #############
    #setup figure
    scatterK1,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    all_X = np.array([])
    all_Y = np.array([])
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['U']/1e15,(ev['K1'])/1e12,
                          label=case,
                          marker=shapes[i],
                          s=200)
                          #cmap='twilight',
                          #c=(np.rad2deg(ev['clock'])+360)%360)
        all_X = np.append(all_X,ev['U'].values/1e15)
        all_Y = np.append(all_Y,ev['K1'].values/1e12)
    slope,intercept,r,p,stderr = linregress(all_X,all_Y)
    linelabel = (f'K1 Flux [TW]:{slope:.2f}U [PJ], r={r:.2f}')
    axis.plot(all_X,(intercept+all_X*slope),
                      label=linelabel,color='grey', ls='--')

    #cbar = plt.colorbar(sc)
    #cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    axis.legend()
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.set_xlabel(r'Integrated Energy $U \left[ PJ\right]$')
    axis.set_ylabel(r'Integrated LobeSheath Flux K1 $\left[ TW\right]$')
    scatterK1.tight_layout(pad=1)
    figurename = path+'/scatter_K1.png'
    scatterK1.savefig(figurename)
    plt.close(scatterK1)
    print('\033[92m Created\033[00m',figurename)
    #############
    #setup figure
    scatterK5,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    all_X = np.array([])
    all_Y = np.array([])
    shapes=['o','<','X','+','>','^']
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['U']/1e15,(ev['K5'])/1e12,
                          label=case,
                          marker=shapes[i],
                          s=200)
                          #cmap='twilight',
                          #c=(np.rad2deg(ev['clock'])+360)%360)
        all_X = np.append(all_X,ev['U'].values/1e15)
        all_Y = np.append(all_Y,ev['K5'].values/1e12)
    slope,intercept,r,p,stderr = linregress(all_X,all_Y)
    linelabel = (f'K5 Flux [TW]:{slope:.2f}U [PJ], r={r:.2f}')
    axis.plot(all_X,(intercept+all_X*slope),
                      label=linelabel,color='grey', ls='--')
    #cbar = plt.colorbar(sc)
    #cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    axis.legend()
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.set_xlabel(r'Integrated Energy $U \left[ PJ\right]$')
    axis.set_ylabel(r'Integrated ClosedSheath Flux K1 $\left[ TW\right]$')
    scatterK5.tight_layout(pad=1)
    figurename = path+'/scatter_K5.png'
    scatterK5.savefig(figurename)
    plt.close(scatterK5)
    print('\033[92m Created\033[00m',figurename)
    #############

def scatter_cuspFlux(tave):
    #############
    #setup figure
    scatterCusp,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['Ulobes']/1e15,(ev['Ks2ac']+ev['M2a'])/1e12,
                          label=case,
                          marker=shapes[i],
                          s=200)
                          #cmap='twilight',
                          #c=(np.rad2deg(ev['clock'])+360)%360)
    #cbar = plt.colorbar(sc)
    #cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    axis.legend()
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    axis.set_xlabel(r'Integrated Lobe Energy $U \left[ PJ\right]$')
    axis.set_ylabel(r'Integrated Cusp Flux 2a $\left[ TW\right]$')
    scatterCusp.tight_layout(pad=1)
    figurename = path+'/scatter_Cusp.png'
    scatterCusp.savefig(figurename)
    plt.close(scatterCusp)
    print('\033[92m Created\033[00m',figurename)
    #############

def innerLobeFlux(tave):
    #############
    #setup figure
    scatterLobe,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['U']/1e15,(ev['Ks3'])/1e12,label=case,
                          marker=shapes[i],
                          s=200,cmap='twilight',
                          c=(np.rad2deg(ev['clock'])+360)%360)
        '''
        sc = axis.scatter(ev['bz'],(ev['K5'])/1e12,label=case,marker='o',
                        s=200,cmap='plasma',
                          #ec=ev['B_T']*1e9,
                          ec='red',fc='none')
        '''
    cbar = plt.colorbar(sc)
    #cbar.set_label('Energy [J]')
    #cbar.set_label(r'$B_{T}\left[ nT\right]$')
    cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    #axis.set_xlim([0,360])
    #axis.set_ylim([-5,15])
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    #axis.set_xlabel(r'Integrated K1 Flux $\left[ TW\right]$')
    #axis.set_xlabel(r'IMF Clock$\left[ TW\right]$')
    axis.set_xlabel(r'Integrated Energy $U \left[ PJ\right]$')
    #axis.set_ylabel(r'-Integrated K1+K5 Flux $\left[ TW\right]$')
    #axis.set_ylabel(r'-Integrated K Flux $\left[ TW\right]$')
    axis.set_ylabel(r'Integrated Inner Lobe Flux K3 $\left[ TW\right]$')
    scatterLobe.tight_layout(pad=1)
    figurename = path+'/scatter_Lobe.png'
    scatterLobe.savefig(figurename)
    plt.close(scatterLobe)
    print('\033[92m Created\033[00m',figurename)
    #############

def scatters(tave):
    #############
    #setup figure
    scatter2,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    shapes=['o','<','X','+','>','^']
    for i,case in enumerate(tave.keys()):
        print(case, shapes[i])
        ev = tave[case]
        sc = axis.scatter(ev['U']/1e15,(ev['Ks7'])/1e12,label=case,
                          marker=shapes[i],
                          s=200,cmap='twilight',
                          c=(np.rad2deg(ev['clock'])+360)%360)
        '''
        sc = axis.scatter(ev['bz'],(ev['K5'])/1e12,label=case,marker='o',
                        s=200,cmap='plasma',
                          #ec=ev['B_T']*1e9,
                          ec='red',fc='none')
        '''
    cbar = plt.colorbar(sc)
    #cbar.set_label('Energy [J]')
    #cbar.set_label(r'$B_{T}\left[ nT\right]$')
    cbar.set_label(r' $\theta_{c}\left[ deg\right]$')
    #Decorations
    #axis.set_xlim([0,360])
    #axis.set_ylim([-5,15])
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    #axis.set_xlabel(r'Integrated K1 Flux $\left[ TW\right]$')
    #axis.set_xlabel(r'IMF Clock$\left[ TW\right]$')
    axis.set_xlabel(r'Integrated Energy $U \left[ PJ\right]$')
    #axis.set_ylabel(r'-Integrated K1+K5 Flux $\left[ TW\right]$')
    #axis.set_ylabel(r'-Integrated K Flux $\left[ TW\right]$')
    axis.set_ylabel(r'Integrated Inner Closed Flux K7 $\left[ TW\right]$')
    scatter2.tight_layout(pad=1)
    figurename = path+'/scatter2.png'
    scatter2.savefig(figurename)
    plt.close(scatter2)
    print('\033[92m Created\033[00m',figurename)
    #############
    #setup figure
    scatter3,(axis) =plt.subplots(1,1,figsize=[20,20])
    #Plot
    for case in tave.keys():
        print(case)
        print(np.min((np.rad2deg(ev['clock'])+360)%360))
        print(np.max((np.rad2deg(ev['clock'])+360)%360),'\n')
        ev = tave[case]
        sc = axis.scatter((np.rad2deg(ev['clock'])+360)%360,
                         (ev['Ks2ac']+ev['M2a'])/1e12,label=case,
                          s=200,cmap='plasma',c=ev['U']*1e9)
    cbar = plt.colorbar(sc)
    cbar.set_label('Energy [J]')
    #cbar.set_label(r'$B_{T}\left[ nT\right]$')
    #Decorations
    axis.set_xlim([0,360])
    #axis.set_ylim([-5,15])
    axis.axhline(0,c='black')
    axis.axvline(0,c='black')
    #axis.set_xlabel(r'Integrated K1 Flux $\left[ TW\right]$')
    axis.set_xlabel(r'IMF Clock$\left[ TW\right]$')
    axis.set_ylabel(r'Wang et. al 2014 $E_{in}\left[TW\right]$')
    #axis.set_ylabel(r'-Integrated K1+K5 Flux $\left[ TW\right]$')
    #axis.set_ylabel(r'-Integrated K Flux $\left[ TW\right]$')
    scatter3.tight_layout(pad=1)
    figurename = path+'/scatter3.png'
    scatter3.savefig(figurename)
    plt.close(scatter3)
    print('\033[92m Created\033[00m',figurename)
    #############


def tab_ranges(dataset):
    events = dataset.keys()
    results_min = {}
    results_max = {}
    swvals = ['pdyn','Beta','Beta*','Ma','r_shue98','B_T','EinWang','Pstorm']
    for val in swvals:
        results_min[val] = min([dataset[e]['obs']['swmf_sw'][val].min() for
                           e in events])
        results_max[val] = max([dataset[e]['obs']['swmf_sw'][val].max() for
                           e in events])
    mpvals = ['Utot [J]']
    for val in mpvals:
        results_min[val] = min([dataset[e]['mpdict']['ms_full'][val].min() for
                           e in events])
        results_max[val] = max([dataset[e]['mpdict']['ms_full'][val].max() for
                           e in events])

def initial_figures(dataset):
    path = unfiled
    tave = {}
    for i,event in enumerate(dataset.keys()):
        ev = refactor(dataset[event],dt.datetime(2022,6,6,0))
        tave[event] = interval_average(ev)
        if False:
            external_flux_timeseries(event,ev['times'],
                                      ev['HM1'],ev['Hs1'],ev['HM5'],ev['Hs5'],
                                      ev['Hs4'],ev['Hs6'],ev['Hs3'],ev['Hs7'],
                                      ev['SM1'],ev['Ss1'],ev['SM5'],ev['Ss5'],
                                      ev['Ss4'],ev['Ss6'],ev['Ss3'],ev['Ss7'],
                                      ev['M1'], ev['Ks1'], ev['M5'],ev['Ks5'],
                                      ev['Ks5'],ev['Ks6'],ev['Ks3'],ev['Ks7'],
                                 path)
        #explorer(event,ev,path)
        #common_x_bar(event,ev)
        #common_x_lineup(event,ev)
        #segments(event,ev)
        #series_segments(event,ev)
    test_matrix(event,ev,path)
    scatter_cuspFlux(tave)
    scatter_externalFlux(tave)
    #scatter_internalFlux(tave)
    energies(dataset,path)
    indices(dataset,path)
    couplers(dataset,path)
    scatters(tave)
    scatter_Ein_compare(tave)
    scatter_Pstorm(tave)
    innerLobeFlux(tave)
    tab_ranges(dataset)

if __name__ == "__main__":
    T0 = dt.datetime(2022,6,6,0,0)
    TSTART = dt.datetime(2022,6,6,0,59)
    DT = dt.timedelta(minutes=60)
    TJUMP = dt.timedelta(hours=2)
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inLogs = os.path.join(sys.argv[-1],'data/logs/')
    inAnalysis = os.path.join(sys.argv[-1],'data/analysis/')
    outPath = os.path.join(inBase,'figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    ## Analysis Data
    dataset = {}
    dataset['LOWnLOWu'] = load_hdf_sort(inAnalysis+'LOWnLOWu.h5')
    dataset['HIGHnHIGHu'] = load_hdf_sort(inAnalysis+'HIGHnHIGHu.h5')
    dataset['LOWnHIGHu'] = load_hdf_sort(inAnalysis+'LOWnHIGHu.h5')
    dataset['HIGHnLOWu'] = load_hdf_sort(inAnalysis+'HIGHnLOWu.h5')
    dataset['MEDnMEDu'] = load_hdf_sort(inAnalysis+'MEDnMEDu.h5')
    dataset['MEDnHIGHu'] = load_hdf_sort(inAnalysis+'MEDnHIGHu.h5')

    #HOTFIX duplicate LOWLOW values
    M5 = dataset['LOWnLOWu']['mpdict']['ms_full']['UtotM5 [W]'].copy()/1e12
    drops = np.where(abs(M5)>15)
    droptimes = dataset['LOWnLOWu']['time'][drops]
    dataset['LOWnLOWu']['time'] = dataset['LOWnLOWu']['time'].drop(droptimes)
    keylist = dataset['LOWnLOWu']['msdict'].keys()
    for key in keylist:
        df = dataset['LOWnLOWu']['msdict'][key]
        dataset['LOWnLOWu']['msdict'][key] = df.drop(index=droptimes)
    keylist = dataset['LOWnLOWu']['mpdict'].keys()
    for key in keylist:
        df = dataset['LOWnLOWu']['mpdict'][key]
        dataset['LOWnLOWu']['mpdict'][key] = df.drop(index=droptimes)

    ## Log Data
    dataset['LOWnLOWu']['obs'] = read_indices(inLogs, prefix='LOWnLOWu_',
                                              read_supermag=False)
    dataset['HIGHnHIGHu']['obs'] = read_indices(inLogs, prefix='HIGHnHIGHu_',
                                              read_supermag=False)
    dataset['LOWnHIGHu']['obs'] = read_indices(inLogs, prefix='LOWnHIGHu_',
                                              read_supermag=False)
    dataset['HIGHnLOWu']['obs'] = read_indices(inLogs, prefix='HIGHnLOWu_',
                                              read_supermag=False)
    dataset['MEDnMEDu']['obs'] = read_indices(inLogs, prefix='MEDnMEDu_',
                                              read_supermag=False)
    dataset['MEDnHIGHu']['obs'] = read_indices(inLogs, prefix='MEDnHIGHu_',
                                              read_supermag=False)
    ######################################################################
    ##Quicklook timeseries figures
    initial_figures(dataset)
