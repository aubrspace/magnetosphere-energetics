#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys,glob,time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
from scipy import signal
from scipy.stats import linregress
import datetime as dt
import pandas as pd
import matplotlib as mpl
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
                                                   refactor,ie_refactor,
                                                   gmiono_refactor)
from global_energetics.analysis.proc_hdf import (load_hdf_sort)
from global_energetics.analysis.proc_indices import read_indices,ID_ALbays
from global_energetics.analysis.analyze_energetics import plot_power
from global_energetics.analysis.proc_timing import (peak2peak,
                                                    pearson_r_shifts)
from global_energetics.analysis.workingtitle import (
                                                     stack_energy_region_fig,
                                                     lobe_balance_fig,
                                                     solarwind_figure)

def rxn(ev,event,path,**kwargs):
    #interval_list = build_interval_list(TSTART,DT,TJUMP,
    #                                    ev['mp'].index)
    if 'zoom' in kwargs:
        zoom = kwargs.get('zoom')
        window = [float(pd.Timedelta(n-TSTART).to_numpy()) for n in zoom]
    else:
        window = None
    #############
    #setup figure
    RXN,(daynight) = plt.subplots(1,1,figsize=[24,8],
                                                    sharex=True)
    Bflux,(bflux,dphidt) = plt.subplots(2,1,figsize=[24,20],
                                                    sharex=True)
    # Get time markings that match the format
    #Plot
    # Dayside/nightside reconnection rates
    daynight.axhline(0,c='black')
    daynight.fill_between(ev['ie_times'],(ev['cdiffRXN']/1e3),label='cdiff',
                          fc='grey')
    daynight.plot(ev['ie_times'],ev['RXNm_Day']/1e3,label='Day',
                  c='dodgerblue',lw=3)
    daynight.plot(ev['ie_times'],ev['RXNm_Night']/1e3,label='Night',c='navy')
    daynight.plot(ev['ie_times'],ev['RXNs']/1e3,label='Static',c='red',ls='--')
    daynight.plot(ev['ie_times'],ev['RXN']/1e3,label='Net',c='gold',ls='--')

    bflux.plot(ev['ie_times'],ev['ie_flux']/1e9,label='1s',c='grey')
    bflux.plot(ev['ie_times'],ev['ie_flux'].rolling('3s').mean()/1e9,
                                                        label='3s',c='gold')
    bflux.plot(ev['ie_times'],ev['ie_flux'].rolling('10s').mean()/1e9,
                                                        label='10s',c='brown')
    bflux.plot(ev['ie_times'],ev['ie_flux'].rolling('30s').mean()/1e9,
                                                        label='30s',c='blue')
    bflux.plot(ev['ie_times'],ev['ie_flux'].rolling('60s').mean()/1e9,
                                                        label='60s',c='black')
    dphidt.plot(ev['ie_times'],central_diff(ev['ie_flux'])/1e3,label='1s'
                                                                    ,c='grey')
    dphidt.plot(ev['ie_times'],central_diff(
                                      ev['ie_flux'].rolling('3s').mean())/1e3,
                                                          label='3s',c='gold')
    dphidt.plot(ev['ie_times'],central_diff(
                                      ev['ie_flux'].rolling('10s').mean())/1e3,
                                                        label='10s',c='brown')
    dphidt.plot(ev['ie_times'],central_diff(
                                      ev['ie_flux'].rolling('30s').mean())/1e3,
                                                         label='30s',c='blue')
    dphidt.plot(ev['ie_times'],central_diff(
                                      ev['ie_flux'].rolling('60s').mean())/1e3,
                                                        label='60s',c='black')
    #Decorations
    general_plot_settings(daynight,do_xlabel=True,legend=True,
                          ylabel=r'$d\phi/dt \left[kV\right]$',
                          xlim = window,
                          timedelta=True)
    general_plot_settings(bflux,do_xlabel=False,legend=True,
                          ylabel=r'$\phi \left[GW\right]$',
                          ylim=[2.13,2.18],
                          xlim = window,
                          timedelta=True)
    general_plot_settings(dphidt,do_xlabel=True,legend=True,
                          ylabel=r'$d\phi/dt \left[kV\right]$',
                          #ylim=[2.13,2.18],
                          xlim = window,
                          timedelta=True)
    if 'zoom' in kwargs:
        '''
        daynight.set_ylim([ev['RXN'][(ev['RXN'].index<zoom[1])&
                                     (ev['RXN'].index>zoom[0])].min()/1e3,
                           ev['RXN'][(ev['RXN'].index<zoom[1])&
                                     (ev['RXN'].index>zoom[0])].max()/1e3])
        '''
        daynight.set_ylim([-500,500])
    for interv in interval_list:
        daynight.axvline(float(pd.Timedelta(interv[0]-TSTART).to_numpy()),
                          c='grey')
    dphidt.axhline(0,c='lightgrey',lw=1)
    daynight.margins(x=0.01)
    bflux.margins(x=0.01)
    dphidt.margins(x=0.01)
    RXN.tight_layout()
    Bflux.tight_layout()
    #Save
    if 'zoom' in kwargs:
        figurename = (path+'/RXN_'+event+'_'+
                      kwargs.get('tag','zoomed')+'.png')
        figurename2 = (path+'/Bflux_'+event+'_'+
                      kwargs.get('tag','zoomed')+'.png')
    else:
        figurename = path+'/RXN_'+event+'.png'
        figurename2 = path+'/Bflux_'+event+'.png'
    # Save in pieces
    RXN.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(RXN)

    Bflux.savefig(figurename2)
    print('\033[92m Created\033[00m',figurename2)
    plt.close(Bflux)

def summary_plot(ev,dataset,event,path,**kwargs):
    #interval_list = build_interval_list(TSTART,DT,TJUMP,
    #                                    ev['mp'].index)
    #############
    #setup figure
    Summary = plt.figure(figsize=(28,22))#,layout="constrained")
    horizChunk = plt.GridSpec(1,2,hspace=0.1,top=0.95,figure=Summary,
                                width_ratios=[5,1])
    leftstacks = horizChunk[0].subgridspec(3,1,hspace=0.05)
    rightstacks = horizChunk[1].subgridspec(1,1,hspace=0.05)
    # Axes
    driving = Summary.add_subplot(leftstacks[0,:])
    ms_state = Summary.add_subplot(leftstacks[1,:])
    coupling = Summary.add_subplot(leftstacks[2,:])
    # Tighten up the window
    #al_values = dataset[event]['obs']['swmf_index']['AL']
    dst_values = dataset[event]['obs']['swmf_log']['dst_sm']
    Ein_values = dataset[event]['obs']['swmf_sw']['EinWang']
    Pstorm_values = dataset[event]['obs']['swmf_sw']['Pstorm']
    if 'zoom' in kwargs:
        #zoom
        zoom = kwargs.get('zoom')
        mp_window = ((ev['mp'].index>zoom[0])&(ev['mp'].index<zoom[1]))
        '''
        index_window = (
                (dataset[event]['obs']['swmf_index']['AL'].index>zoom[0])&
                (dataset[event]['obs']['swmf_index']['AL'].index<zoom[1]))
        log_window = (
                (dataset[event]['obs']['swmf_log']['dst_sm'].index>zoom[0])&
                (dataset[event]['obs']['swmf_log']['dst_sm'].index<zoom[1]))
        sw_window = (
                (dataset[event]['obs']['swmf_sw']['EinWang'].index>zoom[0])&
                (dataset[event]['obs']['swmf_sw']['EinWang'].index<zoom[1]))
        zoomed = {}
        for key in ev.keys():
            if len(mp_window)==len(ev[key]):
                zoomed[key] = np.array(ev[key])[mp_window]
        ev = zoomed
        al_values = dataset[event]['obs']['swmf_index']['AL'][index_window]
        dst_values = dataset[event]['obs']['swmf_log']['dst_sm'][log_window]
        Ein_values = dataset[event]['obs']['swmf_sw']['EinWang']
        Pstorm_values = dataset[event]['obs']['swmf_sw']['Pstorm']
        '''
    # Driving as measured by coupling function
    driving.fill_between(Ein_values.index,Ein_values/1e12,label='Wang2014',
                         fc='orange')
    driving.plot(Pstorm_values.index,-Pstorm_values/1e12,
                 label='Tenfjord and Østgaard 2013',ls='--',c='black')
    # Magnetospheric state as measured by Energy and Dst
    ms_state.fill_between(ev['rawtimes'],-ev['U']/1e15,
                            label=r'-$\int_VU\left[\frac{J}{m^3}\right]$',
                            fc='blue',alpha=0.4)
    rax = ms_state.twinx()
    rax.plot(dst_values.index,dst_values,label='Dst (simulated)',color='black')

    # Coupling measured as K1,5,4,6 fluxes
    coupling.fill_between(ev['mp'].index,ev['K1']/1e12,
                          label='K1 (Open Magnetopause)',fc='blue')
    coupling.fill_between(ev['mp'].index,ev['K5']/1e12,
                          label='K5 (Closed Magnetopause)',fc='red')
    coupling.plot(ev['mp'].index,ev['Ks4']/1e12,
                  label='K4 (Open cuttoff)',c='black')
    coupling.plot(ev['mp'].index,ev['Ks6']/1e12,
                  label='K6 (Closed cuttoff)',c='grey')
    # AL and GridL
    #al.plot(al_values.index,al_values,label='AL (simulated)',color='black')

    #Decorations
    general_plot_settings(driving,do_xlabel=False,legend=True,
                          ylabel=r' $\int_S$Energy Flux $\left[TW\right]$',
                          ylim=[0,3],
                          xlim=kwargs.get('zoom'),
                          timedelta=False,legend_loc='lower left')
    general_plot_settings(ms_state,do_xlabel=False,legend=True,
                          ylabel=r'Energy $\left[PJ\right]$',
                          xlim=kwargs.get('zoom'),
                          #ylim=[-3,3],
                          timedelta=False,legend_loc='lower left')
    general_plot_settings(coupling,do_xlabel=True,legend=True,
              ylabel=r'$\int_{MP}\mathbf{K}\cdot d\mathbf{A}\left[TW\right]$',
                          xlim=kwargs.get('zoom'),
                          ylim=[-3,3],
                          timedelta=False,legend_loc='lower left')
    driving.xaxis.set_ticklabels([])
    ms_state.xaxis.set_ticklabels([])
    coupling.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    coupling.set_xlabel('Time [hr:min]')
    rax.set_ylabel(r'$\Delta B \left[nT\right]$')
    rax.legend(loc='lower right')
    rax.spines
    ms_state.spines['left'].set_color('slateblue')
    ms_state.tick_params(axis='y',colors='slateblue')
    ms_state.yaxis.label.set_color('slateblue')
    if 'zoom' in kwargs:
        rax.set_ylim([dst_values.min(),0])
        pass
    ms_state.axhline(0,c='black')
    driving.margins(x=0.01)
    ms_state.margins(x=0.01)
    Summary.tight_layout()
    #Save
    if 'zoom' in kwargs:
        figurename = (path+'/Summary'+event+'_'+
                      kwargs.get('tag','zoomed')+'.svg')
    else:
        figurename = path+'/Summary_'+event+'.svg'
    # Save in pieces
    Summary.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)

def create_figures(dataset):
    path = unfiled
    # Zoom in
    crossing_window = (dt.datetime(2022,2,2,23,0)-dt.timedelta(hours=2),
                       dt.datetime(2022,2,2,23,50)+dt.timedelta(hours=2))
    # Process data
    ev = refactor(dataset['star4'],dt.datetime(2022,2,2,23,0))
    # Create figures
    summary_plot(ev,dataset,'star4',path,zoom=crossing_window,tag='crossing')

if __name__ == "__main__":
    #Need input path, then create output dir's
    inBase = 'magEx/'
    inLogs = os.path.join(inBase,'data/starlink/logs/')
    inAnalysis = os.path.join(inBase,'data/starlink/analysis/')
    outPath = os.path.join(inBase,'figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    ## Read in data
    data = {}
    GMfile = os.path.join(inAnalysis,'starlink2_results4Re.h5')
    if os.path.exists(GMfile):
        data['star4'] = load_hdf_sort(GMfile)
    else:
        print('couldnt find data!')
        exit

    # Log Data
    prefix = GMfile.split('_')[1]+'_'
    data['star4']['obs'] = read_indices(inLogs,prefix='starlink_',
                                        start=data['star4']['time'][0],
                 end=data['star4']['time'][-1]+dt.timedelta(seconds=1),
                                             read_supermag=False)
    ## Make plots
    create_figures(data)
