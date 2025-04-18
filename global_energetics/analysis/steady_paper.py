#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys
import numpy as np
from numpy import abs
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
#interpackage imports
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings,
                                                   bin_and_describe,
                                                   extended_fill_between,
                                                   refactor)
from global_energetics.analysis.proc_hdf import (load_hdf_sort)
from global_energetics.analysis.proc_indices import read_indices,ID_ALbays



def build_interval_list(tstart,tlength,tjump,alltimes):
    interval_list = []
    tend = tstart+tlength
    while tstart<alltimes[-1]:
        interval_list.append([tstart,tend])
        tstart+=tjump
        tend+=tjump
    return interval_list


def test_matrix(event,ev,path):
    timedelta = [t-T0 for t in dataset[event]['obs']['swmf_log'].index]
    times = [float(n.to_numpy()) for n in timedelta]
    interval_list =build_interval_list(TSTART,DT,TJUMP,dataset[event]['time'])
    #############
    #setup figure
    matrix,(axis) =plt.subplots(1,1,figsize=[20,8],sharex=True)
    axis.plot(ev['swt'],ev['sw']['bz'],label=r'$B_z$',color='blue',lw=4)
    axis.plot(ev['swt'],ev['sw']['by'],label=r'$B_y$',color='magenta')
    axis.fill_between(ev['swt'],np.sqrt(ev['sw']['by']**2+ev['sw']['bz']**2),
                                        fc='black',label=r'B',alpha=0.4)
    for interv in interval_list:
        axis.axvline((interv[0]-T0).total_seconds()*1e9,c='grey',alpha=0.2)
    general_plot_settings(axis,do_xlabel=True,legend=True,
                          legend_loc='upper left',
                          ylabel=r'IMF $\left[nT\right]$',timedelta=True)
    axis.margins(x=0.01)
    matrix.tight_layout(pad=1)
    figurename = path+'/matrix'+event+'.svg'
    matrix.savefig(figurename)
    plt.close(matrix)
    print('\033[92m Created\033[00m',figurename)

def plot_indices(dataset,path):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                       dataset['stretched_MEDnMEDu']['time'])
    colors = ['#80b3ffff','#0066ffff','#0044aaff',
              '#80ffb3ff','#00aa44ff','#005500ff',
              '#ffe680ff','#ffcc00ff','#806600ff']
    styles = [None,None,None,
              '--','--','--',
              None,None,None]
    markers = [None,None,None,
               None,None,None,
               'x','x','x']
    testpoints = ['stretched_LOWnLOWu',
                  'stretched_MEDnLOWu',
                  'stretched_HIGHnLOWu',
                  'stretched_LOWnMEDu',
                  'stretched_MEDnMEDu',
                  'stretched_HIGHnMEDu',
                  'stretched_LOWnHIGHu',
                  'stretched_MEDnHIGHu',
                  'stretched_HIGHnHIGHu']
    #############
    #setup figure
    indices,(axis1,axis2,axis3,axis4) =plt.subplots(4,1,figsize=[20,20],
                                                    sharex=True)
    for i,event in enumerate(testpoints):
        if event not in dataset.keys():
            continue
        # Time shenanigans
        timedelta = [t-T0 for t in dataset[event]['obs']['swmf_log'].index]
        times = [float(n.to_numpy()) for n in timedelta]
        supdelta = [t-T0 for t in dataset[event]['obs']['super_log'].index]
        suptimes = [float(n.to_numpy()) for n in supdelta]
        griddelta = [t-T0 for t in dataset[event]['obs']['gridMin'].index]
        gridtimes = [float(n.to_numpy()) for n in griddelta]
        inddelta = [t-T0 for t in dataset[event]['obs']['swmf_index'].index]
        indtimes = [float(n.to_numpy()) for n in inddelta]
        utimedelta = [t-T0 for t in dataset[event]['mpdict']['ms_full'].index]
        utimes = [float(n.to_numpy()) for n in utimedelta]

        # Plot
        axis1.plot(utimes,
                   dataset[event]['mpdict']['ms_full']['Utot [J]']/1e15,
                   label=event,c=colors[i],lw=3,ls=styles[i],marker=markers[i],
                   markevery=10)
        axis2.plot(times,
                  dataset[event]['obs']['swmf_log']['dst_sm'],
                  label=event,c=colors[i],lw=3,ls=styles[i],marker=markers[i],
                  markevery=50)
        axis3.plot(gridtimes,dataset[event]['obs']['gridMin']['dBmin'],
                   label=event,c=colors[i],ls=styles[i],
                   marker=markers[i],markevery=10)
        axis4.plot(times,dataset[event]['obs']['swmf_log']['cpcpn'],
                   label=event,c=colors[i],ls=styles[i],
                   marker=markers[i],markevery=50)
    for interv in interval_list:
        axis1.axvline((interv[0]-T0).total_seconds()*1e9,c='grey')
        axis2.axvline((interv[0]-T0).total_seconds()*1e9,c='grey')
        axis3.axvline((interv[0]-T0).total_seconds()*1e9,c='grey')
        axis4.axvline((interv[0]-T0).total_seconds()*1e9,c='grey')

    general_plot_settings(axis1,do_xlabel=False,legend=False,
                          ylabel=r'Energy $\left[PJ\right]$',timedelta=True)
    general_plot_settings(axis2,do_xlabel=False,legend=False,
                          ylabel=r'Dst $\left[nT\right]$',timedelta=True)
    general_plot_settings(axis3,do_xlabel=False,legend=False,
                          ylabel=r'GridL $\left[nT\right]$',timedelta=True)
    general_plot_settings(axis4,do_xlabel=True,legend=False,
                        ylabel=r'CPCP North $\left[kV\right]$',timedelta=True)
    axis1.margins(x=0.01)
    axis2.margins(x=0.01)
    axis3.margins(x=0.01)
    indices.tight_layout(pad=0.5)
    figurename = path+'/indices.svg'
    indices.savefig(figurename)
    plt.close(indices)
    print('\033[92m Created\033[00m',figurename)

def plot_2by2_flux(dataset,windows,path):
    interval_list =build_interval_list(TSTART,DT,TJUMP,
                                       dataset['stretched_MEDnHIGHu']['time'])
    #############
    #setup figure
    Fluxes,ax = plt.subplots(2,len(windows),figsize=[32,22])
    #for i,run in enumerate(['stretched_MEDnHIGHu','stretched_HIGHnHIGHu']):
    ylims = [[-6,5],[-20,15]]
    for i,run in enumerate(['stretched_LOWnLOWu','stretched_HIGHnHIGHu']):
        ev = refactor(dataset[run],T0)
        for j,window in enumerate(windows):
            #xlims = [float(pd.Timedelta(t-T0).to_numpy()) for t in window]
            xlims = [(t-T0).total_seconds()*1e9 for t in window]
            # External Energy Flux
            ax[i][j].fill_between(ev['times'],ev['K1']/1e12,label='K1 (OpenMP)',
                                 fc='blue')
            ax[i][j].fill_between(ev['times'],ev['K5']/1e12,
                                 label='K5 (ClosedMP)',fc='red')
            ax[i][j].fill_between(ev['times'],(ev['K1']+ev['K5'])/1e12,
                           label=r'$K1+K5$',fc='black',alpha=0.9)
            ax[i][j].plot(ev['times'],ev['Ksum']/1e12,
                                  label=r'$K_{net}$',c='lightgrey',lw=3)
            ax[i][j].plot(ev['swt'],-ev['sw']['EinWang']/1e12,
                     c='black',lw=4,label='Wang2014')
            ax[i][j].plot(ev['swt'],ev['sw']['Pstorm']/1e12,
                     c='grey',lw=4,label='Tenfjord and Østgaard 2013')
            # Decorate
            general_plot_settings(ax[i][j],do_xlabel=i==1,legend=False,
                          ylabel=r' $\int_S\mathbf{K}\left[TW\right]$',
                          #ylim=[-25,17],
                          ylim=ylims[i],
                          xlim=xlims,
                          timedelta=True,legend_loc='lower left')
            if j!=0: ax[i][j].set_ylabel(None)
            for interv in interval_list:
                ax[i][j].axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
            ax[i][j].margins(x=0.01)
    ax[0][2].legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=6, fancybox=True, shadow=True)
    Fluxes.tight_layout()
    #Save
    figurename = path+'/Fluxes_2by3.svg'
    Fluxes.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)

def all_fluxes(ev,event,path,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                        ev['mp'].index)
    #############
    #setup figure
    Fluxes,(Kexternal,al,cpcp,Energy_dst) = plt.subplots(4,1,
                                                            figsize=[24,24],
                                                            sharex=True)
    test1,(Kinternal) = plt.subplots(1,1,figsize=[24,16],sharex=True)
    # Tighten up the window
    al_values = dataset[event]['obs']['swmf_index']['AL']
    sml_values = dataset[event]['obs']['super_log']['SML']
    dst_values = dataset[event]['obs']['swmf_log']['dst_sm']
    if 'zoom' in kwargs:
        #xlims = [float(pd.Timedelta(t-T0).to_numpy())
        #         for t in kwargs.get('zoom')]
        xlims = [(t-T0).total_seconds()*1e9 for t in kwargs.get('zoom')]
    else:
        #xlims = [float(pd.Timedelta(t-T0).to_numpy())
        #         for t in [TSTART,TEND]]
        xlims = [(t-T0).total_seconds()*1e9 for t in [TSTART,TEND]]
    # Get time markings that match the format
    inddelta = [t-T0 for t in sml_values.index]
    indtimes = [float(n.to_numpy()) for n in inddelta]
    logdelta = [t-T0 for t in dst_values.index]
    logtimes = [float(n.to_numpy()) for n in logdelta]
    #Plot
    # External Energy Flux
    Kexternal.fill_between(ev['times'],ev['K1']/1e12,label='K1 (OpenMP)',
                   fc='blue')
    Kexternal.fill_between(ev['times'],ev['K5']/1e12,label='K5 (ClosedMP)',
                   fc='red')
    Kexternal.plot(ev['times'],ev['Ksum']/1e12,label=r'$K_{net}$',
                   c='lightgrey')
    Kexternal.fill_between(ev['times'],(ev['K1']+ev['K5'])/1e12,
                           label=r'$K1+K5$',fc='black',alpha=0.9)
    Kexternal.plot(ev['swt'],-ev['sw']['EinWang']/1e12,
                   c='black',lw=4,label='Wang2014')
    Kexternal.plot(ev['swt'],ev['sw']['Pstorm']/1e12,
                   c='grey',lw=4,label='Tenfjord and Østgaard 2013')
    # Internal Energy Flux
    Kinternal.fill_between(ev['times'],ev['K2a']/1e12,label='K2a (Cusp)',
                           fc='magenta')
    Kinternal.fill_between(ev['times'],ev['K2b']/1e12,
                           label='K2b (Plasmasheet)',fc='dodgerblue')
    Kinternal.plot(ev['times'],ev['Ks3']/1e12,label='K3 (OpenIB)',
                   color='darkslategrey')
    Kinternal.plot(ev['times'],ev['Ks7']/1e12,label='K7 (ClosedIB)',
                   color='brown')
    # Energy and Dst
    Energy_dst.fill_between(ev['times'],-ev['U']/1e15,-1*ev['U'].min()/1e15,
                            label=r'-$\int_VU\left[\frac{J}{m^3}\right]$',
                            fc='blue',alpha=0.4)
    rax = Energy_dst.twinx()
    rax.plot(logtimes,dst_values,label='Dst (simulated)',color='black')
    # SML and GridL
    colors=['grey','goldenrod','red','blue']
    for i,source in enumerate(['dBMhd','dBFac','dBPed','dBHal']):
       al.plot(ev['times'],ev['maggrid'][source],label=source,c=colors[i])
    al.plot(ev['times'],ev['GridL'],label='Total',color='black')
    # Cross Polar Cap Potential
    log = dataset[event]['obs']['swmf_log']
    logt = [float((t-T0).to_numpy()) for t in log.index]
    cpcp.plot(logt,log['cpcpn'],label='North',c='orange')
    cpcp.plot(logt,log['cpcps'],label='South',c='blue')
    #Decorations
    general_plot_settings(Kexternal,do_xlabel=False,legend=True,
                          ylabel=r' $\int_S\mathbf{K}\left[TW\right]$',
                          ylim=[ev['K1'].quantile(0.01)/1e12*1.1,
                                ev['K5'].quantile(0.99)/1e12*1.1],
                          xlim=xlims,
                          timedelta=True,legend_loc='lower left')
    Kexternal.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                     ncol=3, fancybox=True, shadow=True)
    general_plot_settings(Kinternal,do_xlabel=True,legend=True,
                          ylabel=r' $\int_S\mathbf{K}\left[TW\right]$',
                          xlim=xlims,
                          timedelta=True,legend_loc='lower left')
    Kinternal.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                     ncol=3, fancybox=True, shadow=True)
    general_plot_settings(al,do_xlabel=False,legend=True,xlim=xlims,
                          ylabel=r'$\Delta B_{N} \left[nT\right]$',
                          timedelta=True,legend_loc='lower left')
    general_plot_settings(cpcp,do_xlabel=False,legend=True,xlim=xlims,
                          #ylim=[0,120],
                          ylabel=r'CPCP $\left[kV\right]$',
                          timedelta=True,legend_loc='upper left')
    general_plot_settings(Energy_dst,do_xlabel=True,legend=True,
                          ylabel=r'$\int_V U\left[PJ\right]$',xlim=xlims,
                          timedelta=True,legend_loc='lower left')
    rax.set_ylabel(r'$\Delta B \left[nT\right]$')
    rax.legend(loc='lower right')
    rax.spines
    Energy_dst.spines['left'].set_color('slateblue')
    Energy_dst.tick_params(axis='y',colors='slateblue')
    Energy_dst.yaxis.label.set_color('slateblue')
    if 'zoom' in kwargs:
        rax.set_ylim([dst_values.min(),0])
        pass
    for interv in interval_list:
        Kexternal.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        Kinternal.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        al.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        cpcp.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        Energy_dst.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
    Kexternal.axhline(0,c='black')
    Kinternal.axhline(0,c='black')
    Energy_dst.axhline(0,c='black')
    al.axhline(0,c='black')
    Kexternal.margins(x=0.01)
    Kinternal.margins(x=0.01)
    Energy_dst.margins(x=0.01)
    al.margins(x=0.01)
    Fluxes.tight_layout()
    test1.tight_layout()
    #Save
    if 'zoom' in kwargs:
        figurename = (path+'/Fluxes_'+event+'_'+
                      kwargs.get('tag','zoomed')+'.pdf')
    else:
        figurename = path+'/Fluxes_'+event+'.pdf'
    # Save in pieces
    Fluxes.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(Fluxes)

    figurename = path+'/Internal_'+event+'.pdf'
    test1.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(test1)

def build_events(ev,run,**kwargs):
    events = pd.DataFrame()
    ## ID types of events
    # Construction (test matrix) signatures
    events['imf_transients'] = ID_imftransient(ev,run,**kwargs)
    # Internal process signatures
    events['DIPx'],events['DIPb'],events['DIP']=ID_dipolarizations(ev,**kwargs)
    events['plasmoids_mass'] = ID_plasmoids(ev,mode='mass')
    events['ALbays'],events['ALonsets'],events['ALpsuedos'],_ = ID_ALbays(ev)
    events['MGLbays'],events['MGLonsets'],events['MGLpsuedos'],_=ID_ALbays(ev,
                                                              al_series='MGL')
    # ID MPBs
    events['substorm'] = np.ceil((events['DIPb']+
                                  events['plasmoids_mass']+
                                  events['MGLbays'])
                                  /3)
    events['allsubstorm'] = np.floor((
                                  events['DIPb']+
                                  events['plasmoids_mass']+
                                  events['MGLbays'])
                                  /3)
    # Coupling variability signatures
    events['K1var'],events['K1unsteady'],events['K1err']=ID_variability(
                                                                    ev['K1'])
    events['K1var_5-5'],_,__ = ID_variability(ev['K1'],relative=False)
    events['clean']=np.ceil((1+events['substorm']-events['imf_transients'])/2)
    return events

def ID_imftransient(ev,run,**kwargs):
    """Function that puts a flag for when the imf change is affecting the
        integrated results
    Inputs
        ev
        window (float) - default 30min
        kwargs:
            None
    """
    # If we assume the X length of the simulation domain
    simX = kwargs.get('xmax',32)-kwargs.get('xmin',-150)
    # X velocity denoted by the event name
    vx_dict = {'HIGH':800,'MED':600,'LOW':400}
    vx_str = run.split('n')[1].split('u')[0]
    # Simple balistics
    window = simX*6371/vx_dict[vx_str]/60
    # find the times when imf change has occured
    times = ev['mp'].index
    intervals = build_interval_list(TSTART,dt.timedelta(minutes=window),TJUMP,
                                    times)
    #   add time lag from kwargs
    imftransients = np.zeros(len(times))
    for start,end in intervals:
        # mark all these as "change times"
        imftransients[(times>start) & (times<end)] = 1
    return imftransients

def ID_variability(in_signal,**kwargs):
    dt = [t.seconds for t in in_signal.index[1::]-in_signal.index[0:-1]]
    signal = in_signal.copy(deep=True).reset_index(drop=True)
    lookahead = kwargs.get('lookahead',5)
    lookbehind = kwargs.get('lookbehind',5)
    threshold = kwargs.get('threshold',51.4)
    unsteady = np.zeros(len(signal))
    var = np.zeros(len(signal))
    relvar = np.zeros(len(signal))
    integ_true = np.zeros(len(signal))
    integ_steady = np.zeros(len(signal))
    integ_err = np.zeros(len(signal))
    for i in range(lookbehind+1,len(signal)-lookahead-1):
        # Measure the variability within a given window for each point
        var[i] = np.sum([abs(signal[k+1]-signal[k-1])/(2) for k in
                                        range(i-lookbehind,i+lookahead+1)])
        relvar[i] = var[i]/abs(signal[i])*100
        # Assign 1/0 to if the window variability is > threshold
        unsteady[i] = var[i]>abs(signal[i])*(threshold/100)
        # Calculate the error in window integrated value vs assuming steady
        integ_true[i] = np.sum([signal[k] for k in
                            range(i-lookbehind,i+lookahead+1)])
        integ_steady[i] = len(range(i-lookbehind,i+lookahead+1))*signal[i]
        integ_err[i] = (integ_steady[i]-integ_true[i])/integ_true[i]*100
    #from IPython import embed; embed()
    if kwargs.get('relative',True):
        return relvar,unsteady,integ_err
    else:
        return var,unsteady,integ_err

def ID_plasmoids(ev,**kwargs):
    """Function finds plasmoid release events in timeseries
    Inputs
        ev
        kwargs:
            plasmoid_lookahead (int)- number of points to look ahead
            volume_reduction (float)-
            mass_reduction (float)-
    Returns
        list(bools)
    """
    #ibuffer = kwargs.get('plasmoid_buffer',4)
    lookahead = kwargs.get('plasmoid_ahead',10)
    volume = ev['closed']['Volume [Re^3]'].copy(deep=True).reset_index(
                                                                    drop=True)
    mass = ev['closed']['M_night [kg]'].copy(deep=True).reset_index(drop=True)
    R1 = kwargs.get('volume_reduction',5)
    R2 = kwargs.get('mass_reduction',12e3)
    # 1st measure: closed_nightside volume timeseries
    # R1 reduction in volume
    volume_reduction = np.zeros([len(volume)])
    if 'volume' in kwargs.get('mode','volume_mass'):
        for i,testpoint in enumerate(volume.values[0:-10]):
            v = volume[i]
            # Skip point if we've already got it or the volume isnt shrinking
            if not volume_reduction[i] and volume[i+1]<volume[i]:
                # Check xmin loc against lookahead points
                if (v-volume[i:i+lookahead].min())>(R1):
                    # If we've got one need to see how far it extends
                    i_min = volume[i:i+lookahead].idxmin()
                    volume_reduction[i:i_min+1] = 1
                    # If it's at the end of the window, check if its continuing
                    if i_min==i+lookahead-1:
                        found_min = False
                    else:
                        found_min = True
                    while not found_min:
                        if volume[i_min+1]<volume[i_min]:
                            volume_reduction[i_min+1] = 1
                            i_min +=1
                            if i_min==len(volume)-2:
                                found_min = True
                        else:
                            found_min = True
    # 2nd measure: volume integral of density (mass) on nightside timeseries
    # R2 reduction in mass
    mass_reduction = np.zeros([len(mass)])
    if 'mass' in kwargs.get('mode','volume_mass'):
        for i,testpoint in enumerate(mass.values[0:-10]):
            m = mass[i]
            # Skip point if we've already got it or the mass isnt shrinking
            if not mass_reduction[i] and mass[i+1]<mass[i]:
                # Check xmin loc against lookahead points
                if (m-mass[i:i+lookahead].min())>(R2):
                    # If we've got one need to see how far it extends
                    i_min = mass[i:i+lookahead].idxmin()
                    mass_reduction[i:i_min+1] = 1
                    # If it's at the end of the window, check if its continuing
                    if i_min==i+lookahead-1:
                        found_min = False
                    else:
                        found_min = True
                    while not found_min:
                        if mass[i_min+1]<mass[i_min]:
                            mass_reduction[i_min+1] = 1
                            i_min +=1
                            if i_min==len(mass)-2:
                                found_min = True
                        else:
                            found_min = True
    return [y1 or y2 for y1,y2 in zip(volume_reduction,mass_reduction)]

def ID_dipolarizations(ev,**kwargs):
    """Function finds dipolarizations in timeseries
    Inputs
        ev
        kwargs:
            DIP_ahead (int)- number of points to look ahead
            xline_reduction_percent (float)-
            B1_reduction_percent (float)-
    Returns
        list(bools)
        list(bools)
        list(bools)
    """
    ibuffer = kwargs.get('DIP_buffer',0)#NOTE too complicated...
    lookahead = kwargs.get('DIP_ahead',10)
    xline = ev['mp']['X_NEXL [Re]'].copy(deep=True).reset_index(drop=True)
    R1 = kwargs.get('xline_reduction_percent',15)
    R2 = kwargs.get('B1_reduction_amount',0.3e15)
    # 1st measure: closed[X].min() timeseries
    # R1% reduction in -X_min distance at ANY point in the interval
    xline_reduction = np.zeros([len(xline)])
    for i,testpoint in enumerate(xline.values[0:-10]):
        x = xline[i]
        # Skip point if we've already got it and the distance isnt moving in
        if not xline_reduction[i] and xline[i+1]>xline[i]:
            # Check xmin loc against lookahead points
            if (-x+xline[i:i+lookahead].max())/(-x)>(R1/100):
                # If we've got one need to see how far it extends
                i_max = xline[i:i+lookahead].idxmax()
                xline_reduction[i:i_max+1] = 1
                # If it's at the end of the window, check if its continuing
                if i_max==i+lookahead-1:
                    found_max = False
                else:
                    found_max = True
                while not found_max:
                    if xline[i_max+1]>xline[i_max]:
                        xline_reduction[i_max+1] = 1
                        i_max +=1
                        if i_max==len(xline):
                            found_max = True
                    else:
                        found_max = True
    # 2nd measure: volume_int{(B-Bdipole)} on nightside timeseries
    # R2 reduction in volume_int{B-Bd} at ANY point in the interval
    b1_reduction = np.zeros([len(xline)])
    udb = ev['closed']['u_db_night [J]'].copy(deep=True).reset_index(drop=True)
    for i,testpoint in enumerate(xline.values[ibuffer:-10]):
        b1 = udb[i]
        # Skip point if we've already got it and the distance isnt decreasing
        if not b1_reduction[i] and udb[i+1]>udb[i]:
            # Check U_dB against lookahead points
            i_min = udb[i+ibuffer:i+lookahead].idxmin()
            if np.isnan(i_min):
                continue
            else:
                tdelta = np.sum(ev['dt'][i:i_min])
                if (b1-udb[i_min])>(R2):
                    # If we've got one need to see how far it extends
                    b1_reduction[i:i_min+1] = 1
                    # If it's at the end of the window, check if its continuing
                    if (i_min==i+lookahead-1):#min at the end of the window
                        found_min = False
                    else:
                        found_min = True
                    while not found_min:
                        if udb[i_min+1]<udb[i_min]:
                            b1_reduction[i_min+1] = 1
                            i_min +=1
                            if i_min==(len(udb)-2):
                                found_min = True
                        else:
                            found_min = True
    both = [yes1 or yes2 for yes1,yes2 in zip(xline_reduction,b1_reduction)]
    return xline_reduction,b1_reduction,both

def hide_zeros(values):
    return np.array([float('nan') if x==0 else x for x in values])

def show_full_hist(events,path,**kwargs):
    testpoints = ['stretched_LOWnLOWu',
                  'stretched_MEDnLOWu',
                  'stretched_HIGHnLOWu',
                  'stretched_LOWnMEDu',
                  'stretched_MEDnMEDu',
                  'stretched_HIGHnMEDu',
                  'stretched_LOWnHIGHu',
                  'stretched_MEDnHIGHu',
                  'stretched_HIGHnHIGHu']
    #############
    #setup figure
    fig1,(ax1,ax2) = plt.subplots(1,2,figsize=[20,10])
    fig2,((ax3,ax4),(ax5,ax6)) = plt.subplots(2,2,figsize=[20,20])
    allK1var = np.array([])
    allK1var2 = np.array([])
    anySubstorm = np.array([])
    allSubstorm = np.array([])
    dipb = np.array([])
    moidm= np.array([])
    mgl  = np.array([])
    imf  = np.array([])
    for i,run in enumerate(testpoints):
        if run not in events.keys():
            continue
        evK1var = events[run]['K1var_5-5']/1e12
        evK1var2 = events[run]['K1var']

        evIMF  = events[run]['imf_transients']
        evAny  =(events[run]['substorm']*(1-evIMF))
        evAll  =(events[run]['allsubstorm']*(1-evIMF))
        evDipb =(events[run]['DIPb']*(1-evIMF))
        evMoidm=(events[run]['plasmoids_mass']*(1-evIMF))
        evMgl  =(events[run]['MGLbays']*(1-evIMF))

        allK1var = np.append(allK1var,evK1var.values)
        allK1var2 = np.append(allK1var2,evK1var2.values)
        anySubstorm = np.append(anySubstorm,evAny.values)
        allSubstorm = np.append(allSubstorm,evAll.values)
        dipb  = np.append(dipb,evDipb.values)
        moidm = np.append(moidm,evMoidm.values)
        mgl   = np.append(mgl,evMgl.values)
        imf   = np.append(imf,evIMF.values)
    dfK1var = pd.DataFrame({'K1var':allK1var,
                            'anysubstorm':anySubstorm,
                            'allsubstorm':allSubstorm,
                            'DIPb':dipb,
                            'moidsm':moidm,
                            'MGL':mgl,
                            'IMF':imf})
    dfK1var2 = pd.DataFrame({'K1var':allK1var2,
                            'anysubstorm':anySubstorm,
                            'allsubstorm':allSubstorm,
                            'DIPb':dipb,
                            'moidsm':moidm,
                            'MGL':mgl,
                            'IMF':imf})
    for ax,df,key in[(ax1,dfK1var,'anysubstorm'),(ax2,dfK1var2,'anysubstorm'),
                     (ax3,dfK1var2,'allsubstorm'),
                     (ax4,dfK1var2,'DIPb'),
                     (ax5,dfK1var2,'moidsm'),
                     (ax6,dfK1var2,'MGL')]:
        binsvar = np.linspace(df['K1var'].quantile(0.00),
                              df['K1var'].quantile(0.95),51)
        # Create 3 stacks
        layer1 = df['K1var'][(df[key]==0)&(df['IMF']==0)]
        layer2 = df['K1var'][df['IMF']==1]
        layer3 = df['K1var'][df[key]==1]
        y1,x = np.histogram(layer1,bins=binsvar)
        y2,x2 = np.histogram(layer2,bins=binsvar)
        y3,x3 = np.histogram(layer3,bins=binsvar)
        # Variability
        ax.stairs(y1+y2+y3,edges=x,fill=True,fc='grey',alpha=0.4,label='total')
        ax.stairs(y1,x,fill=True,fc='blue',label=f'not-{key}',alpha=0.4)
        ax.stairs(y3,x,fill=True,fc='green',label=key,alpha=0.4)
        #
        ax.stairs(y1+y2+y3,edges=x,ec='grey',label='_total')
        ax.stairs(y2,x,ec='black',label=f'IMFTransit',lw=4)
        ax.stairs(y1,x,ec='blue',label=f'_not-{key}')
        ax.stairs(y3,x,ec='green',label=f'_{key}')
        #
        ax.axvline(df['K1var'].quantile(0.50),c='grey',lw=4)
        ax.axvline(layer1.quantile(0.50),c='darkblue',lw=2)
        ax.axvline(layer2.quantile(0.50),c='black',lw=2)
        ax.axvline(layer3.quantile(0.50),c='darkgreen',lw=2)
        ax.legend(loc='center right')
        ax.set_ylabel('Counts')
        # Decorate
        q50_var =df['K1var'].quantile(0.50)
        if ax!=ax1:
            ax.text(1,0.94,'Median'+f'={q50_var:.2f}[%]',
                        transform=ax.transAxes,
                        horizontalalignment='right')
        ax.set_xlabel('Relative Total Variation of \n'+
                  r'$\int_{-5}^{5}\mathbf{K_1}\left[\%\right]$')
    q50_var =dfK1var['K1var'].quantile(0.50)
    ax1.text(1,0.94,'Median'+f'={q50_var:.2f}[TW]',
                        transform=ax1.transAxes,
                        horizontalalignment='right')
    ax1.set_xlabel('Total Variation of \n'+
                     r'$\int_{-5}^{5}\mathbf{K_1}\left[ TW\right]$')

    # Save
    fig1.tight_layout(pad=1)
    figurename = path+'/hist_eventsK1_1.pdf'
    fig1.savefig(figurename)
    plt.close(fig1)
    print('\033[92m Created\033[00m',figurename)

    fig2.tight_layout(pad=1)
    figurename = path+'/hist_eventsK1_2.pdf'
    fig2.savefig(figurename)
    plt.close(fig2)
    print('\033[92m Created\033[00m',figurename)

def show_multi_events(evs,events,path,**kwargs):
    interval_list = build_interval_list(TSTART,DT,TJUMP,
                                    evs['stretched_MEDnHIGHu']['mp'].index)
    # Set windows
    window1 = (dt.datetime(2022,6,6,11,30),dt.datetime(2022,6,6,14,30))
    window2 = (dt.datetime(2022,6,6,15,30),dt.datetime(2022,6,6,18,30))
    window3 = (dt.datetime(2022,6,7,17,30),dt.datetime(2022,6,7,20,30))
    window4 = (dt.datetime(2022,6,6,9,30),dt.datetime(2022,6,6,12,30))
    #xlims1 = [float(pd.Timedelta(t-T0).to_numpy()) for t in window1]
    #xlims2 = [float(pd.Timedelta(t-T0).to_numpy()) for t in window2]
    #xlims3 = [float(pd.Timedelta(t-T0).to_numpy()) for t in window3]
    xlims1 = [(t-T0).total_seconds()*1e9 for t in window1]
    xlims2 = [(t-T0).total_seconds()*1e9 for t in window2]
    xlims3 = [(t-T0).total_seconds()*1e9 for t in window3]
    xlims4 = [(t-T0).total_seconds()*1e9 for t in window4]
    #runs = ['stretched_MEDnHIGHu','stretched_HIGHnHIGHu']
    runs = ['stretched_LOWnLOWu','stretched_HIGHnHIGHu']
    # Pull the things we're gonna plot
    run_a = runs[0]
    run_b = runs[0]
    #xlimsa = xlims1
    #xlimsb = xlims2
    xlimsa = xlims4
    xlimsb = xlims2
    # IMF transits
    IMF_a = events[run_a]['imf_transients']
    IMF_b = events[run_b]['imf_transients']
    # Time axis
    t_a = evs[run_a]['times']
    t_b = evs[run_b]['times']
    #   K1 flux
    K1_a = evs[run_a]['K1']/1e12
    K1_b = evs[run_b]['K1']/1e12
    Any_a = events[run_a]['substorm']*(1-events[run_a]['imf_transients'])
    Any_b = events[run_b]['substorm']*(1-events[run_b]['imf_transients'])
    #   Total Variation
    K1var_a = events[run_a]['K1var_5-5']/1e12
    K1var_b = events[run_b]['K1var_5-5']/1e12
    #   deltaB
    dB_a = evs[run_a]['closed']['u_db_night [J]']/1e15
    dB_b = evs[run_b]['closed']['u_db_night [J]']/1e15
    DIP_a = events[run_a]['DIPb']*(1-events[run_a]['imf_transients'])
    DIP_b = events[run_b]['DIPb']*(1-events[run_b]['imf_transients'])
    #   Volume + Mass
    M_a   = evs[run_a]['closed']['M [kg]']/1e3
    M_b   = evs[run_b]['closed']['M [kg]']/1e3
    Moidm_a = events[run_a]['plasmoids_mass']*(
                                          1-events[run_a]['imf_transients'])
    Moidm_b = events[run_b]['plasmoids_mass']*(
                                          1-events[run_b]['imf_transients'])
    #   GridL
    gridL_a = evs[run_a]['GridL']
    gridL_b = evs[run_b]['GridL']
    grid_bay_a = events[run_a]['MGLbays']*(
                                          1-events[run_a]['imf_transients'])
    grid_bay_b = events[run_b]['MGLbays']*(
                                          1-events[run_b]['imf_transients'])
    # Create Figures
    fig,axes = plt.subplots(5,2,figsize=[24,35])
    col_a = axes[:,0]
    col_b = axes[:,1]
    # Plot
    #   K1 flux
    col_a[0].fill_between(t_a,K1_a,fc='blue')
    col_a[0].fill_between(t_a,hide_zeros(K1_a*Any_a.values),fc='green')
    col_b[0].fill_between(t_b,K1_b,fc='blue')
    col_b[0].fill_between(t_b,hide_zeros(K1_b*Any_b.values),fc='green')
    #   Total Variation
    col_a[1].fill_between(t_a,K1var_a,fc='blue')
    col_a[1].fill_between(t_a,hide_zeros(K1var_a*Any_a.values),fc='green')
    col_b[1].fill_between(t_b,K1var_b,fc='blue')
    col_b[1].fill_between(t_b,hide_zeros(K1var_b*Any_b.values),fc='green')
    #   deltaB
    col_a[2].fill_between(t_a,dB_a,fc='grey')
    col_a[2].fill_between(t_a,hide_zeros(dB_a*DIP_a.values),fc='pink')
    col_b[2].fill_between(t_b,dB_b,fc='grey')
    col_b[2].fill_between(t_b,hide_zeros(dB_b*DIP_b.values),fc='pink')
    #   Mass
    col_a[3].fill_between(t_a,M_a,fc='grey')
    col_a[3].fill_between(t_a,hide_zeros(M_a*Moidm_a.values),fc='pink')
    col_b[3].fill_between(t_b,M_b,fc='grey')
    col_b[3].fill_between(t_b,hide_zeros(M_b*Moidm_b.values),fc='pink')
    #   GridL
    col_a[4].fill_between(t_a,gridL_a,fc='grey')
    col_a[4].fill_between(t_a,hide_zeros(gridL_a*grid_bay_a.values),fc='pink')
    col_b[4].fill_between(t_b,gridL_b,fc='grey')
    col_b[4].fill_between(t_b,hide_zeros(gridL_b*grid_bay_b.values),fc='pink')
    # Decorate
    for ax in col_a:
        for interv in interval_list:
            ax.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        ax.fill_between(t_a,ax.get_ylim()[0]*hide_zeros(IMF_a.values),
                        ax.get_ylim()[1]*IMF_a,
                        fc='black',alpha=0.3)
        ax.margins(x=0.1,y=0.1)
    for ax in col_b:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        for interv in interval_list:
            ax.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        ax.fill_between(t_b,ax.get_ylim()[0]*hide_zeros(IMF_b.values),
                        ax.get_ylim()[1]*IMF_b,
                        fc='black',alpha=0.3)
        ax.margins(x=0.1,y=0.1)
    general_plot_settings(col_a[0],do_xlabel=False,legend=False,
                          ylabel=r'$\int\mathbf{K}_1\left[TW\right]$',
                          xlim=xlimsa,
                          #ylim=[-20,0],
                          ylim=[-6,0],
                          timedelta=True)
    general_plot_settings(col_b[0],do_xlabel=False,legend=False,
                          ylabel=r'$\int\mathbf{K}_1\left[TW\right]$',
                          xlim=xlimsb,
                          #ylim=[-20,0],
                          ylim=[-6,0],
                          timedelta=True)
    general_plot_settings(col_a[1],do_xlabel=False,legend=False,
                          ylabel=r'$TV\left(\int\mathbf{K}_1\right)$',
                          xlim=xlimsa,
                          #ylim=[0,6],
                          ylim=[0,6],
                          timedelta=True)
    general_plot_settings(col_b[1],do_xlabel=False,legend=False,
                          ylabel=r'$TV\left(\int\mathbf{K}_1\right)$',
                          xlim=xlimsb,
                          #ylim=[0,6],
                          ylim=[0,6],
                          timedelta=True)
    general_plot_settings(col_a[2],do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int|\Delta B|\left[PJ\right]$',
                          xlim=xlimsa,
                          ylim=[1.2,5.2],
                          timedelta=True)
    general_plot_settings(col_b[2],do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int|\Delta B|\left[PJ\right]$',
                          xlim=xlimsb,
                          ylim=[1.2,5.2],
                          timedelta=True)
    general_plot_settings(col_a[3],do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int n\left[Mg\right]$',
                          xlim=xlimsa,
                          ylim=[70,200],
                          timedelta=True)
    general_plot_settings(col_b[3],do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int n\left[Mg\right]$',
                          xlim=xlimsb,
                          ylim=[70,200],
                          timedelta=True)
    general_plot_settings(col_a[4],do_xlabel=True,legend=False,
                          ylabel=r'$GridL \left[nT\right]$',
                          xlim=xlimsa,
                          ylim=[-1800,-500],
                          timedelta=True)
    general_plot_settings(col_b[4],do_xlabel=True,legend=False,
                          ylabel=r'$GridL \left[nT\right]$',
                          xlim=xlimsb,
                          ylim=[-1800,-500],
                          timedelta=True)
    # Save
    fig.tight_layout(pad=1)
    figurename = path+'/vis_events_multi.svg'
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

def show_events(ev,run,events,path,**kwargs):
    interval_list =build_interval_list(TSTART,DT,TJUMP,ev['mp'].index)
    #Set the zoom
    if 'zoom' in kwargs:
        xlims = [(t-T0).total_seconds()*1e9 for t in kwargs.get('zoom')]
        #xlims = [float(pd.Timedelta(t-T0).to_numpy())
        #         for t in kwargs.get('zoom')]
    else:
        xlims = [(t-T0).total_seconds()*1e9 for t in [TSTART,TEND]]
        #xlims = [float(pd.Timedelta(t-T0).to_numpy())
        #         for t in [TSTART,TEND]]
    #Create handy variables for marking found points on curves
    # Time
    t        = ev['times']
    # K1 flux and K1 flux variation
    K1       = ev['K1']/1e12
    K1var    = events[run]['K1var_5-5']/1e12
    # IMF transits
    IMF      = events[run]['imf_transients']
    # Dipolarizations
    dB       = ev['closed']['u_db_night [J]']/1e15
    DIP      = events[run]['DIPb']*(1-events[run]['imf_transients'])
    # Plasmoids
    M        = ev['closed']['M [kg]']/1e3
    Moid     = events[run]['plasmoids_mass']*(1-events[run]['imf_transients'])
    # Current Wedge
    gridL    = ev['GridL']
    grid_bay = events[run]['MGLbays']*(1-events[run]['imf_transients'])
    # Substorms
    Any      = events[run]['substorm']*(1-events[run]['imf_transients'])
    #############
    #setup figure
    fig1,axes = plt.subplots(5,1,figsize=[24,35],sharex=True)
    k1     = axes[0]
    k1var  = axes[1]
    dip    = axes[2]
    moid   = axes[3]
    glbays = axes[4]
    #Plot
    # K1 flux
    k1.fill_between(t,K1,fc='blue')
    k1.fill_between(t,hide_zeros(K1*Any.values),fc='green')
    # Total Variation
    k1var.fill_between(t,K1var,fc='blue')
    k1var.fill_between(t,hide_zeros(K1var*Any.values),fc='green')
    # deltaB
    dip.fill_between(t,dB,fc='grey')
    dip.fill_between(t,hide_zeros(dB*DIP.values),fc='pink')
    # Mass
    moid.fill_between(t,M,fc='grey')
    moid.fill_between(t,hide_zeros(M*Moid.values),fc='pink')
    # GridL
    glbays.fill_between(t,gridL,fc='grey')
    glbays.fill_between(t,hide_zeros(gridL*grid_bay.values),fc='pink')
    #Decorate
    general_plot_settings(k1,do_xlabel=False,legend=False,
                          ylabel=r'$\int\mathbf{K}_1\left[TW\right]$',
                          xlim=xlims,ylim=[k1.get_ylim()[0],0],
                          timedelta=True)
    general_plot_settings(k1var,do_xlabel=False,legend=False,
                          ylabel=r'$TV\left(\int\mathbf{K}_1\right)$',
                          xlim=xlims,
                          ylim=[0,k1var.get_ylim()[1]],
                          timedelta=True)
    general_plot_settings(dip,do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int|\Delta B|\left[PJ\right]$',
                          xlim=xlims,
                          ylim=[0,dip.get_ylim()[1]],
                          timedelta=True)
    general_plot_settings(moid,do_xlabel=False,legend=False,
                          ylabel=r'(closed)$\int n\left[Mg\right]$',
                          xlim=xlims,
                          ylim=[0,moid.get_ylim()[1]],
                          timedelta=True)
    general_plot_settings(glbays,do_xlabel=True,legend=False,
                          ylabel=r'$GridL \left[nT\right]$',
                          xlim=xlims,
                          ylim=[glbays.get_ylim()[0],1],
                          timedelta=True)
    for ax in axes:
        for interv in interval_list:
            ax.axvline(((interv[0]-T0).total_seconds()*1e9),c='grey')
        ax.fill_between(t,ax.get_ylim()[0]*hide_zeros(IMF.values),
                        ax.get_ylim()[1]*IMF,
                        fc='black',alpha=0.3)
        ax.margins(x=0.1,y=0.1)

    #Save
    fig1.tight_layout(pad=1)
    figurename = path+'/vis_events_'+run+kwargs.get('tag','')+'.pdf'
    fig1.savefig(figurename)
    plt.close(fig1)
    print('\033[92m Created\033[00m',figurename)

def coupling_scatter(events,path):
    interval_list =build_interval_list(TSTART,DT,TJUMP,
                                       dataset['stretched_LOWnLOWu']['time'])
    #############
    #setup figure
    scatterCoupling,(axis) =plt.subplots(1,1,figsize=[15,15])
    colors = ['#80b3ffff','#0066ffff','#0044aaff',
              '#80ffb3ff','#2aff80ff','#00aa44ff',
              '#ffe680ff','#ffcc00ff','#806600ff']
    testpoints = ['stretched_LOWnLOWu',
                  'stretched_MEDnLOWu',
                  'stretched_HIGHnLOWu',
                  'stretched_LOWnMEDu',
                  'stretched_MEDnMEDu',
                  'stretched_HIGHnMEDu',
                  'stretched_LOWnHIGHu',
                  'stretched_MEDnHIGHu',
                  'stretched_HIGHnHIGHu']
    #Plot
    for i,testpoint in enumerate(testpoints):
        if testpoint in dataset.keys():
            sw = dataset[testpoint]['obs']['swmf_sw']
            Ein,Pstorm = np.zeros(23),np.zeros(23)
            for k,tsample in enumerate([j[1] for j in interval_list][0:-1]):
                Ein[k],Pstorm[k] = sw.loc[tsample,['EinWang','Pstorm']]
            axis.scatter(Ein/1e12,-Pstorm/1e12,s=200,c=colors[i],
                         label=str(i+1))
    axis.set_xlabel(r'$E_{in} \left(TW\right)$, (Wang2014)')
    axis.set_ylabel(r'$P_{storm} \left(TW\right)$, '+
                                                '(Tenfjord and Østgaard 2013)')
    #axis.legend()
    scatterCoupling.tight_layout(pad=1)
    figurename = path+'/couplingScatter.svg'
    scatterCoupling.savefig(figurename)
    plt.close(scatterCoupling)
    print('\033[92m Created\033[00m',figurename)

def coupling_model_vs_sim(dataset,events,path,**kwargs):
    raw_data, Ebyrun                    = plt.subplots(1,1,figsize=[24,24])
    model_vs_sim, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=[24,24])
    model_vs_sim2,([ax5,ax6],[ax7,ax8]) = plt.subplots(2,2,figsize=[24,24])
    model_vs_sim3,([ax9,ax10]) = plt.subplots(2,1,figsize=[12,24])
    cpcp          = ax1
    Ebycategory   = ax2
    cpcp_variance = ax3
    Evariance     = ax4
    #
    saturate      = ax5
    energy        = ax6
    sat_variance  = ax7
    energy_var    = ax8
    #
    energy2       = ax9
    energy2_var   = ax10
    colors = ['#80b3ffff','#0066ffff','#0044aaff',
              '#80ffb3ff','#00aa44ff','#005500ff',
              '#ffe680ff','#ffcc00ff','#806600ff']
    markers = ['*','*','*',
               'o','o','o',
               'x','x','x']
    testpoints = ['stretched_LOWnLOWu',
                  'stretched_MEDnLOWu',
                  'stretched_HIGHnLOWu',
                  'stretched_LOWnMEDu',
                  'stretched_MEDnMEDu',
                  'stretched_HIGHnMEDu',
                  'stretched_LOWnHIGHu',
                  'stretched_MEDnHIGHu',
                  'stretched_HIGHnHIGHu']
    allK1var = np.array([])
    anySubstorm = np.array([])
    imf = np.array([])
    allEin,allEin2 = np.array([]),np.array([])
    allK1 = np.array([])
    allCPCP = np.array([])
    allU    = np.array([])
    #Plot
    for i,run in enumerate(testpoints):
        if run not in events.keys():
            continue
        # Redo some basic data wrangling over the missing data times
        tstart = T0+dt.timedelta(minutes=10)
        mp = dataset[run]['mpdict']['ms_full'][
                               dataset[run]['mpdict']['ms_full'].index>tstart]
        mp = mp.resample('60s').asfreq()
        # Get data to a common time axis
        index_log = dataset[run]['obs']['swmf_log'].index
        index_sw = dataset[run]['obs']['swmf_sw'].index
        t_log = [float(t.to_numpy()) for t in index_log-T0]
        t_sw = [float(t.to_numpy()) for t in index_sw-T0]
        t_energy = [float(t.to_numpy()) for t in mp.index-T0]

        # Extract the quantities for this subset of data
        evK1var = events[run]['K1var_5-5']/1e12
        evIMF   = events[run]['imf_transients']
        evAny   = events[run]['substorm']*(1-evIMF)
        evEin   = np.interp(t_energy,t_sw,
                        dataset[run]['obs']['swmf_sw']['EinWang'].values/1e12)
        evEin2  = np.interp(t_energy,t_sw,
                        -dataset[run]['obs']['swmf_sw']['Pstorm'].values/1e12)
        CPCP    = np.interp(t_energy,t_log,
                        dataset[run]['obs']['swmf_log']['cpcpn'].values)
        K1      = (mp['K_netK1 [W]']+mp['UtotM1 [W]'])/-1e12
        U       = (mp['Utot [J]'])/1e15

        # Plot this subset of data with color and marker style
        Ebyrun.scatter(evEin,K1,c=colors[i],marker=markers[i],s=50,
                       alpha=0.3)

        # Append subset of data to a full data array for further vis
        allK1var    = np.append(allK1var,evK1var.values)
        anySubstorm = np.append(anySubstorm,evAny.values)
        imf         = np.append(imf,evIMF.values)
        allK1       = np.append(allK1,K1.values)
        allEin      = np.append(allEin,evEin)
        allEin2     = np.append(allEin2,evEin2)
        allCPCP     = np.append(allCPCP,CPCP)
        allU        = np.append(allU,U)

    df_summary = pd.DataFrame({'K1var':allK1var,
                               'anysubstorm':anySubstorm,
                               'IMF':imf,
                               'Ein':allEin,
                               'Ein2':allEin2,
                               'CPCP':allCPCP,
                               'U':allU,
                               'K1':allK1})
    # Obtain low,50, and high %tiles, and variance binned by our X axis
    #Ein_bins  = np.linspace(0.5,24.5,25)
    Ein_bins  = np.linspace(1,24,11)
    Eindict   = bin_and_describe(df_summary['Ein'],df_summary['K1'],
                               df_summary,Ein_bins,0.05,0.95)
    CPCP_bins = np.linspace(df_summary['CPCP'].quantile(0.005),
                            df_summary['CPCP'].quantile(0.995),11)
    CPCPdict  = bin_and_describe(df_summary['CPCP'],df_summary['K1'],
                                 df_summary,CPCP_bins,0.05,0.95)
    Satdict   = bin_and_describe(df_summary['Ein'],df_summary['CPCP'],
                                 df_summary,Ein_bins,0.05,0.95)
    U_bins    = np.linspace(df_summary['U'].quantile(0.01),
                            df_summary['U'].quantile(0.99),11)
    Udict     = bin_and_describe(df_summary['U'],df_summary['K1'],
                                 df_summary,U_bins,0.05,0.95)
    K_bins    = np.linspace(df_summary['K1'].quantile(0.01),
                            df_summary['K1'].quantile(0.99),11)
    Kdict     = bin_and_describe(df_summary['K1'],df_summary['U'],
                                 df_summary,K_bins,0.05,0.95)
    # Lines on plots
    Ebyrun.plot(df_summary['Ein'],df_summary['Ein'],ls='--',c='grey')
    Ebyrun.plot(df_summary['Ein'],df_summary['Ein2'],ls='--',c='grey')

    # Ax1
    Ebycategory.scatter(df_summary['Ein']*df_summary['IMF'],
                       df_summary['K1']*df_summary['IMF'],
                     marker='+',c='black',s=50,label='IMF Transit',alpha=0.2)
    Ebycategory.scatter(df_summary['Ein']*df_summary['anysubstorm'],
                       df_summary['K1']*df_summary['anysubstorm'],
                     marker='x',c='green',s=50,label='Any Substorm',alpha=0.2)
    Ebycategory.scatter(df_summary['Ein']*(1-df_summary['anysubstorm'])
                                         *(1-df_summary['IMF']),
                       df_summary['K1']*(1-df_summary['anysubstorm'])
                                         *(1-df_summary['IMF']),
                       c='blue',s=50,label='Not Any Substorm',alpha=0.2)
    extended_fill_between(Ebycategory,Ein_bins,
                          Eindict['pLow_imf'],Eindict['pHigh_imf'],
                          'black',0.2)
    extended_fill_between(Ebycategory,Ein_bins,
                          Eindict['pLow_not'],Eindict['pHigh_not'],
                          'blue',0.2)
    extended_fill_between(Ebycategory,Ein_bins,
                          Eindict['pLow_sub'],Eindict['pHigh_sub'],
                          'green',0.2)
    Ebycategory.plot(Ein_bins,Eindict['p50_not'],c='blue',ls='--',lw=4)
    Ebycategory.plot(Ein_bins,Eindict['p50_sub'],c='green',ls='--',lw=4)
    Ebycategory.plot(Ein_bins,Eindict['p50_imf'],c='black',ls='--',lw=4)
    # Ax2
    cpcp.scatter(df_summary['CPCP']*df_summary['IMF'],
                       df_summary['K1']*df_summary['IMF'],
                      marker='+',c='black',s=50,label='IMF Transit',alpha=0.2)
    cpcp.scatter(df_summary['CPCP']*df_summary['anysubstorm'],
                       df_summary['K1']*df_summary['anysubstorm'],
                      marker='x',c='green',s=50,label='Any Substorm',alpha=0.2)
    cpcp.scatter(df_summary['CPCP']*(1-df_summary['anysubstorm'])
                                   *(1-df_summary['IMF']),
                       df_summary['K1']*(1-df_summary['anysubstorm'])
                                       *(1-df_summary['IMF']),
                       c='blue',s=50,label='Not Any Substorm',alpha=0.2)
    extended_fill_between(cpcp,CPCP_bins,
                          CPCPdict['pLow_imf'],CPCPdict['pHigh_imf'],
                          'black',0.2)
    extended_fill_between(cpcp,CPCP_bins,
                          CPCPdict['pLow_not'],CPCPdict['pHigh_not'],
                          'blue',0.2)
    extended_fill_between(cpcp,CPCP_bins,
                          CPCPdict['pLow_sub'],CPCPdict['pHigh_sub'],
                          'green',0.2)
    cpcp.plot(CPCP_bins,CPCPdict['p50_not'],c='blue',ls='--',lw=4)
    cpcp.plot(CPCP_bins,CPCPdict['p50_sub'],c='green',ls='--',lw=4)
    cpcp.plot(CPCP_bins,CPCPdict['p50_imf'],c='black',ls='--',lw=4)

    # Ax3
    Evariance.plot(Ein_bins,Eindict['variance_imf']**0.5,c='black',marker='+')
    Evariance.plot(Ein_bins,Eindict['variance_sub']**0.5,c='green',marker='x')
    Evariance.plot(Ein_bins,Eindict['variance_not']**0.5,c='blue',marker='o')
    Evariance.plot(Ein_bins,Eindict['variance_all']**0.5,c='grey',marker='o')
    Evariance_r = Evariance.twinx()
    Evariance_r.bar(Ein_bins,Eindict['nAll'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='grey')
    Evariance_r.bar(Ein_bins,Eindict['nSub'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='green')
    Evariance_r.bar(Ein_bins,Eindict['nNot'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='blue')
    Evariance_r.bar(Ein_bins,Eindict['nIMF'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='black')

    # Ax4
    cpcp_variance.plot(CPCP_bins,CPCPdict['variance_imf']**0.5,c='black',
                       marker='+')
    cpcp_variance.plot(CPCP_bins,CPCPdict['variance_sub']**0.5,c='green',
                       marker='x')
    cpcp_variance.plot(CPCP_bins,CPCPdict['variance_not']**0.5,c='blue',
                       marker='o')
    cpcp_variance.plot(CPCP_bins,CPCPdict['variance_all']**0.5,c='grey',
                       marker='o')
    cpcp_variance_r = cpcp_variance.twinx()
    cpcp_variance_r.bar(CPCP_bins,CPCPdict['nAll'],alpha=1,
                  width=(CPCP_bins[1]-CPCP_bins[0])/2,fill=False,ec='grey')
    cpcp_variance_r.bar(CPCP_bins,CPCPdict['nSub'],alpha=1,
                  width=(CPCP_bins[1]-CPCP_bins[0])/2,fill=False,ec='green')
    cpcp_variance_r.bar(CPCP_bins,CPCPdict['nNot'],alpha=1,
                  width=(CPCP_bins[1]-CPCP_bins[0])/2,fill=False,ec='blue')
    cpcp_variance_r.bar(CPCP_bins,CPCPdict['nIMF'],alpha=1,
                  width=(CPCP_bins[1]-CPCP_bins[0])/2,fill=False,ec='black')

    # Ax5
    saturate.scatter(df_summary['Ein']*df_summary['IMF'],
                       df_summary['CPCP']*df_summary['IMF'],
                     marker='+',c='black',s=50,label='IMF Transit',alpha=0.2)
    saturate.scatter(df_summary['Ein']*df_summary['anysubstorm'],
                       df_summary['CPCP']*df_summary['anysubstorm'],
                     marker='x',c='green',s=50,label='Any Substorm',alpha=0.2)
    saturate.scatter(df_summary['Ein']*(1-df_summary['anysubstorm'])
                                      *(1-df_summary['IMF']),
                       df_summary['CPCP']*(1-df_summary['anysubstorm'])
                                         *(1-df_summary['IMF']),
                       c='blue',s=50,label='Not Any Substorm',alpha=0.2)
    extended_fill_between(saturate,Ein_bins,
                          Satdict['pLow_imf'],Satdict['pHigh_imf'],
                          'black',0.3)
    extended_fill_between(saturate,Ein_bins,
                          Satdict['pLow_not'],Satdict['pHigh_not'],
                          'blue',0.3)
    extended_fill_between(saturate,Ein_bins,
                          Satdict['pLow_sub'],Satdict['pHigh_sub'],
                          'green',0.3)
    saturate.plot(Ein_bins,Satdict['p50_not'],c='blue',ls='--',lw=4)
    saturate.plot(Ein_bins,Satdict['p50_sub'],c='green',ls='--',lw=4)
    saturate.plot(Ein_bins,Satdict['p50_imf'],c='black',ls='--',lw=4)
    # Ax6
    energy.scatter(df_summary['U']*df_summary['IMF'],
                       df_summary['K1']*df_summary['IMF'],
                     marker='+',c='black',s=50,label='IMF Transit',alpha=0.2)
    energy.scatter(df_summary['U']*df_summary['anysubstorm'],
                       df_summary['K1']*df_summary['anysubstorm'],
                     marker='x',c='green',s=50,label='Any Substorm',alpha=0.2)
    energy.scatter(df_summary['U']*(1-df_summary['anysubstorm'])
                                  *(1-df_summary['IMF']),
                       df_summary['K1']*(1-df_summary['anysubstorm'])
                                       *(1-df_summary['IMF']),
                       c='blue',s=50,label='Not Any Substorm',alpha=0.2)
    extended_fill_between(energy,U_bins,
                          Udict['pLow_imf'],Udict['pHigh_imf'],
                          'black',0.2)
    extended_fill_between(energy,U_bins,
                          Udict['pLow_not'],Udict['pHigh_not'],
                          'blue',0.2)
    extended_fill_between(energy,U_bins,
                          Udict['pLow_sub'],Udict['pHigh_sub'],
                          'green',0.2)
    energy.plot(U_bins,Udict['p50_not'],c='blue',ls='--',lw=4)
    energy.plot(U_bins,Udict['p50_sub'],c='green',ls='--',lw=4)
    energy.plot(U_bins,Udict['p50_imf'],c='black',ls='--',lw=4)

    # Ax7
    sat_variance.plot(Ein_bins,Satdict['variance_imf']**0.5,c='black',
                      marker='+')
    sat_variance.plot(Ein_bins,Satdict['variance_sub']**0.5,c='green',
                      marker='x')
    sat_variance.plot(Ein_bins,Satdict['variance_not']**0.5,c='blue',
                      marker='o')
    sat_variance.plot(Ein_bins,Satdict['variance_all']**0.5,c='grey',
                      marker='o')
    sat_variance_r = sat_variance.twinx()
    sat_variance_r.bar(Ein_bins,Satdict['nAll'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='grey')
    sat_variance_r.bar(Ein_bins,Satdict['nSub'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='green')
    sat_variance_r.bar(Ein_bins,Satdict['nNot'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='blue')
    sat_variance_r.bar(Ein_bins,Satdict['nIMF'],alpha=1,
                  width=(Ein_bins[1]-Ein_bins[0])/2,fill=False,ec='black')

    # Ax8
    energy_var.plot(U_bins,Udict['variance_imf']**0.5,c='black',marker='+')
    energy_var.plot(U_bins,Udict['variance_sub']**0.5,c='green',marker='x')
    energy_var.plot(U_bins,Udict['variance_not']**0.5,c='blue',marker='o')
    energy_var.plot(U_bins,Udict['variance_all']**0.5,c='grey',marker='o')
    energy_var_r = energy_var.twinx()
    energy_var_r.bar(U_bins,Udict['nAll'],alpha=1,
                  width=(U_bins[1]-U_bins[0])/2,fill=False,ec='grey')
    energy_var_r.bar(U_bins,Udict['nSub'],alpha=1,
                  width=(U_bins[1]-U_bins[0])/2,fill=False,ec='green')
    energy_var_r.bar(U_bins,Udict['nNot'],alpha=1,
                  width=(U_bins[1]-U_bins[0])/2,fill=False,ec='blue')
    energy_var_r.bar(U_bins,Udict['nIMF'],alpha=1,
                  width=(U_bins[1]-U_bins[0])/2,fill=False,ec='black')

    # Ax9
    energy2.scatter(df_summary['K1']*df_summary['IMF'],
                       df_summary['U']*df_summary['IMF'],
                     marker='+',c='black',s=50,label='IMF Transit',alpha=0.2)
    energy2.scatter(df_summary['K1']*df_summary['anysubstorm'],
                       df_summary['U']*df_summary['anysubstorm'],
                     marker='x',c='green',s=50,label='Any Substorm',alpha=0.2)
    energy2.scatter(df_summary['K1']*(1-df_summary['anysubstorm'])
                                  *(1-df_summary['IMF']),
                       df_summary['U']*(1-df_summary['anysubstorm'])
                                       *(1-df_summary['IMF']),
                       c='blue',s=50,label='Not Any Substorm',alpha=0.2)
    extended_fill_between(energy2,K_bins,
                          Kdict['pLow_imf'],Kdict['pHigh_imf'],
                          'black',0.2)
    extended_fill_between(energy2,K_bins,
                          Kdict['pLow_not'],Kdict['pHigh_not'],
                          'blue',0.2)
    extended_fill_between(energy2,K_bins,
                          Kdict['pLow_sub'],Kdict['pHigh_sub'],
                          'green',0.2)
    energy2.plot(K_bins,Kdict['p50_not'],c='blue',ls='--',lw=4)
    energy2.plot(K_bins,Kdict['p50_sub'],c='green',ls='--',lw=4)
    energy2.plot(K_bins,Kdict['p50_imf'],c='black',ls='--',lw=4)
    # Ax10
    energy2_var.plot(K_bins,Kdict['variance_imf']**0.5,c='black',marker='+')
    energy2_var.plot(K_bins,Kdict['variance_sub']**0.5,c='green',marker='x')
    energy2_var.plot(K_bins,Kdict['variance_not']**0.5,c='blue',marker='o')
    energy2_var.plot(K_bins,Kdict['variance_all']**0.5,c='grey',marker='o')
    energy2_var_r = energy2_var.twinx()
    energy2_var_r.bar(K_bins,Kdict['nAll'],alpha=1,
                  width=(K_bins[1]-K_bins[0])/2,fill=False,ec='grey')
    energy2_var_r.bar(K_bins,Kdict['nSub'],alpha=1,
                  width=(K_bins[1]-K_bins[0])/2,fill=False,ec='green')
    energy2_var_r.bar(K_bins,Kdict['nNot'],alpha=1,
                  width=(K_bins[1]-K_bins[0])/2,fill=False,ec='blue')
    energy2_var_r.bar(K_bins,Kdict['nIMF'],alpha=1,
                  width=(K_bins[1]-K_bins[0])/2,fill=False,ec='black')

    ##Decorate the plots
    Ebyrun.set_xlim(0,25)
    Ebyrun.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    Ebyrun.set_ylabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    Ebyrun.set_ylim(0,25)

    Ebycategory.set_xlim(0,25)
    Ebycategory.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    Ebycategory.set_ylabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    Ebycategory.set_ylim(0,25)

    cpcp.set_xlim(20,180)
    cpcp.set_xlabel(r'CPCP $\left[kV\right]$')
    cpcp.set_ylabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    cpcp.set_ylim(0,25)

    Evariance.set_xlim(0,25)
    Evariance.set_ylim(0,6.7)
    Evariance.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    Evariance.set_ylabel(r'$\sigma\left[TW\right]$')
    Evariance_r.set_ylabel(r'Counts')

    cpcp_variance.set_xlim(20,180)
    cpcp_variance.set_ylim(0,6.7)
    cpcp_variance.set_xlabel(r'CPCP $\left[kV\right]$')
    cpcp_variance.set_ylabel(r'$\sigma\left[TW\right]$')
    cpcp_variance_r.set_ylabel(r'Counts')

    saturate.set_xlim(0,25)
    saturate.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    saturate.set_ylabel(r'CPCP $\left[kV\right]$')
    saturate.set_ylim(20,180)

    energy.set_xlim(30,70)
    energy.set_xlabel(r'$\int\mathbf{U}$ Energy $\left[PJ\right]$')
    energy.set_ylabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    energy.set_ylim(0,25)

    sat_variance.set_xlim(0,25)
    sat_variance.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    sat_variance.set_ylabel(r'$\sigma\left[kV\right]$')
    sat_variance_r.set_ylabel(r'Counts')

    energy_var.set_xlim(30,70)
    energy_var.set_ylim(0,6.7)
    energy_var.set_xlabel(r'$\int\mathbf{U}$ Energy $\left[PJ\right]$')
    energy_var.set_ylabel(r'$\sigma\left[TW\right]$')
    energy_var_r.set_ylabel(r'Counts')

    energy2.set_xlim(0,25)
    energy2.set_xlabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    energy2.set_ylabel(r'$\int\mathbf{U}$ Energy $\left[PJ\right]$')
    energy2.set_ylim(30,70)

    energy2_var.set_xlabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    energy2_var.set_ylabel(r'$\sigma\left[PJ\right]$')
    energy2_var_r.set_ylabel(r'Counts')

    #Save the plots
    raw_data.tight_layout(pad=1)
    figurename = path+'/compare_raw_data.pdf'
    raw_data.savefig(figurename)
    plt.close(raw_data)
    print('\033[92m Created\033[00m',figurename)

    model_vs_sim.tight_layout(pad=1)
    figurename = path+'/coupling_vs_sim.pdf'
    model_vs_sim.savefig(figurename)
    plt.close(model_vs_sim)
    print('\033[92m Created\033[00m',figurename)

    model_vs_sim2.tight_layout(pad=1)
    figurename = path+'/coupling_vs_sim2.pdf'
    model_vs_sim2.savefig(figurename)
    plt.close(model_vs_sim2)
    print('\033[92m Created\033[00m',figurename)

    model_vs_sim3.tight_layout(pad=1)
    figurename = path+'/coupling_vs_sim3.pdf'
    model_vs_sim3.savefig(figurename)
    plt.close(model_vs_sim3)
    print('\033[92m Created\033[00m',figurename)

def make_figures(dataset):
    path = final
    tave,tv,corr,raw,events,psds = {},{},{},{},{},{}
    evs = {}
    for i,run in enumerate(dataset.keys()):
        # Data wrangling and organization
        ev = refactor(dataset[run],T0)
        evs[run] = ev
        events[run] = build_events(ev,run)
        #Zoomed versions
        window1 = (dt.datetime(2022,6,6,11,30),
                   dt.datetime(2022,6,6,14,30))
        window2 = (dt.datetime(2022,6,6,15,30),
                   dt.datetime(2022,6,6,18,30))
        window3 = (dt.datetime(2022,6,7,17,30),
                   dt.datetime(2022,6,7,20,30))
        bonuswindow  = (dt.datetime(2022,6,8,0,0),
                        dt.datetime(2022,6,9,0,0))
        window4 = (dt.datetime(2022,6,6,9,30),
                   dt.datetime(2022,6,6,12,30))
        # Make event level figures
        test_matrix(run,ev,path)#NOTE only need to run this once
        if 'continued' in run:
            show_events(ev,run,events,path,zoom=bonuswindow)
            all_fluxes(ev,run,path,zoom=bonuswindow)
        else:
            show_events(ev,run,events,path)
            all_fluxes(ev,run,path)
    # Make aggregate level figures
    show_multi_events(evs,events,path)
    show_full_hist(events,path)
    plot_indices(dataset,path)
    coupling_scatter(dataset,path)
    coupling_model_vs_sim(dataset,events,path)
    #plot_2by2_flux(dataset,[window1,window2,window3],path)
    plot_2by2_flux(dataset,[window4,window1,window2],path)

if __name__ == "__main__":
    # Constants
    T0 = dt.datetime(2022,6,6,0,0)
    TSTART = dt.datetime(2022,6,6,0,10)
    DT = dt.timedelta(hours=2)
    TJUMP = dt.timedelta(hours=2)
    TEND = dt.datetime(2022,6,8,0,0)
    # Set paths
    inBase = sys.argv[-1]
    inLogs = os.path.join(sys.argv[-1],'data/logs/')
    inAnalysis = os.path.join(sys.argv[-1],'data/analysis/')
    outPath = os.path.join(inBase,'figures')
    unfiled = os.path.join(outPath,'unfiled')
    final   = os.path.join(outPath,'final')
    # Create output dir's
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)

    # Set pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    # Event list
    events =[
             'stretched_LOWnLOWu',
             'stretched_LOWnMEDu',
             'stretched_LOWnHIGHu',
             #
             'stretched_HIGHnLOWu',
             'stretched_HIGHnMEDu',
             'stretched_HIGHnHIGHu',
             #
             'stretched_MEDnLOWu',
             'stretched_MEDnMEDu',
             'stretched_MEDnHIGHu',
             'stretched_LOWnLOWucontinued',
             ]

    ## Analysis Data
    dataset = {}
    for event in events:
        GMfile = os.path.join(inAnalysis,event+'.h5')
        # GM data
        if os.path.exists(GMfile):
            dataset[event] = load_hdf_sort(GMfile)


    ## Log Data
    for event in events:
        prefix = event.split('_')[1]+'_'
        dataset[event]['obs'] = read_indices(inLogs,prefix=prefix,
                                        #start=dataset[event]['time'][0],
                 #end=dataset[event]['time'][-1]+dt.timedelta(seconds=1),
                                             read_supermag=False)
    ######################################################################
    ## Call all plots
    make_figures(dataset)
