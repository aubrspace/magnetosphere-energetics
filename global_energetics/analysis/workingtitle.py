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
from matplotlib import ticker, colors
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
from global_energetics.analysis.proc_energy_spatial import reformat_lshell

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
    assert (type(indata)==dict or type(indata)==pd.core.frame.DataFrame or
            type(indata)==pd.core.series.Series,
            'Data type only excepts dict, DataFrame, or Series')
    #Hand picked times
    start = dt.timedelta(minutes=kwargs.get('startshift',60))
    #Feb
    feb2014_impact = dt.datetime(2014,2,18,16,15)
    #feb2014_endMain1 = dt.datetime(2014,2,19,4,0)
    #feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    feb2014_endMain1 = dt.datetime(2014,2,19,9,45)
    feb2014_endMain2 = dt.datetime(2014,2,19,9,45)
    feb2014_inter_start = dt.datetime(2014,2,18,15,0)
    feb2014_inter_end = dt.datetime(2014,2,18,17,30)
    #Starlink
    starlink_impact = dt.datetime(2022,2,3,0,0)
    starlink_endMain1 = dt.datetime(2022,2,3,11,15)
    starlink_endMain1 = dt.datetime(2022,2,4,13,10)
    starlink_endMain2 = dt.datetime(2022,2,4,22,0)
    starlink_inter_start = dt.datetime(2022,2,2,22,45)
    starlink_inter_end = dt.datetime(2022,2,3,1,15)
    #May2019
    may2019_impact = dt.datetime(2019,5,13,19,0)
    may2019_endMain1 = dt.datetime(2019,5,14,7,45)
    may2019_endMain2 = dt.datetime(2019,5,14,7,45)
    may2019_inter_start = dt.datetime(2019,5,13,17,45)
    may2019_inter_end = dt.datetime(2019,5,13,20,15)
    #Aug2019
    aug2019_impact = dt.datetime(2019,8,30,20,56)
    aug2019_endMain1 = dt.datetime(2019,8,31,18,0)
    aug2019_endMain2 = dt.datetime(2019,8,31,18,0)
    aug2019_inter_start = dt.datetime(2019,8,30,19,41)
    aug2019_inter_end = dt.datetime(2019,8,30,22,11)

    #Get time information based on given data type
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        if indata.empty:
            return indata, indata
        else:
            times = indata.index
    elif type(indata) == dict:
        times = [df for df in indata.values() if not df.empty][0].index

    #Determine where dividers are based on specific event
    if abs(times-feb2014_impact).min() < dt.timedelta(minutes=15):
        impact = feb2014_impact
        peak1 = feb2014_endMain1
        peak2 = feb2014_endMain2
        inter_start = feb2014_inter_start
        inter_end = feb2014_inter_end
    elif abs(times-starlink_impact).min() < dt.timedelta(minutes=15):
        impact = starlink_impact
        peak1 = starlink_endMain1
        peak2 = starlink_endMain2
        inter_start = starlink_inter_start
        inter_end = starlink_inter_end
    elif abs(times-may2019_impact).min() < dt.timedelta(minutes=15):
        impact = may2019_impact
        peak1 = may2019_endMain1
        peak2 = may2019_endMain2
        inter_start = may2019_inter_start
        inter_end = may2019_inter_end
    elif abs(times-aug2019_impact).min() < dt.timedelta(minutes=15):
        impact = aug2019_impact
        peak1 = aug2019_endMain1
        peak2 = aug2019_endMain2
        inter_start = aug2019_inter_start
        inter_end = aug2019_inter_end
    else:
        impact = times[0]
        peak1 = times[round(len(times)/2)]
        peak2 = times[round(len(times)/2)]
        #peak2 = times[-1]
        inter_start = peak1-dt.timedelta(minute=75)
        inter_end = peak1+dt.timedelta(minute=75)

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
    elif 'interv' in phasekey:
        cond = (times>inter_start) & (times<inter_end)

    #Reload data filtered by the condition
    if (type(indata) == pd.core.series.Series or
        type(indata) == pd.core.frame.DataFrame):
        rel_time = [dt.datetime(2000,1,1)+r for r in
                    indata[cond].index-indata[cond].index[0]]
        return indata[cond], rel_time
    elif type(indata) == dict:
        phase = indata.copy()
        for key in [k for k in indata.keys() if not indata[k].empty]:
            df = indata[key]
            phase.update({key:df[cond]})
            rel_time = [dt.datetime(2000,1,1)+r for r in
                        df[cond].index-df[cond].index[0]]
        return phase, rel_time

def plot_contour(ax,df,lower,upper,xkey,ykey,zkey,**kwargs):
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
    conplot = ax.contourf(X,Y,Z, cmap=kwargs.get('cmap','RdBu_r'),
                          levels=levs,extend='both',**contour_kw)
    cbarconplot = lshell.colorbar(conplot,ax=ax,label=safelabel(zkey),
                                  **cbar_kw)
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
    from IPython import embed; embed()
    time.sleep(3)
    norm = np.linalg.norm(np.array((U, V)), axis=0)
    quiver = ax.quiver(X,Y,U/norm,V/norm,angles='xy',scale=None)
            #cmap=kwargs.get('cmap','RdBu_r'),
            #              levels=levs,extend='both',**contour_kw)
    #cbarconplot = lshell.colorbar(conplot,ax=ax,label=safelabel(zkey),
    #                              **cbar_kw)
    general_plot_settings(ax, **kwargs)
    #ax.grid()

def stack_energy_region_fig(ds,ph,path,hatches):
    """Stack plot Energy by region
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
    #plot
    for i,ev in enumerate(ds.keys()):
        if not ds[ev]['mp'+ph].empty:
            plot_stack_contrib(ax[i],ds[ev]['time'+ph],ds[ev]['mp'+ph],
                               ds[ev]['msdict'+ph], legend=(i==0),
                               value_key='Utot2 [J]',label=ev,ylim=[0,15],
                               factor=1e15,
                               ylabel=r'Energy $\left[ J\right]$',
                               legend_loc='upper right', hatch=hatches[i],
                               do_xlabel=(i==len(ds.keys())-1))
    #save
    contr.tight_layout(pad=1)
    contr.savefig(path+'/contr_energy'+ph+'.png')
    plt.close(contr)

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

        #plot
        for i,ev in enumerate(ds.keys()):
            if not ds[ev]['mp'+ph].empty:
                plot_stack_distr(ax[i],ds[ev]['time'+ph],ds[ev]['mp'+ph],
                                 ds[ev]['msdict'+ph], value_set='Energy2',
                                 doBios=False, label=ev,
                                 ylabel=r'Energy $\left[ J\right]$',
                                 legend_loc='upper left',subzone=sz)
        #save
        distr.tight_layout(pad=1)
        distr.savefig(path+'/distr_energy'+ph+sz+'.png')
        plt.close(distr)

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
    #plot
    for i,ev in enumerate(ds.keys()):
        lobe = ds[ev]['msdict'+ph]['lobes']
        if not lobe.empty:
            ax[i].plot(ds[ev]['time'+ph],lobe['Area [Re^2]'],label=ev)
            general_plot_settings(ax[i],ylabel=r'Area $\left[ Re^2\right]$'
                                  ,do_xlabel=(i==len(ds.keys())-1),
                                  legend=True)
    #save
    pca.tight_layout(pad=1)
    pca.savefig(path+'/polar_cap_area_'+ph+'.png')
    plt.close(pca)

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
    #plot
    for i,ev in enumerate(ds.keys()):
        if not ds[ev]['mp'+ph].empty:
            plot_stack_contrib(ax[i],ds[ev]['time'+ph],ds[ev]['mp'+ph],
                                      ds[ev]['msdict'+ph],
                            value_key='Volume [Re^3]',label=ev,
                            legend=(i==0),
                            ylabel=r'Volume $\left[R_e^3\right]$',
                            legend_loc='upper right', hatch=hatches[i],
                            do_xlabel=(i==len(ds.keys())-1))
            #Calculate quiet time average value and plot as a h line
            rc = ds[ev]['msdict_qt']['rc']['Volume [Re^3]'].mean()
            close=ds[ev]['msdict_qt']['closed']['Volume [Re^3]'].mean()
            lobe=ds[ev]['msdict_qt']['lobes']['Volume [Re^3]'].mean()
            ax[i].axhline(rc,color='grey')
            ax[i].axhline(rc+close,color='grey')
            ax[i].axhline(rc+close+lobe,color='grey')
        #save
        contr.tight_layout(pad=1)
        contr.savefig(path+'/contr_volume'+ph+'.png')
        plt.close(contr)
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
    interface_list = ['Dayside_reg','Flank','PSB']
    ylims = [[-8,8],[-5,5],[-10,10]]
    closed = ['Dayside_reg','Tail_close','L7','PSB','MidLat']
    lobes = ['Flank','Tail_lobe','Poles']
    ring_c = ['LowLat']
    incr = 7.5
    h_ratios=[3,3,1,1,1,1,4,4,1,1,3,3,1,1,1,1,1,1]
    interf_fig, ax = plt.subplots(len(ds.keys())*len(interface_list),
                                      sharex=True,
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
            if not dic[sz].empty and('K_net'+interf+' [W]' in dic[sz]):
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
                    ax[len(ds.keys())*j+i].yaxis.set_label_position(
                                                                  'right')
    #save
    interf_fig.tight_layout(pad=0.2)
    interf_fig.savefig(path+'/interf'+ph+'.png')
    plt.close(interf_fig)


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
                          legend=False, iscontour=True)
    #save
    qt_bar.tight_layout(pad=1)
    qt_bar.savefig(outQT+'/quiet_bar_energy.png')
    plt.close(qt_bar)

def lshell_contour_figure(ds):
    ##Lshell plot
    for qty in ['Utot2','u_db','uHydro','Utot']:
        for ph,path in [('_main',outMN1),('_rec',outRec)]:
            #Plot contours of quantitiy as well as dLdt of quantity
            #for z in [qty+' [J]','dLdt_'+qty]:
            for z in ['dLdt_'+qty]:
                u = 'd'+qty+'dt'
                v = 'd'+qty+' [J]'
                #setup figure
                lshell,ax=plt.subplots(len(ds.keys()),1,sharey=False,
                                 sharex=True,figsize=[14,4*len(ds.keys())])
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
                                #lower = l_copy.min()
                                #upper = l_copy.max()
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
                                       'do_xlabel':(j==1)}
                            plot_quiver(ax[i+j],l_data[cond],
                                        low,up,
                                        't','L',u,v,
                                        **settings)
                            #plot_contour(ax[i+j],l_data[cond],
                            #             l,u,
                            #             't','L',z,
                            #             doLog=('dLdt_' not in z),
                            #             **settings)
                            if j==0 and(
                                    'X_subsolar [Re]'in ds[ev]['mp'+ph]):
                                ax[i+j].plot(mptimes,x_sub,color='white')
                #save
                lshell.tight_layout(pad=0.2)
                lshell.savefig(path+'/'+z.split(' ')[0]+'_lshell'+ph+'.png')
                plt.close(lshell)

def quiet_figures(ds):
    region_interface_averages(ds)

def main_rec_figures(ds):
    ##Main + Recovery phase
    hatches = ['','*','x','o']
    for ph,path in [('_main',outMN1),('_rec',outRec)]:
        stack_energy_type_fig(ds,ph,path)
        stack_energy_region_fig(ds,ph,path,hatches)
        stack_volume_fig(ds,ph,path,hatches)
        interf_power_fig(ds,ph,path,hatches)
        polar_cap_area_fig(ds,ph,path)

def interval_figures(ds):
    hatches = ['','*','x','o']
    for ph,path in [('_interv',outInterv)]:
        stack_energy_type_fig(ds,ph,path)
        stack_energy_region_fig(ds,ph,path,hatches)
        stack_volume_fig(ds,ph,path,hatches)
        interf_power_fig(ds,ph,path,hatches)
        polar_cap_area_fig(ds,ph,path)

def lshell_figures(ds):
    #lshell_contour_figure(ds)
    pass

def bonus_figures(ds):
    pass

def solarwind_figures(ds):
    pass

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
    outInterv = os.path.join(outPath,'interval')
    for path in [outPath,outQT,outSSC,outMN1,outMN2,
                 outRec,unfiled,outInterv]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print_presentation'))

    #HDF data, will be sorted and cleaned
    ds = {}
    ds['feb'] = load_hdf_sort(inPath+'feb2014_results.h5')
    #ds['feb'] = load_hdf_sort(inPath+'feb2014_lshell3.h5')
    ds['star'] = load_hdf_sort(inPath+'starlink_results.h5')
    ds['may'] = load_hdf_sort(inPath+'may2019_results.h5')
    #ds['may'] = load_hdf_sort(inPath+'may2019_lshell.h5')
    #ds['aug'] = load_hdf_sort(inPath+'aug2019_results.h5')

    '''
    #Log files and observational indices
    ds['feb']['obs'] = read_indices(inPath, prefix='feb2014_',
                                    read_supermag=False, tshift=45)
    ds['star']['obs'] = read_indices(inPath, prefix='starlink_',
                                     read_supermag=False)
    #ds['may']['obs'] = read_indices(inPath, prefix='may2019_',
    #                                read_supermag=False)
    #ds['aug']['obs'] = read_indices(inPath, prefix='aug2019_',
    #                                read_supermag=False)
    '''

    #NOTE hotfix for closed region tail_closed
    for ev in ds.keys():
        for t in[t for t in ds[ev]['msdict']['closed'].keys()
                                                    if 'Tail_close'in t]:
            ds[ev]['msdict']['closed'][t] = ds[ev]['mpdict']['ms_full'][t]

    ##Construct "grouped" set of subzones, then get %contrib for each
    for event in ds.keys():
        #ds[event]['mpdict'],ds[event]['msdict'] = get_subzone_contrib(
        #                                               ds[event]['mpdict'],
        #                                               ds[event]['msdict'])
        ds[event]['msdict'] = {
                'rc':ds[event]['msdict'].get('rc',pd.DataFrame()),
                'closed':ds[event]['msdict'].get('closed',pd.DataFrame()),
                'lobes':ds[event]['msdict'].get('lobes',pd.DataFrame())}
                               #'missed':ds[event]['msdict']['missed']}
    ##Parse storm phases
    for ev in ds.keys():
        #obs_srcs = list(ds[ev]['obs'].keys())
        for ph in ['_qt','_main','_rec','_interv']:
            ds[ev]['mp'+ph], ds[ev]['time'+ph] = locate_phase(
                                            ds[ev]['mpdict']['ms_full'],ph)
            ds[ev]['msdict'+ph], _ = locate_phase(ds[ev]['msdict'],ph)
            #for src in obs_srcs:
            #    ds[ev]['obs'][src+ph],ds[ev][src+'_otime'+ph]=(
            #                   locate_phase(ds[ev]['obs'][src],ph))

    ######################################################################
    ##Quiet time
    #quiet_figures(ds)
    ######################################################################
    ##Main + Recovery phase
    #main_rec_figures(ds)
    ######################################################################
    ##Short zoomed in interval
    interval_figures(ds)
    ######################################################################
    ##Lshell plots
    #lshell_figures(ds)
    ######################################################################
    ##Bonus plot
    #bonus_figures(ds)
    ######################################################################
    #Series of solar wind observatioins/inputs/indices
    #solarwind_figures(ds)
    ######################################################################
    """
    ######################################################################
    ##Bonus plot
    #setup figure
    bonus, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    #plot_pearson_r(ax, tx, ty, xseries, yseries, **kwargs):
    for i,ev in enumerate([k for k in ds.keys()
                           if not (ds[k]['msdict']['lobes'].empty)]):
        if i==1:ylabel=''
        else:ylabel=r'Lobe Power/Energy'
        for j,ph in enumerate(['_main','_rec']):
            dic = ds[ev]['msdict'+ph]['lobes']
            obs = ds[ev]['obs']['swmf_sw'+ph]
            if j>0:
                times = [t+(times[-1]-times[0]) for t in ds[ev]['time'+ph]]
                ot = [t+(ot[-1]-ot[0]) for t in ds[ev]['swmf_sw_otime'+ph]]
                h='_'
            else:
                times = ds[ev]['time'+ph]
                ot    = ds[ev]['swmf_sw_otime'+ph]
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
                    legend=(i==0),ylim=[-6,6])
    #save
    bonus.tight_layout(pad=1)
    bonus.savefig(outRec+'/bonus.png')
    plt.close(bonus)

    ######################################################################
    #Series of solar wind observatioins/inputs/indices
    #IMF
    imf, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    for i,ev in enumerate(ds.keys()):
        orange = dt.timedelta(minutes=0)
        for j,ph in enumerate(['_qt','_main','_rec']):
            obs = ds[ev]['obs']['swmf_sw'+ph]
            ot = [t+orange for t in ds[ev]['swmf_sw_otime'+ph]]
            orange += ot[-1]-ot[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].fill_between(ot,obs['B'], ec='dimgrey',fc='thistle',
                               hatch=hatches[i], label=h+r'$|B|$')
            ax[i].plot(ot,obs['bx'],label=h+r'$B_x$',c='maroon')
            ax[i].plot(ot,obs['by'],label=h+r'$B_y$',c='magenta')
            ax[i].plot(ot,obs['bz'],label=h+r'$B_z$',c='tab:blue')
        general_plot_settings(ax[i],ylabel=r'$B\left[nT\right]$'+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    imf.tight_layout(pad=0.8)
    imf.savefig(unfiled+'/imf.png')
    plt.close(imf)

    #Pdyn and Vxyz
    pVel, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    for i,ev in enumerate(ds.keys()):
        orange = dt.timedelta(minutes=0)
        for j,ph in enumerate(['_qt','_main','_rec']):
            obs = ds[ev]['obs']['swmf_sw'+ph]
            ot = [t+orange for t in ds[ev]['swmf_sw_otime'+ph]]
            orange += ot[-1]-ot[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].fill_between(ot,obs['pdyn']*10,ec='dimgrey',fc='thistle',
                               hatch=hatches[i],label=h+r'10x$P_{dyn}$')
            ax2 = ax[i].twinx()
            ax2.plot(ot,-1*obs['vx'],label=h+r'$V_x$',c='maroon')
            ax2.set_ylim([0,800])
            ax[i].plot(ot,obs['vy'],label=h+r'$V_y$',c='magenta')
            ax[i].plot(ot,obs['vz'],label=h+r'$V_z$',c='tab:blue')
        general_plot_settings(ax[i],ylabel=r'$V,P\left[km,nPa\right]$'+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    pVel.tight_layout(pad=0.8)
    pVel.savefig(unfiled+'/pVel.png')
    plt.close(pVel)

    #Alfven Mach number and plasma beta
    betaMa, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    for i,ev in enumerate(ds.keys()):
        orange = dt.timedelta(minutes=0)
        for j,ph in enumerate(['_qt','_main','_rec']):
            obs = ds[ev]['obs']['swmf_sw'+ph]
            ot = [t+orange for t in ds[ev]['swmf_sw_otime'+ph]]
            orange += ot[-1]-ot[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].plot(ot,obs['Ma'],label=h+r'$M_{Alf}$',c='magenta')
            ax[i].plot(ot,obs['Beta'],label=h+r'$\beta$',c='tab:blue')
        general_plot_settings(ax[i],ylabel=r'$M_{Alf},\beta$'+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    betaMa.tight_layout(pad=0.8)
    betaMa.savefig(unfiled+'/betaMa.png')
    plt.close(betaMa)

    #Dst index
    dst, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    for i,ev in enumerate(ds.keys()):
        srange = dt.timedelta(minutes=0)
        orange = dt.timedelta(minutes=0)
        for j,ph in enumerate(['_qt','_main','_rec']):
            sim = ds[ev]['obs']['swmf_log'+ph]
            obs = ds[ev]['obs']['omni'+ph]
            st = [t+srange for t in ds[ev]['swmf_log_otime'+ph]]
            ot = [t+orange for t in ds[ev]['omni_otime'+ph]]
            srange += st[-1]-st[0]
            orange += ot[-1]-ot[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].plot(ot,obs['sym_h'],label=h+r'SYM-H(OMNI)',c='magenta')
            ax[i].plot(st,sim['dst_sm'],label=h+r'sim',c='tab:blue')
        general_plot_settings(ax[i],ylabel=r'$\Delta B\left[nT\right]$ '+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    dst.tight_layout(pad=0.8)
    dst.savefig(unfiled+'/dst.png')
    plt.close(dst)

    #Magnetopause standoff
    mpStan, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    for i,ev in enumerate(ds.keys()):
        trange = dt.timedelta(minutes=0)
        srange = dt.timedelta(minutes=0)
        for j,ph in enumerate(['_qt','_main','_rec']):
            mp = ds[ev]['mp'+ph]
            sw = ds[ev]['obs']['swmf_sw'+ph]
            if not mp.empty:
                times = [t+trange for t in ds[ev]['time'+ph]]
                trange += times[-1]-times[0]
            st = [t+srange for t in ds[ev]['swmf_sw_otime'+ph]]
            srange += st[-1]-st[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].plot(st,sw['r_shue98'],label=h+r'Shue98',c='magenta')
            if not mp.empty:
                ax[i].plot(times,mp['X_subsolar [Re]'],label=h+r'sim',
                           c='tab:blue')
        general_plot_settings(ax[i],ylabel=r'Standoff $R_e$'+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    mpStan.tight_layout(pad=0.8)
    mpStan.savefig(unfiled+'/mpStan.png')
    plt.close(mpStan)

    #Coupling functions and CPCP
    coupl, ax = plt.subplots(len(ds.keys()),1,sharey=True,sharex=True,
                                          figsize=[14,4*len(ds.keys())])
    months = [2,2,5,9]
    for i,ev in enumerate(ds.keys()):
        lrange = dt.timedelta(minutes=0)
        srange = dt.timedelta(minutes=0)
        orange = dt.timedelta(minutes=0)
        ax2 = ax[i].twinx()
        for j,ph in enumerate(['_qt','_main','_rec']):
            log = ds[ev]['obs']['swmf_log'+ph]
            sw = ds[ev]['obs']['swmf_sw'+ph]
            obs = ds[ev]['obs']['omni'+ph]
            lt = [t+lrange for t in ds[ev]['swmf_log_otime'+ph]]
            st = [t+srange for t in ds[ev]['swmf_sw_otime'+ph]]
            ot = [t+orange for t in ds[ev]['omni_otime'+ph]]
            lrange += lt[-1]-lt[0]
            srange += st[-1]-st[0]
            orange += ot[-1]-ot[0]
            if j==0:
                h=''
            else:
                h='_'
            ax[i].plot(st,sw['Newell']/1e3,label=h+r'Newell',c='magenta')
            T = 2*pi*(months[i]/12)
            ax[i].plot(ot,29.28 - 3.31*sin(T+1.49)+17.81*obs['pc_n'],
                       label=h+r'Ridley and Kihn',c='black')
            ax[i].set_ylim([0,250])
            try:
                ax[i].plot(lt,log['cpcpn'],label=h+r'cpcpN',c='tab:blue')
                ax[i].plot(lt,log['cpcps'],label=h+r'cpcpS',c='lightblue')
            except: KeyError
            ax2.plot(st,sw['eps']/1e12,label=h+r'$\epsilon$',c='maroon')
            ax2.spines['right'].set_color('maroon')
            ax2.set_ylabel(r'$\epsilon\left[TW\right]$')
        general_plot_settings(ax[i],
                              ylabel=r'Potential $\left[kV\right]$ '+ev,
                              legend=(i==0),
                              do_xlabel=(i==len(ds.keys())-1))
    #save
    coupl.tight_layout(pad=0.8)
    coupl.savefig(unfiled+'/coupl.png')
    plt.close(coupl)
    """
