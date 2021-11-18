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
                                                  plot_swflowP,plot_swbz,
                                                           plot_dst)

def pyplotsetup(*, mode='presentation', **kwargs):
    """Creates dictionary to send to rcParams for pyplot defaults
    Inputs
        mode(str)- default is "presentation"
        kwargs:
    """
    #Always
    settings={"text.usetex": True,
              "font.family": "sans-serif",
              "font.size": 18,
              "font.sans-serif": ["Helvetica"]}
    if 'presentation' in mode:
        #increase lineweights
        settings.update({'lines.linewidth': 3})
    if 'print' in mode:
        #Change colorcycler
        colorwheel = plt.cycler('color',
                ['tab:blue', 'tab:orange', 'tab:pink', 'tab:brown',
                 'tab:olive','tab:cyan'])
        settings.update({'axes.prop_cycle': colorwheel})
    elif 'digital' in mode:
        #make non white backgrounds, adjust borders accordingly
        settings.update({'axes.edgecolor': 'white',
                         'axes.facecolor': 'grey'})
        #Change colorcycler
        colorwheel = plt.cycler('color',
                   ['cyan', 'magenta', 'peachpuff', 'chartreuse', 'wheat',
                    'lightgrey', 'springgreen', 'coral', 'plum', 'salmon'])
        settings.update({'axes.prop_cycle': colorwheel})
    return settings

def general_plot_settings(ax, ylabel, **kwargs):
    """Sets a bunch of general settings
    Inputs
        ax
        kwargs:
            do_xlabel(boolean)- default False
            color(str)- see matplotlib named colors
            legend_loc(see pyplot)- 'upper right' etc
    """
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(r'Time $\left[ UTC\right]$')
    #if kwargs.get('ylim',None) is not None:
    ax.set_ylim(kwargs.get('ylim',None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    ax.tick_params(which='major', length=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(ylabel)
    ax.legend(loc=kwargs.get('legend_loc',None))

def plot_single(ax, times, mp, msdict, ylabel, **kwargs):
    """Plots energy or dst contribution in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
            subzone(str)- 'ms_full' is default
            value_key(str)- 'Virial_Dst [nT]' default
            color(str)- see matplotlib named colors
    """
    #Gather info
    subzone = kwargs.get('subzone', 'ms_full')
    value = kwargs.get('value_key', 'Virial Volume Total [nT]')
    #Set data
    if 'ms_full' in subzone: data = mp
    else: data = msdict[subzone]
    #Set label
    if 'Virial' in value:
        label='Virial'
        if 'closedRegion' in subzone or 'lobes' in subzone:
            label=label+' (mod)'
    elif 'BioS' in value: label='Biot Savart'
    else: label = None
    #Plot
    if value in data.keys():
        ax.plot(times, data[value], label=label,
                color=kwargs.get('color',None))
    #General plot settings
    general_plot_settings(ax, ylabel, **kwargs)

def plot_distr(ax, times, mp, msdict, ylabel, **kwargs):
    """Plots distribution of energies in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            legend_loc(see pyplot)- 'upper right' etc
            subzone(str)- 'ms_full' is default
            value_set(str)- 'Virialvolume', 'Energy', etc
            ylim(tuple or list)- None
    """
    #Figure out value_keys and subzone
    subzone = kwargs.get('subzone', 'ms_full')
    if 'ms_full' in subzone:
        data = mp
    else:
        data = msdict[subzone]
    value_set = kwargs.get('value_set','Virialvolume')
    values = {'Virialvolume':['Virial 2x Uk [nT]', 'Virial Ub [nT]'],
              'Energy':['uB_dist [J]', 'KE [J]', 'Etherm [J]'],
              'Virialvolume%':['Virial 2x Uk [%]', 'Virial Ub [%]'],
              'Energy%':['uB_dist [%]', 'KE [%]', 'Etherm [%]']}
    #Get % within specific subvolume
    if '%' in value_set:
        for value in values[''.join(value_set.split('%'))]:
                #find total
                if 'Virial' in value_set:
                    if 'volume' in value_set:
                        total = data['Virial Volume Total [nT]']
                    else:
                        total = data['Virial_Dst [nT]']
                elif 'Energy' in value_set:
                    total = (data['Utot [J]']-data['uBtot [J]']+
                             data['uB_dist [J]'])
                data[value.split('[')[0]+'[%]'] = data[value]/total*100
    #Optional layers depending on value_key
    if not '%' in value_set:
        if 'BioS' in value_set:
            ax.fill_between(times, mp['BioS full [nT]'], color='olive')
            ax.plot(times, mp['Virial_Dst [nT]'], color='black', ls='--',
                    label='Virial')
            try:
                ax.plot(omni['Time [UTC]'],omni['sym_h'],
                        color='seagreen',
                        ls='--', label='OMNI obs')
            except NameError:
                print('omni not loaded! No obs comparisons!')
        if 'Virial' in value_set:
            if 'volume' in value_set:
                ax.fill_between(times, data['Virial Volume Total [nT]'],
                                color='lightgrey')
            else:
                ax.fill_between(times, data['Virial_Dst [nT]'],
                                color='lightgrey')
        elif 'Energy' in value_set:
                total = (data['Utot [J]']-data['uBtot [J]']+
                         data['uB_dist [J]'])
                ax.fill_between(times, total, color='lightgrey')
    #Plot line plots
    for value in values[value_set]:
        if value in data.keys():
            safelabel = ' '.join(value.split('_')).split('[%]')[0]
            ax.plot(times, data[value], label=safelabel)
    #General plot settings
    general_plot_settings(ax, ylabel, **kwargs)

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
    #General plot settings
    general_plot_settings(ax, ylabel, **kwargs)

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
            #ax.fill_between(times, mp['BioS full [nT]'], color='olive')
            #ax.plot(times, mp['Virial_Dst [nT]'], color='tab:red', ls='--',
            #        label='Virial')
            ax.fill_between(times, mp[value_key], color='gainsboro')
        elif 'Virial' in value_key:
            #ax.plot(times, mp['BioS [nT]'], color='tab:cyan', ls='--',
            #        label='Biot Savart')
            ax.fill_between(times, mp[value_key], color='gainsboro')
        try:
            ax.plot(omni['Time [UTC]'],omni['sym_h'],color='seagreen',
                    ls='--', label='OMNI obs')
        except NameError:
            print('omni not loaded! No obs comparisons!')
        #ax.fill_between(times, mp[value_key], color='lightgrey')
    #Plot line plots
    for ms in msdict.items():
        if value_key in ms[1].keys():
            ax.plot(times, ms[1][value_key], label=ms[0])
    #Optional plot settings
    if ('BioS' in value_key) and ('%' in value_key):
        ax.set_ylim([-100,100])
    #General plot settings
    general_plot_settings(ax, ylabel, **kwargs)

def get_interzone_stats(mpdict, msdict, **kwargs):
    """Function finds percent contributions and missing amounts
    Inputs
        mpdict(DataFrame)-
        msdict(Dict of DataFrames)-
    Returns
        [MODIFIED] mpdict(Dict of DataFrames)
        [MODIFIED] msdict(Dict of DataFrames)
    """
    #Add total energy
    for m in mpdict.values():
        m['Utot [J]'] = m['uBtot [J]']+m['KE [J]']+m['Etherm [J]']
    for m in msdict.values():
        m['Utot [J]'] = m['uBtot [J]']+m['KE [J]']+m['Etherm [J]']
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
    plt.rcParams.update(pyplotsetup(mode='print_presentation'))
    [swmf_index, swmf_log, swmf_sw,_,omni]= read_indices(datapath,
                                                       read_supermag=False)
    cuttoffstart = dt.datetime(2014,2,18,6,0)
    cuttoffend = dt.datetime(2014,2,20,0,0)
    simdata = [swmf_index, swmf_log, swmf_sw]
    [swmf_index,swmf_log,swmf_sw] = chopends_time(simdata, cuttoffstart,
                                      cuttoffend, 'Time [UTC]', shift=True)
    store = pd.HDFStore(datapath+'energetics.h5')
    #store = pd.HDFStore(datapath+'halfbaked.h5')
    times = store[store.keys()[0]]['Time [UTC]']+dt.timedelta(minutes=45)
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
    #Second type: distribution of types of energy in the system
    ######################################################################
    y1labels = [r'Total Magnetic Energy $\left[J\right]$',
                r'Disturbance Magnetic Energy $\left[J\right]$',
                r'Dipole Magnetic Energy $\left[J\right]$',
                r'Kinetic Energy $\left[J\right]$',
                r'Thermal Energy $\left[J\right]$',
                r'Total Energy $\left[J\right]$']
    y2label = r'Energy fraction $\left[\%\right]$'
    energies = ['uBtot','uB_dist','uB_dipole','KE','Etherm','Utot']
    fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig2,ax2 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig3,ax3 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig4,ax4 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig5,ax5 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig6,ax6 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    fig7,ax7 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    ax = [ax1,ax2,ax3,ax4,ax5,ax6]
    for fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6]):
        plot_contributions(ax[fig[0]][0],times,mp,msdict_3zone,
                       y1labels[fig[0]], value_key=energies[fig[0]]+' [J]')
        plot_contributions(ax[fig[0]][1], times, mp, msdict_3zone, y2label,
                         value_key=energies[fig[0]]+' [%]', do_xlabel=True)
        fig[1].tight_layout(pad=1)
        fig[1].savefig(figureout+energies[fig[0]]+'_line.png')
        plt.close(fig[1])
    plot_contributions(ax7[0], times, mp, msdict_3zone,
                    r'Volume $\left[R_e\right]$',value_key='Volume [Re^3]')
    plot_contributions(ax7[1], times, mp, msdict_3zone,
               r'Volume fraction $\left[\%\right]$',value_key='Volume [%]')
    fig7.tight_layout(pad=1)
    fig7.savefig(figureout+'Volume_line.png')
    plt.close(fig7)
    #Third type: distrubiton of virial and types of energy within subzone
    ######################################################################
    y2label = r'Fraction $\left[\%\right]$'
    valset = ['Virialvolume','Energy']
    for vals in valset:
        if 'Virial' in vals:
            fig2ylabels = {'rc':r'Ring Current Dst $\left[nT\right]$',
                    'closedRegion':r'Closed Region Dst $\left[nT\right]$',
                           'lobes':r'Lobes Dst $\left[nT\right]$'}
        elif 'Energy' in vals:
            fig2ylabels = {'rc':r'Ring Current Energy $\left[J\right]$',
                    'closedRegion':r'Closed Region Energy $\left[J\right]$',
                           'lobes':r'Lobes Energy $\left[J\right]$'}
        y1labels = ['Full MS '+vals+' Distribution']
        fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig2,ax2 = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,14])
        plot_distr(ax1[0],times,mp,msdict_3zone,y1labels[0],value_set=vals)
        plot_distr(ax1[1], times, mp, msdict_3zone, y2label,
                   value_set=vals+'%', do_xlabel=True)
        fig1.tight_layout(pad=1)
        fig1.savefig(figureout+'distr_'+vals+'_fullMS_line.png')
        for subzone in enumerate([k for k in msdict_3zone.keys()][0:-1]):
            y1labels = [subzone[1]+' '+vals+' Distribution']
            fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
            plot_distr(ax[0],times,mp,msdict_3zone,y1labels[0],
                       value_set=vals, subzone=subzone[1])
            plot_distr(ax[1], times, mp, msdict_3zone, y2label,
                   value_set=vals+'%', do_xlabel=True, subzone=subzone[1])
            plot_distr(ax2[subzone[0]],times,mp,msdict_3zone,
                fig2ylabels[subzone[1]],value_set=vals,subzone=subzone[1],
                       ylim=[0,8e15])
                       #do_xlabel=(subzone[0] is len(msdict_3zone)-2))
            fig.tight_layout(pad=1)
            fig.savefig(figureout+'distr_'+vals+subzone[1]+'_line.png')
            plt.close(fig)
        plot_swbz(ax2[3],[swmf_sw],'Time [UTC]',r'$B_z \left[nT\right]$')
        plot_swflowP(ax2[3].twinx(),[swmf_sw],'Time [UTC]',
                     r'$P_{ram} \left[nT\right]$')
        ax2[3].set_xlabel(r'Time $\left[ UTC\right]$')
        ax2[3].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2[3].tick_params(which='major', length=9)
        ax2[3].xaxis.set_minor_locator(AutoMinorLocator(6))
        fig2.tight_layout(pad=1)
        fig2.savefig(figureout+'distr_'+vals+'eachzone_line.png')
        plt.close(fig2)
    #Fourth type: virial vs Biot Savart stand alones
    ######################################################################
    ylabels = {'rc':r'Ring Current Dst $\left[nT\right]$',
               'closedRegion':r'Closed Region Dst $\left[nT\right]$',
               'lobes':r'Lobes Dst $\left[nT\right]$',
               'missing':r'Unaccounted Dst $\left[nT\right]$'}
    fig, ax = plt.subplots(nrows=len(msdict_3zone), ncols=1, sharex=True,
                           figsize=[14,4*len(msdict_3zone)])
    #Hotfix, create "modified" virial contributions with volumetric
    # proportions of surface total contributions
    wetted_volumes = (msdict_3zone['closedRegion']['Volume [Re^3]']+
                      msdict_3zone['lobes']['Volume [Re^3]'])
    msdict_3zone['closedRegion']['Virial Volume Total_mod'] = (
                msdict_3zone['closedRegion']['Virial Volume Total [nT]']+
                mp['Virial Surface Total [nT]']*
              msdict_3zone['closedRegion']['Volume [Re^3]']/wetted_volumes)
    msdict_3zone['lobes']['Virial Volume Total_mod'] = (
                        msdict_3zone['lobes']['Virial Volume Total [nT]']+
                mp['Virial Surface Total [nT]']*
                    msdict_3zone['lobes']['Volume [Re^3]']/wetted_volumes)
    '''
    msdict_3zone['closedRegion']['Virial Volume Total_mod'] = (
                msdict_3zone['closedRegion']['Virial Volume Total [nT]'])
    msdict_3zone['lobes']['Virial Volume Total_mod'] = (
                        msdict_3zone['lobes']['Virial Volume Total [nT]']+
                mp['Virial Surface Total [nT]'])
    '''
    msdict_3zone['rc']['Virial Volume Total_mod'] = (
                            msdict_3zone['rc']['Virial Volume Total [nT]'])
    msdict_3zone['missing']['Virial Volume Total_mod'] = (
                       msdict_3zone['missing']['Virial Volume Total [nT]'])
    mp['Virial Volume Total_mod'] = (mp['Virial Volume Total [nT]']+
                                      mp['Virial Surface Total [nT]'])
    for subzone in enumerate(msdict_3zone):
        plot_single(ax[subzone[0]], times, mp, msdict_3zone,
                   ylabels[subzone[1]],
                   subzone=subzone[1],value_key='Virial Volume Total_mod')
        plot_single(ax[subzone[0]], times, mp, msdict_3zone,
             ylabels[subzone[1]], subzone=subzone[1],value_key='BioS [nT]',
                    do_xlabel=(subzone[0] is len(msdict_3zone)-1))
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'compare_Total_line.png')
    plt.close(fig)
    #Fifth type: virial vs Biot Savart one nice imag with obs dst
    ######################################################################
    y1label = r'Virial $\Delta B\left[ nT\right]$'
    y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    '''
    msdict_3zone.pop('missing')
    plot_contributions(ax[1], times,mp,msdict_3zone,y2label,
                               value_key='BioS [nT]', do_xlabel=True)
    ax[0].plot(times,
               msdict_3zone['closedRegion']['Virial Volume Total [nT]'],
               label='Closed region')
    ax[0].plot(times,msdict_3zone['lobes']['Virial Volume Total [nT]'],
               label='lobes')
    msdict_3zone.pop('lobes')
    msdict_3zone.pop('closedRegion')
    plot_contributions(ax[0], times,mp,msdict_3zone,y1label,
                               value_key='Virial Volume Total_mod',
                               legend_loc='lower left')
    ax[0].plot(times,mp['Virial Surface Total [nT]'],
               label='Surface')
    '''
    msdict_3zone.pop('missing')
    plot_contributions(ax[0], times,mp,msdict_3zone,y1label,
                               value_key='Virial Volume Total_mod',
                               legend_loc='lower left')
    plot_contributions(ax[1], times,mp,msdict_3zone,y2label,
                               value_key='BioS [nT]', do_xlabel=True)
    ax[0].set_ylim([-125,50])
    ax[1].set_ylim([-125,50])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'pretty_Dst_line.png')
    plt.close(fig)

