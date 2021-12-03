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

def pyplotsetup(*,mode='presentation',**kwargs):
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
            value_key(str)- 'Virial [nT]' default
            color(str)- see matplotlib named colors
    """
    #Gather info
    subzone = kwargs.get('subzone', 'ms_full')
    value = kwargs.get('value_key', 'Virial [nT]')
    #Set data
    if 'ms_full' in subzone: data = mp
    else: data = msdict[subzone]
    #Set label
    if 'Virial' in value:
        label='Virial'
    elif 'bioS' in value: label='Biot Savart'
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
    values = {'Virial':['Virial 2x Uk [nT]', 'Virial Ub [nT]','Um [nT]',
                        'Virial Surface Total [nT]'],
              'Energy':['Virial Ub [J]', 'KE [J]', 'Eth [J]'],
              'Virial%':['Virial 2x Uk [%]', 'Virial Ub [%]',
                               'Virial Surface Total [%]'],
              'Energy%':['Virial Ub [%]', 'KE [%]', 'Eth [%]']}
    #Get % within specific subvolume
    if '%' in value_set:
        for value in values[''.join(value_set.split('%'))]:
                #find total
                if 'Virial' in value_set:
                    total = data['Virial [nT]']
                elif 'Energy' in value_set:
                    total = (data['Utot [J]']-data['uB [J]']+
                             data['Virial Ub [J]'])
                data[value.split('[')[0]+'[%]'] = data[value]/total*100
    #Optional layers depending on value_key
    if not '%' in value_set:
        if 'bioS' in value_set:
            ax.fill_between(times, mp['bioS_full [nT]'], color='olive')
            ax.plot(times, mp['Virial [nT]'], color='black', ls='--',
                    label='Virial')
            try:
                ax.plot(omni['Time [UTC]'],omni['sym_h'],
                        color='seagreen',
                        ls='--', label='OMNI obs')
            except NameError:
                print('omni not loaded! No obs comparisons!')
        if 'Virial' in value_set:
            ax.fill_between(times, data['Virial [nT]'],
                            color='lightgrey')
        elif 'Energy' in value_set:
                total = (data['Utot [J]']-data['uB [J]']+
                         data['Virial Ub [J]'])
                ax.fill_between(times, total, color='lightgrey')
    #Plot line plots
    for value in values[value_set]:
        if value in data.keys():
            safelabel = ' '.join(value.split('_')).split('[%]')[0]
            #ax.plot(times, data[value], label=safelabel)
            ax.scatter(times, data[value], label=safelabel)
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
        if (not 'bioS' in value_key) and ('Volume Total' in value_key):
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
    if ('bioS' in value_key) and ('%' in value_key):
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
        if 'bioS' in value_key:
            ax.fill_between(times, mp['bioS_full [nT]'], color='olive')
        elif 'Virial' in value_key:
            pass
        if 'bioS' in value_key or 'Virial' in value_key:
            try:
                ax.plot(omni['Time [UTC]'],omni['sym_h'],color='seagreen',
                    ls='--', label='OMNI obs')
            except NameError:
                print('omni not loaded! No obs comparisons!')
            try:
                ax.plot(swmf_log['Time [UTC]'], swmf_log['dst_sm'],
                    color='tab:red', ls='--',
                    label='SWMFLog BiotSavart')
                ax.plot(swmf_log['Time [UTC]'], swmf_log['dstflx_R=3.0'],
                    color='magenta', ls='--',
                    label='SWMFLog Flux')
            except NameError:
                print('swmf log file not loaded!')
        ax.fill_between(times, mp[value_key], color='gainsboro')
    #Plot line plots
    for ms in msdict.items():
        if value_key in ms[1].keys():
            ax.plot(times, ms[1][value_key], label=ms[0])
    #Optional plot settings
    if ('bioS' in value_key) and ('%' in value_key):
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
    mp = [m for m in mpdict.values()][0]
    for m in msdict.items():
        #Remove empty/uneccessary columns from subvolumes
        for key in m[1].keys():
            if all(m[1][key].isna()):
                m[1].drop(columns=[key], inplace=True)
        #Total energy
        if 'uB' in m[1].keys():
            m['Utot [J]'] = m['uB [J]']+m['KE [J]']+m['Eth [J]']
        #Total virial contribution from surface and volume terms
        if 'Virial 2x Uk [J]' in m[1].keys():
            if 'nlobe' in m[0]:
                m[1]['Virial Surface Total [J]'] = (
                                       mp['Virial Surface TotalOpenN [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            elif 'slobe' in m[0]:
                m[1]['Virial Surface Total [J]'] = (
                                       mp['Virial Surface TotalOpenS [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            elif 'qDp' in m[0]:
                m[1]['Virial Surface Total [J]'] = (
                                      mp['Virial Surface TotalClosed [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            else:
                m[1]['Virial [J]'] = m[1]['Virial Volume Total [J]']
                m[1]['Virial Surface Total [J]'] = 0
            m[1]['Virial Surface Total [nT]'] = (
                                  m[1]['Virial Surface Total [J]']/(-8e13))
            m[1]['Virial [nT]'] = m[1]['Virial [J]']/(-8e13)
            msdict.update({m[0]:m[1]})
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

def calculate_mass_term(df):
    """Calculates the temporal derivative minus surface motion contribution
        to virial term
    Inputs
        df(DataFrame)
    Returns
        df(DataFrame)- modified
    """
    #Forward difference of integrated positionally weighted density
    df['Um_static [J]'] = -1*df['rhoU_r [Js]']/60
    f_n0 = df['rhoU_r [Js]']
    f_n1 = f_n0.copy()
    f_n1.index=f_n1.index-1
    f_n1 = f_n1.drop(index=[-1])
    forward_diff = (f_n1-f_n0)/(60)
    #Save as energy and as virial
    df['Um_static [J]'] = -1*forward_diff
    df['Um_static [nT]'] = -1*forward_diff/(-8e13)
    if 'rhoU_r_net [J]' in df.keys():
        #Get static + motional
        df['Um [J]'] = df['Um_static [J]']+df['rhoU_r_net [J]']
        df['Um [nT]'] = df['Um [J]']/(-8e13)
    else:
        df['Um [J]'] = df['Um_static [J]']
        df['Um [nT]'] = df['Um_static [nT]']
    return df

def virial_mods(df):
    """Function returns DataFrame with modified virial columns
    Inputs
        df(DataFrame)
    Returns
        df(DataFrame)- modified
    """
    #!!Correct pressure term: should be 3x current value (trace of P=3p)
    df['Virial 2x Uk [J]'] = 3*df['Virial 2x Uk [J]']
    if 'rhoU_r [Js]' in df.keys():
        df = calculate_mass_term(df)
        df['Virial Volume Total [J]'] = (df['Virial Volume Total [J]']+
                                         df['Um [J]'])
        df['Virial Volume Total [nT]'] = (df['Virial Volume Total [J]']/
                                                                   (-8e13))
    if (('Virial Volume Total [J]' in df.keys()) and
        ('Virial Surface Total [J]' in df.keys())):
        df['Virial [J]'] = (df['Virial Volume Total [J]']+
                                   df['Virial Surface Total [J]'])
    for key in df.keys():
        if '[J]' in key and key.split(' [J]')[0]+' [nT]' not in df.keys():
            df[key.split(' [J]')[0]+' [nT]'] = df[key]/(-8e13)
    return df

def magnetopause_energy_mods(mpdf):
    """Function returns magnetopause DataFrame with modified energy columns
    Inputs
        mpdf(DataFrame)
    Returns
        mpdf(DataFrame)- modified
    """
    #One-off renames
    if 'Utot_acqu[W]' in mpdf.keys():
        mpdf.rename(columns={'Utot_acqu[W]':'Utot_acqu [W]',
                             'Utot_forf[W]':'Utot_forf [W]',
                             'Utot_net[W]':'Utot_net [W]'}, inplace=True)
    #Define relevant pieces
    fluxes, energies = ['K_','ExB_','P0_'], ['Utot_', 'uB_', 'uHydro_']
    direct = ['injection', 'escape', 'net']
    locations = ['', 'Day', 'Flank', 'Tail', 'OpenN', 'OpenS', 'Closed']
    motion = ['acqu', 'forf', 'net']
    u = ' [W]'#units
    #One-off missing keys
    if 'Utot_acquDay [W]' not in mpdf.keys():
        for loc in locations:
            for m in motion:
                mpdf['Utot_'+m+loc+u] = (mpdf['uB_'+m+loc+u]+
                                         mpdf['uHydro_'+m+loc+u])
    #Surface Volume combinations
    for flux in enumerate(fluxes):
        for d in enumerate(direct):
            for loc in locations:
                #Rename to denote 'static' contribution only
                mpdf.rename(columns={flux[1]+d[1]+loc+u:
                                     flux[1]+d[1]+loc+'_static'+u},
                                     inplace=True)
                #Add in motional terms for proper total
                static = mpdf[flux[1]+d[1]+loc+'_static'+u]
                motional = mpdf[energies[flux[0]]+motion[d[0]]+loc+u]
                mpdf[flux[1]+d[1]+loc+u] = static+motional
    #Drop time column
    mpdf.drop(columns=['Time [UTC]'],inplace=True, errors='ignore')
    return mpdf

def gather_magnetopause(outersurface, volume):
    """Function combines surface and volume terms to construct single df
        with all magnetopause (magnetosphere) terms
    Inputs
        outersurface(DataFrame)-
        volume(DataFrame)-
    Returns
        combined(DataFrame)
    """
    combined = pd.DataFrame()
    for df in [outersurface, volume]:
        for key in df.keys():
            if not all(df[key].isna()):
                combined[key] = df[key]
    if 'K_net [W]' in combined.keys():
        combined = magnetopause_energy_mods(combined)
    if 'Virial Volume Total [nT]' in combined.keys():
        combined = virial_mods(combined)
    return combined

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
    store = pd.HDFStore(datapath+'partial.h5')
    times = store[store.keys()[0]]['Time [UTC]']+dt.timedelta(minutes=45)
    #Clean up and gather statistics
    mp = gather_magnetopause(store['/mp_iso_betastar_surface'],
                             store['/mp_iso_betastar_volume'])
    inner_mp = store['/mp_iso_betastar_inner_surface']
    mpdict = {'ms_full':mp}
    msdict = {}
    for key in store.keys():
        if 'ms' in key:
            if any(['Virial' in k for k in store[key].keys()]):
                cleaned_df = virial_mods(store[key])
            else:
                cleaned_df = store[key]
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_df.drop(
                                                  columns=['Time [UTC]'])})
    mpdict, msdict = get_interzone_stats(mpdict, msdict)
    store.close()
    #Construct "grouped" set of subzones
    msdict_3zone = {'lobes':msdict['nlobe']+msdict['slobe'],
                    'closedRegion':msdict['ps']+msdict['qDp'],
                    'rc':msdict['rc'],
                    'missing':msdict['missed']}
    msdict_lobes ={'nlobe':msdict['nlobe'],
                   'slobe':msdict['slobe'],
                   'remaining':msdict['rc']+msdict['ps']+msdict['qDp'],
                   'missing':msdict['missed']}
    #Begin plots
    #Fist type: line and stacks of various groupings of terms
    ######################################################################
    if 'Virial Volume Total [nT]' in mp.keys()and 'bioS [nT]' in mp.keys():
        #Label setup
        y1labels = [r'Virial $\Delta B\left[ nT\right]$',
                    r'Plasma Virial $\Delta B\left[ nT\right]$',
                    r'Magnetic Virial $\Delta B\left[ nT\right]$',
                    r'Biot Savart $\Delta B\left[ nT\right]$']
        y2label =   r'$\%$ Contribution'
        value_keys=['Virial Volume Total ','Virial 2x Uk ',
                    'Virial Ub ','bioS ']
        #Line plots- total virial, plasma, disturbance, and Biot savart
        for combo in {'allPiece':msdict,'3zone':msdict_3zone,
                    'lobes':msdict_lobes}.items():
            fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig2,ax2 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig3,ax3 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig4,ax4 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            for ax in enumerate([ax1, ax2, ax3, ax4]):
                plot_contributions(ax[1][0], times,mp,combo[1],
                                   y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_contributions(ax[1][1], times, mp, combo[1], y2label,
                                   value_key=value_keys[ax[0]]+'[%]',
                                   do_xlabel=True)
            for fig in{'total':fig1,'plasma':fig2,'mag_perturb':fig3,
                    'bioS':fig4}.items():
                fig[1].tight_layout(pad=1)
                fig[1].savefig(figureout+'virial_line_'+fig[0]+
                               combo[0]+'.png')
                plt.close(fig[1])
        #Stack plots- total virial, plasma, disturbance, and Biot savart
        for combo in {'allPiece':msdict,'3zone':msdict_3zone,
                    'lobes':msdict_lobes}.items():
            fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig2,ax2 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig3,ax3 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            fig4,ax4 = plt.subplots(nrows=2,ncols=1,sharex=True,
                                    figsize=[14,8])
            for ax in enumerate([ax1, ax2, ax3, ax4]):
                plot_stack_contrib(ax[1][0], times,mp,combo[1],
                                   y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_stack_contrib(ax[1][1], times, mp, combo[1], y2label,
                                   value_key=value_keys[ax[0]]+'[%]',
                                   do_xlabel=True)
            for fig in{'total':fig1,'plasma':fig2,'mag_perturb':fig3,
                    'bioS':fig4}.items():
                fig[1].tight_layout(pad=1)
                fig[1].savefig(figureout+'virial_stack_'+fig[0]+
                               combo[0]+'.png')
                plt.close(fig[1])
    #Second type: distribution of types of energy in the system
    ######################################################################
    term_labels = {'uB':r'Total Magnetic Energy $\left[J\right]$',
               'Virial Ub':r'Disturbance Magnetic Energy $\left[J\right]$',
                   'KE':r'Kinetic Energy $\left[J\right]$',
                   'Eth':r'Thermal Energy $\left[J\right]$',
                   'Utot':r'Total Energy $\left[J\right]$'}
    y2label = r'Energy fraction $\left[\%\right]$'
    for (term,label) in term_labels.items():
        if term+' [J]' in mp.keys():
            fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
            plot_contributions(ax[0],times,mp,msdict_3zone,label,
                               value_key=term+' [J]')
            plot_contributions(ax[1],times,mp,msdict_3zone,y2label,
                               value_key=term+' [%]',
                               do_xlabel=True)
            fig.tight_layout(pad=1)
            fig.savefig(figureout+term+'_line.png')
            plt.close(fig)
    if 'Volume [Re^3]' in mp.keys():
        fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        plot_contributions(ax[0], times, mp, msdict_3zone,
                           r'Volume $\left[R_e\right]$',
                           value_key='Volume [Re^3]')
        plot_contributions(ax[1], times, mp, msdict_3zone,
                           r'Volume fraction $\left[\%\right]$',
                           value_key='Volume [%]')
        fig.tight_layout(pad=1)
        fig.savefig(figureout+'Volume_line.png')
        plt.close(fig)
    #Third type: distrubiton of virial and types of energy within subzone
    ######################################################################
    y2label = r'Fraction $\left[\%\right]$'
    valset = ['Virial','Energy']
    for vals in valset:
        if 'Virial' in vals:
            fig2ylabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
                'closedRegion':r'Closed Region $\Delta B\left[nT\right]$',
                           'lobes':r'Lobes $\Delta B\left[nT\right]$'}
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
        plt.close(fig1)
        for subzone in enumerate([k for k in msdict_3zone.keys()][0:-1]):
            y1labels = [subzone[1]+' '+vals+' Distribution']
            fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
            plot_distr(ax[0],times,mp,msdict_3zone,y1labels[0],
                       value_set=vals, subzone=subzone[1])
            plot_distr(ax[1], times, mp, msdict_3zone, y2label,
                   value_set=vals+'%', do_xlabel=True, subzone=subzone[1])
            plot_distr(ax2[subzone[0]],times,mp,msdict_3zone,
                fig2ylabels[subzone[1]],value_set=vals,subzone=subzone[1])
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
    ylabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
               'closedRegion':r'Closed Region $\Delta B\left[nT\right]$',
               'lobes':r'Lobes $\Delta B\left[nT\right]$',
               'missing':r'Unaccounted $\Delta B\left[nT\right]$'}
    fig, ax = plt.subplots(nrows=len(msdict_3zone), ncols=1, sharex=True,
                           figsize=[14,4*len(msdict_3zone)])
    for subzone in enumerate(msdict_3zone):
        plot_single(ax[subzone[0]], times, mp, msdict_3zone,
                   ylabels[subzone[1]],
                   subzone=subzone[1],value_key='Virial [nT]')
        plot_single(ax[subzone[0]], times, mp, msdict_3zone,
             ylabels[subzone[1]], subzone=subzone[1],value_key='bioS [nT]',
                    do_xlabel=(subzone[0] is len(msdict_3zone)-1))
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'compare_Total_line.png')
    plt.close(fig)
    #Fifth type: virial vs Biot Savart one nice imag with obs dst
    ######################################################################
    y1label = r'Virial $\Delta B\left[ nT\right]$'
    y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    msdict_3zone.pop('missing')
    plot_contributions(ax[0], times,mp,msdict_3zone,y1label,
                               value_key='Virial [nT]',
                               legend_loc='lower left')
    plot_contributions(ax[1], times,mp,msdict_3zone,y2label,
                               value_key='bioS [nT]', do_xlabel=True)
    ax[0].set_ylim([-125,50])
    ax[1].set_ylim([-125,50])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'pretty_Dst_line.png')
    plt.close(fig)
    #Just look at surface terms
    ######################################################################
    if False:
        y1label = r'$\Delta B \left[nT\right]$'
        fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=[14,4])
    if False:
        #Open closed
        ax.fill_between(times,mp['Virial Surface TotalClosed [nT]']+
                            mp['Virial Surface TotalOpenN [nT]']+
                         mp['Virial Surface TotalOpenS [nT]'],color='grey')
        ax.plot(times,mp['Virial Surface TotalClosed [nT]'],label='Closed')
        ax.plot(times,mp['Virial Surface TotalOpenN [nT]'],label='North')
        ax.plot(times,mp['Virial Surface TotalOpenS [nT]'],label='South')
    if False:
        #Lobes, closed field, RC
        ax.fill_between(times,mp['Virial Surface Total [nT]']+
                            mp['Virial Volume Total [nT]'],color='grey')
        ax.plot(times,mp['Virial Surface TotalClosed [nT]']+
                  msdict_3zone['closedRegion']['Virial Volume Total [nT]'],
                    label='Closed')
        ax.plot(times,mp['Virial Surface TotalOpenN [nT]']+
                    mp['Virial Surface TotalOpenS [nT]']+
                    msdict_3zone['lobes']['Virial Volume Total [nT]'],
                    label='Lobes')
        ax.plot(times,msdict_3zone['rc']['Virial Volume Total [nT]'],
                    label='RingCurrent')
    if False:
        ax.fill_between(times,mp['Um [nT]'],color='grey')
        ax.plot(times,mp['rhoU_r_acqu [J]']/(-8e13),label='Acquired')
        ax.plot(times,mp['rhoU_r_forf [J]']/(-8e13),label='Forf')
        ax.plot(times,mp['rhoU_r_net [J]']/(-8e13),label='Net')
    if False:
        general_plot_settings(ax, y1label, do_xlabel=True)
        plt.show()
    from IPython import embed; embed()
