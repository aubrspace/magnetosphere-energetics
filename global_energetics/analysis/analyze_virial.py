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
#interpackage imports
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup,
                                                   get_omni_cdas)
from global_energetics.analysis.analyze_energetics import (plot_swflowP,
                                                          plot_swbz,
                                                          plot_dst,plot_al)
from global_energetics.analysis.proc_virial import (process_virial)
from global_energetics.analysis.proc_hdf import(group_subzones,
                                                load_hdf_sort)
def plot_single(ax, times, mp, msdict, **kwargs):
    """Plots energy or dst contribution in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
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
                color=kwargs.get('color',None),
                ls=kwargs.get('ls',None))
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_stack_distr(ax, times, mp, msdict, **kwargs):
    """Plots distribution of energies in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
            legend_loc(see pyplot)- 'upper right' etc
            subzone(str)- 'ms_full' is default
            value_set(str)- 'Virialvolume', 'Energy', etc
            ylim(tuple or list)- None
    """
    #Figure out value_keys and subzone
    subzone = kwargs.get('subzone', 'ms_full')
    kwargs.update({'legend_loc':kwargs.get('legend_loc','lower left')})
    if 'ms_full' in subzone:
        data = mp
    else:
        data = msdict[subzone]
    value_set = kwargs.get('value_set','Virial')
    values = {'Virial':['Virial 2x Uk [nT]', 'Virial Ub [nT]','Um [nT]',
                        'Virial Surface Total [nT]'],
              'Energy':['Virial Ub [J]', 'KE [J]', 'Eth [J]'],
              'Virial%':['Virial 2x Uk [%]', 'Virial Ub [%]',
                               'Virial Surface Total [%]'],
              'Energy%':['Virial Ub [%]', 'KE [%]', 'Eth [%]']}
    #Get % within specific subvolume
    if '%' in value_set:
        kwargs.update({'ylim':kwargs.get('ylim',[0,100])})
        for value in values[''.join(value_set.split('%'))]:
                #find total
                if 'Virial' in value_set:
                    total = data['Virial [nT]']
                elif 'Energy' in value_set:
                    total = (data['Utot [J]']-data['uB [J]']+
                             data['Virial Ub [J]'])
                data[value.split('[')[0]+'[%]'] = data[value]/total*100
    #Optional layers depending on value_key
    starting_value = 0
    if (not '%' in value_set) and (not 'Energy' in value_set):
        ax.plot(times, data['Virial Surface Total [nT]'], color='#05BC54',
                linewidth=4,label='Boundary Stress')#light green color
        starting_value = data['Virial Surface Total [nT]']
        values[value_set].remove('Virial Surface Total [nT]')
        ax.axhline(0,color='white',linewidth=0.5)
    for value in values[value_set]:
        label = value.split(' [')[0].split('Virial ')[-1]
        ax.fill_between(times,starting_value,starting_value+data[value],
                        label=label)
        starting_value = starting_value+data[value]
    if (not '%' in value_set) and kwargs.get('doBios',True):
        if any(['bioS' in k for k in data.keys()]):
            ax.plot(times, data['bioS [nT]'], color='white', ls='--',
                    label='BiotSavart')
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_distr(ax, times, mp, msdict, **kwargs):
    """Plots distribution of energies in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
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
    value_set = kwargs.get('value_set','Virial')
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
                if all(omni['sym_h'].isna()): raise NameError
                ax.plot(omni['Time [UTC]'],omni['sym_h'],
                        color='white',
                        ls='--', label='SYM-H')
            except NameError:
                print('omni not loaded! No obs comparisons!')
        elif 'Virial' in value_set:
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
            ax.plot(times, data[value], label=safelabel)
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_stack_contrib(ax, times, mp, msdict, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
            legend_loc(see pyplot)- 'upper right' etc
    """
    #Figure out value_key
    value_key = kwargs.get('value_key','Virial [nT]')
    #Optional layers depending on value_key
    starting_value = 0
    if not '%' in value_key:
        ax.axhline(0,color='white',linewidth=0.5)
        if ('Virial' in value_key) or ('bioS' in value_key):
            try:
                if all(omni['sym_h'].isna()): raise NameError
                ax.plot(omni['Time [UTC]'],omni['sym_h'],color='white',
                    ls='--', label='SYM-H')
            except NameError:
                print('omni not loaded! No obs comparisons!')
        if ('bioS' in value_key) and ('bioS_ext [nT]' in mp.keys()):
            starting_value = mp['bioS_ext [nT]']
            ax.plot(times, mp['bioS_ext [nT]'],color='#05BC54',linewidth=4,
                    label='External')#light green color
    #Plot stacks
    for ms in msdict.items():
        mslabel = ms[0]
        d = ms[1][value_key]
        times, d = times[~d.isna()], d[~d.isna()]
        ax.fill_between(times,starting_value,starting_value+d,label=mslabel)
        starting_value = starting_value+d
    ax.set_xlim([times.iloc[0],times.iloc[-1]])
    #Optional plot settings
    if ('bioS' in value_key) and ('%' in value_key):
        ax.set_ylim([-100,100])
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_contributions(ax, times, mp, msdict, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
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
                if all(omni['sym_h'].isna()): raise NameError
                ax.plot(omni['Time [UTC]'],omni['sym_h'],color='seagreen',
                    ls='--', label='OMNI obs')
            except NameError:
                print('omni not loaded! No obs comparisons!')
            try:
                pass
                #ax.plot(swmf_log['Time [UTC]'], swmf_log['dst_sm'],
                #    color='tab:red', ls='--',
                #    label='SWMFLog')
                #ax.plot(swmf_log['Time [UTC]'], swmf_log['dstflx_R=3.0'],
                #    color='magenta', ls='--',
                #    label='SWMFLog Flux')
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
    general_plot_settings(ax, **kwargs)

if __name__ == "__main__":
    #handling io paths
    datapath = sys.argv[-1]
    figureout = os.path.join(datapath,'figures')
    os.makedirs(figureout, exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='digital_presentation'))

    ##Loading data
    #Log files and observational indices
    [swmf_index, swmf_log, swmf_sw,_,omni]= read_indices(datapath,
                                                       read_supermag=False)
    #HDF data, will be sorted and cleaned
    [mpdict,msdict,inner_mp,times,get_nonGM]=load_hdf_sort(
                                                datapath+'virial_track.h5')
    if get_nonGM:
        #Check for non GM data
        ie, ua_j, ua_e, ua_non, ua_ie= load_nonGM(datapath+'results.h5')

    ##Apply any mods and gather additional statistics
    [mpdict,msdict,inner_mp] = process_virial(mpdict,msdict,inner_mp,times)
    mp = [m for m in mpdict.values()][0]

    ##Construct "grouped" set of subzones
    msdict = group_subzones(msdict,mode='3zone')

    ##Begin plots
    #MGU image: ie integrated joule heating, ua density@210km, lobe energy
    ######################################################################
    if get_nonGM:
        #gmlabel = r'$\Delta B \left[nT\right]$'
        gmlabel = r'Energy $\left[J\right]$'
        gmlabel2 = r'Energy $\left[\%\right]$'
        lobelabel= r'Lobes Energy$\left[J\right]$'
        ielabel = r'Joule Heating $\left[GW\right]$'
        ielabel3 = r'Power $\left[GW\right]$'
        ualabel = r'$\rho_{210km} \left[amu/cm^3\right]$'
        ualabel2 = r'$\rho_{210km}/\rho_0$'
        #fig,ax = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,18],
        #                      gridspec_kw={'height_ratios':[1,1,1,1]})
        fig,ax = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,15.75],
                              gridspec_kw={'height_ratios':[2,2,2,1]})
        ##Biot Savart Dst
        #plot_stack_contrib(ax[0], times,mp,msdict,ylabel=gmlabel,
        #                   ylim=[-125,30], value_key='bioS [nT]',
        #                   do_xlabel=False)
        ##Energy Contributions- value
        plot_stack_contrib(ax[0], times,mp,msdict,ylabel=gmlabel,
                           value_key='Utot2 [J]',
                           do_xlabel=False)
        ##Energy Contributions- percent
        plot_contributions(ax[1], times,mp,msdict,ylabel=gmlabel2,
                           ylim=[0,100],value_key='Utot2 [%]',
                           do_xlabel=False)
        ##Lobe Energy
        #plot_stack_distr(ax[1],times,mp,msdict,ylabel=lobelabel,
        #                doBios=False,
        #                value_set='Energy',subzone='lobes',do_xlabel=False)
        ##IE integrated joule heating
        ax[2].plot(ie.index,ie['nJouleHeat_W']/1e9,label='north Joule')
        ax[2].plot(ie.index,ie['sJouleHeat_W']/1e9,label='south Joule')
        #general_plot_settings(ax[2],ylabel=ielabel,legend_loc='upper left')
        ##IE integrated Energy Flux
        #ax22 = ax[2].twinx()
        ax[2].plot(ie.index,ie['nEFlux_W']/1e3,label='north EFlux',color='peru')
        ax[2].plot(ie.index,ie['sEFlux_W']/1e3,label='south EFlux',color='chartreuse')
        general_plot_settings(ax[2],ylabel=ielabel3,legend_loc='upper left')
        ##UA Density @210km
        ua_tags = ['nJoule','nEnergy','None', 'IE']
        ua_ls = ['--',':','-.',None]
        fig,ax=plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,15.75],
                                   gridspec_kw={'height_ratios':[2,2,2,2]})
        for ua in enumerate([ua_e,ua_j,ua_non,ua_ie]):
            ax[3].plot(ua[1].index,
                       ua[1]['Rho_mean210']/ua[1]['Rho_mean210'].iloc[0],
                       label='210'+ua_tags[ua[0]],ls=ua_ls[ua[0]])
            ax[2].plot(ua[1].index,
                       ua[1]['Rho_mean310']/ua[1]['Rho_mean310'].iloc[0],
                       label='310'+ua_tags[ua[0]],ls=ua_ls[ua[0]])
            ax[1].plot(ua[1].index,
                       ua[1]['Rho_mean410']/ua[1]['Rho_mean410'].iloc[0],
                       label='410'+ua_tags[ua[0]],ls=ua_ls[ua[0]])
            ax[0].plot(ua[1].index,
                       ua[1]['Rho_mean510']/ua[1]['Rho_mean510'].iloc[0],
                       label='510'+ua_tags[ua[0]],ls=ua_ls[ua[0]])
        general_plot_settings(ax[0],ylabel=ualabel2,do_xlabel=True)
        general_plot_settings(ax[1],ylabel=ualabel2,do_xlabel=True)
        general_plot_settings(ax[2],ylabel=ualabel2,do_xlabel=True)
        general_plot_settings(ax[3],ylabel=ualabel2,do_xlabel=True)
        #                     ylim=[0.8,1.5])
        #ax[3].get_legend().remove()
        #Save
        fig.tight_layout(pad=1)
        fig.savefig(figureout+'/ie_ua_energy')
        plt.close(fig)
    '''
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
        for combo in {'allPiece':msdict,'3zone':msdict,
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
                                   ylabel=y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_contributions(ax[1][1], times, mp, combo[1],
                                   ylabel=y2label,
                                   value_key=value_keys[ax[0]]+'[%]',
                                   do_xlabel=True)
            for fig in{'total':fig1,'plasma':fig2,'mag_perturb':fig3,
                    'bioS':fig4}.items():
                fig[1].tight_layout(pad=1)
                fig[1].savefig(figureout+'virial_line_'+fig[0]+
                               combo[0]+'.png')
                plt.close(fig[1])
        #Stack plots- total virial, plasma, disturbance, and Biot savart
        for combo in {'allPiece':msdict,'3zone':msdict,
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
                                   ylabel=y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_stack_contrib(ax[1][1], times, mp, combo[1],
                                   ylabel=y2label,
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
            plot_contributions(ax[0],times,mp,msdict,ylabel=label,
                               value_key=term+' [J]')
            plot_contributions(ax[1],times,mp,msdict,ylabel=y2label,
                               value_key=term+' [%]',
                               do_xlabel=True)
            fig.tight_layout(pad=1)
            fig.savefig(figureout+term+'_line.png')
            plt.close(fig)
    if 'Volume [Re^3]' in mp.keys():
        fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        plot_contributions(ax[0], times, mp, msdict,
                           ylabel=r'Volume $\left[R_e\right]$',
                           value_key='Volume [Re^3]')
        plot_contributions(ax[1], times, mp, msdict,
                           ylabel=r'Volume fraction $\left[\%\right]$',
                           value_key='Volume [%]')
        fig.tight_layout(pad=1)
        fig.savefig(figureout+'Volume_line.png')
        plt.close(fig)
    '''
    #Third type: distrubiton of virial and types of energy within subzone
    ######################################################################
    y2label = r'Fraction $\left[\%\right]$'
    valset = ['Virial','Energy']
    #valset = ['Virial']
    for vals in valset:
        if 'Virial' in vals:
            fig2ylabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
                'closedRegion':r'Closed Region $\Delta B\left[nT\right]$',
                           'lobes':r'Lobes $\Delta B\left[nT\right]$',
                           'missing':r'missed $\Delta B\left[nT\right]$'}
        elif 'Energy' in vals:
            fig2ylabels = {'rc':r'Ring Current Energy $\left[J\right]$',
                    'closedRegion':r'Closed Region Energy $\left[J\right]$',
                           'lobes':r'Lobe Energy $\left[J\right]$'}
        y1labels = ['Full MS '+vals+' Distribution']
        fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig2,ax2 = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,16])
        plot_stack_distr(ax1[0],times,mp,msdict,ylabel=y1labels[0],
                                                            value_set=vals)
        plot_stack_distr(ax1[1], times, mp, msdict, ylabel=y2label,
                   value_set=vals+'%', do_xlabel=True)
        fig1.tight_layout(pad=1)
        fig1.savefig(figureout+'/distr_'+vals+'_fullMS_line.png')
        plt.close(fig1)
        for subzone in enumerate([k for k in msdict.keys()][0:-1]):
            y1labels = [subzone[1]+' '+vals+' Distribution']
            fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,figsize=[14,4])
            plot_stack_distr(ax,times,mp,msdict,
                             ylabel=fig2ylabels[subzone[1]],
                           value_set=vals,subzone=subzone[1],doBios=False,
                           ylim=[0,10e15],do_xlabel=True,
                           legend_loc='upper left')
            #plot_stack_distr(ax[1], times, mp, msdict, ylabel=y2label,
            #       value_set=vals+'%', do_xlabel=True, subzone=subzone[1])
            plot_stack_distr(ax2[subzone[0]],times,mp,msdict,
                ylabel=fig2ylabels[subzone[1]],value_set=vals,
                subzone=subzone[1],
                             do_xlabel=(subzone[0] is len(msdict)-2))
            fig.tight_layout(pad=1)
            fig.savefig(figureout+'/distr_'+vals+subzone[1]+'_line.png')
            plt.close(fig)
        fig2.tight_layout(pad=1)
        fig2.savefig(figureout+'/distr_'+vals+'eachzone_line.png')
        plt.close(fig2)
    #Fourth type: virial vs Biot Savart stand alones
    ######################################################################
    ylabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
               'closedRegion':r'Closed Region $\Delta B\left[nT\right]$',
               'lobes':r'Lobes $\Delta B\left[nT\right]$',
               'missing':r'Unaccounted $\Delta B\left[nT\right]$'}
    fig, ax = plt.subplots(nrows=len(msdict), ncols=1, sharex=True,
                           figsize=[14,4*len(msdict)])
    for subzone in enumerate(msdict):
        plot_single(ax[subzone[0]], times, mp, msdict,
                   ylabel=ylabels[subzone[1]],
                   subzone=subzone[1],value_key='Virial [nT]')
        plot_single(ax[subzone[0]], times, mp, msdict,
             ylabel=ylabels[subzone[1]], subzone=subzone[1],
             value_key='bioS [nT]',
                    do_xlabel=(subzone[0] is len(msdict)-1))
                    #ylim=[-100,30])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/compare_Total_line.png')
    plt.close(fig)
    #Fifth type: virial vs Biot Savart one nice imag with obs dst
    ######################################################################
    y1label = r'Virial $\Delta B\left[ nT\right]$'
    y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,14])
    msdict.pop('missing')
    plot_stack_contrib(ax[0], times,mp,msdict,ylabel=y1label,
                               value_key='Virial [nT]',
                               legend_loc='lower left')
    plot_stack_contrib(ax[1], times,mp,msdict,ylabel=y2label,
                               value_key='bioS [nT]', do_xlabel=True)
    ax[0].set_ylim([-125,20])
    ax[1].set_ylim([-125,20])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/pretty_Dst_line.png')
    plt.close(fig)
    #AGU image: virial biot savart
    ######################################################################
    y1label = r'Virial $\Delta B\left[ nT\right]$'
    y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
    clabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
               'closedRegion':r'Closed Region $\Delta B\left[nT\right]$',
               'lobes':r'Lobes $\Delta B\left[nT\right]$'}
    fig,ax = plt.subplots(nrows=5,ncols=1,sharex=True,figsize=[14,17],
                          gridspec_kw={'height_ratios':[2,2,1,1,1]})
    #msdict.pop('missing')
    plot_stack_contrib(ax[0], times,mp,msdict,ylabel=y1label,
                               value_key='Virial [nT]',ylim=[-125,25],
                               legend_loc='lower left')
    plot_stack_contrib(ax[1], times,mp,msdict,ylabel=y2label,
                              ylim=[-125,25],
                               value_key='bioS [nT]', do_xlabel=False)
    for subzone in enumerate([k for k in msdict.keys()]):
        ylabels = [subzone[1]+' '+vals+' Distribution']
        plot_stack_distr(ax[2+subzone[0]],times,mp,msdict,
                 ylabel=fig2ylabels[subzone[1]],value_set=vals,
                 subzone=subzone[1],
                             do_xlabel=(subzone[0] is len(msdict)-1))
        ax[2+subzone[0]].get_legend().remove()
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/agu2021_main.png')
    plt.close(fig)
    #AGU image: SW Bz+Bmagnitude, Pdyn
    ######################################################################
    swBlabel = r'$B \left[nT\right]$'
    swPlabel = r'$P_{ram} \left[nPa\right]$'
    Allabel = r'AL $\left[nT\right]$'
    SYMhlabel = r'SYM-H $\left[nT\right]$'
    colorwheel = plt.cycler('color',
                   ['#375e95', '#05BC54', 'black', 'chartreuse', 'wheat',
                    'lightgrey', 'springgreen', 'coral', 'plum', 'salmon'])
                   #matte blue, lightgreen, vibrant red
    plt.rcParams.update({'figure.facecolor':'#FBFFE7',
                         'axes.prop_cycle': colorwheel,
                         'text.color':'black',
                         'ytick.color':'black',
                         'xtick.color':'black',
                         'axes.edgecolor': 'black',
                         'axes.facecolor': '#FBFFE7',
                         'axes.labelcolor': 'black'})
    #figure settings
    fig,ax = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,10],
                          gridspec_kw={'height_ratios':[1,1,1,2]})
    #fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,4.5],
    #                      gridspec_kw={'height_ratios':[1,1]})
    #Solar wind Bz
    plot_swbz(ax[0],[swmf_sw],'Time [UTC]',swBlabel)
    ax[0].fill_between(swmf_sw['Time [UTC]'], np.sqrt(swmf_sw['bx']**2+
                                     swmf_sw['by']**2+swmf_sw['bz']**2),
                                     color='grey',label=r'$|B|$')
    general_plot_settings(ax[0], ylabel=swBlabel, legend_loc='lower left')
    ax[0].set_xlabel(None)
    #Ram pressure
    plot_swflowP(ax[1],[swmf_sw],'Time [UTC]',swPlabel)
    general_plot_settings(ax[1], ylabel=swPlabel)
    ax[1].set_xlabel(None)
    ax[1].get_legend().remove()
    #AL
    plot_al(ax[2], [omni,swmf_index], 'Time [UTC]', Allabel)
    general_plot_settings(ax[2], ylabel=Allabel)
    ax[2].set_xlabel(None)
    ax[2].get_legend().remove()
    #Dst
    plot_dst(ax[3], [omni], 'Time [UTC]', Allabel)
    ax[3].plot(times, mp['bioS [nT]'],linewidth=2,label='Sim')
    general_plot_settings(ax[3], ylabel=SYMhlabel)
    #Save
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/agu2021_side.png')
    plt.close(fig)
    #Pressure, External Perturbations, Volume
    ######################################################################
    Plabel = r'$P_{ram} \left[nPa\right]$'
    Biotlabel = r'External $\Delta B\left[ nT\right]$'
    Vollabel = r'Volume $\left[R_e\right]$'
    fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=[14,6],
                              gridspec_kw={'height_ratios':[1,1,1]})
    #Ram pressure
    plot_swflowP(ax[0],[swmf_sw],'Time [UTC]',Plabel)
    general_plot_settings(ax[0], ylabel=Plabel)
    ax[0].set_xlabel(None)
    ax[0].get_legend().remove()
    #Dst
    ax[1].plot(times,mp['bioS_ext [nT]'],color='#05BC54')
    general_plot_settings(ax[1], ylabel=Biotlabel)
    ax[1].set_xlabel(None)
    ax[1].get_legend().remove()
    #Volume
    ax[2].plot(times,mp['Volume [Re^3]'],color='#05BC54')
    general_plot_settings(ax[2], ylabel=Vollabel,do_xlabel=True)
    ax[2].get_legend().remove()
    #Save
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/external_pressure.png')
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
                  msdict['closedRegion']['Virial Volume Total [nT]'],
                    label='Closed')
        ax.plot(times,mp['Virial Surface TotalOpenN [nT]']+
                    mp['Virial Surface TotalOpenS [nT]']+
                    msdict['lobes']['Virial Volume Total [nT]'],
                    label='Lobes')
        ax.plot(times,msdict['rc']['Virial Volume Total [nT]'],
                    label='RingCurrent')
    if False:
        ax.fill_between(times,mp['Um [nT]'],color='grey')
        ax.plot(times,mp['rhoU_r_acqu [J]']/(-8e13),label='Acquired')
        ax.plot(times,mp['rhoU_r_forf [J]']/(-8e13),label='Forf')
        ax.plot(times,mp['rhoU_r_net [J]']/(-8e13),label='Net')
    if False:
        general_plot_settings(ax, ylabel=y1label, do_xlabel=True)
        plt.show()
    if False:
        y1label = r'$\Delta B \left[nT\right]$'
        fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=[14,4])
        plot_swbz(ax2[3],[swmf_sw],'Time [UTC]',r'$B_z \left[nT\right]$')
        plot_swflowP(ax2[3].twinx(),[swmf_sw],'Time [UTC]',
                     r'$P_{ram} \left[nT\right]$')
        ax2[3].set_xlabel(r'Time $\left[ UTC\right]$')
        ax2[3].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2[3].tick_params(which='major', length=9)
        ax2[3].xaxis.set_minor_locator(AutoMinorLocator(6))
