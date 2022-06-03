#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import os
import sys
import glob
import time
import numpy as np
from scipy import signal
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup, plot_psd,
                                                   plot_pearson_r,
                                                   plot_stack_distr,
                                                   plot_stack_contrib,
                                                   safelabel,
                                                   get_omni_cdas)
from global_energetics.analysis.analyze_energetics import (plot_swflowP,
                                                          plot_swbz,
                                                          plot_dst,plot_al)
from global_energetics.analysis.proc_virial import (process_virial)
from global_energetics.analysis.workingtitle import (locate_phase)
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
                if all(obs['omni']['sym_h'].isna()): raise NameError
                ax.plot(obs['omni']['Time [UTC]'],omni['sym_h'],
                        color='seagreen',ls='--', label='OMNI obs')
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
    #HDF data, will be sorted and cleaned
    feb_results = load_hdf_sort(datapath+'/feb2014_results.h5')
    star_results = load_hdf_sort(datapath+'/starlink_results.h5')
    ccmc_results = load_hdf_sort(datapath+'/ccmc_results.h5')

    #Log files and observational indices
    f14obs = read_indices(datapath,prefix='feb2014_',read_supermag=False,
                          start=feb_results['mpdict']['ms_full'].index[0],
                          end=feb_results['mpdict']['ms_full'].index[-1])
    starobs = read_indices(datapath,prefix='starlink_',read_supermag=False,
                          start=star_results['mpdict']['ms_full'].index[0],
                          end=star_results['mpdict']['ms_full'].index[-1])
    ccmcobs = read_indices(datapath,prefix='ccmc_',read_supermag=False,
                          start=ccmc_results['mpdict']['ms_full'].index[0],
                          end=ccmc_results['mpdict']['ms_full'].index[-1])

    ##Apply any mods and gather additional statistics
    [feb_mpdict,feb_msdict,feb_inner_mp] = process_virial(feb_results,
                                                        f14obs['swmf_log'])
    [star_mpdict,star_msdict,star_inner_mp] = process_virial(star_results,
                                                       starobs['swmf_log'])
    [ccmc_mpdict,ccmc_msdict,ccmc_inner_mp] = process_virial(ccmc_results,
                                                       ccmcobs['swmf_log'])

    feb_mp = [m for m in feb_mpdict.values()][0]
    star_mp = [m for m in star_mpdict.values()][0]
    ccmc_mp = [m for m in ccmc_mpdict.values()][0]

    ##Construct "grouped" set of subzones
    feb_msdict = {'lobes':feb_msdict['lobes'],
                  'closed':feb_msdict['closed'],
                  'rc':feb_msdict['rc'],
                  'missed':feb_msdict['missed']}
    star_msdict = {'lobes':star_msdict['lobes'],
                   'closed':star_msdict['closed'],
                   'rc':star_msdict['rc'],
                   'missed':star_msdict['missed']}
    ccmc_msdict = {'lobes':ccmc_msdict['lobes'],
                   'closed':ccmc_msdict['closed'],
                   'rc':ccmc_msdict['rc'],
                   'missed':ccmc_msdict['missed']}


    ##Begin plots
    ######################################################################
    #Power spectrum plot
    #TODO
    #   > bring in the phase locator from working title
    #   > Run for feb main/recovery
    #   > Look at energy data for both closed field and lobe energy content
    #       * they do oscillate with similar period and negative cor
    #   > See if there is a combination of sw drivers that can give a
    #       similar period
    #   > Rerun results with the latest surface definitions @ 1min cadence
    #   > If results aren't different for the events just choose one
    # WHY is the ~4hr substorm period showing up in the virial results?
    #   Hyp:    Energy being released from the system as plasmoid
    #   Hyp:    Mass being drawn in closer, reducing rhoU dot r
    # COULD the variations be a symptom of missing energy in the accounting?
    #       * Doesn't appear so!
    #Starlink
    #   starlink_impact = dt.datetime(2022,2,3,0,0)
    #   starlink_endMain1 = dt.datetime(2022,2,3,11,15)
    #   starlink_endMain2 = dt.datetime(2022,2,4,13,10)
    for mp,e_obs,msdict,inner,tag in [
            (feb_mp,f14obs,feb_msdict,feb_inner_mp,'feb2014'),
            (star_mp,starobs,star_msdict,star_inner_mp,'starlink'),
            (ccmc_mp,ccmcobs,ccmc_msdict,ccmc_inner_mp,'ccmc')]:
        figureout = os.path.join(datapath,'figures',tag)
        os.makedirs(figureout, exist_ok=True)
        ##Comparisons
        for (sim,comp)in[(mp['Virial [nT]'],e_obs['swmf_sw']['bz']),
                        (mp['bioS_full [nT]'],e_obs['swmf_sw']['bz']),
                        (mp['bioS [nT]'],e_obs['swmf_sw']['bz']),
                        (mp['bioS_ext [nT]'],e_obs['swmf_index']['AL']),
            (msdict['lobes']['Utot2 [J]'], msdict['closed']['Utot2 [J]'])]:
            #Gather labeling info
            simkey, compkey = sim.name, comp.name
            short_simkey = safelabel(simkey).split(' ')[0]
            short_compkey = safelabel(compkey).split(' ')[0]
            sim_u = simkey.split('[')[-1].split(']')[0]
            comp_u = compkey.split('[')[-1].split(']')[0]
            if comp_u == compkey and compkey=='bz': comp_u = 'nT'

            #Figure setup
            figname = ('frequency_analysis_'+short_simkey+'_'
                                            +short_compkey+'.png')
            psd,ax = plt.subplots(nrows=2,ncols=2,figsize=[14,14])
            psd_simy = (r'Power Density $\left['+
                                sim_u+r'^2 / Hz \right]$')
            psd_compy = (r'Power Density $\left['+
                                sim_u+r'^2 / Hz \right]$')
            psd_xlabel = r'Frequency $\left[mHz\right]$'
            rcor_xlabel = r''+short_compkey+r'$\left['+comp_u+r'\right]$'
            rcor_ylabel = r''+short_simkey+r'$\left['+sim_u+r'\right]$'

            for (i, phase) in enumerate(['main','rec']):
                phase_sim = locate_phase(sim,phase)
                phase_comp = locate_phase(comp,phase)
                #Time Series
                t_sim = phase_sim[~phase_sim.isna()].index
                t_comp = phase_comp[~phase_comp.isna()].index
                sim_series = phase_sim[~phase_sim.isna()].values
                comp_series = phase_comp[~phase_comp.isna()].values

                #Power Spectral Density analysis
                plot_psd(ax[i][0], t_sim, sim_series, label=short_simkey,
                        xlabel=psd_xlabel,ylabel=psd_simy)
                plot_psd(ax[i][0], t_comp, comp_series, label=short_compkey,
                        xlabel=psd_xlabel,ylabel=psd_compy)

                #Pearson R Correlation
                plot_pearson_r(ax[i][1], t_sim, t_comp, sim_series,
                               comp_series, xlabel=rcor_xlabel,
                                           ylabel=rcor_ylabel)

            psd.tight_layout(pad=1)
            psd.savefig(figureout+'/'+simkey.split(' ')[0]+figname)
            plt.close(psd)
    figureout = os.path.join(datapath,'figures')
    #mp, msdict, e_obs  = (star_mp, star_msdict, starobs)
    mp, msdict, e_obs  = (feb_mp, feb_msdict, f14obs)
    #MGU image: ie integrated joule heating, ua density@210km, lobe energy
    ######################################################################
    if False:
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
        #plot_stack_contrib(ax[0], mp.index,mp,msdict,ylabel=gmlabel,
        #                   ylim=[-125,30], value_key='bioS [nT]',
        #                   do_xlabel=False)
        ##Energy Contributions- value
        plot_stack_contrib(ax[0], mp.index,mp,msdict,ylabel=gmlabel,
                           value_key='Utot2 [J]',
                           do_xlabel=False)
        ##Energy Contributions- percent
        plot_contributions(ax[1], mp.index,mp,msdict,ylabel=gmlabel2,
                           ylim=[0,100],value_key='Utot2 [%]',
                           do_xlabel=False)
        ##Lobe Energy
        #plot_stack_distr(ax[1],mp.index,mp,msdict,ylabel=lobelabel,
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
                plot_contributions(ax[1][0], mp.index,mp,combo[1],
                                   ylabel=y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_contributions(ax[1][1], mp.index, mp, combo[1],
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
                plot_stack_contrib(ax[1][0], mp.index,mp,combo[1],
                                   ylabel=y1labels[ax[0]],
                                   value_key=value_keys[ax[0]]+'[nT]')
                plot_stack_contrib(ax[1][1], mp.index, mp, combo[1],
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
            plot_contributions(ax[0],mp.index,mp,msdict,ylabel=label,
                               value_key=term+' [J]')
            plot_contributions(ax[1],mp.index,mp,msdict,ylabel=y2label,
                               value_key=term+' [%]',
                               do_xlabel=True)
            fig.tight_layout(pad=1)
            fig.savefig(figureout+term+'_line.png')
            plt.close(fig)
    if 'Volume [Re^3]' in mp.keys():
        fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        plot_contributions(ax[0], mp.index, mp, msdict,
                           ylabel=r'Volume $\left[R_e\right]$',
                           value_key='Volume [Re^3]')
        plot_contributions(ax[1], mp.index, mp, msdict,
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
                'closed':r'Closed $\Delta B\left[nT\right]$',
                           'lobes':r'Lobes $\Delta B\left[nT\right]$',
                           'missed':r'missed $\Delta B\left[nT\right]$'}
        elif 'Energy' in vals:
            fig2ylabels = {'rc':r'Ring Current Energy $\left[J\right]$',
                    'closed':r'Closed Energy $\left[J\right]$',
                           'lobes':r'Lobe Energy $\left[J\right]$'}
        y1labels = ['Full MS '+vals+' Distribution']
        fig1,ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
        fig2,ax2 = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=[14,16])
        plot_stack_distr(ax1[0],mp.index,mp,msdict,ylabel=y1labels[0],
                                                            value_set=vals)
        plot_stack_distr(ax1[1], mp.index, mp, msdict, ylabel=y2label,
                   value_set=vals+'%', do_xlabel=True)
        fig1.tight_layout(pad=1)
        fig1.savefig(figureout+'/distr_'+vals+'_fullMS_line.png')
        plt.close(fig1)
        for subzone in enumerate([k for k in msdict.keys()
                                 if'miss' not in k and 'summed' not in k]):
            y1labels = [subzone[1]+' '+vals+' Distribution']
            fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,figsize=[14,4])
            plot_stack_distr(ax,mp.index,mp,msdict,
                             ylabel=fig2ylabels[subzone[1]],
                           value_set=vals,subzone=subzone[1],doBios=False,
                           ylim=[0,10e15],do_xlabel=True,
                           legend_loc='upper left')
            #plot_stack_distr(ax[1], mp.index, mp, msdict, ylabel=y2label,
            #       value_set=vals+'%', do_xlabel=True, subzone=subzone[1])
            plot_stack_distr(ax2[subzone[0]],mp.index,mp,msdict,
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
               'closed':r'Closed $\Delta B\left[nT\right]$',
               'lobes':r'Lobes $\Delta B\left[nT\right]$',
               'missed':r'Unaccounted $\Delta B\left[nT\right]$'}
    fig, ax = plt.subplots(nrows=len(msdict), ncols=1, sharex=True,
                           figsize=[14,4*len(msdict)])
    for i,subzone in enumerate([sz for sz in msdict if 'summed' not in sz]):
        plot_single(ax[i], mp.index, mp, msdict, ylabel=ylabels[subzone],
                    subzone=subzone,value_key='Virial [nT]')
        plot_single(ax[i], mp.index, mp, msdict, ylabel=ylabels[subzone],
                    subzone=subzone, value_key='bioS [nT]',
                    do_xlabel=(i is len(msdict)-1))
                    #ylim=[-100,30])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/compare_Total_line.png')
    plt.close(fig)
    #Fifth type: virial vs Biot Savart one nice imag with obs dst
    ######################################################################
    for mp,e_obs,msdict,inner,tag in [
            (feb_mp,f14obs,feb_msdict,feb_inner_mp,'feb2014'),
            (star_mp,starobs,star_msdict,star_inner_mp,'starlink'),
            (ccmc_mp,ccmcobs,ccmc_msdict,ccmc_inner_mp,'ccmc')]:
        figureout = os.path.join(datapath,'figures',tag)
        os.makedirs(figureout, exist_ok=True)
        #Setup figure
        y1label = r'Virial $\Delta B\left[ nT\right]$'
        y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
        fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,14])
        msdict.pop('missed')

        #Plot
        plot_stack_contrib(ax[0], mp.index,mp,msdict,ylabel=y1label,
                               value_key='Virial [nT]',
                               legend_loc='lower left',omni=e_obs['omni'])
        plot_stack_contrib(ax[1], mp.index,mp,msdict,ylabel=y2label,
                               value_key='bioS [nT]', do_xlabel=True,
                               omni=e_obs['omni'])
        ax[0].set_ylim([-125,25])
        ax[1].set_ylim([-125,25])

        #Save
        fig.tight_layout(pad=1)
        fig.savefig(figureout+'/pretty_Dst_line.png')
        plt.close(fig)

        #Setup figure
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
        fig,axis = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=[14,8])

        #Plot
        axis.plot(mp.index,mp['Virial [nT]'],label='Virial',color='#375e95')
        axis.plot(mp.index,mp['bioS_full [nT]'],label='Biot Savart',
                  color='#375e95', ls='--')
        axis.plot(e_obs['omni'].index, e_obs['omni']['sym_h'],ls='--',
                  label='SYM-H', color='#05BC54')
        general_plot_settings(axis, ylim=[-125,25], do_xlabel=True,
                              ylabel=r'$\Delta B\left[ nT\right]$')

        #Save
        fig.tight_layout(pad=1)
        fig.savefig(figureout+'/virial_bios_obs_compare.png')
        plt.close(fig)
        plt.rcParams.update(pyplotsetup(mode='digital_presentation'))

    figureout = os.path.join(datapath,'figures')
    #AGU image: virial biot savart
    ######################################################################
    y1label = r'Virial $\Delta B\left[ nT\right]$'
    y2label = r'Biot Savart law $\Delta B\left[ nT\right]$'
    clabels = {'rc':r'Ring Current $\Delta B\left[nT\right]$',
               'closed':r'Closed Region $\Delta B\left[nT\right]$',
               'lobes':r'Lobes $\Delta B\left[nT\right]$'}
    fig,ax = plt.subplots(nrows=5,ncols=1,sharex=True,figsize=[14,17],
                          gridspec_kw={'height_ratios':[2,2,1,1,1]})
    #msdict.pop('missed')
    plot_stack_contrib(ax[0], mp.index,mp,msdict,ylabel=y1label,
                               value_key='Virial [nT]',ylim=[-125,25],
                               legend_loc='lower left')
    plot_stack_contrib(ax[1], mp.index,mp,msdict,ylabel=y2label,
                              ylim=[-125,25],
                               value_key='bioS [nT]', do_xlabel=False)
    for i,subzone in enumerate([sz for sz in msdict if 'missed' not in sz]):
        ylabels = [subzone+' '+vals+' Distribution']
        plot_stack_distr(ax[2+i],mp.index,mp,msdict,subzone=subzone,
                         ylabel=fig2ylabels[subzone],value_set=vals,
                         do_xlabel=(i is len(msdict)-1))
        ax[2+i].get_legend().remove()
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
    fig,ax = plt.subplots(nrows=7,ncols=1,sharex=True,figsize=[14,12],
                          gridspec_kw={'height_ratios':[1,1,1,1,1,1,2]})
    #Solar wind Bz
    for axis,obs in [(ax[0],f14obs), (ax[3],starobs)]:
        obstime = [dt.datetime(2000,1,1)+r for r in
                    obs['swmf_sw'].index-obs['swmf_sw'].index[0]]
        axis.plot(obstime,obs['swmf_sw']['bz'],label=swBlabel)
        axis.fill_between(obstime, np.sqrt(
                                                  obs['swmf_sw']['bx']**2+
                                                  obs['swmf_sw']['by']**2+
                                                  obs['swmf_sw']['bz']**2),
                                     color='grey',label=r'$|B|$')
        general_plot_settings(axis, ylabel=swBlabel,ylim=[-20,20],
                              legend_loc='lower left')
        axis.set_xlabel(None)
    #Ram pressure
    for axis,obs in [(ax[1],f14obs), (ax[4],starobs)]:
        obstime = [dt.datetime(2000,1,1)+r for r in
                    obs['swmf_sw'].index-obs['swmf_sw'].index[0]]
        axis.plot(obstime,obs['swmf_sw']['pdyn'])
        general_plot_settings(axis, ylabel=swPlabel,ylim=[0,15])
        axis.set_xlabel(None)
        axis.get_legend().remove()
    #AL
    for axis,obs in [(ax[2],f14obs), (ax[5],starobs)]:
        obstime = [dt.datetime(2000,1,1)+r for r in
                    obs['swmf_index'].index-obs['swmf_index'].index[0]]
        axis.plot(obstime,obs['swmf_index']['AL'],
                  label='Sim')
        axis.plot(obstime,obs['omni']['al'],label='Obs')
        general_plot_settings(axis, ylabel=Allabel,ylim=[-1500,0],
                              legend_loc='upper left')
        axis.set_xlabel(None)
        #axis.get_legend().remove()
    #Dst
    #for axis,obs in [(ax[6],f14obs), (ax[6],starobs)]:
    for axis,obs in [(ax[6],f14obs)]:
        obstime = [dt.datetime(2000,1,1)+r for r in
                    obs['swmf_log'].index-obs['swmf_log'].index[0]]
        #axis.plot(mp.index, mp['bioS [nT]'],linewidth=2,label='Sim')
        axis.plot(obstime, obs['swmf_log']['dst_sm'],
                  linewidth=2,label='Sim')
        omnitime = [dt.datetime(2000,1,1)+r for r in
                    obs['omni'].index-obs['omni'].index[0]]
        axis.plot(omnitime,obs['omni']['sym_h'],label='Obs')
        general_plot_settings(axis, ylabel=SYMhlabel)
    #Save
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'/egu2022_side.png')
    plt.close(fig)
    #Pressure, External Perturbations, Volume
    ######################################################################
    Plabel = r'$P_{ram} \left[nPa\right]$'
    Biotlabel = r'External $\Delta B\left[ nT\right]$'
    Vollabel = r'Volume $\left[R_e\right]$'
    fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=[14,6],
                              gridspec_kw={'height_ratios':[1,1,1]})
    #Ram pressure
    ax[0].plot(obs['swmf_sw'].index,obs['swmf_sw']['pdyn'])
    general_plot_settings(ax[0], ylabel=Plabel)
    ax[0].set_xlabel(None)
    ax[0].get_legend().remove()
    #Dst
    ax[1].plot(mp.index,mp['bioS_ext [nT]'],color='#05BC54')
    general_plot_settings(ax[1], ylabel=Biotlabel)
    ax[1].set_xlabel(None)
    ax[1].get_legend().remove()
    #Volume
    ax[2].plot(mp.index,mp['Volume [Re^3]'],color='#05BC54')
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
        ax.fill_between(mp.index,mp['Virial Surface TotalClosed [nT]']+
                            mp['Virial Surface TotalOpenN [nT]']+
                         mp['Virial Surface TotalOpenS [nT]'],color='grey')
        ax.plot(mp.index,mp['Virial Surface TotalClosed [nT]'],label='Closed')
        ax.plot(mp.index,mp['Virial Surface TotalOpenN [nT]'],label='North')
        ax.plot(mp.index,mp['Virial Surface TotalOpenS [nT]'],label='South')
    if False:
        #Lobes, closed field, RC
        ax.fill_between(mp.index,mp['Virial Surface Total [nT]']+
                            mp['Virial Volume Total [nT]'],color='grey')
        ax.plot(mp.index,mp['Virial Surface TotalClosed [nT]']+
                  msdict['closed']['Virial Volume Total [nT]'],
                    label='Closed')
        ax.plot(mp.index,mp['Virial Surface TotalOpenN [nT]']+
                    mp['Virial Surface TotalOpenS [nT]']+
                    msdict['lobes']['Virial Volume Total [nT]'],
                    label='Lobes')
        ax.plot(msdict['rc'].index,msdict['rc']['Virial Volume Total [nT]'],
                    label='RingCurrent')
    if False:
        ax.fill_between(mp.index,mp['Um [nT]'],color='grey')
        ax.plot(mp.index,mp['rhoU_r_acqu [J]']/(-8e13),label='Acquired')
        ax.plot(mp.index,mp['rhoU_r_forf [J]']/(-8e13),label='Forf')
        ax.plot(mp.index,mp['rhoU_r_net [J]']/(-8e13),label='Net')
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
