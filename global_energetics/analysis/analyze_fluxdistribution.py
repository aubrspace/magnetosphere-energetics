#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys,glob,time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
from scipy import stats
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings)
from global_energetics.analysis.proc_hdf import (load_hdf_sort)
from global_energetics.analysis.proc_indices import read_indices,ID_ALbays
from global_energetics.analysis.analyze_ideals import ID_variability

def data_to_cdf(data,bins,weights,**kwargs):
    """Function finds cumulative density function from data array
    Inputs
        data (1D arraylike)
        kwargs:
    Returns
        cdf (1D numpy array)
    """
    #pdf,bin_edges = np.histogram(data,bins=bins,weights=weights,density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    data.sort()
    cdf = np.zeros(len(bin_centers))
    for i,x in enumerate(bin_centers):
       cdf[i] = sum(data<=x)
    cdf = cdf/len(data)
    return cdf

def integrate_distribution(df,**kwargs):
    """Function handles dataframe of flux distribution to find some integrated
        quantities
    Inputs
    Returns
    """
    variable = kwargs.get('integrand','K_net [W/Re^2]')
    i_closed = df['Status']==3
    i_open = df['Status']!=3

    fluxes = df[variable]
    area = df['Area']

    K1 = np.dot(fluxes[i_open],area[i_open])
    K5 = np.dot(fluxes[i_closed],area[i_closed])
    return K1,K5

def add_summary_data(dataset,**kwargs):
    """Adds summary data, like integrated quantities to dataset
    Inputs
        dataset (dict{DataFrame})- pandas dataframe NOTE: it's multi indexed
        kwargs:
    Returns
        None
    """
    print('Gathering Summary Data ...')
    storage = {}
    if 'ellipsoid' in dataset:
        ellipsoid_all = dataset['ellipsoid']['ellipsoid_surface']
        smooth_ellips_all = dataset['ellipsoid']['perfectellipsoid_surface']

        # Get times from multi Index dataframe
        ellipsoid_ts = ellipsoid_all.index.get_level_values(0).unique()
        smooth_ellips_ts=smooth_ellips_all.index.get_level_values(0).unique()

        # placeholder arrays
        storage['ellipsoid_k1'] = np.zeros(len(ellipsoid_ts))
        storage['ellipsoid_k5'] = np.zeros(len(ellipsoid_ts))
        storage['smooth_ellips_k1'] = np.zeros(len(ellipsoid_ts))
        storage['smooth_ellips_k5'] = np.zeros(len(ellipsoid_ts))
        storage['ellipsoid_variance'] = np.zeros(len(ellipsoid_ts))
        storage['smooth_ellips_variance'] = np.zeros(len(ellipsoid_ts))
    mp_surface_all = {}
    mp_ts = {}
    for case in [k for k in dataset.keys() if 'ellipsoid' not in k]:
        mp_surface_all[case] = dataset[case]['mp_iso_betastar_surface']
        mp_ts[case] = mp_surface_all[case].index.get_level_values(0).unique()
        storage['mp_k1_'+case] = np.zeros(len(mp_ts[case]))
        storage['mp_k5_'+case] = np.zeros(len(mp_ts[case]))
        storage['mp_variance_'+case] = np.zeros(len(mp_ts[case]))


    if 'ellipsoid' in dataset:
        print('\tEllipsoid')
        for i,itime in enumerate(ellipsoid_ts):
            ellipsoid = ellipsoid_all.loc[itime]
            smooth_ellips = smooth_ellips_all.loc[itime]
            # Integrated K1static,K5static
            k1,k5 = integrate_distribution(ellipsoid)
            storage['ellipsoid_k1'][i] = k1
            storage['ellipsoid_k5'][i] = k5
            k1,k5 = integrate_distribution(smooth_ellips)
            storage['smooth_ellips_k1'][i] = k1
            storage['smooth_ellips_k5'][i] = k5
            # Variance of the distribution #NOTE to account for cell size!
            storage['ellipsoid_variance'][i] = (ellipsoid['K_net [W/Re^2]']*
                                                ellipsoid['Area']).var()
            storage['smooth_ellips_variance'][i] = (
                                            smooth_ellips['K_net [W/Re^2]']*
                                            smooth_ellips['Area']).var()
    #storage['xmin'] = np.zeros(len(mp_ts[case]))
    #for i,itime in enumerate(mp_ts[case]):
    #    storage['xmin'][i]=mp_surface_all['r45_x150_b07']['X'].loc[itime].min()
    for case in [k for k in dataset.keys() if 'ellipsoid' not in k]:
        print(f'\tMagnetopause: {case}')
        for i,itime in enumerate(mp_ts[case]):
            mp_surface = mp_surface_all[case].loc[itime]
            # Integrated K1static,K5static
            k1,k5 = integrate_distribution(mp_surface)
            # Check if the flux has been reversed
            if (k1>0) & (k5<0):
                k1 *= -1
                k5 *= -1
            storage['mp_k1_'+case][i] = k1
            storage['mp_k5_'+case][i] = k5
            # Variance of the distribution #NOTE to account for cell size!
            storage['mp_variance_'+case][i] = (mp_surface['K_net [W/Re^2]']*
                                              mp_surface['Area']).var()
    # Assign into summary data
    summary_data = pd.DataFrame(storage,
                                index=mp_ts[case])#NOTE assuming same size
    # Calculate variability
    print(f'\tCalculating variability')
    for key in [k for k in summary_data.keys() if 'k1' in k]:
        #print(f"\t\t{key}->{key.replace('k1','k1var')}")
        summary_data[key.replace('k1','k1var')],_,_ = ID_variability(
                                                            summary_data[key],
                                                            relative=False)
    return summary_data

def plot_K_histogram_compare2(dataset,path,**kwargs):
    """Plot total energy flux K
    Inputs
        dataset (dict{DataFrame})- pandas dataframe NOTE: it's multi indexed
        path
        summary_data
        kwargs:
    Returns
        None
    """
    # Colorwheel
    colors = ['tab:brown','tab:pink','tab:grey','tab:purple']
    # Holder for data in dataset that's not our control data
    test_surface_all = {}
    # Use control data to set the unique time array so we can iterate
    mp_ts = dataset['r3_x120_b07'][
                 'mp_iso_betastar_surface'].index.get_level_values(0).unique()
    for case in [k for k in dataset.keys() if 'x120_b07' not in k]:
        test_surface_all[case] = dataset[case]['mp_iso_betastar_surface']


    # Loop through each timestep
    for i,itime in enumerate(mp_ts):
        print(f'\ttime: {itime}')
        # Find the statistical information about the control data @this time
        base_surface = dataset['r3_x120_b07'][
                                        'mp_iso_betastar_surface'].loc[itime]
        base_bins = np.linspace(
              (base_surface['K_net [W/Re^2]']*base_surface['Area']).min()/1e9,
              (base_surface['K_net [W/Re^2]']*base_surface['Area']).max()/1e9,
                 250)
        base_bin_cc = 0.5 * (base_bins[:-1] + base_bins[1:])
        # Calculate CDF
        base_cdf = data_to_cdf(base_surface['K_net [W/Re^2]'].values/1e9,
                               base_bins,base_surface['Area'])
        # Holders for the rest of the data
        cdfs = {}
        all_results = {}
        test_bins = {}
        test_bin_cc = {}
        for case in [k for k in dataset.keys() if 'x120_b07' not in k]:
            # Now we loop through each test case and repeat above & compare
            test_surface = test_surface_all[case].loc[itime]
            test_bins[case] = np.linspace(
              (test_surface['K_net [W/Re^2]']*test_surface['Area']).min()/1e9,
              (test_surface['K_net [W/Re^2]']*test_surface['Area']).max()/1e9,
                 250)
            test_bin_cc[case] = 0.5*(test_bins[case][:-1]+test_bins[case][1:])
            # Calculate CDF
            test_cdf = data_to_cdf(test_surface['K_net [W/Re^2]'].values/1e9,
                                   test_bins[case],test_surface['Area'])
            cdfs[case] = test_cdf#add to our storage dict for later vis
            # Calculate the Kolmogorov-Smirnov goodness of fit statistic
            all_results[case] = stats.ks_2samp(
                    (test_surface['K_net [W/Re^2]']*test_surface['Area'])/1e9,
                    (base_surface['K_net [W/Re^2]']*base_surface['Area'])/1e9)
        # Pick out the best and worst fit for this timestep based on KS_2
        r_best = stats._stats_py.KstestResult(statistic=9999,
                                              pvalue=0.0,
                                              statistic_location=0,
                                              statistic_sign=1)
        r_worst = stats._stats_py.KstestResult(statistic=0,
                                              pvalue=0.0,
                                              statistic_location=0,
                                              statistic_sign=1)
        for case,r in all_results.items():
            if r.statistic>r_worst.statistic:
                r_worst = r
                worst_case = case
            if r.statistic<r_best.statistic:
                r_best = r
                best_case = case
        # Now get the XY coordinates so that we can visualize it on the plot
        ytest_worse = np.interp(r_worst.statistic_location,
                                test_bin_cc[worst_case],cdfs[worst_case])
        ybase_worse = np.interp(r.statistic_location,base_bin_cc,base_cdf)
        ytest_best = np.interp(r_best.statistic_location,
                                test_bin_cc[best_case],cdfs[best_case])
        ybase_best = np.interp(r_best.statistic_location,base_bin_cc,base_cdf)
        ks2_worst = r_worst.statistic*r_worst.statistic_sign
        ks2_best = r_best.statistic*r_best.statistic_sign
        # Get the max and min mean value
        mean_max =max([(test_surface_all[k].loc[itime]['K_net [W/Re^2]']*
                        test_surface_all[k].loc[itime]['Area']).mean()/1e9
                      for k in test_surface_all.keys()])
        mean_min =min([(test_surface_all[k].loc[itime]['K_net [W/Re^2]']*
                        test_surface_all[k].loc[itime]['Area']).mean()/1e9
                      for k in test_surface_all.keys()])

        integ_max=max([np.dot(test_surface_all[k].loc[itime]['K_net [W/Re^2]'],
                              test_surface_all[k].loc[itime]['Area'])/1e12
                      for k in test_surface_all.keys()])
        integ_min=min([np.dot(test_surface_all[k].loc[itime]['K_net [W/Re^2]'],
                              test_surface_all[k].loc[itime]['Area'])/1e12
                      for k in test_surface_all.keys()])
        integ_base = np.dot(base_surface['K_net [W/Re^2]'],
                            base_surface['Area'])/1e12
        #############
        #setup figure
        Khist,(histo,cdf) = plt.subplots(2,1,figsize=[20,30],sharex=True)
        #Plot
        histo.hist(base_surface['K_net [W/Re^2]']/1e9,bins=base_bins,
                   weights=base_surface['Area'],label='Base',color='blue',
                   edgecolor=None,alpha=0.5)
        cdf.plot(base_bin_cc,base_cdf,color='blue',label='Base')
        for i,case in enumerate([k for k in dataset.keys()
                                                     if 'x120_b07' not in k]):
            test_surface = test_surface_all[case].loc[itime]
            test_hist,_ = np.histogram(test_surface['K_net [W/Re^2]']/1e9,
                            bins=test_bins[case],weights=test_surface['Area'])
            print(f'{case} cell count: {len(test_surface)}')
            histo.hist(test_surface['K_net [W/Re^2]']/1e9,bins=test_bins[case],
                       weights=test_surface['Area'],label=case,fill=True,
                       color=colors[i],alpha=0.5)
            histo.plot(test_bin_cc[case],test_hist,label='_'+case,c=colors[i])
            cdf.plot(test_bin_cc[case],cdfs[case],label='Case',c=colors[i])
        #Decorate
        tstamp = histo.text(0,1.01,f'Time: {itime}',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='black')
        mean1 = histo.text(1,0.95,f'MaxMean: ={mean_max:.4f}'+r'$[MW/Re^2]$ ',
                       transform=histo.transAxes,
                       horizontalalignment='right',color='black')
        mean2 = histo.text(1,0.90,f'MinMean: ={mean_min:.4f}'+r'$[MW/Re^2]$ ',
                       transform=histo.transAxes,
                       horizontalalignment='right',color='black')
        integr1 = histo.text(0,0.95,r' MAX $\int\mathbf{K}$: ='+
                          f'{integ_max:.4f}[TW]',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='black')
        integr2 = histo.text(0,0.90,r' MIN $\int\mathbf{K}$: ='+
                          f'{integ_min:.4f}[TW]',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='black')
        integr3 = histo.text(0,0.85,r' Base $\int\mathbf{K}$: ='+
                          f'{integ_base:.4f}[TW]',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='black')
        KS1 = cdf.text(0,0.95,f' K-S Best: {ks2_best:.4f}',
                       transform=cdf.transAxes,
                       horizontalalignment='left',color='black')
        KS2 = cdf.text(0,0.90,f' K-S Worst: {ks2_worst:.4f}',
                       transform=cdf.transAxes,
                       horizontalalignment='left',color='black')
        histo.axvline(0,c='black')
        cdf.axvline(0,c='black')
        cdf.plot(2*[r_best.statistic_location],
                 [np.min([ytest_best,ybase_best]),
                  np.max([ytest_best,ybase_best])],
              ls='--',color='red',lw=3)
        cdf.set_xlabel(r'Energy Flux $\left[MW/R_e^2\right]$')
        cdf.set_ylabel(r'Cumulative Probability')
        #cdf.legend(loc='lower left')
        cdf.set_xlim(-10,10)
        cdf.margins(x=0.01)

        histo.axvline((base_surface['K_net [W/Re^2]']*
                       base_surface['Area']).mean()/1e9,c='blue',ls='--')
        histo.set_ylabel('Area Weighted Count')
        #histo.legend(loc='lower left')
        histo.set_xlim(-10,10)
        histo.margins(x=0.01)
        histo.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=3, fancybox=True, shadow=True)
        Khist.tight_layout()
        #Save
        figurename = (path+'/K_distribution_magnetopause_'+
                      f'{itime.hour:02}{itime.minute:02}.png')
        Khist.savefig(figurename)
        plt.close(Khist)
        print('\033[92m Created\033[00m',figurename)

def plot_K_histogram_compare(dataset,path,**kwargs):
    """Plot total energy flux K
    Inputs
        dataset (dict{DataFrame})- pandas dataframe NOTE: it's multi indexed
        path
        summary_data
        kwargs:
    Returns
        None
    """
    ellipsoid_all = dataset['ellipsoid']['ellipsoid_surface']
    smooth_ellips_all = dataset['ellipsoid']['perfectellipsoid_surface']

    # Get times from multi Index dataframe
    ellipsoid_ts = ellipsoid_all.index.get_level_values(0).unique()
    smooth_ellips_ts = smooth_ellips_all.index.get_level_values(0).unique()

    for itime in ellipsoid_ts[0:1]:
        ellipsoid = ellipsoid_all.loc[itime]
        smooth_ellips = smooth_ellips_all.loc[itime]
        bins = np.linspace(
            (ellipsoid['K_net [W/Re^2]']*ellipsoid['Area']).min()/1e9,
            (ellipsoid['K_net [W/Re^2]']*ellipsoid['Area']).max()/1e9,
            250)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        hist_min = np.min([
               (ellipsoid['K_net [W/Re^2]']*ellipsoid['Area']).min()/1e9,
               (smooth_ellips['K_net [W/Re^2]']*ellipsoid['Area']).min()/1e9])
        hist_max = np.max([
               (ellipsoid['K_net [W/Re^2]']*ellipsoid['Area']).max()/1e9,
               (smooth_ellips['K_net [W/Re^2]']*ellipsoid['Area']).max()/1e9])
        #Calculate CDF from ellipsoid and smooth cases
        ellipsoid_cdf = data_to_cdf(ellipsoid['K_net [W/Re^2]'].values/1e9,
                                    bins,ellipsoid['Area'])
        smooth_ellips_cdf = data_to_cdf(
                                smooth_ellips['K_net [W/Re^2]'].values/1e9,
                                    bins,smooth_ellips['Area'])
        delta_cdf = ellipsoid_cdf-smooth_ellips_cdf
        delta_max = abs(delta_cdf).max()
        delta_imax = np.where(abs(delta_cdf)==abs(delta_cdf).max())[0]
        #Calculate the Kolmogorov-Smirnov goodness of fit statistic
        result = stats.ks_2samp(
                (ellipsoid['K_net [W/Re^2]']*ellipsoid['Area'])/1e9,
                (smooth_ellips['K_net [W/Re^2]']*smooth_ellips['Area'])/1e9)
        ks2 = result.statistic*result.statistic_sign
        #TODO see if comparing to reference value make sense?
        #TODO see if the confidence interval can be determined
        #bins = np.linspace(-30,50,151)
        #############
        #setup figure
        Khist,(histo,cdf) = plt.subplots(2,1,figsize=[12,24],sharex=True)
        #Plot
        histo.hist(ellipsoid['K_net [W/Re^2]']/1e9,bins=bins,
                   weights=ellipsoid['Area'],label='Ellipsoid',color='blue',
                   alpha=0.7)
        histo.hist(smooth_ellips['K_net [W/Re^2]']/1e9,bins=bins,
                   weights=smooth_ellips['Area'],label='Smooth',color='grey',
                   alpha=0.7)
        cdf.plot(bin_centers,ellipsoid_cdf,color='blue',label='Ellipsoid')
        cdf.plot(bin_centers,smooth_ellips_cdf,color='grey',label='Smooth')
        cdf.plot(bin_centers,delta_cdf,color='red',label='Difference')
        #Decorate
        ellipsoid_integ = np.dot(ellipsoid['K_net [W/Re^2]'],
                                 ellipsoid['Area'])/1e12
        ellipsoid_mean = ellipsoid_integ*1e12/np.sum(ellipsoid['Area'])/1e9
        smooth_ellips_integ = np.dot(smooth_ellips['K_net [W/Re^2]'],
                                     smooth_ellips['Area'])/1e12
        smooth_ellips_mean = smooth_ellips_integ*1e12/np.sum(
                                                 smooth_ellips['Area'])/1e9
        mean1 = histo.text(1,0.94,f'Mean: ={ellipsoid_mean:.2f}[MW/Re^2]',
                       transform=histo.transAxes,
                       horizontalalignment='right',color='blue')
        mean2 = histo.text(1,0.84,f'Mean: ={smooth_ellips_mean:.2f}[MW/Re^2]',
                       transform=histo.transAxes,
                       horizontalalignment='right',color='grey')
        integr1 = histo.text(0,0.94,r'$\int\mathbf{K}$: ='+
                          f'{ellipsoid_integ:.2f}[TW]',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='blue')
        integr2 = histo.text(0,0.84,r'$\int\mathbf{K}$: ='+
                          f'{smooth_ellips_integ:.2f}[TW]',
                       transform=histo.transAxes,
                       horizontalalignment='left',color='grey')
        KS = cdf.text(0,0.94,f'KS_2Sample: {ks2:.2f}',
                       transform=cdf.transAxes,
                       horizontalalignment='left',color='red')
        maxdiff = cdf.text(0,0.84,f'MaxDiff: {delta_cdf[delta_imax][0]:.2f}',
                       transform=cdf.transAxes,
                       horizontalalignment='left',color='red')
        cdf.plot(2*[bin_centers[delta_imax]],
            [np.min([ellipsoid_cdf[delta_imax],smooth_ellips_cdf[delta_imax]]),
            np.max([ellipsoid_cdf[delta_imax],smooth_ellips_cdf[delta_imax]])],
              ls='--',color='red',lw=3)
        cdf.set_xlabel(r'Energy Flux $\left[MW/R_e^2\right]$')
        cdf.set_ylabel(r'Cumulative Probability')
        #cdf.legend(loc='lower left')
        cdf.set_xlim(-10,10)
        cdf.margins(x=0.01)

        histo.axvline(ellipsoid_mean,c='blue',ls='--')
        histo.axvline(smooth_ellips_mean,c='grey',ls='--')
        #histo.set_xlabel(r'Energy Flux $\left[MW/R_e^2\right]$')
        histo.set_ylabel('Area Weighted Count')
        #histo.legend(loc='lower left')
        histo.set_xlim(-10,10)
        histo.margins(x=0.01)
        histo.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=2, fancybox=True, shadow=True)
        Khist.tight_layout()
        #Save
        figurename = (path+'/K_distribution_ellipsoid_'+
                      f'{itime.hour:02}{itime.minute:02}.png')
        Khist.savefig(figurename)
        plt.close(Khist)
        print('\033[92m Created\033[00m',figurename)

def plot_tseries_ellipsoid(df,path,**kwargs):
    T0 = dt.datetime(2022,6,6,0,0)
    tdelta = [t-T0 for t in df.index]
    times = [float(n.to_numpy()) for n in tdelta]
    #############
    #setup figure
    ellip,(fluxes,variability,variance) = plt.subplots(3,1,figsize=[28,32],
                                                       sharex=True)
    #Plot
    fluxes.fill_between(times,df['ellipsoid_K1']/1e12,fc='blue',alpha=0.4,
                        label='Ellipsoid K1')
    fluxes.fill_between(times,df['ellipsoid_K5']/1e12,fc='red',alpha=0.4,
                        label='Ellipsoid K5')
    fluxes.plot(times,df['smooth_ellips_K1']/1e12,c='blue',label='Smooth K1')
    fluxes.plot(times,df['smooth_ellips_K5']/1e12,c='red',label='Smooth K5')

    variability.fill_between(times,df['ellipsoid_k1var'],fc='blue',alpha=0.4,
                             label='Ellipsoid RelVar[K1]')
    variability.plot(times,df['smooth_ellips_k1var'],c='blue',
                             label='Smooth RelVar[K1]')

    variance.fill_between(times,df['ellipsoid_var']/1e18,fc='blue',
                          alpha=0.4,label='Ellipsoid Variance[K1]')
    variance.plot(times,df['smooth_ellips_var']/1e18,c='blue',
                          label='Ellipsoid Variance[K1]')
    #Decorate
    general_plot_settings(fluxes,do_xlabel=False,legend=True,
             ylabel=r'Static $\int\mathbf{K}\cdot\mathbf{n}\left[ TW\right]$',
                          timedelta=True)
    general_plot_settings(variability,do_xlabel=False,legend=True,
             ylabel='Rel.Variability of '+
                    r'$\int\mathbf{K}\cdot\mathbf{n}\left[\%\right]$',
                          ylim=[0,500],
                          timedelta=True)
    general_plot_settings(variance,do_xlabel=True,legend=True,
          ylabel=r'Variance of $\mathbf{K}\cdot\mathbf{n}\left[ GW\right]^2$',
                          timedelta=True)
    variance.set_xlabel(r'Time $\left[hr:min\right]$')
    fluxes.margins(x=0.1)
    variability.margins(x=0.1)
    variance.margins(x=0.1)
    ellip.tight_layout()

    #Save
    figurename = (path+'/K_timeseries_ellipsoid.png')
    ellip.savefig(figurename)
    plt.close(ellip)
    print('\033[92m Created\033[00m',figurename)

def plot_tseries_magnetopause(df,path,**kwargs):
    print('Preparing magnetopause plots ...')
    # Time handling
    T0 = dt.datetime(2022,6,6,0,0)
    tdelta = [t-T0 for t in df.index]
    times = [float(n.to_numpy()) for n in tdelta]
    mark1 = float(pd.Timedelta(dt.datetime(2022,6,7,0,10)-T0).to_numpy())
    mark2 = float(pd.Timedelta(dt.datetime(2022,6,7,2,10)-T0).to_numpy())
    mark3 = float(pd.Timedelta(dt.datetime(2022,6,7,4,10)-T0).to_numpy())
    mark4 = float(pd.Timedelta(dt.datetime(2022,6,7,6,10)-T0).to_numpy())
    # Means and Standard Errors
    k1keys = [k for k in df.keys() if 'k1_' in k]
    k1mean = df[k1keys].mean(axis='columns')/1e12
    k1se = df[k1keys].std(axis='columns')/np.sqrt(5)/1e12

    k5keys = [k for k in df.keys() if 'k5_' in k]
    k5mean = df[k5keys].mean(axis='columns')/1e12
    k5se = df[k5keys].std(axis='columns')/np.sqrt(5)/1e12

    k1varkeys = [k for k in df.keys() if 'k1var_' in k]
    k1varmean = df[k1varkeys].mean(axis='columns')/1e12
    k1varse = df[k1varkeys].std(axis='columns')/np.sqrt(5)/1e12

    vkeys = [k for k in df.keys() if 'variance_' in k]
    vmean = df[vkeys].mean(axis='columns')/1e18
    vse = df[vkeys].std(axis='columns')/np.sqrt(5)/1e18

    #############
    #setup figure
    mp,(fluxes,variability,variance) = plt.subplots(3,1,figsize=[28,32],
                                                       sharex=True)
    #Plot
    fluxes.fill_between(times,df['mp_k1_r3_x120_b07']/1e12,fc='blue',
                        alpha=0.4,label=r'K1 (Baseline)')
    fluxes.fill_between(times,df['mp_k5_r3_x120_b07']/1e12,fc='red',
                        alpha=0.4,label=r'K5 (Baseline)')
    fluxes.plot(times,df['mp_k1_r2625_x120_b01']/1e12,c='black',
                        #label=r'_($\beta^*=0.1$)')
                        label=r'Variations')
    fluxes.plot(times,df['mp_k5_r2625_x120_b01']/1e12,c='black',
                        label=r'_($\beta^*=0.1$)')
    fluxes.plot(times,df['mp_k1_r275_x120_b14']/1e12,c='black',
                        label=r'_($\beta^*=1.4$)')
    fluxes.plot(times,df['mp_k5_r275_x120_b14']/1e12,c='black',
                        label=r'_($\beta^*=1.4$)')
    fluxes.plot(times,df['mp_k1_r4_x20_b07']/1e12,c='black',
                        label=r'_($X=-20$)')
    fluxes.plot(times,df['mp_k5_r4_x20_b07']/1e12,c='black',
                        label=r'_($X=-20$)')
    fluxes.plot(times,df['mp_k1_r45_x150_b07']/1e12,c='black',
                        label=r'_($X=-150$)')
    fluxes.plot(times,df['mp_k5_r45_x150_b07']/1e12,c='black',
                        label=r'_($X=-150$)')
    fluxes.plot(times,k1mean,c='red',ls='--',label='Mean')
    fluxes.fill_between(times,k1mean-k1se,k1mean+k1se,fc='grey',alpha=0.6,
                        label='Standard Error')
    fluxes.plot(times,k5mean,c='red',ls='--',label='_mean')
    fluxes.fill_between(times,k5mean-k5se,k5mean+k5se,fc='grey',alpha=0.6,
                        label='_standard error')
    MSE1 = fluxes.text(1,0.95,f'MSE: {k5se.mean():.3f}[TW] ',
                       transform=fluxes.transAxes,
                       horizontalalignment='right',color='red')
    MSE1 = fluxes.text(1,0.05,f'MSE: {k1se.mean():.3f}[TW] ',
                       transform=fluxes.transAxes,
                       horizontalalignment='right',color='blue')

    variability.fill_between(times,df['mp_k1var_r3_x120_b07']/1e12,fc='blue',
                             alpha=0.4,label=r'Variability(K1) (Baseline)')
    variability.plot(times,df['mp_k1var_r2625_x120_b01']/1e12,c='black',
                            label=r'($\beta^*=0.1$)')
    variability.plot(times,df['mp_k1var_r275_x120_b14']/1e12,c='black',
                            label=r'($\beta^*=1.4$)')
    variability.plot(times,df['mp_k1var_r4_x20_b07']/1e12,c='black',
                            label=r'($X=-20$)')
    variability.plot(times,df['mp_k1var_r45_x150_b07']/1e12,c='black',
                            label=r'($X=-150$)')
    variability.plot(times,k1varmean,c='red',ls='--',label='Mean')
    variability.fill_between(times,k1varmean-k1varse,k1varmean+k1varse,
                             fc='grey',alpha=0.6,label='Standard Error')
    MSE1 = variability.text(1,0.95,f'MSE: {k1varse.mean():.1f}[TW] ',
                       transform=variability.transAxes,
                       horizontalalignment='right',color='black')

    variance.fill_between(times,df['mp_variance_r3_x120_b07']/1e18,
                fc='blue',alpha=0.4,label=r'Variance(K) (Baseline)')
    variance.plot(times,df['mp_variance_r2625_x120_b01']/1e18,
                           c='black',label=r'($\beta^*=0.1$)')
    variance.plot(times,df['mp_variance_r275_x120_b14']/1e18,
                           c='black',label=r'($\beta^*=1.4$)')
    variance.plot(times,df['mp_variance_r4_x20_b07']/1e18,
                           c='black',label=r'($X=-20$)')
    variance.plot(times,df['mp_variance_r45_x150_b07']/1e18,
                           c='black',label=r'($X=-150$)')
    variance.plot(times,vmean,c='red',ls='--',label='Mean')
    variance.fill_between(times,vmean-vse,vmean+vse,
                             fc='grey',alpha=0.6,label='Standard Error')
    MSE1 = variance.text(1,0.95,f'MSE: {vse.mean():.3f}[GW^2] ',
                       transform=variance.transAxes,
                       horizontalalignment='right',color='black')
    for mark in [mark1,mark2,mark3,mark4]:
        fluxes.axvline(mark,c='grey')
        variability.axvline(mark,c='grey')
        variance.axvline(mark,c='grey')
    #Decorate
    general_plot_settings(fluxes,do_xlabel=False,legend=False,
             ylabel=r'Static $\int\mathbf{K}\cdot\mathbf{n}\left[ TW\right]$',
                          timedelta=True)
    general_plot_settings(variability,do_xlabel=False,legend=False,
             ylabel='Variability of '+
                    r'$\int\mathbf{K}\cdot\mathbf{n}\left[ TW\right]$',
                          #ylim=[0,280],
                          timedelta=True)
    general_plot_settings(variance,do_xlabel=True,legend=False,
          ylabel=r'Variance of $\mathbf{K}\cdot\mathbf{n}\left[ GW^2\right]$',
                          timedelta=True)
    fluxes.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                  ncol=5, fancybox=True, shadow=True)
    variance.set_xlabel(r'Time $\left[hr:min\right]$')
    fluxes.margins(x=0.1)
    variability.margins(x=0.1)
    variance.margins(x=0.1)
    mp.tight_layout()

    #Save
    figurename = (path+'/K_timeseries_magnetopause.png')
    mp.savefig(figurename)
    plt.close(mp)
    print('\033[92m Created\033[00m',figurename)


if __name__ == "__main__":
    T0 = dt.datetime(2022,6,6,0,0)
    TSTART = dt.datetime(2022,6,6,0,10)
    DT = dt.timedelta(hours=2)
    TJUMP = dt.timedelta(hours=2)
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inLogs = os.path.join(sys.argv[-1],'data/logs/')
    inDistribution = os.path.join(sys.argv[-1],'data/analysis/distributions/')
    outPath = os.path.join(inBase,'figures')
    distributions = os.path.join(outPath,'distributions')
    for path in [outPath,distributions]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    # Event list
    events =[#'ellipsoid',
            'r2625_x120_b01',
            'r275_x120_b14',
            'r3_x120_b07',
            'r4_x20_b07',
            'r45_x150_b07'
            ]

    ## Analysis Data
    dataset = {}
    for event in events:
        GMfile = os.path.join(inDistribution,event+'.h5')
        if os.path.exists(GMfile):
            print(f'Loading {GMfile} ...')
            with pd.HDFStore(GMfile) as store:
                dataset[event] = {}
                for k in store.keys():
                    dataset[event][k.replace('/','')] = store[k]
                    df = dataset[event][k.replace('/','')]
                    #df.attrs['time']=store.get_storer(k).attrs.time
        else:
            print(f'MISSING FILE: {GMfile}')

    ## Add summary data for timeseries
    summary_data = add_summary_data(dataset)

    ## Make Plots
    #plot_K_histogram_compare(dataset,distributions)
    #plot_K_histogram_compare2(dataset,distributions)
    #plot_tseries_ellipsoid(summary_data,distributions)
    plot_tseries_magnetopause(summary_data,distributions)
