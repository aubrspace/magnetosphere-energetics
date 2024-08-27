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
    ellipsoid_all = dataset['ellipsoid']['ellipsoid_surface']
    smooth_ellips_all = dataset['ellipsoid']['perfectellipsoid_surface']

    # Get times from multi Index dataframe
    ellipsoid_ts = ellipsoid_all.index.get_level_values(0).unique()
    smooth_ellips_ts = smooth_ellips_all.index.get_level_values(0).unique()

    # placeholder arrays
    ellipsoid_k1 = np.zeros(len(ellipsoid_ts))
    ellipsoid_k5 = np.zeros(len(ellipsoid_ts))
    smooth_ellips_k1 = np.zeros(len(ellipsoid_ts))
    smooth_ellips_k5 = np.zeros(len(ellipsoid_ts))

    for i,itime in enumerate(ellipsoid_ts):
        ellipsoid = ellipsoid_all.loc[itime]
        smooth_ellips = smooth_ellips_all.loc[itime]
        # Integrated K1static,K5static
        ellipsoid_k1[i],ellipsoid_k5[i] = integrate_distribution(ellipsoid)
        smooth_ellips_k1[i],smooth_ellips_k5[i] = integrate_distribution(
                                                                smooth_ellips)
    # Assign into summary data
    summary_data = pd.DataFrame({
                        'ellipsoid_K1':ellipsoid_k1,
                        'ellipsoid_K5':ellipsoid_k5,
                        'smooth_ellips_K1':smooth_ellips_k1,
                        'smooth_ellips_K5':smooth_ellips_k5},
                        index=ellipsoid_ts)
    # Calculate variability
    summary_data['ellipsoid_k1var'],_,_ = ID_variability(
                                                 summary_data['ellipsoid_K1'])
    summary_data['smooth_ellips_k1var'],_,_ = ID_variability(
                                             summary_data['smooth_ellips_K1'])
    return summary_data

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
    #############
    #setup figure
    ellip,(fluxes,variability,variance) = plt.subplots(3,1,figsize=[20,28],
                                                       sharex=True)
    #Plot
    from IPython import embed; embed()
    fluxes.plot(bin_centers,ellipsoid_cdf,color='blue',label='Ellipsoid')
    #TODO pull plotting routines from another place, mostly the xaxis stuff
    # Concept
    #   1st panel
    #       stack plots of K1,K5(note they're STATIC)
    #       include solid line for the smooth surface
    #   2nd panel 
    #       fill for variability of main timeseries
    #       include solid line for alternates to compare
    #   3rd panel
    #       fill for variance of the flux distribution
    #       include solid line for alternates
    #Decorate

    #Save
    figurename = (path+'/K_distribution_ellipsoid_'+
                    f'{itime.hour:02}{itime.minute:02}.png')
    Khist.savefig(figurename)
    plt.close(Khist)
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
    events =['ellipsoid']
    #TODO layer of organization here:
    #   distributtion of flux per time
    #   per run (X-20,X-40,X-80,X-120, etc.)
    #       may want to pull in all of it

    ## Analysis Data
    dataset = {}
    for event in events:
        GMfile = os.path.join(inDistribution,event+'.h5')
        if os.path.exists(GMfile):
            with pd.HDFStore(GMfile) as store:
                dataset[event] = {}
                for k in store.keys():
                    dataset[event][k.replace('/','')] = store[k]
                    df = dataset[event][k.replace('/','')]
                    #df.attrs['time']=store.get_storer(k).attrs.time

    ## Add summary data for timeseries
    for run in dataset.keys():
        summary_data = add_summary_data(dataset)

    ## Make Plots
    #plot_K_histogram_compare(dataset,distributions)
    #plot_tseries_ellipsoid(summary_data,distributions)
    #plot_tseries_magnetopause(summary_data,distributions)
