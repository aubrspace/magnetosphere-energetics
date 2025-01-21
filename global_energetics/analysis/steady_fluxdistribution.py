#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
#interpackage imports
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings)
from global_energetics.analysis.proc_hdf import (load_hdf_sort)
from global_energetics.analysis.steady_paper import ID_variability

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
    mp_surface_all = {}
    mp_ts = {}
    for case in [k for k in dataset.keys()]:
        mp_surface_all[case] = dataset[case]['mp_iso_betastar_surface']
        mp_ts[case] = mp_surface_all[case].index.get_level_values(0).unique()
        storage['mp_k1_'+case] = np.zeros(len(mp_ts[case]))
        storage['mp_k5_'+case] = np.zeros(len(mp_ts[case]))
        storage['mp_variance_'+case] = np.zeros(len(mp_ts[case]))


    for case in [k for k in dataset.keys()]:
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
        summary_data[key.replace('k1','k1var')],_,_ = ID_variability(
                                                            summary_data[key],
                                                            relative=False)
    return summary_data

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
    mp,(fluxes,variability,variance) = plt.subplots(3,1,figsize=[24,21],
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
    MSE1 = fluxes.text(1,0.93,f'MSE: {k5se.mean():.3f}[TW] ',
                       transform=fluxes.transAxes,
                       horizontalalignment='right',color='red')
    MSE1 = fluxes.text(1,0.03,f'MSE: {k1se.mean():.3f}[TW] ',
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
    MSE1 = variability.text(1,0.93,f'MSE: {k1varse.mean():.1f}[TW] ',
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
    MSE1 = variance.text(1,0.93,f'MSE: {vse.mean():.3f}[GW^2] ',
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
                          ylim=[0,variability.get_ylim()[1]],
                          timedelta=True)
    general_plot_settings(variance,do_xlabel=True,legend=False,
          ylabel=r'Variance of $\mathbf{K}\cdot\mathbf{n}\left[ GW^2\right]$',
                          xlim=[times[0],times[-1]],
                          ylim=[0,variance.get_ylim()[1]],
                          timedelta=True)
    fluxes.legend(loc='lower right', bbox_to_anchor=(1.0, 1.03),
                  ncol=5, fancybox=True, shadow=True)
    variance.set_xlabel(r'Time $\left[hr:min\right]$')
    fluxes.margins(x=0.1)
    variability.margins(x=0.1)
    variance.margins(x=0.1)
    mp.tight_layout()

    #Save
    figurename = (path+'/K_timeseries_magnetopause.pdf')
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
    events =['r2625_x120_b01',
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
        else:
            print(f'MISSING FILE: {GMfile}')

    ## Add summary data for timeseries
    summary_data = add_summary_data(dataset)

    ## Make Plots
    plot_tseries_magnetopause(summary_data,distributions)
