#!/usr/bin/env python3
"""Analyze and plot data for the paper "Comprehensive Energy Analysis of
    a Simulated Magnetosphere"(working title)
"""
import os
import sys
import glob
import time
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
from matplotlib import ticker, colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.analysis.workingtitle import central_diff
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings)

if __name__ == "__main__":
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inAnalysis = os.path.join(sys.argv[-1],'analysis','IE')
    inLogs = os.path.join(inBase,'')
    outPath = os.path.join(inBase,'figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    ## Analysis Data
    dataset = {}
    #dataset['analysis'] = load_hdf_sort(inAnalysis+'/ie_results.h5')
    dataset['analysis'] = {}
    with pd.HDFStore(os.path.join(inAnalysis,'ie_results.h5')) as store:
        for key in store.keys():
            dataset['analysis'][key] = store[key]

    ## Logs and observation Data
    dataset['obs'] = read_indices(inLogs, prefix='', read_supermag=False)

    # Name the top level
    surface_north = dataset['analysis']['/ionosphere_north_surface']
    surface_south = dataset['analysis']['/ionosphere_south_surface']
    term_north = dataset['analysis']['/terminatornorth']
    term_south = dataset['analysis']['/terminatorsouth']
    times = term_north.index
    sw = dataset['obs']['swmf_sw']
    swt = sw.index
    log = dataset['obs']['swmf_log']
    logt = log.index
    # Name more specific stuff
    dayFlux_north = surface_north['Bf_injectionPolesDayN [Wb]']
    dayFlux_south = surface_south['Bf_escapePolesDayS [Wb]']
    nightFlux_north = surface_north['Bf_injectionPolesNightN [Wb]']
    nightFlux_south = surface_south['Bf_escapePolesNightS [Wb]']
    night2dayNet_north = term_north['dPhidt_net [Wb/s]']
    night2dayNet_south = term_south['dPhidt_net [Wb/s]']
    # Derive some things
    dphidt_day_north = central_diff(abs(dayFlux_north.rolling(1).mean()))
    dphidt_day_south = central_diff(abs(dayFlux_south.rolling(1).mean()))
    dphidt_night_north = central_diff(abs(nightFlux_north.rolling(1).mean()))
    dphidt_night_south = central_diff(abs(nightFlux_south.rolling(1).mean()))
    rxnDay_north = dphidt_day_north+night2dayNet_north
    rxnDay_south = dphidt_day_south+night2dayNet_south
    rxnNight_north = dphidt_night_north-night2dayNet_north
    rxnNight_south = dphidt_night_south-night2dayNet_south
    if True:
        #############
        #setup figure
        rx,(imf,cpcp,pcFlux,rxn) = plt.subplots(4,1,figsize=[32,16],
                                                    sharex=True)
        ##Plot
        # IMF
        imf.plot(swt,sw['bx'],label='Bx')
        imf.plot(swt,sw['by'],label='By')
        imf.plot(swt,sw['bz'],label='Bz')
        #axis1.plot(swt,sw['Newell'],label='Newell')
        # CPCP
        cpcp.plot(logt,log['cpcpn'],label='CPCPn')
        cpcp.plot(logt,log['cpcps'],label='CPCPs')
        # Polar Cap Flux
        pcFlux.plot(times,abs(dayFlux_north.rolling(1).mean()),label='Day')
        pcFlux.plot(times,abs(nightFlux_north.rolling(1).mean()),label='Night')
        pcFlux.plot(times,(abs(dayFlux_north)+
                          abs(nightFlux_north)).rolling(1).mean(),
                          label='Total')
        # Reconnection Rate
        rxn.plot(times,rxnDay_north.rolling(1).mean(),label='Day')
        rxn.plot(times,rxnNight_north.rolling(1).mean(),label='Night')
        rxn.plot(times,night2dayNet_north,label='TermNight2Day')
        ##Decorations
        # IMF
        general_plot_settings(imf,do_xlabel=False,legend=True,
                              ylabel=r'$B \left[nT\right]$',
                              xlim=[dt.datetime(2014,4,10,3,30),
                                    dt.datetime(2014,4,10,9,30)],
                              timedelta=False)
        # CPCP
        general_plot_settings(cpcp,do_xlabel=False,legend=True,
                              ylabel=r'$CPCP\left[kV\right]$',
                              xlim=[dt.datetime(2014,4,10,3,30),
                                    dt.datetime(2014,4,10,9,30)],
                              timedelta=False)
        # Polar Cap Flux
        general_plot_settings(pcFlux,do_xlabel=False,legend=True,
                              #ylabel=r'$B \left[nT\right]$',
                              ylabel=r'$\vec{B}\cdot\vec{A}\left[Wb\right]$',
                              xlim=[dt.datetime(2014,4,10,3,30),
                                    dt.datetime(2014,4,10,9,30)],
                              timedelta=False)
        # Reconnection Rate
        general_plot_settings(rxn,do_xlabel=True,legend=True,
                              ylabel=r'RXPotential $d\phi dt\left[V\right]$',
                              xlim=[dt.datetime(2014,4,10,3,30),
                                    dt.datetime(2014,4,10,9,30)],
                              timedelta=False)
        #save
        rx.tight_layout(pad=1)
        figurename = path+'/rx.png'
        rx.savefig(figurename)
        plt.close(rx)
        print('\033[92m Created\033[00m',figurename)
        #############
