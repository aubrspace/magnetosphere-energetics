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
'''
from global_energetics.analysis import analyze_bow_shock
from global_energetics.analysis.plot_tools import (pyplotsetup,safelabel,
                                                   general_plot_settings,
                                                   plot_stack_distr,
                                                   plot_pearson_r,
                                                   plot_stack_contrib)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_satellites import read_satellites
from global_energetics.analysis.proc_hdf import (load_hdf_sort,
                                                 group_subzones,
                                                 get_subzone_contrib)
from global_energetics.analysis.analyze_energetics import plot_power
from global_energetics.analysis.proc_energy_spatial import reformat_lshell
from global_energetics.analysis.proc_timing import (peak2peak,
                                                    pearson_r_shifts)
'''

if __name__ == "__main__":
    #Need input path, then create output dir's
    inBase = sys.argv[-1]
    inAnalysis = os.path.join(sys.argv[-1],'analysis','IE')
    inLogs = os.path.join(sys.argv[-1],'')
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
    dphidt_day_north = central_diff(abs(dayFlux_north.rolling(6).mean()))
    dphidt_day_south = central_diff(abs(dayFlux_south.rolling(6).mean()))
    dphidt_night_north = central_diff(abs(nightFlux_north.rolling(6).mean()))
    dphidt_night_south = central_diff(abs(nightFlux_south.rolling(6).mean()))
    rxnDay_north = dphidt_day_north+night2dayNet_north
    rxnDay_south = dphidt_day_south+night2dayNet_south
    rxnNight_north = dphidt_night_north-night2dayNet_north
    rxnNight_south = dphidt_night_south-night2dayNet_south
    if True:
        #############
        #setup figure
        rx,(axis1,axis2,axis3) = plt.subplots(3,1,figsize=[24,16])
        #Plot
        '''
        axis.plot(times,rxnDay_north,label='dayNorth')
        axis.plot(times,rxnDay_south,label='daySouth')
        axis.plot(times,rxnNight_north,label='nightNorth')
        axis.plot(times,rxnNight_south,label='nightSouth')
        '''
        #axis1.plot(swt,sw['bx'],label='Bx')
        #axis1.plot(swt,sw['by'],label='By')
        #axis1.plot(swt,sw['bz'],label='Bz')
        #axis1.plot(swt,sw['Newell'],label='Newell')
        axis1.plot(logt,log['cpcpn'],label='CPCPn')
        axis1.plot(logt,log['cpcps'],label='CPCPs')
        axis2.plot(times,abs(dayFlux_north.rolling(6).mean()),label='Day')
        axis2.plot(times,abs(nightFlux_north.rolling(6).mean()),label='Night')
        axis2.plot(times,(abs(dayFlux_north)+
                          abs(nightFlux_north)).rolling(6).mean(),
                          label='Total')
        #from IPython import embed; embed()
        axis3.plot(times,rxnDay_north.rolling(6).mean(),label='Day')
        axis3.plot(times,rxnNight_north.rolling(6).mean(),label='Night')
        axis3.plot(times,night2dayNet_north,label='TermNight2Day')
        #Decorations
        general_plot_settings(axis1,do_xlabel=True,legend=True,
                              #ylabel=r'$B \left[nT\right]$',
                              ylabel=r'$CPCP\left[kV\right]$',
                              xlim=[dt.datetime(2000,6,24,6,0),
                                    dt.datetime(2000,6,24,7,0)],
                              timedelta=False)
        general_plot_settings(axis2,do_xlabel=True,legend=True,
                              #ylabel=r'$B \left[nT\right]$',
                              ylabel=r'$\vec{B}\cdot\vec{A}\left[Wb\right]$',
                              xlim=[dt.datetime(2000,6,24,6,0),
                                    dt.datetime(2000,6,24,7,0)],
                              timedelta=False)
        general_plot_settings(axis3,do_xlabel=True,legend=True,
                              ylabel=r'RXPotential $d\phi dt\left[V\right]$',
                              xlim=[dt.datetime(2000,6,24,6,0),
                                    dt.datetime(2000,6,24,7,0)],
                              timedelta=False)
        #save
        rx.tight_layout(pad=1)
        figurename = path+'/rx.png'
        rx.savefig(figurename)
        plt.close(rx)
        print('\033[92m Created\033[00m',figurename)
        #############
