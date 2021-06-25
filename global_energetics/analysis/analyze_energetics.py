#!/usr/bin/env python3
"""module calls on processing scripts to gather data then creates plots
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs,pi,cos, sin, sqrt, rad2deg, matmul, deg2rad,linspace
import scipy as sp
from scipy.interpolate import griddata
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import swmfpy
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage imports
from global_energetics.analysis.proc_temporal import read_energetics
from global_energetics.analysis.proc_indices import (read_indices,
                                                     get_expanded_sw,
                                                     df_coord_transform)

def shade_plot(axis, *, do_full=False):
    """Credit Qusai from NSF proposal:
        Shade the ICME regions"""

    # ICME Timings
    def hour(minutes):
        """return hour as int from given minutes"""
        return int((minutes)/60//24)

    def minute(minutes):
        """return minute as int from given minutes"""
        return int(minutes % 60)

    # From Tuija email
    icme = (

        # (dt.datetime(2014, 2, 15, 13+hour(25), minute(25)),  # SH1
        #  dt.datetime(2014, 2, 16, 4+hour(45), minute(45)),  # EJ1
        #  dt.datetime(2014, 2, 16, 16+hour(55), minute(55))),  # ET1

        (dt.datetime(2014, 2, 18, 7+hour(6), minute(6)),  # SH2
         dt.datetime(2014, 2, 18, 15+hour(45), minute(45)),  # EJ2
         dt.datetime(2014, 2, 19, 3+hour(55), minute(55))),  # ET2

        (dt.datetime(2014, 2, 19, 3+hour(56), minute(56)),  # SH3
         dt.datetime(2014, 2, 19, 12+hour(45), minute(45)),  # EJ3
         dt.datetime(2014, 2, 20, 3+hour(9), minute(9))),  # ET3

        # (dt.datetime(2014, 2, 20, 3+hour(9), minute(9)),  # SH4
        #  dt.datetime(2014, 2, 21, 3+hour(15), minute(15)),  # EJ4
        #  dt.datetime(2014, 2, 22, 13+hour(00), minute(00))),  # ET4

        )
    fullicme = (

        (dt.datetime(2014, 2, 15, 13+hour(25), minute(25)),  # SH1
         dt.datetime(2014, 2, 16, 4+hour(45), minute(45)),  # EJ1
         dt.datetime(2014, 2, 16, 16+hour(55), minute(55))),  # ET1

        (dt.datetime(2014, 2, 18, 7+hour(6), minute(6)),  # SH2
         dt.datetime(2014, 2, 18, 15+hour(45), minute(45)),  # EJ2
         dt.datetime(2014, 2, 19, 3+hour(55), minute(55))),  # ET2

        (dt.datetime(2014, 2, 19, 3+hour(56), minute(56)),  # SH3
         dt.datetime(2014, 2, 19, 12+hour(45), minute(45)),  # EJ3
         dt.datetime(2014, 2, 20, 3+hour(9), minute(9))),  # ET3

        (dt.datetime(2014, 2, 20, 3+hour(9), minute(9)),  # SH4
         dt.datetime(2014, 2, 21, 3+hour(15), minute(15)),  # EJ4
         dt.datetime(2014, 2, 22, 13+hour(00), minute(00))),  # ET4

        )
    if do_full:
        icme = fullicme

    for num, times in enumerate(icme, start=2):
        sheath = mpl.dates.date2num(times[0])
        ejecta = mpl.dates.date2num(times[1])
        end = mpl.dates.date2num(times[2])
        axis.axvspan(sheath, ejecta, facecolor='k', alpha=0.1,
                     label=('sheath ' + str(num)))
        axis.axvspan(ejecta, end, facecolor='k', alpha=0.4,
                     label=('ejecta ' + str(num)))

def plot_Power_al(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, multiplier=-1, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        powerin = 'K_injection [W]'
        powerout = 'K_escape [W]'
        powernet = 'K_net [W]'
        if name.find('mp')!=-1:
            #INJECTION
            axis.plot(data[timekey],multiplier*abs(data[powerin]),
                            label=r'$\displaystyle K_{injection}$',
                        linewidth=Size, linestyle=ls,
                        color='gold')
            #ESCAPE
            axis.plot(data[timekey],multiplier*abs(data[powerout]),
                            label=r'$\displaystyle K_{escape}$',
                        linewidth=Size, linestyle=ls,
                        color='deepskyblue')
        else:
            axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='plum')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_Power(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, use_average=False):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    if Color != None:
        override_color = True
        if dflist[0]['name'].iloc[-1].find('aggr')!=-1:
            #TBD!!! somehow switch colors!
            pass
    else:
        override_color = False
    legend_loc = 'upper right'
    for data in dflist:
        name = data['name'].iloc[-1]
        powerin_str = 'K_injection [W]'
        powerout_str = 'K_escape [W]'
        powernet_str = 'K_net [W]'
        if use_inner:
            powerin_str = 'inner'+powerin_str
            powerout_str = 'inner'+powerout_str
            powernet_str = 'inner'+powernet_str
        if (name.find('mp')!=-1) or (name.find('aggr')!=-1):
            if use_shield:
                oneDin = abs(data['1D'+powerin_str])*100+1
                oneDout = abs(data['1D'+powerout_str])*100+1
                oneDnet = abs(data['1D'+powernet_str])*100+1
                powerin = data[powerin_str]/oneDin
                powerout = data[powerout_str]/oneDout
                powernet = data[powernet_str]/oneDnet
                axis.set_ylim([-50, 50])
            elif use_average:
                powerin = data[powerin_str]/data['Area [Re^2]']
                powerout = data[powerout_str]/data['Area [Re^2]']
                powernet = data[powernet_str]/data['Area [Re^2]']
            else:
                powerin = data[powerin_str]
                powerout = data[powerout_str]
                powernet = data[powernet_str]
                axis.set_ylim([-3e13, 3e13])
            #INJECTION
            if not override_color:
                Color = 'gold'
            axis.plot(data[timekey],powerin,
                            label=r'$\displaystyle K_{injection}$',
                        linewidth=Size, linestyle=ls,
                        color=Color)
            dtime = data[timekey][~ data[timekey].isin([np.inf,
                                                           -np.inf])]
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerin,
                                  color='wheat')
            #ESCAPE
            if not override_color:
                Color = 'deepskyblue'
            axis.plot(data[timekey],powerout,
                            label=r'$\displaystyle K_{escape}$',
                        linewidth=Size, linestyle=ls,
                        color=Color)
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey].values,powerout.values,
                              color='lightsteelblue')
            #NET
            if not use_shield:
                if not override_color:
                    Color = 'maroon'
                axis.plot(data[timekey],powernet,
                            label=r'$\displaystyle K_{net}$',
                            linewidth=Size, linestyle=ls,
                            color=Color)
                dtime = data[timekey][~ data[timekey].isin([np.inf,
                                                           -np.inf])]
                powernet = powernet[~ powernet.isin([np.inf,-np.inf])]
                if name.find('aggr')==-1:
                    axis.fill_between(data[timekey],powernet,
                                      color='coral')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_P0Power(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, use_average=False):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dflist:
        name = data['name'].iloc[-1]
        powerin_str = 'P0_injection [W]'
        powerout_str = 'P0_escape [W]'
        powernet_str = 'P0_net [W]'
        if use_inner:
            powerin_str = 'inner'+powerin_str
            powerout_str = 'inner'+powerout_str
            powernet_str = 'inner'+powernet_str
        if (name.find('mp')!=-1) or (name.find('aggr')!=-1):
            if use_shield:
                oneDin = abs(data['1D'+powerin_str])*100+1
                oneDout = abs(data['1D'+powerout_str])*100+1
                oneDnet = abs(data['1D'+powernet_str])*100+1
                powerin = data[powerin_str]/oneDin
                powerout = data[powerout_str]/oneDout
                powernet = data[powernet_str]/oneDnet
                axis.set_ylim([-50, 50])
            elif use_average:
                powerin = data[powerin_str]/data['Area [Re^2]']
                powerout = data[powerout_str]/data['Area [Re^2]']
                powernet = data[powernet_str]/data['Area [Re^2]']
            else:
                powerin = data[powerin_str]
                powerout = data[powerout_str]
                powernet = data[powernet_str]
                axis.set_ylim([-3e13, 3e13])
            #INJECTION
            axis.plot(data[timekey],powerin,
                            label=r'$\displaystyle P0_{injection}$',
                        linewidth=Size, linestyle=ls,
                        color='gold')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerin,
                              color='wheat')
            #ESCAPE
            axis.plot(data[timekey],powerout,
                            label=r'$\displaystyle P0_{escape}$',
                        linewidth=Size, linestyle=ls,
                        color='peru')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerout,
                                  color='peachpuff')
            #NET
            if not use_shield:
                axis.plot(data[timekey],powernet,
                            label=r'$\displaystyle P0_{net}$',
                            linewidth=Size, linestyle=ls,
                            color='maroon')
                if name.find('aggr')==-1:
                    axis.fill_between(data[timekey],powernet,
                                color='coral')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_ExBPower(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, use_average=False):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dflist:
        name = data['name'].iloc[-1]
        powerin_str = 'ExB_injection [W]'
        powerout_str = 'ExB_escape [W]'
        powernet_str = 'ExB_net [W]'
        if use_inner:
            powerin_str = 'inner'+powerin_str
            powerout_str = 'inner'+powerout_str
            powernet_str = 'inner'+powernet_str
        if (name.find('mp')!=-1) or (name.find('aggr')!=-1):
            if use_shield:
                oneDin = abs(data['1D'+powerin_str])*100+1
                oneDout = abs(data['1D'+powerout_str])*100+1
                oneDnet = abs(data['1D'+powernet_str])*100+1
                powerin = data[powerin_str]/oneDin
                powerout = data[powerout_str]/oneDout
                powernet = data[powernet_str]/oneDnet
                axis.set_ylim([-50, 50])
            elif use_average:
                powerin = data[powerin_str]/data['Area [Re^2]']
                powerout = data[powerout_str]/data['Area [Re^2]']
                powernet = data[powernet_str]/data['Area [Re^2]']
            else:
                powerin = data[powerin_str]
                powerout = data[powerout_str]
                powernet = data[powernet_str]
                axis.set_ylim([-3e13, 3e13])
            #INJECTION
            axis.plot(data[timekey],powerin,
                            label=r'$\displaystyle \left(E\times B\right)_{injection}$',
                        linewidth=Size, linestyle=ls,
                        color='mediumvioletred')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerin,
                                  color='palevioletred')
            #ESCAPE
            axis.plot(data[timekey],powerout,
                            label=r'$\displaystyle \left(E\times B\right)_{escape}$',
                        linewidth=Size, linestyle=ls,
                        color='deepskyblue')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerout,
                                  color='lightsteelblue')
            #NET
            if not use_shield:
                axis.plot(data[timekey],powernet,
                            label=r'$\displaystyle \left(E\times B\right)_{net}$',
                            linewidth=Size, linestyle=ls,
                            color='midnightblue')
                if name.find('aggr')==-1:
                    axis.fill_between(data[timekey],powernet,
                                    color='blue')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_PowerSpatDist(axis, dflist, timekey, ylabel, powerkey, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             allpositive=False):
    """Function plots spatatial distribution in terms of percents
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    powerin_str = powerkey+'_injection [W]'
    powerout_str = powerkey+'_escape [W]'
    powernet_str = powerkey+'_net [W]'
    if len(axis) == 1:
        injaxis = axis[0]
        escaxis = axis[0]
    else:
        injaxis = axis[0]
        escaxis = axis[1]
    if (dflist[0]['name'].iloc[-1] !=-1) and (len(dflist)>1):
        total_powerin = dflist[0][~ dflist[0][powerin_str].isna()][
                                                              powerin_str]
        total_powerout = dflist[0][~ dflist[0][powerin_str].isna()][
                                                             powerout_str]
        total_powernet = dflist[0][~ dflist[0][powerin_str].isna()][
                                                             powernet_str]
        for data in dflist[1::]:
            data = data[~ data[powerin_str].isna()]
            total_powerin = total_powerin+data[powerin_str]
            total_powerout = total_powerout+data[powerout_str]
            total_powernet = total_powernet+data[powernet_str]
        for data in dflist:
            name = data['name'].iloc[-1]
            data = data[~ data[powerin_str].isna()]
            #set linestyles
            if name.lower().find('day') != -1:
                marker = 'o'
                name = 'Day'
                KColors = dict({'inj':'gold','esc':'deepskyblue'})
                ExBColors = dict({'inj':'mediumvioletred',
                                  'esc':'deepskyblue'})
                P0Colors = dict({'inj':'gold','esc':'peru'})
            elif name.lower().find('flank') != -1:
                marker = 'v'
                name = 'Flank'
                KColors = dict({'inj':'wheat','esc':'lightsteelblue'})
                ExBColors = dict({'inj':'palevioletred',
                                  'esc':'lightsteelblue'})
                P0Colors = dict({'inj':'wheat','esc':'peachpuff'})
            elif name.lower().find('tail') != -1:
                marker = 'X'
                name = 'Tail'
                KColors = dict({'inj':'goldenrod','esc':'steelblue'})
                ExBColors = dict({'inj':'crimson',
                                  'esc':'steelblue'})
                P0Colors = dict({'inj':'goldenrod','esc':'saddlebrown'})
            Colordict = dict({'K':KColors,'ExB':ExBColors,'P0':P0Colors})
            powerout = data[powerout_str]/total_powerout
            powernet = data[powernet_str]/total_powernet
            if allpositive:
                powerin = data[powerin_str]/total_powerin
                injaxis.set_ylim([0, 1])
                escaxis.set_ylim([0, 1])
            else:
                powerin = -1*data[powerin_str]/total_powerin
                injaxis.set_ylim([-1, 1])
                escaxis.set_ylim([-1, 1])
            #INJECTION
            injaxis.plot(data[timekey],powerin,
                         label=r'$\displaystyle '+name+'_{injection \%}$',
                        linewidth=Size, marker=marker, markersize='14',
                        markevery=100,markerfacecolor='black',
                        color=Colordict[powerkey]['inj'])
            #axis.fill_between(data[timekey],powerin,
            #                  color='palevioletred')
            #ESCAPE
            escaxis.plot(data[timekey],powerout,
                         label=r'$\displaystyle '+name+'_{escape \%}$',
                        linewidth=Size, marker=marker, markersize='14',
                        markevery=100,markerfacecolor='black',
                        color=Colordict[powerkey]['esc'])
            #axis.fill_between(data[timekey],powerout,
            #                  color='lightsteelblue')
            '''
            #NET
            axis.plot(data[timekey],powernet,
                            label=r'$\displaystyle \left(E\times B\right)_{net \%}$',
                            linewidth=Size, marker=marker,
                            color='midnightblue')
                #axis.fill_between(data[timekey],powernet,
                #                color='blue')
            '''
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    injaxis.set_xlabel(r'\textit{Time (UTC)}')
    escaxis.set_xlabel(r'\textit{Time (UTC)}')
    injaxis.set_ylabel(ylabel)
    escaxis.set_ylabel(ylabel)
    injaxis.legend(loc=legend_loc, facecolor='gray')
    escaxis.legend(loc=legend_loc, facecolor='gray')

def plot_TotalEnergy(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        total = 'Total [J]'
        for qtkey in [total]:
            if name.find('mp')!=-1:
                axis.plot(data[timekey],-1*data[qtkey],
                            label=r'$\displaystyle \int_V{U_{total}}$',
                        linewidth=Size, linestyle=':',
                        color='black')
            else:
                axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='coral')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_VoverSA(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'V/SA [Re]'
    if Color == None:
        Color = 'magenta'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            tag = name.split('_')[-1]
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{'+tag+'}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_SA(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'Area [Re^2]'
    if Color == None:
        Color = 'magenta'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            tag = name.split('_')[-1]
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{'+tag+'}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_Standoff(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'X_subsolar [Re]'
    if Color == None:
        Color = 'magenta'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            tag = name.split('_')[-1]
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{'+tag+'}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_Volume(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'Volume [Re^3]'
    if Color == None:
        Color = 'magenta'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            tag = name.split('_')[-1]
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{'+tag+'}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_dst(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower left'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SMR (nT)'
            legend_loc = 'upper right'
            axis.plot(data[timekey],data[qtkey],
                      label='SMR',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
            axis.plot(data[timekey],data[qtkey],
                      label='SWMF Dst',
                      linewidth=Size, linestyle=ls,
                      color='magenta')
        elif name == 'omni':
            qtkey = 'sym_h'
            legend_loc = 'lower left'
            axis.plot(data[timekey],data[qtkey],
                      label='SYM-H',
                      linewidth=Size, linestyle=ls,
                      color='black')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_al(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower left'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SML (nT)'
            legend_loc = 'upper right'
            axis.plot(data[timekey],data[qtkey],
                      label='SML',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_index':
            qtkey = 'AL'
            axis.plot(data[timekey],data[qtkey],
                      label='AL swmf',
                      linewidth=Size, linestyle=ls,
                      color='magenta')
        elif name == 'omni':
            qtkey = 'al'
            legend_loc = 'lower left'
            axis.plot(data[timekey],data[qtkey],
                      label='AL OMNI',
                      linewidth=Size, linestyle=ls,
                      color='black')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_newell(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Newell CF (Wb/s)'
        else:
            qtkey = None
        if qtkey != None:
            #Values should be increased by x10! for now assuming
            #   supermag data is incorrect and including x10 factor
            axis.plot(data[timekey],data[qtkey]/100,
                      label='Newell Coupling Function',
                      linewidth=Size, linestyle=ls, color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_cpcp(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dflist:
        name = data['name'].iloc[-1]
        north = 'cpcpn'
        south = 'cpcps'
        for pole in [north,south]:
            if pole==south:
                axis.plot(data[timekey],data[pole],
                      label='South CPCP',
                      linewidth=Size,linestyle='--',color='magenta')
            else:
                axis.plot(data[timekey],data[pole],
                      label='North CPCP',
                      linewidth=Size, linestyle=ls, color='darkmagenta')
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=pole+'_'+name,
                      linewidth=Size, linestyle=ls, color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swdensity(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Density (#/cm^3)'
            axis.plot(data[timekey],data[qtkey],
                      label='SuperMAG (OMNI)',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            qtkey = 'dens'
            axis.plot(data[timekey],data[qtkey],
                      label='SWMF Input (WIND)',
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            qtkey = 'density'
            axis.plot(data[timekey],data[qtkey],
                      label='OMNI',
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_DesslerParkerSckopke(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for simdata in dflist:
        simname = simdata['name'].iloc[-1]
        if simname.find('mp')!=-1:
            total = simdata['Total [J]']
            uB = simdata['uB [J]']
            uE = simdata['uE [J]']
            uKperp = simdata['KEperp [J]']
            uKpar = simdata['KEpar [J]']
            axis.plot(simdata[timekey], (uB-uB[0]),
                      label=r'$\displaystyle \int_V{U_B-U_{B,0}}$',
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
            axis.plot(simdata[timekey], (total-total[0]),
                      label=r'$\displaystyle \int_V{U_{total}-U_{total,0}}$',
                      linewidth=Size, linestyle='--',
                      color='lightsteelblue')
            '''
            axis.plot(simdata[timekey], uE, label=simname+'_uE',
                      linewidth=Size, linestyle=ls,
                      color='violet')
            axis.plot(simdata[timekey], uKperp, label=simname+'_uKperp',
                      linewidth=Size, linestyle=ls,
                      color='green')
            axis.plot(simdata[timekey], uKpar, label=simname+'_uKpar',
                      linewidth=Size, linestyle='--',
                      color='green')
            '''
    for data in dflist:
        qtkey = None
        name = data['name'].iloc[-1]
        #Get DSP energy from magnetic perturbation
        if name == 'supermag':
            qtkey = 'SMR (nT)'
            Color = 'black'
            tag = 'SuperMAG'
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
            Color = 'gainsboro'
            tag = 'SWMF'
        elif name == 'omni':
            qtkey = 'sym_h'
            Color = 'coral'
            tag = 'OMNI'
        if qtkey != None:
            deltaB = data[qtkey]
            B_e = 31e3
            W_mag = 4*np.pi/3/(4*np.pi*1e-7)*B_e**2*(6371e3)**3 *(1e-18)
            Wtotal = -3*W_mag/B_e * deltaB
            axis.plot(data[timekey],Wtotal,
                    label=r'\textit{'+tag+'}  $\displaystyle W_{total}$',
                    linewidth=Size, linestyle=ls,
                    color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left', facecolor='gray')

def plot_akasofu(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            pass
        elif name == 'swmf_sw':
            bsquared_units = 1e-9**2 / (8*np.pi*1e-7)
            l = 7*6371*1000
            data = df_coord_transform(data, timekey, ['bx','by','bz'],
                                      ('GSE','car'), ('GSM','car'))
            data['v'] = 1000*np.sqrt(data['vx']**2+data['vy']**2+
                                     data['vz']**2)
            data['B^2'] = (data['bx']**2+data['by']**2+data['bz']**2)*(
                                                            bsquared_units)
            data['clock (GSE)'] = np.arctan2(data['by'],data['bz'])
            data['clock (GSM)'] = np.arctan2(data['byGSM'],data['bzGSM'])
            data['eps(GSE) [W]'] = (data['B^2']*data['v']*
                                    np.sin(data['clock (GSE)'])**4/2*l**2)
            data['eps(GSM) [W]'] = (data['B^2']*data['v']*
                                    np.sin(data['clock (GSM)']/2)**4*l**2)
            qtkey = 'eps(GSE) [W]'
            axis.plot(data[timekey],data[qtkey]*1000,
                      label=r'$\displaystyle \epsilon$',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'omni':
            qtkey = 'bz'
            axis.plot(data[timekey],data[qtkey],
                      label=name,
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper right', facecolor='gray')

def plot_pearson_r(axis, dflist, ydf, xlabel, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SMR (nT)'
            Color = 'black'
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
            Color = 'lightsteelblue'
        elif name == 'omni':
            qtkey = 'sym_h'
            Color = 'coral'
        ydata = np.interp(data['Time [UTC]'][0:-1],
                          ydf['Time [UTC]'][0:-1],ydf['Total [J]'][0:-1])
        xdata = data[qtkey][0:-1].values
        #normalized_x = (xdata-np.min(xdata))/(np.max(xdata-np.min(xdata)))
        cov = np.cov(np.stack((xdata,ydata)))[0][1]
        r = cov/(xdata.std()*ydata.std())
        axis.scatter(xdata, ydata, label='r = {:.2f}'.format(r),
                     color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper right')
    axis.grid()

def plot_swbz(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'BzGSM (nT)'
            axis.plot(data[timekey],data[qtkey],
                      label='SuperMAG (OMNI)',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            data = df_coord_transform(data, timekey, ['bx','by','bz'],
                                      ('GSE','car'), ('GSM','car'))
            qtkey = 'bzGSM'
            axis.plot(data[timekey],data[qtkey],
                      label='SWMF input (WIND)',
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            qtkey = 'bz'
            axis.plot(data[timekey],data[qtkey],
                      label='OMNI',
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.axhline(0, color='white')
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)

def plot_swflowP(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dflist:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'Dyn. Pres. (nPa)'
            axis.plot(data[timekey],data[qtkey],
                      label='SuperMAG (OMNI)',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            qtkey = 'Pdyn'
            axis.plot(data[timekey],data[qtkey],
                      label='SWMF input (WIND)',
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['Pdyn'] = data['density']*data['v']**2*convert
            qtkey = 'Pdyn'
            axis.plot(data[timekey],data[qtkey],
                      label='OMNI',
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)

def plot_swPower(axis, dflist, mp, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dflist- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dflist:
        name = data['name'].iloc[-1]
        powerin_str = '1DK_injection [W]'
        powerout_str = '1DK_escape [W]'
        powernet_str = '1DK_net [W]'
        if name.find('mp')!=-1:
            powerin = data[powerin_str]
            powerout = data[powerout_str]
            powernet = data[powernet_str]
            axis.set_ylim([-3e14, 3e14])
            #INJECTION
            axis.plot(data[timekey],powerin,
                        label=r'$\displaystyle K_{injection}$',
                        linewidth=Size, linestyle='--',
                        color='gold')
            #ESCAPE
            axis.plot(data[timekey],powerout,
                        label=r'$\displaystyle K_{escape}$',
                        linewidth=Size, linestyle='--',
                        color='deepskyblue')
            #NET
            axis.plot(data[timekey],powernet,
                        label=r'$\displaystyle K_{net}$',
                        linewidth=Size, linestyle='--',
                        color='maroon')
        elif name == 'supermag':
            pass
        elif name == 'swmf_sw':
            #constants/conversions
            pdyn_convert = 1.6726e-27*1e6*(1e3)**2*1e9
            pmag_convert = 1e-18/(8*np.pi*1e-7)
            total_convert = 1e-9*(6371*1e3)**3
            delta_t = 60
            kb = 1.3806503e-23
            #intermediate variables
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['B^2'] = data['bx']**2+data['by']**2+data['bz']**2
            #pressures in nPa
            data['Pdyn'] = 0.5*data['dens']*data['v']**2*pdyn_convert
            data['Pth'] = kb*data['temp']*data['dens']*1e9*1e6
            data['Pmag'] = data['B^2']*pmag_convert
            data['SW Ptot [nPa]']= (data['Pdyn']+data['Pth']+data['Pmag'])
            #derived power
            total_behind = (data['SW Ptot [nPa]']*total_convert*
                            mp['Volume [Re^3]'])
            total_forward = (data['SW Ptot [nPa]']*total_convert*
                            mp['Volume [Re^3]'])
            total_behind.index = total_behind.index-1
            total_forward.index = total_forward.index+1
            data['SW Power [W]']=(total_behind-total_forward)/(-2*delta_t)
            maxindex = (data[data['SW Power [W]']==
                        data['SW Power [W]'].max()].index)[0]
            minindex = (data[data['SW Power [W]']==
                        data['SW Power [W]'].min()].index)[0]
            plotdata = data['SW Power [W]'].drop(index=[maxindex,minindex])
            plottime = data['Time [UTC]'].drop(index=[maxindex,minindex])
            #energy and power
            qtkey = 'SW Power [W]'
            axis.plot(plottime,plotdata,
                      label=name,
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            pass
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time (UTC)}')
    axis.set_ylabel(ylabel)

def plot_fixed_loc(fig, ax, df,  powerkey, *,
                    contourlim=None, rlim=None, show_colorbar=False):
    """Function plots polar contour plot of power
    Inputs
        ax- axis for the plot
        df- dataframe containing data
    """
    #set variables
    alpha = df[~ df['alpha_rad'].isna()]['alpha_rad']
    power = df[~ df[powerkey].isna()][powerkey]
    timedata = df[~ df['Time_UTC'].isna()]['Time_UTC']
    timedata=pd.to_timedelta(timedata[0]-timedata).values.astype('float64')
    print(timedata)
    start = timedata[0]
    end = timedata[-1]
    #labels
    rectlabels = ['-Y',r'$\displaystyle III$','-Z',
                  r'$\displaystyle IV$','+Y',
                  r'$\displaystyle I$','+Z',
                  r'$\displaystyle II$', '-Y']
    #construct axes
    xi = linspace(start,end, 100)
    yi = linspace(-pi,pi,1441)
    zi = griddata((timedata,alpha),power,(xi[None,:],yi[:,None]),
                    method='linear')
    ax.set(xlim=(start, end))
    ax.set_title(str(df['x_cc'].mean()), pad=12)
    ax.set_yticks([-pi,-0.75*pi,-0.5*pi,-0.25*pi,0,
                    0.25*pi,0.5*pi,0.75*pi,pi])
    ax.set_yticklabels(rectlabels)
    twin = ax.twinx()
    twin.set_yticks([-pi,-0.75*pi,-0.5*pi,-0.25*pi,0,
                    0.25*pi,0.5*pi,0.75*pi,pi])
    twin.set_yticklabels(rectlabels)
    #plot contour
    if powerkey == 'K_net [W/Re^2]':
        colors = 'coolwarm_r'
        axes = fig.get_axes()[0:3]
    elif powerkey == 'ExB_net [W/Re^2]':
        colors = 'BrBG'
        axes = fig.get_axes()[3:6]
    elif powerkey == 'P0_net [W/Re^2]':
        colors = 'RdYlGn'
        axes = fig.get_axes()[6:9]
    cont_lvl = linspace(-3e9, 3e9, 11)
    #xi = xi.astype('timedelta64')+df['Time_UTC'].iloc[0]
    #ax.set(xlim=(df['Time_UTC'].iloc[0], df['Time_UTC'].iloc[-1]))
    cntr = ax.contourf(xi, yi, zi, cont_lvl, cmap=colors, extend='both')
    if show_colorbar:
        fig.colorbar(cntr, ax=axes, ticks=cont_lvl)

def chopends_time(dflist, start, end, timekey):
    """Function chops ends of dataframe based on time
    Inputs
        dflist
        start, end
        timekey
    Outputs
        dflist
    """
    newlist = []
    for df in enumerate(dflist):
        name = pd.Series({'name':df[1]['name'].iloc[-1]})
        cutdf = df[1][(df[1][timekey]>start)&
                      (df[1][timekey]<end)].append(
                            name,ignore_index=True)
        newlist.append(cutdf)
    return newlist

if __name__ == "__main__":
    datapath = sys.argv[1::]
    print('processing indices output at {}'.format(datapath))
    figureout = datapath[-1]+'figures/'
    #set text settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    #figure size defaults
    figx = 14; figy = 6; padsize=2
    #Read in data
    fullstart = dt.datetime(2014, 2, 15, 0)
    fullend = dt.datetime(2014, 2, 23, 0)
    energetics_list = read_energetics(datapath)
    mplist, fixedlist, agglist, dofixed, doagg = [], [], [], False, False
    for df in energetics_list:
        if df['name'].iloc[-1].find('mp_')!=-1:
            mplist.append(df)
        #if df['name'].iloc[-1].find('fixed')!=-1:
        #    fixedlist.append(df)
        #    dofixed = True
        if df['name'].iloc[-1].find('aggr')!=-1:
            agglist.append(df)
            doagg = True
            print('doagg = True!')
        if df['name'].iloc[-1].find('lcb')!=-1:
            lcb = df
    if len(mplist) > 1:
        do_mpcomparisons = True
        main_key = 'betastar'
        mp = mplist.pop(np.argmax(
                   [df.iloc[-1]['name'].find(main_key) for df in mplist]))
        #put for loop here to create 'remaining' category for each key that's not the time data
        remainin_inj = mp['K_injection [W]']-mplist[0]['K_injection [W]']-mplist[1]['K_injection [W]']-mplist[2]['K_injection [W]']
        mpremain = mp.copy()
        mpremain['K_injection [W]'] = remainin_inj
    else:
        do_mpcomparisons = False
        mp = mplist[0]
    [swmf_index, swmf_log, swmf_sw, supermag, omni]= read_indices(
                                                             datapath[-1])
    [supermag_expanded, omni_expanded] = get_expanded_sw(fullstart,
                                                    fullend, datapath[-1])
    #Chop based on time
    cuttoffstart = dt.datetime(2014,2,18,6,0)
    cuttoffend = dt.datetime(2014,2,20,0,0)
    #cuttoffstart = dt.datetime(2013,9,19,4,0)
    #cuttoffend = mp['Time [UTC]'].iloc[-2]
    datalist = [mp, swmf_index, swmf_log, swmf_sw, supermag, omni]
    [mp,swmf_index,swmf_log,swmf_sw,supermag,omni] = chopends_time(
                          datalist, cuttoffstart, cuttoffend, 'Time [UTC]')
    ##Plot data
    ######################################################################
    #Newell Function and Outerbound Net Power, and cpcp
    if True:
        figname = 'Newell_NetOutterPower'
        new_out_pow, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        new_out_pow.tight_layout(pad=1.1*padsize)
        #Time
        timekey = 'Time [UTC]'
        ylabel = r'\textit{Potential} $\displaystyle V \left(Wb/s \right)$'
        y1label = 'Coupling Function [Wb/s]'
        y2label = 'Power Injection [W]'
        y3label = 'Cross Polar Cap Potential [Wb/s]'
        #ax2 = ax1.twinx()
        plot_newell(ax1, [supermag], timekey, ylabel, ylim=[0,250])
        plot_cpcp(ax1, [swmf_log], timekey, ylabel, ylim=[0,250])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        new_out_pow.savefig(figureout+'{}.eps'.format(figname),
                            format='eps')
    ######################################################################
    #Total energy and dst index
    if True:
        figname = 'Energy_dst'
        energy_dst, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[0.8*figx,0.8*figy])
        energy_dst.tight_layout(rect=(0.03,0,0.95,1))
        #Time
        timekey = 'Time [UTC]'
        y1label = r'$\displaystyle Dst$ \textit{equiv.}$\displaystyle \left( nT \right)$'
        y2label = r'$\displaystyle E_{total} \left( J \right)$'
        ax2 = ax1.twinx()
        plot_dst(ax1, [swmf_log, omni, supermag], timekey, y1label)
        plot_TotalEnergy(ax2, [mp], timekey, y2label)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #new_out_pow.autofmt_xdate()
        ax2.legend(loc='lower right', facecolor='gray', fontsize=24)
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        energy_dst.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Correlation for energy and dst
    if True:
        figname = 'Energy_dst_rcor'
        energy_corr, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figy,figy])
        energy_corr.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        xlabel = r'$\displaystyle Dst$ \textit{equiv.}$\displaystyle \left( nT \right)$'
        ylabel = r'$\displaystyle E_{total} \left( J \right)$'
        plot_pearson_r(ax1, [supermag,swmf_log,omni] ,mp,
                       xlabel, ylabel)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #energy_corr.autofmt_xdate()
        #ax1.set_facecolor('olive')
        energy_corr.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Power and AL
    if True:
        figname = 'Power_al'
        power_al, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        power_al.tight_layout(pad=padsize*1.5)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'$\displaystyle AL$ \textit{equiv.}$\displaystyle \left( nT \right)$'
        y2label = r'$\displaystyle -\mid Power \left( W \right) \mid $'
        plot_al(ax1, [supermag, swmf_index, omni], timekey, y1label)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        power_al.savefig(figureout+'{}_justindex.eps'.format(figname))
        ax2 = ax1.twinx()
        power_al.tight_layout(pad=padsize*1.2)
        plot_Power_al(ax2, [mp], timekey, y2label)
        power_al.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Theoretical power from the solar wind
    if True:
        figname = 'swPower'
        swpower, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        swpower.tight_layout(pad=padsize*1.1)
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        #plot_swPower(ax1, [mp], mp,timekey, y1label)
        plot_Power(ax1, [mp], timekey, y1label)
        ax1.set_ylim([-12e13,12e13])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        swpower.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Total Power injection, escape and net
    if True:
        figname = 'Power'
        power, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        innerpower, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        power.tight_layout(pad=padsize*1.1)
        innerpower.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, [mp], timekey, y1label)
        plot_Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        power.savefig(figureout+'{}.eps'.format(figname))
        innerpower.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #ExB Power injection, escape and net
    if True:
        figname = 'ExBPower'
        exbpower, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        innerexbpower, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        exbpower.tight_layout(pad=padsize)
        innerexbpower.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_ExBPower(ax1, [mp], timekey, y1label)
        plot_ExBPower(in_ax1, [mp], timekey, y1label, use_inner=True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        exbpower.savefig(figureout+'{}.eps'.format(figname))
        innerexbpower.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #P0 Power injection, escape and net
    if True:
        figname = 'P0Power'
        p0power, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        innerP0power, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        p0power.tight_layout(pad=padsize)
        innerP0power.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_P0Power(ax1, [mp], timekey, y1label)
        plot_P0Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        p0power.savefig(figureout+'{}.eps'.format(figname))
        innerP0power.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #3panel Power, power_inner, and shielding
    if True:
        figname = '3panelPower'
        panel3power, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True, figsize=[figx,3*figy])
        in3panelpower,(in_ax1, in_ax2, in_ax3)=plt.subplots(
                                            nrows=3,ncols=1, sharex=True,
                                          figsize=[figx,3*figy])
        sh3panelpower,(sh_ax1, sh_ax2, sh_ax3)=plt.subplots(
                                            nrows=3,ncols=1, sharex=True,
                                          figsize=[figx,3*figy])
        panel3power.tight_layout(pad=padsize*1.1)
        in3panelpower.tight_layout(pad=padsize*1.1)
        sh3panelpower.tight_layout(pad=padsize*1.3)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        y2label = r'\textit{Transfer Efficiency}$\displaystyle \left( \% \right)$'
        plot_Power(ax1, [mp], timekey, y1label)
        plot_Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        #plot_Power(sh_ax1, [mp], timekey, y2label, use_shield=True)
        plot_P0Power(ax3, [mp], timekey, y1label)
        plot_P0Power(in_ax3, [mp], timekey, y1label, use_inner=True)
        #plot_P0Power(sh_ax3, [mp], timekey, y2label, use_shield=True)
        plot_ExBPower(ax2, [mp], timekey, y1label)
        plot_ExBPower(in_ax2, [mp], timekey, y1label, use_inner=True)
        #plot_ExBPower(sh_ax2, [mp], timekey, y2label, use_shield=True, ylim=[-350,350])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))

        #shade_plot(ax1), shade_plot(in_ax1), shade_plot(sh_ax1),
        #shade_plot(ax2), shade_plot(in_ax2), shade_plot(sh_ax2),
        #shade_plot(ax3), shade_plot(in_ax3), shade_plot(sh_ax3)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        #ax2.set_facecolor('olive'), in_ax2.set_facecolor('olive')
        #ax3.set_facecolor('olive'), in_ax3.set_facecolor('olive')
        #sh_ax1.set_facecolor('olive'), sh_ax2.set_facecolor('olive')
        #sh_ax3.set_facecolor('olive')
        panel3power.savefig(figureout+'{}.eps'.format(figname))
        in3panelpower.savefig(figureout+'{}_inner.eps'.format(figname))
        sh3panelpower.savefig(figureout+'{}_shield.eps'.format(figname))
    ######################################################################
    #Solarwind, regular sized
    if True:
        figname = 'SolarWind'
        solarwind, (ax2,ax3)= plt.subplots(nrows=2,ncols=1,sharex=True,
                                                     figsize=[0.8*figx,0.8*figy])
        solarwind.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Density} $\displaystyle \left(\#/cm^3\right)$'
        y2label = r'\textit{IMF} $\displaystyle B_z \left(nT\right)$'
        y3label = r'\textit{Flow Pressure}$\displaystyle \left(nPa\right)$'
        #plot_swdensity(ax1, [swmf_sw], timekey, y1label)
        plot_swbz(ax2, [swmf_sw], timekey, y2label)
        plot_swflowP(ax3, [swmf_sw], timekey, y3label)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #shade_plot(ax1); shade_plot(ax2); shade_plot(ax3)
        #ax1.set_facecolor('olive')
        #ax2.set_facecolor('olive')
        #ax3.set_facecolor('olive')
        solarwind.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Solarwind, expanded
    if True:
        figname = 'ExpandedSolarWind'
        bigsw, (ax1,ax2,ax3)= plt.subplots(nrows=3,ncols=1,sharex=True,
                                                     figsize=[figx*1.5,2*figy])
        bigsw.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Density} $\displaystyle \left(\#/cm^3\right)$'
        y2label = r'\textit{IMF} $\displaystyle B_z \left(nT\right)$'
        y3label = r'\textit{Flow Pressure}$\displaystyle \left(nPa\right)$'
        plot_swdensity(ax1, [supermag_expanded, omni_expanded], timekey,
                                                                   y1label)
        plot_swbz(ax2,[supermag_expanded, omni_expanded], timekey, y2label)
        plot_swflowP(ax3, [supermag_expanded, omni_expanded], timekey,
                                                                   y3label)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #ax1.set_facecolor('olive')
        #ax2.set_facecolor('olive')
        #ax3.set_facecolor('olive')
        bigsw.savefig(figureout+'{}_noshade.eps'.format(figname))
        #shade_plot(ax1, do_full=True); shade_plot(ax2, do_full=True)
        #shade_plot(ax3, do_full=True)
        bigsw.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #2panel Akosofu, Newell
    if True:
        figname = '2panelProxies'
        panel2prox, (ax1,ax2)=plt.subplots(nrows=2, ncols=1,
                                          sharex=True, figsize=[figx,2*figy])
        panel2prox.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        y2label = r'\textit{Potential}$\displaystyle \left(kV\right)=\left(kWb/s\right)$'
        #twin1 = ax1.twinx(); twiny1label = 'Epsilon [W]'
        #twin2 = ax2.twinx(); twiny2label = 'Coupling Function [Wb/s]=[J/A]'
        plot_akasofu(ax1, [swmf_sw], timekey, y1label)
        plot_Power_al(ax1, [mp], timekey, y1label, multiplier=1)
        ax1.legend(loc='upper right', facecolor='gray')
        ax1.text(0.01,0.9,'a)', Color='black', fontsize=36, transform=ax1.transAxes)
        plot_newell(ax2, [supermag], timekey, y2label)
        plot_cpcp(ax2, [swmf_log], timekey, y2label)
        ax2.legend(loc='upper right', facecolor='gray')
        ax2.text(0.01,0.9,'b)', Color='black', fontsize=36, transform=ax2.transAxes)
        #shade_plot(ax1); shade_plot(ax2); shade_plot(ax3)
        #ax1.set_facecolor('olive'); ax2.set_facecolor('olive')
        panel2prox.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Dessler-Parker-Sckopke
    if True:
        figname = 'DesslerParkerSckopke'
        DPS, (ax1)=plt.subplots(nrows=1, ncols=1,
                                          sharex=True, figsize=[figx,figy])
        DPS.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        ylabel = r'\textit{Energy} $\displaystyle \left(J\right)$'
        plot_DesslerParkerSckopke(ax1, [swmf_log, supermag, omni, mp],
                                  timekey, ylabel)
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        DPS.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces power
    if do_mpcomparisons:
        figname = 'ComparativePower'
        power_comp, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        power_comp.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        #plot_Power(ax1, [mp], timekey, y1label, Color='midnightblue')
        plot_Power(ax1, [mplist[0]], timekey, y1label, Color='coral')
        plot_Power(ax1, [mplist[1]], timekey, y1label, Color='gold')
        plot_Power(ax1, [mplist[2]], timekey, y1label, Color='plum')
        #plot_Power(ax1, [difference_mp], timekey, y1label, Color='midnightblue')
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        power_comp.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces volumes
    if do_mpcomparisons:
        figname = 'ComparativeVolume'
        volume_comp, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        volume_comp.tight_layout(pad=padsize*2)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Volume} $\displaystyle \left( R_e^3\right)$'
        plot_Volume(ax1, [mp], timekey, y1label, Color='midnightblue')
        plot_Volume(ax1, [mplist[0]], timekey, y1label, Color='coral')
        #plot_Volume(ax1, [mplist[1]], timekey, y1label, Color='gold')
        #plot_Volume(ax1, [mplist[2]], timekey, y1label, Color='plum')
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        volume_comp.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces volumes
    if do_mpcomparisons:
        figname = 'ComparativeSurfaceArea'
        surf_comp, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        surf_comp.tight_layout(pad=padsize*2)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Area} $\displaystyle \left( R_e^2\right)$'
        plot_SA(ax1, [mp], timekey, y1label, Color='midnightblue')
        plot_SA(ax1, [mplist[0]], timekey, y1label, Color='coral')
        #plot_SA(ax1, [mplist[1]], timekey, y1label, Color='gold')
        #plot_SA(ax1, [mplist[2]], timekey, y1label, Color='plum')
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        surf_comp.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'Flank3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        y2label = r'\textit{Transfer Efficiency}$\displaystyle \left( \% \right)$'
        plot_Power(ax1, agglist[1:2], timekey, y1label)
        plot_P0Power(ax2, agglist[1:2], timekey, y1label)
        plot_ExBPower(ax3, agglist[1:2], timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'Tail3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, agglist[2:3], timekey, y1label)
        plot_P0Power(ax2, agglist[2:3], timekey, y1label)
        plot_ExBPower(ax3, agglist[2:3], timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'AverageDay3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, agglist[0:1], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, agglist[0:1], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, agglist[0:1], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'AverageFlank3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, agglist[1:2], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, agglist[1:2], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, agglist[1:2], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'AverageTail3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left( W\right)$'
        y2label = r'\textit{Transfer Efficiency}$\displaystyle \left( \% \right)$'
        plot_Power(ax1, agglist[2:3], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, agglist[2:3], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, agglist[2:3], timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'Day3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, agglist[0:1], timekey, y1label)
        plot_P0Power(ax2, agglist[0:1], timekey, y1label)
        plot_ExBPower(ax3, agglist[0:1], timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'DayFlankTail3panel'
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[0.8*figx,1.6*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left( W\right)$'
        plot_Power(ax1, agglist, timekey, y1label)
        plot_P0Power(ax2, agglist, timekey, y1label)
        plot_ExBPower(ax3, agglist, timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'Spatial_Powers3panel'
        spPower3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        spPower3.tight_layout(pad=padsize*1.3)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power \%} $\displaystyle \left( W\right)$'
        y2label = r'\textit{ExB Power \%} $\displaystyle \left( W\right)$'
        y3label = r'\textit{P0 Power \%} $\displaystyle \left( W\right)$'
        plot_PowerSpatDist([ax1], agglist, timekey, y1label, 'K')
        plot_PowerSpatDist([ax2], agglist, timekey, y2label, 'P0')
        plot_PowerSpatDist([ax3], agglist, timekey, y3label, 'ExB')

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3),
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive'),
        spPower3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if doagg:
        figname = 'Spatial_Powers'
        spPower1, (ax1, ax2)=plt.subplots(nrows=2, ncols=1,
                                          sharex=True,
                                          figsize=[figx,2*figy])
        spPower2, (ax3, ax4)=plt.subplots(nrows=2, ncols=1,
                                          sharex=True,
                                          figsize=[figx,2*figy])
        spPower3, (ax5, ax6)=plt.subplots(nrows=2, ncols=1,
                                          sharex=True,
                                          figsize=[figx,2*figy])
        spPower1.tight_layout(pad=padsize*1.3)
        spPower2.tight_layout(pad=padsize*1.3)
        spPower3.tight_layout(pad=padsize*1.3)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power \%} $\displaystyle \left( W\right)$'
        y2label = r'\textit{P0 Power \%} $\displaystyle \left( W\right)$'
        y3label = r'\textit{ExB Power \%} $\displaystyle \left( W\right)$'
        plot_PowerSpatDist([ax1,ax2], agglist, timekey, y1label, 'K',
                           allpositive=True)
        plot_PowerSpatDist([ax3,ax4], agglist, timekey, y2label, 'P0',
                           allpositive=True)
        plot_PowerSpatDist([ax5,ax6], agglist, timekey, y3label, 'ExB',
                           allpositive=True)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')
        ax4.legend(loc='upper left', facecolor='gray')
        ax5.legend(loc='upper left', facecolor='gray')
        ax6.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1),
        #shade_plot(ax2),
        #shade_plot(ax3),
        #shade_plot(ax4),
        #shade_plot(ax5),
        #shade_plot(ax6),
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive'),
        #ax4.set_facecolor('olive'), ax5.set_facecolor('olive'),
        #ax6.set_facecolor('olive'),
        spPower1.savefig(figureout+'{}.eps'.format(figname+'_K'))
        spPower2.savefig(figureout+'{}.eps'.format(figname+'_P0'))
        spPower3.savefig(figureout+'{}.eps'.format(figname+'_ExB'))
    ######################################################################
    #timeseries of data at a set of fixed points on the surface
    if dofixed:
        figname = 'fixed_spatial_timeseries'
        fixed = plt.figure(figsize=[16,16])
        #toprow
        ax1 = fixed.add_subplot(331,projection='rectilinear')
        ax2 = fixed.add_subplot(332,projection='rectilinear')
        ax3 = fixed.add_subplot(333,projection='rectilinear')
        #middlerow
        ax4 = fixed.add_subplot(334,projection='rectilinear')
        ax5 = fixed.add_subplot(335,projection='rectilinear')
        ax6 = fixed.add_subplot(336,projection='rectilinear')
        #bottomrow
        ax7 = fixed.add_subplot(337,projection='rectilinear')
        ax8 = fixed.add_subplot(338,projection='rectilinear')
        ax9 = fixed.add_subplot(339,projection='rectilinear')
        fixed.tight_layout(pad=2)
        #total
        plot_fixed_loc(fixed, ax1, fixedlist[0], 'K_net [W/Re^2]')
        plot_fixed_loc(fixed, ax2, fixedlist[1], 'K_net [W/Re^2]')
        plot_fixed_loc(fixed, ax3, fixedlist[2], 'K_net [W/Re^2]',
                                              show_colorbar=True)
        #ExB
        plot_fixed_loc(fixed, ax4, fixedlist[0], 'ExB_net [W/Re^2]')
        plot_fixed_loc(fixed, ax5, fixedlist[1], 'ExB_net [W/Re^2]')
        plot_fixed_loc(fixed, ax6, fixedlist[2], 'ExB_net [W/Re^2]',
                                                show_colorbar=True)
        #P0
        plot_fixed_loc(fixed, ax7, fixedlist[0], 'P0_net [W/Re^2]')
        plot_fixed_loc(fixed, ax8, fixedlist[1], 'P0_net [W/Re^2]')
        plot_fixed_loc(fixed, ax9, fixedlist[2], 'P0_net [W/Re^2]',
                                                show_colorbar=True)
        fixed.savefig(figureout+'{}'.format(figname)+'.eps')
        plt.cla()
    ######################################################################
