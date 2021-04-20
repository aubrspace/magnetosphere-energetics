#!/usr/bin/env python3
"""module calls on processing scripts to gather data then creates plots
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
             xlim=None, ylim=None, multiplier=-1, Size=4, ls=None):
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
                            label=powerin.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='gold')
            #ESCAPE
            axis.plot(data[timekey],multiplier*abs(data[powerout]),
                            label=powerout.split(' ')[0],
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
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_Power(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,
             use_inner=False):
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
        powerin = 'K_injection [W]'
        powerout = 'K_escape [W]'
        powernet = 'K_net [W]'
        if use_inner:
            powerin = 'inner'+powerin
            powerout = 'inner'+powerout
            powernet = 'inner'+powernet
        if name.find('mp')!=-1:
            #INJECTION
            axis.plot(data[timekey],data[powerin],
                            label=powerin.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='gold')
            axis.fill_between(data[timekey],data[powerin],
                              color='wheat')
            #ESCAPE
            axis.plot(data[timekey],data[powerout],
                            label=powerout.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='deepskyblue')
            axis.fill_between(data[timekey],data[powerout],
                              color='lightsteelblue')
            #NET
            axis.plot(data[timekey],data[powernet],
                            label=powernet.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='maroon')
            axis.fill_between(data[timekey],data[powernet],
                              color='coral')
        else:
            axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='plum')
    axis.set_ylim([-3e13, 3e13])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_P0Power(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,
             use_inner=False):
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
        powerin = 'P0_injection [W]'
        powerout = 'P0_escape [W]'
        powernet = 'P0_net [W]'
        if use_inner:
            powerin = 'inner'+powerin
            powerout = 'inner'+powerout
            powernet = 'inner'+powernet
        if name.find('mp')!=-1:
            #INJECTION
            axis.plot(data[timekey],data[powerin],
                            label=powerin.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='gold')
            axis.fill_between(data[timekey],data[powerin],
                              color='wheat')
            #ESCAPE
            axis.plot(data[timekey],data[powerout],
                            label=powerout.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='peru')
            axis.fill_between(data[timekey],data[powerout],
                              color='peachpuff')
            #NET
            axis.plot(data[timekey],data[powernet],
                            label=powernet.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='maroon')
            axis.fill_between(data[timekey],data[powernet],
                              color='coral')
        else:
            axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='plum')
    axis.set_ylim([-3e13, 3e13])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_ExBPower(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,
             use_inner=False):
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
        powerin = 'ExB_injection [W]'
        powerout = 'ExB_escape [W]'
        powernet = 'ExB_net [W]'
        if use_inner:
            powerin = 'inner'+powerin
            powerout = 'inner'+powerout
            powernet = 'inner'+powernet
        if name.find('mp')!=-1:
            #INJECTION
            axis.plot(data[timekey],data[powerin],
                            label=powerin.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='mediumvioletred')
            axis.fill_between(data[timekey],data[powerin],
                              color='palevioletred')
            #ESCAPE
            axis.plot(data[timekey],data[powerout],
                            label=powerout.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='deepskyblue')
            axis.fill_between(data[timekey],data[powerout],
                              color='lightsteelblue')
            #NET
            axis.plot(data[timekey],data[powernet],
                            label=powernet.split(' ')[0],
                        linewidth=Size, linestyle=ls,
                        color='midnightblue')
            axis.fill_between(data[timekey],data[powernet],
                              color='blue')
        else:
            axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='plum')
    axis.set_ylim([-3e13, 3e13])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_OuterPower(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots outer surface boundary powers
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
        powerin = 'K_injection [W]'
        powerout = 'K_net [W]'
        powernet = 'K_escape [W]'
        for qtkey in [powerin]:
            if name.find('mp')!=-1:
                axis.plot(data[timekey],-1*data[qtkey],
                            label='magnetopause',
                        linewidth=Size, linestyle=ls,
                        color='lightsteelblue')
            else:
                axis.plot(data[timekey],-1*data[qtkey],
                            label='lcb surface',
                        linewidth=Size, linestyle=ls,
                        color='violet')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_TotalEnergy(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
                            label=name,
                        linewidth=Size, linestyle=':',
                        color='lightsteelblue')
            else:
                axis.plot(data[timekey],-1*data[qtkey],
                            label=name,
                        linewidth=Size, linestyle=ls,
                        color='coral')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_dst(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls,
                      color='black')
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            qtkey = 'sym_h'
            legend_loc = 'lower left'
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_al(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_index':
            qtkey = 'AL'
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            qtkey = 'al'
            legend_loc = 'lower left'
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_newell(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots cross polar cap potential with given data frames
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
        if name == 'supermag':
            qtkey = 'Newell CF (Wb/s)'
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=qtkey+'_'+name,
                      linewidth=Size, linestyle='--', color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_cpcp(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots cross polar cap potential with given data frames
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
        north = 'cpcpn'
        south = 'cpcps'
        for pole in [north,south]:
            if pole==south:
                axis.plot(data[timekey],data[pole]*100,
                      label=pole+'_'+name,
                      linewidth=Size,linestyle='--',color='deepskyblue')
            else:
                axis.plot(data[timekey],data[pole]*100,
                      label=pole+'_'+name,
                      linewidth=Size, linestyle='--', color='gold')
        else:
            qtkey = None
        if qtkey != None:
            axis.plot(data[timekey],data[qtkey],
                      label=pole+'_'+name,
                      linewidth=Size, linestyle='--', color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc, facecolor='gray')

def plot_swdensity(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
                      label=name,
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
                      label=name,
                      linewidth=Size, linestyle=ls,
                      color='coral')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_DesslerParkerSckopke(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
            axis.plot(simdata[timekey], (uB-uB[0]), label=simname+'_duB',
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
            axis.plot(simdata[timekey], (total-total[0]), label=simname+'_dtotal',
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
        elif name == 'swmf_log':
            qtkey = 'dst_sm'
            Color = 'gainsboro'
        elif name == 'omni':
            qtkey = 'sym_h'
            Color = 'coral'
        if qtkey != None:
            deltaB = data[qtkey]
            B_e = 31e3
            W_mag = 4*np.pi/3/(4*np.pi*1e-7)*B_e**2*(6371e3)**3 *(1e-18)
            Wtotal = -3*W_mag/B_e * deltaB
            axis.plot(data[timekey],Wtotal,label=name+'_Wtotal',
                    linewidth=Size, linestyle=ls,
                    color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left', facecolor='gray')

def plot_akasofu(axis, dflist, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
            axis.plot(data[timekey],data[qtkey],
                      label=name,
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
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper right', facecolor='gray')

def plot_pearson_r(axis, dflist, ydf, xlabel, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
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
                      label=name,
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            data = df_coord_transform(data, timekey, ['bx','by','bz'],
                                      ('GSE','car'), ('GSM','car'))
            qtkey = 'bzGSM'
            axis.plot(data[timekey],data[qtkey],
                      label=name,
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
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
    axis.axhline(0, color='white')
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)

def plot_swflowP(axis, dflist, timekey, ylabel, *,
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
            qtkey = 'Dyn. Pres. (nPa)'
            axis.plot(data[timekey],data[qtkey],
                      label=name,
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            qtkey = 'Pdyn'
            axis.plot(data[timekey],data[qtkey],
                      label=name,
                      linewidth=Size, linestyle=ls,
                      color='lightsteelblue')
        elif name == 'omni':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['Pdyn'] = data['density']*data['v']**2*convert
            qtkey = 'Pdyn'
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
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)

def plot_swPower(axis, dflist, mp, timekey, ylabel, *,
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
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(timekey)
    axis.set_ylabel(ylabel)

def chopends_time(dflist, start, end, timekey):
    """Function chops ends of dataframe based on time
    Inputs
        dflist
        start, end
        timekey
    Outputs
        dflist
    """
    for df in enumerate(dflist):
        name = pd.Series({'name':df[1]['name'].iloc[-1]})
        cutdf = df[1][(df[1][timekey]>start)&(df[1][timekey]<end)].append(
                                                    name,ignore_index=True)
    return dflist

if __name__ == "__main__":
    datapath = sys.argv[1]
    print('processing indices output at {}'.format(datapath))
    figureout = datapath+'figures/'
    #set text settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    #Read in data
    fullstart = dt.datetime(2014, 2, 15, 0)
    fullend = dt.datetime(2014, 2, 23, 0)
    energetics_list = read_energetics([datapath])
    for df in energetics_list:
        if df['name'].iloc[-1].find('mp_')!=-1:
            mp = df
        if df['name'].iloc[-1].find('lcb')!=-1:
            lcb = df
    [swmf_index, swmf_log, swmf_sw, supermag, omni]= read_indices(datapath)
    [supermag_expanded, omni_expanded] = get_expanded_sw(fullstart,
                                                         fullend, datapath)
    #Chop based on time
    cuttoffstart = dt.datetime(2014,2,18,7,0)
    #cuttoffend = dt.datetime(2014,2,18,12,0)
    cuttoffend = mp['Time [UTC]'].iloc[-2]
    datalist = [mp, swmf_index, swmf_log, swmf_sw, supermag, omni]
    [mp,swmf_index,swmf_log,swmf_sw,supermag,omni] = chopends_time(
                          datalist, cuttoffstart, cuttoffend, 'Time [UTC]')
    ##Plot data
    ######################################################################
    #Newell Function and Outerbound Net Power, and cpcp
    if True:
        figname = 'Newell_NetOutterPower'
        new_out_pow, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Coupling Function [Wb/s]'
        y2label = 'Power Injection [W]'
        y3label = 'Cross Polar Cap Potential [Wb/s]'
        ax2 = ax1.twinx()
        plot_newell(ax2, [supermag], timekey, y1label)
        plot_OuterPower(ax1, [mp], timekey, y2label)
        plot_cpcp(ax2, [swmf_log], timekey, y3label)
        shade_plot(ax1)
        ax1.set_facecolor('olive')
        new_out_pow.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Total energy and dst index
    if True:
        figname = 'Energy_dst'
        energy_dst, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Dst equiv. [nT]'
        y2label = '-1*Total Energy [J]'
        ax2 = ax1.twinx()
        plot_dst(ax1, [supermag, swmf_log, omni], timekey, y1label)
        plot_TotalEnergy(ax2, [mp], timekey, y2label)
        shade_plot(ax1)
        ax1.set_facecolor('olive')
        energy_dst.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Correlation for energy and dst
    if True:
        figname = 'Energy_dst_rcor'
        energy_corr, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[8,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        xlabel = 'Dst equiv [nT]'
        ylabel = 'Total Energy [J]'
        plot_pearson_r(ax1, [supermag,swmf_log,omni] ,mp,
                       xlabel, ylabel)
        ax1.set_facecolor('olive')
        energy_corr.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Power and AL
    if True:
        figname = 'Power_al'
        power_al, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'AL equiv. [nT]'
        y2label = '-1*|Power [W]|'
        ax2 = ax1.twinx()
        plot_al(ax1, [supermag, swmf_index, omni], timekey, y1label)
        shade_plot(ax1)
        ax1.set_facecolor('olive')
        power_al.savefig(figureout+'{}_justindex.png'.format(figname),
                      facecolor='gainsboro')
        plot_Power_al(ax2, [mp], timekey, y2label)
        power_al.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Theoretical power from the solar wind
    if True:
        figname = 'swPower'
        swpower, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        plot_swPower(ax1, [swmf_sw], mp,timekey, y1label)
        plot_Power(ax1, [mp], timekey, y1label)
        ax1.set_ylim([-12e13,12e13])
        shade_plot(ax1)
        ax1.set_facecolor('olive')
        swpower.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Total Power injection, escape and net
    if True:
        figname = 'Power'
        power, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        innerpower, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        plot_Power(ax1, [mp], timekey, y1label)
        plot_Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        shade_plot(ax1), shade_plot(in_ax1)
        ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        power.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
        innerpower.savefig(figureout+'{}_inner.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #ExB Power injection, escape and net
    if True:
        figname = 'ExBPower'
        exbpower, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        innerexbpower, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        plot_ExBPower(ax1, [mp], timekey, y1label)
        plot_ExBPower(in_ax1, [mp], timekey, y1label, use_inner=True)
        shade_plot(ax1), shade_plot(in_ax1)
        ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        exbpower.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
        innerexbpower.savefig(figureout+'{}_inner.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #P0 Power injection, escape and net
    if True:
        figname = 'P0Power'
        p0power, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        innerP0power, (in_ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        plot_P0Power(ax1, [mp], timekey, y1label)
        plot_P0Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        shade_plot(ax1), shade_plot(in_ax1)
        ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        p0power.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
        innerP0power.savefig(figureout+'{}_inner.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #3panel Power and power_inner
    if True:
        figname = '3panelPower'
        panel3power, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True, figsize=[18,24],
                                          facecolor='gainsboro')
        in3panelpower,(in_ax1, in_ax2, in_ax3)=plt.subplots(
                                            nrows=3,ncols=1, sharex=True,
                                          figsize=[18,24],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        plot_Power(ax1, [mp], timekey, y1label)
        plot_Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        plot_ExBPower(ax2, [mp], timekey, y1label)
        plot_ExBPower(in_ax2, [mp], timekey, y1label, use_inner=True)
        plot_P0Power(ax3, [mp], timekey, y1label)
        plot_P0Power(in_ax3, [mp], timekey, y1label, use_inner=True)
        shade_plot(ax1), shade_plot(in_ax1)
        shade_plot(ax2), shade_plot(in_ax2)
        shade_plot(ax3), shade_plot(in_ax3)
        ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        ax2.set_facecolor('olive'), in_ax2.set_facecolor('olive')
        ax3.set_facecolor('olive'), in_ax3.set_facecolor('olive')
        panel3power.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
        in3panelpower.savefig(figureout+'{}_inner.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Solarwind, regular sized
    if True:
        figname = 'SolarWind'
        solarwind, (ax1,ax2,ax3)= plt.subplots(nrows=3,ncols=1,sharex=True,
                                                     figsize=[18,8],
                                                     facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Density [#/cm^3]'
        y2label = 'IMF Bz [nT]'
        y3label = 'Flow Pressure [nPa]'
        plot_swdensity(ax1, [supermag, swmf_sw], timekey, y1label)
        plot_swbz(ax2, [supermag, swmf_sw], timekey, y2label)
        plot_swflowP(ax3, [supermag, swmf_sw], timekey, y3label)
        shade_plot(ax1); shade_plot(ax2); shade_plot(ax3)
        ax1.set_facecolor('olive')
        ax2.set_facecolor('olive')
        ax3.set_facecolor('olive')
        solarwind.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Solarwind, expanded
    if True:
        figname = 'ExpandedSolarWind'
        bigsw, (ax1,ax2,ax3)= plt.subplots(nrows=3,ncols=1,sharex=True,
                                                     figsize=[24,8],
                                                     facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Density [#/cm^3]'
        y2label = 'IMF Bz [nT]'
        y3label = 'Flow Pressure [nPa]'
        plot_swdensity(ax1, [supermag_expanded, omni_expanded], timekey,
                                                                   y1label)
        plot_swbz(ax2,[supermag_expanded, omni_expanded], timekey, y2label)
        plot_swflowP(ax3, [supermag_expanded, omni_expanded], timekey,
                                                                   y3label)
        ax1.set_facecolor('olive')
        ax2.set_facecolor('olive')
        ax3.set_facecolor('olive')
        bigsw.savefig(figureout+'{}_noshade.png'.format(figname),
                      facecolor='gainsboro')
        shade_plot(ax1, do_full=True); shade_plot(ax2, do_full=True)
        shade_plot(ax3, do_full=True)
        bigsw.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #3panel Newell, Akosofu and ???
    if True:
        figname = '3panelProxies'
        panel3prox, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True, figsize=[18,24],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        y1label = 'Power [W]'
        twin1 = ax1.twinx(); twiny1label = 'Coupling Function [Wb/s]'
        twin2 = ax2.twinx(); twiny2label = 'Epsilon (Akasofu l_0=7) [W]'
        plot_newell(twin1, [supermag], timekey, twiny1label)
        plot_Power_al(ax1, [mp], timekey, y1label, multiplier=1)
        ax1.legend(loc='upper left')
        plot_akasofu(twin2, [swmf_sw], timekey, twiny2label)
        plot_Power_al(ax2, [mp], timekey, y1label, multiplier=1)
        ax2.legend(loc='upper left')
        shade_plot(ax1); shade_plot(ax2); shade_plot(ax3)
        ax1.set_facecolor('olive'); ax2.set_facecolor('olive')
        ax3.set_facecolor('olive')
        panel3prox.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    #Dessler-Parker-Sckopke
    if True:
        figname = 'DesslerParkerSckopke'
        DPS, (ax1)=plt.subplots(nrows=1, ncols=1,
                                          sharex=True, figsize=[18,8],
                                          facecolor='gainsboro')
        #Time
        timekey = 'Time [UTC]'
        ylabel = 'Energy [J]'
        plot_DesslerParkerSckopke(ax1, [swmf_log, supermag, omni, mp],
                                  timekey, ylabel)
        shade_plot(ax1)
        ax1.set_facecolor('olive')
        DPS.savefig(figureout+'{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
