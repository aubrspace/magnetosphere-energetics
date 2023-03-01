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
from scipy.integrate import quad
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#from labellines import labelLines
import swmfpy
#interpackage imports
from global_energetics.extract.shue import (r0_alpha_1998, r_shue)
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup,safelabel,
                                                   get_omni_cdas,
                                                   mark_times, shade_plot)
from global_energetics.analysis.proc_hdf import(group_subzones,
                                                load_hdf_sort)
from global_energetics.analysis.proc_energy_timeseries import(process_energy)
from global_energetics.analysis.proc_indices import (read_indices,
                                                     get_expanded_sw)


def plot_Power_al(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, multiplier=-1, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dfdict:
        name = data['name'].iloc[-1]
        powerin = 'K_injection [W]'
        powerout = 'K_escape [W]'
        powernet = 'K_net [W]'
        print(name)
        if (name.find('mp')!=-1) or (name.find('agg')!=-1):
            #INJECTION
            axis.plot(data[timekey],abs(data[powerin])/1e12,
                            label=r'\textit{Injection}',
                        linewidth=Size, linestyle=ls,
                        color='darkmagenta')
            #ESCAPE
            axis.plot(data[timekey],abs(data[powerout])/1e12,
                            label=r'\textit{Escape}',
                        linewidth=Size, linestyle=ls,
                        color='magenta')
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
    axis.legend(loc=legend_loc)

def plot_power(axis, dfdict, times, **kwargs):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    #Update default legend_loc
    kwargs.update({'legend_loc':kwargs.get('legend_loc','upper right')})
    #helper dictionaries
    keydict = {} #load piece by piece allowing kwargs to get a say
    keydict.update({'inj':kwargs.get('inj','K_injection [W]')})
    keydict.update({'esc':kwargs.get('esc','K_escape [W]')})
    keydict.update({'net':kwargs.get('net','K_net [W]')})

    labeldict = {'inj':r'Injection','esc':r'Escape','net':r'Net'}
    colordict = {'inj':'mediumvioletred','esc':'peru','net':'black'}
    fcolordict = {'inj':'palevioletred','esc':'peachpuff','net':'silver'}
    ##Yvalue dictionary
    if kwargs.get('use_inner',False):
        for item,value in keydict:
            keydict.update({item:'inner'+value})
    elif kwargs.get('use_surface',False):
        keydict={'inj':'Utot_acquired [W]','esc':'Utot_forfeited [W]',
                    'net':'Utot_net [W]'}
    if kwargs.get('use_shield',False):
        pass#NOTE come back to this
    elif kwargs.get('use_average',False):
        powdict = {'inj':dfdict[keydict['inj']]/dfdict['Area [Re^2]'],
                    'esc':dfdict[keydict['esc']]/dfdict['Area [Re^2]'],
                    'net':dfdict[keydict['net']]/dfdict['Area [Re^2]']}
    else:
        powdict = {'inj':dfdict[keydict['inj']]/1e12,
                    'esc':dfdict[keydict['esc']]/1e12,
                    'net':dfdict[keydict['net']]/1e12}
        #axis.set_ylim([-20, 20])
    ##PLOT
    for term in ['inj','esc','net']:
        axis.plot(times,powdict[term],label=labeldict[term],
                    linewidth=kwargs.get('lw',None),
                    linestyle=kwargs.get('ls',None),
                    color=kwargs.get(term+'color',colordict[term]))
        if kwargs.get('dofill',True):
            axis.fill_between(times,powdict[term],
                            fc=kwargs.get(term+'fcolor',fcolordict[term]),
                            ec=kwargs.get(term+'color',colordict[term]),
                            hatch=kwargs.get('hatch'),lw=0.0)
    #General plot settings
    general_plot_settings(axis, **kwargs)

def plot_P0Power(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, use_average=False,
             use_surface=False):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dfdict:
        name = data['name'].iloc[-1]
        powerin_str = 'P0_injection [W]'
        powerout_str = 'P0_escape [W]'
        powernet_str = 'P0_net [W]'
        if use_inner:
            powerin_str = 'inner'+powerin_str
            powerout_str = 'inner'+powerout_str
            powernet_str = 'inner'+powernet_str
        elif use_surface:
            powerin_str = 'uHydro_acquired [W]'
            powerout_str = 'uHydro_forfeited [W]'
            powernet_str = 'uHydro_net [W]'
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
                powerin = data[powerin_str]/1e12
                powerout = data[powerout_str]/1e12
                powernet = data[powernet_str]/1e12
                axis.set_ylim([-20, 20])
            #INJECTION
            axis.plot(data[timekey],powerin,
                            label=r'Injection',
                        linewidth=Size, linestyle=ls,
                        color='gold')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerin,
                              color='wheat')
            #ESCAPE
            axis.plot(data[timekey],powerout,
                            label=r'Escape',
                        linewidth=Size, linestyle=ls,
                        color='peru')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerout,
                                  color='peachpuff')
            #NET
            if not use_shield:
                Size = 1
                axis.plot(data[timekey],powernet,
                            label=r'Net',
                            linewidth=Size, linestyle=ls,
                            color='maroon')
                if name.find('aggr')==-1:
                    axis.fill_between(data[timekey],powernet,
                                color='coral')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_ExBPower(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, use_average=False,
             use_surface=False):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dfdict:
        name = data['name'].iloc[-1]
        powerin_str = 'ExB_injection [W]'
        powerout_str = 'ExB_escape [W]'
        powernet_str = 'ExB_net [W]'
        if use_inner:
            powerin_str = 'inner'+powerin_str
            powerout_str = 'inner'+powerout_str
            powernet_str = 'inner'+powernet_str
        elif use_surface:
            powerin_str = 'uB_acquired [W]'
            powerout_str = 'uB_forfeited [W]'
            powernet_str = 'uB_net [W]'
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
                powerin = data[powerin_str]/1e12
                powerout = data[powerout_str]/1e12
                powernet = data[powernet_str]/1e12
                axis.set_ylim([-20, 20])
            #INJECTION
            axis.plot(data[timekey],powerin,
                            label=r'Injection',
                        linewidth=Size, linestyle=ls,
                        color='mediumvioletred')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerin,
                                  color='palevioletred')
            #ESCAPE
            axis.plot(data[timekey],powerout,
                            label=r'Escape',
                        linewidth=Size, linestyle=ls,
                        color='deepskyblue')
            if name.find('aggr')==-1:
                axis.fill_between(data[timekey],powerout,
                                  color='lightsteelblue')
            #NET
            if not use_shield:
                Size = 1
                axis.plot(data[timekey],powernet,
                            label=r'Net',
                            linewidth=Size, linestyle=ls,
                            color='midnightblue')
                if name.find('aggr')==-1:
                    axis.fill_between(data[timekey],powernet,
                                    color='blue')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_PowerSpatDist(axis, dfdict, timekey, ylabel, powerkey, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             allpositive=False):
    """Function plots spatatial distribution in terms of percents
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    powerin_str = powerkey+'_injection [W]'
    powerout_str = powerkey+'_escape [W]'
    powernet_str = powerkey+'_net [W]'
    if len(axis) == 1:
        injaxis = axis[0]
        escaxis = axis[0]
    else:
        injaxis = axis[0]
        escaxis = axis[1]
    if (dfdict[0]['name'].iloc[-1] !=-1) and (len(dfdict)>1):
        total_powerin = dfdict[0][~ dfdict[0][powerin_str].isna()][
                                                              powerin_str]
        total_powerout = dfdict[0][~ dfdict[0][powerin_str].isna()][
                                                             powerout_str]
        total_powernet = dfdict[0][~ dfdict[0][powerin_str].isna()][
                                                             powernet_str]
        for data in dfdict[1::]:
            data = data[~ data[powerin_str].isna()]
            total_powerin = total_powerin+data[powerin_str]
            total_powerout = total_powerout+data[powerout_str]
            total_powernet = total_powernet+data[powernet_str]
        for data in dfdict:
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
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    injaxis.set_xlabel(r'\textit{Time [UTC]}')
    escaxis.set_xlabel(r'\textit{Time [UTC]}')
    injaxis.set_ylabel(ylabel)
    escaxis.set_ylabel(ylabel)
    injaxis.legend(loc=legend_loc)
    escaxis.legend(loc=legend_loc)
def plot_DFTstack(axis, dfdict, timekey, ylabel, control_key, *,
                   xlim=None, ylim=None, Color=None, Size=2, ls=None,
                   do_percent=False):
    """Function plots power transfer at different vorticity bins
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    day = [df for df in dfdict if df['name'].iloc[-1].split('_')[1].split(
                                                    'aggr')[-1]=='day'][0]
    flank = [df for df in dfdict if df['name'].iloc[-1].split(
                                    '_')[1].split('aggr')[-1]=='flank'][0]
    tail = [df for df in dfdict if df['name'].iloc[-1].split('_')[1].split(
                                                    'aggr')[-1]=='tail'][0]
    Injlist, Injlabels, Esclist, Esclabels, k, kmax = [],[],[],[],0,99
    for data in dfdict:
        name = data['name'].iloc[-1]
        region = name.split('_')[1].split('aggr')[-1].capitalize()
        Colordict = dict({'Day':'deepskyblue',
                          'Flank':'magenta',
                          'Tail':'darkmagenta'})
        Linedict = dict({'Day':1, 'Flank':1, 'Tail':1})
        if (k==0):
            Injlist.append(data[control_key+'injection [W]'].dropna()/1e12)
            Injlabels.append(region)
            Esclist.append(data[control_key+'escape [W]'].dropna()/1e12)
            Esclabels.append(region)
        else:
            Injlist.append(Injlist[k-1]+data[
                                control_key+'injection [W]'].dropna()/1e12)
            Injlabels.append(region)
            Esclist.append(Esclist[k-1]+data[
                                   control_key+'escape [W]'].dropna()/1e12)
            Esclabels.append(region)
        k+=1
    if do_percent:
        newlist=[]
        for dat in enumerate(Injlist):
            newlist.append(-1*dat[1]/Injlist[-1])
        Injlist = newlist
        newlist=[]
        for dat in enumerate(Esclist):
            newlist.append(dat[1]/Esclist[-1])
        Esclist = newlist
        axis.set_ylim([-1,1])
    else:
        axis.set_ylim([-20,20])
    linelabels=[Injlabels, Esclabels]
    for typelist in enumerate([Injlist, Esclist]):
        for dat in enumerate(typelist[1]):
            region = linelabels[typelist[0]][dat[0]]
            if typelist[0]==0:
                axis.plot(data[timekey].dropna(),dat[1],
                  label=r'\textit{'+region+'}',
                  linewidth=Linedict[region], linestyle=ls,
                  color=Colordict[region])
            else:
                axis.plot(data[timekey].dropna(),dat[1],
                  linewidth=Linedict[region], linestyle=ls,
                  color=Colordict[region])
            if dat[0]==0:
                axis.fill_between(data[timekey].dropna(),dat[1],
                                  color=Colordict[region])
            else:
                axis.fill_between(data[timekey].dropna(),dat[1],
                      typelist[1][dat[0]-1], color=Colordict[region])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)
    #labelLines(axis.get_lines(),zorder=2.5,align=False,fontsize=11)

def VortValue(vortstr):
    return float(vortstr.split('vort')[-1].split('[')[-1].split('-')[0])

def plot_VortPower(axis, dfdict, timekey, ylabel, control_key, *,
                   xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots power transfer at different vorticity bins
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    control_flavor = control_key.split('_')[0]
    control_type = control_key.split('_')[-1].split(' ')[0]
    for data in dfdict:
        name = data['name'].iloc[-1]
        region = name.split('_')[1].split('aggr')[-1]
        Klist, Klabels, k, kmax = [], [], 0, 99
        Colorwheel = ['cadetblue','magenta','darkmagenta',
                      'indigo','blue','blueviolet','violet',
                      'royalblue','cyan','teal','dodgerblue',
                      'darkslateblue', 'black']
        for powkey in enumerate(sorted(
                     [key for key in data.keys() if key.find('vort')!=-1],
                                                          key=VortValue)):
            #wlabel = 'W '+powkey[1].split('vort')[-1]
            wlabel = str(VortValue(powkey[1])).split('.')[0]
            flavor = powkey[1].split('_')[0]
            powtype = powkey[1].split('_')[-1].split(' ')[0]
            if (control_flavor==flavor)and(
                powtype==control_type)and(Color==None):
                maxpercent =abs((data[powkey[1]]/data[control_key]).max())
                if (maxpercent>0.1) or (k==0):
                    if (k==0) or (control_type.find('net')!=-1):
                        Klist.append(data[powkey[1]].dropna())
                        Klabels.append(wlabel)
                    else:
                        Klist.append(Klist[k-1]+data[powkey[1]].dropna())
                        Klabels.append(wlabel)
                elif k<kmax:
                    Klist.append(data[control_key].dropna())
                    Klabels.append('Total')
                    Colorwheel[k] = 'black'
                    kmax = k
                print(control_key, wlabel, maxpercent)
                k+=1
        for dat in enumerate(Klist):
            axis.plot(data[timekey].dropna(),dat[1],
                      label=r'\textit{'+Klabels[dat[0]]+'}',
                      linewidth=Size, linestyle=ls,
                      color=Colorwheel[dat[0]])
            if (dat[0]==0) and (control_type.find('net')==-1):
                axis.fill_between(data[timekey].dropna(),dat[1],
                              color=Colorwheel[dat[0]])
            elif control_type.find('net')==-1:
                axis.fill_between(data[timekey].dropna(),dat[1],
                                  Klist[dat[0]-1],
                                  color=Colorwheel[dat[0]])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    #labelLines(axis.get_lines(),zorder=2.5,align=False,fontsize=11)

def plot_stackedPower(axis, dfdict, timekey, ylabel, control_key, *,
                   xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots power transfer at different vorticity bins
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    #TBD
    legend_loc = 'upper right'
    control_flavor = control_key.split('_')[0]
    control_type = control_key.split('_')[-1].split(' ')[0]
    for data in dfdict:
        name = data['name'].iloc[-1]
        region = name.split('_')[1].split('aggr')[-1]
        Klist, Klabels, k, kmax = [], [], 0, 99
        Colorwheel = ['cadetblue','magenta','darkmagenta',
                      'indigo','blue','blueviolet','violet',
                      'royalblue','cyan','teal','dodgerblue',
                      'darkslateblue', 'black']
        for powkey in enumerate(sorted(
                     [key for key in data.keys() if key.find('vort')!=-1],
                                                          key=VortValue)):
            #wlabel = 'W '+powkey[1].split('vort')[-1]
            wlabel = str(VortValue(powkey[1])).split('.')[0]
            flavor = powkey[1].split('_')[0]
            powtype = powkey[1].split('_')[-1].split(' ')[0]
            if (control_flavor==flavor)and(
                powtype==control_type)and(Color==None):
                maxpercent =abs((data[powkey[1]]/data[control_key]).max())
                if (maxpercent>0.1) or (k==0):
                    if (k==0) or (control_type.find('net')!=-1):
                        Klist.append(data[powkey[1]].dropna())
                        Klabels.append(wlabel)
                    else:
                        Klist.append(Klist[k-1]+data[powkey[1]].dropna())
                        Klabels.append(wlabel)
                elif k<kmax:
                    Klist.append(data[control_key].dropna())
                    Klabels.append('Total')
                    Colorwheel[k] = 'black'
                    kmax = k
                print(control_key, wlabel, maxpercent)
                k+=1
        for dat in enumerate(Klist):
            axis.plot(data[timekey].dropna(),dat[1],
                      label=r'\textit{'+Klabels[dat[0]]+'}',
                      linewidth=Size, linestyle=ls,
                      color=Colorwheel[dat[0]])
            if (dat[0]==0) and (control_type.find('net')==-1):
                axis.fill_between(data[timekey].dropna(),dat[1],
                              color=Colorwheel[dat[0]])
            elif control_type.find('net')==-1:
                axis.fill_between(data[timekey].dropna(),dat[1],
                                  Klist[dat[0]-1],
                                  color=Colorwheel[dat[0]])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    labelLines(axis.get_lines(),zorder=2.5,align=False,fontsize=11)

def plot_SurfacePower(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    qtkey = 'KSurface_'+typekey
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_trackEth(axis, dfdict, times, **kwargs):
    """Function plots energy overwrites during IM-GM coupling
    Inputs
        axis- object plotted on
        dfdict- datasets
        times-
        kwargs
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    #Update default legend_loc
    #kwargs.update({'legend_loc':kwargs.get('legend_loc','lower right')})
    for name,data in dfdict.items():
        eth_key = kwargs.get('eth_key','Eth_acc [J]')
        axis.plot(times,data[eth_key]/1e15,
                  label=kwargs.get('altlabel',safelabel(name)),
                  linewidth=kwargs.get('lw'), linestyle=kwargs.get('ls'),
                  color=kwargs.get('color','olive'))
    #General plot settings
    general_plot_settings(axis, **kwargs)

def plot_trackHth(axis, dfdict, times, **kwargs):
    """Function plots energy overwrites during IM-GM coupling
    Inputs
        axis- object plotted on
        dfdict- datasets
        times-
        kwargs
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    #Update default legend_loc
    #kwargs.update({'legend_loc':kwargs.get('legend_loc','lower right')})
    for name,data in dfdict.items():
        eth_key = kwargs.get('Hth_key','Wth [W]')
        axis.plot(times,data[eth_key]/1e12-data[eth_key].iloc[0]/1e12,
                  label=kwargs.get('altlabel',safelabel(name)),
                  linewidth=kwargs.get('lw'), linestyle=kwargs.get('ls'),
                  color=kwargs.get('color','olive'))
    #General plot settings
    general_plot_settings(axis, **kwargs)

def plot_TotalEnergy(axis, dfdict, times, **kwargs):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        times-
        kwargs
            ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    #Update default legend_loc
    kwargs.update({'legend_loc':kwargs.get('legend_loc','lower right')})
    for name,data in dfdict.items():
        totkey = kwargs.get('totkey','Utot [J]')
        axis.plot(times,-1*data[totkey]/1e15,
                  label=kwargs.get('altlabel',safelabel(name)),
                  linewidth=kwargs.get('lw'), linestyle=kwargs.get('ls'),
                  color=kwargs.get('color','coral'))
    #General plot settings
    general_plot_settings(axis, **kwargs)

def plot_VoverSA(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'V/SA [Re]'
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_SA(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'Area [Re^2]'
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_Standoff(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    qtkey = 'X_subsolar [Re]'
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{Simulation}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
        elif name.find('swmf_sw')!=-1:
            #calculate Pdyn
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            r0 = []
            for i in data.index:
                r0_i, _ = r0_alpha_1998(data['bz'].loc[i],
                                        data['Pdyn'].loc[i])
                r0.append(r0_i)
            data['r0'] = r0
            axis.plot(data[timekey],data['r0'],
                        label=r"\textit{Shue '98}",
                        linewidth=Size, linestyle=ls,
                        color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_Asymmetry(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    qtkey = ''
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name.find('asym')!=-1:
            tag = name.split('_')[-1]
            axis.fill_between(data[timekey].dropna(),
                              data['rmax-10'].dropna(),
                              data['rmin-10'].dropna(), color='silver')
            axis.fill_between(data[timekey].dropna(),
                          (data['rmean-10']+1.5*data['rstd-10']).dropna(),
                          (data['rmean-10']-1.5*data['rstd-10']).dropna(),
                              color='grey')
            axis.plot(data[timekey],data['rmean-10'],
                        label=r'\textit{Simulation}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
        elif name.find('swmf_sw')!=-1:
            def integrand1(theta, r0, alpha):
                return 2*pi/3*r0**3*(cos(theta/2))**(-6*alpha)
            def integrand2(theta):
                return 2*pi/3*(-20)**3*(cos(theta))**(-3)
            #calculate Pdyn
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            qtkey = 'Pdyn'
            shue_rmin, shue_rmean, shue_rmax = [], [], []
            for i in data.index:
                r0_i, alpha_i = r0_alpha_1998(data['bz'].loc[i],
                                              data['Pdyn'].loc[i])
                for X in enumerate([-9.5,-10,-10.5]):
                    #bisect to find r-> x=X
                    theta_l, theta_r, done = pi/2, pi*0.9, False
                    threshold, nstep, nmax = 0.01, 0, 100
                    while not done:
                        theta_m = (theta_l+theta_r)/2
                        r = r0_i*cos(theta_m/2)**(-2*alpha_i)
                        x_m = r*cos(theta_m)
                        if abs(x_m)<abs(X[1]):
                            theta_l=theta_m
                        else:
                            theta_r=theta_m
                        if (abs(X[1]-x_m)<threshold) or (nstep>nmax):
                            done=True
                        nstep+=1
                    if X[0]==0:
                        shue_rmin.append(r)
                    elif X[0]==1:
                        shue_rmean.append(r)
                    else:
                        shue_rmax.append(r)
            data['rmin'] = shue_rmin
            data['rmean'] = shue_rmean
            data['rmax'] = shue_rmax
            #axis.fill_between(data[timekey].dropna(),data['rmax'].dropna(),
            #                  data['rmin'].dropna(), color='lightblue')
            axis.plot(data[timekey],data['rmean'],
                        label=r"\textit{Shue '98}",
                        linewidth=Size+1, linestyle=ls,
                        color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        axis.set_ylim([8,40])
        axis.yaxis.set_minor_locator(AutoMinorLocator(5))
    #axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_Volume(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots outer surface boundary powers
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    qtkey = 'Volume [Re^3]'
    if Color == None:
        Color = 'magenta'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name.find('mp')!=-1:
            tag = name.split('_')[-1]
            axis.plot(data[timekey],data[qtkey],
                        label=r'\textit{Simulation}',
                        linewidth=Size, linestyle=ls,
                        color=Color)
        elif name.find('swmf_sw')!=-1:
            def integrand1(theta, r0, alpha):
                return 2*pi/3*r0**3*(cos(theta/2))**(-6*alpha)
            def integrand2(theta):
                return 2*pi/3*(-20)**3*(cos(theta))**(-3)
            #calculate Pdyn
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['v'] = np.sqrt(data['vx']**2+data['vy']**2+
                                data['vz']**2)
            data['Pdyn'] = data['dens']*data['v']**2*convert
            qtkey = 'Pdyn'
            #calculate shue volume
            shue_volume = []
            for i in data.index:
                r0_i, alpha_i = r0_alpha_1998(data['bz'].loc[i],
                                              data['Pdyn'].loc[i])
                #bisect to find theta-> x=-20
                X = -20
                theta_l, theta_r, done = pi/2, pi*0.9, False
                threshold, nstep, nmax = 0.01, 0, 100
                while not done:
                    theta_m = (theta_l+theta_r)/2
                    r = r0_i*cos(theta_m/2)**(-2*alpha_i)
                    x_m = r*cos(theta_m)
                    if abs(x_m)<abs(X):
                        theta_l=theta_m
                    else:
                        theta_r=theta_m
                    if (abs(X-x_m)<threshold) or (nstep>nmax):
                        done=True
                        print(x_m, theta_m)
                    nstep+=1
                #integrate in 3D from 0-theta_m
                I1 = quad(integrand1, 0, theta_m, args=(r0_i,alpha_i))[0]
                I2 = pi*(20/3)*(
                      r0_i*cos(theta_m/2)**(-2*alpha_i)*sin(theta_m))**2
                shue_volume.append(I1+I2)
            data['Volume'] = shue_volume
            axis.plot(data[timekey],data['Volume'],
                        label=r"\textit{Shue '98}",
                        linewidth=Size, linestyle=ls,
                        color='black')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    #axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))
    axis.legend(loc=legend_loc)

def plot_dst(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    pallete = {'supermag':'black','swmf_log':'magenta','omni':'black'}
    if Color != None:
        pallete = {'supermag':Color,'swmf_log':Color,'omni':Color}
    legend_loc = 'lower left'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'SMR (nT)'
            axis.plot(data[timekey],data[qtkey],
                      label='SMR',
                      linewidth=Size, linestyle=ls,
                      color=pallete[name])
        elif name == 'swmf_index':
            qtkey = 'dst_sm'
            axis.plot(data[timekey],data[qtkey],
                      label='Sim',
                      linewidth=Size, linestyle=ls,
                      color=pallete[name])
        elif name == 'omni':
            qtkey = 'sym_h'
            axis.plot(data[timekey],data[qtkey],
                      label='Obs',
                      linewidth=Size, linestyle=ls)
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_al(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots dst (or equivalent index) with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower left'
    for data in dfdict:
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
                      label='SWMF',
                      linewidth=Size, linestyle=ls)
        elif name == 'omni':
            qtkey = 'al'
            legend_loc = 'lower left'
            axis.plot(data[timekey],data[qtkey],
                      label='Obs',
                      linewidth=Size, linestyle=ls)
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_newell(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            data['Newell_corrected'] = data['Newell CF (Wb/s)']*100
            qtkey = 'Newell CF (Wb/s)'
            Color='black'
        elif name == 'omni':
            vsw = data['v']*1000
            By = data['by']*1e-9
            Bz = data['bz']*1e-9
            Cmp = 1000 #Followed
            #https://supermag.jhuapl.edu/info/data.php?page=swdata
            #term comes from Cai and Clauer[2013].
            #Note that SuperMAG lists the term as 100,
            #however from the paper "From our work,
            #                       α is estimated to be on order of 10^3"
            data['Newell']=Cmp*vsw**(4/3)*np.sqrt(
                           By**2+Bz**2)**(2/3)*abs(np.sin(
                                              np.arctan2(By,Bz)/2))**(8/3)
            qtkey='Newell'
            Color='blue'
        elif name == 'swmf_sw':
            #data = df_coord_transform(data, timekey, ['bx','by','bz'],
            #                          ('GSE','car'), ('GSM','car'))
            vsw = 1000*np.sqrt(data['vx']**2+data['vy']**2+data['vz']**2)
            clock = np.arctan2(data['by'],data['bz'])
            By = data['by']*1e-9
            Bz = data['bz']*1e-9
            Cmp = 1000 #Followed
            #https://supermag.jhuapl.edu/info/data.php?page=swdata
            #term comes from Cai and Clauer[2013].
            #Note that SuperMAG lists the term as 100,
            #however from the paper: "From our work,
            #                       α is estimated to be on order of 10^3"
            data['Newell']=Cmp*vsw**(4/3)*np.sqrt(
                           By**2+Bz**2)**(2/3)*abs(np.sin(clock/2))**(8/3)
            qtkey='Newell'
            Color='black'
        else:
            qtkey = None
        if qtkey != None:
            ok_indices = (~ data[qtkey].isna())
            axis.plot(data.loc[ok_indices][timekey],
                      data.loc[ok_indices][qtkey]/1000,
                      label='Newell Coupling Function',
                      linewidth=Size, linestyle=ls, color=Color)
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_cpcp(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots cross polar cap potential with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'omni':
            pci = data['pc_n']
            T = 2*pi*(2/12)
            data['cpcp'] = 29.28 - 3.31*sin(T+1.49)+17.81*pci
            axis.plot(data[timekey],data['cpcp'],
                      label=r'\textit{Ridley and Kihn}',
                      linewidth=Size, linestyle=ls, color='blue')
        else:
            north = 'cpcpn'
            south = 'cpcps'
            for pole in [north,south]:
                if pole==south:
                    axis.plot(data[timekey],data[pole],
                        label='South CPCP',
                        linewidth=Size,linestyle=ls,color='magenta')
                else:
                    axis.plot(data[timekey],data[pole],
                        label='North CPCP',
                        linewidth=Size, linestyle=ls, color='darkmagenta')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_swdensity(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind density with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper left'
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

def plot_DesslerParkerSckopke(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for simdata in dfdict:
        simname = simdata['name'].iloc[-1]
        if simname.find('mp')!=-1:
            total = simdata['Utot [J]']
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
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left')

def plot_akasofu(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            pass
        elif name == 'swmf_sw':
            bsquared_units = 1e-9**2 / (4*np.pi*1e-7)
            l = 7*6371*1000
            #data = df_coord_transform(data, timekey, ['bx','by','bz'],
            #                          ('GSE','car'), ('GSM','car'))
            data['v'] = 1000*np.sqrt(data['vx']**2+data['vy']**2+
                                     data['vz']**2)
            data['B^2'] = (data['bx']**2+data['by']**2+data['bz']**2)*(
                                                            bsquared_units)
            data['clock (GSE)'] = np.arctan2(data['by'],data['bz'])
            #data['clock (GSM)'] = np.arctan2(data['byGSM'],data['bzGSM'])
            data['eps(GSE) [W]'] = (data['B^2']*data['v']*
                                    np.sin(data['clock (GSE)']/2)**4*l**2)
            #data['eps(GSM) [W]'] = (data['B^2']*data['v']*
            #                        np.sin(data['clock (GSM)']/2)**4*l**2)
            qtkey = 'eps(GSE) [W]'
            axis.plot(data[timekey],data[qtkey]/1e12,
                      label=r'$\displaystyle \epsilon$',
                      linewidth=Size, linestyle=ls,
                      color='blue')
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    #axis.legend(loc='upper right')

def plot_pearson_r(axis, dfdict, ydf, xlabel, ylabel, *,
                   qtkeyX='dst',qtkeyY='Utot [J]',
                   xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            if qtkeyX=='dst':
                qtkey = 'SMR (nT)'
            elif qtkeyX=='pram':
                qtkey = 'Dyn. Pres. (nPa)'
            Color = 'black'
        elif name == 'swmf_log':
            if qtkeyX=='dst':
                qtkey = 'dst_sm'
            Color = 'magenta'
        elif name == 'swmf_sw':
            if qtkeyX=='pram':
                convert = 1.6726e-27*1e6*(1e3)**2*1e9
                data['v2'] = data['vx']**2+data['vy']**2+data['vz']**2
                data['pram'] = data['v2']*data['dens']*convert
                qtkey = 'pram'
            Color = 'magenta'
        elif name == 'omni':
            if qtkeyX=='dst':
                qtkey = 'sym_h'
            elif qtkeyX=='pram':
                convert = 1.6726e-27*1e6*(1e3)**2*1e9
                data['pram'] = data['v']**2 * data['density']*convert
                qtkey = 'pram'
            Color = 'lightgrey'
        if qtkeyX=='pram':
            useindex = data[qtkey]>0
        else:
            useindex = (~ data[qtkey].isna())
        ydata = np.interp(data.loc[useindex]['Time [UTC]'][0:-1],
                        ydf['Time [UTC]'][0:-1],ydf[qtkeyY][0:-1])
        xdata = data.loc[useindex][qtkey][0:-1].values
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

def plot_swbz(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dfdict:
        name = data['name'].iloc[-1]
        if name == 'supermag':
            qtkey = 'BzGSM (nT)'
            axis.plot(data[timekey],data[qtkey],
                      label='SuperMAG (OMNI)',
                      linewidth=Size, linestyle='--',
                      color='black')
        elif name == 'swmf_sw':
            #data = df_coord_transform(data, timekey, ['bx','by','bz'],
            #                          ('GSE','car'), ('GSM','car'))
            qtkey = 'bz'
            axis.plot(data[timekey],data[qtkey],
                      label=r'$B_z$',
                      linewidth=Size, linestyle=ls)
        elif name == 'omni':
            qtkey = 'bz'
            axis.plot(data[timekey],data[qtkey],
                      label='OMNI',
                      linewidth=Size, linestyle=ls,
                      color='tab:pink')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    #axis.axhline(0, color='black')
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)

def plot_swflowP(axis, dfdict, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'lower right'
    for data in dfdict:
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
                      linewidth=Size)
        elif name == 'omni':
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            data['Pdyn'] = data['density']*data['v']**2*convert
            qtkey = 'Pdyn'
            axis.plot(data[timekey],data[qtkey],
                      label='OMNI',
                      linewidth=Size, linestyle=ls,
                      color='tab:orange')
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))

def plot_swPower(axis, dfdict, mp, timekey, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None):
    """Function plots solar wind clock angle with given data frames
    Inputs
        axis- object plotted on
        dfdict- datasets
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    legend_loc = 'upper right'
    for data in dfdict:
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
    axis.set_xlabel(r'\textit{Time [UTC]}')
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

def chopends_time(dfdict, start, end, timekey,*,shift=False,
                  shift_minutes=45):
    """Function chops ends of dataframe based on time
    Inputs
        dfdict
        start, end
        timekey
        shift- boolean for shifting entire dataset
        shift_minutes- how much to shift (add time)
    Outputs
        dfdict
    """
    newlist = []
    for df in enumerate(dfdict):
        name = pd.Series({'name':df[1]['name'].iloc[-1]})
        if not any([key==timekey for key in df[1].keys()]):
            df[1][timekey] = df[1][
                 [key for key in df[1].keys() if key.find('Time')!=-1][0]]
        if shift:
            df[1][timekey]=(df[1][timekey]+
                            dt.timedelta(minutes=shift_minutes))
        cutdf = df[1][(df[1][timekey]>start)&
                      (df[1][timekey]<end)].append(
                            name,ignore_index=True)
        newlist.append(cutdf)
    return newlist

if __name__ == "__main__":
    #handling io paths
    datapath = sys.argv[-1]
    figureout = os.path.join(datapath,'figures')
    os.makedirs(figureout, exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print_presentation'))

    ##Loading data
    #Log files and observational indices
    [swmf_index, swmf_log, swmf_sw,_,omni]= read_indices(datapath,
                                                      read_supermag=False)
    #HDF data needs to be read, sorted and cleaned
    [mpdict,msdict,inner_mp,times,get_nonGM]=load_hdf_sort(
                                            datapath+'trackIM_results.h5')
    if get_nonGM:
        #Check for non GM data
        ie, ua_j, ua_e, ua_non, ua_ie= load_nonGM(datapath+'results.h5')

    ##Apply any mods and gather additional statistics
    [mpdict,msdict,inner_mp]=process_energy(mpdict,msdict,inner_mp,times)
    mp = [m for m in mpdict.values()][0]

    ##Construct "grouped" set of subzones
    if msdict!={}:
        msdict = group_subzones(msdict,mode='3zone')

    ##Plot data
    ######################################################################
    #IM Tracking initial values
    if True and 'WKE [W]'in mp.keys():
        #figure specs
        figname = 'Track_IM'
        track_im,(ax1) = plt.subplots(1,1,figsize=[14,6],sharex=True)
        ax2 = ax1.twinx()

        #labels
        Hth_label = r'$H_{therm,IM}\left[ TW \right]$'
        Eth_label = r'$E_{therm,IM}\left[ PJ \right]$'
        KE_label = r'$KE_{acc,IM}\left[ PJ \right]$'
        Knet_label= r'Integ. K Power $\left[TW\right]$'

        #Figure elements
        mark_times(ax1)
        #plot_power(ax1,mpdict,times,ylabel=Knet_label,ylim=[-7.5,7.5])
        plot_trackEth(ax1,mpdict,times,ylabel=Eth_label)
        plot_trackHth(ax2,mpdict,times,ylabel=Hth_label)

        #Additional adjustments
        track_im.tight_layout(pad=1)

        #Save
        track_im.savefig(figureout+'/'+figname+'.png')
    ######################################################################
    #Newell Function and Outerbound Net Power, and cpcp
    if False:
        figname = 'Newell_NetOutterPower'
        new_out_pow, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        new_out_pow.tight_layout(pad=1.1*padsize)
        #Time
        timekey = 'Time [UTC]'
        ylabel = r'\textit{Potential} $\displaystyle V \left[Wb/s \right]$'
        y1label = 'Coupling Function [Wb/s]'
        y2label = 'Power Injection [W]'
        y3label = 'Cross Polar Cap Potential [Wb/s]'
        #ax2 = ax1.twinx()
        plot_newell(ax1, [supermag], timekey, ylabel, ylim=[0,250])
        plot_cpcp(ax1, [swmf_log], timekey, ylabel, ylim=[0,250])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        new_out_pow.savefig(figureout+'/'+figname+'.eps')
    ######################################################################
    #Total energy and dst index
    if True:
        #figure specs
        figname = 'Energy_dst'
        energy_dst,(ax1) = plt.subplots(1,1,figsize=[14,6],sharex=True)
        ax2 = ax1.twinx()

        #labels
        db_label = r'$\Delta B \left[ nT \right]$'
        etot_label = r'$E_{total} \left[ PJ \right]$'

        #Figure elements
        mark_times(ax2)
        plot_dst(ax1, [swmf_log, omni], 'Time [UTC]', db_label)
        plot_TotalEnergy(ax2, mpdict, times, ylabel=etot_label)

        #Additional adjustments
        ax2.spines['right'].set_color('blue')
        ax2.yaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='blue')
        energy_dst.tight_layout(pad=1)

        #Save
        energy_dst.savefig(figureout+'/'+figname+'.png')
    ######################################################################
    #Correlation for energy and dst
    if False:
        figname = 'Energy_dst_rcor'
        energy_corr, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figy,figy])
        #Time
        timekey = 'Time [UTC]'
        xlabel = r'$\displaystyle Dst\textit{equiv.}\left[ nT \right]$'
        ylabel = r'$\displaystyle E_{total} \left[ J \right]$'
        plot_pearson_r(ax1, [omni,swmf_log] ,mp, xlabel, ylabel)
        ax1.set_facecolor('dimgrey')
        energy_corr.tight_layout(pad=0.5)
        #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #ax1.tick_params(which='major', length=7)
        #ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        #ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        energy_corr.savefig(figureout+'{}.png'.format(figname))
    ######################################################################
    #Correlation for ram pressure and volume
    if False:
        figname = 'Pram_volume_rcor'
        pram_corr, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figy,figy])
        #Time
        timekey = 'Time [UTC]'
        xlabel = r'$P_{\textit{ram}} \left[ nPa\right]$'
        ylabel = r'$\textit{Volume}^{(-2.2)} \left[ R_e^3 \right]$'
        plot_pearson_r(ax1, [swmf_sw] ,mp, xlabel, ylabel,
                       qtkeyX='pram', qtkeyY='Volume [Re^3]')
        '''
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        '''
        pram_corr.tight_layout(pad=1)
        pram_corr.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Power and AL
    if False:
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
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        power_al.savefig(figureout+'{}_justindex.eps'.format(figname))
        power_al.savefig(figureout+'{}_justindex.tiff'.format(figname))
        ax2 = ax1.twinx()
        power_al.tight_layout(pad=padsize*1.2)
        plot_Power_al(ax2, [mp], timekey, y2label)
        power_al.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Theoretical power from the solar wind
    if False:
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
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #new_out_pow.autofmt_xdate()
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        swpower.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Total Power injection, escape and net
    if False:
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
        plot_Power(ax1, [mp], timekey, y1label, ylim=[-3e13,3e13])
        plot_Power(in_ax1, [mp], timekey, y1label, use_inner=True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        power.savefig(figureout+'{}.eps'.format(figname))
        innerpower.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #ExB Power injection, escape and net
    if False:
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
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        exbpower.savefig(figureout+'{}.eps'.format(figname))
        innerexbpower.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #P0 Power injection, escape and net
    if False:
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
        ax1.tick_params(which='major', length=7)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(6))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        #power.autofmt_xdate()
        in_ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #innerpower.autofmt_xdate()
        #shade_plot(ax1), shade_plot(in_ax1)
        #ax1.set_facecolor('olive'), in_ax1.set_facecolor('olive')
        p0power.savefig(figureout+'{}.eps'.format(figname))
        innerP0power.savefig(figureout+'{}_inner.eps'.format(figname))
    ######################################################################
    #3panel Power, power_inner, and shielding
    if False:
        figname = '3panelPower'
        panel3power, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True, figsize=[figx,3*figy])
        in3panelpower,(in_ax1, in_ax2, in_ax3)=plt.subplots(
                                            nrows=3,ncols=1, sharex=True,
                                          figsize=[figx,3*figy])
        sh3panelpower,(sh_ax1, sh_ax2, sh_ax3)=plt.subplots(
                                            nrows=3,ncols=1, sharex=True,
                                          figsize=[figx,3*figy])
        in3panelpower.tight_layout(pad=padsize*1.1)
        sh3panelpower.tight_layout(pad=padsize*1.3)
        #Time
        timekey = 'Time [UTC]'
        yKlabel = r'\textit{Integrated  $\mathbf{K}$ Power} $\left[ TW\right]$'
        ySlabel = r'\textit{Integrated  $\mathbf{S}$ Power} $\left[ TW\right]$'
        yPlabel = r'\textit{Integrated  $\mathbf{H}$ Power} $\left[ TW\right]$'
        y2label=r'\textit{Transfer Efficiency}$\left[\%\right]$'
        plot_Power(ax1, [mp], timekey, yKlabel)
        plot_Power(in_ax1, [mp], timekey, yKlabel, use_inner=True)
        #plot_Power(sh_ax1, [mp], timekey, y2label, use_shield=True)
        plot_P0Power(ax3, [mp], timekey, yPlabel)
        plot_P0Power(in_ax3, [mp], timekey, yPlabel, use_inner=True)
        #plot_P0Power(sh_ax3, [mp], timekey, y2label, use_shield=True)
        plot_ExBPower(ax2, [mp], timekey, ySlabel)
        plot_ExBPower(in_ax2, [mp], timekey, ySlabel, use_inner=True)
        #plot_ExBPower(sh_ax2, [mp], timekey, y2label, use_shield=True, ylim=[-350,350])
        for ax in [ax1, ax2, ax3]:
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            mark_times(ax)
            ax.tick_params(which='major', length=9)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.set_xlabel(None)
        ax3.set_xlabel(timekey)
        panel3power.tight_layout(pad=1)
        panel3power.savefig(figureout+'{}.eps'.format(figname))
        panel3power.savefig(figureout+'{}.tiff'.format(figname))
        panel3power.savefig(figureout+'{}.png'.format(figname))
        in3panelpower.savefig(figureout+'{}_inner.eps'.format(figname))
        sh3panelpower.savefig(figureout+'{}_shield.eps'.format(figname))
    ######################################################################
    #Solarwind, regular sized
    if False:
        figname = 'SolarWind'
        solarwind, (ax2,ax3)= plt.subplots(nrows=2,ncols=1,sharex=True,
                                                     figsize=[0.8*figx,0.8*figy])
        solarwind.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Density} $\displaystyle \left[\#/cm^3\right]$'
        y2label = r'\textit{IMF} $\displaystyle B_z \left[nT\right]$'
        y3label = r'\textit{Flow Pressure}$\displaystyle \left[nPa\right]$'
        #plot_swdensity(ax1, [swmf_sw], timekey, y1label)
        plot_swbz(ax2, [swmf_sw], timekey, y2label)
        plot_swflowP(ax3, [swmf_sw], timekey, y3label)
        for ax in [ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            mark_times(ax)
            ax.tick_params(which='major', length=7)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.set_xlabel(None)
        ax3.set_xlabel(timekey)
        solarwind.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Solarwind, expanded
    if False:
        figname = 'ExpandedSolarWind'
        bigsw, (ax1,ax2,ax3)= plt.subplots(nrows=3,ncols=1,sharex=True,
                                                     figsize=[figx*1.5,2*figy])
        bigsw.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Density} $\displaystyle \left[\#/cm^3\right]$'
        y2label = r'\textit{IMF} $\displaystyle B_z \left[nT\right]$'
        y3label = r'\textit{Flow Pressure}$\displaystyle \left[nPa\right]$'
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
    #3panel Akosofu, Newell, Dst
    if False:
        figname = '3panelProxies'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                       'flank_full')!=-1]
        panel2prox,(ax1,ax2,ax3)=plt.subplots(nrows=3,ncols=1,sharex=True,
                                              figsize=[14,12])
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Flank integrated  $\mathbf{K}$ Power }$\left[ TW\right]$'
        y1blabel = r'$\epsilon \left[ TW\right]$'
        y2label = r'\textit{Potential}$\displaystyle \left[kV\right]=\left[kWb/s\right]$'
        y3label = r'$Dst \left[ nT \right]$'
        y3blabel = r'$-1*E_{total} \left[ PJ \right]$'
        ax1twin = ax1.twinx()
        ax3twin = ax3.twinx()
        plot_akasofu(ax1twin, [swmf_sw], timekey, y1blabel)
        plot_Power_al(ax1, aggsublist, 'Time_UTC', y1label, multiplier=1)
        plot_newell(ax2, [swmf_sw], timekey, y2label)
        plot_cpcp(ax2, [swmf_log, omni], timekey, y2label)
        plot_dst(ax3, [swmf_log, omni], timekey, y3label)
        plot_TotalEnergy(ax3twin, [mp], timekey, y3blabel)
        for ax in [ax1, ax1twin, ax2, ax3, ax3twin]:
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            mark_times(ax)
            ax.tick_params(which='major', length=9)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel(None)
        ax3.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        ax3.legend(loc='lower left')
        for ax in enumerate([ax1twin, ax3twin]):
            ax[1].spines['right'].set_color('blue')
            ax[1].yaxis.label.set_color('blue')
            ax[1].tick_params(axis='y', colors='blue')
            if ax[0]==1:
                ax[1].legend(loc='lower right', fontsize=18,
                             labelcolor='blue')
            else:
                ax[1].legend().remove()
        panel2prox.tight_layout(pad=1)
        panel2prox.savefig(figureout+'{}.eps'.format(figname))
        panel2prox.savefig(figureout+'{}.tiff'.format(figname))
    ######################################################################
    #Dessler-Parker-Sckopke
    if False:
        figname = 'DesslerParkerSckopke'
        DPS, (ax1)=plt.subplots(nrows=1, ncols=1,
                                          sharex=True, figsize=[figx,figy])
        DPS.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        ylabel = r'\textit{Energy} $\displaystyle \left[J\right]$'
        plot_DesslerParkerSckopke(ax1, [swmf_log, supermag, omni, mp],
                                  timekey, ylabel)
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        DPS.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces power
    if False:
        figname = 'ComparativePower'
        power_comp, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        power_comp.tight_layout(pad=padsize)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        #plot_Power(ax1, [mp], timekey, y1label, Color='midnightblue')
        plot_Power(ax1, [mplist[0]], timekey, y1label, Color='coral')
        plot_Power(ax1, [mplist[1]], timekey, y1label, Color='gold')
        plot_Power(ax1, [mplist[2]], timekey, y1label, Color='plum')
        #plot_Power(ax1, [difference_mp], timekey, y1label, Color='midnightblue')
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        power_comp.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces volumes
    if False:
        figname = 'MPStandoff'
        mpstand, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[14,4])
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{R$_{MP}$} $\left[ R_E\right]$'
        plot_Standoff(ax1, [swmf_sw, mp], timekey, y1label)
        for ax in [ax1]:
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            mark_times(ax)
            ax.tick_params(which='major', length=9)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel(None)
        ax1.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        mpstand.tight_layout(pad=1)
        mpstand.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of mulitple magnetopause surfaces volumes
    if False:
        figname = 'ComparativeSurfaceArea'
        surf_comp, (ax1) = plt.subplots(nrows=1, ncols=1,sharex=True,
                                          figsize=[figx,figy])
        surf_comp.tight_layout(pad=padsize*2)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'\textit{Area} $\displaystyle \left[ R_e^2\right]$'
        plot_SA(ax1, [mp], timekey, y1label, Color='midnightblue')
        plot_SA(ax1, [mplist[0]], timekey, y1label, Color='coral')
        #plot_SA(ax1, [mplist[1]], timekey, y1label, Color='gold')
        #plot_SA(ax1, [mplist[2]], timekey, y1label, Color='plum')
        #shade_plot(ax1)
        #ax1.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        surf_comp.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'Day3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                          'day_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        plot_Power(ax1, aggsublist, timekey, y1label)
        plot_P0Power(ax2, aggsublist, timekey, y1label)
        plot_ExBPower(ax3, aggsublist, timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'Flank3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                       'flank_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        y2label = r'\textit{Transfer Efficiency}$\displaystyle \left[ \% \right]$'
        plot_Power(ax1, aggsublist, timekey, y1label)
        plot_P0Power(ax2, aggsublist, timekey, y1label)
        plot_ExBPower(ax3, aggsublist, timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'Tail3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                          'tail_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        plot_Power(ax1, aggsublist, timekey, y1label)
        plot_P0Power(ax2, aggsublist, timekey, y1label)
        plot_ExBPower(ax3, aggsublist, timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'AverageDay3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                          'day_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left[ W\right]$'
        plot_Power(ax1, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'AverageFlank3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                        'flank_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left[ W\right]$'
        plot_Power(ax1, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'AverageTail3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                        'flank_full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Average Power} $\displaystyle \left[ W\right]$'
        y2label = r'\textit{Transfer Efficiency}$\displaystyle \left[ \% \right]$'
        plot_Power(ax1, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_P0Power(ax2, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])
        plot_ExBPower(ax3, aggsublist, timekey, y1label, use_average=True,
                   ylim=[-2e10,2e10])

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'DayFlankTail3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                              'full')!=-1]
        dft_3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[0.8*figx,1.6*figy])
        dft_3.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        plot_Power(ax1, aggsublist, timekey, y1label)
        plot_P0Power(ax2, aggsublist, timekey, y1label)
        plot_ExBPower(ax3, aggsublist, timekey, y1label)

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3)
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        dft_3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'Spatial_Powers3panel'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                              'full')!=-1]
        spPower3, (ax1,ax2,ax3)=plt.subplots(nrows=3, ncols=1,
                                          sharex=True,
                                          figsize=[figx,3*figy])
        spPower3.tight_layout(pad=padsize*1.3)
        #Time
        timekey = 'Time_UTC'
        y1label = r'\textit{Power \%} $\displaystyle \left[ W\right]$'
        y2label = r'\textit{ExB Power \%} $\displaystyle \left[ W\right]$'
        y3label = r'\textit{P0 Power \%} $\displaystyle \left[ W\right]$'
        plot_PowerSpatDist([ax1], aggsublist, timekey, y1label, 'K')
        plot_PowerSpatDist([ax2], aggsublist, timekey, y2label, 'P0')
        plot_PowerSpatDist([ax3], aggsublist, timekey, y3label, 'ExB')

        ax1.legend(loc='upper left', facecolor='gray')
        ax2.legend(loc='upper left', facecolor='gray')
        ax3.legend(loc='upper left', facecolor='gray')

        #shade_plot(ax1), shade_plot(ax2), shade_plot(ax3),
        #ax1.set_facecolor('olive'), ax2.set_facecolor('olive'),
        #ax3.set_facecolor('olive'),
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        spPower3.savefig(figureout+'{}.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'Spatial_Powers'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                              'full')!=-1]
        spPower, (ax1)=plt.subplots(nrows=1, ncols=1, sharex=True,
                                         figsize=[14,4])
        timekey = 'Time_UTC'
        y1label = r'\textit{Power Fraction}'
        keys = ['K_']
        axes = [ax1]
        ylabels = [y1label]
        for key in enumerate(keys):
            plot_DFTstack(axes[key[0]],aggsublist,timekey,ylabels[key[0]],
                          key[1], do_percent=True)
            axes[key[0]].xaxis.set_major_formatter(
                                         mdates.DateFormatter('%d-%H:%M'))
            axes[key[0]].tick_params(which='major', length=7)
            axes[key[0]].xaxis.set_minor_locator(AutoMinorLocator(6))
            axes[key[0]].yaxis.set_minor_locator(AutoMinorLocator(5))
            mark_times(axes[key[0]])
            axes[key[0]].set_xlabel(None)
        ax1.axhline(0,color='black')
        ax1.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        spPower.tight_layout(pad=1)
        spPower.savefig(figureout+'{}.eps'.format(figname+'_K'))
    ######################################################################
    #timeseries of data at a set of fixed points on the surface
    if False:
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
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'StackedDFT_K'
        timekey = 'Time_UTC'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                              'full')!=-1]
        keys = ['K_injection [W]', 'K_escape [W]', 'K_net [W]',
                'P0_injection [W]', 'P0_escape [W]', 'P0_net [W]',
                'ExB_injection [W]', 'ExB_escape [W]', 'ExB_net [W]']
        y1 = r'\textit{Power} $\displaystyle _{injection}\left[ W\right]$'
        y2 = r'\textit{Power} $\displaystyle _{escape}\left[ W\right]$'
        y3 = r'\textit{Power} $\displaystyle _{net}\left[ W\right]$'
        ylabels = [y1, y2, y3, y1, y2, y3, y1, y2, y3]
        for key in enumerate(keys):
            fig = plt.figure(figsize=[14,6])
            ax = fig.add_subplot()
            fig.tight_layout(pad=3.2)
            plot_VortPower(ax,aggsublist,timekey,ylabels[key[0]],key[1])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            fig.savefig(figureout+figname+'_'+keys[key[0]].split(' ')[0]+
                        '.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'MPVortDay'
        timekey = 'Time_UTC'
        aggsublist = [ag for ag in energetics_list if ag[
                                       'name'].iloc[-1].find('day')!=-1]
        keys = ['K_injection [W]', 'K_escape [W]', 'K_net [W]',
                'P0_injection [W]', 'P0_escape [W]', 'P0_net [W]',
                'ExB_injection [W]', 'ExB_escape [W]', 'ExB_net [W]']
        y1 = r'\textit{Power} $\displaystyle _{injection}\left( W\right)$'
        y2 = r'\textit{Power} $\displaystyle _{escape}\left( W\right)$'
        y3 = r'\textit{Power} $\displaystyle _{net}\left( W\right)$'
        ylabels = [y1, y2, y3, y1, y2, y3, y1, y2, y3]
        for key in enumerate(keys):
            fig = plt.figure(figsize=[14,6])
            ax = fig.add_subplot()
            fig.tight_layout(pad=3.2)
            plot_VortPower(ax,aggsublist,timekey,ylabels[key[0]],key[1])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            fig.savefig(figureout+figname+'_'+keys[key[0]].split(' ')[0]+
                        '.eps'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'MPVortFlank'
        timekey = 'Time_UTC'
        aggsublist = [ag for ag in energetics_list if ag[
                                       'name'].iloc[-1].find('flank')!=-1]
        keys = ['K_injection [W]', 'K_escape [W]', 'K_net [W]',
                'P0_injection [W]', 'P0_escape [W]', 'P0_net [W]',
                'ExB_injection [W]', 'ExB_escape [W]', 'ExB_net [W]']
        y1 = r'\textit{Power} $\displaystyle _{injection}\left[ W\right]$'
        y2 = r'\textit{Power} $\displaystyle _{escape}\left[ W\right]$'
        y3 = r'\textit{Power} $\displaystyle _{net}\left[ W\right]$'
        ylabels = [y1, y2, y3, y1, y2, y3, y1, y2, y3]
        for key in enumerate(keys):
            fig = plt.figure(figsize=[14,6])
            ax = fig.add_subplot()
            fig.tight_layout(pad=3.2)
            plot_VortPower(ax,aggsublist,timekey,ylabels[key[0]],key[1])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            fig.savefig(figureout+figname+'_'+keys[key[0]].split(' ')[0]+
                        '.eps'.format(figname))
    ######################################################################
    #Total Power injection, escape and net
    if False:
        figname = 'SurfacePower'
        asym=[ag for ag in agglist if ag['name'].iloc[-1].find('asym')!=-1]
        power,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4, ncols=1,sharex=True,
             figsize=[14,15],gridspec_kw={'height_ratios':[1,2,1,1]})
        #Time
        timekey = 'Time [UTC]'
        y1label = r'$P_{\textit{ram}} \left[ nPa\right]$'
        y2label = r'\textit{Integrated  $\mathbf{K}$ Power} $\left[ TW\right]$'
        y3label = r'\textit{Volume} $\left[ R_{e}^3\right]$'
        y4label = r'$\Huge{\mathbf{\rho}}\left[R_e\right]$'
        plot_swflowP(ax1, [swmf_sw], timekey, y1label)
        ax1.set_xlabel(None)
        #plot_Power(ax2, [mp], timekey, y2label, use_surface=True,
        #           ylim=[-1e13, 1e13])
        ax2.plot(mp[timekey],mp['K_net [W]']/1e12,
                 label=r'\textit{Net static $+$ motional}',
                 linewidth=1, linestyle=None, color='silver')
        ax2.fill_between(mp[timekey], mp['K_net [W]']/1e12,color='silver')
        ax2.fill_between(mp[timekey], mp['Utot_net [W]']/1e12,color='magenta')
        ax2.plot(mp[timekey],mp['Utot_net [W]']/1e12,
                 label=r'\textit{Net motional}',
                 linewidth=1, linestyle=None, color='darkmagenta')
        ax2.set_ylabel(y2label)
        ax2.set_ylim([-6,6])
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.legend()
        plot_Volume(ax3, [mp,swmf_sw], timekey, y3label)
        plot_Asymmetry(ax4, [asym[0],swmf_sw], timekey, y4label)
        for ax in [ax1,ax2,ax3,ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax.tick_params(which='major', length=7)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            mark_times(ax)
        ax4.set_xlabel(r'\textit{Time [UTC]}')
        power.tight_layout(pad=1)
        power.savefig(figureout+'{}.eps'.format(figname))
        power.savefig(figureout+'{}.tiff'.format(figname))
    ######################################################################
    #Total Power injection, escape and net
    if False:
        figname = 'SurfacePower3types'
        power, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,sharex=True,
                                          figsize=[figx,2*figy])
        power.tight_layout(pad=padsize*1.1)
        #Time
        timekey = 'Time [UTC]'
        y1label = r'$\displaystyle P_{ram} \left( nPa\right)$'
        y2label = r'\textit{Power} $\displaystyle \left[ W\right]$'
        y3label = r'\textit{Volume} $\displaystyle \left[ R_{e}^3\right]$'
        plot_Power(ax1, [mp], timekey, y2label, use_surface=True,
                   ylim=[-3e12, 3e12])
        plot_P0Power(ax2, [mp], timekey, y2label, use_surface=True,
                   ylim=[-3e12, 3e12])
        plot_ExBPower(ax3, [mp], timekey, y2label, use_surface=True,
                   ylim=[-3e12, 3e12])
        for ax in [ax1,ax2,ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax.tick_params(which='major', length=7)
            ax.xaxis.set_minor_locator(AutoMinorLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            mark_times(ax)
        #power.autofmt_xdate()
        #shade_plot(ax1)
        power.savefig(figureout+'{}.eps'.format(figname))
        power.savefig(figureout+'{}.tiff'.format(figname))
    ######################################################################
    #Comparisons of sections of the magntopause surface
    if False:
        figname = 'MPDFTStacked'
        timekey = 'Time_UTC'
        aggsublist = [ag for ag in agglist if ag['name'].iloc[-1].find(
                                                              'full')!=-1]
        keys = ['K_','ExB_','P0_', 'K_']
        yK = r'\textit{Integrated  $\mathbf{K}$ Power} $\left[ TW\right]$'
        yS = r'\textit{Integrated  $\mathbf{S}$ Power} $\left[ TW\right]$'
        yP = r'\textit{Integrated  $\mathbf{H}$ Power} $\left[ TW\right]$'
        yfrac = r'\textit{$\mathbf{K}$ Power Fraction}'
        ylabels = [yK, yS, yP, yfrac]
        percents = [False, False, False, True]
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1,sharex=True,
                                          figsize=[14,16])
        axes = [ax1,ax2,ax3,ax4]
        for key in enumerate(keys):
            plot_DFTstack(axes[key[0]],aggsublist,timekey,ylabels[key[0]],
                          key[1], do_percent=percents[key[0]])
            axes[key[0]].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            axes[key[0]].tick_params(which='major', length=7)
            axes[key[0]].xaxis.set_minor_locator(AutoMinorLocator(6))
            axes[key[0]].yaxis.set_minor_locator(AutoMinorLocator())
            mark_times(axes[key[0]])
            axes[key[0]].set_xlabel(None)
            axes[key[0]].axhline(0, color='black')
        ax4.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        fig.tight_layout(pad=1)
        fig.savefig(figureout+figname+'_3panel.eps'.format(figname))
        fig.savefig(figureout+figname+'_3panel.tiff'.format(figname))
    ######################################################################
