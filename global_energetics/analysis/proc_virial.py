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
import swmfpy
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage imports
from global_energetics.analysis.proc_temporal import read_energetics

def plot_virial_dst(axis, dflist, timekey, ylabel, *,
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
        if name.find('mp')==-1:
            plot_dst(axis, dflist, timekey, ylabel, xlim=xlim,ylim=ylim,
                     Color=Color, Size=Size, ls=ls)
            return
        else:
            qtkey = None
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    axis.set_xlabel(r'\textit{Time [UTC]}')
    axis.set_ylabel(ylabel)
    axis.legend(loc=legend_loc)

if __name__ == "__main__":
    datapath = sys.argv[1::]
    figureout = datapath[-1]+'figures/'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    energeticslist = chopends_time(energetics_list,cuttoffstart,
                                   cuttoffend, 'Time [UTC]', shift=True)
    mplist=[]
    for df in energetics_list:
        if df['name'].iloc[-1].find('mp_')!=-1:
            mplist.append(df)
    mp=mplist[0]
    ######################################################################
    #Virial contributions
    if True:
        figname = 'VirialMultipanel'
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
