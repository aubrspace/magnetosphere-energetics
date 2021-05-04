#!/usr/bin/env python3
"""Functions for handling and processing spatial magnetopause surface data
    that is static in time
"""
import logging as log
import os
import sys
import glob
import time
from array import array
import numpy as np
from numpy import (abs, pi, cos, sin, sqrt, rad2deg,
                   linspace, deg2rad, arctan2)
import scipy as sp
from scipy.interpolate import griddata
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage
from global_energetics.analysis import proc_temporal

def plot_power_dial(fig, ax, df,  powerkey, *,
                    contourlim=None, rlim=None, show_colorbar=False):
    """Function plots polar contour plot of power
    Inputs
        ax- axis for the plot
        df- dataframe containing data
    """
    #get header info
    name = df[~ df['name'].isna()]['name'].values[0]
    timestamp = df[~ df['Time [UTC]'].isna()]['Time [UTC]'].values[0]
    sector = df[~ df['sector'].isna()]['sector'].values[0]
    #set variables
    alpha = df[~ df['alpha [rad]'].isna()]['alpha [rad]']
    x = df[~ df['x_cc'].isna()]['x_cc']
    h = df[~ df['h [R]'].isna()]['h [R]']
    power = df[~ df[powerkey].isna()][powerkey]
    #construct axes
    if sector.find('day') != -1:
        thi = linspace(-pi,pi,1441)
        xi = linspace(15,-4, 401)
        zi = griddata((alpha,x),power,(thi[None,:],xi[:,None]),
                       method='linear')
        ax.set(rlim=(15, -4))
        ax.set_title('Dayside', pad=12)
        ax.set_theta_zero_location('E')
    if sector.find('flank') != -1:
        thi = linspace(-pi,pi,1441)
        xi = linspace(-2,-22, 361)
        zi = griddata((alpha,x),power,(thi[None,:],xi[:,None]),
                       method='linear')
        ax.set(rlim=(-2, -22))
        ax.set_title('Flank', pad=12)
        ax.set_theta_zero_location('E')
    if sector.find('tail') != -1:
        thi = linspace(-pi,pi,1441)
        xi = linspace(0,30, 161)
        zi = griddata((alpha,h),power,(thi[None,:],xi[:,None]),
                       method='linear')
        ax.set(rlim=(0, 30))
        ax.set_title('Tail', pad=12)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(-1)
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
    cont_lvl = linspace(-1e9, 1e9, 11)
    cntr = ax.contourf(thi, xi, zi, cont_lvl, cmap=colors)
    ax.set_thetagrids([0,45,90,135,180,225,270,315],
                      ['+Y','+45','+Z','+135','-Y','-135','-Z','-45'])
    if show_colorbar:
        fig.colorbar(cntr, ax=axes, ticks=cont_lvl)

def make_spatial_plots(day, flank, tail):
    """Function makes 2D plots of spatial distributions of varialbes
    Inputs
        day, flank, tail- subsets of datframe for single timestep
    """
    ######################################################################
    #3panel Power, power_inner, and shielding
    if True:
        figname = '9dialPower'
        dial9power,([ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9])=plt.subplots(
                                                         nrows=3, ncols=3,
                                       subplot_kw=dict(projection='polar'),
                                                    figsize=[16,16])
        ax1 = dial9power.add_subplot(111,projection='polar')
        dial9power.tight_layout(pad=1)
        #total
        plot_power_dial(dial9power, ax1, day, 'K_net [W/Re^2]')
        plot_power_dial(dial9power, ax2, flank, 'K_net [W/Re^2]')
        plot_power_dial(dial9power, ax3, tail, 'K_net [W/Re^2]',
                                              show_colorbar=True)
        #ExB
        plot_power_dial(dial9power, ax4, day, 'ExB_net [W/Re^2]')
        plot_power_dial(dial9power, ax5, flank, 'ExB_net [W/Re^2]')
        plot_power_dial(dial9power, ax6, tail, 'ExB_net [W/Re^2]',
                                                show_colorbar=True)
        #P0
        plot_power_dial(dial9power, ax7, day, 'P0_net [W/Re^2]')
        plot_power_dial(dial9power, ax8, flank, 'P0_net [W/Re^2]')
        plot_power_dial(dial9power, ax9, tail, 'P0_net [W/Re^2]',
                                                show_colorbar=True)

        #ax1.legend(loc='upper left', facecolor='gray')

        dial9power.savefig(figureout+'{}.png'.format(figname))
    ######################################################################

def split_day_flank_tail(df, *, xdaymin=-2, flank_min_h_buff=10):
    """Function splits dataframe into dayside, flank and tail portions
    Inputs
        df- single dataframe, must have some derived varaiables!!
        xdaymin- limit to cuttoff dayside portion
        flank_min_h_buff- buffer to search for min h in flank (for tail)
    Outputs
        day, flank, tail- 3 separate dataframes
    """
    #cut the name and time to save for later
    name = pd.Series({'name':df['name'].iloc[-2]})
    time = pd.Series({'Time [UTC]':df['Time [UTC]'].iloc[-1]})
    daytag = pd.Series({'sector':'day'})
    flanktag = pd.Series({'sector':'flank'})
    tailtag = pd.Series({'sector':'tail'})
    #Identify dayside based on xlocation only
    daycond = df['x_cc']>xdaymin
    day = df[daycond]
    #Identify tail based on xmax and inner h values (near rxn site)
    flank_min_h = df[(df['x_cc']<xdaymin) &
                     (df['x_cc']>df['x_cc'].min()+
                                         flank_min_h_buff)]['h [R]'].min()
    tailcond = (df['x_cc']==df['x_cc'].min()) | (
            (df['x_cc']<xdaymin-1) & (df['h [R]']< flank_min_h))
    tail = df[tailcond]
    #Flank is everything else
    flank = df[(~ daycond) & (~ tailcond)]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(day['x_cc'], day['y_cc'], day['z_cc'], color='orange')
    ax.scatter(flank['x_cc'], flank['y_cc'], flank['z_cc'], color='cyan')
    ax.scatter(tail['x_cc'], tail['y_cc'], tail['z_cc'], color='red')
    #append name and time and sector
    day = day.append(name,ignore_index=True).append(
                   time,ignore_index=True).append(daytag,ignore_index=True)
    flank = flank.append(flanktag,ignore_index=True)
    tail=tail.append(name,ignore_index=True).append(
                  time,ignore_index=True).append(tailtag,ignore_index=True)
    #plt.show()
    return day, flank, tail

def add_derived_variables(dflist):
    """Function adds columns of data by performing simple operations
    Inputs
        dflist- dataframe
    Outputs
        dflist- dataframe with modifications
    """
    for df in enumerate(dflist):
        if not df[1].empty:
            ###Spatial variables
            dflist[df[0]]['r [R]'] =sqrt(df[1]['x_cc']**2+
                                         df[1]['y_cc']**2+df[1]['z_cc']**2)
            dflist[df[0]]['h [R]'] =sqrt(df[1]['y_cc']**2+df[1]['z_cc']**2)
            dflist[df[0]]['alpha [rad]'] = (arctan2(df[1]['z_cc'],
                                                           df[1]['y_cc']))
    return dflist



if __name__ == "__main__":
    PATH = ['temp/meshdata']
    OPATH = 'temp/'
    NALPHA = 36
    NSLICE = 60
    figureout = OPATH+'figures/'
    dflist = proc_temporal.read_energetics(PATH, add_variables=False)
    dflist = add_derived_variables(dflist)
    day, flank, tail = split_day_flank_tail(dflist[0])
    #set text settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    make_spatial_plots(day, flank, tail)
