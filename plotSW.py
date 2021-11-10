#/usr/bin/env python
"""Plots solar wind from IMF.dat file for checking
"""
import pandas as pd
import numpy as np
import os
import sys
import time
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import swmfpy
from swmfpy.io import read_imf_input

def plot_Temp(ax, data, ylabel, **kwargs):
    """plots B components
    Inputs
    """
    for key in ['times', 'temperature']:
        if key not in data.keys():
            raise KeyError (key+'not in dataset, check column names!')
    #plot components
    ax.plot(data['times'],data['temperature'],color='tomato')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    if kwargs.get('do_xlabel',False):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_Rho(ax, data, ylabel, **kwargs):
    """plots Rho
    Inputs
    """
    for key in ['times', 'density']:
        if key not in data.keys():
            raise KeyError (key+'not in dataset, check column names!')
    #plot components
    ax.plot(data['times'],data['density'],color='green')
    ax.axhline(0,color='black')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    if kwargs.get('do_xlabel',False):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_V(ax, data, ylabel, **kwargs):
    """plots B components
    Inputs
    """
    v = pd.DataFrame()
    for key in ['times', 'vx', 'vy', 'vz']:
        if key not in data.keys():
            raise KeyError (key+'not in dataset, check column names!')
        v[key]=data[key]
    #plot components
    colors = {'vx':'red','vy':'gold','vz':'plum'}
    for comp in enumerate(['vz', 'vy', 'vx']):
        ax.plot(v['times'],v[comp[1]],color=colors[comp[1]],
                label=comp[1])
    ax.axhline(0,color='black')
    ax.axhline(-400,color='black',ls='--')
    ax.set_ylabel(ylabel)
    ax.set_ylim([-1200,300])
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_B(ax, data, ylabel, **kwargs):
    """plots B components
    Inputs
    """
    b = pd.DataFrame()
    for key in ['times', 'bx', 'by', 'bz']:
        if key not in data.keys():
            raise KeyError (key+'not in dataset, check column names!')
        b[key]=data[key]
    ax.fill_between(data['times'],np.sqrt(b['bx']**2+b['by']**2+b['bz']**2)
                    ,color='grey')
    ax.axhline(0,color='black')
    colors = {'bx':'blue','by':'magenta','bz':'cyan'}
    for comp in ['bx', 'by', 'bz']:
        ax.plot(data['times'], b[comp], color=colors[comp], label=comp)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_solarwind(*,imffile='IMF.dat', save=False):
    """plots solar wind from IMF.dat
    """
    imf = read_imf_input(filename=imffile)
    y1label = r'$B \left(nT\right)$'
    y2label = r'$V \left(km/s\right)$'
    y3label = r'$\rho \left(amu/cc\right)$'
    y4label = r'$T \left(K\right)$'
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=[24,8])
    plot_B(ax[0], imf, y1label)
    plot_V(ax[1], imf, y2label)
    plot_Rho(ax[2], imf, y3label)
    plot_Temp(ax[3], imf, y4label)

#Main program
if __name__ == '__main__':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    plot_solarwind()
    plt.show()
