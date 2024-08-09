#/usr/bin/env python
"""Plots solar wind from IMF.dat file for checking
"""
import os,sys,glob,time
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator,AutoLocator,AutoMinorLocator)
import swmfpy
from swmfpy.io import read_imf_input, gather_times

def plot_Temp(ax, data, ylabel, **kwargs):
    """plots B components
    Inputs
    """
    for key in ['times', 'temp']:
        if key not in data.keys():
            if any(['temp' in k for k in data.keys()]):
                data['temp'] = data['temperature']
            else:
                raise KeyError (key+'not in dataset, check column names!')
    #plot components
    ax.plot(data['times'],data['temp'],color='tomato')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    #ax.xaxis.set_major_locator(AutoLocator(12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_Rho(ax, data, ylabel, **kwargs):
    """plots Rho
    Inputs
    """
    for key in ['times', 'dens']:
        if key not in data.keys():
            if any(['dens' in k for k in data.keys()]):
                data['dens'] = data['density']
            else:
                raise KeyError (key+'not in dataset, check column names!')
    #plot components
    ax.plot(data['times'],data['dens'],color='green')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if kwargs.get('do_xlabel',False):
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
    ax.axhline(0,color='grey')
    ax.axhline(-400,color='black',ls='--')
    ax.set_ylabel(ylabel)
    ax.set_ylim([-700,200])
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
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
    ax.axhline(0,color='grey')
    colors = {'bx':'blue','by':'magenta','bz':'cyan'}
    for comp in ['bx', 'by', 'bz']:
        ax.plot(data['times'], b[comp], color=colors[comp], label=comp)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_symh(ax, omni, ylabel, **kwargs):
    """plots symh from omni
    Inputs
    """
    for key in ['times', 'sym_h']:
        if key not in omni.keys():
            raise KeyError (key+' not in dataset, check column names!')
    ax.plot(omni['times'], omni['sym_h'], color='magenta', label='SYM-H')
    ax.axhline(0,color='grey')
    ax.axhline(-50,color='black',ls='--')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_al(ax, omni, ylabel, **kwargs):
    """plots al from omni
    Inputs
    """
    for key in ['times', 'al']:
        if key not in omni.keys():
            raise KeyError (key+' not in dataset, check column names!')
    ax.plot(omni['times'], omni['al'], color='darkmagenta', label='AL')
    ax.set_ylabel(ylabel)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc=kwargs.get('legend_loc',None))
    if kwargs.get('do_xlabel',False):
        ax.set_xlabel(kwargs.get('xlabel',None))

def plot_satellite(ax, satname, data, labels, **kwargs):
    """plots satellite locations in GSM XZ and XY planes
    Inputs
        axes list(axis,axis)- axis objects to plot on
        data (DataFrame)- data to plot
        labels (str,str)- labels for XZ and XY planes respectively
        kwargs:
    """
    #XZ plot
    ax[0].set_aspect('equal')
    ax[0].axvline(0,color='grey')
    ax[0].axhline(0,color='grey')
    ax[0].scatter(data['x'],data['z'],label=satname)
    ax[0].scatter(data['x'][0],data['z'][0],color='black',marker='^')
    ax[0].scatter(data['x'][-1],data['z'][-1],color='black',marker='o')
    ax[0].set_xlim(-50,30); ax[0].set_ylim(-40,40); ax[0].invert_xaxis()
    ax[0].set_xlabel(r'$X \left(R_e\right)$'); ax[0].set_ylabel(labels[0])
    plot_earth(ax[0],color='blue')
    #XY plot
    ax[1].set_aspect('equal')
    ax[1].axvline(0,color='grey')
    ax[1].axhline(0,color='grey')
    ax[1].scatter(data['x'],data['y'],label=satname)
    ax[1].scatter(data['x'][0],data['y'][0],color='black',marker='^')
    ax[1].scatter(data['x'][-1],data['y'][-1],color='black',marker='o')
    ax[1].set_xlim(-50,30); ax[1].set_ylim(-40,40); ax[1].invert_xaxis()
    ax[1].set_xlabel(r'$X \left(R_e\right)$'); ax[1].set_ylabel(labels[1])
    plot_earth(ax[1],color='blue')
    ax[1].legend()

def plot_earth(ax, color):
    circle = plt.Circle((0,0),1,color=color)
    ax.add_patch(circle)

def plot_solarwind(*,imffile='IMF.dat', check_omni=False,
                     check_sattelites=False, save=False, **kwargs):
    """plots solar wind from IMF.dat
    """
    satlist = kwargs.get('SATLIST',['THEMIS','CLUSTER','MMS','GEOTAIL'])
    imf = kwargs.get('WIND',read_imf_input(filename=imffile))
    if 'times' not in imf.keys():
        imf_dic = {}
        for key in imf.keys():
            imf_dic[key] = imf[key].values
        imf_dic['times'] = gather_times(imf_dic,**kwargs)
        imf = imf_dic
    xlabel =  r'Time $\left[ UTC\right]$'
    y1label = r'$B \left(nT\right)$'
    y2label = r'$V \left(km/s\right)$'
    y3label = r'$\rho \left(amu/cc\right)$'
    y4label = r'$T \left(K\right)$'
    rows=4;cols=1
    if check_omni or ('OMNI' in kwargs):
        omni =kwargs.get('OMNI',swmfpy.web.get_omni_data(imf['times'][0],
                                                         imf['times'][-1]))
        y5label = r'$\Delta B\left(nT\right)$'
        y6label = r'$\Delta B\left(nT\right)$'
        rows+=2
        doX = False
    if any([s in kwargs for s in satlist]):
        y7label = r'$Z \left(R_e\right)$'
        y8label = r'$Y \left(R_e\right)$'
        rows+=2;cols+=1
    if cols==1:
        fig,ax = plt.subplots(nrows=rows, ncols=1, sharex=True,
                              figsize=[14,2*rows])
        doX = False
    else:
        ax = []
        plt.figure(figsize=[14,2*rows])
        for i in range(1,rows-1):
            ax.append(plt.subplot(rows,1,i))
        ax.append(plt.subplot(rows/2,2,rows-1))
        ax.append(plt.subplot(rows/2,2,rows))
    plot_B(ax[0], imf, y1label,do_xlabel=doX)
    plot_V(ax[1], imf, y2label,do_xlabel=doX)
    plot_Rho(ax[2], imf, y3label,do_xlabel=doX)
    plot_Temp(ax[3], imf, y4label,do_xlabel=doX)
    if check_omni or ('OMNI' in kwargs):
        plot_symh(ax[4], omni, y5label,do_xlabel=doX)
        plot_al(ax[5], omni, y6label,do_xlabel=True, xlabel=xlabel)
    #if not doX:
    #    ax[-1].set_xlabel(r'Time $\left[ UTC\right]$')
    #    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
    #    ax[-1].xaxis.set_minor_locator(AutoMinorLocator(6))
    #Plot satellites
    if any([s in kwargs for s in satlist]):
        for s in satlist:
            if s in kwargs:
                plot_satellite(ax[6:8],s,kwargs[s],[y7label,y8label])

#Main program
if __name__ == '__main__':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    if '-i' in sys.argv:
        infile = sys.argv[sys.argv.index('-i')+1]
        plot_solarwind(imffile=infile)
    else:
        plot_solarwind()
    plt.show()
