#!/usr/bin/env python3
"""Functions for handling and plotting 2D meshes embeded in 3D space
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
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import swmfpy
#interpackage imports
from global_energetics.analysis.proc_virial import (pyplotsetup,
                                                    general_plot_settings)

def binData(xval, yval, zval, area, **kwargs):
    """Function bins data in prep for plotting nice contour plot
    Inputs
        data(DataFrame)
        xdim, ydim, zval- series for spatial dimensions and value for plot
        area- series for cell area
        kwargs:
            nbin
            limX
            limY
    Returns
        Xplot, Yplot, Zplot- arrays to plot in contourf
    """
    nbin = kwargs.get('nbin',10)
    limX = kwargs.get('limX', [-pi,pi])
    limY = kwargs.get('limY', [-pi,pi])
    xtick, dx = np.linspace(limX[0]*(1-1/nbin), limX[1]*(1-1/nbin),
                            nbin, retstep=True)
    ytick, dy = np.linspace(limY[0]*(1-1/nbin), limY[1]*(1-1/nbin),
                            nbin, retstep=True)
    bin_area, bin_raw, bin_z = [], [], []
    for x in xtick:
        for y in ytick:
            bin_area = area[(xval>x-dx/2) & (xval<x+dx/2) &
                            (yval>y-dy/2) & (yval<y+dy/2)]
            bin_value = zval[(xval>x-dx/2) & (xval<x+dx/2) &
                             (yval>y-dy/2) & (yval<y+dy/2)]
            bin_z.append((bin_area*bin_value).sum())
    Xplot,Yplot = np.meshgrid(xtick,ytick)
    Zplot = np.reshape(bin_z, [len(xtick),len(ytick)])
    return Xplot, Yplot, Zplot

def plot_2Dcontour(ax, data,xdim, ydim, valuekey, **kwargs):
    """Function creates a contour plot on given axis with dims and value
    Inputs
        ax- pyplot axis object
        xdim,ydim- x and y axis variables used to create bins
        valuekey- variable used to display value as countor level
        kwargs:
            levels
            xlim, ylim
            ylabel, xlabel
            legend_loc
            axtitle
            see binData for binning settings
    """
    #Get title from value key
    if kwargs.get('title') is None:
        text, units = valuekey.split('[')
        title=r'${'+text+r'}\left['+units.split(']')[0]+r'\right]$'
    X,Y,Z = binData(data[xdim],data[ydim],data[valuekey],
                    data['Cell Volume'], **kwargs)
    con = ax.contourf(X, Y, Z, levels=kwargs.get('levels'))
    ax.scatter(X,Y)
    kwargs.update({'iscontour':True})
    general_plot_settings(ax,**kwargs)
    ax.get_legend().remove()
    #ax.grid(which='both')
    fig = plt.gcf()
    fig.colorbar(con, ax=ax)
    ax.set_title(kwargs.get('axtitle',title))

def get_spatial_vars(df):
    """Function creates useful spatial variables useful for displaying data
    Inputs
        data(dict{str:DataFrame})
    Returns
        newdata(dict{str:DataFrame})
    """
    #axis angles
    df['aXY'] = [np.arctan2(y,x) for (x,y) in df[['x_cc','y_cc']].values]
    df['aXZ'] = [np.arctan2(z,x) for (x,z) in df[['x_cc','z_cc']].values]
    df['aYZ'] = [np.arctan2(y,z) for (y,z) in df[['y_cc','z_cc']].values]
    return df

def proc_data(keypair):
    """Function modifies data for display
    Inputs
        keypair((str,DataFrame))- one item from data dictionary
    Returns
        updated(dict{str:DataFrame})- modified values put in dict form
    """
    name, data = keypair
    data = get_spatial_vars(data)
    return {name:data}

def read_meshfile(filename):
    """Reads hdf5 file and returns timestamp and dictionary of DataFrames
    Inputs
        filename
    Returns
        timestamp(datetime.datetime)
        data(dict{str:DataFrame})
    """
    data = {}
    with pd.HDFStore(filename) as hdf:
        for key in hdf.keys():
            #Get time
            if 'time' in key.lower():
                storetime = hdf['/Time [UTC]'].loc[0].values.astype(
                                                               dt.datetime)
                timestamp = dt.datetime.fromtimestamp(storetime[0]/1e9)
            else:
                name = key.split('/')[-1]
                data.update({name:hdf[key]})
    return timestamp, data

if __name__ == "__main__":
    datapath = sys.argv[-1]
    figureout = '/'.join(datapath.split('/')[0:-1])+'/figures/mesh/'
    plt.rcParams.update(pyplotsetup(mode='digital_presentation'))
    filelist = glob.glob(datapath+'/*.h5')
    timestamp, data = read_meshfile(filelist[0])
    for keypair in data.items():
        proc_data(keypair)
    mp = data['mp_iso_betastar']
    ######First figure: example contour plot
    plt.rcParams.update({'image.cmap':'magma'})
    XZlabel=r'$\angle_{XZ}\left[ Rad\right]$'
    XYlabel=r'$\angle_{XY}\left[ Rad\right]$'
    YZlabel=r'$\angle_{YZ}\left[ Rad\right]$'
    Xlabel=r'$X\left[ R_e\right]$'
    ExBtitle = r'$ExB_{net} \left[ W\right]$'
    Virialtitle = r'Virial Total $\left[ J\right]$'
    Fadv_title = r'Virial Advection $\left[ J\right]$'
    Floz_title = r'Virial Lorentz $\left[ J\right]$'
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=[24,8])
    plot_2Dcontour(ax[0], mp, 'aXY', 'aXZ', 'virial_surfTotal [J/Re^2]',
                   xlabel=XYlabel, ylabel=XZlabel, timestamp=timestamp,
                   axtitle=Virialtitle,limX=[-pi/2,pi/2],limY=[-pi/2,pi/2],
                   levels=np.linspace(-1.5e14,2e13,16))
    plot_2Dcontour(ax[1], mp, 'x_cc', 'aYZ', 'virial_surfTotal [J/Re^2]',
                   ylabel=YZlabel, xlabel=Xlabel, timestamp=timestamp,
                   axtitle=Fadv_title,limX=[-20,0],limY=[-pi,pi],
                   levels=np.linspace(-1.5e14,2e13,16))
    plot_2Dcontour(ax[2], mp, 'x_cc', 'aYZ', 'virial_surfTotal [J/Re^2]',
                   ylabel=YZlabel, xlabel=Xlabel, timestamp=timestamp,
                   axtitle=Floz_title,limX=[-20,0],limY=[-pi,pi],
                   levels=np.linspace(-1.5e14,2e13,11))
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'contour_example.png')
    plt.close(fig)
    ######Second figure: plot 1* cell area
    mp['const []'] = 1
    plt.rcParams.update({'image.cmap':'YlGn_r'})
    XZlabel=r'$\angle_{XZ}\left[ Rad\right]$'
    XYlabel=r'$\angle_{XY}\left[ Rad\right]$'
    YZlabel=r'$\angle_{YZ}\left[ Rad\right]$'
    Xlabel=r'$X\left[ R_e\right]$'
    title = r'Area $\left[ R_e^2\right]$'
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,6])
    plot_2Dcontour(ax, mp, 'x_cc', 'aYZ', 'const []',
                   ylabel=Xlabel, xlabel=YZlabel, timestamp=timestamp,
                   axtitle=title, limX=[-20,0],limY=[-pi,pi])
    fig.tight_layout(pad=1)
    fig.savefig(figureout+'contour_area.png')
    plt.close(fig)
