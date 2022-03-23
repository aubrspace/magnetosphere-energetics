#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def pyplotsetup(*,mode='presentation',**kwargs):
    """Creates dictionary to send to rcParams for pyplot defaults
    Inputs
        mode(str)- default is "presentation"
        kwargs:
    """
    #Always
    settings={"text.usetex": True,
              "font.family": "sans-serif",
              "font.size": 18,
              "font.sans-serif": ["Helvetica"]}
    if 'presentation' in mode:
        #increase lineweights
        settings.update({'lines.linewidth': 3})
    if 'print' in mode:
        #Change colorcycler
        colorwheel = plt.cycler('color',
                ['tab:blue', 'tab:orange', 'tab:pink', 'tab:brown',
                 'tab:olive','tab:cyan'])
        settings.update({'axes.prop_cycle': colorwheel})
    elif 'digital' in mode:
        #make non white backgrounds, adjust borders accordingly
        settings.update({'axes.edgecolor': 'white',
                         'axes.labelcolor': 'white',
                         'axes.facecolor': '#375e95',
                         'figure.facecolor':'#375e95',
                         'text.color':'white',
                         'ytick.color':'white',
                         'xtick.color':'white'})
        #Change colorcycler
        colorwheel = plt.cycler('color',
                   ['#FAFFB0', 'magenta', 'peru', 'chartreuse', 'wheat',
                    'lightgrey', 'springgreen', 'coral', 'plum', 'salmon'])
        settings.update({'axes.prop_cycle': colorwheel})
    return settings

def general_plot_settings(ax, **kwargs):
    """Sets a bunch of general settings
    Inputs
        ax
        kwargs:
            do_xlabel(boolean)- default False
            color(str)- see matplotlib named colors
            legend_loc(see pyplot)- 'upper right' etc
            iscontour(boolean)- default False
    """
    #Xlabel
    if not kwargs.get('iscontour',False):
        if kwargs.get('do_xlabel',False):
            ax.set_xlabel(r'Time $\left[ UTC\right]$')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    else:
        ax.set_xlim(kwargs.get('xlim',None))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(kwargs.get('xlabel',''))
    #Ylabel
    ax.set_ylim(kwargs.get('ylim',None))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel(kwargs.get('ylabel',''))
    ax.tick_params(which='major', length=9)
    ax.legend(loc=kwargs.get('legend_loc',None))


if __name__ == "__main__":
    print('this module only contains helper functions and other useful '+
          'things for making plots!')
