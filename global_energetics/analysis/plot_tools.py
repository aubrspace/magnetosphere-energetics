#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import numpy as np
import datetime as dt
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

def safelabel(label):
    """Returns str that will not break latex format
    Input
        label
    Return
        label
    """
    culprits = ['_','%']#ones that show up often
    for c in culprits:
        label = ('\\'+c).join(label.split(c))
    return label

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

def get_omni_cdas(start,end,as_df=True,**variables):
    from cdasws import CdasWs
    import pandas as pd
    cdas = CdasWs()
    omni_allvars = cdas.get_variables('OMNI_HRO_1MIN')
    omni_vars = ['SYM_H']
    for var in variables:
        if var in omni_allvars:
            omni_vars.append(var)
    status,omni = cdas.get_data('OMNI_HRO_1MIN',omni_vars,start,end)
    if as_df:
        df = pd.DataFrame(omni['SYM_H'], columns=['symh'],
                              index=omni['Epoch'])
        return df
    else:
        return omni

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

def mark_times(axis):
    """Function makes vertical marks at specified time stamps
    Inputs
        axis- object to mark
    """
    #FROM Tuija Email June 20, 2021:
    '''
    % IMF changes
02 19 10 20
02 19 17 45
% Density peaks 18/12-17 UT, 19/0940-1020 UT, 19/1250-1440
02 18 12 00
02 18 17 00
02 19 09 00
02 19 10 20
02 19 12 50
02 19 14 40
% AL: substorm onsets 18/1430, 18/1615, 18/1850, 19/0030, 19/0356, 19/0900, 19/1255
02 18 14 30
02 18 16 10
02 18 18 50
02 19 00 30
02 19 03 56
02 19 09 00
02 19 12 55
    '''
    timeslist=[]
    #IMF changes
    timeslist.append([dt.datetime(2014,2,19,10,20),'IMF'])
    timeslist.append([dt.datetime(2014,2,19,17,45),'IMF'])
    #Density peaks
    timeslist.append([dt.datetime(2014,2,18,12,0),'Density'])
    timeslist.append([dt.datetime(2014,2,18,17,0),'Density'])
    timeslist.append([dt.datetime(2014,2,19,9,0),'Density'])
    timeslist.append([dt.datetime(2014,2,19,10,20),'Density'])
    timeslist.append([dt.datetime(2014,2,19,12,50),'Density'])
    timeslist.append([dt.datetime(2014,2,19,14,40),'Density'])
    #Substorm onsets based on AL
    timeslist.append([dt.datetime(2014,2,18,14,30),'Substorm'])
    timeslist.append([dt.datetime(2014,2,18,16,10),'Substorm'])
    timeslist.append([dt.datetime(2014,2,18,18,50),'Substorm'])
    timeslist.append([dt.datetime(2014,2,19,0,30),'Substorm'])
    timeslist.append([dt.datetime(2014,2,19,3,56),'Substorm'])
    timeslist.append([dt.datetime(2014,2,19,9,0),'Substorm'])
    timeslist.append([dt.datetime(2014,2,19,12,55),'Substorm'])
    #Colors, IMF, Density, Substorm
    colorwheel = dict({'IMF':'black',
                       'Density':'black',
                       'Substorm':'black'})
    lswheel = dict({'IMF':None,
                       'Density':None,
                       'Substorm':'--'})
    for stamp in timeslist:
        axis.axvline(stamp[0], color=colorwheel[stamp[1]],
                     linestyle=lswheel[stamp[1]], linewidth=1)


if __name__ == "__main__":
    print('this module only contains helper functions and other useful '+
          'things for making plots!')
