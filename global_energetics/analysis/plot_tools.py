#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import numpy as np
import datetime as dt
from scipy import signal
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
              "font.size": 28,
              "font.sans-serif": ["Helvetica"]}
    if 'presentation' in mode:
        #increase lineweights
        settings.update({'lines.linewidth': 3})
    if 'print' in mode:
        #Change colorcycler
        colorwheel = plt.cycler('color',
                ['maroon', 'magenta', 'tab:blue', 'green',
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
    elif 'solar' in mode:
        #make non white backgrounds, adjust borders accordingly
        settings.update({'axes.edgecolor': 'white',
                         'axes.labelcolor': 'white',
                         'axes.facecolor': '#788091',
                         'figure.facecolor':'#788091',
                         'text.color':'white',
                         'ytick.color':'white',
                         'xtick.color':'white'})
        #Change colorcycler
        colorwheel = plt.cycler('color',
                   ['gold', 'aqua', 'salmon', 'darkred', 'wheat',
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
            ax.set_xlabel(kwargs.get('xlabel',r'Time $\left[ UTC\right]$'))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
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
    if kwargs.get('legend',True):
        ax.legend(loc=kwargs.get('legend_loc',None))

def get_omni_cdas(start,end,as_df=True,**variables):
    from cdasws import CdasWs
    import pandas as pd
    if type(start)==pd._libs.tslibs.timestamps.Timestamp:
        start = dt.datetime.fromtimestamp(start.timestamp())
        end = dt.datetime.fromtimestamp(end.timestamp())
    cdas = CdasWs()
    omni_allvars = cdas.get_variables('OMNI_HRO_1MIN')
    omni_vars = ['SYM_H']
    for var in variables:
        if var in omni_allvars:
            omni_vars.append(var)
    status,omni = cdas.get_data('OMNI_HRO_1MIN',omni_vars,start,end)
    if as_df:
        df = pd.DataFrame(omni['SYM_H'], columns=['sym_h'],
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

def plot_stack_distr(ax, times, mp, msdict, **kwargs):
    """Plots distribution of energies in particular zone
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
            legend_loc(see pyplot)- 'upper right' etc
            subzone(str)- 'ms_full' is default
            value_set(str)- 'Virialvolume', 'Energy', etc
            ylim(tuple or list)- None
    """
    #Figure out value_keys and subzone
    subzone = kwargs.get('subzone', 'ms_full')
    kwargs.update({'legend_loc':kwargs.get('legend_loc','lower left')})
    if 'ms_full' in subzone:
        data = mp
    else:
        data = msdict[subzone]
    value_set = kwargs.get('value_set','Virial')
    values = {'Virial':['Virial 2x Uk [nT]', 'Virial Ub [nT]','Um [nT]',
                        'Virial Surface Total [nT]'],
              'Energy':['Virial Ub [J]', 'KE [J]', 'Eth [J]'],
              'Energy2':['u_db [J]', 'uHydro [J]'],
              'Report':['Utot2 [J]', 'Utot [J]','uB [J]','uB_dipole [J]',
                        'u_db [J]', 'uHydro [J]'],
              'Virial%':['Virial 2x Uk [%]', 'Virial Ub [%]',
                               'Virial Surface Total [%]'],
              'Energy%':['Virial Ub [%]', 'KE [%]', 'Eth [%]']}
    #Get % within specific subvolume
    if '%' in value_set:
        kwargs.update({'ylim':kwargs.get('ylim',[0,100])})
        for value in values[''.join(value_set.split('%'))]:
                #find total
                if 'Virial' in value_set:
                    total = data['Virial [nT]']
                elif 'Energy' in value_set:
                    total = (data['Utot [J]']-data['uB [J]']+
                             data['Virial Ub [J]'])
                data[value.split('[')[0]+'[%]'] = data[value]/total*100
    #Optional layers depending on value_key
    starting_value = 0
    if (not '%' in value_set) and ('Virial' in value_set):
        ax.plot(times, data['Virial Surface Total [nT]'], color='#05BC54',
                linewidth=4,label='Boundary Stress')#light green color
        starting_value = data['Virial Surface Total [nT]']
        values[value_set].remove('Virial Surface Total [nT]')
        ax.axhline(0,color='white',linewidth=0.5)
    if kwargs.get('dolog',False):
        for value in values[value_set]:
            label = value.split(' [')[0].split('Virial ')[-1]
            ax.semilogy(times,data[value],label=safelabel(label))
    else:
        for value in values[value_set]:
            label = value.split(' [')[0].split('Virial ')[-1]
            ax.fill_between(times,starting_value,starting_value+data[value],
                            label=safelabel(label))
            starting_value = starting_value+data[value]
    if (not '%' in value_set) and kwargs.get('doBios',True):
        if any(['bioS' in k for k in data.keys()]):
            ax.plot(times, data['bioS [nT]'], color='white', ls='--',
                    label='BiotSavart')
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_stack_contrib(ax, times, mp, msdict, **kwargs):
    """Plots contribution of subzone to virial Dst
    Inputs
        mp(DataFrame)- total magnetosphere values
        msdict(Dict of DataFrames)- subzone values
        kwargs:
            do_xlabel(boolean)- default False
            ylabel(str)- default ''
            legend_loc(see pyplot)- 'upper right' etc
            omni
    """
    #Figure out value_key
    value_key = kwargs.get('value_key','Virial [nT]')
    #Optional layers depending on value_key
    starting_value = 0
    if not '%' in value_key:
        ax.axhline(0,color='white',linewidth=0.5)
        if ('Virial' in value_key) or ('bioS' in value_key):
            if 'omni' in kwargs:
                ax.plot(kwargs.get('omni')['Time [UTC]'],
                        kwargs.get('omni')['sym_h'],color='white',
                        ls='--', label='SYM-H')
            else:
                print('omni not loaded! No obs comparisons!')
            '''
            try:
                if all(omni['sym_h'].isna()): raise NameError
                ax.plot(omni['Time [UTC]'],omni['sym_h'],color='white',
                    ls='--', label='SYM-H')
            except NameError:
                print('omni not loaded! No obs comparisons!')
            '''
        if ('bioS' in value_key) and ('bioS_ext [nT]' in mp.keys()):
            starting_value = mp['bioS_ext [nT]']
            starting_value = starting_value+ mp['bioS_int [nT]']
            ax.plot(times, mp['bioS_ext [nT]']+mp['bioS_int [nT]'],
                    color='#05BC54',linewidth=4,
                    label='External+Internal')#light green color
    #Plot stacks
    for szlabel,sz in msdict.items():
        szval = sz[value_key]
        #times, d = times[~d.isna()], d[~d.isna()]
        if kwargs.get('dolog',False):
            ax.semilogy(times,szval,label=szlabel)
        else:
            ax.fill_between(times,starting_value/kwargs.get('factor',1),
                              (starting_value+szval)/kwargs.get('factor',1),
                        label=szlabel,hatch=kwargs.get('hatch'))
            starting_value = starting_value+szval
    ax.set_xlim([times[0],times[-1]])
    #Optional plot settings
    if ('bioS' in value_key) and ('%' in value_key):
        ax.set_ylim([-100,100])
    #General plot settings
    general_plot_settings(ax, **kwargs)

def plot_psd(ax, t, series, **kwargs):
    """plots estimated power spectral density from scipy.signal.periodogram
    Inputs
        ax (Axis object)- axis to plot on
        t (timeIndex)- timeseries used, only tested with equally spaced
        series array(floats)- signal data used
        kwargs:
            fs (float)- sampling frequency, will scale results
            label-
            ylim,xlabel,ylabel
    Returns
        None
    """
    f, Pxx = signal.periodogram(series,fs=kwargs.get('fs',1/300),nfft=3000)
    T_peak = (1/f[Pxx==Pxx.max()][0])/3600 #period in hours
    ax.semilogy(f*1e3,Pxx,label=kwargs.get('label',r'Virial $\Delta B$')
                                    +' Peak at  {:.2f} hrs'.format(T_peak))
    ax.set_xlim(kwargs.get('xlim',[0,0.5]))
    ax.set_ylim(kwargs.get('ylim',[1e-6*Pxx.max(),10*Pxx.max()]))
    ax.legend()
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))

def plot_pearson_r(ax, tx, ty, xseries, yseries, **kwargs):
    """plots scatter with attached r correlation value for XY series
    Inputs
        ax (Axis object)- axis to plot on
        tx,ty (timeIndex)- timeseries used, only tested with equally spaced
        xseries,yseries array(floats)- signal data used
        kwargs:
            label-
            ylim,xlabel,ylabel
    Returns
        None
    """
    #Pearson R Correlation
    xdata = np.interp(ty, tx, xseries)
    ydata = yseries
    cov = np.cov(np.stack((xdata,ydata)))[0][1]
    r = cov/(xdata.std()*ydata.std())
    #Plot with SW on X and Virial on Y
    ax.scatter(xdata,ydata,label='r = {:.2f}'.format(r))
    ax.legend()
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))


if __name__ == "__main__":
    print('this module only contains helper functions and other useful '+
          'things for making plots!')
