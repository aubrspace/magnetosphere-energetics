#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import numpy as np
import datetime as dt
import scipy
from scipy import signal
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,
                               AutoLocator, FuncFormatter)

def central_diff(dataframe,**kwargs):
    """Takes central difference of the columns of a dataframe
    Inputs
        df (DataFrame)- data
        dt (int)- spacing used for denominator
        kwargs:
            fill (float)- fill value for ends of diff
    Returns
        cdiff (DataFrame)
    """
    times = dataframe.copy(deep=True).index
    df = dataframe.copy(deep=True)
    df = df.reset_index(drop=True).ffill()
    df_fwd = df.copy(deep=True)
    df_bck = df.copy(deep=True)
    df_fwd.index -= 1
    df_bck.index += 1
    if kwargs.get('forward',False):
        # Calculate dt at each time interval
        dt = times[1::]-times[0:-1]
        cdiff = (df_fwd-df)/(dt.seconds+dt.microseconds/1e6)
        cdiff.drop(index=[-1],inplace=True)
    else:
        # Calculate dt at each time interval
        dt = times[2::]-times[0:-2]
        diff = (df_fwd-df_bck).drop(index=[-1,0,df_bck.index[-1],
                                                df_bck.index[-2]])
        cdiff = diff/(dt.seconds+dt.microseconds/1e6)
        cdiff.loc[0] = 0
        cdiff.loc[len(cdiff)]=0
        cdiff.sort_index(inplace=True)
    cdiff.index = dataframe.index
    return cdiff

def pyplotsetup(*,mode='presentation',**kwargs):
    """Creates dictionary to send to rcParams for pyplot defaults
    Inputs
        mode(str)- default is "presentation"
        kwargs:
    """
    #Always
    settings={"text.usetex": False,
              "font.family": "sans-serif",
              "font.size": 28}
    #         "font.sans-serif": ["Helvetica"]}
    if 'presentation' in mode:
        #increase lineweights
        settings.update({'lines.linewidth': 3})
    if 'print' in mode:
        #Change colorcycler
        colorwheel = plt.cycler('color',
                ['orange', 'blue', 'tab:blue', 'green',
                 #['maroon', 'magenta', 'tab:blue', 'green',
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
    #TODO: is the missing xlabel in here?
    #Xlabel
    if kwargs.get('iscontour',False):
        ax.set_xlim(kwargs.get('xlim',None))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if kwargs.get('do_xlabel',True):
            ax.set_xlabel(kwargs.get('xlabel',''))
    if kwargs.get('timedelta',False):
        tmin,tmax = ax.get_xlim()
        def timedelta_hours(x,pos):
            if type(x)==type(dt.timedelta):
                hrs = x.days*24+t1.seconds/3600
            else:
                hrs = x/(1e9*3600)
            hours = int(hrs)
            #minutes = (int(abs(hrs*60)%60))
            return "{:d}".format(hours)
        def timedelta_hour_min(x,pos):
            if type(x)==type(dt.timedelta):
                hrs = x.days*24+t1.seconds/3600
            else:
                hrs = x/(1e9*3600)
            hours = int(hrs)
            minutes = (int(abs(hrs*60)%60))
            #from IPython import embed; embed()
            #import time
            #time.sleep(3)
            return "{:d}:{:02}".format(hours,minutes)
        #Get original limits
        ax.set_xlim(kwargs.get('xlim',None))
        xlims = ax.get_xlim()
        islong = (xlims[1]-xlims[0])*1e-9/3600 > 10
        ismed = (xlims[1]-xlims[0])*1e-9/3600 > 2
        if islong:
            #Manually adjust the xticks
            n = 3600*4/1e-9
            locs = [i*n for i in range(-100,100)]
        elif ismed:
            n = 3600*0.5/1e-9
            locs = [i*n for i in range(-100,100)]
        else:
            n = 3600/60/1e-9
            locs = [i*n for i in range(-2000,2000)]
        locs = [np.round(l/1e10)*1e10 for l in locs]
        ax.xaxis.set_ticks(locs)
        if islong:
            formatter = FuncFormatter(timedelta_hours)
        else:
            formatter = FuncFormatter(timedelta_hour_min)
        ax.xaxis.set_major_formatter(formatter)
        #ax.xaxis.set_minor_locator(AutoMinorLocator(3))
        if islong:
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        #Get original limits
        ax.set_xlim(xlims)
        if kwargs.get('do_xlabel',True):
            ax.set_xlabel(
                    #kwargs.get('xlabel',r'Time $\left[ hr:min\right]$'))
                    kwargs.get('xlabel',r'Time $\left[ hr\right]$'))
        if not kwargs.get('do_xlabel',True):
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
    else:
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
        ax.set_xlim(kwargs.get('xlim',None))
        tmin,tmax = ax.get_xlim()
        time_range = mdates.num2timedelta(tmax-tmin)
        if time_range>dt.timedelta(hours=6):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
        elif time_range<dt.timedelta(minutes=3):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        if kwargs.get('do_xlabel',False):
            ax.set_xlabel(kwargs.get('xlabel',r'Time $\left[ hr\right]$'))
        else:
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%-H:%M'))
            #ax.set_xlabel(
            #         kwargs.get('xlabel',r'Time $\left[ hr:Mn\right]$'))
            pass
        ax.xaxis.set_minor_locator(AutoMinorLocator(6))
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
              #'Energy2':['u_db [J]', 'uHydro [J]'],
              'Energy2':['uB [J]', 'uHydro [J]'],
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
    for i,(szlabel,sz) in enumerate([(l,z) for l,z in msdict.items()
                                     if ('closed_rc' not in l) and
                                        ('slice' not in l)]):
        print(szlabel)
        szval = sz[value_key]
        #NOTE changed just for this work
        #colorwheel = ['magenta', 'magenta', 'tab:blue']
        #colorwheel = ['maroon', 'magenta', 'tab:blue']
        colorwheel = ['magenta', 'tab:blue']
        labelwheel = ['Closed', 'Lobes']
        if szlabel=='rc':
            #kwargs['hatch'] ='x'
            #kwargs['edgecolor'] ='grey'
            pass
        else:
            kwargs['hatch'] =''
            kwargs['edgecolor'] =None
        #times, d = times[~d.isna()], d[~d.isna()]
        if kwargs.get('dolog',False):
            ax.semilogy(times,szval,label=szlabel)
        else:
            ax.fill_between(times,starting_value/kwargs.get('factor',1),
                              (starting_value+szval)/kwargs.get('factor',1),
                        label=labelwheel[i],hatch=kwargs.get('hatch'),
                            fc=colorwheel[i],
                            edgecolor=kwargs.get('edgecolor',None))
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
    f, Pxx = signal.periodogram(series,fs=kwargs.get('fs',1/300),
                                nfft=kwargs.get('nfft',None))
    #T_peak = (1/f[Pxx==Pxx.max()][0])/3600 #period in hours
    fmax = 1/kwargs.get('fs')
    T_peak = (1/f[Pxx==Pxx.max()][0]) #period in minutes
    #ax.semilogy(f*1e3,Pxx,label=kwargs.get('label',r'Virial $\Delta B$')
    #                                +' Peak at  {:.2f} hrs'.format(T_peak))
    ax.semilogy(f,Pxx,label=kwargs.get('label',r'Virial $\Delta B$')
                                    +' Peak at  {:.2f}min'.format(T_peak))
    #ax.set_xlim(kwargs.get('xlim',[0,0.5]))
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
            skipplot- (True)
    Returns
        correlation value
    """
    #Pearson R Correlation
    xdata = np.interp(ty, tx, xseries)
    ydata = yseries.copy(deep=True).fillna(method='bfill')
    r = np.corrcoef(xdata,ydata)[0][1]
    #s = scipy.stats.spearmanr(xdata,ydata).correlation
    if (not kwargs.get('skipplot',False)) and ax!=None:
        #Plot with SW on X and Virial on Y
        ax.scatter(xdata,ydata,label='r = {:.2f}'.format(r))
        ax.legend()
        ax.set_xlabel(kwargs.get('xlabel'))
        ax.set_ylabel(kwargs.get('ylabel'))
    return r

def bin_and_describe(X,Y,df,xbins,pLow,pHigh):
    """bins given dataframe according to X and Y, then returns some key
        statistics based on percentiles pLow and pHigh
    Inputs
        X
        xbins
        pLow,pHigh
    Returns
        Ydict
    """
    Ydict = {'pLow_all':np.array([]),
            'p50_all':np.array([]),
            'pHigh_all':np.array([]),
            'variance_all':np.array([]),
            'pLow_imf':np.array([]),
            'p50_imf':np.array([]),
            'pHigh_imf':np.array([]),
            'variance_imf':np.array([]),
            'pLow_sub':np.array([]),
            'p50_sub':np.array([]),
            'pHigh_sub':np.array([]),
            'variance_sub':np.array([]),
            'pLow_not':np.array([]),
            'p50_not':np.array([]),
            'pHigh_not':np.array([]),
            'variance_not':np.array([]),
            'nAll':np.array([]),
            'nIMF':np.array([]),
            'nSub':np.array([]),
            'nNot':np.array([])}
    for cbin in xbins:
        binwidth = (xbins[1]-xbins[0])*3
        bin_low = cbin-binwidth/2
        bin_high = cbin+binwidth/2

        # all data
        Y_all        = Y[(X<bin_high)&(X>bin_low)]
        pLow_all     = (Y_all).quantile(pLow)
        p50_all      = (Y_all).quantile(0.50)
        pHigh_all    = (Y_all).quantile(pHigh)
        variance_all = (Y_all).var()

        # load into new arrays
        Ydict['pLow_all']     = np.append(Ydict['pLow_all'],pLow_all)
        Ydict['p50_all']      = np.append(Ydict['p50_all'],p50_all)
        Ydict['pHigh_all']    = np.append(Ydict['pHigh_all'],pHigh_all)
        Ydict['variance_all'] = np.append(Ydict['variance_all'],variance_all)

        Ydict['nAll']     = np.append(Ydict['nAll'],len(Y_all))

        if 'IMF' in df.keys():
            # imf transits
            Y_imf        = Y[(X<bin_high)&(X>bin_low)&(df['IMF'])]
            pLow_imf     = (Y_imf).quantile(pLow)
            p50_imf      = (Y_imf).quantile(0.50)
            pHigh_imf    = (Y_imf).quantile(pHigh)
            variance_imf = (Y_imf).var()

            Ydict['pLow_imf']     = np.append(Ydict['pLow_imf'],pLow_imf)
            Ydict['p50_imf']      = np.append(Ydict['p50_imf'],p50_imf)
            Ydict['pHigh_imf']    = np.append(Ydict['pHigh_imf'],pHigh_imf)
            Ydict['variance_imf'] = np.append(Ydict['variance_imf'],
                                              variance_imf)

            Ydict['nIMF']     = np.append(Ydict['nIMF'],len(Y_imf))

        if 'anysubstorm' in df.keys():
            # substorm like
            Y_sub        = Y[(X<bin_high)&(X>bin_low)&(df['anysubstorm'])]
            pLow_sub     = (Y_sub).quantile(pLow)
            p50_sub      = (Y_sub).quantile(0.50)
            pHigh_sub    = (Y_sub).quantile(pHigh)
            variance_sub = (Y_sub).var()

            Ydict['pLow_sub']     = np.append(Ydict['pLow_sub'],pLow_sub)
            Ydict['p50_sub']      = np.append(Ydict['p50_sub'],p50_sub)
            Ydict['pHigh_sub']    = np.append(Ydict['pHigh_sub'],pHigh_sub)
            Ydict['variance_sub'] = np.append(Ydict['variance_sub'],
                                              variance_sub)

            Ydict['nSub']     = np.append(Ydict['nSub'],len(Y_sub))

        if 'anysubstorm' in df.keys() and 'IMF' in df.keys():
            # not substorm like
            Y_not        = Y[(X<bin_high)&(X>bin_low)&(1-df['anysubstorm'])&
                                                    (1-df['IMF'])]
            pLow_not     = (Y_not).quantile(pLow)
            p50_not      = (Y_not).quantile(0.50)
            pHigh_not    = (Y_not).quantile(pHigh)
            variance_not = (Y_not).var()

            Ydict['pLow_not']     = np.append(Ydict['pLow_not'],pLow_not)
            Ydict['p50_not']      = np.append(Ydict['p50_not'],p50_not)
            Ydict['pHigh_not']    = np.append(Ydict['pHigh_not'],pHigh_not)
            Ydict['variance_not'] = np.append(Ydict['variance_not'],
                                              variance_not)

            Ydict['nNot']     = np.append(Ydict['nNot'],len(Y_not))

    return Ydict

def extended_fill_between(ax,X,upper,lower,facecolor,alpha):
    # Extend the fill by duplicating the first and last points
    # we want to show how far the fill spans given centered bins...
    Xstart = [X[0] - (X[1]-X[0])/2]
    Xend   = [X[-1]+ (X[1]-X[0])/2]
    X_extend     = np.concatenate((Xstart,X,Xend))
    upper_extend = np.concatenate(([upper[0]],upper,[upper[-1]]))
    lower_extend = np.concatenate(([lower[0]],lower,[lower[-1]]))
    ax.fill_between(X_extend,lower_extend,upper_extend,
                    fc=facecolor,alpha=alpha)

def refactor(event,t0):
    # Gather segments of the event to pass directly to figure functions
    ev = {}
    # Only use data starting at 'tstart' where the sw is actually different
    tstart = t0+dt.timedelta(minutes=10)
    mp = event['mpdict']['ms_full'][event['mpdict']['ms_full'].index>tstart]
    lobes = event['msdict']['lobes'][event['msdict']['lobes'].index>tstart]
    closed = event['msdict']['closed'][event['msdict']['closed'].index>tstart]
    inner = event['inner_mp'][event['inner_mp'].index>tstart]
    ev['mp'] = mp.resample('60s').asfreq()
    use_i = ev['mp'].index
    ev['lobes'] = lobes.reindex(use_i).resample('60s').asfreq()
    ev['closed'] = closed.reindex(use_i).resample('60s').asfreq()
    ev['inner'] = inner.reindex(use_i).resample('60s').asfreq()
    '''
    #NOTE fill gaps S.T. values fill forward to keep const dt
    sample = str((event['mpdict']['ms_full'].index[-1]-
                  event['mpdict']['ms_full'].index[-2]).seconds)+'S'
    ev['mp'] = event['mpdict']['ms_full'].resample(sample).ffill()
    use_i = ev['mp'].index
    ev['lobes'] = event['msdict']['lobes'].reindex(use_i,method='ffill')
    ev['closed'] = event['msdict']['closed'].reindex(use_i,method='ffill')
    ev['inner'] = event['inner_mp'].reindex(use_i,method='ffill')
    #ev['lobes'] = event['msdict']['lobes'].resample('60s').ffill()
    #ev['closed'] = event['msdict']['closed'].resample('60s').ffill()
    '''
    times =  ev['mp'].index
    ev['rawtimes']=times
    timedelta = [t-t0 for t in times]
    ev['times']=[float(n.to_numpy()) for n in timedelta]
    if 'obs' in event.keys():
    #if False:
        ev['sim'] = event['obs']['swmf_log'].reindex(use_i)#,method='ffill')
        ev['sw'] = event['obs']['swmf_sw']
        #ev['sw'] = event['obs']['swmf_sw'].drop_duplicates().reindex(use_i)#,
                                                               #method='ffill')
        ev['index']=event['obs']['swmf_index'].reindex(use_i)#,method='ffill')
        simtdelta = [t-t0 for t in ev['sim'].index]
        swtdelta = [t-t0 for t in ev['sw'].index]
        ev['simt']=[float(n.to_numpy()) for n in simtdelta]
        ev['swt']=[float(n.to_numpy()) for n in swtdelta]
    if 'gridMin' in event['obs'].keys():
        #ev['maggrid'] = event['obs']['gridMin'].reindex(use_i,method='bfill')
        ev['maggrid'] = event['obs']['gridMin'].reindex(use_i)
        ev['GridL'] = ev['maggrid']['dBmin']
    ev['closedVolume'] = ev['closed']['Volume [Re^3]']

    # Calc dt
    ev['dt'] = [(t1-t0).seconds for t0,t1 in
                zip(times[0:-1],times[1::])]
    ev['dt'].append(ev['dt'][-1])

    ## TOTAL
    #K1,5 from mp
    ev['Ks1'] = ev['mp']['K_netK1 [W]']
    ev['Ks5'] = ev['mp']['K_netK5 [W]']
    #K3,4 from lobes
    ev['Ks3'] = ev['lobes']['K_netK3 [W]']
    ev['Ks4'] = ev['lobes']['K_netK4 [W]']
    #K2,6,7 from closed
    ev['Ks2a'] = ev['closed']['K_netK2a [W]']
    ev['Ks2b'] = ev['closed']['K_netK2b [W]']
    ev['Ks6'] = ev['closed']['K_netK6 [W]']
    ev['Ks7'] = ev['closed']['K_netK7 [W]']

    ## HYDRO
    #H1,5 from mp
    ev['Hs1'] = ev['mp']['P0_netK1 [W]']
    ev['Hs5'] = ev['mp']['P0_netK5 [W]']
    #H3,4 from lobes
    ev['Hs3'] = ev['lobes']['P0_netK3 [W]']
    ev['Hs4'] = ev['lobes']['P0_netK4 [W]']
    #H2,6,7 from closed
    ev['Hs2a'] = ev['closed']['P0_netK2a [W]']
    ev['Hs2b'] = ev['closed']['P0_netK2b [W]']
    ev['Hs6'] = ev['closed']['P0_netK6 [W]']
    ev['Hs7'] = ev['closed']['P0_netK7 [W]']

    ## MAG
    #S1,5 from mp
    ev['Ss1'] = ev['mp']['ExB_netK1 [W]']
    ev['Ss5'] = ev['mp']['ExB_netK5 [W]']
    #S3,4 from lobes
    ev['Ss3'] = ev['lobes']['ExB_netK3 [W]']
    ev['Ss4'] = ev['lobes']['ExB_netK4 [W]']
    #S2,6,7 from closed
    ev['Ss2a'] = ev['closed']['ExB_netK2a [W]']
    ev['Ss2b'] = ev['closed']['ExB_netK2b [W]']
    ev['Ss6'] = ev['closed']['ExB_netK6 [W]']
    ev['Ss7'] = ev['closed']['ExB_netK7 [W]']

    ## TOTAL
    #M1,5,total from mp
    ev['M1'] = ev['mp']['UtotM1 [W]'].fillna(value=0)
    ev['M5'] = ev['mp']['UtotM5 [W]'].fillna(value=0)
    ev['M'] = ev['mp']['UtotM [W]'].fillna(value=0)
    ev['MM'] = ev['mp']['MM [kg/s]'].fillna(value=0)
    #M5a,5b,2a,2b,ic from closed
    ev['M5a'] = ev['closed']['UtotM5a [W]'].fillna(value=0)
    ev['M5b'] = ev['closed']['UtotM5b [W]'].fillna(value=0)
    ev['M2a'] = ev['closed']['UtotM2a [W]'].fillna(value=0)
    ev['M2b'] = ev['closed'].get('UtotM2b [W]')
    ev['Mic'] = ev['closed']['UtotMic [W]'].fillna(value=0)
    for M in ['M1','M5','M','MM','M5a','M5b','M2a','M2b','Mic']:
        if ev[M] is not None:
            ev[M] = ev[M].fillna(value=0)
        else:
            ev[M] = np.zeros(len(ev['times']))

    ev['M_lobes'] = ev['M1']
    ev['M_closed'] = ev['M5a']+ev['M5b']+ev['M2a']+ev['M2b']


    ev['Uclosed'] = ev['closed']['Utot [J]']
    ev['Ulobes'] = ev['lobes']['Utot [J]']
    ev['U'] = ev['mp']['Utot [J]']
    # Central difference of partial volume integrals, total change
    # Total
    ev['K_cdiff_closed'] = -1*central_diff(ev['closed']['Utot [J]'])
    ev['K_cidff_lobes'] = -1*central_diff(ev['lobes']['Utot [J]'])
    ev['K_cdiff_mp'] = -1*central_diff(ev['mp']['Utot [J]'])
    # Hydro
    ev['H_cdiff_closed'] = -1*central_diff(ev['closed']['uHydro [J]'])
    ev['H_cdiff_lobes'] = -1*central_diff(ev['lobes']['uHydro [J]'])
    ev['H_cdiff_mp'] = -1*central_diff(ev['mp']['uHydro [J]'])
    # Mag
    ev['S_cdiff_closed'] = -1*central_diff(ev['closed']['uB [J]'])
    ev['S_cdiff_lobes'] = -1*central_diff(ev['lobes']['uB [J]'])
    ev['S_cdiff_mp'] = -1*central_diff(ev['mp']['uB [J]'])
    #if 'obs' in event.keys():
    #    ev['dDstdt_sim'] = -1*central_diff(ev['sim']['dst_sm'])

    ev['K1'] = ev['Ks1']+ev['M1']
    ev['K5'] = ev['Ks5']+ev['M5']
    ev['K2a'] = ev['Ks2a']+ev['M2a']
    ev['K2b'] = ev['Ks2b']+ev['M2b']
    ev['Ksum'] = (ev['Ks1']+ev['Ks3']+ev['Ks4']+ev['Ks5']+ev['Ks6']+ev['Ks7']+
                  ev['M1']+ev['M5'])
    #ev['Kstatic']= ev['mp']['K_net [W]']-ev['inner']['K_net [W]']
    ev['Kstatic']=ev['Ks1']+ev['Ks3']+ev['Ks4']+ev['Ks5']+ev['Ks6']+ev['Ks7']
    #ev['dUdt'] = -ev['mp']['dUtotdt [J/s]']

    #from IPython import embed; embed()
    #time.sleep(3)

    return ev

def gmiono_refactor(event,t0):
    # Name the top level
    ev = {}
    sample = str((event['GMionoNorth_surface'].index[-1]-
                  event['GMionoNorth_surface'].index[-2]).seconds)+'S'
    #ev['ocflb_north'] = event['GMionoNorth_line'].resample(sample).ffill()
    #ev['ocflb_south'] = event['GMionoSouth_line'].resample(sample).ffill()
    #ev['ie_surface_north']=event['GMionoNorth_surface'].resample(
    #                                                           sample).ffill()
    #ev['ie_surface_south']=event['GMionoSouth_surface'].resample(
    #                                                           sample).ffill()
    ev['ocflb_north'] = event['GMionoNorth_line']
    ev['ocflb_south'] = event['GMionoSouth_line']
    ev['ie_surface_north']=event['GMionoNorth_surface']
    ev['ie_surface_south']=event['GMionoSouth_surface']
    # Data collected piece wise so it doesn't always stack up time-wise
    surf_index = ev['ie_surface_north'].index
    line_index = ev['ocflb_north'].index
    timedelta = [t-t0 for t in ev['ocflb_north'].index]
    ev['ie_times']=[float(n.to_numpy()) for n in timedelta]
    # Name more specific stuff
    # Flux
    ev['ie_flux_north']= ev['ie_surface_north']['Bf_netOpenN [Wb]']
    ev['ie_flux_south']= ev['ie_surface_south']['Bf_netOpenS [Wb]']
    ev['ie_flux'] = (abs(ev['ie_surface_north']['Bf_netOpenN [Wb]'])+
                     abs(ev['ie_surface_south']['Bf_netOpenS [Wb]']))
    # Static reconnection
    ev['RXNs_north'] =  ev['ie_surface_north']['dBfdt_netOpenN [Wb/s]']
    ev['RXNs_south'] = -ev['ie_surface_south']['dBfdt_netOpenS [Wb/s]']
    ev['RXNs'] = ( ev['ie_surface_north']['dBfdt_netOpenN [Wb/s]']-
                   ev['ie_surface_south']['dBfdt_netOpenS [Wb/s]'])
    # Motional reconnection
    ev['RXNm_northDay'] = -ev['ie_surface_north']['Bf_netM2a [Wb/s]']
    ev['RXNm_southDay'] =  ev['ie_surface_south']['Bf_netM2a [Wb/s]']
    ev['RXNm_northNight'] = -ev['ie_surface_north']['Bf_netM2b [Wb/s]']
    ev['RXNm_southNight'] =  ev['ie_surface_south']['Bf_netM2b [Wb/s]']
    ev['RXNm_Day'] = (-ev['ie_surface_north']['Bf_netM2a [Wb/s]']+
                       ev['ie_surface_south']['Bf_netM2a [Wb/s]'])
    ev['RXNm_Night'] = (-ev['ie_surface_north']['Bf_netM2b [Wb/s]']+
                         ev['ie_surface_south']['Bf_netM2b [Wb/s]'])
    ev['RXNm'] = ev['RXNm_Day']+ev['RXNm_Night']
    ev['RXN'] = ev['RXNm']+ev['RXNs']
    # Contour (static) reconnection
    ev['cRXNs_north'] = ev['ocflb_north']['dPhidt_net [Wb/s]']
    ev['cRXNs_south'] =  ev['ocflb_south']['dPhidt_net [Wb/s]']
    ev['cRXNs_northDay'] = ev['ocflb_north']['dPhidtDay_net [Wb/s]']
    ev['cRXNs_southDay'] =  ev['ocflb_south']['dPhidtDay_net [Wb/s]']
    ev['cRXNs_northNight'] = ev['ocflb_north']['dPhidtNight_net [Wb/s]']
    ev['cRXNs_southNight'] =  ev['ocflb_south']['dPhidtNight_net [Wb/s]']

    ev['cRXNs'] = (ev['ocflb_north']['dPhidt_net [Wb/s]']+
                    ev['ocflb_south']['dPhidt_net [Wb/s]'])
    ev['cRXNs_Day'] = (ev['ocflb_north']['dPhidtDay_net [Wb/s]']+
                        ev['ocflb_south']['dPhidtDay_net [Wb/s]'])
    ev['cRXNs_Night'] = (ev['ocflb_north']['dPhidtNight_net [Wb/s]']+
                          ev['ocflb_south']['dPhidtNight_net [Wb/s]'])
    # Finite diferences (for checking)
    ev['cdiffRXN_north'] = central_diff(abs(ev['ie_flux_north']))
    ev['cdiffRXN_south'] = central_diff(abs(ev['ie_flux_south']))
    ev['cdiffRXN'] = -central_diff(ev['ie_flux'])
    #from IPython import embed; embed()
    return ev

def ie_refactor(event,t0):
    # Name the top level
    ev = {}
    ev['ie_surface_north'] = event['ionosphere_north_surface'].resample(
                                                                '60s').ffill()
    ev['ie_surface_south'] = event['ionosphere_south_surface'].resample(
                                                                '60s').ffill()
    ev['ocflb_north'] = event['ionosphere_north_line'].resample('60s').ffill()
    ev['ocflb_south'] = event['ionosphere_south_line'].resample('60s').ffill()
    # Data collected piece wise so it doesn't always stack up time-wise
    surf_index = ev['ie_surface_north'].index
    line_index = ev['ocflb_north'].index
    if not len(surf_index)==len(line_index):
        from IPython import embed; embed()
    timedelta = [t-t0 for t in ev['ocflb_north'].index]
    ev['ie_times']=[float(n.to_numpy()) for n in timedelta]
    #TODO if not already in
    '''
    ev['sw'] = ev['obs']['swmf_sw']
    ev['swt'] = sw.index
    ev['log'] = ev['obs']['swmf_log']
    ev['logt'] = log.index
    '''
    #ev['ielogt = dataset[event]['obs']['ie_log'].index
    # Name more specific stuff
    # Flux
    ev['ie_flux_north']=ev['ie_surface_north']['Bf_injectionOpenN [Wb]']
    ev['ie_flux_south']=ev['ie_surface_south']['Bf_escapeOpenS [Wb]']
    ev['ie_flux'] = (abs(ev['ie_surface_north']['Bf_escapeOpenN [Wb]'])+
                     abs(ev['ie_surface_south']['Bf_escapeOpenS [Wb]']))
    # Motional reconnection
    ev['RXNm_northDay'] = ev['ie_surface_north']['Bf_netDay [Wb/s]']
    ev['RXNm_northNight'] = ev['ie_surface_north']['Bf_netNight [Wb/s]']
    ev['RXNm_southDay'] = ev['ie_surface_south']['Bf_netDay [Wb/s]']
    ev['RXNm_southNight'] = ev['ie_surface_south']['Bf_netNight [Wb/s]']
    ev['RXNm_north'] = ev['RXNm_northDay']+ev['RXNm_northNight']
    ev['RXNm_south'] = ev['RXNm_southDay']+ev['RXNm_southNight']
    ev['RXNm_Day'] = ev['RXNm_northDay']+ev['RXNm_southDay']
    ev['RXNm_Night'] = ev['RXNm_northNight']+ev['RXNm_southNight']
    ev['RXNm'] = ev['RXNm_Day']+ev['RXNm_Night']
    # Static reconnection
    ev['RXNs_northDay'] = ev['ocflb_north']['dPhidtDay_net [Wb/s]']
    ev['RXNs_northNight'] = ev['ocflb_north']['dPhidtNight_net [Wb/s]']
    ev['RXNs_southDay'] = ev['ocflb_south']['dPhidtDay_net [Wb/s]']
    ev['RXNs_southNight'] = ev['ocflb_south']['dPhidtNight_net [Wb/s]']
    ev['RXNs_north'] = ev['RXNs_northDay']+ev['RXNs_northNight']
    ev['RXNs_south'] = ev['RXNs_southDay']+ev['RXNs_southNight']
    ev['RXNs_Day'] = ev['RXNs_northDay']+ev['RXNs_southDay']
    ev['RXNs_Night'] = ev['RXNs_northNight']+ev['RXNs_southNight']
    ev['RXNs'] = ev['RXNs_Day']+ev['RXNs_Night']
    # Combined
    for combo in ['RXN_northDay','RXN_northNight',
                  'RXN_southDay','RXN_southNight',
                  'RXN_north','RXN_south',
                  'RXN_Day','RXN_Night','RXN']:
        ev[combo] = (ev[combo.replace('RXN','RXNs')]+
                     ev[combo.replace('RXN','RXNm')])

    # Finite differences
    ev['dphidt_north'] = central_diff(abs(ev['ie_flux_north']))
    ev['dphidt_south'] = central_diff(abs(ev['ie_flux_south']))
    ev['dphidt'] = central_diff(ev['ie_flux'])
    #TODO: stack of -ev['ie_surface_north']['Bf_netNight [Wb/s]']
    #       and the day one
    #   = cdiff integral of changed flux.
    #   It matches the total flux change => then the other peice is unessesary
    #   Does the other piece match too?
    #
    #   Seems like there is non-zero closed_int(Edl)
    #       verify that it should theoeretically be zero
    #       Can there be a "source" term as in the slowdown of u?
    #       If so then there must be a way to calc an effective heating rate
    #   For now the partial flux rates are correct
    #       In terms of how open flux changes over time
    #       Only have full dayside and full nightside
    #       Nothing about how much is exchanged externally vs internally
    #       Do we track was Open Night -> is Open Day? etc???
    #           This would recover at least the day, night, internal pieces
    #   Why is the dayside rate so lame?
    #       It doesn't cause flux growth as expected
    #       Should be opposing nightside and similar magnitude
    t = ev['ie_flux'].index
    daym = ev['RXNm_northDay']
    nightm = ev['RXNm_northNight']
    days = ev['RXNs_northDay']
    nights = ev['RXNs_northNight']
    motion = daym+nightm
    static = days+nights
    fig,(ax1,ax2,ax3) = plt.subplots(3,layout="constrained",
                                 sharex=True,
                                 sharey=True)
    ax1.plot(t,daym/1e3,label='day_motion',c='blue')
    ax1.plot(t,nightm/1e3,label='night_motion',c='purple')
    ax1.fill_between(t,motion/1e3,label='motion',fc='grey')
    #ax1.legend()

    ax2.plot(t,days/1e3,label='day_static [kV]',c='red')
    ax2.plot(t,nights/1e3,label='night_static [kV]',c='orange')
    ax2.fill_between(t,static/1e3,label='static [kV]',fc='grey')
    #ax2.legend()
    #ax1.set_ylim([-200,200])

    ax3.fill_between(t,ev['ie_flux_north']/2e6,label='openFlux [MWb]',
                     fc='skyblue')
    ax3.plot(t,(motion+static)/1e3,label='combined [kV]',c='red')
    ax3.plot(t,(static.cumsum()*60+ev['ie_flux_north'][0])/2e6,
                label='static_integrated [MWb]',c='blue')
    ax3.plot(t,(motion.cumsum()*60+ev['ie_flux_north'][0])/2e6,
                label='motion_integrated [MWb]',c='black')
    #TODO
    #from IPython import embed; embed()
    return ev

if __name__ == "__main__":
    print('this module only contains helper functions and other useful '+
          'things for making plots!')
