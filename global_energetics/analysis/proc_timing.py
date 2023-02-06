#!/usr/bin/env python3
"""Functions for identifying key timing values from time varying data
"""
import numpy as np
import datetime as dt
import pandas as pd
from scipy.signal import argrelextrema
#Interpackage
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   plot_pearson_r)


def find_peaks(data, **kwargs):
    """Function takes input data and finds indices of the signal peaks
    Inputs
        data (pandas Series)- data to look for peak
        kwargs:
    Returns
        peaktimes (list[dt.datetime])- list of times at which peaks occur
    """
    peaks = argrelextrema(data,np.maximum,order=kwargs.get('window',18))
    peaktimes = data.index[peaks]
    from IPython import embed; embed()

def peak2peak(data1,data2,times,**kwargs):
    """Function generates a list of time deltas between two columns peaks
    Inputs
        data1,data2 (pandas Series)
        times (Series(datetimes))
        kwargs:
    Returns
        dts (list[[datetime,datetime,timedelta]])- nx3 list with deltas
    """
    peaks1 = find_peaks(data1)
    peaks2 = find_peaks(data2)

    if len(peaks1)==len(peaks2):
        dts = len(peaks1)*[[]]
        for i,p1 in enumerate(peaks1):
            dts[i] = [p1,p2,abs(p1-p2)]

def remove_univariate_outliers(data,zthreashold):
    z = (data-data.mean())/data.std()
    data = data[abs(z)<zthreashold]

def pearson_r_shifts(data1,data2,**kwargs):
    """Function takes pearson r correlation value over a range of timeshifts
        to find the maximum correlation time
    Inputs
        data1,data1 (pandas DataFrames)
        kwargs:
            tshiftrange (float)- timeshift range in seconds
            tshift_n (int)- number of intervals to check within range
    Returns
        t_shifts
        r_values
    """
    #Setup
    fixed_data = data1.copy(deep=True).fillna(method='bfill')
    shifted_data = data2.copy(deep=True).fillna(method='bfill')
    remove_univariate_outliers(fixed_data,3)
    remove_univariate_outliers(shifted_data,3)
    fixed_time = [float(t) for t in (fixed_data.index).to_numpy()]
    time_shifts = np.linspace(-kwargs.get('tshiftrange',1800),
                               kwargs.get('tshiftrange',1800),
                                2*kwargs.get('tshift_n',30)+1)
    r_values = np.zeros(len(time_shifts))

    for i,shift_seconds in enumerate(time_shifts):
        tshift = dt.timedelta(seconds=shift_seconds)
        shifted_time = [float(t) for t in
                        (shifted_data.index+tshift).to_numpy()]
        r_values[i] = plot_pearson_r(None,shifted_time,fixed_time,
                                          shifted_data,fixed_data)
    return time_shifts,r_values

