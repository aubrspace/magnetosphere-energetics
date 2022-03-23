#!/usr/bin/env python3
"""Functions for processing 2D magnetopause detection data
"""
import numpy as np
import os, warnings
import pandas as pd
import datetime as dt
import glob
from swmfpy.web import get_omni_data
#interpackage
from global_energetics.makevideo import get_time
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings)


def read_2D_data(datapath, **kwargs):
    """Function reads in results of 2D mp detection
    Inputs
        datapath
        kwargs:
            read_all (False)- True will return all matching files
    Returns
        swmf (DataFrame)- pandas dataframe object with contents of file
        swmflist (dict{DataFrames})- dictionary w/ multiple DataFrames
    """
    #First look for hdf5 files
    hdflist = glob.glob(os.path.join(datapath,'*.h5'))
    if hdflist != []:
        if not kwargs.get('read_all', False):
            #Only read first, warn if multiple files, or multiple keys
            store = pd.HDFStore(hdflist[0])
            if len(hdflist)>1 or len(store.keys())>1:
                warnings.warn("multiple files or keys found, using only "+
                                                  "the first!",UserWarning)
            swmf = store[store.keys()[0]]
            #make the time info the index
            timekey = [k for k in swmf.keys() if 'time' in k.lower()][0]
            swmf.set_index(timekey,inplace=True)
            swmf.replace(to_replace='None',inplace=True)
            return swmf
        else:
            swmflist = {} #Initialize
            for hdf in hdflist:
                store = pd.HDFStore(hdf)
                for key in store.keys():
                    swmflist[key] = store[key]
            return swmflist
    else:
        warnings.warn("No files found, check datapath!",UserWarning)
        if not kwargs.get('read_all', False):
            return pd.DataFrame()
        else:
            return {'empty':pd.DataFrame()}

def omni_match(data):
    """Function pulls omni data and modifies to match with timing of input
    Inputs
        data (DataFrame)- pandas dataframe
    Returns
        omni_df (DataFrame)- using swmfpy
    """
    #download omni data
    omni = get_omni_data(data.index.min(),
                         data.index.max())
    #get omni time column key
    omni_timekey = [k for k in omni.keys() if 'time' in k.lower()][0]
    #change types to make interpolate-able
    omni_times = pd.DataFrame(index=omni['times'])
    omni_df = pd.DataFrame()
    data_interp = data.index.values.astype('float64')
    omni_interp = omni_times.index.values.astype('float64')
    #interpolate to data times
    for k in omni.keys():
        if k != 'times':
            omni_df[k]=np.interp(data.index.values.astype('float64'),
                                 omni_times.index.values.astype('float64'),
                                 omni[k])
    omni_df.index = data.index
    omni_df.interpolate(inplace=True)
    omni_df['Time [UTC]'] = omni_df.index
    return omni_df

if __name__ == "__main__":
    DATAPATH = 'localdbug/2Dcuts/'

    ##Gather data
    swmf = read_2D_data(DATAPATH)
    omni = omni_match(swmf)
    ##Quicklook
    from matplotlib import pyplot as plt
    pyplotsetup(mode='digitalpresentation')
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[14,8])
    for key in swmf.keys():
        if 'newell' not in key:
            ax[0].scatter(swmf.index,swmf[key],label=key.split('_')[0])
    general_plot_settings(ax[0],do_xlabel=True,
                          ylabel=r'Loc $\left[ R_e\right]$')
    from global_energetics.analysis import proc_indices
    proc_indices.plot_dst(ax[1], [omni],'Time [UTC]',
                          r'$\Delta B \left[ nT\right]$',name='omni')
    general_plot_settings(ax[0],do_xlabel=True,
                          ylabel=r'$\Delta B \left[ nT\right]$')
    ax2 = ax[1].twinx()
    proc_indices.plot_swpdyn(ax2, [omni],'Time [UTC]',
                          r'$P_{dyn} \left[ nPa\right]$',Color='orange',
                          name='omni')
    general_plot_settings(ax2,do_xlabel=True,
                          ylabel=r'$P_{dyn} \left[ nPa\right]$')
    plt.show()
