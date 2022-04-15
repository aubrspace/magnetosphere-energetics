#!/usr/bin/env python3
"""Handling hdf files and initial organization of data regardless of
    analysis specifics
"""
import warnings
import numpy as np
import pandas as pd

def load_nonGM(hdf, **kwargs):
    store = pd.HDFStore(hdf)
    data = {}
    for key in store.keys():
        if any(['ie' in key, 'ua' in key]):
            data[key] = store[key]
    store.close()
    return (data['/ie'], data['/ua_njoule'], data['/ua_nenergy'],
                         data['/ua_nohpi'], data['/ua_ie'])

def load_hdf_sort(hdf, **kwargs):
    """loads HDF file then sorts cleans and sorts into subzones
    inputs
        hdf (str) - filename str to load (full path)
        kwargs:
            timekey
            surfacekey
            volumekey
            innersurfacekey
    Return
        mpdict (dict{DataFrames})
        msdict (dict{DataFrames})
        inner_mp (DataFrame)
    """
    #load data
    store = pd.HDFStore(hdf)

    #check if non-gm data is present
    nonGM = any([ngm in k for k in store.keys() for ngm in ['ie','ua']])
    #keep only GM keys
    gmdict = {}
    for key in store.keys():
        if not any(['ie' in key, 'ua' in key]):
            gmdict[key] = store[key].sort_values(by='Time [UTC]')
    store.close()

    #strip times
    times = [df for df in gmdict.values()][0]['Time [UTC]']

    #define magnetopause and inner_magnetopause will relevant pieces
    mp = gather_magnetopause(gmdict['/mp_iso_betastar_surface'],
                             gmdict['/mp_iso_betastar_volume'], times)
    inner_mp = gmdict['/mp_iso_betastar_inner_surface']

    #check units OoM
    mp = check_units(mp)
    inner_mp = check_units(inner_mp)

    #repackage in dictionary
    mpdict = {'ms_full':mp}

    #define subzones with relevant pieces
    msdict = {}
    for key in gmdict.keys():
        print(key)
        if 'ms' in key:
            if any(['Virial' in k for k in gmdict[key].keys()]):
                cleaned_df = check_timing(gmdict[key],times)
                cleaned_df = check_units(cleaned_df)
            else:
                cleaned_df = gmdict[key]
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_df.drop(
                                                  columns=['Time [UTC]'])})
    return mpdict, msdict, inner_mp, times, nonGM

def gather_magnetopause(outersurface, volume, times):
    """Function combines surface and volume terms to construct single df
        with all magnetopause (magnetosphere) terms
    Inputs
        outersurface(DataFrame)-
        volume(DataFrame)-
    Returns
        combined(DataFrame)
    """
    combined = pd.DataFrame()
    for df in [outersurface, volume]:
        for key in df.keys():
            if not all(df[key].isna()):
                combined[key] = df[key]
    return combined


def check_timing(df,times):
    """If times don't match data, interpolate data
    Inputs
        df
        times
    Return
        interp_df
    """
    #Do nothing if times are already matching
    if len(times) == len(df['Time [UTC]']):
        if all(times == df['Time [UTC]']):
            return df
    #Otw reconstruct on times column and interpolate
    interp_df = pd.DataFrame({'Time [UTC]':times})
    for key in df.keys().drop('Time [UTC]'):
        interp_df[key] = df[key]
    interp_df = interp_df.interpolate(col='Time [UTC]')
    return interp_df

def check_units(df,*, smallunit='[nT]', bigunit='[J]', factor=-8e13):
    """Sometimes J -> nT isn't done properly
    Inputs
        df
    Return
        df
    """
    #Check for small unit in entry and watch out for large values
    for key in df.keys():
        if (smallunit in key) and any(abs(df[key])>1e6):
            warnings.warn("Entry "+key+" >1e6! dividing by virial factor",
                          UserWarning)
            df[key] = df[key]/factor
    return df

def group_subzones(msdict, mode='3zone'):
    """Groups subzones into dictionary with certain keys
    inputs
        msdict (dict{DataFrames})
        mode (str)
    returns
        msdict
    """
    if 'missed' not in msdict.keys():
        msdict['missed']=0
    if ('closed' in msdict.keys()) and ('3zone' in mode):
        msdict = {'lobes':msdict['nlobe']+msdict['slobe'],
                        'closedRegion':msdict['closed'],
                        'rc':msdict['rc'],
                        'missing':msdict['missed']}
    elif ('3zone' in mode):
        msdict = {'lobes':msdict['nlobe']+msdict['slobe'],
                        'closedRegion':msdict['ps']+msdict['qDp'],
                        'rc':msdict['rc'],
                        'missing':msdict['missed']}
    elif ('lobes' in mode):
        msdict ={'nlobe':msdict['nlobe'],
                       'slobe':msdict['slobe'],
                       'remaining':msdict['rc']+msdict['ps']+msdict['qDp'],
                       'missing':msdict['missed']}
    return msdict

if __name__ == "__main__":
    print('this module processes hdf data, for plotting'+
          ' see analyze_virial.py or analyze_energy_temporal.py')
