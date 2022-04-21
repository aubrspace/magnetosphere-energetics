#!/usr/bin/env python3
"""Handling hdf files and initial organization of data regardless of
    analysis specifics
"""
import warnings
import numpy as np
import pandas as pd

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
        data (dict{dict{DataFrames}})- ex:
                                            {mpdict:(dict{DataFrames}),
                                             msdict:(dict{DataFrames}),
                                             inner_mp:(DataFrame)}
    """
    data = {}
    #load data
    store = pd.HDFStore(hdf)

    #keep only GM keys
    gmdict, iedict, uadict = {}, {}, {}
    for key in store.keys():
        if 'mp' in key:
            gmdict[key] = store[key].sort_values(by='Time [UTC]')
            gmdict[key].index=gmdict[key]['Time [UTC]'].drop(
                                                    columns=['Time [UTC]'])
        if 'ie' in key:
            iedict[key] = store[key]
        if 'ua' in key:
            uadict[key] = store[key]
    #strip times
    #gmtimes = gmdict[[k for k in gmdict.keys()][0]]['Time [UTC]']
    gmtimes = gmdict[[k for k in gmdict.keys()][0]].index
    data.update({'times':gmtimes})
    store.close()

    if gmdict!={}:
        #define magnetopause and inner_magnetopause will relevant pieces
        mp = gather_magnetopause(gmdict['/mp_iso_betastar_surface'],
                                 gmdict['/mp_iso_betastar_volume'],gmtimes)

        #check units OoM
        mp = check_units(mp)
        inner_mp = check_units(gmdict['/mp_iso_betastar_inner_surface'])

        #repackage in dictionary
        data.update({'mpdict': {'ms_full':mp}})
        data.update({'inner_mp': inner_mp})

    if iedict!={}:
        #repackage in dictionary
        data.update({'iedict': iedict})

    if uadict!={}:
        #repackage in dictionary
        data.update({'uadict': uadict})

    #define subzones with relevant pieces
    msdict = {}
    for key in gmdict.keys():
        print(key)
        if 'ms' in key:
            if any(['Virial' in k for k in gmdict[key].keys()]):
                cleaned_df = check_timing(gmdict[key],gmtimes)
                cleaned_df = check_units(cleaned_df)
            else:
                cleaned_df = gmdict[key]
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_df.drop(
                                                  columns=['Time [UTC]'])})
    if msdict!={}:
        #repackage in dictionary
        data.update({'msdict':msdict})
    return data


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
