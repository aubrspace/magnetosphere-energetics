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
        if ('mp' in key) or ('ms' in key):
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
        #inner_mp = check_units(gmdict['/mp_iso_betastar_inner_surface'])

        #repackage in dictionary
        data.update({'mpdict': {'ms_full':mp}})
        #data.update({'inner_mp': inner_mp})

    if iedict!={}:
        #repackage in dictionary
        data.update({'iedict': iedict})

    if uadict!={}:
        #repackage in dictionary
        data.update({'uadict': uadict})

    #define subzones with relevant pieces
    msdict = {}
    for vol,surf in [(k.split('_surface')[0]+'_volume',k) for k in
                            gmdict.keys() if 'ms' in k and 'surface' in k]:
        print(vol+'\t'+surf)
        key = vol.split('_volume')[0]
        cleaned_vol = check_timing(gmdict[vol],gmtimes)
        cleaned_surf = check_timing(gmdict[surf],gmtimes)
        if any(['Virial' in k for k in gmdict[vol].keys()]):
            cleaned_vol = check_units(cleaned_vol)
            cleaned_surf = check_units(cleaned_surf)
        else:
            cleaned_vol = check_units(cleaned_vol)
            cleaned_surf = check_units(cleaned_surf)
            cleaned_df=gather_magnetopause(cleaned_surf,cleaned_vol,gmtimes)
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
    #interp_df = pd.DataFrame({'Time [UTC]':times})
    interp_df = pd.DataFrame({'Time [UTC]':times},index=times)
    if times[0] not in df.index:
        df.loc[times[0]] = df.iloc[0]
        df.sort_index()
    for key in df.keys().drop('Time [UTC]'):
        interp_df[key] = df[key]
    interp_df.interpolate(method='time',inplace=True)
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
        if (' [' not in key) and ('[' in key):
            fixed = ' ['.join(key.split('['))
            warnings.warn("Entry "+key+" fixed to be "+fixed, UserWarning)
            df[fixed] = df[key]
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

def get_subzone_contrib(mpdict, msdict, **kwargs):
    """Function finds percent contributions and missing amounts
    Inputs
        mpdict(DataFrame)-
        msdict(Dict of DataFrames)-
    Returns
        [MODIFIED] mpdict(Dict of DataFrames)
        [MODIFIED] msdict(Dict of DataFrames)
    """
    #assume mpdict actually has just one entry
    full = [m for m in msdict.values()][0]
    #Quantify amount missing from sum of all subzones
    missing_volume, summed_volume = pd.DataFrame(), pd.DataFrame()
    szkeys = [sz for sz in msdict.values()][0].keys()
    for key in [k for k in full.keys() if k in szkeys and '[J]' in k]:
        if key in szkeys:
            fullvalue = full[key]
            added = [m for m in msdict.values()][0][key]
            for sub in [m for m in msdict.values()][1::]:
                added = added+sub[key]
            missing_volume[key] = fullvalue-added
            summed_volume[key] = added
    msdict.update({'missed':missing_volume,'summed':summed_volume})
    #Identify percentage contribution from each piece
    for ms in msdict.values():
        for key in ms.keys():
            if key in full.keys():
                ms[key.split('[')[0]+'[%]'] = (100*ms[key]/full[key])
    return mpdict, msdict

if __name__ == "__main__":
    print('this module processes hdf data, for plotting'+
          ' see analyze_virial.py or analyze_energy_temporal.py')
