#!/usr/bin/env python3
"""Handling hdf files and initial organization of data regardless of
    analysis specifics
"""
import warnings
import numpy as np
import datetime as dt
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
    gmdict, iedict, uadict, bsdict, crossdict,termdict = {},{},{},{},{},{}
    for key in store.keys():
        if ('mp' in key) or ('ms' in key):
            if 'Time [UTC]' in store[key].keys():
                gmdict[key] = store[key].sort_values(by='Time [UTC]')
                if 'tshift' in kwargs:
                    gmdict[key]['Time [UTC]'] += dt.timedelta(minutes=
                                                    kwargs.get('tshift',0))
                gmdict[key].index=gmdict[key]['Time [UTC]']
                gmdict[key].drop(columns=['Time [UTC]'],inplace=True)
            else:
                gmdict[key] = store[key]
        if 'ie' in key or 'iono' in key:
            iedict[key] = store[key]
        if 'ua' in key:
            uadict[key] = store[key]
        if 'bs' in key:
            bsdict[key] = store[key]
        if 'flow_line' in key or 'cross' in key:
            #crossdict[key] = store[key]
            crossdict[key] = store[key].sort_values(by=['Time [UTC]','X'])
            if 'tshift' in kwargs:
                crossdict[key]['Time [UTC]'] += dt.timedelta(minutes=
                                                    kwargs.get('tshift',0))
            crossdict[key].index=crossdict[key]['Time [UTC]']
            crossdict[key].drop(columns=['Time [UTC]'],inplace=True)
        if 'terminator' in key or 'sphere2' in key:
            termdict[key] = store[key].sort_values(by='Time [UTC]')
            if 'tshift' in kwargs:
                termdict[key]['Time [UTC]'] += dt.timedelta(minutes=
                                                    kwargs.get('tshift',0))
            termdict[key].index=termdict[key]['Time [UTC]']
            termdict[key].drop(columns=['Time [UTC]'],inplace=True)
            for item in termdict[key].keys():
                if all(termdict[key][item].isna()):
                    termdict[key].drop(columns=item,inplace=True)
    #strip times
    for leaddict in [gmdict, iedict, uadict, bsdict, crossdict,termdict]:
        try:
            times = leaddict[[k for k in leaddict.keys()][0]].index
            data.update({'time':times})
            break
        except IndexError:
            print('Dict empty, looking at next to find time!')
    store.close()

    if gmdict!={}:
        #define magnetopause and inner_magnetopause will relevant pieces
        if '/mp_iso_betastar_volume' in gmdict.keys():
            mp = gather_magnetopause(gmdict['/mp_iso_betastar_surface'],
                                     gmdict['/mp_iso_betastar_volume'],
                                     times)
        elif '/mp_iso_betastar_surface' in gmdict.keys():
            mp = gmdict['/mp_iso_betastar_surface']
        else:
            mp = pd.DataFrame()
        if '/mp_iso_betastar_inner_surface' in gmdict.keys():
            inner_mp = gmdict['/mp_iso_betastar_inner_surface']
        else:
            inner_mp = pd.DataFrame()

        #check units OoM
        mp = check_units(mp)
        inner_mp = check_units(inner_mp)

        #repackage in dictionary
        data.update({'mpdict': {'ms_full':mp}})
        data.update({'inner_mp': inner_mp})

    if iedict!={}:
        #clean up entry names and interpolate out any timing gaps
        cleaned_iedict = {}
        for key in iedict.keys():
            cleaned = check_timing(iedict[key],times)
            cleaned_iedict.update({key.replace('/',''):cleaned})
        #repackage in dictionary
        data.update({'iedict': cleaned_iedict})
        #repackage in dictionary
        #data.update({'iedict': iedict})

    if uadict!={}:
        #repackage in dictionary
        data.update({'uadict': uadict})

    if bsdict!={}:
        #repackage in dictionary
        data.update({'bsdict': bsdict})

    if crossdict!={}:
        #repackage in dictionary
        data.update({'crossdict': crossdict})

    if termdict!={}:
        #clean up entry names and interpolate out any timing gaps
        cleaned_termdict = {}
        for key in termdict.keys():
            cleaned = check_timing(termdict[key],times)
            cleaned_termdict.update({key.replace('/',''):cleaned})
        #repackage in dictionary
        data.update({'termdict': cleaned_termdict})

    #define subzones with relevant pieces
    msdict = {}
    vols = [k for k in gmdict.keys() if 'ms' in k and 'volume' in k]
    surfs = [k for k in gmdict.keys() if 'ms' in k and 'surface' in k]
    for i in range(0,max(len(vols),len(surfs))):
        #if vols!=[]:
        if i<len(vols):
            cleaned_vol = check_timing(gmdict[vols[i]],times)
            cleaned_vol = check_units(cleaned_vol)
            key = vols[i].split('_volume')[0]
            #print(vols[i])
        #if surfs!=[]:
        if i<len(surfs):
            cleaned_surf = check_timing(gmdict[surfs[i]],times)
            key = surfs[i].split('_surface')[0]
            #print(surfs[i])
        #if vols!=[] and surfs!=[]:
        if i<len(vols) and i<len(surfs):
            cleaned_df=gather_magnetopause(cleaned_surf,cleaned_vol,times)
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_df})
        #elif surfs==[]:
        elif i<len(vols):
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_vol})
        else:
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_surf})
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
    if volume.index.name in volume.keys():
        volume.drop(columns=[volume.index.name],inplace=True)
    return pd.merge(outersurface,volume,left_on='Time [UTC]',
                                          right_on='Time [UTC]')

    #for df in [outersurface, volume]:
    #    pd.concat([combined,df])
        #for key in df.keys():
        #    if not all(df[key].isna()):
        #        combined[key] = df[key]
    #return combined


def check_timing(df,times):
    """If times don't match data, interpolate data
    Inputs
        df
        times
    Return
        interp_df
    """
    #Do nothing if times are already matching
    if len(times) == len(df.index):
        if all(times == df.index):
            return df
    #Otw reconstruct on times column and interpolate
    #interp_df = pd.DataFrame({'Time [UTC]':times})
    interp_df = pd.DataFrame({'Time [UTC]':times},index=times)
    if times[0] not in df.index:
        df.loc[times[0]] = df.iloc[0]
        df.sort_index()
    for key in df.keys():
        interp_df[key] = df[key]
    #NOTE method='time' used to work with older version of pandas...
    #interp_df.interpolate(method='time',inplace=True)
    interp_df.interpolate(method='ffill',inplace=True)
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
                        'closed':msdict['closed'],
                        'rc':msdict['rc'],
                        'missing':msdict['missed']}
    elif ('3zone' in mode):
        msdict = {'lobes':msdict['nlobe']+msdict['slobe'],
                      'closed':msdict['ps']+msdict['qDp'],
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
    for key in [k for k in full.keys() if k in szkeys and ('[J]' in k
                                                         or 'bioS' in k)]:
        if key in szkeys:
            fullvalue = full[key]
            added = [m for m in msdict.values()][0][key]
            for sub in [m for m in msdict.values()][1::]:
                added = added+sub[key]
            missing_volume[key] = fullvalue-added
            summed_volume[key] = added
            if any(['Virial' in k for k in full.keys()])and'bioS'not in key:
                dBkey = '[nT]'.join(key.split('[J]'))
                missing_volume[dBkey] = (fullvalue-added)/(-8e13)
                summed_volume[dBkey] = added/(-8e13)
    #msdict.update({'missed':missing_volume,'summed':summed_volume})
    msdict.update({'missed':missing_volume})
    #Identify percentage contribution from each piece
    for ms in msdict.values():
        for key in ms.keys():
            if key in full.keys() and ('[J]' in key or '[Re^3]' in key):
                ms[key.split('[')[0]+'[%]'] = (100*ms[key]/full[key])
    return mpdict, msdict

if __name__ == "__main__":
    print('this module processes hdf data, for plotting'+
          ' see analyze_virial.py or analyze_energy_temporal.py')
