#!/usr/bin/env python3
"""Functions for handling and plotting time magnetic indices data
"""
import numpy as np
import pandas as pd

def load_clean_virial(hdf, **kwargs):
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

    #strip times
    times = store[store.keys()[0]]['Time [UTC]']

    #define magnetopause and inner_magnetopause will relevant pieces
    mp = gather_magnetopause(store['/mp_iso_betastar_surface'],
                             store['/mp_iso_betastar_volume'], times)
    inner_mp = store['/mp_iso_betastar_inner_surface']
    mpdict = {'ms_full':mp}

    #define subzones with relevant pieces
    msdict = {}
    for key in store.keys():
        if 'ms' in key:
            if any(['Virial' in k for k in store[key].keys()]):
                cleaned_df = virial_mods(store[key], times)
            else:
                cleaned_df = store[key]
            msdict.update({key.split('/')[1].split('_')[1]:cleaned_df.drop(
                                                  columns=['Time [UTC]'])})
    store.close()
    return mpdict, msdict, inner_mp, times

def get_interzone_stats(mpdict, msdict, inner_mp, **kwargs):
    """Function finds percent contributions and missing amounts
    Inputs
        mpdict(DataFrame)-
        msdict(Dict of DataFrames)-
    Returns
        [MODIFIED] mpdict(Dict of DataFrames)
        [MODIFIED] msdict(Dict of DataFrames)
    """
    mp = [m for m in mpdict.values()][0]
    mp['Virial Surface Total [nT]'] = (mp['Virial Surface Total [nT]']+
                  inner_mp['Virial Fadv [nT]']+inner_mp['Virial b^2 [nT]'])
    mp['Virial [nT]'] = (mp['Virial [nT]'] + inner_mp['Virial Fadv [nT]']+
                         inner_mp['Virial b^2 [nT]'])
    for m in msdict.items():
        #Remove empty/uneccessary columns from subvolumes
        for key in m[1].keys():
            if all(m[1][key].isna()):
                m[1].drop(columns=[key], inplace=True)
        #Total energy
        if 'uB' in m[1].keys():
            m['Utot [J]'] = m['uB [J]']+m['KE [J]']+m['Eth [J]']
        #Total virial contribution from surface and volume terms
        if 'Virial 2x Uk [J]' in m[1].keys():
            if 'nlobe' in m[0]:
                m[1]['Virial Surface Total [J]'] = (
                                       inner_mp['Virial FadvOpenN [J]']+
                                       inner_mp['Virial b^2OpenN [J]']+
                                       mp['Virial Surface TotalOpenN [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            elif 'slobe' in m[0]:
                m[1]['Virial Surface Total [J]'] = (
                                       inner_mp['Virial FadvOpenS [J]']+
                                       inner_mp['Virial b^2OpenS [J]']+
                                       mp['Virial Surface TotalOpenS [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            elif ('qDp' in m[0]) or ('closed' in m[0]):
                m[1]['Virial Surface Total [J]'] = (
                                       inner_mp['Virial FadvClosed [J]']-
                                       inner_mp['Virial FadvLowlat [J]']+
                                       inner_mp['Virial b^2Closed [J]']-
                                       inner_mp['Virial b^2Lowlat [J]']+
                                      mp['Virial Surface TotalClosed [J]'])
                m[1]['Virial [J]'] = (m[1]['Virial Volume Total [J]']+
                                      m[1]['Virial Surface Total [J]'])
            elif 'rc' in m[0]:
                m[1]['Virial [J]'] = m[1]['Virial Volume Total [J]']
                m[1]['Virial Surface Total [J]'] = (
                                       inner_mp['Virial FadvLowlat [J]']+
                                       inner_mp['Virial b^2Lowlat [J]'])
            else:
                m[1]['Virial [J]'] = m[1]['Virial Volume Total [J]']
                m[1]['Virial Surface Total [J]'] = 0
            m[1]['Virial Surface Total [nT]'] = (
                                  m[1]['Virial Surface Total [J]']/(-8e13))
            m[1]['Virial [nT]'] = m[1]['Virial [J]']/(-8e13)
            msdict.update({m[0]:m[1]})
    #Quantify amount missing from sum of all subzones
    missing_volume, summed_volume = pd.DataFrame(), pd.DataFrame()
    for key in [m for m in msdict.values()][0].keys():
        if key in [m for m in mpdict.values()][0].keys():
            fullvolume = [m for m in mpdict.values()][0][key]
            added = [m for m in msdict.values()][0][key]
            for sub in [m for m in msdict.values()][1::]:
                added = added+sub[key]
            missing_volume[key] = fullvolume-added
            summed_volume[key] = added
    msdict.update({'missed':missing_volume})
    msdict.update({'summed':summed_volume})
    #Identify percentage contribution from each piece
    for ms in msdict.values():
        for key in ms.keys():
            ms[key.split('[')[0]+'[%]'] = (100*ms[key]/
                                      [m for m in mpdict.values()][0][key])
    #NOTE maybe mpdict simply wasn't being updated!
    #for key in msdict['summed'].keys():
    #    mp[key] = msdict['summed'][key]
    return mpdict, msdict

def calculate_mass_term(df,times):
    """Calculates the temporal derivative minus surface motion contribution
        to virial term
    Inputs
        df(DataFrame)
    Returns
        df(DataFrame)- modified
    """
    #Get time delta for each step
    ftimes = times.copy()
    ftimes.index=ftimes.index-1
    ftimes.drop(index=[-1],inplace=True)
    dtimes = ftimes-times
    #Forward difference of integrated positionally weighted density
    df['Um_static [J]'] = -1*df['rhoU_r [Js]']/[d.seconds for d in dtimes]
    f_n0 = df['rhoU_r [Js]']
    f_n1 = f_n0.copy()
    f_n1.index=f_n1.index-1
    f_n1 = f_n1.drop(index=[-1])
    forward_diff = (f_n1-f_n0)/[d.seconds for d in dtimes]
    #Save as energy and as virial
    df['Um_static [J]'] = -1*forward_diff
    df['Um_static [nT]'] = -1*forward_diff/(-8e13)
    if 'rhoU_r_net [J]' in df.keys():
        #Get static + motional
        df['Um [J]'] = df['Um_static [J]']+df['rhoU_r_net [J]']
        df['Um [nT]'] = df['Um [J]']/(-8e13)
    else:
        df['Um [J]'] = df['Um_static [J]']
        df['Um [nT]'] = df['Um_static [nT]']
    return df

def biot_mods(df, times):
    """Function returns DataFrame with modified biotsavart columns
    Inputs
        df(DataFrame)
    Returns
        df(DataFrame)- modified
    """
    if ('bioS [nT]' in df.keys()) and ('bioS_full [nT]' in df.keys()):
        df['bioS_ext [nT]'] = df['bioS_full [nT]']-df['bioS [nT]']
    return df

def virial_mods(df, times):
    """Function returns DataFrame with modified virial columns
    Inputs
        df(DataFrame)
    Returns
        df(DataFrame)- modified
    """
    if 'rhoU_r [Js]' in df.keys():
        df = calculate_mass_term(df,times)
        df['Virial Volume Total [J]'] = (df['Virial Volume Total [J]']+
                                         df['Um [J]'])
        df['Virial Volume Total [nT]'] = (df['Virial Volume Total [J]']/
                                                                   (-8e13))
    if (('Virial Volume Total [J]' in df.keys()) and
        ('Virial Surface Total [J]' in df.keys())):
        df['Virial [J]'] = (df['Virial Volume Total [J]']+
                                   df['Virial Surface Total [J]'])
        df['Virial [nT]'] = df['Virial [J]']/(-8e13)
    for key in df.keys():
        if '[J]' in key and key.split(' [J]')[0]+' [nT]' not in df.keys():
            df[key.split(' [J]')[0]+' [nT]'] = df[key]/(-8e13)
    if 'KE [J]' not in df.keys():
        df['KE [J]'] = (df['Virial 2x Uk [J]']-df['Pth [J]'])/2
        df['Eth [J]'] = df['Pth [J]']
        df['Utot [J]'] = df['Pth [J]']+df['KE [J]']+df['Virial Ub [J]']
        df['uB [J]'] = df['Virial Ub [J]']
    return df

def magnetopause_energy_mods(mpdf):
    """Function returns magnetopause DataFrame with modified energy columns
    Inputs
        mpdf(DataFrame)
    Returns
        mpdf(DataFrame)- modified
    """
    #One-off renames
    if 'Utot_acqu[W]' in mpdf.keys():
        mpdf.rename(columns={'Utot_acqu[W]':'Utot_acqu [W]',
                             'Utot_forf[W]':'Utot_forf [W]',
                             'Utot_net[W]':'Utot_net [W]'}, inplace=True)
    #Define relevant pieces
    fluxes, energies = ['K_','ExB_','P0_'], ['Utot_', 'uB_', 'uHydro_']
    direct = ['injection', 'escape', 'net']
    locations = ['', 'Day', 'Flank', 'Tail', 'OpenN', 'OpenS', 'Closed']
    motion = ['acqu', 'forf', 'net']
    u = ' [W]'#units
    #One-off missing keys
    if 'Utot_acquDay [W]' not in mpdf.keys():
        for loc in locations:
            for m in motion:
                mpdf['Utot_'+m+loc+u] = (mpdf['uB_'+m+loc+u]+
                                         mpdf['uHydro_'+m+loc+u])
    #Surface Volume combinations
    for flux in enumerate(fluxes):
        for d in enumerate(direct):
            for loc in locations:
                #Rename to denote 'static' contribution only
                mpdf.rename(columns={flux[1]+d[1]+loc+u:
                                     flux[1]+d[1]+loc+'_static'+u},
                                     inplace=True)
                #Add in motional terms for proper total
                static = mpdf[flux[1]+d[1]+loc+'_static'+u]
                motional = mpdf[energies[flux[0]]+motion[d[0]]+loc+u]
                mpdf[flux[1]+d[1]+loc+u] = static+motional
    #Drop time column
    mpdf.drop(columns=['Time [UTC]'],inplace=True, errors='ignore')
    return mpdf

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
    if 'K_net [W]' in combined.keys():
        combined = magnetopause_energy_mods(combined)
    if 'Virial Volume Total [nT]' in combined.keys():
        combined = virial_mods(combined, times)
    if 'Virial Volume Total [nT]' in combined.keys():
        combined = biot_mods(combined, times)
    return combined

def group_subzones(msdict, mode='3zone'):
    """Groups subzones into dictionary with certain keys
    inputs
        msdict (dict{DataFrames})
        mode (str)
    returns
        msdict
    """
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

def BIGMOD(mpdict,msdict,inner_mp,fix=None):
    """Function makes correction to virial data
    Inputs
        mp, msdict, inner_mp
    Returns
        modified(mp, msdict, inner_mp)
    """
    mp = [m for m in mpdict.values()][0]
    if 'doublePth' in fix:
        mp['Virial Volume Total [nT]'] = (mp['Virial Volume Total [nT]']+
                                          mp['Pth [nT]'])
        mp['Virial [nT]'] = (mp['Virial [nT]']+mp['Pth [nT]'])
        for ms in msdict.items():
            ms[1]['Virial Volume Total [nT]']=(ms[1]['Pth [nT]']+
                                         ms[1]['Virial Volume Total [nT]'])
            ms[1]['Virial [nT]'] = (ms[1]['Virial [nT]']+ms[1]['Pth [nT]'])
            msdict.update({ms[0]:ms[1]})
    if 'scaled' in fix:
        #Find biot savart extrema and scaled virial to match
        scale = mp['bioS [nT]'].min()/mp['Virial [nT]'].min()
        for key in mp.keys():
            if 'Virial' in key and '[nT]' in key:
                mp[key] = mp[key]*scale
        for ms in msdict.items():
            for key in ms[1].keys():
                if 'Virial' in key and '[nT]' in key:
                    ms[1][key] = ms[1][key]*scale
    else:
        scale = 1
    return mp, msdict, inner_mp, scale

if __name__ == "__main__":
    print('this module processes virial and biot savart data, for plotting'+          ' see analyze_virial.py')
