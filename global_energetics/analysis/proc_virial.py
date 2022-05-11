#!/usr/bin/env python3
"""Functions process data in prep for virial theorem analysis
"""
import numpy as np
import pandas as pd
#interpackage
from global_energetics.analysis.proc_hdf import get_subzone_contrib

def virial_construction(mpdict, msdict, inner_mp, **kwargs):
    """Function finds percent contributions and missing amounts
    Inputs
        mpdict(DataFrame)-
        msdict(Dict of DataFrames)-
    Returns
        [MODIFIED] mpdict(Dict of DataFrames)
        [MODIFIED] msdict(Dict of DataFrames)
    """
    mp = [m.copy() for m in mpdict.values()][0]
    mp['Virial Surface Total [nT]'] = (mp['Virial Surface Total [nT]']+
                  inner_mp['Virial Fadv [nT]']+inner_mp['Virial b^2 [nT]'])
    mp['Virial [nT]'] = (mp['Virial [nT]'] + inner_mp['Virial Fadv [nT]']+
                         inner_mp['Virial b^2 [nT]'])
    mp['Utot2 [J]'] = mp['KE [J]']+mp['Eth [J]']+mp['Virial Ub [J]']
    mpdict[[k for k in mpdict.keys()][0]] = mp
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
            m[1]['Utot2 [J]'] = (m[1]['KE [J]']+m[1]['Eth [J]']+
                                 m[1]['Virial Ub [J]'])
            msdict.update({m[0]:m[1]})
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
    ftimes = ftimes.drop(ftimes[0]).append(pd.Index([ftimes[-1]]))
    dtimes = ftimes-times
    #Forward difference of integrated positionally weighted density
    df['Um_static [J]'] = -1*df['rhoU_r [Js]']/[d.seconds for d in dtimes]
    f_n0 = df['rhoU_r [Js]']
    f_n1 = f_n0.copy()
    f_n1.index=f_n1.index-dtimes
    f_n1 = f_n1.drop(index=[f_n1.index[0]])
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
    if 'Eth [J]' not in df.keys():
        df['Eth [J]'] = df['Pth [J]']
    if 'Utot [J]' not in df.keys():
        df['Utot [J]'] = df['Pth [J]']+df['KE [J]']+df['Virial Ub [J]']
    if 'uB [J]' not in df.keys():
        df['uB [J]'] = df['Virial Ub [J]']#mostly so it doesn't crash
    return df

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

def process_virial(results,**kwargs):
    """Wrapper function calls all processing steps for timeseries
        data intent for virial analysis
    Inputs
        mpdict,msdict,inner_mp (dict{DataFrames})- data that will be moded
        kwargs:
    Returns
        mpdict,msdict,inner_mp- same as input, MODIFIED
    """
    ##Term modification (usually just name changes, 
     #                   sometimes to hotfix simple calculation errors)
    for group_name,group in results.items():
        print(group_name)
        if not 'time' in group_name:
            for sz in group.keys():
                df = group[sz]#copy subzone values to DataFrame
                if 'Virial Volume Total [nT]' in df.keys():
                    moded = virial_mods(df, df.index)
                    moded = biot_mods(df, df.index)
                    group[sz] = moded

    ##Grouping surface pieces on subzones into appropriate sub-totals
    mpdict, msdict = virial_construction(results['mpdict'],
                                         results['msdict'],
                                         results['inner_mp'])

    ## %contribution, diff between sums etc
    mpdict, msdict = get_subzone_contrib(results['mpdict'],
                                         results['msdict'])
    return mpdict,msdict,results['inner_mp']

if __name__ == "__main__":
    print('this module processes virial and biot savart data, for plotting'+          ' see analyze_virial.py')
