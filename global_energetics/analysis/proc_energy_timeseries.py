#!/usr/bin/env python3
"""Functions for handling and processing time varying magnetopause surface
    data that is spatially averaged, reduced, etc
"""
import glob
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import Bar

#TODO: 
#       -> will also want a separate energy flux term when looking for 
#           % of energy leaving one subzone to ext,set(other sz's)

def energy_mods(mpdf):
    """Function returns magnetopause DataFrame with modified energy columns
    Inputs
        mpdf(DataFrame)
    Returns
        mpdf(DataFrame)- modified
    """
    #Define relevant pieces
    fluxes, energies = ['K_','ExB_','P0_'], ['Utot_', 'uB_', 'uHydro_']
    direct = ['injection', 'escape', 'net']
    locations = ['', 'Day', 'Flank', 'Tail', 'OpenN', 'OpenS', 'Closed']
    motion = ['acqu', 'forf', 'net']
    u = ' [W]'#units
    #Surface Volume combinations
    for flux in enumerate(fluxes):
        for d in enumerate(direct):
            for loc in locations:
                st_combo = flux[1]+d[1]+loc+u
                m_combo = energies[flux[0]]+motion[d[0]]+loc+u
                if (m_combo in mpdf.keys()) and st_combo in mpdf.keys():
                    #Rename to denote 'static' contribution only
                    mpdf.rename(columns={st_combo:
                                         flux[1]+d[1]+loc+'_static'+u},
                                         inplace=True)
                    #Add in motional terms for proper total
                    static = mpdf[flux[1]+d[1]+loc+'_static'+u]
                    motional = mpdf[m_combo]
                    mpdf[flux[1]+d[1]+loc+u] = static+motional
    #Drop time column
    mpdf.drop(columns=['Time [UTC]'],inplace=True, errors='ignore')
    return mpdf

#TODO: figure out if this can be deleted
def add_derived_variables(dflist):
    """Function adds variables based on existing columns
    Inputs
        dflist - list of dataframes
    Outputs
        dflist
    """
    for df in enumerate(dflist):
        if not df[1].empty:
            if len(df[1]) > 1 and (df[1]['name'].iloc[-1].find('fixed')==-1
                              and  df[1]['name'].iloc[-1].find('agg')==-1):
                ###Add cumulative energy terms
                #Compute cumulative energy In, Out, and Net
                start = df[1].index[0]
                totalE = df[1]['Utot [J]']
                delta_t = (df[1]['Time [UTC]'].loc[start+1]-
                        df[1]['Time [UTC]'].loc[start]).seconds
                #use pandas cumulative sum method
                cumu_E_net = df[1]['K_net [W]'].cumsum()*delta_t*-1
                cumu_E_in = df[1]['K_injection [W]'].cumsum()*delta_t*-1
                cumu_E_out = df[1]['K_escape [W]'].cumsum()*delta_t*-1
                #readjust to initialize error to 0 at start
                cumu_E_net = (cumu_E_net+totalE.loc[start]-
                              cumu_E_net.loc[start])
                E_net_error = cumu_E_net - totalE
                E_net_rel_error = E_net_error/totalE*100
                #Add column to dataframe
                dflist[df[0]]['CumulE_net [J]'] = cumu_E_net
                dflist[df[0]]['CumulE_injection [J]'] = cumu_E_in
                dflist[df[0]]['CumulE_escape [J]'] = cumu_E_out
                dflist[df[0]]['Energy_error [J]'] = E_net_error
                dflist[df[0]]['RelativeE_error [%]'] =E_net_rel_error
                ###Add derivative power terms
                Power_dens = df[1]['K_net [W]']/df[1]['Volume [Re^3]']
                #Compute derivative of energy total using central diff
                total_behind = totalE.copy()
                total_forward = totalE.copy()
                total_behind.index = total_behind.index-1
                total_forward.index = total_forward.index+1
                derived_Power = (total_behind-total_forward)/(-2*delta_t)
                derived_Power_dens = derived_Power/df[1]['Volume [Re^3]']
                #Estimate error in power term
                rms_Power = abs(df[1]['K_escape [W]'])
                power_error = df[1]['K_net [W]']-derived_Power
                power_error_rel = (df[1]['K_net [W]']-derived_Power)/(
                                   rms_Power/100)
                power_error_dens = power_error/df[1]['Volume [Re^3]']
                dflist[df[0]]['Power_density [W/Re^3]'] = Power_dens
                dflist[df[0]]['Power_derived [W]'] = derived_Power
                dflist[df[0]]['Power_derived_density [W/Re^3]'] = (
                                                        derived_Power_dens)
                dflist[df[0]]['Power_error [W]'] = power_error
                dflist[df[0]]['Power_error [%]'] = power_error_rel
                dflist[df[0]]['Power_error_density [W/Re^3]'] = (
                                                          power_error_dens)
                ###Add 1step energy terms
                predicted_energy = total_behind+df[1]['K_net [W]']*delta_t
                predicted_error = predicted_energy-totalE
                predicted_error_rel = (predicted_energy-totalE)/totalE*100
                dflist[df[0]]['1step_Predict_E [J]'] = (
                                                          predicted_energy)
                dflist[df[0]]['1step_Predict_E_error [J]'] = (
                                                           predicted_error)
                dflist[df[0]]['1step_Predict_E_error [%]'] = (
                                                       predicted_error_rel)
                ###Add volume/surface area and estimated error
                dflist[df[0]]['V/SA [Re]'] = (df[1]['Volume [Re^3]']/
                                                    df[1]['Area [Re^2]'])
                dflist[df[0]]['V/(SA*X_ss)'] = (df[1]['V/SA [Re]']/
                                                  df[1]['X_subsolar [Re]'])
                dflist[df[0]]['nVolume'] = (df[1]['Volume [Re^3]']/
                                              df[1]['Volume [Re^3]'].max())
                dflist[df[0]]['nArea'] = (df[1]['Area [Re^2]']/
                                              df[1]['Area [Re^2]'].max())
                dflist[df[0]]['nX_ss'] = (df[1]['X_subsolar [Re]']/
                                            df[1]['X_subsolar [Re]'].max())
    return dflist

def process_energy(mpdict,msdict,inner_mp,times):
    """Wrapper function calls all processing steps for timeseries
        data intent for energy analysis
    Inputs
        mpdict,msdict,inner_mp (dict{DataFrames})- data that will be moded
        kwargs:
    Returns
        mpdict,msdict,inner_mp- same as input, MODIFIED
    """
    ##Term modification (always combo motional+static, checknames,
     #                   sometimes hotfix simple calculation errors)
    for group in [mpdict,msdict]:
        for sz in group.keys():
            df = group[sz]#copy subzone values to DataFrame
            if any(['acqu' in k for k in df.keys()]):
                moded = energy_mods(df)#changes for energy terms
                group[sz] = moded
    return mpdict,msdict,inner_mp

if __name__ == "__main__":
    pass
