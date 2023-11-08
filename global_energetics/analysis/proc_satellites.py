#!/usr/bin/env python3
"""module processes observation/simulation satellite traces
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import swmfpy
#interpackage imports
from global_energetics.analysis.analyze_energetics import mark_times
from global_energetics.analysis.proc_indices import (read_indices,
                                                     datetimeparser,
                                                     datetimeparser2,
                                                     datetimeparser3,
                                                     datetimeparser4,
                                                     df_coord_transform)
def mark_cross_themis(axis, crossingdata, *, probe='a',
                      timerange=[dt.datetime(2014,2,18,6),
                                 dt.datetime(2014,2,20,0)]):
    """Function marks themis crossings in time range for given probe based
        on crossing data input
    Input
        axis- pyplot axis object
        crossingdata- pandas DataFrame with all crossings
        probe, timerange- optional inputs for specific probe and time range
    """
    data = crossingdata[(crossingdata['UT']<timerange[1]) &
                        (crossingdata['UT']>timerange[0]) &
                        (crossingdata['PROBE']==probe)]
    for cross in data['UT'].values:
        axis.axvline(cross, color='red', linestyle=None, linewidth=1)
def mark_cross_geotail(axis, crossingdata, *,
                      timerange=[dt.datetime(2014,2,18,6),
                                 dt.datetime(2014,2,20,0)]):
    """Function marks geotail crossings in time range for given probe based
        on crossing data input
    Input
        axis- pyplot axis object
        crossingdata- pandas DataFrame with all crossings
        probe, timerange- optional inputs for specific probe and time range
    """
    data = crossingdata[(crossingdata['UT']<timerange[1]) &
                        (crossingdata['UT']>timerange[0])]
    for cross in data['UT'].values:
        axis.axvline(cross, color='red', linestyle=None, linewidth=1)
def plot_BetastarScatter(axis, dflist, *, ylabel=None,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise','geotail',
               'cluster1', 'cluster2', 'cluster3', 'cluster4']
    markerdict = dict({'themisa':'^','themisd':'o','themise':'x',
                       'cluster4':'d'})
    colordict = dict({'themisa':'black','themisd':'darkmagenta',
                      'themise':'maroon','cluster4':'magenta'})
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('TH')!=-1:
            probe = name.split('_obs')[0].split('TH')[-1]
            for simdf in dflist:
                simname = simdf['name'].iloc[-1]
                print(simname, probe)
                if simname.find('themis'+probe.lower())!=-1:
                    xdata = np.interp(simdf['Time [UTC]'][0:-1],
                                      df['UT'][0:-1],df['Betastar'][0:-1])
                    ydata = simdf['Betastar'][0:-1]
                    xcolors=((simdf['Time [UTC]'][0:-1]-
                              simdf['Time [UTC]'][0]).values/
                                           (simdf['Time [UTC]'][-2:-1]-
                                            simdf['Time [UTC]'][0]).values)
                    #contingency = conting.fromBoolean(ydata,xdata)
                    scatterdata = pd.DataFrame({'time':simdf['Time [UTC]'][0:-1],
                                                'betastar_sim':ydata,
                                                'betastar_obs':xdata})
                    scatterdata['diff'] = scatterdata['betastar_sim']-scatterdata['betastar_obs']
                    scatterdata['match'] = abs(scatterdata['diff'])<0.2
                    fwd_total, back_total = dt.timedelta(seconds=0), dt.timedelta(seconds=0)
                    scatterdata['fwd_total']=dt.timedelta(seconds=0)
                    scatterdata['back_total']=dt.timedelta(seconds=0)
                    scatterdata['timediff']=dt.timedelta(seconds=0)
                    for i in scatterdata.index[0:-1]:
                        if scatterdata['match'].loc[i]:
                            fwd_total=dt.timedelta(seconds=0)
                        else:
                            timedelta = scatterdata['time'][i+1]-scatterdata['time'][i]
                            fwd_total+=timedelta
                        scatterdata['fwd_total'].loc[i]=fwd_total
                    for i in reversed(scatterdata.index[1::]):
                        if scatterdata['match'].loc[i]:
                            back_total=dt.timedelta(seconds=0)
                        else:
                            timedelta = scatterdata['time'][i]-scatterdata['time'][i-1]
                            back_total+=timedelta
                        scatterdata['back_total'].loc[i]=back_total
                        scatterdata['timediff'].loc[i] = min([scatterdata['fwd_total'].loc[i],
                                                              scatterdata['back_total'].loc[i]],
                                                              key=abs)
                    axis.scatter(xdata,ydata,
                      label=r'\textit{Themis}$\displaystyle_'+probe+'$',
                      c=scatterdata['timediff'], marker=markerdict[simname])
        elif name.find('GE')!=-1:
            pass
        elif (name.find('C')!=-1):
            probe = name.split('_obs')[0].split('C')[-1]
            for simdf in dflist:
                simname = simdf['name'].iloc[-1]
                print(simname, probe)
                if simname.find('cluster'+probe)!=-1:
                    xdata = np.interp(simdf['Time [UTC]'][0:-1],
                                      df['UT'][0:-1],df['Betastar'][0:-1])
                    ydata = simdf['Betastar'][0:-1]
                    xcolors=((simdf['Time [UTC]'][0:-1]-
                              simdf['Time [UTC]'][0]).values/
                                           (simdf['Time [UTC]'][-2:-1]-
                                            simdf['Time [UTC]'][0]).values)
                    scatterdata = pd.DataFrame({'time':simdf['Time [UTC]'][0:-1],
                                                'betastar_sim':ydata,
                                                'betastar_obs':xdata})
                    scatterdata['diff'] = scatterdata['betastar_sim']-scatterdata['betastar_obs']
                    scatterdata['match'] = abs(scatterdata['diff'])<0.2
                    fwd_total, back_total = dt.timedelta(seconds=0), dt.timedelta(seconds=0)
                    scatterdata['fwd_total']=dt.timedelta(seconds=0)
                    scatterdata['back_total']=dt.timedelta(seconds=0)
                    scatterdata['timediff']=dt.timedelta(seconds=0)
                    for i in scatterdata.index[0:-1]:
                        if scatterdata['match'].loc[i]:
                            fwd_total=dt.timedelta(seconds=0)
                        else:
                            timedelta = scatterdata['time'][i+1]-scatterdata['time'][i]
                            fwd_total+=timedelta
                        scatterdata['fwd_total'].loc[i]=fwd_total
                    for i in reversed(scatterdata.index[1::]):
                        if scatterdata['match'].loc[i]:
                            back_total=dt.timedelta(seconds=0)
                        else:
                            timedelta = scatterdata['time'][i]-scatterdata['time'][i-1]
                            back_total+=timedelta
                        scatterdata['back_total'].loc[i]=back_total
                        scatterdata['timediff'].loc[i] = min([scatterdata['fwd_total'].loc[i],
                                                              scatterdata['back_total'].loc[i]],
                                                              key=abs)
                    axis.scatter(xdata,ydata,
                      label=r'\textit{Cluster}$\displaystyle_'+probe+'$',
                      c=xcolors, marker=markerdict[simname])
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        axis.set_xlim([0,2])
        axis.set_ylim([0,2])
    axis.set_xlabel(r'$\displaystyle\beta^*_{\textit{Observed}}$')
    axis.set_ylabel(r'$\displaystyle\beta^*_{\textit{Simulation}}$')
    axis.legend(loc='upper left')
def plot_Betastar(axis, dflist, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise','geotail',
               'cluster1', 'cluster2', 'cluster3', 'cluster4']
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('TH')!=-1:
            probe = name.split('_obs')[0].split('TH')[-1]
            axis.plot(df['UT'],df['Betastar'],
                      label=r'\textit{Themis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif name.find('GE')!=-1:
            pass
        elif (name.find('C')!=-1) and (name.find('_obs')!=-1):
            probe = name.split('_obs')[0].split('C')[-1]
            axis.plot(df['UT'], df['Betastar'],
                      label=r'\textit{Cluster}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif any([name.find(key)!=-1 for key in simkeys]):
            if name.find('themis')!=-1:
                probe = name.split('themis')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Betastar'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            elif name.find('cluster')!=-1:
                probe = name.split('cluster')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Betastar'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            else:
                axis.plot(df['Time [UTC]'],df['Betastar'],
                          label=r'\textit{Sim}$\displaystyle$',
                       linewidth=Size, linestyle=ls,color='magenta')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        axis.set_ylim([0,5])
    axis.set_xlabel(r'\textit{Time (UT)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left')
def plot_Magnetosphere(axis, dflist, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise','geotail',
               'cluster1', 'cluster2', 'cluster3', 'cluster4']
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('TH')!=-1:
            probe = name.split('_obs')[0].split('TH')[-1]
            axis.plot(df['UT'],df['Magnetosphere_state'],
                      label=r'\textit{Themis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif name.find('GE')!=-1:
            pass
        elif (name.find('C')!=-1) and (name.find('_obs')!=-1):
            probe = name.split('_obs')[0].split('C')[-1]
            axis.plot(df['UT'], df['Magnetosphere_state'],
                      label=r'\textit{Cluster}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif any([name.find(key)!=-1 for key in simkeys]):
            if name.find('themis')!=-1:
                probe = name.split('themis')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Magnetosphere_state'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            elif name.find('cluster')!=-1:
                probe = name.split('cluster')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Magnetosphere_state'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            else:
                axis.plot(df['Time [UTC]'],df['Magnetosphere_state'],
                          label=r'\textit{Sim}$\displaystyle$',
                       linewidth=Size, linestyle=ls,color='magenta')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        #axis.set_ylim([0,200])
        pass
    axis.set_xlabel(r'\textit{Time (UT)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left')
def plot_Bmag(axis, dflist, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=2, ls=None,
             use_inner=False, use_shield=False, legend_loc=None):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise','geotail',
               'cluster1', 'cluster2', 'cluster3', 'cluster4']
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('TH')!=-1:
            probe = name.split('_obs')[0].split('TH')[-1]
            axis.plot(df['UT'],df['FGS-D_B_TOTAL'],
                      label=r'\textit{Themis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif name.find('GE')!=-1:
            axis.plot(df['UT'], df['IB']*0.1,
                      label=r'\textit{Geotail}$\displaystyle$',
                      linewidth=Size, linestyle=ls,color='black')
        elif (name.find('C')!=-1) and (name.find('_obs')!=-1):
            probe = name.split('_obs')[0].split('C')[-1]
            axis.plot(df['UT'], df['B'],
                      label=r'\textit{Cluster}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='black')
        elif name.find('omni')!=-1:
            axis.plot(df['Time [UTC]'], df['b'],
                      label=r'\textit{OMNI}',
                      linewidth=Size, linestyle=ls,color='black')
        elif any([name.find(key)!=-1 for key in simkeys]):
            if name.find('themis')!=-1:
                probe = name.split('themis')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Bmag [nT]'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            elif name.find('cluster')!=-1:
                probe = name.split('cluster')[-1].upper()
                axis.plot(df['Time [UTC]'],df['Bmag [nT]'],
                          label=r'\textit{Sim}$\displaystyle_'+probe+'$',
                       linewidth=Size, linestyle=ls,color='magenta')
            else:
                axis.plot(df['Time [UTC]'],df['Bmag [nT]'],
                          label=r'\textit{Sim}$\displaystyle$',
                       linewidth=Size, linestyle=ls,color='magenta')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        axis.set_ylim([0,125])
    #axis.set_xlabel(r'\textit{Time (UT)}')
    axis.set_ylabel(ylabel)
    if legend_loc==None:
        axis.legend(loc='upper left')
    else:
        axis.legend(loc=legend_loc)

def combine_obs_sats(indict):
    """Combines plasma and magnetic field data to a common time axis
    Inputs
        indict (dict{DataFrames})
    Returns
        combined (DataFrame)
    """
    combined = pd.DataFrame()
    # check for 'plasma' and 'bfield' in the dictionary
    if indict['plasma'].empty or indict['bfield'].empty:
        print('Cant combine, plasma or bfield is empty')
    else:
        # set the lead to the one that is larger
        # set follower to smaller one
        if len(indict['bfield'])>len(indict['plasma']):
            leader = indict['bfield']
            follower = indict['plasma']
        else:
            leader = indict['plasma']
            follower = indict['bfield']
        # interpolate follower -> leader
        xtime = [t.value for t in follower.index]
        ytime = [t.value for t in leader.index]
        combined = leader.copy(deep=True)
        for var in follower.keys():
            combined[var] = np.interp(ytime,xtime,follower[var])
    return combined

def smooth_data(indf,**kwargs):
    """Function smooths data, typically to 1min resolution
    Inputs
        indf (DataFrame)
        kwargs:
    Returns
        df
    """
    df = indf.copy(deep=True) #copy so we can control what's changing
    if df.empty:
        print('Empty data frame, skipping derived variables')
    else:
        df = df.resample('1T').mean()
    return df
def add_derived_variables2(indf, name, datatype,**kwargs):
    """Function returns dataframe with derived variables
    Inputs
        indf (DataFrame)
        name (str)
        datatype (str)
        kwargs:
    Returns
        df
    """
    df = indf.copy(deep=True) #copy so we can control what's changing
    if df.empty:
        print('Empty data frame, skipping derived variables')
    else:
        if datatype == 'virtual':
            n = df['Rho']
            ux = df['U_x']
            uy = df['U_y']
            uz = df['U_z']
            p = df['P']
            bx = df['B_x']
            by = df['B_y']
            bz = df['B_z']
        elif datatype == 'combined':
            # Thermal Pressure
            if ('tpar' in [k.lower() for k in df.keys()] and
                'tperp' in [k.lower() for k in df.keys()]):
                if 'cluster' in name:
                    Tfactor = 1e6 # from MK to K
                    T =np.sqrt(df['Tpar']**2+(2*df['Tperp'])**2)*Tfactor #K
                elif 'mms' in name:
                    Tfactor = 1.1604e4 # from eV to K
                    T =np.sqrt(df['tpar']**2+(2*df['tperp'])**2)*Tfactor #K
                df['p'] = df['n']*1e6*1.3807e-23*T*1e9 # nPa
                p = df['p']
            elif 'ptot' in df.keys():
                pass
            elif 'themis' in name and 'p' in df.keys():
                df.rename(columns={'p':'ptot'},inplace=True)
            n = df['n']
            ux = df['vx']
            uy = df['vy']
            uz = df['vz']
            bx = df['bx']
            by = df['by']
            bz = df['bz']
        # Dynamic pressure
        rho = n*1e6*1.6605e-27 #kg/m^3
        df['pdyn'] = np.sqrt(ux**2+uy**2+uz**2) * rho*1e9 #nPa
        if 'ptot' in df.keys() and 'p' not in df.keys():
            df['p'] = df['ptot']*1.6022e-4-df['pdyn'] #nPa
            p = df['p']
        # Betastar
        df['betastar'] =(p+df['pdyn'])/((bx**2+by**2+bz**2)/(2*4*pi*1e-7))
        # H total energy transfer vector
        df['Hx'] = 0.5*df['pdyn']+2.5*p*6371**2*ux #W/Re^2
        df['Hy'] = 0.5*df['pdyn']+2.5*p*6371**2*uy #W/Re^2
        df['Hz'] = 0.5*df['pdyn']+2.5*p*6371**2*uz #W/Re^2
        # S total energy transfer vector
        df['Sx'] = (bx**2+by**2+bz**2)/(4*np.pi*1e-7)*1e-9*6371**2*(
                    ux)-bx*(bx*ux+by*uy+bz*uz)/(4*np.pi*1e-7)*1e-9*6371**2
        df['Sy'] = (bx**2+by**2+bz**2)/(4*np.pi*1e-7)*1e-9*6371**2*(
                    uy)-by*(bx*ux+by*uy+bz*uz)/(4*np.pi*1e-7)*1e-9*6371**2
        df['Sz'] = (bx**2+by**2+bz**2)/(4*np.pi*1e-7)*1e-9*6371**2*(
                    uz)-bz*(bx*ux+by*uy+bz*uz)/(4*np.pi*1e-7)*1e-9*6371**2
        #W/Re^2
        # K total energy transfer vector
        df['Kx'] = df['Hx']+df['Sx'] #W/Re^2
        df['Ky'] = df['Hy']+df['Sy'] #W/Re^2
        df['Kz'] = df['Hz']+df['Sz'] #W/Re^2
    return df

def add_derived_variables(dflist, *, obs=False):
    """Function adds columns of data by performing simple operations
    Inputs
        dflist- dataframe
    Outputs
        dflist- dataframe with modifications
    """
    for df in enumerate(dflist):
        if (not df[1].empty) and (not obs):
            ###B field
            B = sqrt(df[1]['Bx']**2+df[1]['By']**2+df[1]['Bz']**2)
            dflist[df[0]]['Bmag [nT]'] = B
            ###Flow field
            U = sqrt(df[1]['Ux']**2+df[1]['Uy']**2+df[1]['Uz']**2)
            dflist[df[0]]['Umag [km/s]'] = U
            ###Betastar
            Dp = df[1]['Rho']*1e6*1.6605e-27*U**2*1e6*1e9
            dflist[df[0]]['Betastar'] =(df[1]['P']+Dp)/(
                                        B**2/(2*4*pi*1e-7)*1e-9)
            ###Magnetosphere state
            state=[]
            for index in df[1].index:
                if dflist[df[0]]['Betastar'].iloc[index]>0.7:
                    if df[1]['status'].iloc[index]!=1:
                        state.append(0)
                    else:
                        state.append(1)
                else:
                    state.append(1)
            dflist[df[0]]['Magnetosphere_state']=state
        elif not df[1].empty:
            name = df[1]['name'].iloc[-1]
            if name.find('TH')!=-1:
                #Get P, Dp, B in J/m^3 and T
                probe = name.split('_obs')[0].split('TH')[-1]
                Ptot = df[1]['P_ION_MOM_ESA-'+probe] #eV/cm^3
                N = df[1]['N_ION_MOM_ESA-'+probe]*1e6*1.6726e-27 #kg/m^3
                Vx = df[1]['VX_ION_GSM_MOM_ESA-'+probe] #km/s
                Vy = df[1]['VY_ION_GSM_MOM_ESA-'+probe] #km/s
                Vz = df[1]['VZ_ION_GSM_MOM_ESA-'+probe] #km/s
                V = np.sqrt(Vx**2+Vy**2+Vz**2)*1000 #m/s
                Dp = V**2 * N #J/m^3
                P = Ptot*1.6022e-13-Dp/2 #J/m^3
                B = df[1]['FGS-D_B_TOTAL']*1e-9 #T
            elif name.find('C4')!=-1:
                #Get P, Dp, B in J/m^3 and T
                probe = 4
                N = df[1]['N(P)']*1e6*1.6726e-27 #kg/m^3
                Vx = df[1]['VX_P_GSE'] #km/s
                Vy = df[1]['VY_P_GSE'] #km/s
                Vz = df[1]['VZ_P_GSE'] #km/s
                V = np.sqrt(Vx**2+Vy**2+Vz**2)*1000 #m/s
                T =np.sqrt(df[1]['T(P)_PAR']**2+(2*df[1]['T(P)_PERP'])**2)/1e6 #K
                P = N*1.3807e-23*T
                Dp = V**2 * N #J/m^3
                B = df[1]['B']*1e-9 #T
            else:
                P, Dp, B = 0, 0, 1
            dflist[df[0]]['Betastar'] = (P+Dp)/(B**2/(2*4*pi*1e-7))
            ###Magnetosphere state
            state=[]
            for index in df[1].index:
                if dflist[df[0]]['Betastar'].iloc[index]>0.7:
                    state.append(0)
                else:
                    state.append(1)
            dflist[df[0]]['Magnetosphere_state']=state
    return dflist
def split_geotail(dflist):
    """Function splits themis satellite data in big list into each probe
    Inputs
        dflist
    Outputs
        themis_a.. themis_e
    """
    geotail, cross = [], []
    for df in dflist:
        name = df['name'].iloc[-1]
        if (name.find('GE')!=-1) or (name.find('geotail')!=-1):
            geotail.append(df)
            if (name.find('GE')!=-1) and (len(geotail)>1):
                geotail = [interp_combine_dfs(geotail[0],geotail[-1],
                                              'UT', 'UT')]
        elif (name.find('crossing')!=-1):
            cross.append(df)
    return geotail, cross
def split_themis(dflist):
    """Function splits themis satellite data in big list into each probe
    Inputs
        dflist
    Outputs
        themis_a.. themis_e
    """
    themis_a,themis_b,themis_c,themis_d,themis_e,cross=[],[],[],[],[],[]
    for df in dflist:
        name = df['name'].iloc[-1]
        if (name.find('THA')!=-1) or (name.find('themisa')!=-1):
            themis_a.append(df)
            if (name.find('THA')!=-1) and (len(themis_a)>1):
                themis_a = [interp_combine_dfs(themis_a[0],themis_a[-1],
                                              'UT', 'UT')]
        elif (name.find('THB')!=-1) or (name.find('themisb')!=-1):
            themis_b.append(df)
            if (name.find('THB')!=-1) and (len(themis_b)>1):
                themis_b = [interp_combine_dfs(themis_b[0],themis_b[-1],
                                              'UT', 'UT')]
        elif (name.find('THC')!=-1) or (name.find('themisc')!=-1):
            themis_c.append(df)
            if (name.find('THC')!=-1) and (len(themis_c)>1):
                themis_c = [interp_combine_dfs(themis_c[0],themis_c[-1],
                                              'UT', 'UT')]
        elif (name.find('THD')!=-1) or (name.find('themisd')!=-1):
            themis_d.append(df)
            if (name.find('THD')!=-1) and (len(themis_d)>1):
                themis_d = [interp_combine_dfs(themis_d[0],themis_d[-1],
                                              'UT', 'UT')]
        elif (name.find('THE')!=-1) or (name.find('themise')!=-1):
            themis_e.append(df)
            if (name.find('THE')!=-1) and (len(themis_e)>1):
                themis_e = [interp_combine_dfs(themis_e[0],themis_e[-1],
                                              'UT', 'UT')]
        elif (name.find('crossing')!=-1):
            cross.append(df)
    return themis_a, themis_b, themis_c, themis_d, themis_e, cross
def split_cluster(dflist):
    """Function splits themis satellite data in big list into each probe
    Inputs
        dflist
    Outputs
        themis_a.. themis_e
    """
    cluster1,cluster2,cluster3,cluster4=[],[],[],[]
    for df in dflist:
        name = df['name'].iloc[-1]
        if (name.find('C1')!=-1) or (name.find('cluster1')!=-1):
            cluster1.append(df)
            if (name.find('C1')!=-1) and (len(cluster1)>1):
                cluster1 = [interp_combine_dfs(cluster1[0],cluster1[-1],
                                              'UT', 'UT')]
        elif (name.find('C2')!=-1) or (name.find('cluster2')!=-1):
            cluster2.append(df)
            if (name.find('C2')!=-1) and (len(cluster2)>1):
                cluster2 = [interp_combine_dfs(cluster2[0],cluster2[-1],
                                              'UT', 'UT')]
        elif (name.find('C3')!=-1) or (name.find('cluster3')!=-1):
            cluster3.append(df)
            if (name.find('C3')!=-1) and (len(cluster3)>1):
                cluster3 = [interp_combine_dfs(cluster3[0],cluster3[-1],
                                              'UT', 'UT')]
        elif (name.find('C4')!=-1) or (name.find('cluster4')!=-1):
            cluster4.append(df)
            if (name.find('C4')!=-1) and (len(cluster4)>1):
                cluster4 = [interp_combine_dfs(cluster4[0],cluster4[-1],
                                              'UT', 'UT')]
    return cluster1, cluster2, cluster3, cluster4
def geotail_to_df(obsdict, *, satkey='geotail', crosskey='crossings'):
    """Function returns data frame using dict to find file and satkey for
        which satellite data to pull
    Inputs
        obsdict- dictionary with filepaths for different satellites
        satkey- satellite key for dictionary
        crosskey- crossing file indication key for dictionary
    Outputs
        dflist
    """
    dflist = []
    for satfile in obsdict.get(satkey,[]):
        if os.path.exists(satfile):
            #name = satfile.split('/')[-1].split('_')[1]
            filename = satfile.split('/')[-1]
            name = filename.split('_')[0]
            if (crosskey in filename) and (satkey in filename):
                df = pd.read_csv(satfile, header=0, sep='\s+',
                                parse_dates={'TIME':['YYYYMMDD', 'UT']},
                                date_parser=datetimeparser4,
                                infer_datetime_format=True,index_col=False)
                df = df.rename(columns={'TIME':'UT'})
                nametag = pd.Series({'name':(name+'_crossings').lower()})
            elif ('MGF' in filename) or ('CPI' in filename):
                heads, feet, skiplen, total_len = [], [], 0, 0
                headerlines = []
                with open(satfile,'r') as mgffile:
                    for line in enumerate(mgffile):
                        if line[1].find('@')!=-1:
                            heads.append([line[0]-1,line[0]+1])
                            feet.append(skiplen)
                            skiplen=0
                            headerlines.append(prev_line)
                        elif line[1].find('#')!=-1:
                            skiplen+=1
                        total_len +=1
                        prev_line = line[1]
                    feet.append(skiplen)
                    for head in enumerate(heads[0:-1]):
                        df=pd.read_csv(satfile,header=head[1][-1],sep='\s+',
                                skipfooter=(total_len-heads[head[0]+1][0]+
                                         feet[head[0]+1]),engine='python',
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                        df.columns = pd.Index(headerlines[head[0]].split())
                        nametag = pd.Series({'name':name+'_ELEC'})
                        dflist.append(df.append(nametag,ignore_index=True))
                    #last dataset in the file
                    df = pd.read_csv(satfile, header=heads[-1][-1],
                                     engine='python', sep='\s+',
                                     skipfooter=feet[-1],
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                    df.columns = pd.Index(headerlines[-1].split())
                    if (satfile.find('MGF')!=-1):
                        name = name+'_MGF'
                        df = df.rename(columns={'EPOCH':'UT'})
                    elif (satfile.find('CPI')!=-1):
                        name = name+'_CPI'
                        df = df.rename(columns={'CDF_EPOCH':'UT'})
                    nametag = pd.Series({'name':name})
            else:
                df = pd.DataFrame()
            dflist.append(df.append(nametag,ignore_index=True))
            print('{} loaded'.format(nametag))
    return dflist
def cluster_to_df(obsdict, *, satkey='cluster', crosskey='crossings'):
    """Function returns data frame using dict to find file and satkey for
        which satellite data to pull
    Inputs
        obsdict- dictionary with filepaths for different satellites
        satkey- satellite key for dictionary
        crosskey- crossing file indication key for dictionary
    Outputs
        dflist
    """
    dflist = []
    for satfile in obsdict.get(satkey,[]):
        if os.path.exists(satfile):
            #name = satfile.split('/')[-1].split('_')[1]
            filename = satfile.split('/')[-1]
            name = filename.split('_')[1]
            if (crosskey in filename) and (satkey in filename):
                df = pd.read_csv(satfile, header=0, sep='\s+',
                                parse_dates={'TIME':['YYYYMMDD', 'UT']},
                                date_parser=datetimeparser4,
                                infer_datetime_format=True,index_col=False)
                df = df.rename(columns={'TIME':'UT'})
                nametag = pd.Series({'name':(name+'_crossings').lower()})
            elif ('PP' in filename) or ('CP' in filename):
                heads, feet, skiplen, total_len = [], [], 0, 0
                headerlines = []
                with open(satfile,'r') as clfile:
                    for line in enumerate(clfile):
                        if line[1].find('@')!=-1:
                            heads.append([line[0]-1,line[0]+1])
                            feet.append(skiplen)
                            skiplen=0
                            headerlines.append(prev_line)
                        elif line[1].find('#')!=-1:
                            skiplen+=1
                        total_len +=1
                        prev_line = line[1]
                    feet.append(skiplen)
                    df = pd.read_csv(satfile, header=heads[-1][-1],
                                     engine='python', sep='\s+',
                                     skipfooter=feet[-1],
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                    df.columns = pd.Index(headerlines[-1].split())
                    if (satfile.find('PP')!=-1):
                        name = name+'_PP'
                        df = df.rename(columns={'EPOCH':'UT'})
                    elif (satfile.find('CP')!=-1):
                        name = name+'_CP'
                    nametag = pd.Series({'name':name})
            else:
                df = pd.DataFrame()
            dflist.append(df.append(nametag,ignore_index=True))
            print('{} loaded'.format(nametag))
    return dflist
def themis_to_df(obsdict, *, satkey='themis', crosskey='crossings'):
    """Function returns data frame using dict to find file and satkey for
        which satellite data to pull
    Inputs
        obsdict- dictionary with filepaths for different satellites
        satkey- satellite key for dictionary
        crosskey- crossing file indication key for dictionary
    Outputs
        dflist
    """
    dflist = []
    for satfile in obsdict.get(satkey,[]):
        if os.path.exists(satfile):
            #name = satfile.split('/')[-1].split('_')[1]
            filename = satfile.split('/')[-1]
            name = filename.split('_')[1]
            if (crosskey in filename) and (satkey in filename):
                df = pd.read_csv(satfile, header=42, sep='\s+',
                                parse_dates={'UT':['TIMESTAMP']},
                                date_parser=datetimeparser3,
                                infer_datetime_format=True,index_col=False)
                nametag = pd.Series({'name':(name+'_crossings').lower()})
            elif ('MOM' in filename) or ('FGM' in filename):
                heads, feet, skiplen, total_len = [], [], 0, 0
                headerlines = []
                with open(satfile,'r') as momfile:
                    for line in enumerate(momfile):
                        if line[1].find('@')!=-1:
                            heads.append([line[0]-1,line[0]+1])
                            feet.append(skiplen)
                            skiplen=0
                            headerlines.append(prev_line)
                        elif line[1].find('#')!=-1:
                            skiplen+=1
                        total_len +=1
                        prev_line = line[1]
                    feet.append(skiplen)
                    '''
                    for head in enumerate(heads[0:-1]):
                        df=pd.read_csv(satfile,header=head[1][-1],sep='\s+',
                                skipfooter=(total_len-heads[head[0]+1][0]+
                                         feet[head[0]+1]),engine='python',
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                        df.columns = pd.Index(headerlines[head[0]].split())
                        nametag = pd.Series({'name':name+'_ELEC'})
                        dflist.append(df.append(nametag,ignore_index=True))
                        print('{} loaded'.format(nametag))
                    '''
                    df = pd.read_csv(satfile, header=heads[-1][-1],
                                     engine='python', sep='\s+',
                                     skipfooter=feet[-1],
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                    df.columns = pd.Index(headerlines[-1].split())
                    df = df.rename(columns={'Time [UT]':'UT'})
                    if satfile.find('FGM')!=-1:
                        name = name+'_FGM'
                    elif satfile.find('MOM')!=-1:
                        name = name+'_MOM'
                    nametag = pd.Series({'name':name})
            else:
                df = pd.DataFrame()
            dflist.append(df.append(nametag,ignore_index=True))
            print('{} loaded'.format(nametag))
    return dflist
def simdata_to_df(simdict, satkey):
    """Function returns data frame using dict to find file and satkey for
        which satellite data to pull
    Inputs
        simdict- dictionary with filepaths for different satellites
        satkey- satellite key for dictionary
    Outputs
        dflist
    """
    dflist = []
    for satfile in simdict.get(satkey,[]):
        if os.path.exists(satfile):
            df = pd.read_csv(satfile, sep='\s+', skiprows=1,
                          parse_dates={'Time [UTC]':['year','mo','dy','hr',
                                                         'mn','sc','msc']},
                          date_parser=datetimeparser,
                          infer_datetime_format=True, keep_date_col=True)
            df['Time [UTC]'] = df['Time [UTC]']+dt.timedelta(minutes=45)
            name = satfile.split('/')[-1].split('_')[1]
            nametag = pd.Series({'name':name})
            dflist.append(df.append(nametag,ignore_index=True))
    return dflist
def interp_combine_dfs(from_df,to_df, from_xkey, to_xkey, *,
                       using_datetimes=True):
    """Function combines two dataframes by interpolating based on given
        x_keys and adding remaining columns from the from_df to to_df
    Inputs
        from_df, to_df- pandas dataframe objects to be combined
        from_xkey, to_xkey- keys used to ID column to interpolate on
        using_datetimes- default true, otw interp should be easier
    Outputs
        combo_df- combined dataframe
    """
    #sort by given keys
    from_df = from_df.sort_values(by=from_xkey)
    to_df = to_df.sort_values(by=to_xkey)
    #pop name to give back later
    name, _ = from_df.pop('name'), to_df.pop('name')
    from_df = from_df.dropna().reset_index(drop=True)
    to_df = to_df.dropna().reset_index(drop=True)
    if using_datetimes:
        #starting time
        starttime = min(to_df[to_xkey].loc[0], from_df[from_xkey].loc[0])
        dfs, xkeys = [from_df, to_df], [from_xkey, to_xkey]
        for df in enumerate([from_df, to_df]):
            #turn xdata into timedelta
            dfs[df[0]]['delta'] = df[1][xkeys[df[0]]]-starttime
        for key in from_df:
            if key!='UT':
                to_df[key]=np.interp(to_df['delta'].values.astype('float'),
                                   from_df['delta'].values.astype('float'),
                                   from_df[key].values.astype('float'))
        to_df['name'] = name.iloc[-1].split('_')[0]+'_obs'
    return to_df

def determine_satelliteIDs(pathtofiles, *,keylist=['cluster','geotail',
                                                 'goes','themis','rbspb']):
    """Function returns dict w/ sat filepaths for each type of satellite
    Inputs
        pathtofiles- can be simulation or observation files
        keylist- used to construct dict, what satellites are expected
    Outputs
        satfiledict- dict with entries for each type, none if not found
    """
    satfiledict = dict.fromkeys(keylist,[])
    for satfile in glob.glob(pathtofiles+'/*'):
        for satkey in keylist:
            if satfile.lower().find(satkey)!=-1:
                satfiledict.update({satkey:satfiledict[satkey]+[satfile]})
    return satfiledict

def todimensional(df):
    """Function modifies dimensionless variables -> dimensional variables
    Inputs
        dataset (frame.dataset)- tecplot dataset object
        kwargs:
    """
    out_df = df.copy(deep=True)
    proton_mass = 1.6605e-27
    cMu = pi*4e-7
    #SWMF sets up two conversions to go between
    # No (nondimensional) Si (SI) and Io (input output),
    # what we want is No2Io = No2Si_V / Io2Si_V
    #Found these conversions by grep'ing for 'No2Si_V' in ModPhysics.f90
    No2Si = {'X':6371*1e3,                             #planet radius
             'Y':6371*1e3,
             'Z':6371*1e3,
             'Rho':1e6*proton_mass,                    #proton mass
             'U_x':6371*1e3,                           #planet radius
             'U_y':6371*1e3,
             'U_z':6371*1e3,
             'P':1e6*proton_mass*(6371*1e3)**2,        #Rho * U^2
             'B_x':6371*1e3*sqrt(cMu*1e6*proton_mass), #U * sqrt(M*rho)
             'B_y':6371*1e3*sqrt(cMu*1e6*proton_mass),
             'B_z':6371*1e3*sqrt(cMu*1e6*proton_mass),
             'J_x':(sqrt(cMu*1e6*proton_mass)/cMu),    #B/(X*cMu)
             'J_y':(sqrt(cMu*1e6*proton_mass)/cMu),
             'J_z':(sqrt(cMu*1e6*proton_mass)/cMu)
            }
    #Found these conversions by grep'ing for 'Io2Si_V' in ModPhysics.f90
    Io2Si = {'X':6371*1e3,                  #planet radius
             'Y':6371*1e3,                  #planet radius
             'Z':6371*1e3,                  #planet radius
             'Rho':1e6*proton_mass,         #Mp/cm^3
             'U_x':1e3,                     #km/s
             'U_y':1e3,                     #km/s
             'U_z':1e3,                     #km/s
             'P':1e-9,                      #nPa
             'B_x':1e-9,                    #nT
             'B_y':1e-9,                    #nT
             'B_z':1e-9,                    #nT
             'J_x':1e-6,                    #microA/m^2
             'J_y':1e-6,                    #microA/m^2
             'J_z':1e-6,                    #microA/m^2
             }#'theta_1':pi/180,              #degrees
             #'theta_2':pi/180,              #degrees
             #'phi_1':pi/180,                #degrees
             #'phi_2':pi/180                 #degrees
            #}
    for var in out_df.keys():
        # Default is no change
        conversion = No2Si.get(var,1)/Io2Si.get(var,1)
        out_df[var] *= conversion
    return out_df

def read_satellites(pathtofiles):
    """
    Inputs
    Returns
    """
    virtualsats = {}
    obssats = {}
    # Glob files
    filelist = glob.glob(os.path.join(pathtofiles,'*.h5'))
    # Figure out which satellites we're looking for
    virtual_filename=[f for f in filelist if 'virtual' in f.split('/')[-1]][0]
    ###### Virtual Satellites
    virtualfile = pd.HDFStore(virtual_filename)
    satlist = [k.replace('/','') for k in virtualfile.keys()]
    # See if data is in dimensionless form
    for sat in satlist:
        df = virtualfile[sat]
        if 'time' in df:
            df.index = df['time']
            df.drop(columns=['time'],inplace=True)
        if df['B_x'].min()>-50:
            df = todimensional(df)
        virtualsats[sat] = df.sort_index()
        virtualsats[sat] = add_derived_variables2(virtualsats[sat],
                                                  sat,'virtual')
    virtualfile.close()
    endtime = virtualsats[sat].index[-1]
    ######
    ###### Real Satellites
    # Initialize a dict for each satellite
    for sat in satlist:
        obssats[sat] = {}
    # Gather, sort and organize data from each file
    for satfile in [f for f in filelist if 'virtual' not in f.split('/')[-1]]:
        # Extract useful information from the filename
        filename = satfile.split('/')[-1]
        satset,vartype = filename.replace('.h5','').split('_')
        print(filename,satset,vartype)
        # Read the file
        datafile = pd.HDFStore(pathtofiles+filename)
        for sat in datafile.keys():
            print('\t',sat)
            df = datafile[sat.replace('/','')]
            if 'time' in df:
                df.index = df['time']
                df.drop(columns=['time'],inplace=True)
                df.sort_index(inplace=True)
            obssats[sat.replace('/','').lower()][vartype]=df[df.index<endtime]
        datafile.close()
    for sat in obssats.keys():
        if 'bx' in obssats[sat]['pos'].keys():
            copy_b_comps=obssats[sat]['pos'][['bx','by','bz']].copy(deep=True)
            obssats[sat]['bfield'] = copy_b_comps
        if 'bfield' in obssats[sat] and 'plasma' in obssats[sat]:
            print('\t',sat)
            obssats[sat]['combined'] = combine_obs_sats(obssats[sat])
            obssats[sat] = add_derived_variables2(obssats[sat]['combined'],
                                                  sat,'combined')
            obssats[sat] = smooth_data(obssats[sat])
        else:
            obssats[sat] = pd.DataFrame()
    ######
    return virtualsats, obssats

if __name__ == "__main__":
    datapath = sys.argv[1]
    print('processing satellite output at {}'.format(datapath))
    obspath = os.path.join(datapath, 'observation')
    simpath = os.path.join(datapath, 'simulation')
    outpath = os.path.join(datapath, 'figures')
    obsdict = determine_satelliteIDs(obspath)
    simdict = determine_satelliteIDs(simpath)
    #geotail
    geo_dfs_sim = simdata_to_df(simdict, 'geotail')
    geo_dfs_sim = add_derived_variables(geo_dfs_sim)
    geo_dfs_obs = geotail_to_df(obsdict)
    [geo, geo_cross] = split_geotail(geo_dfs_obs)
    [sgeo, _] = split_geotail(geo_dfs_sim)
    #themis
    themis_dfs_sim = simdata_to_df(simdict, 'themis')
    themis_dfs_sim = add_derived_variables(themis_dfs_sim)
    themis_dfs_obs = themis_to_df(obsdict)
    [th_a,th_b,th_c,th_d,th_e,th_cross] = split_themis(themis_dfs_obs)
    [sth_a,sth_b,sth_c,sth_d,sth_e,_] = split_themis(themis_dfs_sim)
    #cluster
    cluster_dfs_sim = simdata_to_df(simdict, 'cluster')
    cluster_dfs_sim = add_derived_variables(cluster_dfs_sim)
    cluster_dfs_obs = cluster_to_df(obsdict)
    [cl1,cl2,cl3,cl4] = split_cluster(cluster_dfs_obs)
    [scl1,scl2,scl3,scl4] = split_cluster(cluster_dfs_sim)
    #omni
    index_data = read_indices(None, read_swmf=False,
                                              read_supermag=False)
    omni = index_data['omni']
    #Add derived variables for observation datasets
    [th_a,th_b,th_c,th_d,th_e] = add_derived_variables(
                                                [th_a[0],th_b[0],th_c[0],th_d[0],th_e[0]],
                                                obs=True)
    [cl1,cl2,cl3,cl4] = add_derived_variables([cl1[0],cl2[0],cl3[0],cl4[0]],obs=True)
    [geo] = add_derived_variables([geo[0]],obs=True)
    #combine simulation and observation into one list
    th_a = [th_a, sth_a[0]]
    th_b = [th_b, sth_b[0]]
    th_c = [th_c, sth_c[0]]
    th_d = [th_d, sth_d[0]]
    th_e = [th_e, sth_e[0]]
    geo = [geo, sgeo[0]]
    cl1 = [cl1, scl1[0]]
    cl2 = [cl2, scl2[0]]
    cl3 = [cl3, scl3[0]]
    cl4 = [cl4, scl4[0]]
    #set text settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"],
        "image.cmap": "twilight"})
    ######################################################################
    if True:
        from IPython import embed; embed()
    #B magnitude
    if True:
        figname = 'Bmagnitude'
        Bmag_themisBC, (axB,axC) = plt.subplots(nrows=2, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,8])
        Bmag_themisADE, (axA,axD,axE) = plt.subplots(nrows=3, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,12])
        Bmag_geo, (omniax, geoax1,clax4) = plt.subplots(nrows=3, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,12])
        ylabel = r'$B \left[nT\right]$'
        plot_Bmag(axA, th_a, ylabel)
        plot_Bmag(axB, th_b, ylabel, ylim=[0,40])
        plot_Bmag(axC, th_c, ylabel, ylim=[0,40])
        plot_Bmag(axD, th_d, ylabel, ylim=[0,100], legend_loc='lower right')
        plot_Bmag(axE, th_e, ylabel, ylim=[0,150], legend_loc='lower left')
        plot_Bmag(omniax, [omni], ylabel, ylim=[0,20])
        plot_Bmag(geoax1, geo, ylabel, ylim=[0,40])
        plot_Bmag(clax4, cl4, ylabel, ylim=[0,75])
        probe = ['a','b','c','d','e']
        for ax in enumerate([axA,axB,axC,axD,axE]):
            mark_cross_themis(ax[1], th_cross[0], probe=probe[ax[0]])
            mark_times(ax[1])
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator())
            ax[1].set_xlabel(None)
        axC.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        axE.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        for ax in enumerate([omniax,geoax1,clax4]):
            mark_times(ax[1])
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator())
            ax[1].set_xlabel(None)
        clax4.set_xlabel(r'\textit{Time }$\left[\textit{UTC}\right]$')
        Bmag_themisBC.tight_layout(pad=1)
        Bmag_themisADE.tight_layout(pad=1)
        Bmag_geo.tight_layout(pad=1)
        #EPS
        Bmag_themisBC.savefig(outpath+'/{}_themisBC.eps'.format(figname))
        Bmag_themisADE.savefig(outpath+'/{}_themisADE.eps'.format(figname))
        Bmag_geo.savefig(outpath+'/{}_geo.eps'.format(figname))
        #TiFF
        Bmag_themisBC.savefig(outpath+'/{}_themisBC.tiff'.format(figname))
        Bmag_themisADE.savefig(outpath+'/{}_themisADE.tiff'.format(figname))
        Bmag_geo.savefig(outpath+'/{}_geo.tiff'.format(figname))
    ######################################################################
    # Betastar magnetosphere
    if False:
        figname = 'Magnetosphere'
        Mag_themis, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,20])
        Mag_geo, (geoax1) = plt.subplots(nrows=1, ncols=1,
                                                   figsize=[12,4])
        Mag_cluster, (clax1,clax2,clax3,clax4) = plt.subplots(nrows=4,
                                                   ncols=1,
                                                   sharex=True,
                                                   figsize=[12,16])
        Mag_themis.tight_layout(pad=2)
        Mag_geo.tight_layout(pad=2)
        Mag_cluster.tight_layout(pad=2)
        ylabel = r'$\textit{Magnetosphere}$'
        plot_Magnetosphere(ax1, th_a, ylabel)
        plot_Magnetosphere(ax2, th_b, ylabel)
        plot_Magnetosphere(ax3, th_c, ylabel)
        plot_Magnetosphere(ax4, th_d, ylabel)
        plot_Magnetosphere(ax5, th_e, ylabel)
        plot_Magnetosphere(geoax1, geo, ylabel)
        plot_Magnetosphere(clax1, cl1, ylabel)
        plot_Magnetosphere(clax2, cl2, ylabel)
        plot_Magnetosphere(clax3, cl3, ylabel)
        plot_Magnetosphere(clax4, cl4, ylabel)
        for ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
            mark_cross_themis(ax[1], th_cross[0], probe=probe[ax[0]])
            mark_times(ax[1])
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        for ax in enumerate([geoax1]):
            mark_cross_geotail(ax[1], geo_cross[0])
            mark_times(ax[1])
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        for ax in enumerate([clax1, clax2, clax3, clax4]):
            mark_times(ax[1])
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        Mag_themis.savefig(outpath+'/{}_themis.eps'.format(figname))
        Mag_geo.savefig(outpath+'/{}_geo.eps'.format(figname))
        Mag_cluster.savefig(outpath+'/{}_cluster.eps'.format(figname))
    ######################################################################
    # Betastar magnetosphere
    if False:
        figname = 'Betastar'
        Betastar, (axA,axD,axE,clax4) = plt.subplots(nrows=4, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,12])
        ylabel = r'$\displaystyle \beta^{*}$'
        plot_Betastar(axA, th_a, ylabel)
        plot_Betastar(axD, th_d, ylabel)
        plot_Betastar(axE, th_e, ylabel)
        plot_Betastar(clax4, cl4, ylabel)
        probe = ['a','d','e']
        for ax in enumerate([axA,axD,axE]):
            mark_cross_themis(ax[1], th_cross[0], probe=probe[ax[0]])
            mark_times(ax[1])
            ax[1].axhline(0.7, color='black', linestyle=None, linewidth=1)
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
        for ax in enumerate([clax4]):
            mark_times(ax[1])
            ax[1].axhline(0.7, color='black', linestyle=None, linewidth=1)
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H:%M'))
            ax[1].tick_params(which='major', length=7)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
        Betastar.savefig(outpath+'/{}.eps'.format(figname))
    ######################################################################
    # Betastar magnetosphere
    if False:
        figname = 'BetastarScatter'
        Betastar, ([ax1,ax2],[ax3,ax4]) = plt.subplots(nrows=2, ncols=2,
                                                 sharex=True,sharey=True,
                                                   figsize=[12,12])
        #ylabel = r'$\displaystyle \beta^{*}$'
        '''
        cmap = mpl.cm.Purples
        norm = mpl.colors.Normalize(vmin=th_a[1]['Time [UTC]'][0:1].values,
                                vmax=th_a[1]['Time [UTC]'][-2:-1].values)
        Betastar.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=ax4,orientation='vertical',label='Time [UTC]')
        '''
        #mpl.rc('image', cmap='Purples')
        plot_BetastarScatter(ax1, th_a)
        plot_BetastarScatter(ax2, th_d)
        plot_BetastarScatter(ax3, th_e)
        plot_BetastarScatter(ax4, cl4)
        probe = ['a','d','e']
        for ax in enumerate([ax1,ax2,ax3,ax4]):
            #ax[1].tick_params(which='major', length=11)
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
            #Box around target zone
            ax[1].hlines([0.6,0.8],0.6,0.8)
            ax[1].vlines([0.6,0.8],0.6,0.8)
            #Dashed lines showing < targets
            ax[1].hlines([0.6,0.8],0,0.6, ls='--', color='green')
            ax[1].vlines([0.6,0.8],0,0.6, ls='--', color='green')
            #Dashed lines showing > targets
            ax[1].hlines([0.6,0.8],0.8,2, ls='--', color='red')
            ax[1].vlines([0.6,0.8],0.8,2, ls='--', color='red')
        #plt.colorbar(ax=ax4)
        Betastar.savefig(outpath+'/{}.eps'.format(figname))
    ######################################################################
