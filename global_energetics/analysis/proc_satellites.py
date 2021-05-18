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
import swmfpy
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
#interpackage imports
from global_energetics.analysis.proc_temporal import read_energetics
from global_energetics.analysis.proc_indices import (read_indices,
                                                     datetimeparser,
                                                     datetimeparser2,
                                                     datetimeparser3,
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
        axis.axvline(cross)
def plot_Magnetosphere(axis, dflist, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,
             use_inner=False, use_shield=False):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise']
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('FGM')!=-1:
            probe = name.split('_FGM')[0].split('TH')[-1]
            pass
            #axis.plot(df['UT'],df['FGS-D_B_TOTAL'],
            #          label=r'\textit{Themis}$\displaystyle_'+probe+'$',
            #          linewidth=Size, linestyle=ls,color='coral')
        elif any([name.find(key)!=-1 for key in simkeys]):
            probe = name.split('themis')[-1].upper()
            axis.plot(df['Time [UTC]'],df['Magnetosphere_state'],
                      label=r'\textit{SimThemis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='lightsteelblue')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        #axis.set_ylim([0,200])
        pass
    axis.set_xlabel(r'\textit{Time (UT)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left', facecolor='gray')
def plot_Bmag(axis, dflist, ylabel, *,
             xlim=None, ylim=None, Color=None, Size=4, ls=None,
             use_inner=False, use_shield=False):
    """Function plots B field magnitude for trajectories
    Inputs
        axis- object plotted on
        dflist- datasets
        dflabels- labels used for legend
        timekey- used to located column with time and the qt to plot
        ylabel, xlim, ylim, Color, Size, ls,- plot/axis settings
    """
    simkeys = ['themisa','themisb','themisc','themisd','themise']
    for df in dflist:
        name = df['name'].iloc[-1]
        if name.find('FGM')!=-1:
            probe = name.split('_FGM')[0].split('TH')[-1]
            axis.plot(df['UT'],df['FGS-D_B_TOTAL'],
                      label=r'\textit{Themis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='coral')
        elif any([name.find(key)!=-1 for key in simkeys]):
            probe = name.split('themis')[-1].upper()
            axis.plot(df['Time [UTC]'],df['Bmag [nT]'],
                      label=r'\textit{SimThemis}$\displaystyle_'+probe+'$',
                      linewidth=Size, linestyle=ls,color='lightsteelblue')
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        axis.set_ylim([0,200])
    axis.set_xlabel(r'\textit{Time (UT)}')
    axis.set_ylabel(ylabel)
    axis.legend(loc='upper left', facecolor='gray')
def add_derived_variables(dflist):
    """Function adds columns of data by performing simple operations
    Inputs
        dflist- dataframe
    Outputs
        dflist- dataframe with modifications
    """
    for df in enumerate(dflist):
        if not df[1].empty:
            ###B field
            B = sqrt(df[1]['Bx']**2+df[1]['By']**2+df[1]['Bz']**2)
            dflist[df[0]]['Bmag [nT]'] = B
            ###Flow field
            U = sqrt(df[1]['Ux']**2+df[1]['Uy']**2+df[1]['Uz']**2)
            dflist[df[0]]['Umag [km/s]'] = U
            ###Betastar
            Dp = df[1]['Rho']*1e6*1.6605e-27*U**2*1e6*1e9
            dflist[df[0]]['Betastar'] =(df[1]['P']+0.5*Dp)/(
                                        B**2/(2*4*pi*1e-7)*1e-9)
            ###Magnetosphere state
            state=[]
            for index in df[1].index:
                if dflist[df[0]]['Betastar'].iloc[index]>0.7:
                    print('High betastar!')
                    if df[1]['status'].iloc[index]!=1:
                        state.append(0)
                    else:
                        state.append(1)
                else:
                    state.append(1)
            dflist[df[0]]['Magnetosphere_state']=state
    return dflist
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
        elif (name.find('THB')!=-1) or (name.find('themisb')!=-1):
            themis_b.append(df)
        elif (name.find('THC')!=-1) or (name.find('themisc')!=-1):
            themis_c.append(df)
        elif (name.find('THD')!=-1) or (name.find('themisd')!=-1):
            themis_d.append(df)
        elif (name.find('THE')!=-1) or (name.find('themise')!=-1):
            themis_e.append(df)
        elif (name.find('crossing')!=-1):
            cross.append(df)
    return themis_a, themis_b, themis_c, themis_d, themis_e, cross
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
            name = satfile.split('/')[-1].split('_')[1]
            if satfile.find(crosskey)!=-1:
                print("""
                df = pd.read_csv(satfile, header=42, sep='\s+',
                                parse_dates={'UT':['TIMESTAMP']},
                                date_parser=datetimeparser3,
                                infer_datetime_format=True,index_col=False)
                """)
                df = pd.read_csv(satfile, header=42, sep='\s+',
                                parse_dates={'UT':['TIMESTAMP']},
                                date_parser=datetimeparser3,
                                infer_datetime_format=True,index_col=False)
                nametag = pd.Series({'name':(name+'_crossings').lower()})
            elif (satfile.find('MOM')!=-1) or (satfile.find('FGM')!=-1):
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
                    #last dataset in the file
                    if len(heads)==1:
                        name = name+'_FGM'
                    else:
                        name = name+'_IONS'
                    df = pd.read_csv(satfile, header=heads[-1][-1],
                                     engine='python', sep='\s+',
                                     skipfooter=feet[-1],
                                parse_dates={'Time [UT]':['dd-mm-yyyy',
                                                          'hh:mm:ss.ms']},
                                date_parser=datetimeparser2,
                                infer_datetime_format=True,index_col=False)
                    df.columns = pd.Index(headerlines[-1].split())
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
            name = satfile.split('/')[-1].split('_')[1]
            nametag = pd.Series({'name':name})
            dflist.append(df.append(nametag,ignore_index=True))
    return dflist
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
if __name__ == "__main__":
    datapath = sys.argv[1]
    print('processing satellite output at {}'.format(datapath))
    obspath = os.path.join(datapath, 'observation')
    simpath = os.path.join(datapath, 'simulation')
    outpath = os.path.join(datapath, 'figures')
    obsdict = determine_satelliteIDs(obspath)
    simdict = determine_satelliteIDs(simpath)
    themis_dfs_sim = simdata_to_df(simdict, 'themis')
    themis_dfs_sim = add_derived_variables(themis_dfs_sim)
    themis_dfs_obs = themis_to_df(obsdict)
    [th_a,th_b,th_c,th_d,th_e,th_cross] = split_themis(themis_dfs_obs)
    [sth_a,sth_b,sth_c,sth_d,sth_e,_] = split_themis(themis_dfs_sim)
    print('length of sth_a: {}'.format(len(sth_a)))
    #set text settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 18,
        "font.sans-serif": ["Helvetica"]})
    ######################################################################
    #B magnitude
    if True:
        figname = 'Bmagnitude'
        Bmag, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,20],
                                                   facecolor='gainsboro')
        Bmag.tight_layout(pad=2)
        ylabel = r'$\displaystyle B_{mag} (nT)$'
        th_a.extend(sth_a)
        th_b.extend(sth_b)
        th_c.extend(sth_c)
        th_d.extend(sth_d)
        th_e.extend(sth_e)
        plot_Bmag(ax1, th_a, ylabel)
        plot_Bmag(ax2, th_b, ylabel, ylim=[0,40])
        plot_Bmag(ax3, th_c, ylabel, ylim=[0,40])
        plot_Bmag(ax4, th_d, ylabel)
        plot_Bmag(ax5, th_e, ylabel)
        probe = ['a','b','c','d','e']
        for ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
            ax[1].set_facecolor('olive')
            mark_cross_themis(ax[1], th_cross[0], probe=probe[ax[0]])
        Bmag.savefig(outpath+'/{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
    # Betastar magnetosphere
    if True:
        figname = 'Betastar'
        Bmag, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, ncols=1,
                                                   sharex=True,
                                                   figsize=[12,20],
                                                   facecolor='gainsboro')
        Bmag.tight_layout(pad=2)
        ylabel = r'$\displaystyle Magnetosphere_{\beta ^*}$'
        plot_Magnetosphere(ax1, th_a, ylabel)
        plot_Magnetosphere(ax2, th_b, ylabel)
        plot_Magnetosphere(ax3, th_c, ylabel)
        plot_Magnetosphere(ax4, th_d, ylabel)
        plot_Magnetosphere(ax5, th_e, ylabel)
        probe = ['a','b','c','d','e']
        for ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
            ax[1].set_facecolor('olive')
            mark_cross_themis(ax[1], th_cross[0], probe=probe[ax[0]])
        Bmag.savefig(outpath+'/{}.png'.format(figname),
                      facecolor='gainsboro')
    ######################################################################
