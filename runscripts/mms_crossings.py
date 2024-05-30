#/usr/bin/env python
"""accesses MMS data from nasa CDA and creates plots for SWMF comparisons
"""
#General file IO/debugging
import os,sys,time,glob
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
#The standards for math and data handling
import datetime as dt
import numpy as np
from numpy import pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
#import scipy as sp
#Pandas *shrugs* because its easy
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,
                               AutoLocator, FuncFormatter)
#NASA tools
from cdasws import CdasWs
cdas = CdasWs()
from sscws.sscws import SscWs
ssc = SscWs()
from sscws.coordinates import CoordinateSystem as coordsys
#Geopack for coord transforms
from geopack import geopack as gp
#Custom packages for calling CDAweb/OMNI + post processing
import swmfpy
from global_energetics.wind_to_swmfInput import (collect_themis,collect_mms)
from global_energetics.extract.shue import r0_alpha_1998
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup)
from global_energetics.analysis.proc_indices import (ID_ALbays)
from global_energetics.analysis.proc_satellites import(add_derived_variables2)

def datetimeparser(datetimestring):
    return dt.datetime.strptime(datetimestring,'%Y %m %d %H %M')

def read_substorm_list(infile,**kwargs):
    """
    """
    substorms = pd.read_csv(infile,skiprows=range(0,37),sep='\t',
                            parse_dates={'times':['<year>','<month>','<day>',
                                                  '<hour>','<min>']},
                            date_parser=datetimeparser,
                            infer_datetime_format=True)
    return substorms

def get_ss_matches(crossings,substorms):
    crossings['ss'] = False
    for i,ID in enumerate(crossings['ID']):
        misc = crossing_misc.loc[ID]
        start = dt.datetime.fromisoformat(misc['DateStart'])
        i_closest=np.where(abs(substorms['times']-start)==
                           abs(substorms['times']-start).min())
        #ss_mindt = (substorms['times'][i_closest]-start).total_seconds()
        ss_mindt=(substorms.loc[i_closest,'times']-start).max().total_seconds()
        #ss_mindt = abs(substorms['times']-start).min().total_seconds()
        crossings.loc[i,'ss'] = ss_mindt
    return crossings

def read_fastflow_list(inpath,**kwargs):
    """
    """
    full_list = glob.glob(inpath+'/*.txt')
    read_list = [f for f in full_list if '2015' in f or
                                         '2016' in f or
                                         '2017' in f or
                                         '2018' in f]
    ffs = list(np.zeros(len(read_list)))
    for i,infile in enumerate(read_list):
        ff = pd.read_csv(infile,sep='\s+')
        ff.index = [dt.datetime.fromisoformat(s) for s in
                                                      ff['Time_of_fast_flow']]
        ffs[i] = ff
    fastflows = pd.concat(ffs)
    return fastflows

def get_ff_matches(crossings,fastflows):
    crossings['ff'] = False
    for i,ID in enumerate(crossings['ID']):
        misc = crossing_misc.loc[ID]
        start = dt.datetime.fromisoformat(misc['DateStart'])
        ff_mindt = abs(fastflows.index-start).min().total_seconds()
        #if ff_mindt < dt.timedelta(seconds=300):
        #    crossings.loc[i,'ff'] = True
        crossings.loc[i,'ff'] = ff_mindt
    return crossings

def test_THEMIS(pos_data,**kwargs):
    """Function to test if THEMIS data is present in the tail
    Inputs
        pos_data
    Returns
        good_placement (bool)
    """
    good_placement = False
    print('spacecraft\ty_position:')
    print('*******************************')
    for sc in pos_data.keys():
        if not pos_data[sc]['y'].empty:
            x = pos_data[sc]['x']
            y = pos_data[sc]['y']
            pos_data[sc]['r'] = np.sqrt(x**2+y**2)
            pos_data[sc]['mlt'] = (12+np.rad2deg(np.arctan2(y,x))*12/180)%24
            print('\t',sc,'X\t','{:.2f}'.format(x[0]))
            print('\t',sc,'Y\t','{:.2f}'.format(y[0]))
            print('\t',sc,'R\t','{:.2f}'.format(pos_data[sc]['r'][0]))
            print('\t',sc,'MLT\t','{:.2f}\n'.format(pos_data[sc]['mlt'][0]))
            #if any([(y>-20)&(y<20) for y in pos_data[sc]['y'].values]):
            #    print('\t',sc,'X\t','{:.2f}'.format(pos_data[sc]['x'][0]))
            #    if any([(x>-20)&(x<-6) for x in pos_data[sc]['x'].values]):
            #        good_placement = True
            if any([(r>6)&(r<20) for r in pos_data[sc]['r'].values]):
                if any([(mlt>19)|
                        (mlt<5) for mlt in pos_data[sc]['mlt'].values]):
                    good_placement = True
        else:
            print('\t',sc,'\t','no data')
    print('placement: ',good_placement)
    return good_placement


def nan_help(y):#thanks stack exchange
    return np.isnan(y), lambda z:z.nonzero()[0]

def upgrade_OMNI(omni,**kwargs):
    """Function adds derived variables to omni dataset
    Inputs
        omni
    Returns
        omni
    """
    t0 = dt.datetime(1970,1,1)
    # Clean data (linear interpolate over data gaps)
    omni['dt'] = [(f-b).seconds for f,b in
                                zip(omni['times'][1::],omni['times'][0:-1])]
    for key in ['bx','by_gse','bz_gse',
                'vx_gse','vy_gse','vz_gse',
                'density','b','v','sym_h','dst']:
        if key in omni.keys():
            nans, x= nan_help(omni[key])
            omni[key][nans]= np.interp(x(nans), x(~nans), omni[key][~nans])
    # Get GSM variables
    for gse_keys,gsm_keys in [[['bx','by_gse','bz_gse'],
                               ['bx','by','bz']],
                              [['vx_gse','vy_gse','vz_gse'],
                               ['vx','vy','vz']]]:
        if gse_keys[0] in omni.keys():
            gse = [omni[gse_keys[0]],omni[gse_keys[1]],omni[gse_keys[1]]]
        else:
            continue
        x,y,z=np.zeros(len(gse[0])),np.zeros(len(gse[0])),np.zeros(len(gse[0]))
        for i,t in enumerate(omni['times']):
            ut = (t-t0).total_seconds()
            gp.recalc(ut)
            gsm = gp.gsmgse(gse[0][i],gse[1][i],gse[2][i],-1)
            x[i],y[i],z[i] = gsm
        omni[gsm_keys[0]] = x
        omni[gsm_keys[1]] = y
        omni[gsm_keys[2]] = z
    # Dynamic pressure
    convert = 1.6726e-27*1e6*(1e3)**2*1e9
    omni['pdyn'] = omni['density']*omni['v']**2*convert
    # Shue model
    omni['r_shue98'],omni['alpha'] = (r0_alpha_1998(omni['bz'],omni['pdyn']))
    # Coupling functions
    Cmp = 1000 #Followed
    #           https://supermag.jhuapl.edu/info/data.php?page=swdata
    #           term comes from Cai and Clauer[2013].
    #           Note that SuperMAG lists the term as 100,
    #           however from the paper: "From our work,
    #                                   α is estimated to be on order of 10^3"
    omni['B_T'] = np.sqrt((omni['by']*1e-9)**2+(omni['bz']*1e-9)**2)
    omni['M_A'] = (np.sqrt(omni['pdyn']*1e-9*(4*pi*1e-7))
                    /omni['B_T'])
    omni['clock'] = np.arctan2(omni['by'],omni['bz'])
    omni['Newell']=Cmp*((omni['v']*1e3)**(4/3)*np.sqrt(
                (omni['by']*1e-9)**2+(omni['bz']*1e-9)**2)**(2/3)*
                            abs(np.sin(omni['clock']/2))**(8/3))
    l = 7*6371*1000
    omni['eps'] = (omni['b']**2*omni['v']*
                                np.sin(omni['clock']/2)**4*l**2*
                                        1e3*1e-9**2 / (4*pi*1e-7))
    #Wang2014
    omni['EinWang'] = (3.78e7*omni['density']**0.24*
                        omni['v']**1.47*(omni['B_T']*1e9)**0.86*
                        (abs(sin(omni['clock']/2))**2.70+0.25))
    #Tenfjord2013
    if 'vx' not in omni.keys():
        omni['vx'] = omni['v']
    omni['Pstorm'] = (omni['B_T']**2*omni['vx']*1e3/(4*pi*1e-7)*
                    omni['M_A']*abs(sin(omni['clock']/2))**4*
                    135/(5e-5*omni['bz']**3+1)*6371e3**2)
    return omni

def classify_OMNI(omni,start,end,**kwargs):
    """Function that classifies upstream sw from OMNI data
    Inputs
        omni
    Returns
        driving(str('quiet','low','med','high','failed','strange'))
        energized(str('quiet','low','med','high','failed','strange'))
        storm(str('quiet','sc','main','recovery','failed','strange'))
        substorm(str('quiet','psuedo','substorm','failed','strange'))
    """
    #############
    # Set Limits
    #############
    quiet_driving_limit = 0.25e12 #TW of energy input from coupling fnc
    low_driving_limit = 7.5e12
    med_driving_limit = 15e12 #> is 'highdriving'

    unenergized_limit = -10  #nT of dst representing energy state
    lowenergized_limit = -50
    medenergized_limit = -150 #> is 'highenergized'
    #############
    # 
    #############

    window = [(t<=end)&(t>=start) for t in omni['times']]
    if (not any(window) and any([t>=start for t in omni['times']]) and
                            any([t<=end for t in omni['times']])):
        iend = np.where([t>=start for t in omni['times']])[0][0]
        istart = iend-1
    else:
        istart = np.where(window)[0][0]
        iend = np.where(window)[0][-1]+1
    #Coupling
    Ein = np.mean(omni['EinWang'][istart:iend])
    driving = Ein
    '''
    if Ein <= quiet_driving_limit:
        driving = 'quiet'
    elif Ein <= low_driving_limit:
        driving = 'low'
    elif Ein <= med_driving_limit:
        driving = 'med'
    elif np.isnan(Ein):
        driving = 'failed'
    elif Ein > med_driving_limit:
        driving = 'high'
    else:
        driving = 'strange'
    '''
    #Energization
    if 'sym_h' in omni.keys():
        symh = omni['sym_h'][istart:iend]
    else:
        symh = omni['dst'][istart:iend]
    symh_ave = np.mean(symh)
    energized = symh_ave*-8e13
    '''
    if symh_ave >= unenergized_limit:
        energized = 'quiet'
    elif symh_ave >= lowenergized_limit:
        energized = 'low'
    elif symh_ave >= medenergized_limit:
        energized = 'med'
    elif np.isnan(symh_ave):
        energized = 'failed'
    elif symh_ave < medenergized_limit:
        energized = 'high'
    else:
        energized = 'strange'
    '''
    #Storm
    # Is there a storm anywhere in this interval?
    if 'sym_h' in omni.keys():
        anyStorm = omni['sym_h'].min() < -50
        # find min location
        i_min = np.where(omni['sym_h']==omni['sym_h'].min())[0]
    else:
        anyStorm = omni['dst'].min() < -50
        # find min location
        i_min = np.where(omni['dst']==omni['dst'].min())[0]
    if not anyStorm:
        storm = 'quiet'
    else:
        # find local 1hr slope
        if 'sym_h' in omni.keys():
            sym_1hr = omni['sym_h'][istart-30:iend+30]
            dt_1hr = omni['dt'][istart-30:iend+30]
            slope_1hr = np.mean((sym_1hr[1::]-sym_1hr[0:-1])/dt_1hr[0:-1])
        else:
            sym_1hr = omni['dst'][istart-1:iend+1]
            dt_1hr = omni['dt'][istart-1:iend+1]
            slope_1hr = np.mean((sym_1hr[1::]-sym_1hr[0:-1])/dt_1hr[0:-1])
        # Now find out when it is relative to our current position
        if i_min[0] > iend:
            if symh_ave < unenergized_limit:
                storm = 'main'
            else:
                storm = 'quiet'
            '''
            # The end of main phase prob hasn't happened yet
            # So let's check the current level and local slope
            if symh_ave < unenergized_limit and slope_1hr < 0:
                storm = 'main'
            elif symh_ave < unenergized_limit and slope_1hr > 0:
                storm = 'strange'
            elif symh_ave > 10:
                storm = 'ssc' #Wow!
            else:
                storm = 'quiet'
            '''
        elif i_min[-1] < istart:
            if symh_ave < unenergized_limit:
                storm = 'recovery'
            else:
                storm = 'quiet'
            '''
            # The end of main phase prob already ended
            # So let's check the current level and local slope
            if symh_ave < unenergized_limit and slope_1hr > 0:
                storm = 'recovery'
            elif symh_ave < unenergized_limit and slope_1hr < 0:
                storm = 'strange'
            elif symh_ave >= unenergized_limit:
                storm = 'quiet'
            else:
                storm = 'strange'
            '''
        else: #our crossing window is between minimums? I guess use slope..
            if slope_1hr <= 0:
                storm = 'main'
            elif slope_1hr > 0:
                storm = 'recovery'
            #else:
            #    storm = 'strange'

    #Substorm
    albays, onsets, psuedos, substorms = ID_ALbays(omni,al_series='al')
    if any(substorms[istart-10:iend+10]):
        substorm = 'substorm'
    elif any(psuedos[istart-10:iend+10]):
        substorm = 'psuedo'
    else:
        substorm = 'quiet'

    #Convection
    pci = omni['pc_n']
    T = 2*np.pi*(start.month/12)
    cpcp = 29.28 - 3.31*np.sin(T+1.49)+17.81*pci
    convection = np.mean(cpcp[istart:iend])

    return driving, energized, storm, substorm, convection

#TODO create function that classifies OMNI data in terms of storm
#   Inputs
#       omni = dict{str(key):np.array(values)}
#   Returns
#       eventtype(str('main','recovery','quiet','failed','strange'))

#TODO create function that classifies THEMIS + OMNI data in terms of tail
#   Inputs
#       themis = dict{str(key):np.array(values)}
#       omni = dict{str(key):np.array(values)}
#   Returns
#       eventtype(str('onset','growth','expansion','quiet','failed','strange'))

#TODO helper function merges arrays with different time cadences
#   Inputs
#       times1,array1,times2,array2
#   Returns
#       array3 (resampled down array2)

def count_bins(x,y,**kwargs):
    factor = kwargs.get('factor',1)
    #xbins = np.linspace(-20,15,36)
    #ybins = np.linspace(-20,20,41)
    xbins = kwargs.get('xbins',np.linspace(20,0,21))
    ybins = kwargs.get('ybins',np.linspace(20,-20,41))
    counts = np.zeros([len(xbins),len(ybins)])
    for i,xb in enumerate(xbins[0:-1]):
        for j,yb in enumerate(ybins[0:-1]):
            counts[i,j] = factor*len(np.where((x>xbins[i+1])&(x<xbins[i])&
                                              (y>ybins[j+1])&(y<ybins[j]))[0])
    return counts

def plot_coverage(crossings,path,**kwargs):
    #############
    #setup figure
    cover,[day,tail]=plt.subplots(2,1,figsize=[15,15])
    # mms crossing counts
    mmsX = crossings['X'].values
    mmsY = crossings['Y'].values
    mmsCounts = count_bins(mmsX,mmsY)
    # themis crossing counts
    all_x = np.array([])
    all_y = np.array([])
    for sc in ['thA','thD','thE']:
        all_x = np.append(all_x,crossings[sc+'x'].values)
        all_y = np.append(all_y,crossings[sc+'y'].values)
    thCounts = count_bins(all_x,all_y,factor=1,
                          xbins=np.linspace(0,-20,21))

    # Creat plot
    c1 = day.imshow(mmsCounts,extent=[20,-20,0,20],cmap='inferno',
                       alpha=1)
    c2 = tail.imshow(thCounts,extent=[20,-20,-20,0],cmap='inferno',
                       alpha=1,vmax=mmsCounts.max())
    # Decorations
    cbar1 = cover.colorbar(c1)
    cover.colorbar(c2)
    cbar1.set_label('Counts')
    day.xaxis.tick_top()
    day.grid(c='grey')
    tail.grid(c='grey')
    #day.set_xlabel('MMS Magnetopause Crossings')
    tail.set_xlabel(r'$Y\left[ R_E\right]$')
    tail.set_ylabel(r'$X\left[ R_E\right]$')
    day.set_ylabel(r'$X\left[ R_E\right]$')
    cover.tight_layout()
    # Save
    figurename = (path+'/heatmap.svg')
    cover.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(cover)

def plot_histograms(crossings,path,**kwargs):
    #############
    #setup figure
    #hists,[[drive,energy],[storm,substorm],[convection,bursts]]=plt.subplots(
    #                                                      3,2,figsize=[18,24],
    #                                                             sharex=False)
    hists,[[drive,storm,convection],[energy,substorm,bursts]]=plt.subplots(
                                                          2,3,figsize=[24,18],
                                                                 sharex=False)
    # driving levels
    drive_values,drive_bins,drive_bars = drive.hist(
                           crossings['driving']/1e12,color='orange',ec='black')
    # energization levels
    energy_values,energy_bins,energy_bars = energy.hist(
                          crossings['energized']/-8e13,color='black',ec='grey')
    # storm conditions
    storm_values,storm_bins,storm_bars = storm.hist(crossings['storm'],
                                                  bins=3,
                                                  color='lightblue',ec='black')
    # substorm conditions
    sub_values,sub_bins,sub_bars = substorm.hist(crossings['substorm'],
                                                    bins=3,
                                                        color='red',ec='black')
    #substorm.hist(crossings['ss'],
    #                bins=np.linspace(-360,360,7),
    #                color='red',ec='black')
    # convection conditions
    conv_values,conv_bins,conv_bars = convection.hist(
                            crossings['convection'],bins=np.linspace(0,200,9),
                            color='gold',ec='black')
    # flow bursts
    flow_values,flow_bins,flow_bars = bursts.hist(crossings['ff'],
                                                  bins=np.linspace(0,720,7),
                                                 #bins=np.logspace(1,5,12),
                                                  color='blue',ec='black')
    # cuttoff markers
    #Decorations
    for ax,bars in zip([drive,energy,storm,substorm,convection,bursts],
            [drive_bars,energy_bars,storm_bars,sub_bars,conv_bars,flow_bars]):
        ax.set_ylabel('Counts')
        ax.bar_label(bars)
    drive.set_title(r'Driving',weight='bold')
    drive.set_xlabel(r'$E_{in}\left[ TW\right]$')
    energy.set_title(r'Energization',weight='bold')
    energy.set_xlabel(r'$SYM_H$ $\Delta B \left[ nT\right]$')
    storm.set_title('Storm Phase',weight='bold')
    substorm.set_title(r'Substorm',weight='bold')
    convection.set_title(r'Convection',weight='bold')
    convection.set_xlabel(r'CPCP $\left[ kV\right]$')
    bursts.set_title(r'Flow Burst',weight='bold')
    bursts.set_xlabel(r'Proximity $\left[ s\right]$')
    hists.tight_layout()
    #general_plot_settings(drive,do_xlabel=False,legend=True,
    #                      ylabel=r'$B \left[nT\right]$',
    #                    #xlim=[event['mms_b'].index[0],
    #                    #    event['mms_b'].index[0]+dt.timedelta(seconds=60)],
    #                      timedelta=False)
    # Save
    figurename = path+'/hists.svg'
    hists.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(hists)

def plot_timeseries(event,path,**kwargs):
    #############
    #setup figure
    tseries,[[mvab,sw],[flux,al]] = plt.subplots(2,2,figsize=[36,18],
                                                 sharex=False)
    # MVA plot to prove we got the crossing
    mvab.plot(event['mms_b'].index,event['mms_b']['bm'],label=r'$B_M$',
                                                                c='grey')
    mvab.plot(event['mms_b'].index,event['mms_b']['bn'],label=r'$B_N$',
                                                                c='brown')
    mvab.plot(event['mms_b'].index,event['mms_b']['bl'],label=r'$B_L$',
                                                                c='lightblue')
    mvab.plot(event['mms_b'].index,event['mms_b']['bl'].rolling('5S').mean(),
                                            label=r'$B_{L,5s}$',lw=3,c='blue')
    # MHD Flux to show that we can do the calculation
    flux.plot(event['mms_mhd'].index,event['mms_mhd']['Km']/1e9,label=r'$K_M$',
              c='grey')
    flux.plot(event['mms_mhd'].index,event['mms_mhd']['Kn']/1e9,label=r'$K_N$',
              c='brown',lw=3)
    flux.plot(event['mms_mhd'].index,event['mms_mhd']['Kn_static']/1e9,
              label=r'$K_{N,Static}$',c='red',ls='--')
    flux.plot(event['mms_mhd'].index,event['mms_mhd']['Kl']/1e9,label=r'$K_L$',
              c='blue')
    # SW to give context to 'driving'
    sw_r = sw.twinx()
    sw.fill_between(event['omni']['times'],event['omni']['EinWang']/1e12,
                    label=r'$E_{in}$ Wang et al. 2014',color='orange',
                    ec='black')
    #sw_r.plot(event['omni']['times'],event['omni']['bx'],label=r'$B_X$',
    #        color='grey')
    #sw_r.plot(event['omni']['times'],event['omni']['by'],label=r'$B_Y$',
    #        color='brown')
    sw_r.plot(event['omni']['times'],event['omni']['bz'],label=r'$B_Z$',
            color='blue',lw=3)
    sw_r.set_ylabel(r'IMF $\left[nT\right]$')
    sw_r.legend(loc='lower right')
    # DST to give context to give context to 'energized' and 'storm'
    #dst.plot(event['omni']['times'],event['omni']['sym_h'],label='SYM-H',
    #         color='black',lw=3)
    # AL to give context to 'substorm'
    al.plot(event['omni']['times'],event['omni']['al'],label='AL',
             color='brown',lw=3)
    #Time markers
    mvab.axvline(event['BL50'],c='orange',lw=3)
    sw_r.axhline(0,c='black')
    for ax in [mvab,flux,sw,al]:
        ax.axvline(event['start'],c='black')
        ax.axvline(event['HTstart'],c='green',ls='--',lw=3)
        ax.axvline(event['HTend'],c='green',ls='--',lw=3)
        ax.margins(x=0.01)
    #Decorations
    med_zoom= [event['HTstart']-dt.timedelta(minutes=45),
               event['HTend']+dt.timedelta(minutes=45)]
    window = np.where([(t>med_zoom[0])&(t<med_zoom[1])
                                             for t in event['omni']['times']])
    general_plot_settings(mvab,do_xlabel=False,legend=True,
                          ylabel=r'$B \left[nT\right]$',
                          xlim=[event['start'],
                                event['HTend']+dt.timedelta(seconds=15)],
                          timedelta=False)
    general_plot_settings(flux,do_xlabel=False,legend=True,
                          ylabel=r'Energy Flux $K \left[GW/{R_E}^2\right]$',
                          xlim=[event['start'],
                                event['HTend']+dt.timedelta(seconds=15)],
                          timedelta=False)
    general_plot_settings(sw,do_xlabel=False,legend=True,
                          xlim=med_zoom,
                       ylim=[0,
                            event['omni']['EinWang'][window].max()/1e12*1.02],
                          ylabel=r'Energy Input $\left[ TW\right]$',
                          timedelta=False)
    #general_plot_settings(dst,do_xlabel=False,legend=True,
    #                      ylabel=r'$\Delta B \left[nT\right]$',
    #          xlim=[event['mms_mhd'].index[0]-dt.timedelta(seconds=(12*3600)),
    #               event['mms_mhd'].index[0]+dt.timedelta(seconds=(12*3600))],
    #                      ylim=[-150,10],
    #                      timedelta=False)
    general_plot_settings(al,do_xlabel=False,legend=True,
                          ylabel=r'$\Delta B \left[nT\right]$',
                          xlim=med_zoom,
                          ylim=[event['omni']['al'][window].min()*1.02,0],
                          timedelta=False)
    sw.spines['left'].set_color('tab:orange')
    sw.tick_params(axis='y',colors='tab:orange')
    sw.yaxis.label.set_color('tab:orange')
    sw.legend(loc='upper left')
    sw.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    al.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    flux.set_xlabel('Time [min:sec]')
    al.set_xlabel('Time [min:sec]')
    tseries.suptitle(event['ID'],fontsize=32,fontweight='bold',
                     x=0.99,y=0.99,
                     verticalalignment='top',
                     horizontalalignment='right')
    tseries.tight_layout()
    # Save
    figurename = path+'/exampletimeseries_'+event['ID']+'.svg'
    tseries.savefig(figurename)
    print('\033[92m Created\033[00m',figurename)
    plt.close(tseries)

#TODO eventtype_hists
#   Inputs
#       good/bad crossings
#   Returns
#       None (creates and saves plots)

#TODO SEA_analysis(crossing_subdict)
#   Inputs
#       dict with some of good_crossings
#   Returns
#       SEA products

#TODO plot_SEA
#   Inputs
#       SEA products
#   Returns
#       None (line plots of SEA)


def calc_LMN(bfield,vfield,normal,l,**kwargs):
    """Calcs LMN coords given bfield and normal direction
    Inputs
        bfield {}
        normal
        l
    Returns
        crossing
        BL_50, HT_start, HT_end
    """
    #NOTE now just assuming everything passed here is GSM coordinates
    cross_v = vfield.copy(deep=True)
    cross_b = bfield.copy(deep=True)
    # B
    cross_b['bn'] = (bfield['bx']*normal[0]+
                      bfield['by']*normal[1]+
                      bfield['bz']*normal[2])
    cross_b['bl'] = (bfield['bx']*l[0]+
                      bfield['by']*l[1]+
                      bfield['bz']*l[2])
    cross_b['bm'] = np.sqrt(bfield['b']**2-
                             cross_b['bn']**2-cross_b['bl']**2)
    # V
    cross_v['v'] = np.sqrt(vfield['vx']**2+vfield['vy']**2+vfield['vz']**2)
    cross_v['vn'] = (vfield['vx']*normal[0]+
                      vfield['vy']*normal[1]+
                      vfield['vz']*normal[2])
    cross_v['vl'] = (vfield['vx']*l[0]+
                      vfield['vy']*l[1]+
                      vfield['vz']*l[2])
    cross_v['vm'] = np.sqrt(cross_v['v']**2-
                             cross_v['vn']**2-cross_v['vl']**2)
    # Get BL 50% mark
    #TODO: update this to not find the 'closest' but to march instead
    runningBL = cross_b['bl'].rolling('5s').mean()
    i_min = np.where(runningBL==runningBL.min())[0][0]
    i_max = np.where(runningBL==runningBL.max())[0][0]
    bl76 = runningBL[min(i_min,i_max):max(i_min,i_max)+1]

    # 50%
    dB = runningBL.max()-runningBL.min()
    blmin = bl76.min()
    bl50_value = runningBL.min()+(runningBL.max()-runningBL.min())*0.5
    i_50 = np.where(abs(bl76-bl50_value)==abs(bl76-bl50_value).min())[0][0]
    bl50_time = bl76.index[i_50]
    if bl76[0]>bl76[-1]:
        down = bl76[i_50::].items()
        up = zip(reversed(bl76[0:i_50].index),reversed(bl76[0:i_50]))
    else:
        up = bl76[i_50::].items()
        down = zip(reversed(bl76[0:i_50].index),reversed(bl76[0:i_50]))
    # 38%
    for i,(t,value) in enumerate(down):
        percent = (value-blmin)/dB*100
        if percent<32:
            i_38 = i
            bl38_time = t
            break
    # 88%
    for i,(t,value) in enumerate(up):
        percent = (value-blmin)/dB*100
        if percent>88:
            i_88 = i
            bl88_time = t
            break


    '''
    bl38_value = runningBL.min()+(runningBL.max()-runningBL.min())*0.12
    bl88_value = runningBL.min()+(runningBL.max()-runningBL.min())*0.70
    # Constrain to only between the max and min
    i_min = np.where(runningBL==runningBL.min())[0][0]
    i_max = np.where(runningBL==runningBL.max())[0][0]
    bl76 = runningBL[min(i_min,i_max):max(i_min,i_max)+1]
    # Find the times associated with those % (just using min distance)
    bl38_time = bl76.index[abs(bl76-bl38_value)==
                                abs(bl76-bl38_value).min()][0]
    bl50_time = bl76.index[abs(bl76-bl50_value)==
                                abs(bl76-bl50_value).min()][0]
    bl88_time = bl76.index[abs(bl76-bl88_value)==
                                abs(bl76-bl88_value).min()][0]
    '''
    # Get HT interval as CurrentSheet width x 1.5
    cs_window = max(bl38_time,bl88_time)-min(bl38_time,bl88_time)
    HT_start = (min(bl38_time,bl88_time)+cs_window/2)-(cs_window*1.5/2)
    HT_end = (min(bl38_time,bl88_time)+cs_window/2)+(cs_window*1.5/2)
    return cross_b,cross_v,HT_start,bl50_time,HT_end

def calc_MHD(cross_b,cross_v,HT_start,HT_end,n_gsm,u_n):
    bdata = cross_b.copy(deep=True)
    vdata = cross_v.copy(deep=True)
    # Merge the two datasets -> the coarser resolution
    b_dseconds = [t.total_seconds() for t in cross_b.index-cross_b.index[0]]
    v_dseconds = [t.total_seconds() for t in cross_v.index-cross_v.index[0]]
    for key in bdata.keys():
        vdata[key] = np.interp(v_dseconds,b_dseconds,bdata[key].values)
    # Call existing function to calc MHD energy fluxes, for xyz and LMN
    dfxyz = add_derived_variables2(vdata,'mms','combined',xyz=['x','y','z'])
    dfLMN = add_derived_variables2(vdata,'mms','combined',xyz=['n','m','l'])
    # Call again to get the 'static' flux taking into account boundary motion
    un_x,un_y,un_z = [n*u_n for n in n_gsm]
    dfxyz_static = add_derived_variables2(vdata,'mms','combined',
                                          xyz=['x','y','z'],
                                          un=[un_x,un_y,un_z])
    dfLMN_static = add_derived_variables2(vdata,'mms','combined',
                                          xyz=['n','m','l'],
                                          un=[u_n,0,0])
    # Combine both statics  & LMN -> xyz
    for var in [k for k in dfLMN.keys() if 'K' in k or 'S' in k or 'H' in k]:
        dfxyz[var] = dfLMN[var]
        dfxyz[var+'_static'] = dfLMN_static[var]
    for var in [k for k in dfxyz_static.keys()if 'K' in k or 'S' in k
                                                          or 'H' in k]:
        dfxyz[var+'_static'] = dfxyz_static[var]

    # Specifically we want to sample the magnetosphere side Kn and Kn_static
    ht_start_dseconds = (HT_start-cross_v.index[0]).total_seconds()
    d_start = np.array([abs(t-ht_start_dseconds) for t in v_dseconds])
    i_start = np.where(d_start==d_start.min())[0]

    ht_end_dseconds = (HT_end-cross_v.index[0]).total_seconds()
    d_end = np.array([abs(t-ht_end_dseconds) for t in v_dseconds])
    i_end = np.where(d_end==d_end.min())[0]
    if dfxyz['bl'].values[i_start][0] < dfxyz['bl'].values[i_end][0]:
        i_outside = i_start
        i_inside = i_end
        #Kn_outside = dfxyz['Kn'].values[i_start][0]
        #Kn_inside = dfxyz['Kn'].values[i_end][0]
        #Kn_static_outside = dfxyz['Kn'].values[i_start][0]
        #Kn_static_inside = dfxyz['Kn'].values[i_end][0]
    else:
        i_inside = i_start
        i_outside = i_end
        #Kn_outside = dfxyz['Kn'].values[i_end][0]
        #Kn_inside = dfxyz['Kn'].values[i_start][0]
        #Kn_static_outside = dfxyz['Kn_static'].values[i_end][0]
        #Kn_static_inside = dfxyz['Kn_static'].values[i_start][0]

    return dfxyz, i_inside, i_outside

def process_crossing(crossing_misc,ID,t0,**kwargs):
    """Main function
    """
    # Load event
    misc = crossing_misc.loc[ID]
    start = dt.datetime.fromisoformat(misc['DateStart'])
    ut = (start-t0).total_seconds()
    gp.recalc(ut)
    interval = [start,start+dt.timedelta(seconds=300)]
    event = {'ID':ID}
    # Load and handle MMS data
    if kwargs.get('load_mms',True):
        # Misc. data has found the boundary normal and u_norm already
        eigenvalues = misc['l1_mfr'],misc['l2_mfr'],misc['l3_mfr']
        min_vector = eigenvalues.index(min(eigenvalues))+1
        max_vector = eigenvalues.index(max(eigenvalues))+1
        n_gse = [misc[f'x{min_vector}x_mfr'],
                 misc[f'x{min_vector}y_mfr'],
                 misc[f'x{min_vector}z_mfr']]
        l_gse = [misc[f'x{max_vector}x_mfr'],
                 misc[f'x{max_vector}y_mfr'],
                 misc[f'x{max_vector}z_mfr']]
        n_gsm = gp.gsmgse(*n_gse,-1)
        l_gsm = gp.gsmgse(*l_gse,-1)
        u_n = misc['Vn_MFR']
        pos,bfield,plasma=collect_mms(interval[0],interval[1],
                                      writeData=False,
                                      fgm_mode='brst',
                                      fpi_mode='brst')

        if pos['mms1'].empty:
            print('BAD DATA!')
            return
        event['mms_b'],event['mms_v'],HTstart,BL50,HTend = calc_LMN(
                                                bfield['mms1'],
                                                plasma['mms1'],n_gsm,l_gsm)
        event['start'] = start
        event['BL50'] = BL50
        event['HTstart'] = HTstart
        event['HTend'] = HTend
        event['mms_mhd'],event['i_in'],event['i_out'] = calc_MHD(
                                                    event['mms_b'],
                                                    event['mms_v'],
                                                    HTstart,HTend,n_gsm,u_n)
    # Get THEMIS data and check for good positioning
    th_pos,th_bfield,th_plasma = collect_themis(
                                        interval[0],
                                        interval[1],
                                        writeData=False,
                                        skip_bfield_data=True,
                                        skip_plasma_data=True)
    if not th_pos['themisA'].empty:
        good_themis = test_THEMIS(th_pos)
        event['thAx'],event['thAy'],event['thAz'] = th_pos['themisA'][
                                          ['x_gse','y_gse','z_gse']].values[0]
        event['thDx'],event['thDy'],event['thDz'] = th_pos['themisD'][
                                          ['x_gse','y_gse','z_gse']].values[0]
        event['thEx'],event['thEy'],event['thEz'] = th_pos['themisE'][
                                          ['x_gse','y_gse','z_gse']].values[0]
    else:
        print('BAD DATA!')
        good_themis = False
    # Get OMNI data and check for good positioning
    omni_window = [interval[0]-dt.timedelta(hours=(24)),
                   interval[1]+dt.timedelta(hours=(24))]
    omni = swmfpy.web.get_omni_data(omni_window[0],omni_window[1])
    # Check that the omni pull was successful
    max_tdiff = max(abs(omni['times'][0]-omni_window[0]).total_seconds()/60,
                    abs(omni['times'][-1]-omni_window[1]).total_seconds()/60)
    if max_tdiff > 60:
        old_omni = omni
        omni = swmfpy.web.get_omni_data(omni_window[0],omni_window[1],
                                        resolution='low')
    event['omni'] = upgrade_OMNI(omni)
    driving,energized,storm,substorm,convection = classify_OMNI(event['omni'],
                                            interval[0]-dt.timedelta(hours=0),
                                            interval[1]-dt.timedelta(hours=0))
    # Store results in event dict
    event['good_themis'] = good_themis
    event['driving'] = driving
    event['energized'] = energized
    event['storm'] = storm
    event['substorm'] = substorm
    event['convection'] = convection

    return event



#Main program
if __name__ == '__main__':
    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))
    start_time = time.time()
    inBase = 'magEx'
    inPath = os.path.join(inBase,'data')
    outPath = os.path.join(inBase,'figures')
    #Read in crossing list- from:
    #  Paschmann, G., Haaland, S. E., Phan, T. D., Sonnerup, B. U. Ö., Burch, J. L., Torbert, R. B., et al. (2018). Large-scale survey of the structure of the dayside magnetopause by MMS. Journal of Geophysical Research: Space Physics, 123, 2018–2033. https://doi.org/10.1002/2017JA025121 
    crossing_list = pd.read_csv(os.path.join(inPath,
                                           'MMS_MPdatabase_mp_crossings.csv'))
    crossing_list.index = crossing_list['EventId']
    crossing_ids = crossing_list['EventId'].values
    crossing_misc=pd.read_csv(os.path.join(inPath,'MMS2up_misc.csv'),sep=';')
    crossing_misc.index = crossing_misc['EventId']

    #Filter based on Event Classification (Appendix A2)
    # 1. the crossing type (magnetopause, magnetosphere, magnetosheath,
    #                       bow shock, and solar wind)
    # 2. the nature of the BL profile across the current sheet (“monotonic,”
    #                                                          “nonmonotonic”)
    # 3. the quality of the current sheet fit (“good,” “mediocre,” “bad”)
    # 4. the number of current sheets in the burst interval (“single,”
    #                                                              “multiple”)
    # 5. whether the current sheet crossing was “complete,” “incomplete,” or 
    #               “overlapping” with neighboring current sheets; Harris-like
    #               crossings were also noted
    filtered_list = []
    x,y,z = [],[],[]
    for i,ID in enumerate(crossing_list['EventId'].values):
        flag = crossing_list.loc[ID,'FlagStr']
        if ('mp' in flag and
            'mon' in flag and
            'gf' in flag and
            's' in flag and
            'compl' in flag and
            ID in crossing_misc.index):
            filtered_list.append(ID)
            x.append(crossing_list.loc[ID,'X'])
            y.append(crossing_list.loc[ID,'Y'])
            z.append(crossing_list.loc[ID,'Z'])

    good_crossings = pd.DataFrame({'ID':filtered_list,
                                   'X':x,'Y':y,'Z':z,
                                   'good_themis':[np.nan]*len(filtered_list),
                                   'thAx':[np.nan]*len(filtered_list),
                                   'thAy':[np.nan]*len(filtered_list),
                                   'thAz':[np.nan]*len(filtered_list),
                                   'thDx':[np.nan]*len(filtered_list),
                                   'thDy':[np.nan]*len(filtered_list),
                                   'thDz':[np.nan]*len(filtered_list),
                                   'thEx':[np.nan]*len(filtered_list),
                                   'thEy':[np.nan]*len(filtered_list),
                                   'thEz':[np.nan]*len(filtered_list),
                                   'driving':[np.nan]*len(filtered_list),
                                   'energized':[np.nan]*len(filtered_list),
                                   'storm':[np.nan]*len(filtered_list),
                                   'substorm':[np.nan]*len(filtered_list)})
    good_crossing_data = {}

    # Initialize the geopack routines by finding the universal time
    t0 = dt.datetime(1970,1,1)

    success = 0
    failed = 0
    '''
    for i,ID in enumerate(good_crossings['ID'].values):
        print(i,ID)
        # Store in compiled list
        #try:
        if True:
            event = process_crossing(crossing_misc,ID,t0,load_mms=False)
            #event = process_crossing(crossing_misc,'20151206_232844',t0)
            #plot_timeseries(event,outPath)
            for key in [k for k in event.keys() if 'mms' not in k and
                                                   'omni' not in k]:
                good_crossings.loc[i,key] = event[key]
            success+=1
        #except:
        #    print(ID,' Failed!')
        #    failed+=1
        #    continue
    print(f'success: {success/(i+1)*100}%, failed: {failed/(i+1)*100}%')
    '''
    if False:
        # Save good_crossings to an hdf5 file
        fileout = 'magEx/data/analysis/categorized_mms_crossings.h5'
        good_crossings.to_hdf(fileout,'best_quality')
        print('\033[92m Saved\033[00m',fileout)
    if False:
        #old_good_crossings = good_crossings
        fileout = 'magEx/data/analysis/categorized_mms_crossings.h5'
        good_crossings = pd.HDFStore(fileout)['/best_quality']
        #good_crossings['X'] = old_good_crossings['X']
        #good_crossings['Y'] = old_good_crossings['Y']
        #good_crossings['Z'] = old_good_crossings['Z']

        # Read in themis fast flow events from Li et al. 2021:
        ff_inpath = os.path.join('magEx','data','Li2021_themis_fastflows',
                                 'fast_flow_full_info','refined_20210224')
        fastflows = read_fastflow_list(ff_inpath)
        good_crossings = get_ff_matches(good_crossings,fastflows)
        # Crossref times with substorm list from Ohtani and G
        substorm_infile = ('magEx/data/'+
                   'substorms-ohtani-20150101_000000_to_20190101_000000.ascii')
        substorms = read_substorm_list(substorm_infile)
        good_crossings = get_ss_matches(good_crossings,substorms)

        # Plot coverage
        plot_coverage(good_crossings,outPath)
        # Plot histograms
        plot_histograms(good_crossings,outPath)


    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
