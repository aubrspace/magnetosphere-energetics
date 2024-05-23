#/usr/bin/env python
"""accesses MMS data from nasa CDA and creates plots for SWMF comparisons
"""
#General file IO/debugging
import os,sys,time
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
from global_energetics.analysis.plot_tools import (general_plot_settings)
from global_energetics.analysis.proc_indices import (ID_ALbays)
from global_energetics.analysis.proc_satellites import(add_derived_variables2)

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
            print('\t',sc,'Y\t','{:.2f}'.format(pos_data[sc]['y'][0]))
            if any([(y>-20)&(y<20) for y in pos_data[sc]['y'].values]):
                print('\t',sc,'X\t','{:.2f}'.format(pos_data[sc]['x'][0]))
                if any([(x>-20)&(x<-10) for x in pos_data[sc]['x'].values]):
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
    for key in ['x_gse','y_gse','z_gse',
                'bx','by_gse','bz_gse',
                'vx_gse','vy_gse','vz_gse',
                'density','b','v']:
        nans, x= nan_help(omni[key])
        omni[key][nans]= np.interp(x(nans), x(~nans), omni[key][~nans])
    # Get GSM variables
    for gse,gsm_keys in [[[omni['x_gse'],omni['y_gse'],omni['z_gse']],
                          ['x','y','z']],
                         [[omni['bx'],omni['by_gse'],omni['bz_gse']],
                          ['bx','by','bz']],
                         [[omni['vx_gse'],omni['vy_gse'],omni['vz_gse']],
                          ['vx','vy','vz']]]:
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

    window = [(t<end)&(t>start) for t in omni['times']]
    istart = np.where(window)[0][0]
    iend = np.where(window)[0][-1]+1
    #Coupling
    Ein = np.mean(omni['EinWang'][istart:iend])
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
    #Energization
    symh = np.mean(omni['sym_h'][istart:iend])
    if symh >= unenergized_limit:
        energized = 'quiet'
    elif symh >= lowenergized_limit:
        energized = 'low'
    elif symh >= medenergized_limit:
        energized = 'med'
    elif np.isnan(symh):
        energized = 'failed'
    elif symh < medenergized_limit:
        energized = 'high'
    else:
        energized = 'strange'
    #Storm
    # Is there a storm anywhere in this interval?
    anyStorm = omni['sym_h'].min() < -50
    if not anyStorm:
        storm = 'quiet'
    elif omni['sym_h'].min() > -50:
        # find min location
        i_min = np.where(omni['sym_h']==omni['sym_h'].min())[0]
        # find local 1hr slope
        sym_1hr = omni['sym_h'][istart-30:iend+30]
        dt_1hr = omni['dt'][istart-30:iend+30]
        slope_1hr = np.mean((sym_1hr[1::]-sym_1hr[0:-1])/dt_1hr[0:-1])
        # Now find out when it is relative to our current position
        if i_min[0] > i_end:
            # The end of main phase prob hasn't happened yet
            # So let's check the current level and local slope
            if symh < unenergized_limit and slope_1hr < 0:
                storm = 'main'
            elif symh < unenergized_limit and slope_1hr > 0:
                storm = 'strange'
            elif symh > 10:
                storm = 'ssc' #Wow!
            else:
                storm = 'quiet'
        elif i_min[-1] < i_start:
            # The end of main phase prob already ended
            # So let's check the current level and local slope
            if symh < unenergized_limit and slope_1hr > 0:
                storm = 'recovery'
            elif symh < unenergized_limit and slope_1hr < 0:
                storm = 'strange'
            elif symh >= unerenergized_limit:
                storm = 'quiet'
            else:
                storm = 'strange'
        else: #our crossing window is between minimums? I guess use slope..
            if slope_1hr <= 0:
                storm = 'main'
            elif slope_1hr > 0:
                storm = 'recovery'
            else:
                storm = 'strange'
    else:
        storm = 'failed'

    #Substorm
    albays, onsets, psuedos = ID_ALbays(omni,al_series='al')
    if any(albays[istart-10:iend+10]):
        substorm = 'substorm'
    elif any(psuedos[istart-10:iend+10]):
        substorm = 'psuedo'
    else:
        substorm = 'quiet'

    return driving, energized, storm, substorm

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

def plot_timeseries(event,path,**kwargs):
    #############
    #setup figure
    tseries,[mvab,flux,sw,dst,al] = plt.subplots(5,1,figsize=[12,18],
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
                    label=r'$E_{in}$ Wang et al. 2014',color='tab:orange',
                    alpha=0.4)
    sw_r.plot(event['omni']['times'],event['omni']['bx'],label=r'$B_X$',
            color='grey')
    sw_r.plot(event['omni']['times'],event['omni']['by'],label=r'$B_Y$',
            color='brown')
    sw_r.plot(event['omni']['times'],event['omni']['bz'],label=r'$B_Z$',
            color='blue')
    sw_r.set_ylabel(r'IMF $\left[nT\right]$')
    sw_r.legend(loc='lower right')
    # DST to give context to give context to 'energized' and 'storm'
    dst.plot(event['omni']['times'],event['omni']['sym_h'],label='SYM-H',
             color='black',lw=3)
    # AL to give context to 'substorm'
    al.plot(event['omni']['times'],event['omni']['al'],label='AL',
             color='brown')
    #Time markers
    mvab.axvline(event['BL50'],c='orange',lw=3)
    mvab.axvline(event['start'],c='black')
    sw_r.axhline(0)
    for ax in [mvab,flux,sw,dst,al]:
        ax.axvline(event['HTstart'],c='green',ls='--',lw=3)
        ax.axvline(event['HTend'],c='green',ls='--',lw=3)
        ax.margins(x=0.01)
    #Decorations
    general_plot_settings(mvab,do_xlabel=False,legend=True,
                          ylabel=r'$B \left[nT\right]$',
                        #xlim=[event['mms_b'].index[0],
                        #    event['mms_b'].index[0]+dt.timedelta(seconds=60)],
                          timedelta=False)
    general_plot_settings(flux,do_xlabel=False,legend=True,
                          ylabel=r'$K \left[GW/R_E^2\right]$',
                        #xlim=[event['mms_mhd'].index[0],
                        #  event['mms_mhd'].index[0]+dt.timedelta(seconds=60)],
                          timedelta=False)
    general_plot_settings(sw,do_xlabel=False,legend=True,
                          ylabel=r'$\int_{MP}E_{input}\left[ TW\right]$',
              xlim=[event['mms_mhd'].index[0]-dt.timedelta(seconds=(12*3600)),
                    event['mms_mhd'].index[0]+dt.timedelta(seconds=(12*3600))],
                          timedelta=False)
    general_plot_settings(dst,do_xlabel=False,legend=True,
                          ylabel=r'$\Delta B \left[nT\right]$',
              xlim=[event['mms_mhd'].index[0]-dt.timedelta(seconds=(12*3600)),
                    event['mms_mhd'].index[0]+dt.timedelta(seconds=(12*3600))],
                          ylim=[-150,10],
                          timedelta=False)
    general_plot_settings(al,do_xlabel=True,legend=True,
                          ylabel=r'$\Delta B \left[nT\right]$',
              xlim=[event['mms_mhd'].index[0]-dt.timedelta(seconds=(12*3600)),
                    event['mms_mhd'].index[0]+dt.timedelta(seconds=(12*3600))],
                          timedelta=False)
    sw.spines['left'].set_color('tab:orange')
    sw.tick_params(axis='y',colors='tab:orange')
    sw.yaxis.label.set_color('tab:orange')
    sw.legend(loc='upper left')
    tseries.suptitle(event['ID'],fontsize=32,fontweight='bold',
                     x=0.99,y=0.99,
                     verticalalignment='top',
                     horizontalalignment='right')
    tseries.tight_layout()
    # Save
    figurename = path+'/exampletimeseries_'+event['ID']+'.png'
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
    bl38_value = runningBL.min()+(runningBL.max()-runningBL.min())*0.12
    bl50_value = runningBL.min()+(runningBL.max()-runningBL.min())*0.5
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

def process_crossing(crossing_misc,ID,t0):
    """Main function
    """
    # Use start time +100s to find a prelim interval
    #misc = crossing_misc.loc['20151206_232844']
    misc = crossing_misc.loc[ID]
    #misc = crossing_misc.loc['20151206_002304']#NOTE failed case
    start = dt.datetime.fromisoformat(misc['DateStart'])
    ut = (start-t0).total_seconds()
    gp.recalc(ut)
    interval = [start,start+dt.timedelta(seconds=100)]
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
    # Get MMS data and hone in on actual window
    #event = {'ID':'20151206_232844'}
    #event = {'ID':'20151206_002304'}#NOTE failed case
    event = {'ID':ID}
    pos,bfield,plasma=collect_mms(interval[0]-dt.timedelta(hours=5),
                                    interval[1]-dt.timedelta(hours=5),
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
                                        interval[0]-dt.timedelta(hours=5),
                                        interval[1]-dt.timedelta(hours=5),
                                        writeData=False)
    if not th_pos['themisA'].empty:
        print('BAD DATA!')
        good_themis = test_THEMIS(th_pos)
    else:
        good_themis = False
    # Get OMNI data and check for good positioning
    omni_window = [interval[0]-dt.timedelta(hours=(24)),
                       interval[1]+dt.timedelta(hours=(24))]
    omni = swmfpy.web.get_omni_data(omni_window[0],omni_window[1])
    event['omni'] = upgrade_OMNI(omni)
    driving, energized, storm, substorm = classify_OMNI(event['omni'],
                                            interval[0]-dt.timedelta(hours=5),
                                            interval[1]-dt.timedelta(hours=5))
    # Store results in event dict
    event['good_themis'] = good_themis
    event['driving'] = driving
    event['energized'] = energized
    event['storm'] = storm
    event['substorm'] = substorm

    return event



#Main program
if __name__ == '__main__':
    inBase = 'magEx'
    inPath = os.path.join(inBase,'data')
    outPath = os.path.join(inBase,'figures')
    #Read in crossing list- from:
    #  Paschmann, G., Haaland, S. E., Phan, T. D., Sonnerup, B. U. Ö., Burch, J. L., Torbert, R. B., et al. (2018). Large-scale survey of the structure of the dayside magnetopause by MMS. Journal of Geophysical Research: Space Physics, 123, 2018–2033. https://doi.org/10.1002/2017JA025121 
    crossing_list = pd.read_csv(os.path.join(inPath,
                                           'MMS_MPdatabase_mp_crossings.csv'))
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
    for i,ID in enumerate(crossing_list['EventId'].values):
        flag = crossing_list.loc[i,'FlagStr']
        if ('mp' in flag and
            'mon' in flag and
            'gf' in flag and
            's' in flag and
            'compl' in flag):
            filtered_list.append(ID)

    good_crossings = pd.DataFrame({'ID':filtered_list,
                                   'good_themis':[np.nan]*len(filtered_list),
                                   'driving':[np.nan]*len(filtered_list),
                                   'energized':[np.nan]*len(filtered_list),
                                   'storm':[np.nan]*len(filtered_list),
                                   'substorm':[np.nan]*len(filtered_list)})
    good_crossing_data = {}

    # Initialize the geopack routines by finding the universal time
    t0 = dt.datetime(1970,1,1)

    success = 0
    failed = 0
    for i,ID in enumerate(good_crossings['ID'].values):
        print(i,ID)
        # Store in compiled list
        try:
            event = process_crossing(crossing_misc,ID,t0)
            plot_timeseries(event,outPath)
            for key in [k for k in event.keys() if 'mms' not in k]:
                good_crossings.loc[i,key] = event[key]
            success+=1
        except:
            print(ID,' Failed!')
            failed+=1
            continue

        '''
        good_crossing_data[ID] = event
        good_crossings.loc[i,'ID'] = event['ID']
        good_crossings.loc[i,'good_themis'] = event['good_themis']
        good_crossings.loc[i,'driving'] = event['driving']
        good_crossings.loc[i,'energized'] = event['energized']
        good_crossings.loc[i,'storm'] = event['storm']
        good_crossings.loc[i,'substorm'] = event['substorm']
        '''
    print(f'success: {success}, failed: {failed}')
    #from IPython import embed; embed()
    #TODO
    #   #Count up the number of crossings of different types
    #   eventtype_hists(good_crossings,bad_crossings)
    #   For each good crossing
    #       plot mms+MHD: B, VxVyVz, K,S,H, etc.
    #   For type in [types]:
    #       pull just type from good_crossings
    #       sea_results = SEA(type)
    #       plot_sea(sea_results)
    #themis_pos, themis_b,themis_plasma = collect_themis(start, end)
    #mms_pos, mms_b,mms_plasma = collect_mms(start, end)
