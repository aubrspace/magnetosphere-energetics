#!/usr/bin/env python3
"""Functions for handling and plotting data related to bow shock
"""
import os
import sys
import glob
import time
import numpy as np
from scipy import signal
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#interpackage imports
from global_energetics.extract.shue import (r0_alpha_1998,
                                            r0_bow_shock_Jerab2005)
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup, plot_psd,
                                                   plot_pearson_r,
                                                   plot_stack_distr,
                                                   plot_stack_contrib,
                                                   safelabel,
                                                   get_omni_cdas)
from global_energetics.analysis.analyze_energetics import (plot_swflowP,
                                                          plot_swbz,
                                                          plot_dst,plot_al)
from global_energetics.analysis.proc_virial import (process_virial)
from global_energetics.analysis.workingtitle import (locate_phase)
from global_energetics.analysis.proc_hdf import(group_subzones,
                                                load_hdf_sort)

def marktimes(ax,event,minutes=False):
    """function puts vertical lines at event times
    """
    ##Times for ICME phases
    feb_start =       dt.datetime(2014,2,18,6,45)
    feb_1_ICMEstart = dt.datetime(2014,2,18,5,59)
    feb_1_MOstart =   dt.datetime(2014,2,18,15,7)
    feb_1_ICMEend =   dt.datetime(2014,2,19,9,35)
    #   Shock: N
    #   Type:  Flux rope
    feb_2_ICMEstart = dt.datetime(2014,2,19,3,9)
    feb_2_MOstart =   dt.datetime(2014,2,19,3,9)
    feb_2_ICMEend =   dt.datetime(2014,2,20,2,37)
    #   Shock: Y
    #   Type:  Complex
    may_start =     dt.datetime(2019,5,13,17,35)
    may_ICMEstart = dt.datetime(2019,5,13,22,54)
    may_MOstart =   dt.datetime(2019,5,14,9,8)
    may_ICMEend =   dt.datetime(2019,5,15,15,41)
    #   Shock: N
    #   Type:  Ejecta
    if 'feb' in event:
        marks = [feb_1_ICMEstart, feb_1_MOstart, feb_1_ICMEend,
                 feb_2_ICMEstart, feb_2_MOstart, feb_2_ICMEend]
        start = feb_start
    elif 'may' in event:
        marks = [may_ICMEstart, may_MOstart, may_ICMEend]
        start = may_start
    for t in marks:
        if minutes:
            m = (t-start).days*1440 + (t-start).seconds/60
            ax.axvline(m, color='black',ls='--')
        else:
            ax.axvline(t, color='black',ls='--')


if __name__ == "__main__":
    #handling io paths
    datapath = sys.argv[-1]
    figureout = os.path.join(datapath,'figures')
    os.makedirs(figureout, exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='solar_presentation'))

    ##Loading data
    #HDF data, will be sorted and cleaned
    feb_results = load_hdf_sort(datapath+'/feb2014_results.h5')
    #star_results = load_hdf_sort(datapath+'/starlink_results.h5')
    #ccmc_results = load_hdf_sort(datapath+'/ccmc_results.h5')

    #Log files and observational indices
    f14obs = read_indices(datapath,prefix='feb2014_',read_supermag=False,
                          start=feb_results['mpdict']['ms_full'].index[0],
                          end=feb_results['mpdict']['ms_full'].index[-1])
    #starobs = read_indices(datapath,prefix='starlink_',read_supermag=False,
    #                      start=star_results['mpdict']['ms_full'].index[0],
    #                      end=star_results['mpdict']['ms_full'].index[-1])
    #ccmcobs = read_indices(datapath,prefix='ccmc_',read_supermag=False,
    #                      start=ccmc_results['mpdict']['ms_full'].index[0],
    #                      end=ccmc_results['mpdict']['ms_full'].index[-1])

    ##Apply any mods and gather additional statistics
    feb_mp = feb_results['mpdict']['ms_full']
    #star_mp = star_results['mpdict']['ms_full']
    #ccmc_mp = ccmc_results['mpdict']['ms_full']

    for mp,e_obs,bsdict,crossings,tag in [
                                   (feb_mp,f14obs,feb_results['bsdict'],
                                       feb_results['crossdict'],'feb2014')]:
                                   #(ccmc_mp,ccmcobs,ccmc_results['bsdict'],
                                   # ccmc_results['crossdict'],'may2019')]:
        #Easy references and quick formatting
        bs_up = bsdict['/ext_bs_up_surface']
        bs_up = bs_up.sort_values(by=['Time [UTC]'])
        bs_up.index = bs_up['Time [UTC]']
        bs_up = bs_up.drop(columns=['Time [UTC]'])

        bs_dn = bsdict['/ext_bs_down_surface']
        bs_dn = bs_dn.sort_values(by=['Time [UTC]'])
        bs_dn.index = bs_dn['Time [UTC]']
        bs_dn = bs_dn.drop(columns=['Time [UTC]'])

        #Some derived variables
        for key in [k for k in crossings.keys() if 'flow_line' in k]:
            cross = crossings[key]
            #Trace data
            convert = 1.6726e-27*1e6*(1e3)**2*1e9
            cross['U']=np.sqrt(cross['U_x']**2+cross['U_y']**2+
                               cross['U_z']**2)
            cross['B']=np.sqrt(cross['B_x']**2+cross['B_y']**2+
                               cross['B_z']**2)
            cross['Pdyn']= cross['Rho']*cross['U']**2*convert
            cross['Beta'] = cross['P']/(cross['B']**2/(4*np.pi*1e-7*1e9))
            cross['Beta*'] = (cross['P']+cross['Pdyn'])/(cross['B']**2/
                                                    (4*np.pi*1e-7*1e9))
            cross['Cs'] = np.sqrt(5/3*cross['P']/cross['Rho']/1.67)*1e6
            cross['Va'] = np.sqrt(cross['B']**2/(4*np.pi)/
                                  cross['Rho']/1.67)*1e5
            cross['Ms'] = cross['U']*1e3/cross['Cs']
            cross['Ma'] = cross['U']*1e3/cross['Va']
            cross['Mms'] = cross['U']*1e3/np.sqrt(
                                            cross['Va']**2+cross['Cs']**2)
            cross['s'] = cross['P']/cross['Rho']**(5/3)
            cross['JdotE'] = (
        cross['J_x']*(cross['U_z']*cross['B_y']-cross['U_y']*cross['B_z'])+
        cross['J_y']*(cross['U_x']*cross['B_z']-cross['U_z']*cross['B_x'])+
        cross['J_z']*(cross['U_y']*cross['B_x']-cross['U_x']*cross['B_y'])
                             )
        ma = np.zeros(len(bs_up.index))
        ms = np.zeros(len(bs_up.index))
        mms = np.zeros(len(bs_up.index))
        for i,t in enumerate(bs_up.index):
            curve = crossings['/flow_line_nose'][
                                    crossings['/flow_line_nose'].index==t]
            nose = curve[curve['X']==bs_up['X_subsolar [Re]'].iloc[i]+
                 np.min(abs(curve['X']-bs_up['X_subsolar [Re]'].iloc[i]))]
            ma[i] = nose['Ma'][0]
            ms[i] = nose['Ms'][0]
            mms[i] = nose['Mms'][0]
        bs_up['Ma'] = ma
        bs_up['Ms'] = ms
        bs_up['Mms'] = mms

        #Swmf solar wind data
        e_obs['swmf_sw']['B']= np.sqrt(e_obs['swmf_sw']['bx']**2+
                                       e_obs['swmf_sw']['by']**2+
                                       e_obs['swmf_sw']['bz']**2)
        e_obs['swmf_sw']['cone']= np.arctan2(e_obs['swmf_sw']['bx'],
                     (e_obs['swmf_sw']['by']**2+e_obs['swmf_sw']['bz']**2))
        e_obs['swmf_sw']['P']= (e_obs['swmf_sw']['density']*
                                e_obs['swmf_sw']['temperature']*
                                1e6*1.3807e-23*1e9)
        e_obs['swmf_sw']['Beta'] = (e_obs['swmf_sw']['P']/
                (e_obs['swmf_sw']['B']**2/(4*np.pi*1e-7*1e9)))
        e_obs['swmf_sw']['Beta*'] = ((e_obs['swmf_sw']['P']+
                                                 e_obs['swmf_sw']['pdyn'])/
                             (e_obs['swmf_sw']['B']**2/(4*np.pi*1e-7*1e9)))
        e_obs['swmf_sw']['Cs'] = (np.sqrt(5/3*e_obs['swmf_sw']['P']/
                                     e_obs['swmf_sw']['density']/1.67)*1e6)
        e_obs['swmf_sw']['Va']=(np.sqrt(e_obs['swmf_sw']['B']**2/(4*np.pi)/
                                     e_obs['swmf_sw']['density']/1.67)*1e5)
        e_obs['swmf_sw']['Ms'] = (e_obs['swmf_sw']['v']*1e3/
                                  e_obs['swmf_sw']['Cs'])
        e_obs['swmf_sw']['Ma'] = (e_obs['swmf_sw']['v']*1e3/
                                  e_obs['swmf_sw']['Va'])
        e_obs['swmf_sw']['s'] = (e_obs['swmf_sw']['P']/
                                 e_obs['swmf_sw']['density']**(5/3))
        e_obs['swmf_sw']['r_shue98'], e_obs['swmf_sw']['alpha'] = (
                                  r0_alpha_1998(e_obs['swmf_sw']['bz'],
                                                e_obs['swmf_sw']['pdyn']))
        e_obs['swmf_sw']['r_Jerab05'] = r0_bow_shock_Jerab2005(
                                        e_obs['swmf_sw']['density'],
                                        e_obs['swmf_sw']['v'],
                                        e_obs['swmf_sw']['Ma'],
                                        e_obs['swmf_sw']['B'],
                                        gamma=2.15)

        #Create subdirectory
        figureout = os.path.join(datapath,'figures',tag)
        os.makedirs(figureout, exist_ok=True)

        ##################################################################
        #Plot bs nose and mp subsolar
        figname = '/standoff.png'
        stdof,ax = plt.subplots(nrows=1,ncols=1,figsize=[14,8])
        ax.plot(bs_up.index, bs_up['X_subsolar [Re]'], label='Bow Shock')
        ax.plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['r_Jerab05'],
                label='Jerab05 '+r'$\gamma=2.15$',color='goldenrod',ls='--')
        ax.fill_between(bs_up.index, bs_up['X_subsolar [Re]'],
                        bs_dn['X_subsolar [Re]'], color='thistle')
        ax.plot(mp.index, mp['X_subsolar [Re]'], label='Magnetopause',
                color='aqua')
        ax.plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['r_shue98'],
                label='Shue98', color='teal',ls='--')
        general_plot_settings(ax,ylabel=r'Distance $\left[R_e\right]$',
                              ylim=[5,30],do_xlabel=True)
        marktimes(ax,tag)
        stdof.tight_layout(pad=1)
        stdof.savefig(figureout+figname)
        plt.close(stdof)

        ##################################################################
        #Plot mass flux
        figname = '/mass.png'
        mass,ax = plt.subplots(nrows=1,ncols=1,figsize=[14,8])
        ax.plot(bs_up.index, bs_up['M_net [kg/s]'], label='Bow Shock')
        general_plot_settings(ax,ylabel=r'Mass Flux $\left[kg/s\right]$',
                              do_xlabel=True)
        ax.legend().remove()
        mass.tight_layout(pad=1)
        mass.savefig(figureout+figname)
        plt.close(mass)

        ##################################################################
        #Plot energy fluxes
        figname = '/energy.png'
        energy,ax = plt.subplots(nrows=1,ncols=1,figsize=[14,12])
        #ax.plot(bs_up.index, bs_up['K_net [W]'], label='Up Total')
        #ax.plot(bs_dn.index, bs_dn['K_net [W]'], label='Down Total')
        ax.fill_between(bs_up.index, bs_up['M_net [kg/s]']/1e4,
                         label='Mass Flux', color='thistle')
        ax.plot(bs_up.index,bs_up['P0_net [W]']/1e14,
                label='Plasma Energy Up')
        ax.plot(bs_dn.index,bs_dn['P0_net [W]']/1e14,
                label='Plasma Energy Down',color='goldenrod',ls='--')
        ax.plot(bs_up.index,bs_up['ExB_net [W]']/1e14,label='Poynting Up',
                color='salmon')
        ax.plot(bs_dn.index,bs_dn['ExB_net [W]']/1e14,label='Poynting Down',
                color='indianred',ls='--')
        general_plot_settings(ax,ylabel=r'Flux $\left[10^{2}TW\right]$'+
                                            r' $\left[10^{4}kg\right]$',
                              do_xlabel=True)
        marktimes(ax,tag)
        energy.tight_layout(pad=1)
        energy.savefig(figureout+figname)
        plt.close(energy)

        ##################################################################
        #Plot density ranges
        '''
        figname = '/ranges.png'
        ranges,ax = plt.subplots(nrows=3,ncols=1,figsize=[14,20],
                                   sharex=True)
        for t in bs_up.index:
            curve = cross.copy()[cross['Time [UTC]']==t]
            curve = curve.sort_values(by=['X'])
            ax[0].plot(curve['Time [UTC]'],curve['Rho'])
            ax[1].plot(curve['Time [UTC]'],curve['Beta'])
            ax[2].plot(curve['Time [UTC]'],curve['Beta*'])
        ax[0].legend().remove()
        general_plot_settings(ax[0],ylabel=r'$\rho\left[amu/cm^3\right]$',
                              do_xlabel=False)
        ax[1].legend().remove()
        general_plot_settings(ax[1],ylabel=r'$\beta$',do_xlabel=False)
        ax[2].legend().remove()
        general_plot_settings(ax[2],ylabel=r'$\beta^*$',do_xlabel=True)
        ranges.tight_layout(pad=1)
        ranges.savefig(figureout+figname)
        plt.close(ranges)
        '''

        ##################################################################
        #Plot contour for nose crossing
        for key in [k for k in crossings.keys() if 'flow_line' in k]:
            cross = crossings[key]
            figname = key+'_contours.png'
            contours,ax = plt.subplots(nrows=3,ncols=1,figsize=[14,10],
                                       sharex=True)
            jdot,ax2 = plt.subplots(nrows=1,ncols=1,figsize=[14,4])
            cond = (cross['X']>8) & (cross['X']<30) & (~cross['X'].isna())
            cross = cross[cond]
            times = ((cross.index-cross.index[0]).days*1440+
                     (cross.index-cross.index[0]).seconds/60)
            bs_up_times = ((bs_up.index-bs_up.index[0]).days*1440+
                        (bs_up.index-bs_up.index[0]).seconds/60)
            #Density
            cRho_ = ax[0].tricontourf(times,cross['X'],cross['Rho'],
                                      levels=np.linspace(0,30,31),
                                      cmap='plasma',extend='both')
            cbarRho = contours.colorbar(cRho_,ax=ax[0])
            #ax[0].plot(bs_up_times,bs_up['X_subsolar [Re]'],color='grey')
            #ax[0].set_ylabel(r'$\rho \left[ amu/cm^3\right]$')
            marktimes(ax[0],tag,minutes=True)
            #Mach number
            cMa_ = ax[1].tricontourf(times,cross['X'],cross['Ma'],
                                     levels=np.linspace(0,20,21),
                                     cmap='viridis',extend='both')
            cbarMa = contours.colorbar(cMa_,ax=ax[1])
            #ax[1].plot(bs_up_times,bs_up['X_subsolar [Re]'],color='grey')
            #ax[1].set_ylabel(r'$M_A$')
            marktimes(ax[1],tag,minutes=True)
            #Specific Entropy
            cS_ = ax[2].tricontourf(times,cross['X'],np.log10(cross['s']),
                                    levels=np.linspace(-4,-1,11),
                                    cmap='cividis',extend='both')
            cbarS = contours.colorbar(cS_,ax=ax[2])
            #ax[2].plot(bs_up_times,bs_up['X_subsolar [Re]'],color='grey')
            #ax[2].set_ylabel(r'$log_{10}$ s $\left[\frac{nPa}'+
            #                           r'{(amu/cm^3)^{\gamma}}\right]$')
            ax[2].set_xlabel(r'Time $\left[ min\right]$')
            marktimes(ax[2],tag,minutes=True)

            #Joule heating
            jE_ = ax2.tricontourf(times,cross['X'],cross['JdotE'],
                                    cmap='inferno',extend='both')
            cbarJ = contours.colorbar(jE_,ax=ax2)
            ax2.set_ylabel(r'$J\cdot E\left[10^{-12} W/m^3\right]$')
            ax2.set_xlabel(r'Time $\left[ min\right]$')
            marktimes(ax2,tag,minutes=True)
            jdot.tight_layout(pad=1)
            jdot.savefig(figureout+key+'_jdot.png')
            plt.close(jdot)

            contours.tight_layout(pad=1)
            contours.savefig(figureout+figname)
            plt.close(contours)

        ##################################################################
        #Plot solar wind conditions
        figname = '/solar_wind.png'
        sw,ax = plt.subplots(nrows=3,ncols=1,figsize=[14,10],
                                   sharex=True)
        #Magnetic field
        ax[0].fill_between(e_obs['swmf_sw'].index, np.sqrt(
                                                e_obs['swmf_sw']['bx']**2+
                                                e_obs['swmf_sw']['by']**2+
                                                e_obs['swmf_sw']['bz']**2),
                                                color='thistle')
        ax[0].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['bx'],
                   label=r'$B_x$')
        ax[0].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['by'],
                   label=r'$B_y$')
        ax[0].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['bz'],
                   label=r'$B_z$')
        general_plot_settings(ax[0],ylabel=r'$B\left[nT\right]$',
                              do_xlabel=False)
        marktimes(ax[0],tag)
        #Veloctity
        twinax = ax[1].twinx()
        twinax.spines['right'].set_color('gold')
        twinax.plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['vx']*-1,
                           label=r'$-V_x$')
        twinax.legend(loc='lower right')
        ax[1].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['vy'],
                   label=r'$V_y$',color='aqua')
        ax[1].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['vz'],
                   label=r'$V_z$',color='salmon')
        general_plot_settings(ax[1],ylabel=r'$V\left[km/s\right]$',
                              do_xlabel=False,legend_loc='upper left')
        marktimes(ax[1],tag)
        #Mach number and plasma beta
        ax[2].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['Ma'],
                   label=r'$M_A$')
        ax[2].plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['Beta'],
                   label=r'$\beta$')
        twinax = ax[2].twinx()
        twinax.spines['right'].set_color('salmon')
        twinax.set_ylabel(r'$P_{dyn}\left[nPa\right]$')
        twinax.plot(e_obs['swmf_sw'].index, e_obs['swmf_sw']['pdyn'],
                    color='salmon')
        general_plot_settings(ax[2],ylabel=r'$M_A$ $\beta$',
                              do_xlabel=True,ylim=[0,25])
        marktimes(ax[2],tag)
        sw.tight_layout(pad=1)
        sw.savefig(figureout+figname)
        plt.close(sw)

        ##################################################################
        #Plot compression ratio
        compress,ax = plt.subplots(nrows=1,ncols=1,figsize=[14,6],
                                       sharex=True)
        figname = '/nose_compressions.png'
        if 'r_Rho' in bs_up.keys():
            ax.plot(bs_up.index, bs_up['r_Rho'],lw=1.5,label=r'$r_{\rho}$')
            #ax.plot(bs_up.index, bs_up['r_Rho'].rolling(5).mean(),
            #        label=r'$r_{\rho}$')
            #ax.plot(bs_up.index, bs_up['r_Bmag'].rolling(5).mean(),
            #        label=r'$r_{B}$')
            ax.plot(bs_up.index, bs_up['r_Bmag'],lw=1,label=r'$r_{B}$')
        general_plot_settings(ax,ylabel=r'Compression Ratio', ylim=[0,10],
                              do_xlabel=True)
        marktimes(ax,tag)
        compress.tight_layout(pad=1)
        compress.savefig(figureout+figname)
        plt.close(compress)

        ##################################################################
        #Plot correlation of Ma, Mms vs compression ratio
        corr,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,6])
        figname = '/compression_correlation.png'
        plot_pearson_r(ax, bs_up.index,bs_up.index,
                           bs_up['r_Rho'],bs_up['Ma'],
                           xlabel=r'$r_{\rho}$',ylabel=r'$M_{A}$')
        plot_pearson_r(ax, bs_up.index,bs_up.index,
                           bs_up['r_Rho'],bs_up['Ms'],
                           xlabel=r'$r_{\rho}$',ylabel=r'$M_{S}$')
        plot_pearson_r(ax, bs_up.index,bs_up.index,
                           bs_up['r_Rho'],bs_up['Mms'],
                           xlabel=r'$r_{\rho}$',ylabel=r'$M_{MS}$')
        corr.tight_layout(pad=1)
        corr.savefig(figureout+figname)
        plt.close(corr)
