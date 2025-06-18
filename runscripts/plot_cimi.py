#!/usr/bin/env python3
"""Analyze and plot CIMI data for a quiet month with basic settings
"""
import os,sys
import numpy as np
from numpy import abs
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm,ticker
from matplotlib import dates as mdates
import swmfpy
#interpackage imports
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings,
                                                   bin_and_describe,
                                                   extended_fill_between,
                                                   refactor)
from global_energetics.analysis import proc_indices
from cdasws import CdasWs
cdas = CdasWs()
#from plot_rbsp import call_cdaweb_rbsp

def match_up_E(sim_lvls,hep,Elow,Ehigh)-> list[int,int,int,int]:
    E_lvls_L = hep['FEDO_L_Energy_MEAN']
    E_lvls_H = hep['FEDO_H_Energy_MEAN']
    iEmatch_L = np.where([(E_lvls_L>=Elow)&(E_lvls_L<=Ehigh)])[1]
    iEmatch_H = np.where([(E_lvls_H>=Elow)&(E_lvls_H<=Ehigh)])[1]

    return iEmatch_L[0],iEmatch_L[-1],iEmatch_H[0],iEmatch_H[-1]


def call_cdaweb_rbsp(start:dt.datetime,
                       end:dt.datetime,probe,**kwargs):
    """ Returns 2 cdaweb spacepy.pycdf.CDFCopy for MagEIS and REPT (electrons)
    """
    print(f"Calling CDAWeb for RBSP {probe}\n\t from {start} to {end}")
    #status,data =cdas.get_data(f'RBSP{probe}_REL03_ECT-REPT-SCI-L3',
    #                           ['Position','L_star','FEDU'],start,end)
    status,data_MagEIS = cdas.get_data(f'RBSP{probe}_REL03_ECT-MAGEIS-L2',
                                 ['Position','MLT','L_star','FESA'],
                                                                    start,end)
    data_MagEIS['FESA']=np.nan_to_num(data_MagEIS['FESA'].replace_invalid(),0)
    status,data_REPT = cdas.get_data(f'RBSP{probe}_REL03_ECT-REPT-SCI-L2',
                                 ['Position','MLT','L_star','FESA','FEDU'],
                                                                    start,end)
    data_REPT['FESA'] = np.nan_to_num(data_REPT['FESA'].replace_invalid(),0)
                            #-dt.timedelta(minutes=kwargs.get('padtime',360)),
                            #+dt.timedelta(minutes=kwargs.get('padtime',360)))
    #NOTE from url below, FEDU is time x pitch angle x energies
    #                     FESA is time x energies
    # https://cdaweb.gsfc.nasa.gov/misc/NotesR.html#RBSPA_REL03_ECT-REPT-SCI-L3
    return data_MagEIS,data_REPT

def call_cdaweb_arase(start:dt.datetime,
                       end:dt.datetime,**kwargs):
    """ Returns 2 cdaweb spacepy.pycdf.CDFCopy for HEP and XEP (electrons)
    """
    print(f"Calling CDAWeb for ARASE \n\t from {start} to {end}")
    status,hep = cdas.get_data('ERG_HEP_L2_OMNIFLUX',['FEDO_H', 'FEDO_L'],
                               start,end)
    status,orb = cdas.get_data('ERG_ORB_L2',['pos_Lm'],start,end)
    hep_time = [(t-T0).total_seconds() for t in hep['Epoch_L']]
    orb_time = [(t-T0).total_seconds() for t in orb['epoch']]
    hep['L'] = np.interp(hep_time,orb_time,orb['pos_Lm'][:,2])
    #status,xep = cdas.get_data('ERG_XEP_L2_OMNIFLUX',['FEDO_SSD'],start,end)
    return hep

def read_db_file(infile:str) -> dict[str:np.ndarray]:
    data_dict = {}
    with open(infile,'r') as f:
        headers = f.readline().split()
        raw = f.readlines()
    data = np.zeros([len(raw),len(headers)])
    for i,line in enumerate(raw):
        data[i,:] =np.array([float(v) for v in line.replace('\n','').split()])
    for icol in range(0,len(headers)):
        data_dict[headers[icol]] = data[:,icol]
    return data_dict

def integrate_f_dAlpha(sina:np.ndarray,
                       flux:np.ndarray,pa_axis:int) -> np.ndarray:
    # Get dmu from the sin of pitch angle array
    mu_bins_left  = np.concat([[0],[0.5*(sina[i]+sina[i-1])
                                        for i in range(1,len(sina))]])
    mu_bins_right = np.concat([[0.5*(sina[i+1]+sina[i])
                                        for i in range(0,len(sina)-1)],[1]])
    dmu = np.sqrt(1-mu_bins_left**2)-np.sqrt(1-mu_bins_right**2)

    # Get para/perp factors
    para_factor = 3*np.cos(np.arcsin(sina))**2
    perp_factor = 1.5*sina**2

    # Integrate using flux*dmu
    flux_PAave = np.sum(flux*dmu,axis=pa_axis)/np.sum(dmu)
    flux_para  = np.sum(flux*dmu*para_factor,axis=pa_axis)/np.sum(dmu)
    flux_perp  = np.sum(flux*dmu*perp_factor,axis=pa_axis)/np.sum(dmu)

    return flux_PAave,flux_para,flux_perp

def mlt2rad(mlt:float) -> float:
    return (12-mlt)*np.pi/12

def find_nearest(array:np.ndarray, value:float) -> tuple[int,float]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]


#############################################################################
def add_Lstar_time(data:np.lib.npyio.NpzFile,axis:plt.axis,
                  ilowE:int,ihighE:int,
                                **kwargs:dict) -> mpl.contour.QuadContourSet:
    # Average flux @Energy level over all pitch angles
    Elvls = data['E_lvls'][ilowE:ihighE]
    flux_Echannel = np.trapezoid(data['flux'][:,:,:,ilowE:ihighE,:],
                                 x=Elvls,axis=3)# t,Lat,MLT,PA
    flux_PAave,flux_para,flux_perp = integrate_f_dAlpha(data['alpha_lvls'],
                                                        flux_Echannel,3)

    if 'Lstar' not in list(data.keys()): #TODO handle this better
        lats = data['latN']
        Lparam = 1/np.cos(np.deg2rad(lats))**2
    else:
        Lparam = data['Lstar']
    # Creat L* bins and take average at each bin
    L_bins = np.linspace(-10,10,len(data['lat_lvls'])+1)
    L_ave_flux = np.zeros([len(data['times']),len(L_bins)-1])
    for it in range(0,len(data['times'])):
        flux_now  = flux_PAave[it,:,:]
        L_now     = Lparam[it,:,:]
        for iL in range(0,len(L_bins)-1):
            flux_at_L = flux_now[(L_now>=L_bins[iL])&
                                 (L_now<=L_bins[iL+1])]
            if flux_at_L.size>0:
                L_ave_flux[it,iL] = flux_at_L.mean()
            else:
                L_ave_flux[it,iL] = 0

    # Plot in log scale
    L_bins_c = [(L_bins[i]+L_bins[i+1])/2 for i in range(0,len(L_bins)-1)]
    T,L = np.meshgrid(data['times'],L_bins_c)
    #times = [T0+dt.timedelta(hours=t) for t in data['times']]
    #T,L = np.meshgrid(times,L_bins_c)
    clevels = kwargs.get('clevels',np.logspace(2.0,7.0))
    cs = axis.contourf(T,L,L_ave_flux.T,clevels,cmap=mpl.cm.plasma,
                                 norm=mpl.colors.LogNorm(),extend='both')
    return cs

def add_Lstar_time_vsat(satflux,axis:plt.axis) -> plt.scatter:
    # Get all flux from spin averaged
    all_flux = np.trapezoid(satflux['flux'][:,43:55],
                            x=satflux['E_lvls'][43:55],axis=1) # time,flux
    scat = axis.scatter(satflux['times'],satflux['Lshell'],c=all_flux,
                        cmap=cm.plasma,vmin=1e2,vmax=1e6,norm='log')
    return scat

def add_Lstar_time_rbsp(mageis,rept,axis:plt.axis) -> plt.scatter:
    # Get all flux from 800keV onwards
    #TODO identify which energy channels on MAGEIS to integrate over
    #       - then maybe also include all REPT channels??
    #           see how the magnitudes turn out
    E_lvls = np.array([float(S.split()[0].split('_')[-1])
                      for S in mageis['metavar1'] if '-1' not in S])
    mflux = np.delete(mageis['FESA'],[0,1,2,3,17],axis=1)
    all_mageis = np.trapezoid(mflux[:,9::],x=E_lvls[9::],axis=1) # time,flux
    #all_rept = np.trapezoid(rept['FESA'][:,:],axis=1) # time,flux
    scat = axis.scatter(mageis['Epoch'],mageis['L_star'],c=all_mageis,
                        cmap=cm.plasma,vmin=1e2,vmax=1e6,norm='log')
    return scat

def add_Lstar_time_arase(hep,ax1:plt.axis,ax2:plt.axis,
                        ilowE_L:int,ihighE_L:int,ilowE_H:int,
                        ihighE_H:int) -> list[plt.scatter,plt.scatter]:
    # Get all flux from 800keV onwards
    #TODO identify which energy channels on MAGEIS to integrate over
    #       - then maybe also include all REPT channels??
    #           see how the magnitudes turn out
    flux_L =np.trapezoid(hep['FEDO_L'][:,ilowE_L:ihighE_L],
                         x=hep['FEDO_L_Energy_MEAN'][ilowE_L:ihighE_L],axis=1)
    flux_H =np.trapezoid(hep['FEDO_H'][:,ilowE_H:ihighE_H],
                         x=hep['FEDO_H_Energy_MEAN'][ilowE_H:ihighE_H],axis=1)

    scat_L = ax1.scatter(hep['Epoch_L'][hep['L']>0],hep['L'][hep['L']>0],
                          c=flux_L[hep['L']>0],
                          cmap=cm.plasma,vmin=1e2,vmax=1e6,norm='log')
    scat_H = ax2.scatter(hep['Epoch_H'][hep['L']>0],hep['L'][hep['L']>0],
                          c=flux_H[hep['L']>0],
                          cmap=cm.plasma,vmin=1e2,vmax=1e6,norm='log')
    return scat_L,scat_H

def add_dst(swmf_log:pd.DataFrame,omni:dict,axis:plt.axis) -> None:
    # could return lines as list[mpl.lines.Line2D]
    # Omni Sym-H
    axis.plot(omni['times'],omni['sym_h'],c='black',lw=2,label='Dst')
    # SWMF dB
    axis.plot(swmf_log.index,swmf_log['dst_sm'],c='navy',lw=2,ls='--',
              label='SWMF')
    return

def plot_figure2(flux:np.lib.npyio.NpzFile,
                 swmf_log:dict,
                 hep,
                 omni:dict,
                 vobs:np.lib.npyio.NpzFile,**kwargs:dict) -> None:
    """ Plot L-time plot with flux at a certain energy level
    """
    ilowE  = kwargs.pop('ilowE',7)
    ihighE = kwargs.pop('ihighE',11)

    # Create figure
    fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=[24,15],sharex=True)

    # Add L-time all CIMI
    cs1   = add_Lstar_time(flux,ax1,ilowE,ihighE,**kwargs)
    cbar1 = fig.colorbar(cs1,format="{x:.1e}",pad=0.1,ticks=[1e2,1e4,1e6])
    rax = ax1.twinx()
    add_dst(swmf_log['swmf_log'],omni,rax)

    # Add L-time from ARASE
    ilow_L,ihigh_L,ilow_H,ihigh_H = match_up_E(flux['E_lvls'],hep,
                                               flux['E_lvls'][ilowE],
                                               flux['E_lvls'][ihighE])
    cs2a,cs2b = add_Lstar_time_arase(hep,ax2,ax3,ilow_L,ihigh_L,ilow_H,ihigh_H)
    cbar2a = fig.colorbar(cs2a,format="{x:.1e}",pad=0.1,ticks=[1e2,1e4,1e6])
    cbar2b = fig.colorbar(cs2b,format="{x:.1e}",pad=0.1,ticks=[1e2,1e4,1e6])

    # Decorate
    cbar1.set_label(
      f"Flux {flux['E_lvls'][ilowE]:.0f}-{flux['E_lvls'][ihighE]:.0f} keV\n"+
        r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')
    cbar2a.set_label(
      f"Flux {hep['FEDO_L_Energy_MEAN'][ilow_L]:.0f}-"+
           f"{hep['FEDO_L_Energy_MEAN'][ihigh_L]:.0f} keV\n"+
        r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')
    cbar2b.set_label(
      f"Flux {hep['FEDO_H_Energy_MEAN'][ilow_H]:.0f}-"+
           f"{hep['FEDO_H_Energy_MEAN'][ihigh_H]:.0f} keV\n"+
        r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')

    ax1.text(0.01,0.90,"May 2024",transform=ax1.transAxes,
              horizontalalignment='left')
    for axis in [ax1,ax2,ax3]:
        axis.set_ylim([1,10])
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
        axis.set_ylabel(r'$L\left[R_E\right]$')
        axis.margins(x=0.01)
    ax3.set_xlabel('Day-Hour')
    rax.set_ylabel(r'Sym-H $\left[nT\right]$')
    rax.set_ylim([-500,200])
    rax.legend(loc='upper right')
    fig.tight_layout(pad=1)

    # Save
    figurename = f"{OUTPATH}/Lstar_time.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

#############################################################################
def add_eq_flux(itime:int,data:np.lib.npyio.NpzFile,
                 axis:plt.axis,ilowE:int,ihighE:int,
                                **kwargs:dict) -> mpl.contour.QuadContourSet:
    # Pull necessary pieces from data
    r_eq   = data['ro'][itime]
    mlt_eq = data['mlto'][itime]
    nLat,nMLT = r_eq.shape
    # X - GSM-X
    x_eq = np.zeros((nLat,nMLT+1))# Repeat for period bound
    x_eq[:,0:nMLT] = r_eq * np.cos(mlt2rad(mlt_eq))# Lat,MLT
    x_eq[:,nMLT] = x_eq[:,0]
    # Y - GSM-Y
    y_eq = np.zeros((nLat,nMLT+1))# Repeat for period bound
    y_eq[:,0:nMLT] = r_eq * np.sin(mlt2rad(mlt_eq))# Lat,MLT
    y_eq[:,nMLT] = y_eq[:,0]
    # Z - flux @ energy level
    #iE,energy_keV = find_nearest(data['E_lvls'],kwargs.get('energy_keV',500))
    flux_single_E = np.trapezoid(data['flux'][itime,:,:,ilowE:ihighE+1,:],
                                 x=data['E_lvls'][ilowE:ihighE+1],axis=2)

    # Integrate over pitch angles
    flux_PAave = np.zeros((nLat,nMLT+1))

    f_PA,fpara,fperp = integrate_f_dAlpha(data['alpha_lvls'],flux_single_E,2)
    flux_PAave[:,0:nMLT] = f_PA

    flux_PAave[:,nMLT] = flux_PAave[:,0]

    # Plot
    clevels = np.logspace(2.0,7)
    cs = axis.contourf(x_eq,y_eq,flux_PAave,clevels,cmap=mpl.cm.plasma,
                                 norm=mpl.colors.LogNorm(),extend='both')
    return cs

def add_eq_aniso(itime:int,data:np.lib.npyio.NpzFile,
                  axis:plt.axis,ilowE:int,ihighE:int,
                                **kwargs:dict) -> mpl.contour.QuadContourSet:
    # Pull necessary pieces from data
    r_eq   = data['ro'][itime]
    mlt_eq = data['mlto'][itime]
    nLat,nMLT = r_eq.shape
    # X - GSM-X
    x_eq = np.zeros((nLat,nMLT+1))# Repeat for period bound
    x_eq[:,0:nMLT] = r_eq * np.cos(mlt2rad(mlt_eq))# Lat,MLT
    x_eq[:,nMLT] = x_eq[:,0]
    # Y - GSM-Y
    y_eq = np.zeros((nLat,nMLT+1))# Repeat for period bound
    y_eq[:,0:nMLT] = r_eq * np.sin(mlt2rad(mlt_eq))# Lat,MLT
    y_eq[:,nMLT] = y_eq[:,0]
    # Z - flux @ energy level
    Elvls = data['E_lvls'][ilowE:ihighE+1]
    flux_Echannel =np.trapezoid(data['flux'][itime,:,:,ilowE:ihighE+1,:],
                                 x=Elvls,axis=2)# Lat,MLT,PA
    #iE,energy_keV = find_nearest(data['E_lvls'],kwargs.get('energy_keV',500))
    #flux_single_E = data['flux'][itime,:,:,iE,:]

    # Integrate over pitch angles
    flux_aniso = np.zeros((nLat,nMLT+1))

    f_PA,fpara,fperp = integrate_f_dAlpha(data['alpha_lvls'],flux_Echannel,2)
    flux_aniso[:,0:nMLT] = (fperp-fpara)/(fpara+fperp)

    flux_aniso[:,nMLT] = flux_aniso[:,0]

    # Plot
    clevels = np.linspace(-1,1,11)
    cs = axis.contourf(x_eq,y_eq,flux_aniso,clevels,cmap=mpl.cm.bwr,
                       extend='both')
    return cs

def plot_figure1(data:np.lib.npyio.NpzFile,itime:int,**kwargs:dict) -> None:
    """ Plot of equatorial plane flux at a specific energy level
    """
    ilowE  = kwargs.get('ilowE',7)
    ihighE = kwargs.get('ihightE',11)
    # Create figure
    fig,[ax1,ax2] = plt.subplots(1,2,figsize=[30,12])

    # Add flux equatorial panel
    cs1 = add_eq_flux(itime,data,ax1,ilowE,ihighE)
    cbar1 = fig.colorbar(cs1,format="{x:.2e}")

    # Add pitch angle anisotropy panel
    cs2 = add_eq_aniso(itime,data,ax2,ilowE,ihighE)
    cbar2 = fig.colorbar(cs2,format="{x:.2f}")

    # Decorate
    #iE,energy_keV = find_nearest(data['E_lvls'],kwargs.get('energy_keV',500))
    cbar1.set_label(
        f"Flux {data['E_lvls'][ilowE]:.0f}-{data['E_lvls'][ihighE]:.0f} keV")
    cbar2.set_label("Pitch Angle Anisotropy"+
                    r"$\left[\frac{f_{\perp}-f_{\|}}{f_{tot}}\right]$")
    for axis in [ax1,ax2]:
        axis.set_xlim([10,-10])
        axis.set_ylim([-10,10])
        axis.set_xlabel(r'X $\left[R_E\right]$')
        axis.set_ylabel(r'Y $\left[R_E\right]$')
        if type(data['times'][itime])==dt.datetime:
            axis.text(1,0.94,f"Time:{data['times'][itime]}",
                        transform=axis.transAxes,
                        horizontalalignment='right')
        else:
            axis.text(1,0.94,f'Time:{T0+dt.timedelta(hours=itime)}',
                        transform=axis.transAxes,
                        horizontalalignment='right')
        axis.margins(x=0.01)
    fig.tight_layout(pad=1)

    # Save
    figurename = f"{OUTPATH}/eq_flux/eq_flux{itime:03}.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)
#############################################################################

def plot_figure0(swmf_log:dict,omni:dict,**kwargs:dict) -> None:
    """ Plot of equatorial plane flux at a specific energy level
    """
    # Create figure
    fig,axis = plt.subplots(1,1,figsize=[24,8])

    # Plot dst
    add_dst(swmf_log['swmf_log'],omni,axis)
    axis.plot(omni['times'],omni['sym_h'],c='black',lw=2,label='Dst')

    # Decorate
    axis.set_xlabel('Day-Hour')
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
    axis.set_ylabel(r'Sym-H $\left[nT\right]$')
    axis.set_ylim([-500,200])
    fig.tight_layout(pad=1)

    # Save
    figurename = f"{OUTPATH}/dst.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)
#############################################################################


def main() -> None:
    # Collect data
    flux = np.load(f"{INPATH}/{FLUXFILE}",allow_pickle=True)
    hep = call_cdaweb_arase(T0,TEND)
    v_arase = {}
    omni = swmfpy.web.get_omni_data(T0,TEND)
    swmf_log = proc_indices.read_indices("gannon-storm/data/logs/",
                                         read_supermag=False)

    # Draw figures
    #plot_figure0(swmf_log,omni)
    #plot_figure1(flux,1080)
    #for i in range(0,len(flux['times'])):
    #    plot_figure1(flux,i)
    #    pass
    plot_figure2(flux,swmf_log,hep,omni,v_arase)
                 #ilowE=3,ihighE=5,clevels=np.logspace(4,8))

if __name__ == "__main__":
    # Globals
    global T0,INPATH,INFILE,OUTPATH

    T0 = dt.datetime(2024,5,10,13,0)
    TEND = dt.datetime(2024,5,11,17,0)
    #INPATH = "gannon_rad_belt/analysis"
    INPATH = "gannon-storm/data/large/RB"
    #FLUXFILE = "20240511_170000_e_fls.npz"
    FLUXFILE = "new_fls.npz"
    #DBFILE = "2018p001.db"
    #VSATFILE = "2018p001_rbsp-A_e_flux.npz"
    #OUTPATH = os.path.join(INPATH,"output")
    OUTPATH = os.path.join("gem2025")

    # Set pyplot configurations
    #plt.rcParams.update(pyplotsetup(mode='print'))
    settings={"text.usetex": True,
              "font.family": "sans-serif",
              "font.size": 28}
    plt.rcParams.update(settings)

    main()
