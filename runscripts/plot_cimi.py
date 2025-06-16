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
from cdasws import CdasWs
cdas = CdasWs()
#from plot_rbsp import call_cdaweb_rbsp

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
                                **kwargs:dict) -> mpl.contour.QuadContourSet:
    # Average flux @Energy level over all pitch angles
    Erange = kwargs.get('Erange',[16,25])
    Elvls = data['E_lvls'][Erange[0]:Erange[1]]
    flux_Echannel = np.trapezoid(data['flux'][:,:,:,Erange[0]:Erange[-1],:],
                                 x=Elvls,axis=3)# t,Lat,MLT,PA
    flux_PAave,flux_para,flux_perp = integrate_f_dAlpha(data['alpha_lvls'],
                                                        flux_Echannel,3)
    #sina = data['alpha_lvls']
    #mu_bins_left  = np.concat([[0],[0.5*(sina[i]+sina[i-1])
    #                                    for i in range(1,len(sina))]])
    #mu_bins_right = np.concat([[0.5*(sina[i+1]+sina[i])
    #                                    for i in range(0,len(sina)-1)],[1]])
    #dmu = np.sqrt(1-mu_bins_left**2)-np.sqrt(1-mu_bins_right**2)
    #flux_PAave = np.sum(flux_Echannel*dmu,axis=3)# t,Lat,MLT

    # Creat L* bins and take average at each bin
    Lstar_bins = np.linspace(-10,10,len(data['lat_lvls'])+1)
    L_ave_flux = np.zeros([len(data['times']),len(Lstar_bins)-1])
    for it in range(0,len(data['times'])):
        flux_now  = flux_PAave[it,:,:]
        lstar_now = data['Lstar'][it,:,:]
        for iL in range(0,len(Lstar_bins)-1):
            flux_at_L = flux_now[(lstar_now>=Lstar_bins[iL])&
                                 (lstar_now<=Lstar_bins[iL+1])]
            if flux_at_L.size>0:
                L_ave_flux[it,iL] = flux_at_L.mean()
            else:
                L_ave_flux[it,iL] = 0

    # Plot in log scale
    Lstar_bins_c = [(Lstar_bins[i]+Lstar_bins[i+1])/2
                                         for i in range(0,len(Lstar_bins)-1)]
    #T,L = np.meshgrid(data['times'],Lstar_bins_c)
    times = [T0+dt.timedelta(hours=t) for t in data['times']]
    T,L = np.meshgrid(times,Lstar_bins_c)
    clevels = kwargs.get('clevels',np.logspace(2.0,6.0))
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

def add_dst(cimi_db:dict,omni:dict,axis:plt.axis) -> None:
    # could return lines as list[mpl.lines.Line2D]
    # Omni Sym-H
    axis.plot(omni['times'],omni['sym_h'],c='black',lw=2,label='Dst')
    # CIMI magnetic perturbation
    #cimi_times = [T0+dt.timedelta(hours=t) for t in cimi_db['hour']]
    #axis.plot(cimi_times,cimi_db['DstRC'],c='black',lw=2,ls='--',label='CIMI')
    return
            

def plot_figure2(flux:np.lib.npyio.NpzFile,
                 cimi_db:dict,
                 mageis,rept,
                 omni:dict,
                 vobs:np.lib.npyio.NpzFile,**kwargs:dict) -> None:
    """ Plot L*-time plot with flux at a certain energy level
    """
    #TODO
    #   - add Mageis/Rept actual data as third panel
    #   - Fix the channels/contour limits for fair comparison
    #   - Update L label and check that it's consistent w real & vsat

    # Create figure
    fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=[24,12],sharex=True)

    # Add L*-time all CIMI
    cs1   = add_Lstar_time(flux,ax1,Erange=[15,24])
    cbar1 = fig.colorbar(cs1,format="{x:.1e}",pad=0.1,ticks=[1e2,1e4,1e6])
    rax = ax1.twinx()
    add_dst(cimi_db,omni,rax)

    # Add L*-time from virtual sat
    scat2 = add_Lstar_time_vsat(vobs,ax2)
    cbar2 = fig.colorbar(scat2,format="{x:.1e}",pad=0.1,extend='both')

    # Add L*-time from virtual sat
    scat3 = add_Lstar_time_rbsp(mageis,rept,ax3)
    cbar3 = fig.colorbar(scat3,format="{x:.1e}",pad=0.1,extend='both')

    # Decorate
    cbar1.set_label(
            f"Flux {flux['E_lvls'][15]:.0f}-{flux['E_lvls'][23]:.0f} keV\n"+
            r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')
    cbar2.set_label(
            f"Flux {vobs['E_lvls'][43]:.0f}-{vobs['E_lvls'][54]:.0f} keV\n"+
            r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')
    cbar3.set_label(
            f"Flux {464:.0f}-{4216:.0f} keV\n"+
            r'$\left[cm^{-2}sr^{-1}s^{-1}\right]$')

    ax1.text(0,0.84,"Jan 2018",transform=ax1.transAxes,
              horizontalalignment='left')
    for axis in [ax1,ax2,ax3]:
        axis.set_ylim([1,10])
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        axis.set_ylabel(r'$L^*\left[R_E\right]$')
        axis.margins(x=0.01)
    ax3.set_xlabel('Day')
    rax.set_ylabel(r'Dst/Sym-H $\left[nT\right]$')
    rax.set_ylim([-100,40])
    rax.legend(loc='lower right')
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
    clevels = np.logspace(2.0,6)
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

def plot_figure0(cimi_db,omni:dict,**kwargs:dict) -> None:
    """ Plot of equatorial plane flux at a specific energy level
    """
    # Create figure
    fig,axis = plt.subplots(1,1,figsize=[24,8])

    # Plot dst
    #add_dst(cimi_db,omni,axis)
    axis.plot(omni['times'],omni['sym_h'],c='black',lw=2,label='Dst')

    # Decorate
    axis.set_xlabel('Day')
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    axis.set_ylabel(r'Sym-H $\left[nT\right]$')
    axis.set_ylim([-100,40])
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
    #cimi_db = read_db_file(f"{INPATH}/{DBFILE}")
    #rbspA_mageis,rbspA_rept = call_cdaweb_rbsp(T0,TEND,'A')
    rbsp_reptL2 = {}
    #v_rbsp = np.load(f"{INPATH}/{VSATFILE}")
    omni = swmfpy.web.get_omni_data(T0,TEND)

    # Draw figures
    #plot_figure0(cimi_db,omni)
    for i,t in enumerate(flux['times'][0::60]):
        plot_figure1(flux,i)
        pass
    #plot_figure2(flux,cimi_db,rbspA_mageis,rbspA_rept,omni,v_rbsp)

if __name__ == "__main__":
    # Globals
    global T0,INPATH,INFILE,OUTPATH

    T0 = dt.datetime(2024,5,10,13,0)
    TEND = dt.datetime(2024,5,11,17,0)
    INPATH = "gannon_rad_belt/analysis"
    #FLUXFILE = "20240511_170000_e_fls.npz"
    FLUXFILE = "new_fls.npz"
    #DBFILE = "2018p001.db"
    #VSATFILE = "2018p001_rbsp-A_e_flux.npz"
    OUTPATH = os.path.join(INPATH,"output")

    # Set pyplot configurations
    #plt.rcParams.update(pyplotsetup(mode='print'))
    settings={"text.usetex": True,
              "font.family": "sans-serif",
              "font.size": 28}
    plt.rcParams.update(settings)

    main()
