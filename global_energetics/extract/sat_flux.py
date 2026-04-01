#!/usr/bin/env python3
"""Extracting data related to particle flux distributions from satellite trajs
"""
import os,sys,time
from glob import glob
from tqdm import tqdm
import numpy as np
from numpy import (sin,cos,deg2rad,pi)
import datetime as dt
from matplotlib import pyplot as plt
from geopack import geopack as gp
#
from global_energetics.makevideo import time_sort

def read_sat_flux(infile:str) -> np.ndarray:
    flux_dict = {}
    print(f"\t\tReading file {infile} ...")
    with open(infile,'r') as f:
        # Grid
        gridline = f.readline()
        ntimes,nE = [int(v) for v in gridline.split('!')[0].split()]
        Ebin_headline = f.readline()
        if 'energy bins' not in Ebin_headline:
            print(f"Unexpected result for line: \n\t{Ebin_headline}")
        unknown_lvls = np.array([])
        E_lvls = np.array([])
        done = False
        i = 0
        while not done:
            lvl_line = [float(v) for v in f.readline().split()]
            unknown_lvls = np.concat([unknown_lvls,lvl_line])
            if len(unknown_lvls)==nE:
                E_lvls = unknown_lvls
                print(f'\t\t\t{nE} Energy Levels')
                unknown_lvls = np.array([])
                done = True
            elif len(unknown_lvls)>2*nE:
                print(f'FAILED i={i}... check file')
                return {}
            i+=1
        param_head1 = f.readline()
        if 'iyr' not in param_head1:
            print(f"Unexpected result, no iyr line: \n\t{param_head1}")
            #NOTE params2 8 values:
            #   iyr          - year
            #   idy          - day
            #   ihr          - hour
            #   min          - minute
            #   ss           - second
            #   Lshell       - Lshell (Lstar or just L???)
            #   plsDen(m^-3) - plasma(sphere??) density
            #   Efield(V/m)  - Electric field
        flux_dict['times'] = np.array([dt.datetime(2018,1,1)]*ntimes,
                                       dtype='datetime64')
        flux_dict['Lshell'] = np.zeros([ntimes])
        flux_dict['plsDen'] = np.zeros([ntimes])
        flux_dict['Efield'] = np.zeros([ntimes])
        param_head2 = f.readline()
        if 'xgeo' not in param_head2:
            print(f"Unexpected result, no xgeo line: \n\t{param_head2}")
            #NOTE params2 6 values:
            #   xgeo     
            #   ygeo     
            #   zgeo     
            #   xsm      
            #   ysm      
            #   zsm(RE)  
        flux_dict['xgeo'] = np.zeros([ntimes])
        flux_dict['ygeo'] = np.zeros([ntimes])
        flux_dict['zgeo'] = np.zeros([ntimes])
        flux_dict['xsm']  = np.zeros([ntimes])
        flux_dict['ysm']  = np.zeros([ntimes])
        flux_dict['zsm']  = np.zeros([ntimes])

        flux_dict['flux'] = np.zeros([ntimes,nE])
        for itime in range(0,ntimes):
            print(f'\t\tLoading time:{itime+1}/{ntimes}')
            params1 = f.readline().split()
            yr     = int(params1[0])
            dy     = int(params1[1])
            hr     = int(params1[2])
            minute = int(params1[3])
            sec    = int(params1[4])
            flux_dict['times'][itime] = dt.datetime.strptime(
                              f"{yr:04}-{dy:03} {hr:02}:{minute:02}:{sec:02}",
                                                             "%Y-%j %H:%M:%S")
            flux_dict['Lshell'][itime] = params1[5]
            flux_dict['plsDen'][itime] = params1[6]
            flux_dict['Efield'][itime] = params1[7]

            params2 = f.readline().split()
            flux_dict['xgeo'][itime] = params2[0]
            flux_dict['ygeo'][itime] = params2[1]
            flux_dict['zgeo'][itime] = params2[2]
            flux_dict['xsm'][itime]  = params2[3]
            flux_dict['ysm'][itime]  = params2[4]
            flux_dict['zsm'][itime]  = params2[5]

            iflux  = np.array([])
            Ecount = 0
            while(Ecount<nE):
                line = [float(v) for v in f.readline().split()]
                iflux = np.concat([iflux,line])
                Ecount= len(iflux)
                if Ecount > 2*nE:
                    print(f'FAILED len(iflux)={Ecount}...')
                    return {}
            flux_dict['flux'][itime,:] = iflux
    # Gather in dictionary
    flux_dict.update({'E_lvls':E_lvls})

    return flux_dict

def read_coupled_cimi_sat_flux(infile:str) -> dict:
    iflux = 13#NOTE come back to this!!!

    # Reads .sat file from IM output
    sat_flux = {}
    with open(infile,'r') as f:
        title   = f.readline()
        headers = f.readline().split()
        raw     = f.readlines()
    nE,nAlpha =np.array(title.replace('nEnergy=',''
                            ).replace('nAngle=','').split()[-2::],dtype='int')
    sat_flux['E_lvls'] = np.array([h.split('keV')[0] for h in headers
                                              if '@00deg' in h],dtype='float')
    sat_flux['alpha_lvls'] = np.array([h.split('@')[-1].split('deg')[0]
                                       for h in headers if '1.0keV' in h],
                                                                dtype='int')
    # Convert string information to one long array of floats
    flat_data = np.array(''.join(raw).split(),dtype='float')
    # Save off pieces that aren't part of the flux matrix
    for icol,header in enumerate([h for h in headers if 'keV' not in h]):
        if icol<8:
            dtype = 'int'
        else:
            dtype = 'float'
        sat_flux[header] = np.array(flat_data[icol::len(headers)],dtype=dtype)
    nt = len(sat_flux['it'])
    # Save the flux matrix
    sat_flux['flux'] = np.array(
                           [flat_data[iflux+len(headers)*i:len(headers)*(i+1)]
                               for i in range(0,nt)],
                                       dtype='float').reshape([nt,nAlpha,nE])
    sat_flux['flux'] = sat_flux['flux'].transpose([0,2,1])
    sat_flux['flux'][sat_flux['flux']<1e-2] = 1e-2
    # Combine the time info into numpy datetime64 type
    time_stack = np.stack([sat_flux['year'],
                           sat_flux['mo'],
                           sat_flux['dy'],
                           sat_flux['hr'],
                           sat_flux['mn'],
                           sat_flux['sc']],axis=1)
    sat_flux['time'] = np.array([dt.datetime(*stack) for stack in time_stack],
                                dtype=np.datetime64)
    # Remove piecewise time objects
    sat_flux.pop('year')
    sat_flux.pop('mo')
    sat_flux.pop('dy')
    sat_flux.pop('hr')
    sat_flux.pop('mn')
    sat_flux.pop('sc')
    sat_flux.pop('msc')
    return sat_flux

def condense_time_info(sat_flux:dict) -> dict:
    new_data = sat_flux.copy()


def main() -> None:
    filelist = sorted(glob(f"{FILEPATH}/{FILEKEY}"),key=time_sort)
    print(f"Converting {FILEKEY} -> {OUTFILE}")
    # TODO (optional) generate sat_flux file from .fls and trajectory
    # Read sat_flux file

    if False:
        sat_flux = read_sat_flux(filelist[0])
    else:
        sat_flux = {}
        for ifile,f in enumerate(tqdm(filelist)):
            latest_flux = read_coupled_cimi_sat_flux(f)
            if ifile==0:
                sat_flux = latest_flux
            else:
                keep = sat_flux['it']<latest_flux['it'][0]
                for key in sat_flux:
                    if 'lvls' not in key:
                        sat_flux[key] = np.concat([sat_flux[key][keep],
                                                   latest_flux[key]])
    # Save as .npz
    np.savez_compressed(f"{OUTPATH}/{OUTFILE}",**sat_flux)
    print(f"SAVED: {OUTPATH}/{OUTFILE}")

if __name__ == "__main__":
    start_time = time.time()
    global FILEPATH,FILEKEY,OUTPATH,OUTFILE

    for filekey in [
            'sat_arase_eflux_*.sat',
            'sat_arase_hflux_*.sat',
            'sat_arase_oflux_*.sat',
            'sat_cluster1_eflux_*.sat',
            'sat_cluster1_hflux_*.sat',
            'sat_cluster1_oflux_*.sat',
            'sat_cluster2_eflux_*.sat',
            'sat_cluster2_hflux_*.sat',
            'sat_cluster2_oflux_*.sat',
            'sat_cluster3_eflux_*.sat',
            'sat_cluster3_hflux_*.sat',
            'sat_cluster3_oflux_*.sat',
            'sat_cluster4_eflux_*.sat',
            'sat_cluster4_hflux_*.sat',
            'sat_cluster4_oflux_*.sat',
            'sat_goes16_eflux_*.sat',
            'sat_goes16_hflux_*.sat',
            'sat_goes16_oflux_*.sat',
            'sat_goes17_eflux_*.sat',
            'sat_goes17_hflux_*.sat',
            'sat_goes17_oflux_*.sat',
            'sat_mms1_eflux_*.sat',
            'sat_mms1_hflux_*.sat',
            'sat_mms1_oflux_*.sat',
            'sat_themisA_eflux_*.sat',
            'sat_themisA_hflux_*.sat',
            'sat_themisA_oflux_*.sat',
            'sat_themisD_eflux_*.sat',
            'sat_themisD_hflux_*.sat',
            'sat_themisD_oflux_*.sat',
            'sat_themisE_eflux_*.sat',
            'sat_themisE_hflux_*.sat',
            'sat_themisE_oflux_*.sat']:
        FILEPATH = 'data/large/IM/plots'
        #FILEKEY  = 'sat_themisE_eflux_*.sat'
        FILEKEY  = filekey
        OUTPATH  = 'data/sat'
        OUTFILE  = FILEKEY.replace('_*.sat','.npz')

        main()

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),np.mod(ltime,60)))
