#!/usr/bin/env python3
"""Extracting data related to particle flux distributions from satellite trajs
"""
import os,sys,time
#sys.path.append(os.getcwd().split('swmf-energetics')[0]+
#                                      'swmf-energetics/')
import glob
import numpy as np
from numpy import (sin,cos,deg2rad,pi)
import datetime as dt
from matplotlib import pyplot as plt
from geopack import geopack as gp

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

def main() -> None:
    infile = "2018p001_rbsp-A_e.flux"
    # TODO (optional) generate sat_flux file from .fls and trajectory
    # Read sat_flux file
    sat_flux = read_sat_flux("2018p001_rbsp-A_e.flux")
    # Save as .npz
    print(f"Converting {infile.split('/')[-1]} -> "+
          f"{infile.replace('.flux','_flux.npz').split('/')[-1]}")
    np.savez_compressed(infile.replace('.flux','_flux.npz'),**sat_flux)

if __name__ == "__main__":
    start_time = time.time()

    main()

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),np.mod(ltime,60)))
