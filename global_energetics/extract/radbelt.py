#!/usr/bin/env python3
"""Extracting data related to radiation belt particle distributions
"""
import os,sys,time
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import glob
import numpy as np
from numpy import (sin,cos,deg2rad,pi)
import datetime as dt
from matplotlib import pyplot as plt
from geopack import geopack as gp

def read_flux(infile:str,**kwargs:dict) -> dict:
    flux_dict = {}
    print(f"\t\treading file")
    with open(infile,'r') as f:
        # Grid
        gridline = f.readline()
        rinner,nLat,nMLT,nE,nAlpha = gridline.split('!')[0].split()[0:5]
        rinner = float(rinner)
        ntimes = 0
        nLat,nMLT,nE,nAlpha = [int(n) for n in [nLat,nMLT,nE,nAlpha]]
        #NOTE the order of which grid appears first can differ
        if any([nLat==nE,nE==nAlpha,nLat==nAlpha]):
            if kwargs.get('safe',True):
                print('ERROR!! Uncertain grid due to matching sizes ...')
                exit
            else:
                print('WARNING!! Uncertain grid due to matching sizes ...')
        grid_start = f.tell()
        total_ngrid = 0
        done = False
        while not done:
            line = f.readline()
            if '!' in line:
                done = True
            else:
                total_ngrid += len(line.split())
        if total_ngrid - (nLat+nMLT+nE+nAlpha)>0: #MLT grid inferred
            # MLT bins
            print(f'\t\t\t{nMLT} MLTs')
            mlt_lvls = list(np.linspace(0,23.5,nMLT))
            skip_mlt = True
        else:
            skip_mlt = False
        f.seek(grid_start)
        done = False
        i = 0
        unknown_lvls = np.array([])
        E_lvls = np.array([])
        alpha_lvls = np.array([])
        lat_lvls = np.array([])
        if not skip_mlt:
            mlt_lvls = np.array([])
        while not done:
            lvl_line = [float(v) for v in f.readline().split()]
            unknown_lvls = np.concat([unknown_lvls,lvl_line])
            if len(unknown_lvls)==nE:
                E_lvls = unknown_lvls
                print(f'\t\t\t{nE} Energy Levels')
                unknown_lvls = np.array([])
            elif len(unknown_lvls)==nMLT and not skip_mlt:
                mlt_lvls = unknown_lvls
                print(f'\t\t\t{nLat} MLT Levels')
                unknown_lvls = np.array([])
            elif len(unknown_lvls)==nAlpha:
                alpha_lvls = unknown_lvls
                print(f'\t\t\t{nAlpha} Pitch Angle Levels')
                unknown_lvls = np.array([])
            elif len(unknown_lvls)==nLat:
                lat_lvls = unknown_lvls
                print(f'\t\t\t{nLat} Latitude/Radial Levels')
                unknown_lvls = np.array([])
            if all([E_lvls.any(),alpha_lvls.any(),
                    lat_lvls.any(),mlt_lvls.any()]):
                done = True
            if i>(nLat+nE+nAlpha):
                print(f'FAILED i={i}... check file')
            i+=1
        data_start = f.tell()
        i_first_parmod = -999
        for i,line in enumerate(f.readlines()):
            if 'parmod' in line:
                ntimes+=1
                i_first_parmod = i
            if i==(i_first_parmod+1):
                params2 = line.split()
            if ntimes==kwargs.get('maxtimes',999): break
        print(f'\t\t\t{ntimes} Times')
        times = np.zeros(ntimes)
        lstarMax = np.zeros(ntimes)
        if len(params2)>6:
            #NOTE What I'm calling 'params2'
            # from cimi.f90 from cimipack_bb June2025, 10 values:
            #   xlati1(i,j)         - northern footpoint lat (deg)
            #   xmlt(j)             - mlt position (hours)
            #   xlatS1(i,j)         - southern footpoint lat (deg)
            #   xmltS(i,j)          - southern footpoint mlt (hours)
            #   ro1=ro(i,j)         - inner radius
            #   xmlto1=xmlto(i,j)   - ???
            #   BriN(i,j)
            #   BriS(i,j)
            #   bo(i,j)
            #   ba(j)
            # i,j are lat and lon coordinates respectively
            # o denotes inner/near planet ie Bo=earths surface field
            flux_dict['latN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mltN'] = np.zeros([ntimes,nMLT])
            flux_dict['latS'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mltS'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['ro']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mlto'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['BriN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['BriS'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['bo']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['ba']   = np.zeros([ntimes,nMLT])
            params3 = f.readline().split()
            #NOTE What I'm calling 'params3'
            # from cimi.f90 form cimipack_bb June2025, 12 values:
            #   density(i,j)
            #   ompe(i,j)
            #   CHpower(i,j)
            #   HIpower(i,j)
            #   denWP(n,i,j)
            #   TparaWP(n,i,j)
            #   TperpWP(n,i,j)
            #   HRPee(n,i,j)
            #   HRPii(n,i,j)
            #   rppa(j)
            #   Lstar(i,j,0)
            #   volume(i,j)
            # n,i,j,0 are species? lat lon ??? respectively
            flux_dict['density'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['ompe']    = np.zeros([ntimes,nLat,nMLT])
            flux_dict['CHpower'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['HIpower'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['denWP']   = np.zeros([ntimes,nLat,nMLT]) #NOTE
            flux_dict['TparaWP'] = np.zeros([ntimes,nLat,nMLT]) #NOTE
            flux_dict['TperpWP'] = np.zeros([ntimes,nLat,nMLT]) #NOTE
            flux_dict['HRPee']   = np.zeros([ntimes,nLat,nMLT]) #NOTE
            flux_dict['HRPii']   = np.zeros([ntimes,nLat,nMLT]) #NOTE
            flux_dict['rppa']    = np.zeros([ntimes,nMLT])
            flux_dict['Lstar']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['volume']  = np.zeros([ntimes,nLat,nMLT])
        else:
            #TODO for the old case
            pass
        f.seek(data_start)
        # Flux data
        flux = np.zeros([ntimes,nLat,nMLT,nE,nAlpha])
        rcross = np.zeros([ntimes,nLat,nMLT])
        if mlt_lvls.max()>7:
            mltangles = np.array([(12-mlt)*np.pi/12 for mlt in mlt_lvls])
        else:
            mltangles = mlt_lvls # already in radians for mLon
        for itime in range(0,ntimes):
            print(f'\t\tLoading time:{itime+1}/{ntimes}')
            params1 = f.readline().split('!')[0].split()
            times[itime] = params1[0]
            lstarMax[itime]  = params1[1]
            found = False
            for ilat in range(0,nLat):
                for imlt in range(0,nMLT):
                    params2 = f.readline().split()
                    if len(params2)>6:
                        flux_dict['latN'][itime,ilat,imlt] = params2[0]
                        flux_dict['mltN'][itime,imlt]      = params2[1]
                        flux_dict['latS'][itime,ilat,imlt] = params2[2]
                        flux_dict['mltS'][itime,ilat,imlt] = params2[3]
                        flux_dict['ro'][itime,ilat,imlt]   = params2[4]
                        flux_dict['mlto'][itime,ilat,imlt] = params2[5]
                        flux_dict['BriN'][itime,ilat,imlt] = params2[6]
                        flux_dict['BriS'][itime,ilat,imlt] = params2[7]
                        flux_dict['bo'][itime,ilat,imlt]   = params2[8]
                        flux_dict['ba'][itime,imlt]        = params2[9]

                        params3 = f.readline().split()
                        flux_dict['density'][itime,ilat,imlt] = params3[0]
                        flux_dict['ompe'][itime,ilat,imlt]    = params3[1]
                        flux_dict['CHpower'][itime,ilat,imlt] = params3[2]
                        flux_dict['HIpower'][itime,ilat,imlt] = params3[3]
                        flux_dict['denWP'][itime,ilat,imlt]   = params3[4]
                        flux_dict['TparaWP'][itime,ilat,imlt] = params3[5]
                        flux_dict['TperpWP'][itime,ilat,imlt] = params3[6]
                        flux_dict['HRPee'][itime,ilat,imlt]   = params3[7]
                        flux_dict['HRPii'][itime,ilat,imlt]   = params3[8]
                        flux_dict['rppa'][itime,imlt]         = params3[9]
                        flux_dict['Lstar'][itime,ilat,imlt]   = params3[10]
                        flux_dict['volume'][itime,ilat,imlt]  = params3[11]

                        rcross[itime,ilat,imlt] = params2[2]
                    else:
                        params3 = None
                        rcross[itime,ilat,imlt] = params2[2]
                    #if mlt_lvls.max()>7:
                    #    mlt_angle = (12-imlt)*np.pi/12
                    #else:
                    #    mlt_angle = imlt
                    #xcross[ilat,imlt] = rcross[ilat,imlt]*np.cos(mlt_angle)
                    #ycross[ilat,imlt] = rcross[ilat,imlt]*np.sin(mlt_angle)
                    for iE in range(0,nE):
                        pacount = 0
                        iflux = np.array([])
                        while(pacount<nAlpha):
                            line = f.readline().split()
                            iflux = np.concat([iflux,line])   
                            pacount = len(iflux)
                        flux[itime,ilat,imlt,iE,:] = iflux
        flux_dict.update({'times':times,
                          'lat_lvls':lat_lvls,'mlt_lvls':mlt_lvls,
                          'E_lvls':E_lvls,'alpha_lvls':alpha_lvls,
                          'rcross':rcross,'flux':flux})
        return flux_dict

def main() -> None:
    start_time = time.time()
    herepath=os.getcwd()
    inpath = os.path.join('/Users/ambrenne/Code/SWMF/run/IM/plots/')
    outpath = os.path.join(herepath,'gannon_rad_belt/analysis/')
    #inpath = os.path.join(herepath,'month_CIMI/')
    #outpath = os.path.join(herepath,'month_CIMI/')
    filelist = glob.glob(f'{inpath}*_e.fls')
    print(f'INPATH: {inpath}')
    #renderView = GetActiveViewOrCreate('RenderView')
    t0 = dt.datetime(1970,1,1)
    for i,infile in enumerate(filelist[0:1]):
        print(f"\t{i+1}/{len(filelist)}\t{infile.split('/')[-1]} ...")
        fname = infile.split('/')[-1][:-6]
        #tevent = dt.datetime.strptime(fname,'%Y%m%d_%H%M%S')
        tevent = dt.datetime(2018,1,1,0)
        ut = (tevent-t0).total_seconds()
        gp.recalc(ut)
        flux_dict = read_flux(infile,maxtimes=999)
        print(f"Converting {infile.split('/')[-1]} -> "+
              f"{infile.replace('.fls','_fls.npz').split('/')[-1]}")
        np.savez_compressed(infile.replace('.fls','_fls.npz'),**flux_dict)

if __name__ == "__main__":
    main()
