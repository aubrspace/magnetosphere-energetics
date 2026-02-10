#!/usr/bin/env python3
"""Extracting data related to radiation belt particle distributions
"""
import os,sys,time
import glob
import numpy as np
from numpy import (sin,cos,deg2rad,pi)
import datetime as dt
from matplotlib import pyplot as plt
from geopack import geopack as gp
from tqdm import tqdm
##
from global_energetics.makevideo import time_sort

def is_data_line(line:str) -> bool:
    return all([e.replace('.','').replace('-','').replace(
                                        '+','').replace('E','').isnumeric()
                for e in line.split('!')[0].split()])

def pull(line:str) -> np.ndarray:
    return np.array([float(v) for v in line.split()])

def read_rtp(infile:str,**kwargs:dict) -> dict:
    rtp = {}
    print(f"READING .rtp file: {infile} ...")
    with open(infile,'r') as f:
        #####################################################################
        # Grid
        gridline = f.readline()
        nr,nth,nphi = [int(v) for v in gridline.split('!')[0].split()[0:3]]
        i,done = 0,False
        buffer = np.array([])
        r_lvls = np.array([])
        th_lvls = np.array([])
        phi_lvls = np.array([])
        while not done:
            line = pull(f.readline())
            print(line)
            buffer = np.concat([buffer,line])
            if len(buffer) == nr and min(buffer)>1.0 :
                r_lvls = buffer[0:nr]
                buffer = buffer[nr:]
                if kwargs.get('verbose',False):
                    print(f'\t\t\t{nr} R Levels')
            elif len(buffer)==nth and max(buffer)<(2*np.pi):
                th_lvls = buffer[0:nth]
                buffer = buffer[nth:]
                if kwargs.get('verbose',False):
                    print(f'\t\t\t{nth} Theta Levels')
            elif (len(buffer) == nphi and max(buffer)>(np.pi)
                  and min(buffer)<1.0):
                phi_lvls = buffer[0:nphi]
                buffer = buffer[nphi:]
                if kwargs.get('verbose',False):
                    print(f'\t\t\t{nphi} Phi Levels')
            if all([r_lvls.any(),th_lvls.any(),phi_lvls.any()]):
                done = True
            if i>(nr+nth+nphi):
                done = True
                if kwargs.get('verbose',False):
                    print(f'FAILED i={i}... check file')
            i+=1
        ####################################################################
        # Get ntimes
        ntimes = 0
        data_start = f.tell()
        for line in f.readlines():
            if 'hour' in line:
                ntimes+=1
        f.seek(data_start)
        times = np.zeros(ntimes)
        #####################################################################
        lines = f.readlines()
    ngrid = nr*nth*nphi
    Bx   = np.zeros([ntimes,nr,nth,nphi])
    By   = np.zeros([ntimes,nr,nth,nphi])
    Bz   = np.zeros([ntimes,nr,nth,nphi])
    Ro   = np.zeros([ntimes,nr,nth,nphi])
    MLTo = np.zeros([ntimes,nr,nth,nphi])
    iline = 0
    for ihour in tqdm(range(0,ntimes)):
        times[ihour] = pull(lines[iline].split('!')[0])[0]
        iline = int(iline+1)
        bigline = ''.join(lines[iline:int(iline+ngrid*5/7)])
        buffer = pull(bigline.replace('\n',''))
        Bx[ihour,:,:,:]   = buffer[0:ngrid].reshape(nr,nth,nphi)*1e9
        By[ihour,:,:,:]   = buffer[ngrid:2*ngrid].reshape(nr,nth,nphi)*1e9
        Bz[ihour,:,:,:]   = buffer[2*ngrid:3*ngrid].reshape(nr,nth,nphi)*1e9
        Ro[ihour,:,:,:]   = buffer[3*ngrid:4*ngrid].reshape(nr,nth,nphi)
        MLTo[ihour,:,:,:] = buffer[4*ngrid:5*ngrid].reshape(nr,nth,nphi)
        iline = int(iline+5*ngrid/7)
    rtp.update({'times':times,
                'r_lvls':r_lvls,'th_lvls':th_lvls,'phi_lvls':phi_lvls,
                'Bx':Bx,'By':By,'Bz':Bz,
                'Ro':Ro,'MLTo':MLTo})
    return rtp

def read_flux_data(f,
            flux_dict:dict,
                 flux:np.ndarray,
                times:np.ndarray,
                itime:int,
                 nLat:int,
                 nMLT:int,
                   nE:int,
               nAlpha:int) -> None:#NOTE modifies flux_dict
    line = f.readline()
    if is_data_line(line):
        params1 = line.split('!')[0].split()
    else:
        params1 = pull(f.readline())
    times[itime] = params1[0]
    #lstarMax[itime]  = params1[-1]
    found = False
    for ilat in range(0,nLat):
        for imlt in range(0,nMLT):
            line = f.readline()
            if is_data_line(line):
                params2 = pull(line)
            else:
                # Read two lines of header and 2 lines of values
                head1 = line.split()
                head2 = f.readline().split()
                params2 = pull(f.readline())
                params2 = np.concat([params2,pull(f.readline())])
                # One more header line for the flux
                flux_header = f.readline()
            if len(params2)>9:
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

                if len(params2)<20:
                    params3 = f.readline().split()
                else:
                    params3 = params2[10:22]

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

            elif len(params2)>6:
                flux_dict['latN'][itime,ilat,imlt] = params2[0]
                flux_dict['mltN'][itime,ilat,imlt] = params2[1]
                flux_dict['ro'][itime,ilat,imlt]   = params2[2]
                flux_dict['mlto'][itime,ilat,imlt] = params2[3]
                flux_dict['bo'][itime,ilat,imlt]   = params2[4]
                flux_dict['irm'][itime,ilat,imlt]  = params2[5]
                flux_dict['iba'][itime,ilat,imlt]  = params2[6]
                params3 = None
            else:
                flux_dict['latN'][itime,ilat,imlt] = params2[0]
                flux_dict['mltN'][itime,ilat,imlt] = params2[1]
                flux_dict['ro'][itime,ilat,imlt]   = params2[2]
                flux_dict['mlto'][itime,ilat,imlt] = params2[3]
                flux_dict['bo'][itime,ilat,imlt]   = params2[4]
                flux_dict['rm'][itime,ilat,imlt]   = params2[5]
                params3 = None
            done = False
            flux_segment = np.array([])
            while not done:
                line = f.readline()
                flux_segment =np.concat([flux_segment,pull(line)])
                if len(flux_segment)>=(nE*nAlpha):
                    done = True
            flux[itime,ilat,imlt,:,:] = flux_segment.reshape(nE,nAlpha)

def read_flux(infile:str,**kwargs:dict) -> dict:
    flux_dict = {}
    if kwargs.get('verbose',False):
        print(f"\t\treading file")
    with open(infile,'r') as f:
        #####################################################################
        # Read Grid info
        line = f.readline()
        if is_data_line(line):
            gridline = line
        else:
            gridline = f.readline()
            if not is_data_line(gridline):
                print('ERROR!! bad gridline')
                return
        rinner,nLat,nMLT,nE,nAlpha = gridline.split('!')[0].split()[0:5]
        rinner = float(rinner)
        ntimes = 0
        nLat,nMLT,nE,nAlpha = [int(n) for n in [nLat,nMLT,nE,nAlpha]]
        #NOTE the order of which grid appears first can differ
        if any([nLat==nE,nE==nAlpha,nLat==nAlpha]) and kwargs.get(
                                                             'verbose',False):
            if kwargs.get('safe',False):
                print('ERROR!! Uncertain grid due to matching sizes ...')
                return
            else:
                print('WARNING!! Uncertain grid due to matching sizes ...')
        grid_start = f.tell()
        total_ngrid = 0
        done = False
        no_grid_headers = True
        while not done:
            line = f.readline()
            if 'parmod' in line:
                done = True
            elif is_data_line(line):
                total_ngrid += len(line.split())
            else:#we've found a header line
                no_grid_headers = False
        if total_ngrid - (nLat+nMLT+nE+nAlpha)<0: #MLT grid inferred
            # MLT bins
            if kwargs.get('verbose',False):
                print(f'\t\t\t{nMLT} MLTs')
            mlt_lvls = np.linspace(0,23.5,nMLT)
            skip_mlt = True
        else:
            skip_mlt = False
        f.seek(grid_start)
        if not no_grid_headers:
            grid_header = f.readline()
        done = False
        i = 0
        unknown_lvls = np.array([])
        E_lvls = np.array([])
        alpha_lvls = np.array([])
        lat_lvls = np.array([])
        if not skip_mlt:
            mlt_lvls = np.array([])
        while not done:
            line = f.readline()
            if is_data_line(line):
                lvl_line = [float(v) for v in line.split()]
                unknown_lvls = np.concat([unknown_lvls,lvl_line])
                if len(unknown_lvls)==nE and max(unknown_lvls)>90:
                    E_lvls = unknown_lvls
                    if kwargs.get('verbose',False):
                        print(f'\t\t\t{nE} Energy Levels')
                    unknown_lvls = np.array([])
                elif len(unknown_lvls)==nMLT and not skip_mlt:
                    mlt_lvls = unknown_lvls
                    if kwargs.get('verbose',False):
                        print(f'\t\t\t{nMLT} MLT Levels')
                    unknown_lvls = np.array([])
                elif len(unknown_lvls)==nAlpha and max(unknown_lvls)<1:
                    alpha_lvls = unknown_lvls
                    if kwargs.get('verbose',False):
                        print(f'\t\t\t{nAlpha} Pitch Angle Levels')
                    unknown_lvls = np.array([])
                elif (len(unknown_lvls)==nLat and max(unknown_lvls)<90 and
                      max(unknown_lvls)>(2*np.pi)):
                    lat_lvls = unknown_lvls
                    if kwargs.get('verbose',False):
                        print(f'\t\t\t{nLat} Latitude/Radial Levels')
                    unknown_lvls = np.array([])
                if all([E_lvls.any(),alpha_lvls.any(),
                        lat_lvls.any(),mlt_lvls.any()]):
                    done = True
                if i>(nLat+nE+nAlpha):
                    done = True
                    if kwargs.get('verbose',False):
                        print(f'FAILED i={i}... check file')
                i+=1
            else:
                grid_header = line
        if mlt_lvls.max()>7:
            mltangles = np.array([(12-mlt)*np.pi/12 for mlt in mlt_lvls])
        else:
            mltangles = mlt_lvls # already in radians for mLon
        data_start = f.tell()
        #####################################################################
        # Loop through and verify the number of time steps
        i_first_parmod = -999
        for i,line in enumerate(f.readlines()):
            if 'parmod' in line:
                ntimes+=1
                if is_data_line(line) and i_first_parmod==-999:
                    i_first_parmod = i
                elif i_first_parmod==-999:
                    i_first_parmod = i+1
            if i==(i_first_parmod+1):
                params2 = line.split()
            if ntimes==kwargs.get('maxtimes',999): break
        if kwargs.get('verbose',False):
            print(f'\t\t\t{ntimes} Times')
        times = np.zeros(ntimes)
        lstarMax = np.zeros(ntimes)
        f.seek(data_start)
        #####################################################################
        # Initialize aux data
        #   First hour, ("parmods"), L*max
        line = f.readline()
        if not is_data_line(line):
            params1 = pull(f.readline())
        #   Then set of params
        line = f.readline()
        if not is_data_line(line):
            # Read two lines of header and then 2 lines of values
            head1 = line.split()
            head2 = f.readline().split()
            head = np.concat([head1,head2])
            params2 = pull(f.readline())
            params2 = np.concat([params2,pull(f.readline())])
        else:
            params2 = pull(line)
        if len(params2)>9:
            #NOTE What I'm calling 'params2'
            # from cimi.f90 from cimipack_bb June2025, 10 values:
            #   xlati1(i,j)         - northern footpoint lat (deg)
            #   xmlt(j)             - mlt position (hours)
            #   xlatS1(i,j)         - southern footpoint lat (deg)
            #   xmltS(i,j)          - southern footpoint mlt (hours)
            #   ro1=ro(i,j)         - inner radius
            #   xmlto1=xmlto(i,j)   - eq mapped mlt
            #   BriN(i,j)           - North mirror point B
            #   BriS(i,j)           - South mirror point B
            #   bo(i,j)             - eq mapped B field
            #   ba(j)               - index of last B field
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
        elif len(params2)>6:
            #NOTE What I'm calling 'params2'
            #NOTE 'params2' from the coupled version
            # lat   - footpoint lat in N ionosphere
            # mlt   - MLT in N ionosphere
            # ro    - eq mapped radius
            # mlto  - eq mapped MLT
            # bo    - eq magnetic field strength
            # irm   - index of latitude of last modeled field line
            # iba   - ???
            flux_dict['latN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mltN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['ro']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mlto'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['bo']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['irm']  = np.zeros([ntimes,nLat,nMLT])
            flux_dict['iba']  = np.zeros([ntimes,nLat,nMLT])
        else:
            flux_dict['latN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mltN'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['ro']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['mlto'] = np.zeros([ntimes,nLat,nMLT])
            flux_dict['bo']   = np.zeros([ntimes,nLat,nMLT])
            flux_dict['rm']  = np.zeros([ntimes,nLat,nMLT])
        f.seek(data_start)
        #####################################################################
        # Read Flux data
        flux = np.zeros([ntimes,nLat,nMLT,nE,nAlpha])
        if ntimes>1:
            for itime in tqdm(range(0,ntimes)):
                read_flux_data(f,flux_dict,flux,times,
                                           itime,nLat,nMLT,nE,nAlpha)
        else:
            read_flux_data(f,flux_dict,flux,times,
                                          0,nLat,nMLT,nE,nAlpha)
    flux_dict.update({'times':times,
                      'lat_lvls':lat_lvls,'mlt_lvls':mlt_lvls,
                      'E_lvls':E_lvls,'alpha_lvls':alpha_lvls,
                      'flux':flux})
    return flux_dict

def main() -> None:
    herepath=os.getcwd()
    arguments = sys.argv
    if '-i' in arguments:
        indir = arguments[arguments.index('-i')+1]
    else:
        indir = 'run_quiet_RBSP2/'
    inpath = os.path.join(herepath,indir)
    outpath = os.path.join(herepath,indir)
    print(f'INPATH: {inpath}/')

    if '-rtp' in arguments:
        infile = glob.glob('*.rtp')[0]
        rtp = read_rtp(infile,verbose=True)
        np.savez_compressed(f"{inpath}/{infile.replace('.rtp','_rtp.npz')}",
                            **rtp)
        print(f"SAVED: {inpath}/{infile.replace('.rtp','_rtp.npz')}")
    else:
        filelist = sorted(glob.glob(f'{inpath}/*_o.fls'),key=time_sort)
        t0 = dt.datetime(1970,1,1)
        flux_dict = {}
        if len(filelist)==1:
            fname = infile.split('/')[-1][:-6]
            flux_dict = read_flux(infile,maxtimes=999,verbose=i==0)
        else:
            for i,infile in enumerate(tqdm(filelist)):
                fname = infile.split('/')[-1][:-6]
                flux = read_flux(infile,maxtimes=999,verbose=i==0)
                if i==0:
                    flux_dict = flux
                else:
                    for key in flux:
                        if len(flux[key].shape) > 2 or 'time' in key:
                            flux_dict[key] = np.concat([flux_dict[key],
                                                        flux[key]])
        print(f"Converting {infile.split('/')[-1]} -> "+
            f"{infile.replace('.fls','_fls.npz').split('/')[-1]}")
        np.savez_compressed(infile.replace('.fls','_fls.npz'),**flux_dict)
    # TODO
    # - see Roeder 1970 book section IV.4 application for mapping flux
    #    in B-L coordinates
    # - Then Suk-Bins code calc_Lstar2.f90 from cimipak bb from 2015
    # - Implement numerical integrations to recover Lstar if only given particle
    #    fluxes and foot points
    # - See how the same calculations can be leveraged to provide better Lstar
    #    values for data-model comparison (use sim fields to get integrals)
    # - Is there a way to more easily extract this info from BATSRUS?

    # TODO
    # Try to feed in RBE flux to sat_flux.f90
    #   - might have to trick it by rewriting the flux file to look like the
    #       cimi one...
    #   - alternatively, could try to also rewrite this part in python but ...

if __name__ == "__main__":
    start_time = time.time()
    main()
