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
from geopack import geopack as gp

def read_flux(infile):
    print(f"\t\treading file")
    with open(infile,'r') as f:
        # Grid
        gridline = f.readline()
        rinner,nLat,nMLT,nE,ny = gridline.split()[0:5]
        rinner = float(rinner)
        nLat,nMLT,nE,ny= int(nLat),int(nMLT),int(nE),int(ny)
        # Energy bins
        energyline1 = f.readline()
        energyline2 = f.readline()
        E_lvls = [float(v) for v in (energyline1+energyline2).split()]
        # Pitch angle bins
        yline1 = f.readline()
        yline2 = f.readline()
        y_lvls = [float(v) for v in (yline1+yline2).split()]
        # Latitude bins
        latlines = f.readline()
        latlines += f.readline()
        latlines += f.readline()
        latlines += f.readline()
        latlines += f.readline()
        latlines += f.readline()
        lat_lvls = [np.round(float(v)+1e-5,2) for v in latlines.split()]
        # MLT bins
        mlt_lvls = list(np.linspace(0,23.5,nMLT))
        # Other parameters
        paramline = f.readline()
        # Double check lats because of a dumb rounding issue
        data_start = f.tell()
        actual_lats = []
        actual_mlts = []
        for i,line in enumerate(f.readlines()):
        #TODO would love to not have to do this step...
            k = i%(1+nE)
            if k==0:
                lat,mlt,r,_,__,___,____ = [float(v) for v in line.split()]
                actual_lats.append(lat)
        lat_lvls = list(np.unique(actual_lats))
        f.seek(data_start)
        # Flux data
        flux = np.zeros([nLat,nMLT,nE,ny])
        rcross = np.zeros([nLat,nMLT])
        for i,line in enumerate(f.readlines()):
            k = i%(1+nE)
            if k==0:
                lat,mlt,r,_,__,___,____ = [float(v) for v in line.split()]
                ilat = lat_lvls.index(lat)#NOTE downgraded to .01 precision
                imlt = mlt_lvls.index(mlt)
                rcross[ilat,imlt] = r
            else:
                iE = k-1
                flux[ilat,imlt,iE,:] = [float(v) for v in line.split()]
        return lat_lvls,mlt_lvls,E_lvls,y_lvls,rcross,flux

if __name__ == "__main__":
    start_time = time.time()
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'gannon_storm/outputs/large/RB/')
    outpath = os.path.join(herepath,'gannon_storm/outputs/analysis/')
    filelist = glob.glob(f'{inpath}*.fls')
    print(f'INPATH: {inpath}')
    #renderView = GetActiveViewOrCreate('RenderView')
    t0 = dt.datetime(1970,1,1)
    for i,infile in enumerate(filelist[0:1]):
        print(f"\t{i+1}/{len(filelist)}\t{infile.split('/')[-1]} ...")
        fname = infile.split('/')[-1][:-6]
        tevent = dt.datetime.strptime(fname,'%Y%m%d_%H%M%S')
        ut = (tevent-t0).total_seconds()
        gp.recalc(ut)
        lats,mlts,Es,ys,rs,flux = read_flux(infile)
