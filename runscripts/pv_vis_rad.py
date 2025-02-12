#/usr/bin/env python
import time
import glob
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
from dateutil import parser
import numpy as np
from scipy import interpolate
import datetime as dt
from geopack import geopack as gp
### Paraview stuff
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
from paraview.simple import *
#### Custom packages added manually to paraview build
import pv_magnetopause
from pv_input_tools import (read_aux, read_tecplot)
from pv_tools import  (merge_rad_mhd,project_to_iono,create_globe)
from pv_magnetopause import (setup_pipeline)
from radbelt import(read_flux)

if True:
    start_time = time.time()
    herepath=os.getcwd()+'/'
    inpath = f'{herepath}gannon_storm/outputs/large/'
    radpath= f'{inpath}RB/'
    mhdpath= f'{inpath}GM/'
    outpath = f'{herepath}gannon_storm/outputs/analysis/'
    radlist = glob.glob(f'{radpath}*.fls')
    mhdlist = glob.glob(f'{mhdpath}*.fls')
    print(f'INPATH: {inpath}')
    #renderView = GetActiveViewOrCreate('RenderView')
    t0 = dt.datetime(1970,1,1)
    for i,inrad in enumerate(radlist[0:1]):
        print(f"\t{i+1}/{len(radlist)}\t{inrad.split('/')[-1]} ...")
        fname = inrad.split('/')[-1][:-6]
        tevent = dt.datetime.strptime(fname,'%Y%m%d_%H%M%S')
        tkey=''.join(str(tevent).split('-')).replace(' ','-').replace(':','')
        inmhd = glob.glob(f'{mhdpath}*{tkey}*.plt')[0]
        aux = read_aux(inmhd.replace('.plt','.aux'))
        ut = (tevent-t0).total_seconds()
        gp.recalc(ut)
        # Create globe of earth
        create_globe(tevent)
        # Read RBE output file
        lats,mlts,Es,ys,rs,flux = read_flux(inrad)
        # Read GM 3D output file and find MP and some other stuff ...
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       inmhd,
                                                       #convert='gsm',
                                                       ut=ut,
                                                       dimensionless=False,
                                                       localtime=tevent,
                                             tilt=float(aux['BTHETATILT']),
                                                       doFieldlines=False,
                                                       path=herepath,
                                                       ffj=True,
                                                       doEntropy=True,
                                                       tail_x=-60,
                                                       aux=aux,
                                                       doEnergyFlux=False)
        # Project 3D solution down to ionosphere radius
        #north,south = project_to_iono(field,tevent)
        # Show polar caps
        #northDisplay = Show(north,GetActiveViewOrCreate('RenderView'))
        #southDisplay = Show(south,GetActiveViewOrCreate('RenderView'))
        '''
        # Create threshold of theta_1
        th_thresh = Threshold(registrationName='theta_thresh', Input=field)
        th_thresh.Scalars = ['POINTS', 'theta_1_deg']
        th_thresh.LowerThreshold = 15
        th_thresh.UpperThreshold = 72
        # Bring in radiation belt data
        rad = merge_rad_mhd(th_thresh,inrad)
        # Clip out a corner to see the cross secion
        clip = Clip(registrationName='radclip',Input=rad)
        clip.ClipType = 'Box'
        clip.Position = [0,0,-10]
        clip.Rotation = [0,0,0]
        clip.Length = [20,20,20]
        clip.Invert = 0
        '''
