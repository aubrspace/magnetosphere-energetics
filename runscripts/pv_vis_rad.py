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
paraview.compatibility.major = 6
paraview.compatibility.minor = 0
from paraview.simple import *
#### Custom packages added manually to paraview build
import pv_magnetopause
from pv_input_tools import (read_aux, read_tecplot,read_npz)
from pv_tools import  (merge_rad_mhd,project_to_iono,create_globe)
from pv_magnetopause import (setup_pipeline)
from radbelt import(read_flux)

def get_time(fname:str) -> dt.datetime:
    return dt.datetime.strptime(fname.split('/')[-1],'%Y%m%d_%H%M%S_e.fls')

if True:
    start_time = time.time()
    herepath=os.getcwd()+'/'
    inpath = f'{herepath}gannon-storm/data/large/'
    radpath= f'{inpath}RB/'
    mhdpath= f'{inpath}GM/IO2/'
    outpath = f'{herepath}gannon-storm/outputs/vis/'
    radlist = sorted(glob.glob(f'{radpath}*.fls'),key=get_time)
    print(f'INPATH: {inpath}')
    #renderView = GetActiveViewOrCreate('RenderView')
    t0 = dt.datetime(1970,1,1)
    #rb_data = dict(np.load(f"{radpath}new_fls.npz",allow_pickle=True))
    #rb = read_npz(f"{radpath}new_fls.npz",'e_flux_from_npz')
    for i,inrad in enumerate(radlist[65:66]):
        #print(f"\t{i+1}/{len(radlist)}\t{inrad.split('/')[-1]} ...")
        tevent = get_time(inrad)
        tkey=''.join(str(tevent).split('-')).replace(' ','-').replace(':','')
        inmhd = glob.glob(f'{mhdpath}*{tkey}*.plt')[0]
        aux = read_aux(inmhd.replace('.plt','.aux'))
        ut = (tevent-t0).total_seconds()
        gp.recalc(ut)
        # Create globe of earth
        create_globe(tevent)
        # Read RBE output file
        #lats,mlts,Es,ys,rs,flux = read_flux(inrad)
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
                                                       ffj=False,
                                                       doEntropy=False,
                                                       tail_x=-60,
                                                       aux=aux,
                                                       doEnergyFlux=False)
        #rad = merge_rad_mhd(field,f"{radpath}new_fls.npz",65)

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
