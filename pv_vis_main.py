import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

import os
import time
import glob
import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *
#import global_energetics.extract.pv_magnetopause
import pv_magnetopause
from pv_magnetopause import (get_time, time_sort, read_aux, setup_pipeline,
                             display_visuals)
import magnetometer
from magnetometer import(get_stations_now)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    #path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
    path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
    outpath = 'vis_com_pv/'
    #from IPython import embed; embed()
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=pv_magnetopause.time_sort)
    #magfile = path+'../magnetometers_e20220202-050000.mag'
    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        #oldsource,pipelinehead,field,mp=setup_pipeline(infile,aux=aux,
        setup_pipeline(infile,aux=aux,
                                                       doEnergy=False,
                                                       dimensionless=True,
                                                       doFieldlines=True,
                                                       ffj=False,
                                                       localtime=localtime,
                                             tilt=float(aux['BTHETATILT']))
        #add_fieldlines(field, station_file=magfile,localtime=localtime,
        #               tilt=float(aux['BTHETATILT']))
        #IDs, station_df = magnetometer.get_stations_now(magfile,localtime,
        #                                   tilt=float(aux['BTHETATILT']))
        '''
        renderView1 = GetActiveViewOrCreate('RenderView')
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=False)
        layout = GetLayout()
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
        '''
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
