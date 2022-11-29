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
                             display_visuals,update_rotation,read_tecplot)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)

if __name__ == "__main__":
#if True:
    start_time = time.time()
    path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
    #path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
    outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
    #from IPython import embed; embed()
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=pv_magnetopause.time_sort)
    #magfile = path+'../magnetometers_e20220202-050000.mag'
    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        oldsource,pipelinehead,field,mp=setup_pipeline(infile,aux=aux,
                                                       doEnergy=False,
                                                       dimensionless=True,
                                                       doFieldlines=True,
                                                       ffj=False,
                                                       localtime=localtime,
                                             tilt=float(aux['BTHETATILT']))
        renderView1 = GetActiveViewOrCreate('RenderView')
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=False)
        layout = GetLayout()
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
    for i,infile in enumerate(filelist[1::]):
        print('processing '+infile.split('/')[-1]+'...')
        outfile=outpath+infile.split('/')[-1].split('.plt')[0]+'.png'
        if os.path.exists(outfile):
            print(outfile+' already exists, skipping')
        else:
            #Read in new file unattached to current pipeline
            SetActiveSource(None)
            newsource = read_tecplot(infile)

            #Attach pipeline to the new source file and delete the old
            pipelinehead.Input = newsource
            Delete(oldsource)
            #Update time varying filters
            aux = read_aux(infile.replace('.plt','.aux'))
            localtime = get_time(infile)
            stations_head = FindSource('stations_pv.loc')
            stations_head.Script = update_stationHead(localtime,n=i+20)
            rotation = FindSource('rotate2GSM')
            rotation.Script = update_rotation(float(aux['BTHETATILT']))
            renderView.Update()
            # Render and save screenshot
            RenderAllViews()
            # layout/tab size in pixels
            layout.SetSize(2162, 1079)
            SaveScreenshot(outfile,layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
