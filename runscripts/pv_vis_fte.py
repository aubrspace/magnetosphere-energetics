"""loads, analyzes, and plots file with an FTE in it (hopefully)
"""
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')

import time
import glob
import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *
#import global_energetics.extract.pv_magnetopause
from makevideo import (get_time, time_sort)
from pv_tools import (update_rotation)
from pv_input_tools import (read_aux, read_tecplot)
import pv_surface_tools
from pv_magnetopause import (setup_pipeline)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)
from pv_visuals import (display_visuals)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    #NOTE execute script (or start program) from "swmf-energetics"
    herepath = os.getcwd()
    path=os.path.join(herepath,'localdbug/fte/')
    outpath=os.path.join(path,'png/')
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=time_sort)
    print(path)
    print(outpath)
    print(filelist)
    renderView1 = GetActiveViewOrCreate('RenderView')

    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        tstart = dt.datetime(2015,11,18,0,2,5)
        outfile = 'fronton'+infile.split('_1_')[-1].split('.')[0]+'.png'
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       ffj=False,
                                                       doFTE=True,
                                                       doEnergyFlux=True)
        #TODO Create the following static images:
        #   1. Image of the grid next to identified FTE in slice
        #   2. Image of 3D field line traces of FTE next to iso volume of FTE
        #   3. Image of Energy transport on MP surface
        #get_surface_flux(mp, 'B_nT','Bnormal_net')
        #mp_Bnorm = FindSource('Bnormal_net')
        # Adjust visuals
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=True,doFluxVol=False,
                        fontsize=40,localtime=localtime,
                        #mpContourBy='B_x_nT',
                        #    contourMin=-5,
                        #    contourMax=5,
                        #fteContourBy='K_W_Re2',
                        #    ftecontourMin=-3e8,
                        #    ftecontourMax=3e8,
                        tstart=tstart,doFFJ=False,
                        show_fte=True,
                        show_mp=True,timestamp=True)
        # Save screenshot
        layout = GetLayout()
        layout.SetSize(1280, 720)# Single hyperwall screen
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,720])
    #for i,infile in enumerate(filelist):
    if False:
        print(str(i+2)+'/'+str(len(filelist))+
              ' processing '+infile.split('/')[-1]+'...')
        outfile = 'fronton'+infile.split('_1_')[-1].split('.')[0]+'.png'
        if os.path.exists(outpath+outfile):
            print(outfile+' already exists, skipping')
        else:
            #Read in new file unattached to current pipeline
            SetActiveSource(None)
            newsource = read_tecplot(infile)

            #Attach pipeline to the new source file and delete the old
            pipelinehead.Input = newsource
            Delete(oldsource)

            ###Update time varying filters
            aux = read_aux(infile.replace('.plt','.aux'))
            localtime = get_time(infile)
            timestamp1 = FindSource('tstamp')
            timestamp1.Text = str(localtime)
            timestamp2 = FindSource('tsim')
            timestamp2.Text = 'tsim: '+str(localtime-tstart)
            #datacube.Script = update_datacube(path=outpath,filename=outfile)

            #Reload the view with all the updates
            renderView1.Update()

            # Render and save screenshot
            RenderAllViews()
            layout.SetSize(1280, 720)# Single hyperwall screen
            SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,720])

            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
