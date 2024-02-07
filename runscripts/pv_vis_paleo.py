#/usr/bin/env python
import os,time
import glob
from dateutil import parser
import numpy as np
import datetime as dt
from geopack import geopack as gp
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
#### import the simple module from paraview
from paraview.simple import *
#### Custom packages added manually to paraview build
import pv_magnetopause
from pv_input_tools import (read_aux, read_tecplot)
#from makevideo import get_time, time_sort
from pv_magnetopause import (setup_pipeline)
import pv_surface_tools
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)
import pv_ionosphere
import pv_tools
from pv_visuals import (display_visuals)
from pv_tabular_tools import (setup_table,save_table_data)
import equations

if True:
    start_time = time.time()
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'paleo/')
    outpath = os.path.join(herepath,'paleo/output/')
    filelist = glob.glob(inpath+'*paraview*.plt')
    print(inpath)
    print(inpath+'*paraview*.plt')
    print(filelist)
    renderView1 = GetActiveViewOrCreate('RenderView')
    for infile in filelist[0:1]:
        # Setup file specific quantities
        aux = read_aux(infile.replace('.plt','.aux'))
        outfile = ('-'.join(infile.split('_')[-2::]).replace('plt','png'))
        outfile2 = outfile.replace('.png','_2.png')
        # Initialize the geopack routines by finding the universal time
        t0 = dt.datetime(1970,1,1)
        tevent = parser.parse(aux['TIMEEVENT'])
        ut = (tevent-t0).total_seconds()
        # Setup pipeline for Magnetosphere
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       #convert='gsm',
                                                       ut=ut,
                                                       dimensionless=False,
                                                       localtime=tevent,
                                   station_file='uniform_station_grid.csv',
                                   n='all',
                                             tilt=float(aux['BTHETATILT']),
                                                       doFieldlines=True,
                                                       path=herepath,
                                                       ffj=False,
                                                       tail_x=-50,
                                                       doEnergyFlux=False)
        # Visuals
        layout = GetLayout()
        layout.SetSize(1280, 1280)# Single hyperwall screen
        display_visuals(field,mp,renderView1,doSlice=False,doFluxVol=False,
                        fontsize=20,localtime=tevent,
                        tstart=tevent,doFFJ=False,
                            show_fte=False,
                            mpContourBy='Status',
                             contourMin=-1,
                             contourMax=3,
                             cmap='Black, Blue and White',
                             show_legend=False,
                             opacity=.01,
                            show_mp=True,timestamp=False)
        fieldlines = FindSource('station_field')
        ## First image
        Show(mp)
        Hide(fieldlines)
        # Overwrite camera positions
        renderView1.CameraPosition = [48.26, 38.63, 22.51]
        renderView1.CameraFocalPoint = [34.17, 29.73, 17.72]
        renderView1.CameraViewUp = [-0.19, -0.21, 0.95]
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
        ## Second image
        Hide(mp)
        Show(fieldlines)
        # Overwrite camera positions
        renderView1.CameraPosition = [48.26, 38.63, 22.51]
        renderView1.CameraFocalPoint = [34.17, 29.73, 17.72]
        renderView1.CameraViewUp = [-0.19, -0.21, 0.95]
        # Save screenshot
        SaveScreenshot(outpath+outfile2,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))

    if False:
    #for i,infile in enumerate(filelist[1::]):
        print(str(i+2)+'/'+str(len(filelist))+
              ' processing '+infile.split('/')[-1]+'...')
        #Read in new file unattached to current pipeline
        SetActiveSource(None)
        newsource = read_tecplot(infile)
        aux = read_aux(infile.replace('.plt','.aux'))
        outfile = ('-'.join(infile.split('_')[-2::]).replace('plt','png'))

        #Attach pipeline to the new source file and delete the old
        pipelinehead.Input = newsource
        Delete(oldsource)

        # Initialize the geopack routines by finding the universal time
        t0 = dt.datetime(1970,1,1)
        tevent = parser.parse(aux['TIMEEVENT'])
        ut = (tevent-t0).total_seconds()
        if False:
            ###Update time varying filters
            # Coord transform
            gp.recalc(ut)
            gse_to_gsm = FindSource('gsm')
            gsm.Script = pv_tools.update_gse_to_gsm()

            # magnetometer stations
            station_head = FindSource('stations_input')
            station_head.Script = update_stationHead(localtime,
                                        file_in='uniform_station_grid.csv',
                                        n=nstation,path=herepath)

            #Rotation matrix from MAG->GSM
            rotation = FindSource('rotate2GSM')
            rotation.Script = update_rotation(float(aux['BTHETATILT']))

        #Reload the view with all the updates
        renderView1.Update()
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
