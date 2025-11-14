import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10
import os,sys

import time
import glob
import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *
#### custom packages
import global_energetics
from global_energetics.extract.pv_magnetopause import (setup_pipeline)
from global_energetics.makevideo import (get_time, time_sort)
from global_energetics.extract.pv_tools import (update_rotation)
from global_energetics.extract.pv_input_tools import (read_aux, read_tecplot)
from global_energetics.extract import pv_surface_tools
from global_energetics.extract.pv_visuals import (display_visuals)
from global_energetics.extract.magnetometer import (get_stations_now,
                                                    update_stationHead)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    herepath=os.getcwd()
    #inpath = os.path.join(herepath,'localdbug/starlink/')
    #outpath= os.path.join(inpath,'test_output/')
    #inpath = os.path.join(herepath,'ccmc_2022-02-02/copy_paraview/')
    #outpath= os.path.join(herepath,'jgr2023/figures/unfiled/')
    #inpath = os.path.join(herepath,'localdbug/parameter_study/LOWnLOWu/')
    #outpath= os.path.join(herepath,'parameter_study/figures/unfiled/')
    inpath = os.path.join(herepath,'theta_aurora/outputs/data/')
    outpath= os.path.join(herepath,'theta_aurora/outputs/figures/unfiled/')

    filelist = sorted(glob.glob(inpath+'*paraview*.plt'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    '''
    filelist = [f for f in filelist if ('03-1154' in f)or#t0
                                       ('03-1204' in f)or#t1
                                       ('03-1214' in f)or#t2
                                       ('03-1224' in f)]#t3
    '''
    for i,infile in enumerate(filelist[-1::]):
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        outfile = 't'+str(i)+infile.split('_1_')[-1].split('.')[0]+'.png'
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       tail_x=-120,
                                                       dimensionless=False,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       repair_status=True,
                                                       ffj=True,
                                                       doEnergyFlux=True)
        # Split and display another
        layout = GetLayout()
        #layout.SetSize(1280, 1280)# Single hyperwall screen
        layout.SetSize(800, 1280)# Skinny for a horizontal colloage
        layout.SplitVertical(0, 0.5)
        renderView2 = CreateView('RenderView')
        AssignViewToLayout(view=renderView2, layout=layout, hint=2)
        # Adjust visuals
        for i,renderView in enumerate([renderView1,renderView2]):
            SetActiveView(renderView)
            display_visuals(field,mp,renderView,doSlice=False,doFluxVol=False,
                            fontsize=20,localtime=localtime,
                            tstart=tstart,doFFJ=True,
                            show_fte=False,
                            mpContourBy='Status',
                             contourMin=-1,
                             contourMax=3,
                             cmap='Inferno (matplotlib)',
                             show_legend=False,
                            show_mp=True,timestamp=False)
            '''
            stamp = Text(registrationName='stamp')
            if i==0:
                stamp.Text = 'NORTH'
                stampDisplay = Show(stamp,renderView,'TextSourceRepresentation')
                stampDisplay.WindowLocation = 'Upper Right Corner'
                stampDisplay.FontSize=80
            elif i==1:
                stamp.Text = 'SOUTH'
                stampDisplay = Show(stamp,renderView,'TextSourceRepresentation')
                stampDisplay.WindowLocation = 'Lower Right Corner'
                stampDisplay.FontSize=80
            stampDisplay.Color = [0.652, 0.652, 0.652]
            '''
            renderView.OrientationAxesVisibility = 0
            # paraview doesn't have the capability to change OA size from
            # python??? :'(
            #renderView.OrientationAxesInteractivity = 1

        # Overwrite camera positions
        renderView1.CameraPosition = [42.58, -6.76, 49.43]
        renderView1.CameraFocalPoint = [30.11, -4.72, 37.56]
        renderView1.CameraViewUp = [-0.69, -0.02, 0.72]

        # current camera placement for renderView2
        renderView2.CameraPosition = [43.61, -0.54, -35.94]
        renderView2.CameraFocalPoint = [-139.74, 4.88, 102.00]
        renderView2.CameraViewUp = [0.59, 0.10, 0.79]
        renderView2.CameraParallelScale = 59.4
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       #SaveAllViews=1,ImageResolution=[1280,1280])
                       SaveAllViews=1,ImageResolution=[800,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
    '''
    for i,infile in enumerate(filelist[1::]):
        print(str(i+2)+'/'+str(len(filelist))+
              ' processing '+infile.split('/')[-1]+'...')
        outfile = 't'+str(i+1)+infile.split('_1_')[-1].split('.')[0]+'.png'
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
            #timestamp1 = FindSource('tstamp')
            #timestamp1.Text = str(localtime)
            #timestamp2 = FindSource('tsim')
            #timestamp2.Text = 'tsim: '+str(localtime-tstart)
            #datacube.Script = update_datacube(path=outpath,filename=outfile)

            #Reload the view with all the updates
            renderView1.Update()
            renderView2.Update()

            # Render and save screenshot
            RenderAllViews()
            #layout.SetSize(1280, 1280)# Single hyperwall screen
            layout.SetSize(800, 1280)# Skinny for a horizontal colloage
            SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[800,1280])
            print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                             os.getcwd()))

            # Set the current source to be replaced on next loop
            oldsource = newsource
    '''
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
