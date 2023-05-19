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
from pv_input_tools import (get_time, time_sort, read_aux, read_tecplot)
from pv_magnetopause import (setup_pipeline,display_visuals,update_rotation,
                             update_fluxVolume,update_fluxResults)
import pv_surface_tools
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    if 'Users' in os.getcwd():
        path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
        outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
        herepath=os.getcwd()
    elif 'aubr' in os.getcwd():
        path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
        outpath='/home/aubr/Code/swmf-energetics/output_hyperwall3_redo/'
        herepath=os.getcwd()
    elif os.path.exists('/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'):
        path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
        outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
        herepath='/Users/ngpdl/Code/swmf-energetics/'
    elif os.path.exists('/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'):
        path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
        outpath='/home/aubr/Code/swmf-energetics/output_hyperwall3_redo/'
        herepath='/home/aubr/Code/swmf-energetics/'
    #Overwrite
    #path='/nfs/solsticedisk/tuija/amr_fte/thirdrun/GM/IO2/'
    #path = '/home/aubr/Code/swmf-energetics/localdbug/fte/'
    #outpath='/nfs/solsticedisk/tuija/amr_fte/thirdrun/'
    path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
    outpath='/home/aubr/Code/swmf-energetics/output_ffj/'
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=time_sort)
    renderView1 = GetActiveViewOrCreate('RenderView')

    #or ('02200' in f)]
    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        tstart = dt.datetime(2022,2,2,5,1,0)
        outfile = 'fronton'+infile.split('_1_')[-1].split('.')[0]+'.png'
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       dimensionless=True,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       ffj=True,
                                                       doEnergyFlux=False)
        '''
        # Create lobes and closed surface
        closed = pv_surface_tools.create_iso_surface(field,'Status',
                                                     'closed_surface',
                                                     iso_value=3)
        north = pv_surface_tools.create_iso_surface(field,'Status',
                                                     'north_surface',
                                                     iso_value=2)
        south = pv_surface_tools.create_iso_surface(field,'Status',
                                                     'south_surface',
                                                     iso_value=1)
        '''
        # Split and display another
        layout = GetLayout()
        layout.SetSize(1280, 1280)# Single hyperwall screen
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
                            show_mp=True,timestamp=(i==0))
            '''
            closedDisplay=Show(closed,renderView,'GeometryRepresentation')
            ColorBy(closedDisplay, None)
            closedDisplay.AmbientColor = [1.0, 1.0, 0.0]
            closedDisplay.DiffuseColor = [1.0, 1.0, 0.0]

            northDisplay=Show(north,renderView,'GeometryRepresentation')
            ColorBy(northDisplay, None)
            northDisplay.AmbientColor = [0.333, 1.0, 1.0]
            northDisplay.DiffuseColor = [0.333, 1.0, 1.0]

            southDisplay=Show(south,renderView,'GeometryRepresentation')
            ColorBy(southDisplay, None)
            southDisplay.AmbientColor = [0.333, 0.666, 1.0]
            southDisplay.DiffuseColor = [0.333, 0.666, 1.0]
            '''
        # Overwrite camera positions

        renderView1.CameraPosition = [42.585901233081, -6.769056543479001, 49.439340467258006]
        renderView1.CameraFocalPoint = [30.11, -4.72, 37.56]
        renderView1.CameraViewUp = [-0.6917662588186009, -0.02005119590778553, 0.7218430526802792]

        # current camera placement for renderView2
        renderView2.CameraPosition = [43.614642774448406, -0.5419440783687373, -35.94152705741683]
        renderView2.CameraFocalPoint = [-139.74339938398316, 4.881931788875821, 102.00015934086913]
        renderView2.CameraViewUp = [0.5999383634463533, 0.10271526877958286, 0.7934251909442058]
        renderView2.CameraParallelScale = 59.4
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
    for i,infile in enumerate(filelist[1::]):
    #if False:
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
            renderView2.Update()

            # Render and save screenshot
            RenderAllViews()
            layout.SetSize(1280, 1280)# Single hyperwall screen
            SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])

            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
