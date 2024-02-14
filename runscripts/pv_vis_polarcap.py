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
import pv_magnetopause
from pv_input_tools import (read_aux, read_tecplot)
from makevideo import get_time, time_sort
from pv_magnetopause import (setup_pipeline,display_visuals,update_rotation,
                             update_fluxVolume,update_fluxResults)
import pv_surface_tools
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)
import pv_ionosphere
import pv_tools
import equations

#if __name__ == "__main__":
if True:
    start_time = time.time()
    #Set Paths NOTE if running script from open paraview pwd will be
    #               where paraview was launched!
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'localdbug/polarcap2000/')
    iepath = os.path.join(herepath,'localdbug/polarcap2000/')
    outpath = os.path.join(herepath,'pc_vis_test/')
    filelist = sorted(glob.glob(inpath+'*paraview*.plt'),
                      key=time_sort)
    renderView1 = GetActiveViewOrCreate('RenderView')
    tstart = get_time(filelist[0])

    for infile in filelist[-1::]:
        # Setup file specific quantities
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        simtime = localtime-tstart
        iefile = ('it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      localtime.year-2000,
                      localtime.month,
                      localtime.day,
                      localtime.hour,
                      localtime.minute,
                      localtime.second))
        outfile = ('pc{:02d}{:02d}{:02d}-{:02d}{:02d}{:02d}-000.png'.format(
                      localtime.year,
                      localtime.month,
                      localtime.day,
                      localtime.hour,
                      localtime.minute,
                      localtime.second))
        # Setup pipeline for Magnetosphere
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       dimensionless=False,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       ffj=True,
                                                       doEnergyFlux=False)
        # Extract a discretized isosurface at R=3
        r3 = pv_tools.get_sphere_filter(field)
        sphere3 = pv_surface_tools.create_iso_surface(r3,'r_state','sphere3',
                                            calc_normals=False)
        #TODO: Create tracing routine in pv_tools
        #       Call it here to check and update all sphere3 points
        #TODO: Map sphere3 -> mapped_sphere3 according to th_1,ph_1
        #TODO: Look into interpolation functions
        # Load ionosphere data
        ie = pv_ionosphere.load_ie(iepath+iefile)
        # Rotate all the vectors from SM -> GSM
        pipeline = ie
        for i,xbase in enumerate(['x','U_x_km_s','J_x_uA_m^2','E_x_mV_m']):
            pipeline = pv_tools.rotate_vectors(pipeline,
                                               float(aux['BTHETATILT']),
                                               xbase=xbase,
                                               coordinates=(i==0))
        # Calculate the dipole field at the new locations
        alleq = equations.equations(aux=aux)
        pipeline = pv_tools.eqeval(alleq['dipole'],pipeline)
        #TODO: make Bdipole the actual B components
        # Calculate the rest of the variables
        pipeline = pv_tools.eqeval(alleq['basic3d'],pipeline)
        pipeline = pv_tools.eqeval(alleq['basic_physics'],pipeline)
        ###Energy flux variables
        '''
        if kwargs.get('doEnergyFlux',False):
            pipeline = pv_tools.eqeval(alleq['energy_flux'],pipeline)
        if kwargs.get('doVolumeEnergy',False):
            pipeline = pv_tools.eqeval(alleq['volume_energy'],pipeline)
        '''
        ###Get Vectors from field variable components
        pipeline = get_vectors(pipeline)
        #TODO: Put 'finalized' zone through the same tracing to check edges
        #TODO: Call get_global variables to get all derived quantities
    """
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
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',outpath+outfile)
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
            renderView2.Update()

            # Render and save screenshot
            RenderAllViews()
            layout.SetSize(1280, 1280)# Single hyperwall screen
            SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])

            # Set the current source to be replaced on next loop
            oldsource = newsource
    """
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
