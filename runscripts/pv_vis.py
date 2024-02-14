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

    #filelist = [f for f in filelist if ('020500' in f)]
    filelist = [f for f in filelist if ('03-070300' in f)]
    #or ('02200' in f)]
    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        tstart = dt.datetime(2015,11,18,1,50,0)
        outfile = 'fronton'+infile.split('_1_')[-1].split('.')[0]+'.png'
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       ffj=True,
                                                       doEnergyFlux=True)
        #get_surface_flux(mp, 'B_nT','Bnormal_net')
        #mp_Bnorm = FindSource('Bnormal_net')
        # Adjust visuals
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=False,doFluxVol=False,
                        fontsize=20,localtime=localtime,
                        mpContourBy='B_x_nT',
                            contourMin=-5,
                            contourMax=5,
                        fteContourBy='K_W_Re2',
                            ftecontourMin=-3e8,
                            ftecontourMax=3e8,
                        tstart=tstart,doFFJ=True,
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
