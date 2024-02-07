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
                             display_visuals,update_rotation,read_tecplot,
                             get_dipole_field,tec2para,update_fluxVolume,
                             update_fluxResults,export_datacube,
                             update_datacube, fix_names)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)

if __name__ == "__main__":
#if True:
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
    path='/nfs/solsticedisk/tuija/amr_fte/thirdrun/GM/IO2/'
    #path = '/home/aubr/Code/swmf-energetics/localdbug/fte/'
    outpath='/nfs/solsticedisk/tuija/amr_fte/thirdrun/'
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=pv_magnetopause.time_sort)
    renderView1 = GetActiveViewOrCreate('RenderView')

    for infile in filelist[-2:-1]:
        outfile = 'datacube'+infile.split('1118-')[-1].split('-')[0]+'.npz'
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       path=herepath,
                                                       ffj=True)
        '''
        oldsource = read_tecplot(infile)
        pipelinehead = MergeBlocks(registrationName='MergeBlocks1',
                                   Input=oldsource)
        field = fix_names(pipelinehead)
        '''
        clip = PointVolumeInterpolator(registrationName='datacubesource',
                                       Input=field, Source='Bounded Volume')
        clip.Kernel = 'VoronoiKernel'
        clip.Locator = 'Static Point Locator'
        clip.Source.Origin = [-8,-10,-8]
        clip.Source.Scale = [20,20,20]
        clip.Source.RefinementMode = 'Use cell-size'
        clip.Source.CellSize = 0.0625
        datacube = export_datacube(clip,path=outpath,filename=
             'datacube'+infile.split('1118-')[-1].split('-')[0]+'.npz')
        datacubeDisplay = Show(datacube, renderView1,
                               'UniformGridRepresentation')

    for i,infile in enumerate(filelist):
        print(str(i)+'/'+str(len(filelist))+
              ' processing '+infile.split('/')[-1]+'...')
        outfile = 'datacube'+infile.split('1118-')[-1].split('-')[0]+'.npz'
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
            datacube.Script = update_datacube(path=outpath,filename=outfile)
            #Reload the view with all the updates
            renderView1.Update()

            # Render and save screenshot
            RenderAllViews()

            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
