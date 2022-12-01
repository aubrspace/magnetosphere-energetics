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
                             update_fluxResults)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)

if __name__ == "__main__":
#if True:
    start_time = time.time()
    #path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
    #outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
    path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
    outpath='/home/aubr/Code/swmf-energetics/output_hyperwall_scene3/'
    #from IPython import embed; embed()
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=pv_magnetopause.time_sort)
    #magfile = path+'../magnetometers_e20220202-050000.mag'
    nstation = 379
    for infile in filelist[0:1]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        tstart = localtime
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,aux=aux,
                                                       doEnergyFlux=False,
                                                       doVolumeEnergy=True,
                                                       dimensionless=True,
                                                       doFieldlines=False,
                                                       doFluxVol=True,
                                                       blanktail=True,
                                path='/home/aubr/Code/swmf-energetics/',
                                                       ffj=False,
                                                       n=nstation,
                                                       localtime=localtime,
                                             tilt=float(aux['BTHETATILT']))
        #path='/Users/ngpdl/Code/swmf-energetics/',
        renderView1 = GetActiveViewOrCreate('RenderView')
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=False,doFluxVol=True,
                        n=nstation,fluxResults=fluxResults,fontsize=60,
                        localtime=localtime,tstart=tstart)
        layout = GetLayout()
        layout.SetSize(3840, 2160)# 4k :-)
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[3840,2160])
    for i,infile in enumerate(filelist[1:7]):
        nstation = np.minimum(nstation+i,379)
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

            ###Update time varying filters
            #auxillary data + time
            aux = read_aux(infile.replace('.plt','.aux'))
            localtime = get_time(infile)
            ##dipole field values
            #Get a new dipole field equation
            Bdx_eq,Bdy_eq,Bdz_eq = get_dipole_field(aux)#just new strings
            for comp,eq in [('Bdx',Bdx_eq),('Bdy',Bdy_eq),('Bdz',Bdz_eq)]:
                source = FindSource(comp)
                source.Function = tec2para(eq.split('=')[-1])
            #magnetometer stations
            station_head = FindSource('stations_input')
            station_head.Script = update_stationHead(localtime,n=nstation)
            #Rotation matrix from MAG->GSM
            rotation = FindSource('rotate2GSM')
            rotation.Script = update_rotation(float(aux['BTHETATILT']))
            #FluxVolume
            fluxVolume = FindSource('fluxVolume_hits')
            fluxVolume.Script = update_fluxVolume(localtime=localtime,
                                                 n=nstation)
            flux_int = FindSource('fluxInt')
            total_int = FindSource('totalInt')
            fluxResults = update_fluxResults(flux_int,total_int)
            #Annotations
            station_num = FindSource('station_num')
            station_num.Text = str(nstation)
            vol_num = FindSource('volume_num')
            vol_num.Text = '{:.2f}%'.format(fluxResults['flux_volume']/
                                          fluxResults['total_volume']*100)
            bflux_num = FindSource('bflux_num')
            bflux_num.Text = '{:.2f}%'.format(fluxResults['flux_Umag']/
                                            fluxResults['total_Umag']*100)
            dbflux_num = FindSource('dbflux_num')
            dbflux_num.Text = '{:.2f}%'.format(fluxResults['flux_Udb']/
                                            fluxResults['total_Udb']*100)
            stamp1 = FindSource('tstamp')
            stamp1.Text = str(localtime)
            stamp2 = FindSource('tsim')
            stamp2.Text = 'tsim: '+str(localtime-tstart)
            #Reload the view with all the updates
            renderView1.Update()

            # Render and save screenshot
            RenderAllViews()

            # layout/tab size in pixels
            layout.SetSize(3840, 2160)
            SaveScreenshot(outfile,layout,
                       SaveAllViews=1,ImageResolution=[3840,2160])
            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
