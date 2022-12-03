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
    outpath='/home/aubr/Code/swmf-energetics/output_hyperwall1_redo/'
    #from IPython import embed; embed()
    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=pv_magnetopause.time_sort)
    #magfile = path+'../magnetometers_e20220202-050000.mag'
    renderView1 = GetActiveViewOrCreate('RenderView')

    if False:
        ###ADD a sphere at 0.99 Re
        earth = Sphere(registrationName='earth')
        earth.Radius = 0.99
        earth.ThetaResolution = 64
        earth.PhiResolution = 64
        earthDisplay = Show(earth, renderView1, 'GeometryRepresentation')
        # change solid color
        earthDisplay.AmbientColor = [0.266, 0.266, 0.266]
        earthDisplay.DiffuseColor = [0.266, 0.266, 0.266]
        ###

    nstation = 379
    #for infile in filelist[1:2]:
    for infile in filelist[1835:1836]:
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        #tstart = localtime
        tstart = dt.datetime(2022,2,2,5,1,0)
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,aux=aux,
                                                       doEnergyFlux=False,
                                                       doVolumeEnergy=True,
                                                       dimensionless=True,
                                                       doFieldlines=True,
                                                       doFluxVol=True,
                                                       blanktail=False,
                               path='/home/aubr/Code/swmf-energetics/',
                                                       ffj=False,
                                                       n=nstation,
                                                       localtime=localtime,
                                             tilt=float(aux['BTHETATILT']))
        #path='/Users/ngpdl/Code/swmf-energetics/',
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=False,doFluxVol=True,
                        n=nstation,fontsize=60,localtime=localtime,
                        tstart=tstart,
                        station_tag=True,show_mp=True,timestamp=True,
                        fluxResults=fluxResults)
        layout = GetLayout()
        layout.SetSize(3840, 2160)# 4k :-)
        #layout.SetSize(1280, 720)# Single hyperwall screen

        if False:
            ###ADD Boarder around viewing area
            #top
            topLine = Line(registrationName='topboarder')
            topLine.Point1 = [1.6683606866950118, -0.5274813310723943, 1.0005165752517748]
            topLine.Point2 = [-0.7685966178322872, 1.7414831498223506, 1.0912349526270746]
            topLine.Resolution = 124
            topLineDisplay=Show(topLine,renderView1,'GeometryRepresentation')
            topLineDisplay.SetRepresentationType('Point Gaussian')
            #bot
            botLine = Line(registrationName='botboarder')
            botLine.Point1 = [1.855088901264601, -0.23956512402252628, -0.8380529085398969]
            botLine.Point2 = [-0.5803520914464642, 2.0226136883153067, -0.7258559430376827]
            botLine.Resolution = 124
            botLineDisplay=Show(botLine,renderView1,'GeometryRepresentation')
            botLineDisplay.SetRepresentationType('Point Gaussian')
            #left
            leftLine = Line(registrationName='leftboarder')
            leftLine.Point1 = [1.8567580235664725, -0.24175455023961234, -0.835480258237455]
            leftLine.Point2 = [1.6776231014853114, -0.49949578020298446, 0.9996421812086711]
            leftLine.Resolution = 72
            leftLineDisplay=Show(leftLine,renderView1,'GeometryRepresentation')
            leftLineDisplay.SetRepresentationType('Point Gaussian')
            #right
            rightLine = Line(registrationName='rightboarder')
            rightLine.Point1 = [-0.793705544431262, 1.7271580274801437, 1.0961207825756434]
            rightLine.Point2 = [-0.580981501635307, 2.0213264921316583, -0.7183947482197919]
            rightLine.Resolution = 72
            rightLineDisplay=Show(rightLine,renderView1,'GeometryRepresentation')
            rightLineDisplay.SetRepresentationType('Point Gaussian')
            ###
    '''
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[3840,2160])
        #SaveAllViews=1,ImageResolution=[1280,720])
    nstation_start = nstation
    for i,infile in enumerate(filelist[481:1081]):
        #for i,infile in enumerate(filelist[481:483]):
        #for i,infile in enumerate(filelist[1836:1866]):
        nstation = np.minimum(nstation_start+i+1,379)
        print('n= ',nstation,'processing '+infile.split('/')[-1]+'...')
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
            if False:
                #FluxVolume
                fluxVolume = FindSource('fluxVolume_hits')
                fluxVolume.Script = update_fluxVolume(localtime=localtime,
                                                      n=nstation)
                flux_int = FindSource('fluxInt')
                total_int = FindSource('totalInt')
                fluxResults = update_fluxResults(flux_int,total_int)
            if True:
                #Annotations
                station_num = FindSource('station_num')
                station_num.Text = str(nstation)
                stamp1 = FindSource('tstamp')
                stamp1.Text = str(localtime)
                stamp2 = FindSource('tsim')
                stamp2.Text = 'tsim: '+str(localtime-tstart)

            if False:
                vol_num = FindSource('volume_num')
                vol_num.Text = '{:.2f}%'.format(fluxResults['flux_volume']/
                                          fluxResults['total_volume']*100)
                bflux_num = FindSource('bflux_num')
                bflux_num.Text = '{:.2f}%'.format(fluxResults['flux_Umag']/
                                            fluxResults['total_Umag']*100)
                dbflux_num = FindSource('dbflux_num')
                dbflux_num.Text = '{:.2f}%'.format(fluxResults['flux_Udb']/
                                            fluxResults['total_Udb']*100)
            #Reload the view with all the updates
            renderView1.Update()

            # Render and save screenshot
            RenderAllViews()

            # layout/tab size in pixels
            layout.SetSize(3840, 2160)
            #layout.SetSize(1280, 720)# Single hyperwall screen
            SaveScreenshot(outfile,layout,
                       SaveAllViews=1,ImageResolution=[3840,2160])
            #SaveAllViews=1,ImageResolution=[1280,720])
            # Set the current source to be replaced on next loop
            oldsource = newsource
    '''
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
