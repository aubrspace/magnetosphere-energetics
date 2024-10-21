#/usr/bin/env python
import time
import glob
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
from dateutil import parser
import numpy as np
from scipy import interpolate
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
import pv_mapping

if True:
    start_time = time.time()
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'paleo/')
    outpath = os.path.join(herepath,'paleo/output/final/')
    filelist = glob.glob(inpath+'*paraview*.plt')
    print(inpath)
    print(inpath+'*paraview*.plt')
    print(filelist)
    #renderView = GetActiveViewOrCreate('RenderView')
    t0 = dt.datetime(1970,1,1)
    tevent = dt.datetime(2010,3,20,3)
    ut = (tevent-t0).total_seconds()
    gp.recalc(ut)
    # load state
    LoadState('/home/aubr/Code/swmf-energetics/paleo/output/'+
              'final_state_jpar_earth_projections.pvsm')
    for i,infile in enumerate(filelist):
        print(f"\t{i+1}/{len(filelist)}\t{infile.split('/')[-1]}")
        # Locate the start of the pipeline and the old data
        pipehead = FindSource('MergeBlocks1')
        oldData = pipehead.Input
        # Read in new data and feed into the pipe, delete old data
        newData = read_tecplot(infile)
        pipehead.Input = newData
        Delete(oldData)
        del oldData
        # Save screenshot
        outfile = ('-'.join(infile.split('_')[-2::]).replace('plt','png'))
        layouts = GetLayouts()
        for i,layout in enumerate(layouts.values()):
            SaveScreenshot(outpath+outfile.replace('.png',f'_{i}.png'),layout)
        # Save all the views to separate files
        '''
        # Setup file specific quantities
        aux = read_aux(infile.replace('.plt','.aux'))
        outfile = ('-'.join(infile.split('_')[-2::]).replace('plt','png'))
        outfile2 = outfile.replace('.png','_2.png')
        # Initialize the geopack routines by finding the universal time
        t0 = dt.datetime(1970,1,1)
        #tevent = parser.parse(aux['TIMEEVENT'])
        tevent = dt.datetime(2013,4,21,12)
        pv_tools.create_globe(tevent)
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
                                                       tail_x=-60,
                                                       aux=aux,
                                                       doEnergyFlux=False)
        # Calculate Jpar
        field = Calculator(registrationName='Jpar',Input=field)
        field.Function = "dot(J_uA_m2,B_nT)/mag(B_nT)"
        field.ResultArrayName = 'Jpar'

        r1 = 1+110/6371
        r2 = 2.35
        # Create inner boundary sphere
        sphere = Sphere(registrationName='Sphere')
        sphere.Radius = r1
        sphere.ThetaResolution = 300
        sphere.PhiResolution = 300
        # Project along dipole
        stretch = pv_mapping.bfield_project(sphere,r1,r2)
        # Interpolate from the field data
        interpolate = PointDatasetInterpolator(registrationName='interp',
                                               Input=field,Source=stretch)
        # Reset to the original position
        squish = pv_mapping.bfield_project(interpolate,r2,r1)
        # Visuals
        jparDisplay=Show(squish,renderView,'GeometryRepresentation')
        jparDisplay.Opacity = 0.8
        ColorBy(jparDisplay, ('POINTS', 'Jpar'))
        jparLUT = GetColorTransferFunction('Jpar')
        jparPWF = GetOpacityTransferFunction('Jpar')
        jparLUT.ApplyPreset('Cool to Warm (Extended)',True)
        jparLUT.RescaleTransferFunction(-0.05,0.05)

        # Create quick contour at r=r2
        contour = Contour(registrationName='innerBound',Input=field)
        contour.ContourBy = ['POINTS','r_R']
        contour.ComputeNormals = 0
        contour.Isosurfaces = [r2]
        # Seed field lines at Jmax
        npoints = 20
        thresh = 100
        jmax_table = ProgrammableFilter(registrationName='jmaxTab',
                                        Input=contour)
        jmax_table.OutputDataSetType = 'vtkTable'
        jmax_table.Script = f"""
        import numpy as np
        data = inputs[0]
        jpar = data.PointData['Jpar']
        x = data.PointData['x']
        y = data.PointData['y']
        z = data.PointData['z']

        jpar_top = jpar[np.where(z>0)]
        x_top = x[np.where(z>0)]
        y_top = y[np.where(z>0)]
        z_top = z[np.where(z>0)]

        jpar_bot = jpar[np.where(z<0)]
        x_bot = x[np.where(z<0)]
        y_bot = y[np.where(z<0)]
        z_bot = z[np.where(z<0)]

        max_jpar_x = np.array([])
        max_jpar_y = np.array([])
        max_jpar_z = np.array([])

        sorted_index_top = jpar_top.argsort()
        # Max Top
        max_jpar_x = np.append(max_jpar_x,x_top[sorted_index_top[
                                            -{thresh}::{int(thresh/npoints)}]])
        max_jpar_y = np.append(max_jpar_y,y_top[sorted_index_top[
                                            -{thresh}::{int(thresh/npoints)}]])
        max_jpar_z = np.append(max_jpar_z,z_top[sorted_index_top[
                                            -{thresh}::{int(thresh/npoints)}]])

        # Min Top
        max_jpar_x = np.append(max_jpar_x,x_top[sorted_index_top[
                                            0:{npoints}:{int(thresh/npoints)}]])
        max_jpar_y = np.append(max_jpar_y,y_top[sorted_index_top[
                                            0:{npoints}:{int(thresh/npoints)}]])
        max_jpar_z = np.append(max_jpar_z,z_top[sorted_index_top[
                                            0:{npoints}:{int(thresh/npoints)}]])

        sorted_index_bot = jpar_bot.argsort()
        # Max Bot
        max_jpar_x = np.append(max_jpar_x,x_bot[sorted_index_bot[
                                            -{thresh}::{int(thresh/npoints)}]])
        max_jpar_y = np.append(max_jpar_y,y_bot[sorted_index_bot[
                                            -{thresh}::{int(thresh/npoints)}]])
        max_jpar_z = np.append(max_jpar_z,z_bot[sorted_index_bot[
                                            -{thresh}::{int(thresh/npoints)}]])
        # Min Bot
        max_jpar_x = np.append(max_jpar_x,x_bot[sorted_index_bot[
                                            0:{npoints}:{int(thresh/npoints)}]])
        max_jpar_y = np.append(max_jpar_y,y_bot[sorted_index_bot[
                                            0:{npoints}:{int(thresh/npoints)}]])
        max_jpar_z = np.append(max_jpar_z,z_bot[sorted_index_bot[
                                            0:{npoints}:{int(thresh/npoints)}]])

        # standard code to display the result
        to = self.GetTableOutput()
        arrx = vtk.vtkFloatArray()
        arrx.SetName("x")
        arrx.SetNumberOfComponents(1)
        arry = vtk.vtkFloatArray()
        arry.SetName("y")
        arry.SetNumberOfComponents(1)
        arrz = vtk.vtkFloatArray()
        arrz.SetName("z")
        arrz.SetNumberOfComponents(1)
        for i,(xi,yi,zi) in enumerate(zip(max_jpar_x,max_jpar_y,max_jpar_z)):
            arrx.InsertNextValue(xi)
            arry.InsertNextValue(yi)
            arrz.InsertNextValue(zi)
        to.AddColumn(arrx)
        to.AddColumn(arry)
        to.AddColumn(arrz)
        """
        spreadSheetView1 = CreateView('SpreadSheetView')
        jmaxTabDisplay = Show(jmax_table,spreadSheetView1,
                              'SpreadSheetRepresentation')
        # Convert table to special vtk data
        jmax = TableToPoints(registrationName='jmax',Input=jmax_table)
        jmax.XColumn = 'x'
        jmax.YColumn = 'y'
        jmax.ZColumn = 'z'
        # Trace
        maxTrace = StreamTracerWithCustomSource(registrationName='jmaxTrace',
                                                Input=field, SeedSource=jmax)
        """
        layout = GetLayout()
        layout.SetSize(1280, 1280)# Single hyperwall screen
        display_visuals(field,mp,renderView,doSlice=False,doFluxVol=False,
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
        renderView.CameraPosition = [48.26, 38.63, 22.51]
        renderView.CameraFocalPoint = [34.17, 29.73, 17.72]
        renderView.CameraViewUp = [-0.19, -0.21, 0.95]
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
        '''
        '''
        ## Second image
        Hide(mp)
        Show(fieldlines)
        # Overwrite camera positions
        renderView.CameraPosition = [48.26, 38.63, 22.51]
        renderView.CameraFocalPoint = [34.17, 29.73, 17.72]
        renderView.CameraViewUp = [-0.19, -0.21, 0.95]
        # Save screenshot
        SaveScreenshot(outpath+outfile2,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
        '''
        """

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
        renderView.Update()
        # Save screenshot
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[1280,1280])
        print('\033[92m Created\033[00m',os.path.relpath(outpath+outfile,
                                                         os.getcwd()))
        """
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
