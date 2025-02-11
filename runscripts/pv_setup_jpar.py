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
from paraview.vtk.numpy_interface import dataset_adapter as dsa
#import global_energetics.extract.pv_magnetopause
import pv_magnetopause
from makevideo import (get_time, time_sort)
from pv_tools import (update_rotation,slice_and_calc_applied_voltage)
from pv_input_tools import (read_aux, read_tecplot,find_IE_matched_file)
import pv_surface_tools
from pv_magnetopause import (setup_pipeline)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)
import pv_ionosphere
from pv_visuals import (display_visuals)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'gannon-storm/data/large/')
    GMpath = os.path.join(inpath,'GM/')
    IEpath = os.path.join(inpath,'IE/')
    IMpath = os.path.join(inpath,'IM/')
    outpath= os.path.join(herepath,'gannon-storm/outputs/vis/')

    filelist = sorted(glob.glob(GMpath+'*paraview*.plt'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    for i,infile in enumerate(filelist[-1::]):
        # Process GM file
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
                                                       aux=aux,
                                                       doEnergyFlux=True)
        # Calculate Solar Wind Electric Field
        Esw = Calculator(registrationName='Esw',Input=field)
        Esw.Function = 'cross(B_nT,U_km_s)/1000'# E [mV/m]
        Esw.ResultArrayName = 'E_mV_m'
        # Create two hemispheres at r=rCurrents
        rC = Contour(registrationName='rCurrents',Input=Esw)
        rC.ContourBy = ['POINTS','r_R']
        rC.ComputeNormals = 1
        rC.Isosurfaces = 2.375 #NOTE rCurrents from PARAM.in

        rC_north = Clip(registrationName='rCurrents_north',Input=rC)
        rC_north.ClipType = 'Scalar'
        rC_north.Scalars  = 'Zd_R'
        rC_north.Value    = 0
        rC_north.Invert   = 0

        rC_south = Clip(registrationName='rCurrents_south',Input=rC)
        rC_south.ClipType = 'Scalar'
        rC_south.Scalars  = 'Zd_R'
        rC_south.Value    = 0
        rC_south.Invert   = 1
        # Create 2 Contours on Each Hemisphere for Max/Min FAC
        #   First find the Max/Min threshold
        north_data = servermanager.Fetch(rC_north)
        north_data = dsa.WrapDataObject(north_data)
        Jpar_north = north_data.PointData['J_par_uA_m2']
        Jpar_north_down = np.quantile(Jpar_north,0.991)#NOTE J// rel to B!!
        Jpar_north_up   = np.quantile(Jpar_north,0.001)#NOTE J// rel to B!!

        south_data = servermanager.Fetch(rC_south)
        south_data = dsa.WrapDataObject(south_data)
        Jpar_south = south_data.PointData['J_par_uA_m2']
        Jpar_south_up   = np.quantile(Jpar_south,0.991)
        Jpar_south_down = np.quantile(Jpar_south,0.001)
        #   Then set the contour(s)
        # North-UP
        contour_north_up = Contour(registrationName='contourN_up',
                                   Input=rC_north)
        contour_north_up.ContourBy = ['POINTS','J_par_uA_m2']
        contour_north_up.Isosurfaces = Jpar_north_up

        # North-DOWN
        contour_north_down = Contour(registrationName='contourN_down',
                                     Input=rC_north)
        contour_north_down.ContourBy = ['POINTS','J_par_uA_m2']
        contour_north_down.Isosurfaces = Jpar_north_down

        # South-UP
        contour_south_up = Contour(registrationName='contourS_up',
                                   Input=rC_south)
        contour_south_up.ContourBy = ['POINTS','J_par_uA_m2']
        contour_south_up.Isosurfaces = Jpar_south_up

        # South-DOWN
        contour_south_down = Contour(registrationName='contourS_down',
                                     Input=rC_south)
        contour_south_down.ContourBy = ['POINTS','J_par_uA_m2']
        contour_south_down.Isosurfaces = Jpar_south_down
        #   Then use stream trace w/ custom source for B
        Bslice_points = {}
        Bstream_tags = ['_N_up','_S_up','_N_down','_S_down']
        for i,seed in enumerate([contour_north_up,contour_south_up,
                                 contour_north_down,contour_south_down]):
            Bstream = StreamTracerWithCustomSource(Input=Esw,SeedSource=seed,
                                   registrationName='Bstream'+Bstream_tags[i])
            Bstream.Vectors = ['POINTS','B_nT']
            Bstream.MaximumStreamlineLength = 120
        Jpar_results = slice_and_calc_applied_voltage(['Bstream'+n
                                                   for n in Bstream_tags],Esw)
        print(Jpar_results)
        ####################################################################
        # Now compare with vtkVectorFieldTopology traced field lines
        ####################################################################
        # Split off B on it's own
        Bsolo = ProgrammableFilter(registrationName='Bsolo',
                                   Input=FindSource('B'))
        Bsolo.Script = f"""
        B = inputs[0].PointData['B_nT']
        x = inputs[0].PointData['x']
        y = inputs[0].PointData['y']
        z = inputs[0].PointData['z']
        output.PointData.append(B,'B_nT')
        output.PointData.append(x,'x')
        output.PointData.append(y,'y')
        output.PointData.append(z,'z')
        """
        # Call the vtkVectorTopology tools to get the critical points
        nulls = ProgrammableFilter(registrationName='nulls',Input=Bsolo)
        nulls.OutputDataSetType = 'vtkPolyData'
        nulls.Script = f"""
        import vtk
        input = self.GetInputDataObject(0,0)
        output = self.GetOutputDataObject(0)
        topology = vtk.vtkVectorFieldTopology()
        topology.SetInputDataObject(input)
        topology.Update()
        output.DeepCopy(topology.GetOutput(0))
        """
        # Use a Calculator filter to reveal the coordinates of the points
        nulls_XYZ = Calculator(registrationName='XYZ',Input=nulls)
        nulls_XYZ.Function = 'coordsX*iHat+coordsY*jHat+coordsZ*kHat'
        nulls_XYZ.ResultArrayName = 'XYZ'
        # Fetch the results and find the dayside endpoints
        null_data = servermanager.Fetch(nulls_XYZ)
        null_data = dsa.WrapDataObject(null_data)
        null_points = null_data.PointData['XYZ']
        day_points = null_points[null_points[:,0]>-10] # where X> -10Re
        P_left = null_points[np.argmin(null_points[:,1])]# min Y
        P_right = null_points[np.argmax(null_points[:,1])]# max Y
        Bstream_tags = ['_N_up','_S_up','_N_down','_S_down']
        for i,corner in enumerate(['Null'+n for n in Bstream_tags]):
            # Generate streamlines near points
            stream = StreamTracer(registrationName=corner,Input=Esw)
            stream.Vectors = 'B_nT'
            if '_N' in corner:#NOTE w/ S IMF should go +Z
                stream.IntegrationDirection ='BACKWARD'
            else:
                stream.IntegrationDirection ='FORWARD'
            stream.SeedType = 'Point Cloud'
            if '_down' in corner:
                stream.SeedType.Center = P_left
            else:
                stream.SeedType.Center = P_right
            stream.SeedType.Radius = 0.1
            stream.SeedType.NumberOfPoints = 8
        Null_results = slice_and_calc_applied_voltage(['Null'+n
                                                   for n in Bstream_tags],Esw)
        print(Null_results)

        # Find corresponding IE file
        iefile = find_IE_matched_file(IEpath,localtime)

        if os.path.exists(iefile):
            pv_ionosphere.load_ie(iefile,coord='GSM',
                                  tilt=float(aux['BTHETATILT']))
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
