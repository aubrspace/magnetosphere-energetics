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
import pandas as pd
#### import the simple module from paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
#import global_energetics.extract.pv_magnetopause
import pv_magnetopause
from makevideo import (get_time, time_sort)
from pv_tools import (update_rotation)
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

    all_results = pd.DataFrame()

    # Load master state
    LoadState(outpath+'cpcp_vis_state2.pvsm')
    for i,infile in enumerate(filelist):
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        timestamp = FindSource('time')
        timestamp.Text = str(localtime)

        print(f"\t{i+1}/{len(filelist)}\t{infile.split('/')[-1]}")
        # Locate the start of the pipeline and the old data
        pipehead = FindSource('MergeBlocks1')
        oldData = pipehead.Input
        # Read in new data and feed into the pipe, delete old data
        newData = read_tecplot(infile)
        pipehead.Input = newData
        Delete(oldData)
        del oldData

        # Find corresponding IE file
        iefile = find_IE_matched_file(IEpath,localtime)
        iehead = FindSource('MergeBlocks')
        oldIE = iehead.Input
        # Read in new data and feed into the pipe, delete old data
        newIE = read_tecplot(iefile,binary=False)
        iehead.Input = newIE
        Delete(oldIE)
        del oldIE

        # Update dipole
        tilt = float(aux['BTHETATILT'])
        mXhat_x = FindSource('mXhat_x')
        mXhat_x.Function = f"sin(({tilt}+90)*3.14159/180)"

        mXhat_z = FindSource('mXhat_z')
        mXhat_z.Function = f"-1*cos(({tilt}+90)*3.14159/180)"

        mZhat_x = FindSource('mXhat_x')
        mZhat_x.Function = f"sin(({tilt})*3.14159/180)"

        mZhat_z = FindSource('mZhat_z')
        mZhat_z.Function = f"-1*cos(({tilt})*3.14159/180)"

        IErotate = FindSource('rotate2GSM')
        IErotate.Script = update_rotation(tilt)

        # Find new up/down FAC contour levels
        north_data = servermanager.Fetch(FindSource('rCurrents_north'))
        north_data = dsa.WrapDataObject(north_data)
        Jpar_north = north_data.PointData['J_par_uA_m2']
        Jpar_north_down = np.quantile(Jpar_north,0.990)#NOTE J// rel to B!!
        Jpar_north_up   = np.quantile(Jpar_north,0.010)#NOTE J// rel to B!!

        south_data = servermanager.Fetch(FindSource('rCurrents_south'))
        south_data = dsa.WrapDataObject(south_data)
        Jpar_south = south_data.PointData['J_par_uA_m2']
        Jpar_south_up     = np.quantile(Jpar_south,0.990)
        Jpar_south_down   = np.quantile(Jpar_south,0.010)

        contour_north_up = FindSource('contourN_up')
        contour_north_up.Isosurfaces = Jpar_north_up
        contour_north_down = FindSource('contourN_down')
        contour_north_down.Isosurfaces = Jpar_north_down

        contour_south_up = FindSource('contourS_up')
        contour_south_up.Isosurfaces = Jpar_south_up
        contour_south_down = FindSource('contourS_down')
        contour_south_down.Isosurfaces = Jpar_south_down

        # Update null points
        null_XYZ = FindSource('XYZ')
        null_XYZ.UpdatePipeline()
        # Fetch the results and find the dayside endpoints
        null_data = servermanager.Fetch(null_XYZ)
        null_data = dsa.WrapDataObject(null_data)
        null_points = null_data.PointData['XYZ']
        day_points = null_points[null_points[:,0]>-30] # where X> -10Re
        P_left = day_points[np.argmin(day_points[:,1])]# min Y
        P_right = day_points[np.argmax(day_points[:,1])]# max Y
        for source in ['Null_N_up','Null_N_down','Null_S_up','Null_S_down']:
            stream = FindSource(source)
            if '_down' in source:
                stream.SeedType.Center = P_left
            else:
                stream.SeedType.Center = P_right

        # Find new sliced points
        Bslice_points = {}
        for slice_source in ['Bstream_Slice_N_up','Bstream_Slice_N_down',
                             'Bstream_Slice_S_up','Bstream_Slice_S_down',
                             'Null_Slice_N_up','Null_Slice_N_down',
                             'Null_Slice_S_up','Null_Slice_S_down']:
            head = slice_source.split('_')[0]
            tag = slice_source.split('Slice')[-1]
            # Update a dict with the info
            slice = FindSource(slice_source)
            slice.SliceType.Radius = 40
            slice_points = servermanager.Fetch(FindSource(slice_source))
            slice_points = dsa.WrapDataObject(slice_points)
            P_x = np.mean(slice_points.PointData['x'])
            P_y = np.mean(slice_points.PointData['y'])
            P_z = np.mean(slice_points.PointData['z'])
            Bslice_points['P_'+head+tag] = (P_x,P_y,P_z)
        # Find new lengths and voltage drops across points
        results = {}
        for length in ['Bstream_N','Bstream_S',
                       'Null_N','Null_S']:
            P = Bslice_points
            up_x,up_y,up_z = Bslice_points['P_'+length+'_up']
            dn_x,dn_y,dn_z = Bslice_points['P_'+length+'_down']
            L_AB = np.sqrt((up_x-dn_x)**2+(up_y-dn_y)**2+(up_z-dn_z)**2)
            AB_x = (up_x-dn_x)/L_AB
            AB_y = (up_y-dn_y)/L_AB
            AB_z = (up_z-dn_z)/L_AB
            results['Length_'+length] = L_AB
            # Update Electric field calculator
            E_AB = FindSource('E_AB_'+length)
            E_AB.Function = (
                         f"dot(E_mV_m,({AB_x}*iHat+{AB_y}*jHat+{AB_z}*kHat))")
            # Update the line integrator
            AB_line = FindSource('Line_'+length)
            AB_line.Source.Point1 = Bslice_points['P_'+length+'_up']
            AB_line.Source.Point2 = Bslice_points['P_'+length+'_down']
            # Collect result
            result = FindSource('LineIntegral_'+length)
            result_data = servermanager.Fetch(result)
            result_data = dsa.WrapDataObject(result_data)
            dV = result_data.PointData['E_AB_mV_m']*6.371 # NOTE mV/m * Re
            results['dV_'+length] = dV

        results['time'] = localtime
        all_results = pd.concat([all_results,pd.DataFrame(data=results)])
        # Save screenshot(s)
        outfile = ('-'.join(infile.split('_')[-2::]).replace('plt','png'))
        layouts = GetLayouts()
        for i,layout in enumerate(layouts.values()):
            SaveScreenshot(outpath+outfile.replace('.png',f'_{i}.png'),layout)
    all_results.index = all_results['time']
    all_results.drop(columns=['time'],inplace=True)
    all_results.to_csv(outpath+'calc_potentials.csv')
    print(all_results)
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
