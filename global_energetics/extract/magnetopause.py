#!/usr/bin/env python3
"""Extraction routine for magnetopause surface
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, deg2rad, linspace
import matplotlib.pyplot as plt
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
from progress.bar import Bar
#interpackage modules
from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
from global_energetics.extract import swmf_access
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import volume_tools
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (streamfind_bisection,
                                                    get_global_variables,
                                                    dump_to_pandas,
                                                    setup_isosurface,
                                                    calc_iso_rho_state,
                                                    calc_iso_rho_uB_state,
                                                    calc_iso_rho_beta_state,
                                                    calc_iso_beta_state,
                                                 calc_transition_rho_state,
                                                    calc_shue_state,
                                                    calc_sphere_state,
                                                    calc_box_state,
                                                    abs_to_timestamp,
                                                    write_to_timelog)

def get_magnetopause(field_data, datafile, *, outputpath='output/',
                     mode='iso_betastar', include_core=True, source='swmf',
                     longitude_bounds=10, n_fieldlines=5,rmax=30,rmin=3,
                     dx_probe=-1,
                     sp_x=0, sp_y=0, sp_z=0, sp_r=4,
                     box_xmax=-5, box_xmin=-8,
                     box_ymax=5, box_ymin=-5,
                     box_zmax=5, box_zmin=-5,
                     itr_max=100, tol=0.1,
                     tail_cap=-20, tail_analysis_cap=-20,
                     integrate_surface=True, integrate_volume=True,
                     xyzvar=[1,2,3], zone_rename=None):
    """Function that finds, plots and calculates energetics on the
        magnetopause surface.
    Inputs
        General
            field_data- tecplot DataSet object with 3D field data
            datafile- field data filename, assumes .plt file
            outputpath- path for output of .csv of points
            mode- iso_betastar, sphere, box or shue97/shue98
            include_core- setting to omit the inner boundary of sim domain
            source- only capable of handling SWMF input
        Streamtracing
            longitude_bounds, nlines- bounds and density of search
            rmax, rmin, itr, tol- parameters for bisection algorithm
        Isosurface selection
            dx_probe- how far from x_subsolar to probe for iso creation
        Surface
            tail_cap- X position of tail cap
            tail_analysis_cap- X position where integration stops
            integrate_surface/volume- booleans for settings
            xyzvar- for X, Y, Z variables in field data variable list
            zone_rename- optional rename if calling multiple times
    """
    approved = ['iso_betastar', 'shue97', 'shue98', 'shue', 'box', 'sphere']
    if not any([mode == match for match in approved]):
        print('Magnetopause mode "{}" not recognized!!'.format(mode))
        print('Please set mode to one of the following:')
        for choice in approved:
            print('\t{}'.format(choice))
        return
    display = ('Analyzing Magnetopause with the following settings:\n'+
               '\tdatafile: {}\n'.format(datafile)+
               '\toutputpath: {}\n'.format(outputpath)+
               '\tmode: {}\n'.format(mode)+
               '\tinclude_core: {}\n'.format(include_core)+
               '\tsource: {}\n'.format(source))
    if mode == 'sphere':
        #sphere settings
        display = (display +
               '\tSphere settings:\n'+
               '\t\tsp_x: {}\n'.format(sp_x)+
               '\t\tsp_y: {}\n'.format(sp_y)+
               '\t\tsp_z: {}\n'.format(sp_z)+
               '\t\tsp_r: {}\n'.format(sp_r))
    if mode == 'box':
        #box settings
        display = (display +
               '\tBox settings:\n'+
               '\t\txmax: {}\n'.format(box_xmax)+
               '\t\txmin: {}\n'.format(box_xmin)+
               '\t\tymax: {}\n'.format(box_ymax)+
               '\t\tymin: {}\n'.format(box_ymin)+
               '\t\tzmax: {}\n'.format(box_zmax)+
               '\t\tzmin: {}\n'.format(box_zmin))
        #field_data.zone('global_field').aux_data['x_subsolar']=0
    #field line settings
    display = (display +
               '\tlongitude_bounds: {}\n'.format(longitude_bounds)+
               '\tn_fieldlines: {}\n'.format(n_fieldlines)+
               '\trmax: {}\n'.format(rmax)+
               '\trmin: {}\n'.format(rmin)+
               '\titr_max: {}\n'.format(itr_max)+
               '\ttol: {}\n'.format(tol))
    #general surface settings
    display = (display+
               '\ttail_cap: {}\n'.format(tail_cap)+
               '\ttail_analysis_cap: {}\n'.format(tail_analysis_cap)+
               '\tintegrate_surface: {}\n'.format(integrate_surface)+
               '\tintegrate_volume: {}\n'.format(integrate_volume)+
               '\txyzvar: {}\n'.format(xyzvar)+
               '\tzone_rename: {}\n'.format(zone_rename))
    if os.path.exists('banner.txt') & (
                               not os.path.exists(outputpath+'/meshdata')):
        with open('banner.txt') as image:
            print(image.read())
    print('**************************************************************')
    print(display)
    print('**************************************************************')
    #get date and time info based on data source
    if source == 'swmf':
        eventtime = swmf_access.swmf_read_time()
        datestring = (str(eventtime.year)+'-'+str(eventtime.month)+'-'+
                      str(eventtime.day)+'-'+str(eventtime.hour)+'-'+
                      str(eventtime.minute))
    else:
        print("Unknown data source, cant find date/time and won't be able"+
              "to consider dipole orientation!!!")
        datestring = 'Date & Time Unknown'

    with tp.session.suspend():
        main_frame = tp.active_frame()
        aux = field_data.zone('global_field').aux_data
        #set frame name and calculate global variables
        if field_data.variable_names.count('r [R]') ==0:
            print('Calculating global energetic variables')
            main_frame.name = 'main'
            get_global_variables(field_data)
        else:
            main_frame = [fr for fr in tp.frames('main')][0]
        #Get x_subsolar if not already there
        if any([key.find('x_subsolar')!=-1 for key in aux.keys()]):
            x_subsolar = float(aux['x_subsolar'])
        else:
            frontzoneindicies = streamfind_bisection(field_data,'dayside',
                                            longitude_bounds, n_fieldlines,
                                            rmax, rmin, itr_max, tol)
            x_subsolar = 1
            for index in frontzoneindicies:
                x_subsolar = max(x_subsolar,
                                field_data.zone(index).values('X *').max())
            print('x_subsolar found at {}'.format(x_subsolar))
            aux['x_subsolar'] = x_subsolar
            #delete streamzones
            for zone_index in reversed(frontzoneindicies):
                field_data.delete_zones(field_data.zone(zone_index))
        #Create isosurface zone depending on mode setting
        ################################################################
        if mode == 'sphere':
            zonename = mode
            if zone_rename != None:
                zonename = zone_rename
            #calculate surface state variable
            sp_state_index = calc_sphere_state(zonename,sp_x, sp_y, sp_z,
                                               sp_r)
            state_var_name = field_data.variable(sp_state_index).name
            #make iso zone
            sp_zone = setup_isosurface(1, sp_state_index,
                                        7, 7, zonename)
            zoneindex = sp_zone.index
        ################################################################
        if mode == 'box':
            zonename = mode
            if zone_rename != None:
                zonename = zone_rename
            #calculate surface state variable
            box_state_index = calc_box_state(zonename,box_xmax, box_xmin,
                                                  box_ymax, box_ymin,
                                                  box_zmax, box_zmin)
            state_var_name = field_data.variable(box_state_index).name
            #make iso zone
            box_zone = setup_isosurface(1, box_state_index,
                                        7, 7, zonename)
            zoneindex = box_zone.index
        ################################################################
        if mode.find('shue') != -1:
            zonename = 'mp_'+mode
            if zone_rename != None:
                zonename = zone_rename
            #calculate surface state variable
            shue_state_index = calc_shue_state(field_data, zonename,
                                              x_subsolar, tail_cap)
            state_var_name = field_data.variable(shue_state_index).name
            #make iso zone
            shue_zone = setup_isosurface(1, shue_state_index,
                                                  7, 7, zonename)
            zoneindex = shue_zone.index
        ################################################################
        if mode == 'iso_betastar':
            zonename = 'mp_'+mode
            iso_betastar_index = calc_iso_beta_state(x_subsolar, tail_cap,
                                                     50, 1, include_core,4)
            state_var_name = field_data.variable(iso_betastar_index).name
            #remake iso zone using new equation
            iso_betastar_zone = setup_isosurface(1, iso_betastar_index,
                                                  7, 7, zonename)
            zoneindex = iso_betastar_zone.index
            if zone_rename != None:
                iso_betastar_zone.name = zone_rename
                zonename = zone_rename
        ################################################################
        #save mesh to hdf file as key=mode, along with time in key='time'
        mp_mesh, _ = dump_to_pandas(main_frame, [zoneindex], xyzvar,
                                    'temp.csv')
        path_to_mesh = outputpath+'meshdata'
        if not os.path.exists(outputpath+'meshdata'):
            os.system('mkdir '+outputpath+'meshdata')
        meshfile = datestring+'_mesh.h5'
        mp_mesh.to_hdf(path_to_mesh+'/'+meshfile, key=zonename)
        pd.Series(eventtime).to_hdf(path_to_mesh+'/'+meshfile, 'time')

        #perform integration for surface and volume quantities
        mp_powers = pd.DataFrame()
        mp_energies = pd.DataFrame()
        if integrate_surface:
            mp_powers = surface_analysis(main_frame, zonename,
                                    cuttoff=tail_analysis_cap)
            print('\nMagnetopause Power Terms')
            print(mp_powers)
        if integrate_volume:
            print('Zonename is: {}'.format(zonename))
            if zonename == 'sphere':
                doblank = False
            else:
                doblank = True
            mp_energies = volume_analysis(main_frame, state_var_name,
                                          cuttoff=tail_analysis_cap,
                                          blank=False)
            print('\nMagnetopause Energy Terms')
            print(mp_energies)
        if integrate_surface or integrate_surface:
            integralfile = outputpath+'mp_integral_log.h5'
            cols = mp_powers.keys().append(mp_energies.keys())
            mp_energetics = pd.DataFrame(columns=cols, data=[np.append(
                                     mp_powers.values,mp_energies.values)])
            #Add time and x_subsolar column
            mp_energetics.loc[:,'Time [UTC]'] = eventtime
            mp_energetics.loc[:,'X_subsolar [Re]'] = x_subsolar
            with pd.HDFStore(integralfile) as store:
                if any([key == '/'+zonename for key in store.keys()]):
                    mp_energetics = store[zonename].append(mp_energetics,
                                                         ignore_index=True)
                store[zonename] = mp_energetics
    #Display result from this step
    result = ('Result\n'+
               '\tmeshdatafile: {}\n'.format(path_to_mesh+'/'+meshfile))
    if integrate_volume or integrate_surface:
        result = (result+
               '\tintegralfile: {}\n'.format(integralfile)+
               '\tzonename_added: {}\n'.format(zonename))
        with pd.HDFStore(integralfile) as store:
            result = result+'\tmp_energetics:\n'
            for key in store.keys():
                result = (result+
                '\t\tkey={}\n'.format(key)+
                '\t\t\tn_values: {}\n'.format(len(store[key])))
    print('**************************************************************')
    print(result)
    print('**************************************************************')



# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()
    os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()

    #Load .plt file, come back to this later for batching
    SWMF_DATA = tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')

    #Set parameters
    #DaySide
    N_AZIMUTH_DAY = 15
    AZIMUTH_MAX = 122
    R_MAX = 30
    R_MIN = 3.5
    ITR_MAX = 100
    TOL = 0.1

    #Tail
    N_AZIMUTH_TAIL = 15
    RHO_MAX = 50
    RHO_STEP = 0.5
    X_TAIL_CAP = -20

    #YZ slices
    N_SLICE = 40
    N_ALPHA = 50

    #Visualization
    RCOLOR = 4

    get_magnetopause('./3d__mhd_2_e20140219_123000-000.plt')
'''
{magnetopause}=IF({X [R]}<10.7&&{X [R]}>-40&&{h}<50,IF({Rho [amu/cm^3]}<0.16,1,0),0)
{plasma_sphere}=IF({P [nPa]}<0.02&&{uB [J/Re^3]}>1e11,1,0)
{lobes}=IF({X [R]}>-40&&abs({lon [deg]})>120&&{h}<28,IF({P [nPa]}<0.005&&{uB [J/Re^3]}<1e11&&{Rho [amu/cm^3]}<0.16,1,0),0)
{plasma_sheet}=IF({X [R]}>-40&&abs({lon [deg]})>120&&{h}<28,IF({P [nPa]}>0.005&&{uB [J/Re^3]}<1e11&&{Rho [amu/cm^3]}<0.08,1,0),0)
{cloak}=IF({P [nPa]}>0.03&&{uB [J/Re^3]}>1e11&&{r [R]}<9.5,1,0)
'''
