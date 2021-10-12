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
                                                    calc_lobe_state,
                                                    calc_rc_state,
                                                    calc_ps_qDp_state,
                                                    calc_iso_rho_state,
                                                    calc_betastar_state,
                                                    calc_closed_state,
                                                 calc_transition_rho_state,
                                                    calc_shue_state,
                                                    calc_delta_state,
                                                    calc_sphere_state,
                                                    calc_box_state,
                                                    abs_to_timestamp,
                                                    get_1D_sw_variables,
                                             get_surfaceshear_variables,
                                                    write_to_timelog)
from global_energetics.write_disp import (write_mesh, write_to_hdf,
                                          display_progress)

def get_magnetopause(field_data, datafile, *, outputpath='output/',
                     mode='iso_betastar', include_core=False, source='swmf',
                     do_1Dsw=False, oneDmx=30, oneDmn=-30, n_oneD=121,
                     do_trace=False, lon_bounds=10, n_fieldlines=5,
                     rmax=30, rmin=3,
                     dx_probe=-1,
                     sp_x=0, sp_y=0, sp_z=0, sp_rmax=3, sp_rmin=0,
                     box_xmax=-5, box_xmin=-8,
                     box_ymax=5, box_ymin=-5,
                     box_zmax=5, box_zmin=-5,
                     lshelllim=7, bxmax=10,
                     itr_max=100, tol=0.1,
                     tail_cap=-20, tail_analysis_cap=-20,
                     integrate_surface=True, integrate_volume=True,
                     save_mesh=True, do_virial=True,
                     do_cms=True, cms_dataset_key='future',
                     do_blank=False, blank_variable='W *',
                     blank_value=50,
                     xyzvar=[0,1,2], zone_rename=None,
                     write_data=True, disp_result=True):
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
        1DSolarWind
            do_1Dsw- boolean for projecting a 1D "pristine" sw solution
            oneDmx, oneDmn, n_oneD- parameters for 1D curve (probe points)
        Streamtracing
            do_trace- boolean turns on fieldline tracing to find x_subsolar
            lon_bounds, nlines- bounds and density of search
            rmax, rmin, itr, tol- parameters for bisection algorithm
        Isosurface selection
            dx_probe- how far from x_subsolar to probe for iso creation
        Surface
            tail_cap- X position of tail cap
            tail_analysis_cap- X position where integration stops
            integrate_surface/volume/docms- booleans for settings
            savemesh- booleans for saving mesh info
            xyzvar- for X, Y, Z variables in field data variable list
            zone_rename- optional rename if calling multiple times
    """
    approved= ['iso_betastar', 'shue97', 'shue98', 'shue', 'box', 'sphere',
               'lcb', 'nlobe', 'slobe', 'rc', 'ps', 'qDp']
    if not any([mode == match for match in approved]):
        print('Magnetopause mode "{}" not recognized!!'.format(mode))
        print('Please set mode to one of the following:')
        for choice in approved:
            print('\t{}'.format(choice))
        return
    if field_data.variable_names.count('Status') ==0:
        has_status = False
        do_trace = True
        print('Status variable not included in dataset!'+
                'do_trace -> True')
    else:
        has_status = True
    if do_trace and mode == 'lcb':
        print('last_closed boundary not setup with trace mode!')
        return
    if do_cms:
        if len([zn for zn in tp.active_frame().dataset.zones()])<2:
            print('not enough data to do moving surfaces!')
            do_cms = False
    display = ('Analyzing Magnetopause with the following settings:\n'+
               '\tdatafile: {}\n'.format(datafile)+
               '\toutputpath: {}\n'.format(outputpath)+
               '\tmode: {}\n'.format(mode)+
               '\tinclude_core: {}\n'.format(include_core)+
               '\tsource: {}\n'.format(source)+
               '\tdo_trace: {}\n'.format(do_trace))
    if do_1Dsw:
        #1D solar wind settings
        display = (display +
               '\t1Dsw settings:\n'+
               '\t\toneDmx: {}\n'.format(oneDmx)+
               '\t\toneDmn: {}\n'.format(oneDmn)+
               '\t\tn_oneD: {}\n'.format(n_oneD))
    if mode == 'sphere':
        #sphere settings
        display = (display +
               '\tSphere settings:\n'+
               '\t\tsp_x: {}\n'.format(sp_x)+
               '\t\tsp_y: {}\n'.format(sp_y)+
               '\t\tsp_z: {}\n'.format(sp_z)+
               '\t\tsp_rmax: {}\n'.format(sp_rmax))
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
    if do_trace == True:
        #field line settings
        display = (display +
               '\tlon_bounds: {}\n'.format(lon_bounds)+
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
               '\tdo_blank: {}\n'.format(do_blank)+
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
        if do_cms:
            futuretime = swmf_access.swmf_read_time(zoneindex=1)
            deltatime = (futuretime-eventtime).seconds
            futurezonename = tp.active_frame().dataset.zone(1).name
        else:
            deltatime=0
    else:
        print("Unknown data source, cant find date/time and won't be able"+
              "to consider dipole orientation!!!")
        datestring = 'Date & Time Unknown'

    main_frame = tp.active_frame()
    aux = field_data.zone('global_field').aux_data
    if do_cms:
        future_aux = field_data.zone(futurezonename).aux_data
    #set frame name and calculate global variables
    if field_data.variable_names.count('r [R]') ==0:
        print('Calculating global energetic variables')
        main_frame.name = 'main'
        get_global_variables(field_data)
        if do_1Dsw:
            print('Calculating 1D "pristine" Solar Wind variables')
            get_1D_sw_variables(field_data, 30, -30, 121)
    else:
        main_frame = [fr for fr in tp.frames('main')][0]
    #Add imfclock angle if not there already
    if not any([key.find('imf_clock_deg')!=-1 for key in aux.keys()]):
        rho,ux,uy,uz,IMFbx,IMFby,IMFbz= tp.data.query.probe_at_position(
                                                       31.5, 0, 0)[0][3:10]
        clockangle = np.rad2deg(np.arctan2(IMFby, IMFbz))
        IMFBmag = sqrt(IMFbx**2+IMFby**2+IMFbz**2)
        pdyn = rho*(ux**2+uy**2+uz**2)*1.6726e-27*1e6*(1e3)**2*1e9
        aux['imf_clock_deg'] = clockangle
        aux['imf_mag'] = IMFBmag
        aux['sw_pdyn'] = pdyn
    #Get x_subsolar if not already there
    if any([key.find('x_subsolar')!=-1 for key in aux.keys()]):
        x_subsolar = float(aux['x_subsolar'])
        if do_cms:
            future_x_subsolar = float(future_aux['x_subsolar'])
        closed_index = None
        closed_zone = None
        #Assign closed zone info if already exists
        if len([zn for zn in field_data.zones('*lcb*')]) > 0:
            closed_index = field_data.variable('lcb').index
            closed_zone = field_data.zone('*lcb*')
        elif has_status == True:
            closed_index = calc_closed_state('Status', 3, tail_cap)
            closed_zone, _ = setup_isosurface(1,closed_index,'lcb')
        #Assign magnetopause variable if exists
        mpvar = field_data.variable('mp*')
    else:
        if do_trace:
            closedzone_index = streamfind_bisection(field_data,
                                                        'dayside',
                                                lon_bounds, n_fieldlines,
                                                rmax, rmin, itr_max, tol)
            if do_cms:
                future_closedzone_index = streamfind_bisection(field_data,
                                                        'dayside',
                                                lon_bounds, n_fieldlines,
                                                rmax, rmin, itr_max, tol,
                                                global_key=futurezonename)
        else:
            closed_index = calc_closed_state('lcb','Status', 3, tail_cap,0)
            closed_zone, _ = setup_isosurface(1,closed_index,'lcb')
            closedzone_index = closed_zone.index
            if do_cms:
                future_closed_index = calc_closed_state('future_lcb',
                                                        'Status', 3,
                                                        tail_cap, 1)
                future_closed_zone, _ = setup_isosurface(1,
                                                       future_closed_index,
                                                       'future_lcb')
                future_closedzone_index = future_closed_zone.index
        x_subsolar = 1
        x_subsolar = max(x_subsolar,
                field_data.zone(closedzone_index).values('X *').max())
        print('x_subsolar found at {}'.format(x_subsolar))
        aux['x_subsolar'] = x_subsolar
        if do_cms:
            future_x_subsolar = 1
            future_x_subsolar = max(future_x_subsolar,
              field_data.zone(future_closedzone_index).values('X *').max())
            future_aux['x_subsolar'] = future_x_subsolar
        if do_trace:
            #delete streamzone
            field_data.delete_zones(closedzone_index)
            closed_index = None
            closed_zone = None
    #Create isosurface zone depending on mode setting
    ################################################################
    if mode == 'sphere':
        zonename = mode
        if zone_rename != None:
            zonename = zone_rename
        #calculate surface state variable
        sp_state_index = calc_sphere_state(zonename,sp_x, sp_y, sp_z,
                                            sp_rmax, rmin=3)
        state_var_name = field_data.variable(sp_state_index).name
        #make iso zone
        sp_zone, _ = setup_isosurface(1, sp_state_index, zonename)
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
        box_zone, _ = setup_isosurface(1, box_state_index, zonename)
        zoneindex = box_zone.index
        if do_cms:
            future_index = calc_box_state('future_'+zonename,
                                                5, -12,
                                                7, -3,
                                                10, -10)
            future_state_var_name = field_data.variable(future_index).name
            __, _ = setup_isosurface(1,future_index,
                                     'future_'+zonename)
    ################################################################
    if mode.find('shue') != -1:
        zonename = 'mp_'+mode
        #calculate surface state variable
        shue_state_index = calc_shue_state(field_data, mode,
                                            x_subsolar, tail_cap)
        state_var_name = field_data.variable(shue_state_index).name
        #make iso zone
        shue_zone, _ = setup_isosurface(1, shue_state_index, zonename)
        zoneindex = shue_zone.index
        if zone_rename != None:
            shue_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode == 'iso_betastar':
        zonename = 'mp_'+mode
        if zone_rename != None:
            zonename = zone_rename
        iso_betastar_index = calc_betastar_state(zonename, 0, x_subsolar,
                                                    tail_cap, 50, 0.7,
                                                    include_core, sp_rmax,
                                                    closed_zone)
        state_var_name = field_data.variable(iso_betastar_index).name
        if do_cms:
            future_index = calc_betastar_state('future_'+zonename, 1,
                                    future_x_subsolar,
                                    tail_cap, 50, 0.7, include_core,
                                    sp_rmax, future_closed_zone)
            future_state_var_name = field_data.variable(future_index).name
        #remake iso zone using new equation
        if not include_core:
            cond = 'sphere'
        else:
            cond = None
        iso_betastar_zone, innerbound_zone = setup_isosurface(1,
                                                iso_betastar_index,
                                                zonename,
                                                keep_condition=cond,
                                                keep_cond_value=sp_rmax)
        zoneindex = iso_betastar_zone.index
        if do_cms:
            __, _ = setup_isosurface(1,future_index,
                                     'future_'+zonename,
                                     keep_condition=cond,
                                     keep_cond_value=sp_rmax)
            #get state variable representing acquisitions/forfeitures
            delta_magnetopause_index = calc_delta_state(state_var_name,
                                                     future_state_var_name)
        #get_surfaceshear_variables(field_data, 'beta_star', 0.7, 2.8)
        if zone_rename != None:
            iso_betastar_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode.find('lcb') != -1:
        assert do_trace == False, (
                            "lcb mode only works with do_trace==False!")
        assert closed_index != None, (
                                    'No closed_zone present! Cant do lcb')
        state_var_name = 'lcb'
        zonename = closed_zone.name
        zoneindex = closed_zone.index
        if zone_rename != None:
            closed_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode.find('lobe') != -1:
        assert do_trace == False, (
                            "lobe mode only works with do_trace==False!")
        assert type(mpvar) != type(None),('magnetopause variable not found'+
                                          ' cannot calculate lobe zone!')
        zonename = 'ms_'+mode
        #calculate surface state variable
        northsouth = 'north'
        if mode.lower().find('s') != -1:
            northsouth = 'south'
        lobe_index = calc_lobe_state(mpvar.name, northsouth)
        state_var_name = field_data.variable(lobe_index).name
        #make iso zone
        lobe_zone, _ = setup_isosurface(1, lobe_index, zonename)
        zoneindex = lobe_zone.index
        if zone_rename != None:
            lobe_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode.find('rc') != -1:
        assert closed_index != None, ('No closed_zone present! Cant do rc')
        zonename = 'ms_'+mode
        #calculate surface state variable
        rc_index = calc_rc_state(closed_zone.name, str(lshelllim))
        state_var_name = field_data.variable(rc_index).name
        #make iso zone
        rc_zone, _ = setup_isosurface(1, rc_index, zonename)
        zoneindex = rc_zone.index
        if zone_rename != None:
            rc_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode.find('ps') != -1:
        assert closed_index != None, ('No closed_zone present! Cant do ps')
        zonename = 'ms_'+mode
        #calculate surface state variable
        ps_index = calc_ps_qDp_state('ps', closed_zone.name,
                                     str(lshelllim), str(bxmax))
        state_var_name = field_data.variable(ps_index).name
        #make iso zone
        ps_zone, _ = setup_isosurface(1, ps_index, zonename)
        zoneindex = ps_zone.index
        if zone_rename != None:
            ps_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    if mode.find('qDp') != -1:
        assert closed_index != None, ('No closed_zone present! Cant do qDp')
        zonename = 'ms_'+mode
        #calculate surface state variable
        qDp_index = calc_ps_qDp_state('qDp', closed_zone.name,
                                      str(lshelllim), str(bxmax))
        state_var_name = field_data.variable(qDp_index).name
        #make iso zone
        qDp_zone, _ = setup_isosurface(1, qDp_index, zonename)
        zoneindex = qDp_zone.index
        if zone_rename != None:
            qDp_zone.name = zone_rename
            zonename = zone_rename
    ################################################################
    #perform integration for surface and volume quantities
    mp_powers, innerbound_powers = pd.DataFrame(), pd.DataFrame()
    mp_energies = pd.DataFrame()
    mp_mesh = pd.DataFrame()
    zonelist = [zoneindex.real]
    savemeshvars = []
    if integrate_surface:
        #integrate power on main surface
        mp_powers, hmin = surface_analysis(main_frame, zonename, do_cms,
                                do_1Dsw, find_DFT=(mode.find('mp')!=-1),
                                cuttoff=tail_analysis_cap, blank=do_blank,
                                blank_variable=blank_variable,
                                blank_value=blank_value,
                                timedelta=deltatime)
        if save_mesh:
            cc_length = len(field_data.zone(zonename).values(
                                                  'x_cc').as_numpy_array())
            for name in field_data.variable_names:
                var_length = len(field_data.zone(zonename).values(
                                  name.split(' ')[0]+'*').as_numpy_array())
                if var_length==cc_length:
                    savemeshvars.append(name)
        if do_1Dsw and save_mesh:
            for name in ['1DK_net [W/Re^2]','1DP0_net [W/Re^2]',
                            '1DExB_net [W/Re^2]']:
                savemeshvars.append(name)
        if mode=='iso_betastar':
            #integrate power on innerboundary surface
            inner_mesh = pd.DataFrame()
            innerbound_powers, _ = surface_analysis(main_frame,
                                    field_data.zone('*innerbound*').name,
                                                    False, do_1Dsw,
                                                    find_DFT=False,
                                            cuttoff=tail_analysis_cap)
            innerbound_powers = innerbound_powers.add_prefix('inner')
            print('\nInnerbound surface power calculated')
            if save_mesh:
                #save inner boundary mesh to the mesh file
                inner_index =(field_data.zone('*innerbound*').index.real)
                for var in savemeshvars:
                    inner_mesh[var] = field_data.zone(inner_index).values(
                                    var.split(' ')[0]+'*').as_numpy_array()
        print('\nMagnetopause Power Terms')
        print(mp_powers)
    else:
        hmin = do_cms
    if integrate_volume:
        print('Zonename is: {}'.format(zonename))
        if zonename == 'sphere':
            doblank = False
        else:
            doblank = True
        mp_energies = volume_analysis(main_frame, state_var_name,
                                        do_1Dsw, do_cms, sp_rmax,
                                        cuttoff=tail_analysis_cap,
                                        blank=False, dt=deltatime,
                                        tail_h=hmin)
        print('\nMagnetopause Energy Terms')
        print(mp_energies)
    if save_mesh:
        #save mesh to hdf file
        for var in savemeshvars:
            mp_mesh[var] = field_data.zone(zoneindex.real
                           ).values(var.split(' ')[0]+'*').as_numpy_array()
    if integrate_volume and integrate_surface and do_virial:
        #Complete virial predicted Dst
        mp_powers['Virial_Dst [nT]']=(mp_powers['Virial Surface Total [J]']+
                                    mp_energies['Virial 2x Uk [J]']+
                                   mp_energies['uB_dist [J]'])/(8e13)*-3/2
    #Add time and x_subsolar
    mp_powers['Time [UTC]'] = eventtime
    mp_powers['X_subsolar [Re]'] = x_subsolar
    if write_data:
        write_mesh(outputpath+'/meshdata/mesh_'+datestring+'.h5',
                    zonename, pd.Series(eventtime), mp_mesh)
        write_to_hdf(outputpath+'/energeticsdata/energetics_'+
                        datestring+'.h5', zonename,
                        mp_energies=mp_energies, mp_powers=mp_powers,
                        mp_inner_powers=innerbound_powers)
    if disp_result:
        display_progress(outputpath+'/meshdata/mesh_'+datestring+'.h5',
                            outputpath+'/energeticsdata/energetics_'+
                            datestring+'.h5', zonename)
    return mp_mesh, mp_powers, mp_energies

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
