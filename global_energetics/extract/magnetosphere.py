#!/usr/bin/env python3
"""Extraction routine for magnetosphere objects (surfaces and/or volumes)
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
from global_energetics.extract import swmf_access
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract.stream_tools import (streamfind_bisection,
                                                    get_global_variables,
                                                    calc_state,
                                                    get_surf_geom_variables,
                                                    setup_isosurface,
                                                    calc_closed_state,
                                                    calc_delta_state,
                                                    get_1D_sw_variables,
                                             get_surfaceshear_variables)
from global_energetics.write_disp import (write_mesh, write_to_hdf,
                                          display_progress)


def validate_preproc(field_data, mode, source, outputpath, do_cms, verbose,
                     do_trace):
    """Function checks compatibility of selected settings and displays
        if in verbose mode
    Inputs
        mode- iso_betastar (default), sphere, box, shue97/shue98,
              subzones (n/slobe, ps, rc, qDp)
        source- only capable of handling SWMF input
        outputpath- path for output of .h5 files
        do_cms- for determining surface velocity at each cell (broken)
        do_trace- boolean turns on fieldline tracing to find x_subsolar
    Return
        do_trace- could by forced True if no Status variable
    """
    approved= ['iso_betastar', 'shue97', 'shue98', 'shue', 'box', 'sphere',
               'lcb', 'nlobe', 'slobe', 'rc', 'ps', 'qDp']
    if not any([mode == match for match in approved]):
        assert False, ('Magnetopause mode "{}" not recognized!!'.format(
                                                                    mode)+
                       'Please set mode to one of the following:'+
                       '\t'.join(approved))
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
    #get date and time info based on data source
    if source == 'swmf':
        eventtime = swmf_access.swmf_read_time()
        if do_cms:
            futuretime = swmf_access.swmf_read_time(zoneindex=1)
            deltatime = (futuretime-eventtime).seconds
        else:
            deltatime=0
    #assert not badsettings, ('Bad input settings! Check file(s) and mode'+
    #                         'selection compatibility')
    return do_trace, eventtime, deltatime

def show_settings(**kwargs):
    """Function prints nice summary of all active settings
    Inputs- all optional
        see docstring for get_magnetopause
    """
    display = ('Analyzing Magnetopause with the following settings:\n')
    for arg in kwargs:
        display = display+'\t\t'+arg+'+: {}\n'.format(kwargs[arg])

def prep_field_data(field_data, **kwargs):
    """Function modifies global field data before 3D objects are ID'd
    Inputs
        field_data
        **kwargs- see get_magnetopause for full list
    Return
        aux
        closed_index
        future_closed_index (only used if do_cms=True)
    """
    #pass along some kwargs from get_magnetopause
    analysis_type = kwargs.get('analysis_type', 'energy')
    do_trace = kwargs.get('do_trace', False)
    do_cms = kwargs.get('do_cms', True)
    do_1Dsw = kwargs.get('do_1Dsw', False)
    tail_cap = kwargs.get('tail_cap', -20)
    lon_bounds = kwargs.get('lon_bounds', 10)
    n_fieldlines = kwargs.get('n_fieldlines', 5)
    rmax = kwargs.get('rmax', 30)
    rmin = kwargs.get('rmin', 3)
    itr_max = kwargs.get('itr_max', 100)
    tol = kwargs.get('tol', 0.1)
    if kwargs.get('verbose',False):
        show_settings(kwargs)
    #Auxillary data from tecplot file
    aux = field_data.zone('global_field').aux_data
    if do_cms:
        futurezone = field_data.zone('future*')
        future_aux = futurezone.aux_data
    #set frame name and calculate global variables
    if field_data.variable_names.count('r [R]') ==0:
        main_frame = tp.active_frame()
        print('Calculating global energetic variables')
        main_frame.name = 'main'
        get_global_variables(field_data, analysis_type)
        if do_1Dsw:
            print('Calculating 1D "pristine" Solar Wind variables')
            get_1D_sw_variables(field_data, 30, -30, 121)
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
            closed_index = calc_closed_state('lcb','Status', 3, tail_cap,0)
            closed_zone, _ = setup_isosurface(1,closed_index,'lcb')
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
            if do_cms:
                future_closed_index = calc_closed_state('future_lcb',
                                                        'Status', 3,
                                                        tail_cap, 1)
                future_closed_zone, _ = setup_isosurface(1,
                                                       future_closed_index,
                                                       'future_lcb')
        x_subsolar = 1
        x_subsolar = max(x_subsolar,
                field_data.zone(closed_zone.index).values('X *').max())
        print('x_subsolar found at {}'.format(x_subsolar))
        aux['x_subsolar'] = x_subsolar
        if do_cms:
            future_x_subsolar = 1
            future_x_subsolar = max(future_x_subsolar,
              field_data.zone(future_closed_zone.index).values('X *').max())
            future_aux['x_subsolar'] = future_x_subsolar
        if do_trace:
            #delete streamzone
            field_data.delete_zones(closedzone_index)
            closed_index = None
            closed_zone = None
    if do_cms:
        return aux, closed_zone, future_closed_zone
    else:
        return aux, closed_zone, None

def get_magnetosphere(field_data, *, mode='iso_betastar', **kwargs):
    """Function that finds, plots and calculates quantities on
        magnetospheric regions
    Inputs
        Required
            field_data- tecplot DataSet object with 3D field data
            mode- iso_betastar (default), sphere, box, shue97/shue98,
                  subzones (n/slobe, ps, rc, qDp)
        General
            outputpath- path for output of .h5 files
            source- only capable of handling SWMF input
            xyzvar- for X, Y, Z variables in field data variable list
            zone_rename- optional rename if calling multiple times
            analysis_type- 'energy' (default), if not developed preset,
                           will calculate all available equations, used to
                           narrow scope of calculations
            integrate_surface, integrate_volume- booleans for analysis
            save_mesh, write_data, disp_result- booleans
            verbose- boolean

        Types of Surfaces:
        *Betastar magnetopause (iso_betastar mode)
            (inner_cond, inner_r)- ('sphere', 3) (default) geometry of
                                   inner boundary
            mpbetastar- 0.7(default) value for isosurface
            tail_cap, tail_analysis_cap- where to draw/calculate mp
        *Sphere
            sp_x, sp_y, sp_z, sp_rmax, sp_rmin- where to create object
        *Box
            box_xyzmax, box_xyzmin- where to create object
        *Subzones (ps,rc,qDp,nlobe,slobe modes)
            lshelllim- L shell threshold value
            bxmax- magnitude of B_x determines where plasma sheet is

        Optional modes
        >1DSolarWind
            do_1Dsw- boolean for projecting a 1D "pristine" sw solution
            oneDmx, oneDmn, n_oneD- parameters for 1D curve (probe points)
        >Streamtracing
            do_trace- boolean turns on fieldline tracing to find x_subsolar
            lon_bounds, n_fieldlines- bounds and density of search
            rmax, rmin, itr_max, tol- parameters for bisection algorithm
        >CMS
            do_cms- for determining surface velocity at each cell (broken)
        >Blanking
            do_blank, blank_variable, blank_value- use tp blank feature
    """
    #Setup default values based on any given kwargs
    outputpath = kwargs.get('outputpath', 'output/')
    source = kwargs.get('source', 'swmf')
    xyzvar = kwargs.get('xyzvar', [0,1,2])
    zone_rename = kwargs.get('zone_rename', None)
    analysis_type = kwargs.get('analysis_type', 'energy')
    integrate_surface = kwargs.get('integrate_surface', True)
    tail_analysis_cap = kwargs.get('tail_analysis_cap', -20)
    integrate_volume = kwargs.get('integrate_volume', True)
    save_mesh = kwargs.get('save_mesh', True)
    write_data = kwargs.get('write_data', True)
    disp_result = kwargs.get('disp_result', True)
    verbose = kwargs.get('verbose', True)
    do_cms = kwargs.get('do_cms', 'energy' in analysis_type)
    do_blank = kwargs.get('do_blank', False)
    blank_variable = kwargs.get('blank_variable', 'r *')
    blank_value = kwargs.get('blank_value', 3)
    blank_operator = kwargs.get('blank_operator', RelOp.LessThan)
    extra_surf_terms = kwargs.get('customTerms', {})
    do_1Dsw = kwargs.get('do_1Dsw', False)
    globalzone = field_data.zone('global_field')
    futurezone = field_data.zone('future*')
    if mode == 'iso_betastar':
        inner_cond = kwargs.get('inner_cond', 'sphere')
        inner_r = kwargs.get('inner_r', 3)
        mpbetastar = kwargs.get('mpbetastar', 0.7)
    if any([mode==match for match in ['iso_betastar', 'shue97', 'shue98',
                                      'shue']]):
        tail_cap = kwargs.get('tail_cap', -20)
    if mode == 'sphere':
        sp_x = kwargs.get('sp_x', 0)
        sp_y = kwargs.get('sp_y', 0)
        sp_z = kwargs.get('sp_z', 0)
        sp_rmin = kwargs.get('sp_rmin', 0)
        sp_rmax = kwargs.get('sp_rmax', 3)
    if mode == 'box':
        box_xmax = kwargs.get('box_xmax', -5)
        box_xmin = kwargs.get('box_xmin', -8)
        box_ymax = kwargs.get('box_ymax', 5)
        box_ymin = kwargs.get('box_ymin', -5)
        box_zmax = kwargs.get('box_zmax', 5)
        box_zmin = kwargs.get('box_zmin', -5)
    if any([mode==match for match in ['ps','rc','qDp','nlobe','slobe']]):
        lshelllim = kwargs.get('lshelllim', 7)
        bxmax = kwargs.get('bxmax', 10)
    if do_1Dsw:
        oneDmx = kwargs.get('oneDmx', 30)
        oneDmn = kwargs.get('oneDmn', -30)
        n_oneD = kwargs.get('n_oneD', 121)
    if do_cms:
        pass
    do_trace, eventtime, deltatime = validate_preproc(field_data, mode,
                                                      source,outputpath,
                                                      do_cms, verbose,
                                              kwargs.get('do_trace',False))
    if do_trace:
        lon_bounds = kwargs.get('lon_bounds', 10)
        n_fieldlines = kwargs.get('n_fieldlines', 5)
        rmax = kwargs.get('rmax', 30)
        rmin = kwargs.get('rmin', 3)
        itr_max = kwargs.get('itr_max', 100)
        tol = kwargs.get('tol', 0.1)
    #prepare field data
    aux, closed_zone, future_closed_zone = prep_field_data(field_data,
                                                           **kwargs)
    kwargs.update({'x_subsolar':float(aux['x_subsolar'])})
    kwargs.update({'closed_zone':closed_zone})
    kwargs.update({'future_closed_zone':future_closed_zone})
    main_frame = [fr for fr in tp.frames('main')][0]
    #Create isosurface zone depending on mode setting
    ################################################################
    zonelist, state_indices = [], []
    if 'virial' in analysis_type:
        modes = [mode, 'ps', 'qDp', 'rc', 'nlobe', 'slobe']
    else:
        modes = [mode, 'ps', 'qDp', 'rc', 'nlobe', 'slobe']
        #modes = [mode]
    for m in modes:
        zone, inner_zone, state_index = calc_state(m, globalzone, **kwargs)
        if zone_rename != None:
            zone.name = zone_rename+'_'+m
        zonelist.append(zone)
        state_indices.append(state_index)
    #Assign magnetopause variable and get geometry vars (others will ref)
    if field_data.variable('mp*') is not None:
        kwargs.update({'mpvar':field_data.variable('mp*').name})
        get_surf_geom_variables(field_data.zone('mp*'))
    if do_cms:
        future_mp,_,future_state_index=calc_state(mode,futurezone,**kwargs)
        #get state variable representing acquisitions/forfeitures
        calc_delta_state(zonelist[0].name, future_mp.name)
    ################################################################
    #perform integration for surface and volume quantities
    mp_powers, innerbound_powers = pd.DataFrame(), pd.DataFrame()
    mp_mesh = {}
    data_to_write={}
    savemeshvars = {}
    for mode in modes:
        savemeshvars.update({mode:[]})
    ################################################################
    if integrate_surface:
        #integrate power on main surface
        mp_powers = surface_analysis(zonelist[0],**kwargs)
        if save_mesh:
            cc_length = len(zonelist[0].values('x_cc').as_numpy_array())
            for name in field_data.variable_names:
                var_length = len(zonelist[0].values(
                                  name.split(' ')[0]+'*').as_numpy_array())
                if var_length==cc_length:
                    savemeshvars[modes[0]].append(name)
        if do_1Dsw and save_mesh:
            for name in ['1DK_net [W/Re^2]','1DP0_net [W/Re^2]',
                            '1DExB_net [W/Re^2]']:
                savemeshvars[modes[0]].append(name)
        if mode=='iso_betastar':
            #integrate power on innerboundary surface
            inner_mesh = pd.DataFrame()
            innerbound_powers=surface_analysis(
                                  field_data.zone('*innerbound*'),**kwargs)
            innerbound_powers = innerbound_powers.add_prefix('inner')
            print('\nInnerbound surface power calculated')
            if save_mesh:
                #save inner boundary mesh to the mesh file
                inner_index =(field_data.zone('*innerbound*').index.real)
                for var in savemeshvars[modes[0]]:
                    inner_mesh[var] = field_data.zone(inner_index).values(
                                    var.split(' ')[0]+'*').as_numpy_array()
        print('\nMagnetopause Power Terms\n{}'.format(mp_powers))
    ################################################################
    if integrate_volume:
        for state_index in enumerate(state_indices):
            region = zonelist[state_index[0]]
            print('Working on: '+region.name)
            energies = volume_analysis(field_data.variable(state_index[1]),
                                       **kwargs)
            energies['Time [UTC]'] = eventtime
            data_to_write.update({region.name:energies})
            for var in ['beta_star','uB [J/Re^3]','Pth [J/Re^3]',
                        'KE [J/Re^3]','uHydro [J/Re^3]','Utot [J/Re^3]']:
                usename = var.split(' ')[0]+'_cc'
                tp.data.operate.execute_equation('{'+usename+'}={'+var+'}',
                  zones=[region],value_location=ValueLocation.CellCentered)
                savemeshvars[modes[state_index[0]]].append(usename)
    ################################################################
    if save_mesh:
        #save mesh to hdf file
        for state_index in enumerate(state_indices):
            region = zonelist[state_index[0]]
            meshvalues = pd.DataFrame()
            for var in [m for m in savemeshvars.values()][state_index[0]]:
                usename = var.split(' ')[0]+'*'
                meshvalues[var] = region.values(usename).as_numpy_array()
            mp_mesh.update({region.name:meshvalues})
    ################################################################
    if integrate_volume and integrate_surface and(
                      'virial' in analysis_type or analysis_type=='all'):
        #Complete virial predicted Dst
        region = zonelist[0]
        mp_powers['Virial_Dst [nT]']=(mp_powers['Virial Surface Total [J]']+
             data_to_write[region.name]['Virial Volume Total [J]'])/(-8e13)
    ################################################################
    #Add time and x_subsolar
    mp_powers['Time [UTC]'] = eventtime
    mp_mesh.update({'Time [UTC]':pd.DataFrame({'Time [UTC]':[eventtime]})})
    mp_powers['X_subsolar [Re]'] = float(aux['x_subsolar'])
    region = zonelist[0]
    if write_data:
        for key in mp_powers:
            data_to_write[region.name][key] = mp_powers[key]
        datestring = (str(eventtime.year)+'-'+str(eventtime.month)+'-'+
                      str(eventtime.day)+'-'+str(eventtime.hour)+'-'+
                      str(eventtime.minute))
        write_to_hdf(outputpath+'/meshdata/mesh_'+datestring+'.h5',mp_mesh)
        write_to_hdf(outputpath+'/energeticsdata/energetics_'+
                        datestring+'.h5', data_to_write)
    if disp_result:
        display_progress(outputpath+'/meshdata/mesh_'+datestring+'.h5',
                            outputpath+'/energeticsdata/energetics_'+
                            datestring+'.h5',
                            [zn.name for zn in zonelist])
    return mp_mesh, data_to_write

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
