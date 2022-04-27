#!/usr/bin/env python3
"""Extraction routine for magnetosphere objects (surfaces and/or volumes)
"""
import logging as log
import os,sys,time,warnings
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, deg2rad, linspace
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#interpackage modules
from global_energetics.extract import swmf_access
from global_energetics.extract.surface_tools import (surface_analysis,
                                                     calc_integral)
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

def todimensional(dataset, **kwargs):
    """Function modifies dimensionless variables -> dimensional variables
    Inputs
        dataset (frame.dataset)- tecplot dataset object
        kwargs:
    """
    eq = tp.data.operate.execute_equation
    proton_mass = 1.6605e-27
    cMu = pi*4e-7
    #SWMF sets up two conversions to go between
    # No (nondimensional) Si (SI) and Io (input output),
    # what we want is No2Io = No2Si_V / Io2Si_V
    #Found these conversions by grep'ing for 'No2Si_V' in ModPhysics.f90
    No2Si = {'X':6371*1e3,                             #planet radius
             'Y':6371*1e3,
             'Z':6371*1e3,
             'Rho':1e6*proton_mass,                    #proton mass
             'U_x':6371*1e3,                           #planet radius
             'U_y':6371*1e3,
             'U_z':6371*1e3,
             'P':1e6*proton_mass*(6371*1e3)**2,        #Rho * U^2
             'B_x':6371*1e3*sqrt(cMu*1e6*proton_mass), #U * sqrt(M*rho)
             'B_y':6371*1e3*sqrt(cMu*1e6*proton_mass),
             'B_z':6371*1e3*sqrt(cMu*1e6*proton_mass),
             'J_x':(sqrt(cMu*1e6*proton_mass)/cMu),    #B/(X*cMu)
             'J_y':(sqrt(cMu*1e6*proton_mass)/cMu),
             'J_z':(sqrt(cMu*1e6*proton_mass)/cMu)
            }
    #Found these conversions by grep'ing for 'Io2Si_V' in ModPhysics.f90
    Io2Si = {'X':6371*1e3,                  #planet radius
             'Y':6371*1e3,                  #planet radius
             'Z':6371*1e3,                  #planet radius
             'Rho':1e6*proton_mass,         #Mp/cm^3
             'U_x':1e3,                     #km/s
             'U_y':1e3,                     #km/s
             'U_z':1e3,                     #km/s
             'P':1e-9,                      #nPa
             'B_x':1e-9,                    #nT
             'B_y':1e-9,                    #nT
             'B_z':1e-9,                    #nT
             'J_x':1e-6,                    #microA/m^2
             'J_y':1e-6,                    #microA/m^2
             'J_z':1e-6,                    #microA/m^2
             'theta_1':pi/180,              #degrees
             'theta_2':pi/180,              #degrees
             'phi_1':pi/180,                #degrees
             'phi_2':pi/180                 #degrees
            }
    units = {'X':'[R]','Y':'[R]','Z':'[R]',
             'Rho':'[amu/cm^3]',
             'U_x':'[km/s]','U_y':'[km/s]','U_z':'[km/s]',
             'P':'[nPa]',
             'B_x':'[nT]','B_y':'[nT]','B_z':'[nT]',
             'J_x':'[uA/m^2]','J_y':'[uA/m^2]','J_z':'[uA/m^2]',
             'theta_1':'[deg]',
             'theta_2':'[deg]',
             'phi_1':'[deg]',
             'phi_2':'[deg]'
            }
    for var in dataset.variable_names:
        if 'status' not in var.lower() and '[' not in var:
            if var in units.keys():
                unit = units[var]
                conversion = No2Si[var]/Io2Si[var]
                print('Changing '+var+' by multiplying by '+str(conversion))
                eq('{'+var+' '+unit+'} = {'+var+'}*'+str(conversion))
                print('Removing '+var)
                dataset.delete_variables([dataset.variable(var).index])
            else:
                print('unable to convert '+var+' not in dictionary!')
        elif 'status' in var.lower():
            dataset.variable(var).name = 'Status'
    #Reset XYZ variables so 3D plotting makes sense
    dataset.frame.plot().axes.x_axis.variable = dataset.variable('X *')
    dataset.frame.plot().axes.y_axis.variable = dataset.variable('Y *')
    dataset.frame.plot().axes.z_axis.variable = dataset.variable('Z *')

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
    #Validate mode selection
    approved= ['iso_betastar', 'shue97', 'shue98', 'shue', 'box', 'sphere',
               'lcb', 'nlobe', 'slobe', 'rc', 'ps', 'qDp','closed','bs']
    if not any([mode == match for match in approved]):
        assert False, ('Magnetopause mode "{}" not recognized!!'.format(
                                                                    mode)+
                       'Please set mode to one of the following:'+
                       '\t'.join(approved))

    #See if status variable in dataset
    if 'Status' not in field_data.variable_names:
        has_status = False
        do_trace = True
        print('Status variable not included in dataset!'+
                'do_trace -> True')
    else:
        has_status = True

    #This specific case won't work
    if do_trace and mode == 'lcb':
        print('last_closed boundary not setup with trace mode!')
        return

    #Make sure if cms is selected that we have a second zone to do d/dt
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

    #Check to make sure that dimensional variables are given
    if ('X' in field_data.variable_names and
        'X [R]' not in field_data.variable_names):
        print('dimensionless variables given! converting to dimensional...')
        todimensional(field_data)
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
        future_aux = field_data.zone('future*').aux_data
    #set frame name and calculate global variables
    if field_data.variable_names.count('r [R]') ==0:
        main_frame = tp.active_frame()
        print('Calculating global energetic variables')
        main_frame.name = 'main'
        get_global_variables(field_data, analysis_type,aux=aux)
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
            closed_zone = setup_isosurface(1,closed_index,'lcb',
                                        blankvalue=kwargs.get('inner_r',3))
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
                                                global_key='future*')
        else:
            closed_index = calc_closed_state('lcb','Status', 3, tail_cap,0)
            closed_zone = setup_isosurface(1,closed_index,'lcb',
                                        blankvalue=kwargs.get('inner_r',3))
            if do_cms:
                future_closed_index = calc_closed_state('future_lcb',
                                                        'Status', 3,
                                                        tail_cap, 1)
                future_closed_zone =setup_isosurface(1,future_closed_index,
                                                     'future_lcb',
                                        blankvalue=kwargs.get('inner_r',3))
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
        print('closed_zone: '+closed_zone.name)
        return aux, closed_zone, None

def generate_3Dobj(sourcezone, **kwargs):
    """Creates isosurface zone depending on mode setting
    Inputs
        sourcezone (Zone)- tecplot Zone with data used for generation
        kwargs:
            analysis_type (str)- kw for family of measurements
            integrate_volume (bool)
            mode (str)- kw for what to generate
            modes (list[str])- !watch out! this overwrites 'mode' setting
    Returns
        zonelist (list[Zone])- list of tp Zone objects of whats generated.
            This useful so we know where to apply SURFACE equations and
            integration operations.
        state_indices (list[VariableIndex])- list of tp Variable Index's.
            This used to apply conditions to achieve VOLUME integrations
        kwargs- we modify at least one arg depending on settings
    """
    zonelist, state_indices = [], []
    if kwargs=={}: print('Generating default zones:\n\t'+
                         'mode=iso_betastar\n\tanalysis_type=energy')

    #Decide what kind of things we will be generating
    if ('virial' in kwargs.get('analysis_type','energy')) and (
                                    kwargs.get('integrate_volume',True)):
        modes = [kwargs.get('mode','iso_betastar'),
                 'closed', 'rc', 'nlobe', 'slobe']
    else:
        modes = [kwargs.get('mode','iso_betastar')]
    modes=kwargs.get('modes',modes)
    if 'modes' in kwargs:
        warnings.warn('Overwriting zone generation modes!, could be '+
                      'incompattable w/ analysis type', UserWarning)
    else:
        kwargs.update({'modes':modes})

    #Create the tecplot objects and store into lists
    for m in modes:
        zone, inner_zone, state_index = calc_state(m, sourcezone, **kwargs)
        if (type(zone) != type(None) or type(inner_zone)!=type(None)):
            if 'zone_rename' in kwargs:
                zone.name = kwargs.get('zone_rename','')+'_'+m
            zonelist.append(zone)
            state_indices.append(state_index)

    #Assign magnetopause variable and get geometry vars (others will ref)
    if sourcezone.dataset.variable('mp*') is not None:
        kwargs.update({'mpvar':sourcezone.dataset.variable('mp*').name})
        get_surf_geom_variables(sourcezone.dataset.zone('mp*'))
        #get_surf_geom_variables(sourcezone.dataset.zone('ext_bs*'))
    else:#now 'mpvar' could be a bit of a misnomer
        kwargs.update({'mpvar':sourcezone.dataset.variable(state_index).name})
        get_surf_geom_variables(zonelist[0])
    if kwargs.get('do_cms','energy'in kwargs.get('analysis_type','energy')):
        futurezone = sourcezone.dataset.zone('future*')
        _,j,future_state_index=calc_state(kwargs.get('mode','iso_betastar'),
                                          futurezone,**kwargs)
        #get state variable representing acquisitions/forfeitures
        calc_delta_state(sourcezone.dataset.variable(state_index).name,
                  sourcezone.dataset.variable(future_state_index).name)
    return zonelist, state_indices, kwargs


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
    tail_analysis_cap = kwargs.get('tail_analysis_cap',
                                   kwargs.get('tail_cap',-20))
    integrate_volume = kwargs.get('integrate_volume', True)
    save_mesh = kwargs.get('save_mesh', True)
    write_data = kwargs.get('write_data', True)
    disp_result = kwargs.get('disp_result', True)
    verbose = kwargs.get('verbose', True)
    do_cms = kwargs.get('do_cms', 'energy' in analysis_type)
    inner_cond = kwargs.get('inner_cond', 'sphere')
    inner_r = kwargs.get('inner_r', 3)
    do_blank = kwargs.get('do_blank', False)
    blank_variable = kwargs.get('blank_variable', 'r *')
    blank_value = kwargs.get('blank_value', inner_r)
    blank_operator = kwargs.get('blank_operator', RelOp.LessThan)
    extra_surf_terms = kwargs.get('customTerms', {})
    do_1Dsw = kwargs.get('do_1Dsw', False)
    globalzone = field_data.zone('global_field')
    mpbetastar = kwargs.get('mpbetastar', 0.7)
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
        pass#TODO does anything go here?
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
    # !!! kwargs updated !!!
    kwargs.update({'x_subsolar':float(aux['x_subsolar'])})
    kwargs.update({'closed_zone':closed_zone})
    kwargs.update({'future_closed_zone':future_closed_zone})
    main_frame = [fr for fr in tp.frames('main')][0]

    # !!! kwargs updated !!!
    zonelist, state_indices, kwargs = generate_3Dobj(globalzone, **kwargs)

    #perform integration for surface and volume quantities
    mp_powers, innerbound_powers = pd.DataFrame(), pd.DataFrame()
    mp_mesh = {}
    data_to_write={}
    savemeshvars = {}
    for mode in kwargs.get('modes'):
        savemeshvars.update({mode:[]})
    ################################################################
    if integrate_surface:
        for zone in zonelist[0:1]:
            #integrate power on main surface
            print('Working on: '+zone.name+' surface')
            surf_results = surface_analysis(zone,**kwargs)
            #Add time and x_subsolar
            surf_results['Time [UTC]'] = eventtime
            surf_results['X_subsolar [Re]'] = float(aux['x_subsolar'])
            mp_mesh.update({'Time [UTC]':
                                 pd.DataFrame({'Time [UTC]':[eventtime]})})
            data_to_write.update({zone.name+'_surface':surf_results})
            if save_mesh:
                cc_length =len(zone.values('x_cc').as_numpy_array())
                for name in field_data.variable_names:
                    var_length = len(zone.values(
                                  name.split(' ')[0]+'*').as_numpy_array())
                    if var_length==cc_length:
                        savemeshvars[kwargs.get('modes')[0]].append(name)
            if do_1Dsw and save_mesh:
                for name in ['1DK_net [W/Re^2]','1DP0_net [W/Re^2]',
                                '1DExB_net [W/Re^2]']:
                    savemeshvars[kwargs.get('modes')[0]].append(name)
        if 'iso_betastar' in kwargs.get('modes'):
            #integrate power on innerboundary surface
            inner_mesh = {}
            inner_zone = field_data.zone('*innerbound*')
            print('Working on: '+inner_zone.name)
            inner_surf_results = surface_analysis(inner_zone, **kwargs)
            inner_surf_results['Time [UTC]'] = eventtime
            data_to_write.update({zonelist[0].name+'_inner_surface':
                                                       inner_surf_results})
            if save_mesh:
                #save inner boundary mesh to the mesh file
                for var in savemeshvars[kwargs.get('modes')[0]]:
                    inner_mesh[var]=inner_zone.values(
                                    var.split(' ')[0]+'*').as_numpy_array()
                inner_mesh.update({'Time [UTC]':
                                 pd.DataFrame({'Time [UTC]':[eventtime]})})
    ################################################################
    if integrate_volume:
        for state_index in enumerate(state_indices):
            region = zonelist[state_index[0]]
            print('Working on: '+region.name+' volume')
            energies = volume_analysis(field_data.variable(state_index[1]),
                                       **kwargs)
            energies['Time [UTC]'] = eventtime
            data_to_write.update({region.name+'_volume':energies})
            if save_mesh:
                for var in ['beta_star','uB [J/Re^3]','Pth [J/Re^3]',
                      'KE [J/Re^3]','uHydro [J/Re^3]','Utot [J/Re^3]']:
                    usename = var.split(' ')[0]+'_cc'
                    eq = tp.data.operate.execute_equation
                    eq('{'+usename+'}={'+var+'}',zones=[region],
                            value_location=ValueLocation.CellCentered)
                    savemeshvars[kwargs.get('modes')[state_index[0]]].append(usename)
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
    if write_data:
        datestring = (str(eventtime.year)+'-'+str(eventtime.month)+'-'+
                      str(eventtime.day)+'-'+str(eventtime.hour)+'-'+
                      str(eventtime.minute))
        write_to_hdf(outputpath+'/energeticsdata/energetics_'+
                        datestring+'.h5', data_to_write)
    if save_mesh:
        write_to_hdf(outputpath+'/meshdata/mesh_'+datestring+'.h5',mp_mesh)
    if disp_result:
        display_progress(outputpath+'/meshdata/mesh_'+datestring+'.h5',
                            outputpath+'/energeticsdata/energetics_'+
                            datestring+'.h5',
                            [zn.name for zn in zonelist])
    return mp_mesh, data_to_write

if __name__ == "__main__":
    pass#TODO could make this a simple (!and fast) test function...
