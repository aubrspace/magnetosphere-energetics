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
from global_energetics.extract import line_tools
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import post_proc_interface2
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract.mapping import reversed_mapping
from global_energetics.extract.tec_tools import (streamfind_bisection,
                                                    get_global_variables,
                                                    calc_state,
                                                    get_surf_geom_variables,
                                                    setup_isosurface,
                                                    setup_solidvolume,
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
             }#'theta_1':pi/180,              #degrees
             #'theta_2':pi/180,              #degrees
             #'phi_1':pi/180,                #degrees
             #'phi_2':pi/180                 #degrees
            #}
    units = {'X':'[R]','Y':'[R]','Z':'[R]',
             'Rho':'[amu/cm^3]',
             'U_x':'[km/s]','U_y':'[km/s]','U_z':'[km/s]',
             'P':'[nPa]',
             'B_x':'[nT]','B_y':'[nT]','B_z':'[nT]',
             'J_x':'[uA/m^2]','J_y':'[uA/m^2]','J_z':'[uA/m^2]',
             'theta_1':'[deg]',
             'theta_2':'[deg]',
             'phi_1':'[deg]',
             'phi_2':'[deg]',
             'dvol':'[R]^3'
            }
    for var in dataset.variable_names:
        if 'status' not in var.lower() and '[' not in var:
            if var in units.keys():
                unit = units[var]
                #NOTE default is no modification
                conversion = No2Si.get(var,1)/Io2Si.get(var,1)
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
                     do_trace, tshift):
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
               'lcb', 'nlobe', 'slobe', 'rc', 'ps', 'qDp','closed','bs',
               'ellipsoid','plasmasheet']
    if not any([mode == match for match in approved]):
        assert False, ('Magnetopause mode "{}" not recognized!!'.format(
                                                                    mode)+
                       'Please set mode to one of the following:'+
                       '\t'.join(approved))

    #See if status variable in dataset
    if 'Status' not in field_data.variable_names:
        has_status = False
        do_trace = True
        warnings.warn('Status variable not included in dataset!'+
                      'do_trace -> True', UserWarning)
    else:
        has_status = True

    #This specific case won't work
    if do_trace and mode == 'lcb':
        warnings.warn('last_closed boundary not setup with trace mode!',
                      UserWarning)
        return

    #Make sure if cms is selected that we have a future/past zones to do d/dt
    if do_cms:
        if len([zn for zn in tp.active_frame().dataset.zones()])<3:
            warnings.warn('not enough data to do moving surfaces!\n'+
                          'do_cms -> False', UserWarning)
            do_cms = False

    #get date and time info based on data source
    if source == 'swmf':
        if do_cms:
            eventtime = (swmf_access.swmf_read_time(zoneindex=1)+
                                                  dt.timedelta(minutes=tshift))
            pasttime = (swmf_access.swmf_read_time(zoneindex=0)+
                        dt.timedelta(minutes=tshift))
            futuretime = (swmf_access.swmf_read_time(zoneindex=2)+
                          dt.timedelta(minutes=tshift))
            deltatime = (futuretime-pasttime).seconds/2
        else:
            eventtime = (swmf_access.swmf_read_time(zoneindex=0)+
                                                  dt.timedelta(minutes=tshift))
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
    """
    #pass along some kwargs from get_magnetopause
    analysis_type = kwargs.get('analysis_type', 'energy')
    do_trace = kwargs.get('do_trace', False)
    do_cms = kwargs.get('do_cms', False)
    do_1Dsw = kwargs.get('do_1Dsw', False)
    tail_cap = kwargs.get('tail_cap', -20)
    lon_bounds = kwargs.get('lon_bounds', 10)
    n_fieldlines = kwargs.get('n_fieldlines', 5)
    rmax = kwargs.get('rmax', 30)
    rmin = kwargs.get('rmin', 3)
    itr_max = kwargs.get('itr_max', 100)
    tol = kwargs.get('tol', 0.1)
    if kwargs.get('verbose',False):
        show_settings(**kwargs)
    #Auxillary data from tecplot file
    aux = field_data.zone('global_field').aux_data
    eq,cc=tp.data.operate.execute_equation,ValueLocation.CellCentered
    # If the Status == -3 anywhere it means the B field trace failed somewhere
    # What we want to do is try to use th_1 th_2 to fill in the portions
    #   of the trace that were completed before it was abandoned and set to -3
    if 'Status' in field_data.variable_names:
        if ((field_data.zone('global_field').values('Status').min() == -3) and
            ('theta_1 [deg]' in field_data.variable_names)):
            eq('{Status} = if({Status}==-3,'+#Recalculate from theta's
             'if(({theta_1 [deg]}>=0 && {theta_2 [deg]}>=-90),3,'+#closed
             'if(({theta_1 [deg]}<0  && {theta_2 [deg]}>=-90),1,'+#south
             'if(({theta_1 [deg]}>=0 && {theta_2 [deg]}< -90),2,0))),'+#north
                           '{Status})')#Don't change the non -3 Status values
    # If a truegrid file is given, then take that information in
    if 'truegridfile' in kwargs:
        if 'dvol [R]^3' in field_data.variable_names:
            #eq('{trueCellVolume} = {dvol [R]^3}',zones=[field_data.zone(0)])
            eq('{trueCellVolume} = {dvol [R]^3}')
        elif '.plt' in kwargs.get('truegridfile'):
            tp.data.load_tecplot(kwargs.get('truegridfile'),reset_style=False)
            truegrid = field_data.zone(-1)
            truegrid.name = 'truegrid'
            if 'dvol [R]^3' in field_data.variable_names:#Added along w zone
                tp.data.operate.interpolate_linear(
                                    field_data.zone('global_field'),
                                    source_zones=truegrid,
                                    variables=field_data.variable('dvol *'))
                if do_cms:
                    tp.data.operate.interpolate_linear(
                                    field_data.zone('past'),
                                    source_zones=truegrid,
                                    variables=field_data.variable('dvol *'))
                    tp.data.operate.interpolate_linear(
                                    field_data.zone('future'),
                                    source_zones=truegrid,
                                    variables=field_data.variable('dvol *'))
                eq('{trueCellVolume} = {dvol [R]^3}')
            else:
                ## Extract the dual and true grid info and sort using pandas
                field_data.add_variable('trueCellVolume')
                # Create a cellcentered XYZ for the true grid
                eq('{Xcc}={X [R]}',value_location=cc,zones=[truegrid.index])
                eq('{Ycc}={Y [R]}',value_location=cc,zones=[truegrid.index])
                eq('{Zcc}={Z [R]}',value_location=cc,zones=[truegrid.index])
                # Set numpy objects
                x = field_data.zone('global_field').values('X *'
                                                            ).as_numpy_array()
                y = field_data.zone('global_field').values('Y *'
                                                            ).as_numpy_array()
                z = field_data.zone('global_field').values('Z *'
                                                            ).as_numpy_array()
                xCell = truegrid.values('Xcc').as_numpy_array()
                yCell = truegrid.values('Ycc').as_numpy_array()
                zCell = truegrid.values('Zcc').as_numpy_array()
                trueCellVols = truegrid.values('Cell Volume').as_numpy_array()
                # Combine into data frames
                target = pd.DataFrame({'X':x,'Y':y,'Z':z})
                source = pd.DataFrame({'X':xCell,'Y':yCell,'Z':zCell,
                                    'trueVolume':trueCellVols})
                # Sort and load back into the main zone
                target.sort_values(by=['X','Y','Z'],inplace=True)
                source.sort_values(by=['X','Y','Z'],inplace=True)
                target['trueVolume'] = source['trueVolume'].values
                field_data.zone('global_field').values('trueCellVolume')[::]=(
                                     target['trueVolume'].sort_index().values)
    #set frame name and calculate global variables
    if 'r [R]' not in field_data.variable_names:
        main_frame = tp.active_frame()
        print('Calculating global energetic variables')
        main_frame.name = 'main'
        get_global_variables(field_data, analysis_type,aux=aux,
                             modes=kwargs.get('modes',[]),
                             verbose=kwargs.get('verbose',False),
                             customTerms=kwargs.get('customTerms',{}),
                            do_interfacing=kwargs.get('do_interfacing',False))
        if do_1Dsw or 'bs' in kwargs.get('modes',[]):
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
    if any(['x_subsolar' in k for k in aux.keys()]):
        # We already have x_subsolar and such
        x_subsolar = float(aux['x_subsolar'])
        x_nexl = float(aux['x_nexl'])
        inner_l = float(aux['inner_l'])
        closed_index = None
        closed_zone = None
        #Assign closed zone info if already exists
        if len([zn for zn in field_data.zones('*lcb*')]) > 0:
            closed_index = field_data.variable('lcb').index
            closed_zone = field_data.zone('*lcb*')
        elif has_status == True:
            closed_index = calc_closed_state('lcb','Status', 3, tail_cap,
                                        field_data.zone('global_field').index,
                                             kwargs.get('inner_r',3))
            closed_zone = setup_isosurface(1,closed_index,'lcb',
                                        blankvalue=kwargs.get('inner_r',3),
                                           blankvar='')

    else:
        # We don't already have x_subsolar and such
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
        elif (('modes' in kwargs) and
              ('iso_betastar' not in kwargs.get('modes',[])) and
              ('closed' not in kwargs.get('modes',[])) and
              ('xslice' not in kwargs.get('modes',[]))):
            # Edge case where we don't actually want x_subsolar for anything
            aux['x_subsolar'] = 0
            aux['x_nexl'] = 0
            aux['inner_l'] = 0
            return aux, None
        else:
            # Most of the time we need to go find it
            tp.data.operate.execute_equation('{status_cc}={Status}',
                                    value_location=ValueLocation.CellCentered)
            closed_index = calc_closed_state('lcb','Status', 3, tail_cap,
                                   field_data.zone('global_field').index,
                                             kwargs.get('inner_r',3))
            closed_zone = setup_isosurface(1,closed_index,'lcb',
                                        blankvalue=kwargs.get('inner_r',3),
                                           blankvar='')
            if do_cms:
                # Repeat closed zone search for past and future zones
                calc_closed_state('lcb','Status', 3,tail_cap,
                                   field_data.zone('past').index,
                                                   kwargs.get('inner_r',3))
                calc_closed_state('lcb','Status', 3,tail_cap,
                                   field_data.zone('future').index,
                                                   kwargs.get('inner_r',3))
        x_subsolar = 1
        x_subsolar = max(x_subsolar,
                field_data.zone(closed_zone.index).values('X *').max())
        x_nexl = -1*kwargs.get('inner_r',3)
        x_nexl = min(x_nexl,
                field_data.zone(closed_zone.index).values('X *').min())
        inner_l = min(kwargs.get('lshelllim',7),
                field_data.zone(closed_zone.index).values('Lshell').min())
        print('x_subsolar: {}, x_nexl: {},inner_L: {}'.format(
                                              x_subsolar, x_nexl,inner_l))
        aux['x_subsolar'] = x_subsolar
        aux['x_nexl'] = x_nexl
        aux['inner_l'] = inner_l
        if do_trace:
            #delete streamzone
            field_data.delete_zones(closedzone_index)
            closed_index = None
            closed_zone = None
    print('closed_zone: '+closed_zone.name)
    #Update global variables with the centroid adjusted magnetic mapping
    if (kwargs.get('do_interfacing',False) or
        'plasmasheet' in kwargs.get('modes')) and ('phi_1 [deg]' in
                                               field_data.variable_names):
        eq('{trace_limits}=if({Status}==3 && '+
                            '{r [R]}>'+str(kwargs.get('inner_r',3)-1)+',1,0)')
        reversed_mapping(field_data.zone('global_field'),'trace_limits',
                         **kwargs)
        if do_cms:
            reversed_mapping(field_data.zone('past'),'trace_limits',**kwargs)
            reversed_mapping(field_data.zone('future'),'trace_limits',**kwargs)
    return aux, closed_zone

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
        state_names (list[str])- list of tp Variable Index's.
            This used to apply conditions to achieve VOLUME integrations
        kwargs- we modify at least one arg depending on settings
    """
    zonelist, zonelist1D, state_names = [], [], []
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

    #If 'closed' in modes but 'rc' is not then we need to make sure that we
    # get the full closed region and not just L>threshold
    if 'closed' in modes and 'rc' not in modes:
        kwargs.update({'full_closed':True})

    #Will have both a past and future source if cms is on
    if kwargs.get('do_cms',False):
        pastzone = sourcezone.dataset.zone('past*')
        futurezone = sourcezone.dataset.zone('future*')
        sources = [pastzone,sourcezone,futurezone]
    else:
        sources = [sourcezone]

    #Create the tecplot objects from the source and store into lists
    for m in modes:
        zone,inner_zone,state_index=calc_state(m, sources,**kwargs,
                                               mainZoneIndex=sourcezone.index)
        print(m,zone,state_index)
        #if 'nlobe' in m:
        #    from IPython import embed; embed()
        if (type(zone)!=type(None)or type(inner_zone)!=type(None)):
            state_name = zone.dataset.variable(state_index).name
            if 'zone_rename' in kwargs:
                zone.name = kwargs.get('zone_rename','')+'_'+m
            if 'terminator' in m:
                zonelist1D.append(zone)#North
                zonelist1D.append(inner_zone)#South
                kwargs.update({'zonelist1D':zonelist1D})
            else:
                zonelist.append(zone)
                if 'perfect' not in zone.name:
                    state_names.append(state_name)
                if 'bs' in kwargs.get('modes',[]):
                    zonelist.append(inner_zone)
                    state_names.append(state_name)
                    #get_surf_geom_variables(inner_zone,**kwargs)
                if 'modes' in kwargs:
                    if (inner_zone is not None):
                        #and'iso_betastar' in zone.name):
                        zonelist.append(inner_zone)
    #Get the geometry variables AFTER all zones are created
    for zone in zonelist:
        get_surf_geom_variables(zone,**kwargs)
        if 'innerbound' in zone.name:
            copy_kwargs = kwargs.copy()
            copy_kwargs.update({'innerbound':True})
            get_surf_geom_variables(zone,**copy_kwargs)

    #Assign magnetopause variable
    kwargs.update({'mpvar':sourcezone.dataset.variable('mp*')})

    #Update kwargs dictionary rather than return
    kwargs.update({'zonelist':zonelist})
    kwargs.update({'state_names':state_names})

    return kwargs #NOTE variable return based on what zones are made!


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
    start_time = time.time()
    #Setup default values based on any given kwargs
    outputpath = kwargs.get('outputpath', 'output/')
    source = kwargs.get('source', 'swmf')
    xyzvar = kwargs.get('xyzvar', [0,1,2])
    zone_rename = kwargs.get('zone_rename', None)
    analysis_type = kwargs.get('analysis_type', 'energy')
    integrate_line = kwargs.get('integrate_line', False)
    integrate_surface = kwargs.get('integrate_surface', True)
    tail_analysis_cap = kwargs.get('tail_analysis_cap',
                                   kwargs.get('tail_cap',-20))
    integrate_volume = kwargs.get('integrate_volume', True)
    save_mesh = kwargs.get('save_mesh', False)
    write_data = kwargs.get('write_data', True)
    disp_result = kwargs.get('disp_result', False)
    verbose = kwargs.get('verbose', True)
    do_cms = kwargs.get('do_cms', False)
    #do_central_diff = kwargs.get('do_central_diff',False)
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
                                              kwargs.get('do_trace',False),
                                              kwargs.get('tshift',0))
    if 'tdelta' not in kwargs:
        kwargs.update({'tdelta':deltatime})
    print(kwargs.get('tdelta'))
    #timestamp
    ltime = time.time()-start_time
    print('PREPROC--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    if do_trace:
        # Only relevant for tracing routines
        lon_bounds = kwargs.get('lon_bounds', 10)
        n_fieldlines = kwargs.get('n_fieldlines', 5)
        rmax = kwargs.get('rmax', 30)
        rmin = kwargs.get('rmin', 3)
        itr_max = kwargs.get('itr_max', 100)
        tol = kwargs.get('tol', 0.1)
    #prepare field data
    aux, closed_zone = prep_field_data(field_data,**kwargs)
    #timestamp
    ltime = time.time()-start_time
    print('PREP:--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    # !!! kwargs updated !!!
    kwargs.update({'x_subsolar':float(aux['x_subsolar'])})
    kwargs.update({'x_nexl':float(aux['x_nexl'])})
    #NOTE for const L lim kwargs.update({'lshelllim':float(aux['inner_l'])})
    kwargs.update({'closed_zone':closed_zone})
    main_frame = [fr for fr in tp.frames('main')][0]

    # !!! kwargs updated !!!
    #TODO add in the 1D objects as zones as well
    kwargs = generate_3Dobj(globalzone, **kwargs)
    zonelist = kwargs.get('zonelist')
    state_names = kwargs.get('state_names')
    #timestamp
    ltime = time.time()-start_time
    print('GEN3D--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))

    #perform integration for surface and volume quantities
    mp_mesh = {}
    data_to_write={}
    savemeshvars = {}
    distribution_data={}
    for mode in kwargs.get('modes'):
        savemeshvars.update({mode:[]})
    ################################################################
    if integrate_surface:
        for zone in zonelist:
            #integrate power on created surface
            print('\nWorking on: '+zone.name+' surface')
            surf_results,dists = surface_tools.surface_analysis(zone,**kwargs)
            if ('mp_' in zone.name) and ('inner' not in zone.name):
                #Add x_subsolar
                surf_results['X_subsolar [Re]'] = float(aux['x_subsolar'])
                surf_results['X_NEXL [Re]'] = float(aux['x_nexl'])
                surf_results['L_min [L]'] = float(aux['inner_l'])
                mp_mesh.update({'Time [UTC]':
                                 pd.DataFrame({'Time [UTC]':[eventtime]})})
            elif 'bs' in zone.name:
                #Add x_subsolar
                surf_results['X_subsolar [Re]'] = zone.values('X *').max()
                if 'up' in zone.name:
                    #Add compression ratio's
                    for var in ['Rho*','Bmag*']:
                        dn_zone=field_data.zone(
                                              zone.name.split('up')[0]+'down')
                        up_data = zone.values(var).as_numpy_array()
                        dn_data = dn_zone.values(var).as_numpy_array()
                        upcond = (zone.values('X *').as_numpy_array()==
                                  zone.values('X *').as_numpy_array().max())
                        dncond = (dn_zone.values('X *').as_numpy_array()==
                                  dn_zone.values('X *').as_numpy_array().max())
                        upstream = up_data[upcond].mean()
                        dnstream = dn_data[dncond].mean()
                        surf_results['r_'+var.split('*')[0]]=dnstream/upstream
            surf_results['Time [UTC]'] = eventtime
            data_to_write.update({zone.name+'_surface':surf_results})
            if kwargs.get('save_surface_flux_dist',False):
                dists.attrs['time'] = eventtime
                distribution_data[zone.name+'_surface'] = dists
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
            '''
            if(('iso_betastar'in kwargs.get('modes'))and('mp_'in zone.name)
                                           and ('inner' not in zone.name)):
                #integrate power on innerboundary surface
                inner_mesh = {}
                inner_zone = field_data.zone('*innerbound*')
                print('\nWorking on: '+inner_zone.name)
                inner_surf_results,inner_dists=surface_tools.surface_analysis(
                                                       inner_zone,**kwargs)
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
            '''
        #quick post_proc, saves on number of integration calls
        data_to_write = surface_tools.post_proc(data_to_write,
                         do_interfacing=kwargs.get('do_interfacing',False))
        if kwargs.get('save_surface_flux_dist',False):
            distribution_data = surface_tools.post_proc(distribution_data,
                                              save_surface_flux_dist=True,
                                                        do_interfacing=False)
            #distribution_data['time'] = eventtime
    if any(['innerbound' in z.name for z in zonelist]):
        zonelist.remove(field_data.zone('*innerbound'))
    #timestamp
    ltime = time.time()-start_time
    print('SURF--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    ################################################################
    if integrate_volume:
        for i,state in enumerate(state_names):
            if 'Br 'not in state:
                region = zonelist[i]
                print('\nWorking on: '+region.name+' volume')
                energies = volume_analysis(field_data.variable(state),
                                           **kwargs)
                '''
                if kwargs.get('do_central_diff',False):
                    # Drop the non-motional terms
                    motion_keys=[k for k in energies.keys() if ('[W]' in k) or
                                                             ('[kg/s]' in k)]
                    energies = energies[motion_keys]
                    energies['Time [UTC]'] = (eventtime+dt.timedelta(seconds=
                                                    kwargs.get('tdelta',60)))
                else:
                '''
                energies['Time [UTC]'] = eventtime
                data_to_write.update({region.name+'_volume':energies})
            #elif not kwargs.get('do_central_diff',False):
            elif True:
                region = zonelist[i]
                print('\nWorking on: '+region.name+' averages')
                #Save the average Ux velocities on the plasmasheet
                plasmasheet = field_data.zone('ms_plasmasheet')
                ux = plasmasheet.values('U_x *').as_numpy_array()
                betastar = plasmasheet.values('beta_star').as_numpy_array()
                L = plasmasheet.values('Lshell').as_numpy_array()
                tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'NODAL')
                area = plasmasheet.values('Cell Volume').as_numpy_array()
                averages = pd.DataFrame()
                # Ux
                averages['ux_positive'] = [np.sum(ux[ux>0]*area[ux>0])/
                                           np.sum(area[ux>0])]
                averages['ux'] = [np.sum(ux*area)/np.sum(area)]
                averages['ux_negative'] = [np.sum(ux[ux<0]*area[ux<0])/
                                           np.sum(area[ux<0])]
                averages['positive_coverage']=[np.sum(area[ux>0])/np.sum(area)]
                # beta*
                averages['betastar_L>10'] = [np.sum(betastar[L>10]*area[L>10])/
                                             np.sum(area[L>10])]
                averages['Time [UTC]'] = eventtime
                data_to_write.update({region.name+'_averages':averages})
        if kwargs.get('do_interfacing',False):
            #Combine north and south lobes into single 'lobes'
            #North
            if 'ms_nlobe_volume' in data_to_write.keys():
                n = data_to_write.pop('ms_nlobe_volume')
                t = n['Time [UTC]']
            elif 'ms_slobe_volume' in data_to_write.keys():
                num_columns = len(data_to_write['ms_slobe_volume'].keys())
                n = pd.DataFrame(columns=data_to_write['ms_slobe_volume'].keys(),
                                 data=[np.zeros(num_columns)])
                t = data_to_write['ms_slobe_volume']['Time [UTC]']
                if 'log' in kwargs:
                    kwargs.get('logger').debug('North lobe not found!!')
            #South
            if 'ms_slobe_volume' in data_to_write.keys():
                s = data_to_write.pop('ms_slobe_volume')
            else:
                num_columns = len(data_to_write['ms_nlobe_volume'].keys())
                s = pd.DataFrame(columns=n.keys(),
                                 data=[np.zeros(num_columns)])
                if 'log' in kwargs:
                    kwargs.get('logger').debug('South lobe not found!!')
            lobes=[n.drop(columns=['Time [UTC]'])+
                   s.drop(columns=['Time [UTC]'])][0]
            lobes['Time [UTC]'] = t
            data_to_write.update({'ms_lobes_volume':lobes})
            data_to_write.update(post_proc_interface2(data_to_write,
                                                      type='volume'))
            if save_mesh:
                for var in ['beta_star','uB [J/Re^3]','Pth [J/Re^3]',
                      'KE [J/Re^3]','uHydro [J/Re^3]','Utot [J/Re^3]']:
                    usename = var.split(' ')[0]+'_cc'
                    eq = tp.data.operate.execute_equation
                    eq('{'+usename+'}={'+var+'}',zones=[region],
                            value_location=ValueLocation.CellCentered)
                    savemeshvars[kwargs.get('modes')[i]].append(usename)
    #timestamp
    ltime = time.time()-start_time
    print('VOL--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    ################################################################
    if integrate_line:
        for zone in kwargs.get('zonelist1D'):
            if type(zone) is not str:
                #integrate fluxes across the 1D curve or line
                print('\nWorking on: '+zone.name+' line')
                line_results = line_tools.line_analysis(zone,**kwargs)
                line_results['Time [UTC]'] = eventtime
                data_to_write.update({zone.name:line_results})
            else:
                print('\n'+zone+' line Empty!')
                line_results = pd.DataFrame()
                line_results['Time [UTC]'] = eventtime
                data_to_write.update({zone:line_results})
    ################################################################
    if kwargs.get('extract_flowline',False):
        for crossing in field_data.zones('flow_line*'):
            #upper = 31
            lower = 5
            cond = crossing.values('X *').as_numpy_array()>lower
            results = pd.DataFrame()
            for var in['X *','Y *','Z *','U_x *','U_y *','U_z *','P *',
                       'Rho *','B_x *','B_y *','B_z *',
                       'J_x *','J_y *','J_z *','1Ds *']:
                results[var.split(' ')[0]] = crossing.values(
                                                var).as_numpy_array()[cond]
            results['Time [UTC]'] = eventtime
            data_to_write.update({crossing.name:results})
    ################################################################
    if save_mesh:
        #save mesh to hdf file
        for state in enumerate(state_names):
            region = zonelist[0]
            meshvalues = pd.DataFrame()
            for var in [m for m in savemeshvars.values()][0]:
                usename = var.split(' ')[0]+'*'
                meshvalues[var] = region.values(usename).as_numpy_array()
            mp_mesh.update({region.name:meshvalues})
    ################################################################
    if write_data:
        datestring = ('{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(
                                            eventtime.year,eventtime.month,
                                            eventtime.day,eventtime.hour,
                                            eventtime.minute,eventtime.second))
        write_to_hdf(outputpath+'/energeticsdata/GM/energetics_'+
                        datestring+'.h5', data_to_write)
        if kwargs.get('save_surface_flux_dist',False):
            write_to_hdf(outputpath+'/fluxdistribution/GM/fluxdistribution__'+
                         datestring+'.h5', distribution_data)
    if save_mesh:
        write_to_hdf(outputpath+'/meshdata/mesh_'+datestring+'.h5',mp_mesh)
    if disp_result:
        display_progress(outputpath+'/meshdata/mesh_'+datestring+'.h5',
                            outputpath+'/energeticsdata/GM/energetics_'+
                            datestring+'.h5',
                            data_to_write.keys())
    #timestamp
    ltime = time.time()-start_time
    print('WRAPUP--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    return mp_mesh, data_to_write

if __name__ == "__main__":
    pass#TODO could make this a simple (!and fast) test function...
