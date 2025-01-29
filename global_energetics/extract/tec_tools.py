#!/usr/bin/env python3
"""Functions for identifying surfaces from field data
"""
#import logging as log
import os, warnings
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad, arctan2
import datetime as dt
import scipy.spatial as space
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#Interpackage modules
from global_energetics.extract.equations import (equations,rotation)
from global_energetics.extract import shue
from global_energetics.extract.shue import (r_shue, r0_alpha_1997,
                                                    r0_alpha_1998)

def standardize_vars(**kwargs):
    """Function attempts to standarize variable names for consistency
    Inputs
        kwargs:
            See dictionary for potential names to switch from
    """
    #Initialize and standardize variables
    known_swaps = {'/x':'X [R]',
                   '/y':'Y [R]',
                   '/z':'Z [R]',
                   '/ux':'U_x [km/s]',
                   '/uy':'U_y [km/s]',
                   '/uz':'U_z [km/s]',
                   '/Ux':'U_x [km/s]',
                   '/Uy':'U_y [km/s]',
                   '/Uz':'U_z [km/s]',
                   '/jx':'J_x [uA/m^2]',
                   '/jy':'J_y [uA/m^2]',
                   '/jz':'J_z [uA/m^2]',
                   '/Jx':'J_x [uA/m^2]',
                   '/Jy':'J_y [uA/m^2]',
                   '/Jz':'J_z [uA/m^2]',
                   '/bx':'B_x [nT]',
                   '/by':'B_y [nT]',
                   '/bz':'B_z [nT]',
                   '/Bx':'B_x [nT]',
                   '/By':'B_y [nT]',
                   '/Bz':'B_z [nT]',
                   '/b1x':'Bx_diff [nT]',
                   '/b1y':'By_diff [nT]',
                   '/b1z':'Bz_diff [nT]',
                   '/p':'P [nPa]',
                   '/P':'P [nPa]',
                   '/rho':'Rho [amu/cm^3]',
                   '/Rho':'Rho [amu/cm^3]',
                   '/status':'Status'}
    ds = tp.active_frame().dataset
    for var in ds.variable_names:
        ds.variable(var).name = kwargs.get(var,known_swaps[var])

def create_stream_zone(field_data, x1start, x2start, x3start,
                       zone_name, *, line_type=None, cart_given=False,
                       save=False):
    """Function to create a streamline, created in 2 directions from
       starting point
    Inputs
        field_data- Dataset class from tecplot with 3D field data
        x1start- starting position for streamline, default radius [R]
        x2start- starting position for streamline, default latitude [deg]
        x3start- starting position for streamline, default longitude [deg]
        zone_name
        line_type- day, north or south for determining stream direction
        cart_given- optional input for giving cartesian coord [x,y,z]
    Outputs
        streamzone
    """
    do2D = False
    if type(x3start)==type(None):
        x3start=0
        do2D=True
    if cart_given==False:
        # Get starting position in cartesian coordinates
        [x_start, y_start, z_start] = sph_to_cart(x1start, x2start,
                                                  x3start)
    else:
        x_start = x1start
        y_start = x2start
        z_start = x3start
    # Create streamline
    tp.active_frame().plot().show_streamtraces = True
    field_line = tp.active_frame().plot().streamtraces
    if line_type == 'south':
        if do2D:
            field_line.add(seed_point=[x_start, y_start],
                       stream_type=Streamtrace.TwoDLine,
                       direction=StreamDir.Reverse)
        else:
            field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Reverse)
    elif line_type == 'north':
        if do2D:
            field_line.add(seed_point=[x_start, y_start],
                       stream_type=Streamtrace.TwoDLine,
                       direction=StreamDir.Forward)
        else:
            field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Forward)
    else:
        if do2D:
            field_line.add(seed_point=[x_start, y_start],
                       stream_type=Streamtrace.TwoDLine,
                       direction=StreamDir.Both)
        else:
            field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Both)
    # Create zone
    field_line.extract()
    field_data.zone(-1).name = zone_name
    # Delete streamlines
    field_line.delete_all()


def check_streamline_closed(field_data, zone_name, r_seed, line_type,
                            **kwargs):
    """Function to check if a streamline is open or closed
    Inputs
        field_data- tecplot Dataset class with 3D field data
        zone_name
        r_seed [R]- position used to seed field line
        line_type- dayside, north or south from tail
        kwargs
    Outputs
        isclosed- boolean, True for closed
    """
    # Get starting and endpoints of streamzone
    r_values = field_data.zone(zone_name+'*').values('r *').as_numpy_array()
    if line_type == 'north':
        r_end_n = r_values[-1]
        r_end_s = 0
        r_seed = 2
    elif line_type == 'south':
        r_end_n = 0
        r_end_s = r_values[0]
        r_seed = 2
    elif line_type == 'inner_mag' or line_type=='inner_magXZ':
        xmax, xmin = field_data.zone(zone_name+'*').values('X *').minmax()
        zmax, zmin = field_data.zone(zone_name+'*').values('Z *').minmax()
        if line_type != 'inner_magXZ':
            ymax,ymin=field_data.zone(zone_name+'*').values('Y *').minmax()
            extrema = [[xmax, ymax, zmax],
                       [xmin, ymin, zmin]]
            defbounds = [[-220,-126,-126],
                      [31.5, 126, 126]]
        else:
            extrema = [[xmax, zmax],
                       [xmin, zmin]]
            defbounds = [[-220,-126],
                      [31.5, 126]]
        isclosed = True
        bounds = kwargs.get('bounds',defbounds)
        for minmax in enumerate(extrema):
            for value in enumerate(minmax[1]):
                if abs(value[1])>0.9*abs(bounds[minmax[0]][value[0]]):
                    isclosed = False
                    return isclosed
        return isclosed
    elif line_type == 'flowline':
        '''
        rmin = field_data.zone(zone_name+'*').values('r *').min()
        if rmin < r_seed:
            isclosed = True
            return isclosed
        else:
            isclosed = False
            return isclosed
        '''
        free_x = r_seed
        xmax = field_data.zone(zone_name+'*').values('X *').max()
        if xmax < free_x:
            solar_wind = False
            return not solar_wind
        else:
            solar_wind = True
            return not solar_wind
    else:
        r_end_n, r_end_s = r_values[0], r_values[-1]
    #check if closed
    if (r_end_n > r_seed) or (r_end_s > r_seed):
        isclosed = False
    else:
        isclosed = True
    return isclosed

def sph_to_cart(radius, lat, lon):
    """Function converts spherical coordinates to cartesian coordinates
    Inputs
        radius- radial position
        lat- latitude [deg]
        lon- longitude [deg]
    Outputs
        [x_pos, y_pos, z_pos]- list of x y z_pos coordinates
    """
    x_pos = (radius * cos(deg2rad(lat)) * cos(deg2rad(lon)))
    y_pos = (radius * cos(deg2rad(lat)) * sin(deg2rad(lon)))
    z_pos = (radius * sin(deg2rad(lat)))
    return [x_pos, y_pos, z_pos]
'''
def mag2gsm(x1, x2, x3, time, *, inputtype='sph'):
    """Function converts magnetic spherical coordinates to cartesian
        coordinates in GSM
    Inputs
        if inputtype='sph':
            x1,x2,x3- radius, latitue and longitude
        if inputtype='car':
            x1,x2,x3- x,y,z
        time- spacepy Ticktock object with time information
    Output
        xgsm, ygsm, zgsm- in form of 3 element np array
    """
    coordinates = coord.Coords([x1, x2, x3], 'SM',
                               inputtype)
    coordinates.ticks = time
    return coordinates.convert('GSM', 'car').data[0][0:3]

def sm2gsm_temp(radius, latitude, longitude, time):
    """Function converts solar magnetic spherical coordinates to cartesian
        coordinates in GSM
    Inputs
        radius
        latitude- relative to magnetic dipole axis
        longitude- relative to magnetic dipole axis
        time- spacepy Ticktock object with time information
    Output
        xgsm, ygsm, zgsm- in form of 3 element np array
    """
    xsm = radius * cos(deg2rad(latitude))*cos(deg2rad(longitude))
    ysm = radius * cos(deg2rad(latitude))*sin(deg2rad(longitude))
    zsm = radius * sin(deg2rad(latitude))
    #Read from file generated using spacepy, temp fix only
    dpdf = pd.read_csv('dipole_loc.csv')
    xdp = dpdf[dpdf['time'] == str(time.data[0])][' xgsm'].values[0]
    ydp = dpdf[dpdf['time'] == str(time.data[0])][' ygsm'].values[0]
    zdp = dpdf[dpdf['time'] == str(time.data[0])][' zgsm'].values[0]
    mu = np.arctan2(-xdp, np.sqrt(zdp**2+ydp**2))
    Tmat = rotation(-mu, 'y')
    #getTmat function give similar results but not identical
    #Tmat = getTmat(time)
    xgsm, ygsm, zgsm = np.matmul(Tmat, [[xsm], [ysm], [zsm]])
    return xgsm[0], ygsm[0], zgsm[0]
'''

def getTmat(time):
    """Function takes spacepy ticktock object and returns trans matrix
    Inputs
        time
    Output
        T4_t - transformation matrix for sm2gsm
    """
    '''
    y2k = dt.datetime(2000,1,1,12)
    y2ktick = spacetime.Ticktock(y2k, 'UTC')
    T0 = time.getUNX()[0] - y2ktick.getUNX()[0]
    '''
    MJD = time.getMJD()[0]
    T0 = (MJD-51544.5)/36525
    theta = deg2rad(100.461+36000.770*T0+15.04107)
    eps = deg2rad(23.439-0.013*T0)
    M = 357.528+35999.05*T0+0.04107
    lam = 280.460+36000.772*T0+0.04107
    lam_sun = deg2rad(lam+(1.915-0.0048*T0)*sin(M)+0.02*sin(2*M))
    T1 = rotation(theta, 'z')
    T2 = np.matmul(rotation(lam_sun, 'z'), rotation(eps, 'x'))
    xg = cos(deg2rad(287.45))*cos(deg2rad(80.25))
    yg = cos(deg2rad(287.45))*sin(deg2rad(80.25))
    zg = sin(deg2rad(287.45))
    T = np.matmul(T2, np.transpose(T1))
    xe, ye, ze = np.matmul(T, [[xg], [yg], [zg]])
    mu = np.arctan2(-xe,np.sqrt(ye**2+ze**2))
    T4_t = np.transpose(rotation(-mu[0], 'y'))
    return T4_t

def get_lateral_arc(zone, last_angle, last_r, method, *, dx=1):
    '''Function to calculate lateral angle, used to ensure coverage in the
        tail when seeding streams on the dayside
    Inputs
        zone- streamzone to probe values from
        last_angle
        last_r
        method- flow, day, tail, cps determines how to define r
    Outputs
        l_arc defined by dangle*mean_r
        rnew same as mean_r
        angle
    '''
    xzonevals = zone.values('X *').as_numpy_array()
    yzonevals = zone.values('Y *').as_numpy_array()
    zzonevals = zone.values('Z *').as_numpy_array()
    if method == 'dayside':
        dayindices = np.where((zzonevals>-dx/2)&
                              (zzonevals<dx/2))
        yave_day = yzonevals[dayindices].mean()
        xave_day = zzonevals[dayindices].mean()
        angle = np.rad2deg(np.arctan2(yave_day,xave_day))
        r = np.sqrt(yave_day**2+xave_day**2)
    if (method == 'flow') or (method == 'tail'):
        xzonevals = zone.values('X *').as_numpy_array()
        yzonevals = zone.values('Y *').as_numpy_array()
        zzonevals = zone.values('Z *').as_numpy_array()
        tail_indices = np.where((xzonevals>-10-dx/2)&
                                (xzonevals<-10+dx/2))
        yave_tail = yzonevals[tail_indices].mean()
        zave_tail = zzonevals[tail_indices].mean()
        angle = np.rad2deg(np.arctan2(zave_tail,yave_tail))
        r = np.sqrt(yave_tail**2+zave_tail**2)
    dangle = deg2rad(abs(angle-last_angle))
    mean_r = abs(r-last_r)/2
    return dangle * mean_r, mean_r, angle

def streamfind_bisection(field_data, method,
                         dimmax, nstream, rmax, rmin, itr_max, tolerance,
                         *, rcheck=7, time=None,
                         field_key_x='B_x*', global_key='global_field',
                         disp_search=False):
    """Generalized function for generating streamlines for a 'last closed'
        condition based on a bisection algorithm
    Inputs
        General
            field_data- Tecplot Dataset with flowfield information
            method- 'dayside', 'tail', 'flow', or 'plasmasheet'
        Parameters
            dimmmax- X location for tail/flow, maxextent for
                     dayside/plasmasheet
            nstream- number of streamlines to generate
            rmax, rmin- outer/inner bounds of search in whichever direction
            itr_max, tolerance- bisection alg settings
        Special
            rcheck- consideration for "closed" for 'flow' method only
            time- spacepy Ticktock of current time for 'plasmasheet' only
        Other
            field_key_x- string ID for x var to be traced, y&z assumed
            global_key- string ID for zone containing field data
    Outputs
       zonelist- list of indices of created zones
    """
    #validate method form
    approved_methods = ['dayside', 'tail', 'flow', 'plasmasheet',
                        'daysideXZ', 'inner_magXZ']
    reverse_if_flow = 0
    if all([test.find(method)==-1 for test in approved_methods]):
        print('WARNING: method for streamfind_bisection not recognized!!'+
              '\nno streamzones created')
        return None

    #set streamline seed positions in the fixed dimension
    if method == 'dayside':
        #set of points in z=0 plane from + to - dimmax angle from +x ax
        positions = np.linspace(-1*dimmax, dimmax, nstream)
        disp_message = 'Finding Magnetopause Dayside Field Lines'
        lin_type = 'dayside'
    elif (method == 'tail') or (method == 'flow'):
        #set of points 360deg around disc in the tail at x=dimmax
        positions = np.linspace(-180*(1-1/nstream), 180, nstream)
        if method == 'tail':
            disp_message = 'Finding Magnetopause Tail Field Lines'
        elif method == 'flow':
            disp_message = 'Finding Magnetopause Flow Lines'
            lin_type = 'flowline'
            reverse_if_flow = 1
    elif method == 'plasmasheet':
        #set of points on r=1Re at different longitudes N and S poles
        positions = np.append(np.linspace(-dimmax, -180,int(nstream/4)),
                              np.linspace(180, dimmax, int(nstream/4)))
        disp_message = 'Finding Plasmasheet Field Lines'
        lin_type = 'inner_mag'
    elif method == 'daysideXZ':
        positions = [0]
        disp_message = 'Tracing Bx Bz fieldlines'
        lin_type = 'dayside'
    elif method == 'inner_magXZ':
        positions = [0]
        disp_message = 'Tracing Bx Bz fieldlines'
        lin_type = method

    #set vector field
    field_key_y = field_key_x.split('x')[0]+'y'+field_key_x.split('x')[-1]
    field_key_z = field_key_x.split('x')[0]+'z'+field_key_x.split('x')[-1]
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable(field_key_x)
    plot.vector.v_variable = field_data.variable(field_key_y)
    if method != 'daysideXZ' and method != 'inner_magXZ':
        plot.vector.w_variable = field_data.variable(field_key_z)
    seedlist = []
    streamtrace = tp.active_frame().plot().streamtraces
    for a in positions:
        notfound = True
        #Set getseed function based only on search variable r
        #all getseeds return val: dim1, dim2, dim3,rcheck,lin_type
        if method == 'dayside':
            def getseed(r):return r, 0, a, r, lin_type
            cartesian = False
        elif method == 'tail':
            def getseed(r):
                x, y, z = dimmax, r*cos(deg2rad(a)), r*sin(deg2rad(a))
                if tp.data.query.probe_at_position(x, y, z)[0][7] < 0:
                    lin_type = 'south'
                else:
                    lin_type = 'north'
                return [dimmax, r*cos(deg2rad(a)), r*sin(deg2rad(a)),
                        rcheck, lin_type]
            cartesian = True
        elif method == 'flow':
            def getseed(r):
                return [dimmax, r*cos(deg2rad(a)), r*sin(deg2rad(a)),
                        rcheck, lin_type]
            cartesian = True
        elif method == 'plasmasheet':
            def getseed(r):return (sm2gsm_temp(1, r, a, time), None,
                                    lin_type)
            cartesian = True
        elif method == 'daysideXZ':
            plot.vector.v_variable = field_data.variable(field_key_z)
            def getseed(r):return r, 0, None, r, lin_type
            cartesian = True
        elif method=='inner_magXZ':
            plot.vector.v_variable = field_data.variable(field_key_z)
            def getseed(r):return r, 0, None, r, lin_type
            cartesian = True
        #Create initial max/min to lines
        x1, x2, x3, rchk, lin_type = getseed(rmax)
        if disp_search:
            print(x1,x2,x3,rchk,lin_type)
        create_stream_zone(field_data, x1, x2, x3,
                            'max line',
                            cart_given=cartesian)
        x1, x2, x3, rchk, lin_type = getseed(rmin)
        create_stream_zone(field_data, x1, x2, x3,
                            'min line',
                            cart_given=cartesian)
        #Check that last closed is bounded, delete min/max
        max_closed = check_streamline_closed(field_data, 'max*', rchk,
                                            line_type=lin_type)
        min_closed = check_streamline_closed(field_data, 'min*', rchk,
                                            line_type=lin_type)
        field_data.delete_zones(field_data.zone('min*'),
                                field_data.zone('max*'))
        if max_closed and min_closed:
            notfound = False
            if disp_search:
                print('Warning: line closed at {}'.format(rmax))
            x1, x2, x3, rchk, lin_type = getseed(rmax)
        elif not max_closed and not min_closed:
            notfound = False
            if disp_search:
                print('Warning: line open at {}'.format(rmin))
            x1, x2, x3, rchk, lin_type = getseed(rmin)
        else:
            rmid = (rmax+rmin)/2
            itr = 0
            rout = rmax
            rin = rmin
            #Enter bisection search algorithm
            while(notfound and itr < itr_max):
                #create mid
                x1, x2, x3, rchk, lin_type = getseed(rmid)
                create_stream_zone(field_data, x1, x2, x3,
                                    'mid line',
                                    cart_given=cartesian)
                #check midclosed
                mid_closed = check_streamline_closed(field_data,
                                                        'mid*', rchk,
                                                    line_type=lin_type)
                if mid_closed:
                    rin = rmid
                else:
                    rout = rmid
                if abs(rout-rin) < tolerance and (bool(
                                    int(mid_closed)-reverse_if_flow)):
                    #keep closed on boundary unless in 'flowline' method
                    notfound = False
                    field_data.delete_zones(field_data.zone(-1))
                else:
                    rmid = (rin+rout)/2
                    field_data.delete_zones(field_data.zone(-1))
                itr += 1
        seedlist.append([x1,x2,x3])
    #Regenerate all best find streamlines and extracting together
    for seed in seedlist:
        if cartesian == False:
            seed = sph_to_cart(seed[0],seed[1],seed[2])
        if disp_search:
            print(seed)
        if method!='daysideXZ' and method!='inner_magXZ':
            streamtrace.add(seed_point=seed,
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Both)
        else:
            streamtrace.add(seed_point=seed[0:2],
                       stream_type=Streamtrace.TwoDLine,
                       direction=StreamDir.Both)
    streamtrace.extract(concatenate=True)
    field_data.zone(-1).name = lin_type+'stream'
    zoneindex = field_data.zone(-1).index
    streamtrace.delete_all()
    return zoneindex

def dump_to_pandas(frame, zonelist, varlist, filename):
    """Function to hand zone data to pandas to do processing
    Inputs-
        frame- tecplot frame object to export
        zonelist- array like object of which zones to export
        varlist- which variables
        filename
    Outputs-
        loc_data- DataFrame of stream zone data
        x_max
    """
    frame.activate()
    with open(filename,'wb') as fileobject: fileobject.close()
    #Export 3D point data to csv file
    export_command=('VarNames:'+
                    'FrOp=1:'+
                    'ZnCount={:d}:'.format(len(zonelist))+
                    'ZnList=[')
    for zone in zonelist:
        export_command = (export_command+str(zone+1)+',')
    export_command = ','.join(export_command.split(',')[:-1])
    export_command = (export_command+
                    ']:VarCount={:d}:'.format(len(varlist))+
                    'VarList=[')
    for var in varlist:
        export_command = (export_command+str(var+1)+',')
    export_command = ','.join(export_command.split(',')[:-1])
    export_command = (export_command + ']:ValSep=",":'+
                    'FNAME="'+os.getcwd()+'/'+filename+'"')
    tp.macro.execute_extended_command(command_processor_id='excsv',
                                      command=export_command)
    loc_data = pd.read_csv('./'+filename)
    if any(col == 'x_cc' for col in loc_data.columns):
        loc_data = loc_data.drop(
                           columns=['Unnamed: {:d}'.format(len(varlist))])
        loc_data = loc_data.sort_values(by=['x_cc'])
        loc_data = loc_data.reset_index(drop=True)
        x_max = loc_data['x_cc'].max()
    else: x_max = []
    #Delete csv file
    os.remove(os.getcwd()+'/'+filename)
    return loc_data, x_max


def get_surface_velocity_estimate(field_data, currentindex, futureindex,*,
                                  nalpha=36, nphi=30, ntheta=30, nx=5):
    """Function finds the surface velocity given a single other timestep
    Inputs
        field_data- tecplot dataset object
        zone_name- name for the current zone
    """
    current_mesh, future_mesh = pd.DataFrame(), pd.DataFrame()
    eq = tp.data.operate.execute_equation
    eq('{x_cc}={X [R]}', value_location=ValueLocation.CellCentered)
    eq('{y_cc}={Y [R]}', value_location=ValueLocation.CellCentered)
    eq('{z_cc}={Z [R]}', value_location=ValueLocation.CellCentered)
    eq('{d_cc}=0', value_location=ValueLocation.CellCentered)
    eq('{Expansion_cc}=0', value_location=ValueLocation.CellCentered)
    eq('{SectorID}=0', value_location=ValueLocation.CellCentered)
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'CELLCENTERED')
    varnames = ['x_cc', 'y_cc', 'z_cc', 'Cell Volume']
    #load data from tecplot to pandas dataframes
    for var in varnames:
        current_mesh[var] = field_data.zone(currentindex.real
                        ).values(var.split(' ')[0]+'*').as_numpy_array()
        future_mesh[var] = field_data.zone(futureindex.real
                        ).values(var.split(' ')[0]+'*').as_numpy_array()
    #setup element spacings
    alphas, da= np.linspace(-pi, pi, nalpha, retstep=True, endpoint=False)
    phis, dphi = np.linspace(-pi/2, pi/2, nphi, retstep=True,
                                                endpoint=False)
    thetas, dtheta = np.linspace(-pi/2, pi/2, ntheta, retstep=True,
                                                      endpoint=False)
    x_s, dx = np.linspace(-20, 0, nx, retstep=True, endpoint=False)
    #add angles and radii
    flank_min_h_buff = 10
    for df in [current_mesh, future_mesh]:
        df['alpha'] = np.arctan2(df['z_cc'], df['y_cc'])
        df['phi'] = np.arctan2(df['y_cc'], df['x_cc'])
        df['theta'] = np.arctan2(df['z_cc'], df['x_cc'])
        df['h'] = np.sqrt(df['y_cc']**2+df['z_cc']**2)
        df['r'] = np.sqrt(df['x_cc']**2+df['y_cc']**2+df['z_cc']**2)
        flank_min_h = df[(df['x_cc']<0) &
                         (df['x_cc']>df['x_cc'].min()+
                                         flank_min_h_buff)]['h'].min()
        tailcond = (df['x_cc']==df['x_cc'].min()) | (
                   (df['x_cc']<-1) & (df['h']< flank_min_h))
        flankcond = (~ tailcond)
        df['flank'] = False
        df.at[flankcond,'flank'] = True
    #initialize distance column
    current_mesh['d'] = 0
    current_mesh['ExpR'] = 1
    current_mesh['ID'], future_mesh['ID'], k = 1, 0, 0
    #setup zones, cylindrical flank portion
    for a in alphas:
        for x in x_s:
            current_sector_ind = ((current_mesh['alpha']<a+da) &
                                  (current_mesh['alpha']>a) &
                                  (current_mesh['x_cc']>x) &
                                  (current_mesh['x_cc']<x+dx) &
                                  (current_mesh['flank']==True))
            future_sector_ind = ((future_mesh['alpha']<a+da) &
                                  (future_mesh['alpha']>a) &
                                  (future_mesh['x_cc']>x) &
                                  (future_mesh['x_cc']<x+dx) &
                                  (future_mesh['flank']==True))
            csector = current_mesh.loc[current_sector_ind][['x_cc',
                                                            'y_cc',
                                                            'z_cc',
                                                            'h']]
            fsector = future_mesh.loc[future_sector_ind][['x_cc',
                                                          'y_cc',
                                                          'z_cc',
                                                          'h']]
            current_mesh.at[current_sector_ind,'ID']=k
            future_mesh.at[future_sector_ind,'ID']=k
            k+=1
            cArea = current_mesh.loc[current_sector_ind][
                                                      'Cell Volume'].sum()
            fArea = future_mesh.loc[future_sector_ind][
                                                      'Cell Volume'].sum()
            cH = current_mesh.loc[current_sector_ind]['h'].mean()
            fH = future_mesh.loc[future_sector_ind]['h'].mean()
            outIn_sign = np.sign(fH-cH)
            #Calculate distance for each point in csector
            if (len(csector.values)>0) and (len(fsector.values)>0):
                if cArea==0:
                    expansion_ratio=0
                else:
                    expansion_ratio = fArea/cArea
                current_mesh.at[current_sector_ind,'ExpR']=expansion_ratio
                avedist = fsector['h'].mean()-csector['h'].mean()
                current_mesh.at[current_sector_ind,'d']=avedist
    #setup zones, cylindrical inside portion
    for a in alphas:
        for x in x_s:
            current_sector_ind = ((current_mesh['alpha']<a+da) &
                                  (current_mesh['alpha']>a) &
                                  (current_mesh['x_cc']>x) &
                                  (current_mesh['x_cc']<x+dx) &
                                  (current_mesh['flank']==False))
            future_sector_ind = ((future_mesh['alpha']<a+da) &
                                  (future_mesh['alpha']>a) &
                                  (future_mesh['x_cc']>x) &
                                  (future_mesh['x_cc']<x+dx) &
                                  (future_mesh['flank']==False))
            csector = current_mesh.loc[current_sector_ind][['x_cc',
                                                            'y_cc',
                                                            'z_cc',
                                                            'h']]
            fsector = future_mesh.loc[future_sector_ind][['x_cc',
                                                          'y_cc',
                                                          'z_cc',
                                                          'h']]
            current_mesh.at[current_sector_ind,'ID']=k
            future_mesh.at[future_sector_ind,'ID']=k
            k+=1
            cArea = current_mesh.loc[current_sector_ind][
                                                      'Cell Volume'].sum()
            fArea = future_mesh.loc[future_sector_ind][
                                                      'Cell Volume'].sum()
            cH = current_mesh.loc[current_sector_ind]['h'].mean()
            fH = future_mesh.loc[future_sector_ind]['h'].mean()
            outIn_sign = np.sign(cH-fH)
            #Calculate distance for each point in csector
            if (len(csector.values)>0) and (len(fsector.values)>0):
                if cArea==0:
                    expansion_ratio=0
                else:
                    expansion_ratio = fArea/cArea
                current_mesh.at[current_sector_ind,'ExpR']=expansion_ratio
                avedist = fsector['h'].mean()-csector['h'].mean()
                current_mesh.at[current_sector_ind,'d']=avedist
    #setup zones, semi_sphere section
    for phi in phis:
        for theta in thetas:
            current_sector_ind = ((current_mesh['theta']<theta+dtheta) &
                                  (current_mesh['theta']>theta) &
                                  (current_mesh['phi']>phi) &
                                  (current_mesh['phi']<phi+dphi))
            future_sector_ind = ((future_mesh['theta']<theta+dtheta) &
                             (future_mesh['theta']>theta) &
                             (future_mesh['phi']>phi) &
                             (future_mesh['phi']<phi+dphi))
            csector = current_mesh.loc[current_sector_ind][['x_cc',
                                                            'y_cc',
                                                            'z_cc',
                                                            'r']]
            fsector = future_mesh.loc[future_sector_ind][['x_cc',
                                                          'y_cc',
                                                          'z_cc',
                                                          'r']]
            current_mesh.at[current_sector_ind,'ID']=k
            future_mesh.at[future_sector_ind,'ID']=k
            k+=1
            cArea = current_mesh.loc[current_sector_ind][
                                                      'Cell Volume'].sum()
            fArea = future_mesh.loc[future_sector_ind][
                                                      'Cell Volume'].sum()
            cR = current_mesh.loc[current_sector_ind]['r'].mean()
            fR = future_mesh.loc[future_sector_ind]['r'].mean()
            outIn_sign = np.sign(fR-cR)
            #Calculate distance for each point in csector
            if (len(csector.values)>0) and (len(fsector.values)>0):
                if cArea==0:
                    expansion_ratio=0
                else:
                    expansion_ratio = fArea/cArea
                current_mesh.at[current_sector_ind,'ExpR']=expansion_ratio
                avedist = fsector['r'].mean()-csector['r'].mean()
                current_mesh.at[current_sector_ind,'d']=avedist
    #Transfer data back into tecplot
    field_data.zone(currentindex).values('d_cc')[::]=current_mesh[
                                                               'd'].values
    field_data.zone(currentindex).values('Expansion_cc')[::]=current_mesh[
                                                            'ExpR'].values
    field_data.zone(currentindex).values('SectorID')[::]=current_mesh[
                                                            'ID'].values
    field_data.zone(futureindex).values('SectorID')[::]=future_mesh[
                                                            'ID'].values

def pass_time_adjacent_variables(past,present,future,**kwargs):
    """Function passes variables between past-present-future zones
    NOTE: This assumes that all three zones have IDENTICAL STRUCTURE
    Inputs
        past, present, future
        kwargs
            analysis_type
    Returns
        None
    """
    eq = tp.data.operate.execute_equation
    # Get the variable indices
    past_index = str(past.index+1)
    future_index = str(future.index+1)
    # Create list of variables to pass
    variable_list = ['Status','daynight']
    if 'mag' in kwargs.get('analysis_type',''):
        variable_list.append('B_x [nT]')
        variable_list.append('B_y [nT]')
        variable_list.append('B_z [nT]')
    for variable in variable_list:
        eq('{past'+variable+'} = {'+variable+'}['+past_index+']',
                                                            zones=[present])
        eq('{future'+variable+'} = {'+variable+'}['+future_index+']',
                                                            zones=[present])

def get_surf_geom_variables(zone,**kwargs):
    """Function calculates variables for new zone based only on geometry,
        independent of what analysis will be performed on surface
    Inputs
        zone(Zone)- tecplot zone to calculate variables
    """
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    zonelist = kwargs.get('zonelist',[zone])
    if ('X Grid K Unit Normal' in zone.dataset.variable_names and
        zone.values('X Grid K Unit Normal').location != CC):
        #Delete the variables if theyre stuck as Nodal
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'X Grid K Unit Normal'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'Y Grid K Unit Normal'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'Z Grid K Unit Normal'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'Cell Area'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'surface_normal_x'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'surface_normal_y'))
        zone.dataset.delete_variables(zone.dataset.variable(
                                                      'surface_normal_z'))
    if 'X Grid K Unit Normal' not in zone.dataset.variable_names:
        #Get grid dependent variables
        tp.macro.execute_extended_command('CFDAnalyzer3',
                                          'CALCULATE FUNCTION = '+
                                          'GRIDKUNITNORMAL VALUELOCATION = '+
                                          'CELLCENTERED')
        tp.macro.execute_extended_command('CFDAnalyzer3',
                                          'CALCULATE FUNCTION = '+
                                          'CELLVOLUME VALUELOCATION = '+
                                          'CELLCENTERED')
        if kwargs.get('is1D',False):
            eq('{Cell Length}={Cell Volume}',zones=zonelist,value_location=CC)
        else:
            eq('{Cell Area} = {Cell Volume}',zones=zonelist,value_location=CC)
        zone.dataset.delete_variables([zone.dataset.variable('Cell Volume')])
    if ('Cell Area' in zone.dataset.variable_names and
                                         zone.values('Cell Area').max()==0.0):
        tp.macro.execute_extended_command('CFDAnalyzer3',
                                          'CALCULATE FUNCTION = '+
                                          'CELLVOLUME VALUELOCATION = '+
                                          'CELLCENTERED')
        if kwargs.get('is1D',False):
            eq('{Cell Length}={Cell Volume}',zones=zonelist,value_location=CC)
        else:
            eq('{Cell Area} = {Cell Volume}',zones=zonelist,value_location=CC)
        zone.dataset.delete_variables([zone.dataset.variable('Cell Volume')])
    #Generate cellcentered versions of postitional variables
    for var in ['X [R]','Y [R]','Z [R]','r [R]','Xd [R]','Zd [R]',
                'B_x [nT]','B_y [nT]','B_z [nT]','Bdx','Bdy','Bdz',
                'theta_1 [deg]','phi_1 [deg]','theta_2 [deg]','phi_2 [deg]',
                'daynight','Utot [J/Re^3]','Status']:
        if var in zone.dataset.variable_names:
            newvar = var.split(' ')[0].lower()+'_cc'
            if (newvar in zone.dataset.variable_names and
                zone.values(newvar).location != CC):
                #Delete the variable if its stuck as Nodal
                print('deleting ',newvar)
                zone.dataset.delete_variables(zone.dataset.variable(newvar))
            if newvar not in zone.dataset.variable_names:
                eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                             zones=zonelist)
            elif zone.values(newvar).passive:
                eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                             zones=zonelist)
    for var in [v for v in zone.dataset.variable_names if ('past' in v or
                                                           'future' in v) and
                                                           '_cc' not in v]:
        newvar = var.split(' ')[0].lower()+'_cc'
        eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                        zones=zonelist)
    #Create a DataFrame for easy manipulations
    x_ccvalues =zone.values('x_cc').as_numpy_array()
    xnormals = zone.values('X Grid K Unit Normal').as_numpy_array()
    df = pd.DataFrame({'x_cc':x_ccvalues,'normal':xnormals})
    #Check that surface normals are pointing outward from surface
    #Spherical inner boundary surface case (want them to point inwards)
    if kwargs.get('innerbound',False):
    #if 'innerbound' in zone.name:
        if df[df['x_cc']==df['x_cc'].min()]['normal'].mean() < 0:
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    elif 'bs' in zone.name:
        #Look at dayside max plane for bowshock
        if (len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']>0)]) <
            len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']<0)])):
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    elif 'xslice' in zone.name:
        #These ones should be assigned a normal on creation (flat plane)
        eq('{surface_normal_x} = {X Grid K Unit Normal}',
            zones=zonelist, value_location=CC)
        eq('{surface_normal_y} = {Y Grid K Unit Normal}',
            zones=zonelist, value_location=CC)
        eq('{surface_normal_z} = {Z Grid K Unit Normal}',
            zones=zonelist, value_location=CC)
    elif 'ocflb' in zone.name:
        #Look at the extreme +X location where the polar cap looks nice
        if (len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']<0)]) >
            len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']>0)])):
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    elif 'ms_closed' in zone.name or 'mp_iso_betastar' in zone.name:
        #Look at forward set of points which should be roughly planar
        if (len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']>0)]) >
            len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']<0)])):
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    elif 'ms_plasmasheet' in zone.name:
        y_cc =zone.values('y_cc').as_numpy_array()
        #Look to see if things point "up" generally
        if (len(y_cc[y_cc>0]) > len(y_cc[y_cc<0])):
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    else:
        #Look at tail cuttoff plane for other cases
        if (len(df[(df['x_cc']==df['x_cc'].min())&(df['normal']>0)]) >
            len(df[(df['x_cc']==df['x_cc'].min())&(df['normal']<0)])):
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=zonelist, value_location=CC)
    if ('mp' in zone.name) and ('innerbound' not in zone.name):
        #Store a helpful 'htail' value in aux data for potential later use
        xvals = zone.values('X *').as_numpy_array()
        hvals = zone.values('h').as_numpy_array()
        zone.aux_data.update({'hmin':
               hvals[np.where(np.logical_and(xvals<-5,xvals>-8))].min()})

def general_rotation(vec1,vec2):
    r1 = np.sqrt(vec1[0]**2+vec1[1]**2+vec1[2]**2)
    r2 = np.sqrt(vec2[0]**2+vec2[1]**2+vec2[2]**2)
    axis = np.cross(vec1,vec2)
    axis_mag = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
    ax, ay, az = axis/axis_mag
    sn = axis_mag/(r1*r2)
    cs = np.dot(vec1,vec2)/(r1*r2)
    R = np.array(
              [[cs+ax**2*(1-cs), ax*ay*(1-cs)-az*sn, ax*az*(1-cs)+ay*sn],
               [ay*ax*(1-cs)+az*sn, cs+ay**2*(1-cs), ay*az*(1-cs)-ax*sn],
               [az*ax*(1-cs)-ay*sn, az*ay*(1-cs)+ax*sn, cs+az**2*(1-cs)]]
                )
    return R

def croissant_trace(spherezone,**kwargs):
    """The polar cap can sometimes look like a croissant with extrusions of
        the nightside OCFLB existing forward of the dipole terminator line.
        This function will create a new theta/phi magnetic mapping variable
        for each hemisphere relative to the polar cap centroid, rather than
        the dipole center. This corrects identification of the
        dayside/nightside boundaries.
    Inputs
        spherezone (Zone) - Tecplot zone we want the map variables on
        kwargs:
    Returns
        map_limits (dict{str:list[float,float]}) - new variable values of the
                                                 original terminator ymax/min
            keys:   theta_limits_north
                    theta_limits_south
                    phi_limits_north
                    phi_limits_south
    """
    map_limits = {}
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    # Pull XYZ, Status, and area values from the sphere to find the centroid
    all_x = spherezone.values('xd_cc').as_numpy_array()
    all_y = spherezone.values('y_cc').as_numpy_array()
    all_z = spherezone.values('zd_cc').as_numpy_array()
    status = spherezone.values('status_cc').as_numpy_array()
    areas = spherezone.values('Cell Area').as_numpy_array()
    # Filter the Values according to Status
    areasnorth = areas[(status==2)]
    xnorth = all_x[(status==2)]
    ynorth = all_y[(status==2)]
    znorth = all_z[(status==2)]
    areassouth = areas[(status==1)]
    xsouth = all_x[(status==1)]
    ysouth = all_y[(status==1)]
    zsouth = all_z[(status==1)]
    # Calculate the centroid as: Cxyz = Sum(Pxyz*Area)/Sum(Area)
    X_north = np.sum(xnorth*areasnorth)/np.sum(areasnorth)
    Y_north = np.sum(ynorth*areasnorth)/np.sum(areasnorth)
    Z_north = np.sum(znorth*areasnorth)/np.sum(areasnorth)
    X_south = np.sum(xsouth*areassouth)/np.sum(areassouth)
    Y_south = np.sum(ysouth*areassouth)/np.sum(areassouth)
    Z_south = np.sum(zsouth*areassouth)/np.sum(areassouth)
    # Find the rotation matrix R using positions of centroid and pole
    radius_north = np.sqrt(X_north**2+Y_north**2+Z_north**2)
    radius_south = np.sqrt(X_south**2+Y_south**2+Z_south**2)
    pole_north = [0,0,radius_north]
    pole_south = [0,0,-radius_south]
    centroid_north = [X_north,Y_north,Z_north]
    centroid_south = [X_south,Y_south,Z_south]
    R_north = general_rotation(centroid_north,pole_north)
    R_south = general_rotation(centroid_south,pole_south)
    # Then each footpoint into r=1 spherical coordinates
    eq('{xfoot_1}=sin(-{theta_1 [deg]}*pi/180+pi/2)*cos({phi_1 [deg]}*pi/180)')
    eq('{yfoot_1}=sin(-{theta_1 [deg]}*pi/180+pi/2)*sin({phi_1 [deg]}*pi/180)')
    eq('{zfoot_1}=cos(-{theta_1 [deg]}*pi/180+pi/2)')
    eq('{xfoot_2}=sin(-{theta_2 [deg]}*pi/180+pi/2)*cos({phi_2 [deg]}*pi/180)')
    eq('{yfoot_2}=sin(-{theta_2 [deg]}*pi/180+pi/2)*sin({phi_2 [deg]}*pi/180)')
    eq('{zfoot_2} = 1*cos(-{theta_2 [deg]}*pi/180+pi/2)')
    status_node = spherezone.values('Status').as_numpy_array()
    xnorth_node = spherezone.values('Xd *').as_numpy_array()[(status_node==2)]
    ynorth_node = spherezone.values('Y *').as_numpy_array()[(status_node==2)]
    xsouth_node = spherezone.values('Xd *').as_numpy_array()[(status_node==1)]
    ysouth_node = spherezone.values('Y *').as_numpy_array()[(status_node==1)]
    xfoot_north=spherezone.values('xfoot_1').as_numpy_array()[(status_node==2)]
    yfoot_north=spherezone.values('yfoot_1').as_numpy_array()[(status_node==2)]
    zfoot_north=spherezone.values('zfoot_1').as_numpy_array()[(status_node==2)]
    xfoot_south=spherezone.values('xfoot_2').as_numpy_array()[(status_node==1)]
    yfoot_south=spherezone.values('yfoot_2').as_numpy_array()[(status_node==1)]
    zfoot_south=spherezone.values('zfoot_2').as_numpy_array()[(status_node==1)]
    # Store the cross polar cap dipole terminator points
    iterm_min_north = ynorth_node==ynorth_node[abs(xnorth_node)<0.5].max()
    iterm_max_north = ynorth_node==ynorth_node[abs(xnorth_node)<0.5].min()
    iterm_min_south = ysouth_node==ysouth_node[abs(xsouth_node)<0.5].max()
    iterm_max_south = ysouth_node==ysouth_node[abs(xsouth_node)<0.5].min()
    # Parse out some strings to make it easier to read here
    xnew_1 = ('('+str(R_north[0][0])+'*{xfoot_1}+'+
                  str(R_north[0][1])+'*{yfoot_1}+'+
                  str(R_north[0][2])+'*{zfoot_1})')
    ynew_1 = ('('+str(R_north[1][0])+'*{xfoot_1}+'+
                  str(R_north[1][1])+'*{yfoot_1}+'+
                  str(R_north[1][2])+'*{zfoot_1})')
    znew_1 = ('('+str(R_north[2][0])+'*{xfoot_1}+'+
                  str(R_north[2][1])+'*{yfoot_1}+'+
                  str(R_north[2][2])+'*{zfoot_1})')

    xnew_2 = ('('+str(R_south[0][0])+'*{xfoot_2}+'+
                  str(R_south[0][1])+'*{yfoot_2}+'+
                  str(R_south[0][2])+'*{zfoot_2})')
    ynew_2 = ('('+str(R_south[1][0])+'*{xfoot_2}+'+
                  str(R_south[1][1])+'*{yfoot_2}+'+
                  str(R_south[1][2])+'*{zfoot_2})')
    znew_2 = ('('+str(R_south[2][0])+'*{xfoot_2}+'+
                  str(R_south[2][1])+'*{yfoot_2}+'+
                  str(R_south[2][2])+'*{zfoot_2})')
    # Calculate the new theta and phi positions
    eq('{theta_centroid_1} = if({theta_1 [deg]}<0,{theta_1 [deg]},'+
                               '(-acos('+znew_1+')+pi/2)*180/pi)')
    eq('{phi_centroid_1} = if({phi_1 [deg]}<0,{phi_1 [deg]},'+
                               '-atan2('+ynew_1+',-'+xnew_1+')*180/pi+180)')
    eq('{theta_centroid_2} = if({theta_2 [deg]}<0,{theta_2 [deg]},'+
                               '(-acos('+znew_2+')+pi/2)*180/pi)')
    eq('{phi_centroid_2} = if({phi_2 [deg]}<0,{phi_2 [deg]},'+
                               '-atan2('+ynew_2+',-'+xnew_2+')*180/pi+180)')
    theta_centr_north =spherezone.values('theta_centroid_1').as_numpy_array()[
                                                             (status_node==2)]
    phi_centr_north = spherezone.values('phi_centroid_1').as_numpy_array()[
                                                             (status_node==2)]
    map_limits['theta_min_north'] = theta_centr_north[iterm_min_north][0]
    map_limits['theta_max_north'] = theta_centr_north[iterm_max_north][0]
    map_limits['phi_min_north'] = phi_centr_north[iterm_min_north][0]
    map_limits['phi_max_north'] = phi_centr_north[iterm_max_north][0]

    theta_centr_south =spherezone.values('theta_centroid_2').as_numpy_array()[
                                                             (status_node==1)]
    phi_centr_south = spherezone.values('phi_centroid_2').as_numpy_array()[
                                                             (status_node==1)]
    map_limits['theta_min_south'] = theta_centr_south[iterm_min_south][0]
    map_limits['theta_max_south'] = theta_centr_south[iterm_max_south][0]
    map_limits['phi_min_south'] = phi_centr_south[iterm_min_south][0]
    map_limits['phi_max_south'] = phi_centr_south[iterm_max_south][0]
    return map_limits

def get_daymapped_nightmapped(zone,**kwargs):
    reversed_mapping(zone,1)

def SECOND_get_daymapped_nightmapped(zone,**kwargs):
    phi_1min = zone.dataset.zone(0).aux_data['phi_min_north']
    phi_1max = zone.dataset.zone(0).aux_data['phi_max_north']
    phi_2min = zone.dataset.zone(0).aux_data['phi_min_south']
    phi_2max = zone.dataset.zone(0).aux_data['phi_max_south']
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    if 'lobe' in zone.name:
        if 'nlobe' in zone.name:
            eq('{daymapped_nlobe_cc}=IF'+
               '({phi_centroid_1}>='+phi_1max+'||'+
                '({phi_centroid_1}<='+phi_1min+'&&{phi_centroid_1}>=0),1,0)',
                                            value_location=CC,zones=[zone])
            eq('{nightmapped_nlobe_cc}=IF'+
                        '({phi_centroid_1}<'+phi_1max+
                       '&&{phi_centroid_1}>'+phi_1min+',1,0)',
                                            value_location=CC,zones=[zone])
        elif 'slobe' in zone.name:
            eq('{daymapped_slobe_cc}=IF'+
                        '({phi_centroid_2}>='+phi_2max+'||'+
                        '({phi_centroid_2}<='+phi_2min+
                              '&&{phi_centroid_2}>=0),1,0)',
                                            value_location=CC,zones=[zone])
            eq('{nightmapped_slobe_cc}=IF'+
                        '({phi_centroid_2}<'+phi_2max+
                       '&&{phi_centroid_2}>'+phi_2min+',1,0)',
                                            value_location=CC,zones=[zone])
    elif 'mp' in zone.name or 'closed' in zone.name:
        eq('{daymapped_cc}=IF'+
                    '(({phi_centroid_1}>='+phi_1max+'||'+
                        '({phi_centroid_1}<='+phi_1min+
                       '&&{phi_centroid_1}>=0))||'+
                            '({phi_centroid_2}>='+phi_2max+'||'+
                            '({phi_centroid_2}<='+phi_2min+
                           '&&{phi_centroid_2}>=0)),1,0)',
                                            value_location=CC,zones=[zone])
        eq('{nightmapped_cc}=IF'+
                        '(({phi_centroid_1}<'+phi_1max+
                        '&&{phi_centroid_1}>'+phi_1min+')&&'+
                             '({phi_centroid_2}<'+phi_2max+
                            '&&{phi_centroid_2}>'+phi_2min+'),1,0)',
                                            value_location=CC,zones=[zone])
    elif 'global' in zone.name:
        if 'NLobe' in kwargs.get('state_var').name:
            eq('{daymapped_nlobe}=IF'+
               '({phi_centroid_1}>'+phi_1max+'||'+
                '({phi_centroid_1}<'+phi_1min+
               '&&{phi_centroid_1}>0),1,0)',zones=[zone])
            eq('{nightmapped_nlobe}=IF'+
                        '({phi_centroid_1}<'+phi_1max+
                       '&&{phi_centroid_1}>'+phi_1min+',1,0)',
                                                           zones=[zone])
        elif 'SLobe' in kwargs.get('state_var').name:
            eq('{daymapped_slobe}=IF'+
                        '({phi_centroid_2}>'+phi_2max+'||'+
                               '({phi_centroid_2}<'+phi_2min+
                              '&&{phi_centroid_2}>0),1,0)',
                                                           zones=[zone])
            eq('{nightmapped_slobe}=IF'+
                        '({phi_centroid_2}<'+phi_2max+
                       '&&{phi_centroid_2}>'+phi_2min+',1,0)',
                                                           zones=[zone])
        else:
            if 'mp' in kwargs.get('state_var').name:
                targetname = 'mp_iso_betastar'
            elif 'lcb' in kwargs.get('state_var').name:
                targetname = 'closed'
            eq('{daymapped'+targetname+'}=IF'+
                    '(({phi_centroid_1}>='+phi_1max+'||'+
                        '({phi_centroid_1}<='+phi_1min+
                       '&&{phi_centroid_1}>=0))||'+
                            '({phi_centroid_2}>='+phi_2max+'||'+
                            '({phi_centroid_2}<='+phi_2min+
                           '&&{phi_centroid_2}>=0)),1,0)',
                                                            zones=[zone])
            eq('{nightmapped'+targetname+'}=IF'+
                        '(({phi_centroid_1}<'+phi_1max+
                        '&&{phi_centroid_1}>'+phi_1min+')&&'+
                             '({phi_centroid_2}<'+phi_2max+
                            '&&{phi_centroid_2}>'+phi_2min+'),1,0)',
                                                            zones=[zone])
    elif 'ionosphere' in zone.name:
        eq('{daymapped} = IF({Xd [R]}>0,1,0)',zones=[zone])
        eq('{nightmapped} = IF({Xd [R]}<0,1,0)',zones=[zone])

def OLD_get_daymapped_nightmapped(zone,**kwargs):
    """Function assigns variables to represent day and night mapped,
    Inputs
        zone(Zone)- tecplot Zone object to do calculation
    """
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    if 'lobe' in zone.name:
        if 'nlobe' in zone.name:
            eq('{daymapped_nlobe_cc}=IF'+
               '({phi_1 [deg]}>270||'+
                '({phi_1 [deg]}<90&&{phi_1 [deg]}>0),1,0)',
                                            value_location=CC,zones=[zone])
            eq('{nightmapped_nlobe_cc}=IF'+
                        '({phi_1 [deg]}<270&&{phi_1 [deg]}>90,1,0)',
                                            value_location=CC,zones=[zone])
        elif 'slobe' in zone.name:
            eq('{daymapped_slobe_cc}=IF'+
                        '({phi_2 [deg]}>270||'+
                               '({phi_2 [deg]}<90&&{phi_2 [deg]}>0),1,0)',
                                            value_location=CC,zones=[zone])
            eq('{nightmapped_slobe_cc}=IF'+
                        '({phi_2 [deg]}<270&&{phi_2 [deg]}>90,1,0)',
                                            value_location=CC,zones=[zone])
    elif 'mp' in zone.name or 'closed' in zone.name:
        eq('{daymapped_cc}=IF'+
                    '(({phi_1 [deg]}>=270||'+
                        '({phi_1 [deg]}<=90&&{phi_1 [deg]}>=0))||'+
                            '({phi_2 [deg]}>=270||'+
                            '({phi_2 [deg]}<=90&&{phi_2 [deg]}>=0)),1,0)',
                                            value_location=CC,zones=[zone])
        eq('{nightmapped_cc}=IF'+
                        '(({phi_1 [deg]}<270&&{phi_1 [deg]}>90)&&'+
                             '({phi_2 [deg]}<270&&{phi_2 [deg]}>90),1,0)',
                                            value_location=CC,zones=[zone])
    elif 'global' in zone.name:
        if 'NLobe' in kwargs.get('state_var').name:
            eq('{daymapped_nlobe}=IF'+
               '({phi_1 [deg]}>270||'+
                '({phi_1 [deg]}<90&&{phi_1 [deg]}>0),1,0)',zones=[zone])
            eq('{nightmapped_nlobe}=IF'+
                        '({phi_1 [deg]}<270&&{phi_1 [deg]}>90,1,0)',
                                                           zones=[zone])
        elif 'SLobe' in kwargs.get('state_var').name:
            eq('{daymapped_slobe}=IF'+
                        '({phi_2 [deg]}>270||'+
                               '({phi_2 [deg]}<90&&{phi_2 [deg]}>0),1,0)',
                                                           zones=[zone])
            eq('{nightmapped_slobe}=IF'+
                        '({phi_2 [deg]}<270&&{phi_2 [deg]}>90,1,0)',
                                                           zones=[zone])
        else:
            if 'mp' in kwargs.get('state_var').name:
                targetname = 'mp_iso_betastar'
            elif 'lcb' in kwargs.get('state_var').name:
                targetname = 'closed'
            eq('{daymapped'+targetname+'}=IF'+
                    '(({phi_1 [deg]}>=270||'+
                        '({phi_1 [deg]}<=90&&{phi_1 [deg]}>=0))||'+
                            '({phi_2 [deg]}>=270||'+
                            '({phi_2 [deg]}<=90&&{phi_2 [deg]}>=0)),1,0)',
                                                            zones=[zone])
            eq('{nightmapped'+targetname+'}=IF'+
                        '(({phi_1 [deg]}<270&&{phi_1 [deg]}>90)&&'+
                             '({phi_2 [deg]}<270&&{phi_2 [deg]}>90),1,0)',
                                                            zones=[zone])
    elif 'ionosphere' in zone.name:
        eq('{daymapped} = IF({Xd [R]}>0,1,0)',zones=[zone])
        eq('{nightmapped} = IF({Xd [R]}<0,1,0)',zones=[zone])

def get_day_flank_tail(zone,**kwargs):
    """Function assigns variables to represent dayside, flank, and tail,
        typically for the magnetopause zone
    Inputs
        zone(Zone)- tecplot Zone object to do calculation
    """
    eq, cc = tp.data.operate.execute_equation, ValueLocation.CellCentered
    #Check that geometry variables have already been calculated
    if 'mp' in zone.name:
        tail_h = float(zone.dataset.zone('mp*').aux_data['hmin'])
        eq('{Day} = IF({X [R]}>0,1,0)', zones=[zone.index])
        eq('{Tail} = IF(({X [R]}<-5&&{h}<'+str(tail_h)+'*0.8)||'+
             '({X [R]}<-10&&{h}<'+str(tail_h)+')||'+
             '(abs({X [R]}-'+str(zone.values('X *').min())+')<0.5)'+
             ',1,0)',zones=[zone.index],value_location=cc)
        eq('{Flank} = IF({Day}==0&&{Tail}==0,1,0)', zones=[zone.index])
    #TODO check this visually
    elif 'lobe' in zone.name or 'closed' in zone.name:
        eq('{Tail} = IF(abs({X [R]}-'+str(zone.values('X *').min())+
                          ')<0.5,1,0)',zones=[zone.index],value_location=cc)
    '''
    elif 'global' in zone.name:
        statevalues = zone.values(kwargs.get('state_var').name
                                  ).as_numpy_array()
        X=zone.values('x_cc').as_numpy_array()[np.where(statevalues==1)].min()
        if 'mp' in kwargs.get('state_var').name:
            h = zone.values('h').as_numpy_array()[np.where(statevalues==1)]
            tail_h = float(zone.dataset.zone('mp*').aux_data['hmin'])
            eq('{Tail} = IF(({'+kwargs.get('state_var').name+'}==1)&&'+
               '({X [R]}<-5&&{h}<'+str(tail_h)+'*0.8)||'+
                '({X [R]}<-10&&{h}<'+str(tail_h)+')||'+
                '(abs({X [R]}-'+str(X)+')<0.5)'+
                ',1,0)',zones=[zone.index],value_location=cc)
        elif ('Lobe' in kwargs.get('state_var').name or
              'lcb' in kwargs.get('state_var').name):
            eq('{Tail} = IF(({'+kwargs.get('state_var').name+'}==1)&&'+
               'abs({X [R]}-'+str(X)+')<0.5,1,0)',zones=[zone.index],
                                                    value_location=cc)
    '''
    '''
    ##############################################################
    #Day, flank, tail definitions
    #2-dayside
    #1-flank
    #0-tail
    eq('{Day} = IF({x_cc}>0,1,0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
    ds = tp.active_frame().dataset
    h_vals = ds.zone(zone.index).values('h').as_numpy_array()
    x_vals = ds.zone(zone.index).values('X*').as_numpy_array()
    hmin =h_vals[np.where(np.logical_and(x_vals<-5,x_vals>-10))].min()
    eq('{Tail} = IF({X [R]}<-5 && '+
                     '({surface_normal_x}<-.8 || {h}<'+str(hmin)+'),1,0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
    eq('{Flank} = IF({Day}==0 && {Tail}==0, 1, 0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
    eq('{DayFlankTail} = IF({Day}==1,2,IF({Flank}==1,1,0))',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
    '''
    pass

def get_surface_variables(zone, analysis_type, **kwargs):
    """Function calculated variables for a specific 3D surface
    Inputs
        zone(Zone)- tecplot Zone object to expand data
        kwargs:
            find_DFT(bool) - False,
            do_cms(bool) - False,
            dt(float) - 60
            surface_unevaluated_type (str)- creates variables here as if it
                                            were that analysis type, only
                                            read if analysis_type==''
    """
    eq,CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    zonelist = kwargs.get('zonelist',[zone])
    #Check that geometry variables have already been calculated
    assert any([x!=0 for x in
        zone.values('surface_normal_x').as_numpy_array()]), ('Surface '+
                                              'geometry not calculated!')
    if analysis_type=='' and 'surface_unevaluated_type' in kwargs:
        assert('Utot [J/Re^3]' in zone.dataset.variable_names,
                            'Cannot set unevaluated surface analysis type: ',
                             kwargs.get('surface_unevaluated_type'),
                            ' without associated variables!')
        analysis_type = kwargs.get('surface_unevaluated_type')
    #Throw-away variables, these will be overwritten each time
    #eq('{status_cc}={Status}',value_location=ValueLocation.CellCentered,
    #                               zones=[zone])
    '''
    ##Throw-away variables, these will be overwritten each time
    eq('{ux_cc}={U_x [km/s]}', value_location=ValueLocation.CellCentered,
                               zones=[zone])
    eq('{uy_cc}={U_y [km/s]}', value_location=ValueLocation.CellCentered,
                               zones=[zone])
    eq('{uz_cc}={U_z [km/s]}', value_location=ValueLocation.CellCentered,
                               zones=[zone])
    if 'energy' in analysis_type or analysis_type == 'all':
        eq('{Utot_cc}={Utot [J/Re^3]}',
                         value_location=ValueLocation.CellCentered)
    eq('{Bx_cc}={B_x [nT]}', value_location=ValueLocation.CellCentered,
                             zones=[zone])
    eq('{By_cc}={B_y [nT]}', value_location=ValueLocation.CellCentered,
                             zones=[zone])
    eq('{Bz_cc}={B_z [nT]}', value_location=ValueLocation.CellCentered,
                             zones=[zone])
    '''
    '''
    REMOVE THIS
    if ('status_cc' in zone.dataset.variable_names):
        if (zone.values('status_cc').minmax()==(0.0,0.0)):
            eq('{status_cc}={Status}',value_location=ValueLocation.CellCentered,
                                   zones=[zone])
    else:
        eq('{status_cc}={Status}',value_location=ValueLocation.CellCentered,
                                   zones=[zone])
    REMOVE THIS
    '''
    #eq('{W_cc}={W [km/s/Re]}', value_location=ValueLocation.CellCentered)
    #eq('{W_cc}=0', value_location=ValueLocation.CellCentered,
    #               zones=[zone.index])

    if (('mp' in zone.name and 'innerbound' not in zone.name) or
                                            kwargs.get('find_DFT',False)):
        get_day_flank_tail(zone)
        #get_day_flank_tail(zone.dataset.zone(0))
    '''
    dt = str(dt)
    Old surface velocity calculation
        #If x<0 dot n with YZ vector, else dot with R(XYZ) vector
        eq('{Csurface_x} = IF({x_cc}<0,'+
                '0,'+
        '{x_cc}/sqrt({x_cc}**2+{y_cc}**2+{z_cc}**2))'+
            '*6371*{d_cc}/'+dt+'*(1+{Expansion_cc})/2',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
    '''
    if 'virial' in analysis_type:
        eq('{Bdx_cc}={Bdx}', value_location=ValueLocation.CellCentered,
                             zones=zonelist)
        eq('{Bdy_cc}={Bdy}', value_location=ValueLocation.CellCentered,
                             zones=zonelist)
        eq('{Bdz_cc}={Bdz}', value_location=ValueLocation.CellCentered,
                             zones=zonelist)
        get_virials()
    ##Different prefixes allow for calculation of surface fluxes using 
    #   multiple sets of flowfield variables (denoted by the prefix)
    if 'energy' in analysis_type:
        if kwargs.get('do_1Dsw',False):
            prefixlist = ['1D']
        else:
            prefixlist = ['']
        for add in prefixlist:
            ##############################################################
            #Velocity normal to the surface
            eq('{Unorm [km/s]}=sqrt(({U_x [km/s]}*{surface_normal_x})**2+'+
                                '({U_y [km/s]}*{surface_normal_y})**2+'+
                                '({U_z [km/s]}*{surface_normal_z})**2)',
                            value_location=ValueLocation.CellCentered,
                            zones=zonelist)
            ##############################################################
            #Normal Poynting Flux
            eq('{'+add+'ExB_net [W/Re^2]}='+
                        '{'+add+'Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                                  '{'+add+'U_x [km/s]}*{surface_normal_x}'+
                                 '+{'+add+'U_y [km/s]}*{surface_normal_y}'+
                                 '+{'+add+'U_z [km/s]}*{surface_normal_z})-'+
                '({'+add+'B_x [nT]}*({U_x [km/s]})+'+
                 '{'+add+'B_y [nT]}*({U_y [km/s]})+'+
                 '{'+add+'B_z [nT]}*({U_z [km/s]}))'+
                                  '*({'+add+'B_x [nT]}*{surface_normal_x}+'+
                                    '{'+add+'B_y [nT]}*{surface_normal_y}+'+
                                    '{'+add+'B_z [nT]}*{surface_normal_z})'+
                                            '/(4*pi*1e-7)*1e-9*6371**2',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'ExB_escape [W/Re^2]} = max({'+add+'ExB_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'ExB_injection [W/Re^2]} = min({'+add+'ExB_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            ##############################################################
            #Normal Total Pressure Flux
            eq('{'+add+'P0_net [W/Re^2]} = '+
                    '(1/2*{'+add+'Dp [nPa]}+2.5*{'+add+'P [nPa]})*6371**2*('+
                                    '{'+add+'U_x [km/s]}*{surface_normal_x}'+
                                   '+{'+add+'U_y [km/s]}*{surface_normal_y}'+
                                   '+{'+add+'U_z [km/s]}*{surface_normal_z})',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'P0_escape [W/Re^2]} ='+
                        'max({'+add+'P0_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'P0_injection [W/Re^2]} ='+
                        'min({'+add+'P0_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            ##############################################################
            #Normal Total Energy Flux
            eq('{'+add+'K_net [W/Re^2]}='+
                   '{'+add+'P0_net [W/Re^2]}+{'+add+'ExB_net [W/Re^2]}',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'K_escape [W/Re^2]}=max({'+add+'K_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'K_injection [W/Re^2]} ='+
                         'min({'+add+'K_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
        if 'wave' in analysis_type:
            ##############################################################
            #Shear alfven wave energy flux
            eq('{'+add+'sawS_net [W/Re^2]}='+
                                    '{sawS_x [W/Re^2]}*{surface_normal_x}+'+
                                    '{sawS_y [W/Re^2]}*{surface_normal_y}+'+
                                    '{sawS_z [W/Re^2]}*{surface_normal_z}',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'sawS_escape [W/Re^2]}=max({'+
                                                add+'sawS_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'sawS_injection [W/Re^2]} ='+
                         'min({'+add+'sawS_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
    if 'mag' in analysis_type:
        if kwargs.get('do_1Dsw',False):
            prefixlist = ['1D']
        elif kwargs.get('do_cms',False) and 'iono' in zone.name:
            prefixlist = ['past','','future']
        else:
            prefixlist = ['']
        for add in prefixlist:
            # If the variable was accidentially inherited it becomes stuck
            #   as a nodal variable, we then need to delete and remake it
            if add+'Bf_net [Wb/Re^2]' in zone.dataset.variable_names:
                if zone.values(add+'Bf_net ?Wb/Re^2?').location != CC:
                    zone.dataset.delete_variables(zone.dataset.variable(
                                                add+'Bf_net ?Wb/Re^2?'))
                    zone.dataset.delete_variables(zone.dataset.variable(
                                                add+'Bf_escape ?Wb/Re^2?'))
                    zone.dataset.delete_variables(zone.dataset.variable(
                                                add+'Bf_injection ?Wb/Re^2?'))
            ##############################################################
            #Normal Magnetic Flux
            eq('{'+add+'Bf_net [Wb/Re^2]} =('+
                                      '{'+add+'B_x [nT]}*{surface_normal_x}'+
                                     '+{'+add+'B_y [nT]}*{surface_normal_y}'+
                                     '+{'+add+'B_z [nT]}*{surface_normal_z})'+
                                      '*6.371**2*1e3',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'Bf_escape [Wb/Re^2]} ='+
                        'max({'+add+'Bf_net [Wb/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'Bf_injection [Wb/Re^2]} ='+
                        'min({'+add+'Bf_net [Wb/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
        if kwargs.get('do_cms',False) and 'iono' in zone.name:
            #dflux/dt based on central difference
            tdelta=str(kwargs.get('tdelta',60)*2)
            eq('{dBfdt_net [Wb/s/Re^2]} = ({futureBf_net [Wb/Re^2]}-'+
                                          '{pastBf_net [Wb/Re^2]})/'+tdelta,
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
    if 'mass' in analysis_type:
        if kwargs.get('do_1Dsw',False):
            prefixlist = ['1D']
        else:
            prefixlist = ['']
        for add in prefixlist:
            ##############################################################
            #Normal Mass Flux
            eq('{'+add+'RhoU_net [kg/s/Re^2]} = {'+add+'Rho [amu/cm^3]}*'+
                                            '1.67*10e-12*6371**2*('+
                                 '{'+add+'U_x [km/s]}*{surface_normal_x}'+
                                 '+{'+add+'U_y [km/s]}*{surface_normal_y}'+
                                 '+{'+add+'U_z [km/s]}*{surface_normal_z})',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            #Split into + and - flux
            eq('{'+add+'RhoU_escape [kg/s/Re^2]} ='+
                        'max({'+add+'RhoU_net [kg/s/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)
            eq('{'+add+'RhoU_injection [kg/s/Re^2]} ='+
                        'min({'+add+'RhoU_net [kg/s/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=zonelist)


def get_1D_sw_variables(field_data, xmax, xmin, nx):
    """Function calculates values for energetics of 1D pristine solar wind
    Inputs
        field_data- tecplot Dataset class containing 3D field data
        xmax, xmin, nx- discretization for 1D extraction of variables
    """
    eq = tp.data.operate.execute_equation
    #extract data in 1D line at the edge of the simulation domain
    #Assumption is that this is 'unpreterbed'
    yvalue = 120; zvalue = 120
    xvalues, dx = np.linspace(xmax, xmin, nx, retstep=True)
    dx = abs(dx)
    oneD_data = pd.DataFrame(columns=field_data.variable_names)
    #Extract 1D line of field data
    for x in xvalues:
        #probe data at x,y,z to get all energetic field variables
        data = tp.data.query.probe_at_position(x,yvalue,zvalue)[0]
        indata = pd.DataFrame([data],columns=field_data.variable_names)
        oneD_data = pd.concat([indata,oneD_data],ignore_index=False)
        '''
        oneD_data = oneD_data.append(pd.DataFrame([data],
                                     columns=field_data.variable_names),
                                     ignore_index=False)
        '''
    #Create new global variables
    #for var in [v for v in field_data.variable_names if 's ' in v]:
    varlist1D = ['Rho [amu/cm^3]',
                 'U_x [km/s]',
                 'U_y [km/s]',
                 'U_z [km/s]',
                 'B_x [nT]',
                 'B_y [nT]',
                 'B_z [nT]',
                 'P [nPa]',
                 'Dp [nPa]',
                 'Bmag [nT]']
    for var in (varlist1D+
                [v for v in field_data.variable_names if '/Re^2' in v]):
        #Make polynomial fit bc tec equation length is very limited
        p = np.polyfit(oneD_data['X [R]'], oneD_data[var],3)
        fx = xvalues**3*p[0]+xvalues**2*p[1]+xvalues*p[2]+p[3]
        eqstr = ('{1D'+str(var)+'} = IF({X [R]}<'+str(xmax)+'&&'+
                                      '{X [R]}>'+str(xmin)+','+
                                     str(p[0])+'*{X [R]}**3+'+
                                     str(p[1])+'*{X [R]}**2+'+
                                     str(p[2])+'*{X [R]}+'+
                                     str(p[3])+
                                     ', 0)')
        eq(eqstr, value_location=ValueLocation.CellCentered)

def get_surfaceshear_variables(field_data, field, minval, maxval,*,
                               reverse=False):
    """Function calculates values for energetics of 1D pristine solar wind
    Inputs
        field_data- tecplot Dataset class containing 3D field data
        field- string indicating which variable to use to find N vector
        minval, maxval- range for boundary layer where shear variables
                        will be non-zero
        reverse- if True reverse initial gradient for N
    """
    eq = tp.data.operate.execute_equation
    #TBD add reverse functionality
    #Modified field variable that saturates at given limits(boundarylayer)
    eq('{'+field+'_bl} = IF({'+field+'}>'+str(minval)+'&&'+
                           '{'+field+'}<'+str(maxval)+
         ',{'+field+'},IF({'+field+'}<'+str(minval)+','+str(minval)+','+
                                                  str(maxval)+'))',
                         value_location=ValueLocation.CellCentered)
    #value_location=ValueLocation.CellCentered)
    """
    #Un normalized vector representing gradient of created boundary layer
    eq('{N_x} = ddx({'+field+'_bl})',
            value_location=ValueLocation.CellCentered)
    eq('{N_y} = ddy({'+field+'_bl})',
            value_location=ValueLocation.CellCentered)
    eq('{N_z} = ddz({'+field+'_bl})',
            value_location=ValueLocation.CellCentered)
    eq('{N} = sqrt({N_x}**2+{N_y}**2+{N_z}**2)',
            value_location=ValueLocation.CellCentered)
    #Normalized N, with conditon to avoid div by 0
    eq('{nx} = IF({N}>1e-10, {N_x}/{N}, 0)',
            value_location=ValueLocation.CellCentered)
    eq('{ny} = IF({N}>1e-10, {N_y}/{N}, 0)',
            value_location=ValueLocation.CellCentered)
    eq('{nz} = IF({N}>1e-10, {N_z}/{N}, 0)',
            value_location=ValueLocation.CellCentered)
    #Velocity perpendicular to the boundary layer gradient
    eq('{Udotn} = {U_x [km/s]}*{nx}+{U_y [km/s]}*{ny}+{U_z [km/s]}*{nz}',
            value_location=ValueLocation.CellCentered)
    eq('{Uperp_x} = {nx}*{Udotn}',
            value_location=ValueLocation.CellCentered)
    eq('{Uperp_y} = {ny}*{Udotn}',
            value_location=ValueLocation.CellCentered)
    eq('{Uperp_z} = {nz}*{Udotn}',
            value_location=ValueLocation.CellCentered)
    #Velocity parallel to the boundary layer gradient
    eq('{Upar_x} = {U_x [km/s]} - {Uperp_x}',
            value_location=ValueLocation.CellCentered)
    eq('{Upar_y} = {U_y [km/s]} - {Uperp_y}',
            value_location=ValueLocation.CellCentered)
    eq('{Upar_z} = {U_z [km/s]} - {Uperp_z}',
            value_location=ValueLocation.CellCentered)
    eq('{Upar} = sqrt({Upar_x}**2+{Upar_y}**2+{Upar_z}**2)',
            value_location=ValueLocation.CellCentered)
    #Directional derivative normal to boundary layer
    #eq('{D_n Upar} = ddx({Upar})*{nx}+ddy({Upar})*{ny}+ddz({Upar})*{nz}',
    #        value_location=ValueLocation.CellCentered)
    """

def temporal_FD_variable(zone_current, zone_future, field):
    """Function creates field variable representing forward difference in
        time
    Inputs
        zone_past, zone_current- ID's for data at two time locations
        field- variable that is being differenced
    """
    eq = tp.data.operate.execute_equation
    eq('{d'+field+'}={'+field+'}['+str(zone_future.index+1)+']'
                          '-{'+field+'}['+str(zone_current.index+1)+']',
                                                    zones=[zone_current],
                              value_location=ValueLocation.CellCentered)

def get_surface_velocity(zone, field, field_surface_0, delta_field,
                         delta_time, *, reverse=False):
    """Function calculates field variable of surface velocity
    Inputs
        zone- where to calcuate field variables
        field- primary field used to ID surface and get surf velocity
        field_surface_0- value which defines the surface
        delta_field- field variable for IDing displacement gradient
        delta_time- time differential (s) used to calculate velocity
        reverse- boolean for swapping normal direction (based on field)
    """
    eq = tp.data.operate.execute_equation
    eq('{h_'+field+'} = {'+field+'}-'+str(field_surface_0), zones=[zone],
                               value_location=ValueLocation.CellCentered)
    #Un normalized vector representing gradient of created boundary layer
    eq('{N_x} = ddx({'+field+'})*IF('+str(int(reverse))+',-1,1)',
            value_location=ValueLocation.CellCentered)
    eq('{N_y} = ddy({'+field+'})*IF('+str(int(reverse))+',-1,1)',
            value_location=ValueLocation.CellCentered)
    eq('{N_z} = ddz({'+field+'})*IF('+str(int(reverse))+',-1,1)',
            value_location=ValueLocation.CellCentered)
    eq('{N} = sqrt({N_x}**2+{N_y}**2+{N_z}**2)',
            value_location=ValueLocation.CellCentered)
    #Normalized N, with conditon to avoid div by 0
    eq('{nx}=IF({N}!=0,{N_x}/({N}+1e-15),0)',
            value_location=ValueLocation.CellCentered)
    eq('{ny}=IF({N}!=0,{N_y}/({N}+1e-15),0)',
            value_location=ValueLocation.CellCentered)
    eq('{nz}=IF({N}!=0,{N_z}/({N}+1e-15),0)',
            value_location=ValueLocation.CellCentered)
    #Un normalized vector representing gradient of temporal change
    eq('{'+delta_field+'_x} = -ddx({'+delta_field+'})', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{'+delta_field+'_y} = -ddy({'+delta_field+'})', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{'+delta_field+'_z} = -ddz({'+delta_field+'})', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{'+delta_field+'_nx} = {'+delta_field+'_x}*{nx}', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{'+delta_field+'_ny} = {'+delta_field+'_y}*{ny}', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{'+delta_field+'_nz} = {'+delta_field+'_z}*{nz}', zones=[zone],
            value_location=ValueLocation.CellCentered)
    eq('{Grad'+delta_field+'} = sqrt(({'+delta_field+'_x}*{nx})**2+'+
                                    '({'+delta_field+'_y}*{ny})**2+'+
                                    '({'+delta_field+'_z}*{nz})**2)',
                          value_location=ValueLocation.CellCentered,
                                                       zones=[zone])
    eq('{d_surface_'+field+'} = IF({Grad'+delta_field+'}!=0,'+
                    '-{'+delta_field+'}/({Grad'+delta_field+'}+1e-15),0)',
                          value_location=ValueLocation.CellCentered,
                                                       zones=[zone])
def get_virials():
    """Function constructs strings representing virial boundary terms
    Inputs
        None: TBD evaluate/get current variable names for robustness
    """
    terms, eq = [], tp.data.operate.execute_equation
    #Create shorthand objects for readability
    dSx='{surface_normal_x}';rx='{x_cc}'
    dSy='{surface_normal_y}';ry='{y_cc}'
    dSz='{surface_normal_z}';rz='{z_cc}'
    #Scalar pressures at radial distance
    pressures = ['{Pth [J/Re^3]}', '{uB [J/Re^3]}','{uB_dipole [J/Re^3]}',
                 '{Virial Ub [J/Re^3]}']
    signs = ['-1*','-1*','', '']
    for p in enumerate(pressures):
        ptag = p[1].split(' ')[0].split('{')[-1]
        termP=('{virial_scalar'+ptag+'}='+signs[p[0]]+'('+dSx+'*'+rx+'+'+
                                             dSy+'*'+ry+'+'+
                                             dSz+'*'+rz+')'+'*'+p[1])
        terms.append(termP)
    #Momentum advection
    termA = ('{virial_advect}=-1*(({rhoUx_cc}*'+dSx+
                                   '+{rhoUy_cc}*'+dSy+
                                   '+{rhoUz_cc}*'+dSz+')*'+
             '({ux_cc}*'+rx+'+{uy_cc}*'+ry+'+{uz_cc}*'+rz+'))*'+
             '1.6605*6371**3*1e-6')
    terms.append(termA)
    #Magnetic stress non scalar contributions
    termBB = ('{virial_BB}=('
     +dSx+'*({B_x_cc}*{B_x_cc}*{x_cc}+'+
            '{B_x_cc}*{B_y_cc}*{y_cc}+'+
            '{B_x_cc}*{B_z_cc}*{z_cc})+'+
     dSy+'*({B_y_cc}*{B_x_cc}*{x_cc}+'+
            '{B_y_cc}*{B_y_cc}*{y_cc}+'+
            '{B_y_cc}*{B_z_cc}*{z_cc})+'+
     dSz+'*({B_z_cc}*{B_x_cc}*{x_cc}+'+
            '{B_z_cc}*{B_y_cc}*{y_cc}+'+
            '{B_z_cc}*{B_z_cc}*{z_cc}))'+
                     '*1e-9*6371**3/(4*pi*1e-7)')
    termBdBd = ('{virial_BdBd}=0*('
     +dSx+'*({Bdx_cc}*{Bdx_cc}*{x_cc}+'+
            '{Bdx_cc}*{Bdy_cc}*{y_cc}+'+
            '{Bdx_cc}*{Bdz_cc}*{z_cc})+'+
     dSy+'*({Bdy_cc}*{Bdx_cc}*{x_cc}+'+
            '{Bdy_cc}*{Bdy_cc}*{y_cc}+'+
            '{Bdy_cc}*{Bdz_cc}*{z_cc})+'+
     dSz+'*({Bdz_cc}*{Bdx_cc}*{x_cc}+'+
            '{Bdz_cc}*{Bdy_cc}*{y_cc}+'+
            '{Bdz_cc}*{Bdz_cc}*{z_cc}))'+
                     '*1e-9*6371**3/(4*pi*1e-7)')
    termBBd = ('{virial_BBd}=-1*('
     +dSx+'*({B_x_cc}*{Bdx_cc}*{x_cc}+'+
            '{B_x_cc}*{Bdy_cc}*{y_cc}+'+
            '{B_x_cc}*{Bdz_cc}*{z_cc})+'+
     dSy+'*({B_y_cc}*{Bdx_cc}*{x_cc}+'+
            '{B_y_cc}*{Bdy_cc}*{y_cc}+'+
            '{B_y_cc}*{Bdz_cc}*{z_cc})+'+
     dSz+'*({B_z_cc}*{Bdx_cc}*{x_cc}+'+
            '{B_z_cc}*{Bdy_cc}*{y_cc}+'+
            '{B_z_cc}*{Bdz_cc}*{z_cc}))'+
                     '*1e-9*6371**3/(4*pi*1e-7)')
    terms.append(termBB)
    terms.append(termBdBd)
    terms.append(termBBd)
    #Total surface contribution
    term_titles = ['{'+term.split('{')[1].split('}')[0]+'}'
                                                     for term in terms]
    total_adv=('{virial_Fadv [J/Re^2]}={virial_scalarPth}+{virial_advect}')
    terms.append(total_adv)
    total_loz=('{virial_Floz [J/Re^2]}={virial_scalaruB}+{virial_BB}+'+
                                      '{virial_scalaruB_dipole}+'+
                                      '{virial_BBd}+{virial_BdBd}')
    terms.append(total_loz)
    total = ('{virial_surfTotal [J/Re^2]}={virial_Fadv [J/Re^2]}+'+
                                         '{virial_Floz [J/Re^2]}')
    terms.append(total)
    #Debug
    '''
    test_volume = ('{example_volume [#/Re^3]}=1')
    test_surface = ('{example_surface [#/Re^2]}=-1*('+dSx+'*'+rx+'+'+
                                                      dSy+'*'+ry+'+'+
                                                      dSz+'*'+rz+')')
    terms.append(test_volume)
    terms.append(test_surface)
    '''
    for term in terms:
        eq(term, value_location=ValueLocation.CellCentered)

def mag2gsm(lat,lon,btheta,*,r=1,**kwargs):
    """
    """
    if kwargs.get('cartIN',False) and 'z_mag' in kwargs:
        x_mag, y_mag, z_mag = lat,lon,kwargs.get('z_mag')
    else:
        #find xyz_mag
        x_mag, y_mag, z_mag = sph_to_cart(r,lat,lon)
    #get rotation matrix
    rot = rotation(-btheta*pi/180,axis='y')
    #find new points by rotation
    return np.matmul(rot,[x_mag,y_mag,z_mag])

def make_trade_eq(from_state,to_state,tagname,tstep,**kwargs):
    """Creates the equation string and evaluates 'trade' state
        Fix from states with [1] designating the currentzone
        Fix to states with [2] designating the futurezone
              Reverse for opposite sign

          ex. if( dayclosed[past] & ext[future]) then +M5a[now]/dt
              elif( dayclosed[future] & ext[past] then -M5a[now]/dt
    Inputs
        from_state,to_state (str(variablename))- denotes sign convention
        tagname (str)- tag put on variable
        kwargs:
            source_list (list[int])- ordered list w index of [past,pres,futr]
    Returns
        tradestr (str)- equation to be used to evaluate equation
    """
    #NOTE this assumes past, present, future are zones 1,2,3 (tecplot indexing)
    source_list = kwargs.get('source_list',[1,2,3])
    past = '['+str(source_list[0])+']'
    present = '['+str(source_list[1])+']'
    future = '['+str(source_list[2])+']'
    tradestr = ('if('+from_state.replace('}','}'+past)+'&&'+
                     to_state.replace('}','}'+future)+
                     ',{value}'+present+'/'+tstep+','+
                'if('+from_state.replace('}','}'+future)+'&&'+
                     to_state.replace('}','}'+past)+
                     ',-1*{value}'+present+'/'+tstep+',0))')
    return '{name'+tagname+'} = '+tradestr

def make_alt_trade_eq(from_state,to_state,tagname,tstep,**kwargs):
    #TODO: streamline two primary 'cms' integrals:
    #       1. dBf/dt integration over OPEN north/south
    #       2. Bf integration over Opening and Closing north/south
    #           a. Bf_acqu Day(M2a)/Night(M2b) (Opening)
    #           b. Bf_forf Day(M2a)/Night(M2b) (Closing)
    #   X   3. (duplicate) Br(ut . n) integration around the OPEN north/south
    #NOTE this assumes past, present, future are listed with prepends
    tradestr = ('if('+from_state.replace('{','{past')+'&&'+
                      to_state.replace('{','{future')+
                     ',{value}/'+tstep+','+
                'if('+from_state.replace('{','{future')+'&&'+
                      to_state.replace('{','{past')+
                     ',-1*{value}/'+tstep+',0))')
    return '{name'+tagname+'} = '+tradestr

def eqeval(eqset,**kwargs):
    for lhs,rhs in eqset.items():
        tp.data.operate.execute_equation(lhs+'='+rhs,
                              zones=kwargs.get('zones'),
                              value_location=kwargs.get('value_location'),
                              ignore_divide_by_zero=True)
        '''
        try:
            tp.data.operate.execute_equation(lhs+'='+rhs,
                              zones=kwargs.get('zones'),
                              value_location=kwargs.get('value_location'),
                              ignore_divide_by_zero=True)
        except TecplotSystemError:
            from IPython import embed; embed()
        '''

def get_global_variables(field_data, analysis_type, **kwargs):
    """Function calculates values for energetics tracing
    Inputs
        field_data- tecplot Dataset class containing 3D field data
        kwargs:
            aux- if dipole equations and corresponding energies are wanted
            is3D- if all 3 dimensions are present
    """
    alleq = equations(aux=kwargs.get('aux'))
    cc = ValueLocation.CellCentered
    nodal = ValueLocation.Nodal
    eq = tp.data.operate.execute_equation
    #Testing variables
    if kwargs.get('verbose',False)or('test'in
                                     kwargs.get('customTerms',{}).keys()):
        eqeval(alleq['interface_testing'])
    #General equations
    if (any([var.find('J_')!=-1 for var in field_data.variable_names])and
        any([var.find('`mA')!=-1 for var in field_data.variable_names])):
        field_data.variable('J_x*').name = 'J_x [uA/m^2]'
        field_data.variable('J_y*').name = 'J_y [uA/m^2]'
        field_data.variable('J_z*').name = 'J_z [uA/m^2]'
    #Useful spatial variables
    if kwargs.get('is3D',True):
        tp.macro.execute_extended_command('CFDAnalyzer3',
                                          'CALCULATE FUNCTION = '+
                                          'CELLVOLUME VALUELOCATION = '+
                                          'CELLCENTERED')
        if 'aux' in kwargs:
            aux = kwargs.get('aux')
        else:
            aux = field_data.zone('global_field').aux_data
        eqeval(alleq['basic3d'])
        if 'dvol [R]^3' in field_data.variable_names:
            eq('{Cell Size [Re]}={dvol [R]^3}**(1/3)',
                                 zones=[field_data.zone('global_field')],
                                 value_location=cc)
        else:
            tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'CELLCENTERED')
            eq('{Cell Size [Re]}={Cell Volume}**(1/3)',
                                 zones=[field_data.zone('global_field')],
                                 value_location=cc)
        eqeval(alleq['dipole_coord'])
        eqeval(alleq['dipole'])
        #eqeval(alleq['dipole'],value_location=cc)
        field_data.delete_variables([field_data.variable('Cell Volume')])
        if kwargs.get('only_dipole',False):#use the dipole as the whole field
            eqeval({'{B_x [nT]}':'{Bdx}',
                    '{B_y [nT]}':'{Bdy}',
                    '{B_z [nT]}':'{Bdz}'})
    else:
        if 'XY_zone_index' in kwargs:
            eqeval(alleq['basic2d_XY'],
                   zones=[kwargs.get('XY_zone_index',1),
                          kwargs.get('XYTri_index',6)])
        else:
            eqeval(alleq['basic2d_XZ'],
                   zones=[kwargs.get('XZ_zone_index',0),
                          kwargs.get('XZTri_index',2)])
    #Physical quantities including Pdyn,Beta's,Bmag,Cs:
    eqeval(alleq['basic_physics'])
    #Fieldlinemaping
    if ('OCFLB' in analysis_type or analysis_type=='all') and (
                                                kwargs.get('is3D',True)):
        eqeval(alleq['fieldmapping'])
    #Volumetric energy terms
    if 'energy' in analysis_type or analysis_type=='all':
        eqeval(alleq['volume_energy'])
    #Volumetric mass term (kg/Re^3)
    if 'mass' in analysis_type or analysis_type=='all':
        eqeval(alleq['volume_mass'])
    #eqeval(alleq['volume_energy'],value_location=cc)
    if kwargs.get('do_interfacing',False) and ('phi_1 [deg]' in
                                               field_data.variable_names):
        pass
        #eqeval(alleq['daynightmapping'])
        #eqeval(alleq['daynightmapping'],value_location=cc)
    #Virial volume terms
    if 'virial' in analysis_type or analysis_type=='all':
        eqeval(alleq['virial_intermediate'],value_location=cc)
        eqeval(alleq['virial_volume_energy'],value_location=cc)
    #Biot savart
    if ('biotsavart' in analysis_type) or analysis_type=='all':
        eqeval(alleq['biot_savart'],value_location=cc)
    #Energy flux
    if 'energy' in analysis_type or analysis_type=='all':
        eqeval(alleq['energy_flux'])
        #eqeval(alleq['energy_flux'],value_location=cc)
    if 'wave' in analysis_type or analysis_type=='all':
        eqeval(alleq['wave_energy'],value_location=cc)
    #Reconnection variables
    if 'reconnect' in analysis_type:
        eqeval(alleq['reconnect'])
        #eqeval(alleq['reconnect'],value_location=cc)
    if 'ffj' in analysis_type:
        eqeval(alleq['ffj_setup'],value_location=nodal)
        eqeval(alleq['ffj'],value_location=cc)
    #trackIM
    if'trackIM'in analysis_type:eqeval(alleq['trackIM'],value_location=cc)
    #specific entropy
    if 'bs' in kwargs.get('modes',[]):
        eqeval(alleq['entropy'],value_location=cc)
    #plasmasheet
    if 'plasmasheet' in kwargs.get('modes',[]):
        pass
        #eqeval(alleq['local_curv'],value_location=nodal)
    #user_selected
    if 'add_eqset' in kwargs:
        for eq in [eq for eq in alleq if eq in kwargs.get('add_eqset')]:
            eqeval(alleq[eq],value_location=cc)
    if 'global_eq' in kwargs:
        eqeval(kwargs.get('global_eq'))

def integrate_tecplot(var, zone, *, VariableOption='Scalar'):
    """Function to calculate integral of variable on a 3D exterior surface
    Inputs
        var(Variable)- variable to be integrated
        zone(Zone)- zone to perform integration
        VariableOption- default scalar, can choose others
    Output
        integrated_total from result dataframe
    """
    #setup integration command
    integrate_command=("Integrate [{:d}] ".format(zone.index+1)+
                         "VariableOption="+VariableOption+" ")
    if VariableOption == 'Scalar':
        integrate_command = (integrate_command+
                         "ScalarVar={:d} ".format(var.index+1))
    integrate_command = (integrate_command+
                         "XVariable=1 "+
                         "YVariable=2 "+
                         "ZVariable=3 "+
                         "ExcludeBlanked='T' "+
                         " PlotResults='F' ")
    #integrate
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command)
    #access data via aux data variable that saves last total integr qty
    result = float(tp.active_frame().aux_data['CFDA.INTEGRATION_TOTAL'])
    return result

def setup_solidvolume(source, blankindex, state_variable,zonename,**kwargs):
    plt = tp.active_frame().plot()
    #turn on blanking
    plt.value_blanking.active = True
    #set to "primary value" for blanking
    plt.value_blanking.cell_mode = ValueBlankCellMode.PrimaryValue
    #clear all conditions
    for index in range(0,8):
        plt.value_blanking.constraint(index).active=False
    #set blank condition to not= statevariable
    inverse_stateblank = plt.value_blanking.constraint(blankindex)
    inverse_stateblank.variable = source.dataset.variable(state_variable)
    inverse_stateblank.comparison_operator = RelOp.NotEqualTo
    inverse_stateblank.comparison_value = kwargs.get('state_value',1)
    inverse_stateblank.active=True
    #extract blanked regions
    [newzone] = tp.data.extract.extract_blanked_zones(source)
    #set zone name
    newzone.name = zonename
    #turn off blanked condition
    inverse_stateblank.active=False
    #turn off blanking
    plt.value_blanking.active = False
    return newzone

def setup_slicezone(dataset, location_var, location_value, state_name,
                    **kwargs):
    """Function creates a slice zone and returns the new zone
    Inputs
    Returns
    """
    state_variable = dataset.variable(state_name)
    plot = tp.active_frame().plot()
    #turn on blanking
    plot.value_blanking.active = True
    #set to "primary value" for blanking
    plot.value_blanking.cell_mode = ValueBlankCellMode.PrimaryValue
    #turn on slices
    plot.show_slices = True
    #clear all conditions and slices
    for index in range(0,8):
        plot.value_blanking.constraint(index).active=False
        if index!=kwargs.get('sliceindex',3):
            plot.slice(index).show = False
        else:
            plot.slice(index).show = True
            state_slice = plot.slice(index)
    #set blank condition to not= statevariable
    inverse_stateblank = plot.value_blanking.constraint(
                                                   kwargs.get('blankindex',3))
    inverse_stateblank.variable = dataset.variable(state_variable)
    inverse_stateblank.comparison_operator = RelOp.NotEqualTo
    inverse_stateblank.comparison_value = kwargs.get('state_value',1)
    inverse_stateblank.active=True
    #setup slice properties
    if location_var=='x':
        state_slice.orientation = SliceSurface.XPlanes
        origin = (location_value,0,0)
        normal = (1,0,0)
    state_slice.origin[0] = 0
    zone = tp.data.extract.extract_slice(origin=origin,normal=normal,
                                         copy_cellcenters=True)
    return zone

def setup_isosurface(iso_value, varindex, zonename, *,
                     contindex=7, isoindex=7, global_key='global_field',
                                            blankvar='',blankvalue=3,
                                              blankop=RelOp.LessThan,
                                              keep_zones='largest'):
    """Function creates an isosurface and then extracts and names the zone
    Inputs
        iso_value
        varindex, contindex, isoindex- storage locations on tecplot side
        zonename
    Outputs
        newzone- primary zone created (w/ max elements)
    """
    ds = tp.active_frame().dataset
    plt = tp.active_frame().plot()
    #hide all zones not matching global_key
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if zone.name==global_key:
                plt.fieldmap(map_index).show = True
            else:
                plt.fieldmap(map_index).show = False
    plt.show_isosurfaces = True
    iso = plt.isosurface(isoindex)
    iso.show = True
    iso.definition_contour_group_index = contindex
    plt.contour(contindex).variable_index = varindex
    iso.isosurface_values[0] = iso_value
    print('creating isosurface of {}={:.2f}'.format(
                                    ds.variable(varindex).name,
                                    iso_value))
    orig_nzones = ds.num_zones
    #Check for blanking conditions
    if blankvar != '':
        plt.value_blanking.active = True
        plt.value_blanking.cell_mode = ValueBlankCellMode.PrimaryValue
        blank = plt.value_blanking.constraint(1)
        blank.active = True
        blank.variable = ds.variable(blankvar)
        blank.comparison_operator = blankop
        blank.comparison_value = blankvalue
    try:
        macro = tp.macro.execute_command
        macro_cmd='$!ExtractIsoSurfaces Group = {:d} '.format(isoindex+1)
        #if zonename!='ms_lobes':
        if keep_zones=='largest' or keep_zones=='all':
            macro_cmd+='ExtractMode = OneZonePerConnectedRegion'
        macro(macro_cmd)
    except TecplotMacroError:
        print('Unable to create '+zonename+'!')
        return None
    iso.show = False
    #Turn off blanking
    if blankvar != '':
        plt.value_blanking.active = False
    if keep_zones=='largest':
        #only keep zone with the highest number of elements
        zsizes=pd.DataFrame([(z, z.num_elements)for z in
                              ds.zones('*region*')],columns=['zone','size'])
        newzone=zsizes[zsizes['size']==zsizes['size'].max()]['zone'].values[0]
        if newzone.num_elements>200:
            newzone.name = zonename
            if len(zsizes) != 1:
                for i in reversed([z.index for z in ds.zones('*region*')]):
                    ds.delete_zones([i])
            return ds.zone(-1)#NOTE we return this way bc the index may
                              # change! so the 'newzone' reference is no
                              # longer safe
        else:
            ds.delete_zones([z.index for z in ds.zones('*region*')])
            return None
    elif keep_zones=='all':
        return [z for z in ds.zones('*region*')]
    else:
        return ds.zone(-1)

def calc_state(mode, zones, **kwargs):
    """Function selects which state calculation method to use
    Inputs
        mode- eg. "iso_betastar" "ps" "shue97" selects what to create
        sourcezoneID- which tecplot field data to use to create new state
        kwargs- see magnetosphere.py get_magnetosphere for details
    Returns
        zone- tecplot zone object
        innerzone- for magnetopause surface only, tecplot zone obj
        state_index- index for accessing new variable in tecplot
        zonename- name of the zone, (eg. appends mp for magnetopause)
    """
    #####################################################################
    #   This is where new subzones/surfaces can be put in
    #       Create a function calc_MYNEWSURF_state and call it
    #       Recommend: zonename as input so variable name is automated
    #       See example use of 'assert' if any pre_recs are needed
    #####################################################################
    i_primary = kwargs.get('mainZoneIndex',0)
    dataset = zones[i_primary].dataset
    #Call calc_XYZ_state and return state_index and create zonename
    closed_zone = kwargs.get('closed_zone')
    clean_blanks = False #flag to clear out blanks if get used here
    iso_value = 1 #default is state==1 for isosurface creation
    if 'iso_betastar' in mode:
        zonename = 'mp_'+mode
        state_index = calc_betastar_state(zonename,zones,**kwargs)

    elif mode == 'perfectsphere':
        zonename = mode+str(kwargs.get('sp_rmax',3))
        state_index = dataset.variable('r *').index
        iso_value = kwargs.get('sp_rmax',3)
    elif mode == 'sphere':
        zonename = mode+str(kwargs.get('sp_rmax',3))
        state_index = calc_sphere_state(mode, kwargs.get('sp_x',0),
                                kwargs.get('sp_y',0),
                                kwargs.get('sp_z',0),
                                kwargs.get('sp_rmax',3), zones,
                                rmin=kwargs.get('sp_rmin',0))
    elif mode == 'perfectellipsoid':
        zonename = mode
        tp.data.operate.execute_equation('{perfectellipsoid} = '+
            '({X [R]}+25)**2/35**2+{Y [R]}**2/20**2+{Z [R]}**2/20**2',
                                         zones=zones)
        state_index = dataset.variable('perfectellipsoid').index
        iso_value = 1
    elif mode == 'ellipsoid':
        zonename = mode
        kwargs.update({'keep_zones':'all'})
        tp.data.operate.execute_equation('{ellipsoid} = '+
            '({X [R]}+25)**2/35**2+{Y [R]}**2/20**2+{Z [R]}**2/20**2',
                                         zones=zones)
        tp.data.operate.execute_equation('{ellipsoid2} = if({ellipsoid}<1 &&'+
                                                             '{r [R]}>3,1,0)',
                                         zones=zones)
        state_index = dataset.variable('ellipsoid2').index
    elif mode == 'terminator':
        zonename = mode+str(kwargs.get('sp_rmax',3))
        assert dataset.zone('sphere*') is not None, (
                "No spherical zone, can't apply terminator!")
        sp_zone = dataset.zone('sphere*')
        north,south = calc_terminator_zone(zonename,sp_zone,**kwargs)
        return north, south, None
    elif mode == 'box':
        zonename = mode
        state_index = calc_box_state(zonename,
                                     kwargs.get('box_xmax',-5),
                                     kwargs.get('box_xmin',-8),
                                     kwargs.get('box_ymax',5),
                                     kwargs.get('box_ymin',-5),
                                     kwargs.get('box_zmax',5),
                                     kwargs.get('box_zmin',-5),
                                     zones)
    elif 'shue' in mode:
        if mode =='shue97':
            zonename = 'shu97'
        else:
            zonename = 'shue98'
        state_index = calc_shue_state(dataset, mode,
                                      kwargs.get('x_subsolar'),
                                      kwargs.get('tail_cap', -20),
                                      zones)
    elif 'lcb' in mode or ('closed' in mode and
                           kwargs.get('full_closed',False)):
        assert kwargs.get('do_trace',False) == False, (
                            "lcb mode only works with do_trace==False!")
        assert closed_zone is not None,('No closed_zone present!'+
                                                 ' Cant do lcb')
        #zonename = closed_zone.name
        zonename = 'ms_'+mode
        state_index=dataset.variable(closed_zone.name).index
    elif 'lobe' in mode:
        mpvar = kwargs.get('mpvar',dataset.variable('mp*'))
        assert kwargs.get('do_trace',False) == False, (
                            "lobe mode only works with do_trace==False!")
        #assert mpvar is not None,('magnetopause variable not found'+
        #                          'cannot calculate lobe zone!')
        #NOTE replaced assertion w emergency magnetopause variable creation
        if mpvar is None:
            calc_betastar_state('mp_iso_betastar', zones,**kwargs)
            mpvar = dataset.variable('mp*')
        zonename = 'ms_'+mode
        if 'slobe' in mode.lower():
            state_index = calc_lobe_state(mpvar.name, 'south',
                                          zones,**kwargs)
        elif 'nlobe' in mode.lower():
            state_index = calc_lobe_state(mpvar.name, 'north',
                                          zones,**kwargs)
        else:
            state_index = calc_lobe_state(mpvar.name, 'both',
                                          zones,**kwargs)
    elif 'plasmasheet' in mode:
        assert 'daynight' in dataset.variable_names, ('No'+
                                       ' daynightmapping present!'+
                                       ' Cant do plasmasheet')
        #Br=0, NOTE this makes a flat 2D planar surface (0 captured volume)
        zonename = 'ms_'+mode
        state_index = dataset.variable('Br *').index
        iso_value = 0
        # Blank daynight>-1 implies both closed and mapped to nightside
        tp.active_frame().plot().value_blanking.active = True
        tp.active_frame().plot().value_blanking.cell_mode = (
                                               ValueBlankCellMode.PrimaryValue)
        blank = tp.active_frame().plot().value_blanking.constraint(7)
        blank.active = True
        blank.variable = dataset.variable('daynight')
        blank.comparison_operator = RelOp.GreaterThan
        blank.comparison_value = -1
        clean_blanks = True
    elif 'rc' in mode:
        assert closed_zone is not None, ('No'+
                                       ' closed_zone present! Cant do rc')
        zonename = 'ms_'+mode
        state_index = calc_rc_state(closed_zone.name,
                                    str(kwargs.get('lshelllim',7)),
                                    zones, **kwargs)
    elif 'ps' in mode:
        assert closed_zone is not None, ('No'+
                                       ' closed_zone present! Cant do ps')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('ps', closed_zone.name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)),
                                        zones)
    elif 'qDp' in mode:
        assert closed_zone is not None, ('No'+
                                      ' closed_zone present! Cant do qDp')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('qDp', closed_zone.name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)),
                                        zones)
    elif 'closed' in mode:
        assert closed_zone is not None, ('No'+
                                   ' closed_zone present! Cant do closed')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('closed', closed_zone.name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)),
                                        zones)
    elif 'xslice' in mode:
        assert 'lcb' in dataset.variable_names, ('No'+
                                       ' closed_zone variable! Cant do xslice')
        zonename = 'ms_'+mode
        #state_index = calc_xslice_state(zonename,closed_zone.name,zones)
        zone = setup_slicezone(dataset,'x',0,'lcb')
        zone.name = zonename
        return zone, None, dataset.variable('lcb').index

    elif 'bs' in mode:
        #TODO: revive and refresh this to give a consistant result
        #       -> then integrate only the forward projected area
        #       -> Determine summary of geometric results: SA, standoff, flare
        #       -> Main Q is: How does energy partition upstream of shock 
        #                       affect the energy flux through the shock?
        #                 or: Is the low beta ejecta portion of the event
        #                       transfering more or less energy through the
        #                       shock?
        zonename = 'ext_'+mode
        state_index = calc_bs_state2(kwargs.get('deltaS',3),
                                    kwargs.get('betastarblank',0.8),
                                    kwargs.get('tail_cap',-20),
                                    zones,
                                    mpexists=('mpvar' in kwargs.keys()))
        if kwargs.get('create_zone',True):
            upstream = setup_isosurface(1,state_index,zonename,
                                        blankvar = 'X *',
                                        blankvalue=kwargs.get('tail_cap',-20))
        #TEMPORARY REROUTE FOR BOW SHOCK MODE
        #   Bow shock detection finds upstream edge where sw properties are
        #   still unshocked, to work around we will:
        #       1. find the overshoot max * some factor (1.2) at the nose
        #       2. copy and shift the whole surface back by this distance
        #       3. reinterpolate the field data to the new surface
        #       4. recalculate derived (global) variables

        #       1.1 find the nose
        ds = upstream.dataset
        nose = upstream.values('X *').max()
        X = upstream.values('X *').as_numpy_array()
        Y = upstream.values('Y *').as_numpy_array()
        Z = upstream.values('Z *').as_numpy_array()

        #       1.2 extract values along a flow line passing through nose
        tp.active_frame().plot().vector.u_variable = ds.variable('U_x *')
        tp.active_frame().plot().vector.v_variable = ds.variable('U_y *')
        tp.active_frame().plot().vector.w_variable = ds.variable('U_z *')
        flow_line = tp.active_frame().plot().streamtraces
        flow_line.add([nose,0,0],Streamtrace.VolumeLine,
                      direction=StreamDir.Both)
        flow_line.extract()
        flow_line.delete_all()
        ds.zone(-1).name = 'flow_line_nose'
        for yseed,zseed,tag in [[10,0,'+10Y'],[-10,0,'-10Y'],
                                [0,10,'+10Z'],[0,-10,'-10Z']]:
            if X[(abs(Y-yseed)<.5)&(abs(Z-zseed)<.5)] != []:
                flow_line.add([X[(abs(Y-yseed)<.5)&(abs(Z-zseed)<.5)].max(),
                           yseed,zseed], Streamtrace.VolumeLine,
                           direction=StreamDir.Both)
            else:
                flow_line.add([X[(abs(Y-yseed)<1)&(abs(Z-zseed)<1)].max(),
                           yseed,zseed], Streamtrace.VolumeLine,
                           direction=StreamDir.Both)
            flow_line.extract()
            flow_line.delete_all()
            ds.zone(-1).name = 'flow_line'+tag
        #       1.3 find the overshoot max * some factor (1.2)
        overshoot = ds.zone('flow_line_nose').values('X *').as_numpy_array()[
                (ds.zone('flow_line_nose').values('Rho *').as_numpy_array()==
                         ds.zone('flow_line_nose').values('Rho *').max())][0]
        #       2. copy and shift the whole surface back by this distance
        downstream = ds.copy_zones(upstream)[0]
        downstream.name = downstream.name+'_down'
        upstream.name = upstream.name+'_up'
        downstream.values('X *')[:]=(downstream.values('X *').as_numpy_array()
                                     +1.2*(overshoot-nose))
        #       3. reinterpolate the field data to the new surface
        tp.data.operate.interpolate_linear(downstream,
                                       source_zones=zones,
                                       variables=[3,4,5,6,7,8,9,10,11,12,13])
        #       4. recalculate derived (global) variables
        get_global_variables(ds, kwargs.get('analysis_type'),
                          aux=upstream.dataset.zone('global_field').aux_data,
                             modes=kwargs.get('modes',[]),zones=[downstream])
        return upstream, downstream, state_index
    elif 'Jpar' in mode:
        #Make sure we have a place to find the regions of intense FAC's
        assert any(
              ['inner' in zn for zn in dataset.zone_names]),(
                                        'No inner boundary zone created, '
                                            +'unclear where to calc FAC!')
        #Warn user if there is multiple valid "inner" zone targets
        if (['inner' in zn for zn in
                         dataset.zone_names].count(True) >1):
            warnings.warn("multiple 'inner' zones found, "+
                                   "default to first listed!",UserWarning)
        zonename = 'ms_'+mode
        state_index = calc_Jpar_state(mode, zones)
    else:
        assert False, ('mode not recognized!! Check "approved" list with'+
                       'available calc_state functions')
    if 'iso_betastar' in mode and kwargs.get('create_zone',True):
        #Generate outersurface with blanking the inner boundary
        zone = setup_isosurface(1, state_index, zonename,blankvar='r *',
                                blankvalue=kwargs.get('inner_r',3))
        #Sphere at fixed radius
        innerzone = setup_isosurface(kwargs.get('inner_r',3),
                            dataset.variable('r *').index,
                                     zonename+'innerbound',blankvar='')
        #PALEO update subsolar point
        new_subsolar = zone.values('X *').max()
        if 'x_subsolar' in zones[i_primary].aux_data:
            if new_subsolar>float(zones[i_primary].aux_data['x_subsolar']):
                print('x_subsolar updated to {}'.format(new_subsolar))
                zones[i_primary].aux_data['x_subsolar'] = new_subsolar
    elif kwargs.get('create_zone',True):
        if kwargs.get('keep_zones')=='all':
            newzones = setup_isosurface(iso_value, state_index,
                                    zonename,
                                    blankvar=kwargs.get('blankvar',''),
                                    blankvalue=kwargs.get('blankvalue',3),
                              keep_zones=kwargs.get('keep_zones','largest'))
            if 'sphere' in mode and kwargs.get('sp_rmin',0)>0:
                # Should have an outer an inner shell as top two largest
                sizes = [z.num_points for z in newzones]
                i_biggest = sizes.index(max(sizes))
                sizes.remove(max(sizes))
                zone = newzones.pop(i_biggest)
                i_next_biggest = sizes.index(max(sizes))
                innerzone = newzones.pop(i_next_biggest)
                innerzone.name = zonename+'_inner'
                # Delete the rest
                if newzones != []:
                    dataset.delete_zones(newzones)
                #NOTE need to refresh variable for innerzone to fix index
                innerzone = dataset.zone(zonename+'_inner')
                print('created '+innerzone.name)

                # Name the zones
                zone.name = zonename
                print('created '+zone.name)
            else:
                zone, innerzone = newzones
                # Name the zones
                zone.name = zonename
                print('created '+zone.name)
                innerzone.name = zonename+'_innerbound'
                print('created '+zone.name+'_innerbound')
        else:
            zone = setup_isosurface(iso_value, state_index, zonename,
                                    blankvar=kwargs.get('blankvar',''),
                                    blankvalue=kwargs.get('blankvalue',3),
                              keep_zones=kwargs.get('keep_zones','largest'))
            if 'sphere' in mode and kwargs.get('sp_rmin',0)>0:
                innerzone = setup_isosurface(kwargs.get('sp_rmin',0),
                                    state_index, zonename+'_inner',
                                    blankvar=kwargs.get('blankvar',''),
                                    blankvalue=kwargs.get('blankvalue',3),
                              keep_zones=kwargs.get('keep_zones','largest'))
            else:
                innerzone = None
    else:
        zone = None
        innerzone = None
    if clean_blanks:
        # loop through and deactivate any blanking conditions used in creation
        for i in range(0,8):
            tp.active_frame().plot().value_blanking.constraint(i).active=False
    return zone, innerzone, state_index

def extrema(array,factor):
    """Function to get mean+factor*sigma
    Input
        array
        factor
        sigma
    Returns
        extrema
    """
    return np.mean(array)+np.sign(np.mean(array))*factor*np.std(array)

def foot_dist(foot,target,tol):
    """Function returns True if foot (2 items) is near target (2 items)
    """
    return np.sqrt((foot[0]-target[0])**2+(foot[1]-target[1])**2)<tol

def forced_polarcap(sphere_zone, terminator_zone,*,
                 x='Xd *',y='Y *',z='rSigned*',status_key=2):
    """Function modifies the given zone to follow the open flux contour
    Inputs
        terminator_zone (Zone)- 1D tecplot Zone object
    Returns
        None (modifies given Zone object)
    """
    #Isolate values from the spherical zone and the 1D terminator curve
    terminator_x = terminator_zone.values(x).as_numpy_array()
    terminator_y = terminator_zone.values(y).as_numpy_array()
    terminator_z = terminator_zone.values(z).as_numpy_array()
    terminator_Status = terminator_zone.values('Status').as_numpy_array()

    sphere_x = sphere_zone.values(x).as_numpy_array()
    sphere_y = sphere_zone.values(y).as_numpy_array()
    sphere_z = sphere_zone.values(z).as_numpy_array()
    sphere_Status = sphere_zone.values('Status').as_numpy_array()
    n = 300
    xdims = np.linspace(-3,3,n)
    ydims = np.linspace(-terminator_y.min(),terminator_y.max(),n)
    X,Y = np.meshgrid(xdims,ydims)
    polarcap = sphere_zone.dataset.add_ordered_zone(
                                           'forced_north_polarcap',[n,n])
    polarcap.values('Xd*')[:] = X
    polarcap.values('Y *')[:] = Y
    polarcap.values('rSigned*')[:] = X*0+terminator_z.max()
    for i, (xtest,ytest,ztest) in enumerate(zip(terminator_x,
                                                terminator_y,
                                                terminator_z)):
        xx = np.linspace(-1,1,n)
        yy = np.ones(n)*ytest
        zz = np.ones(n)*ztest
        temp_zone = tp.data.extract.extract_line(zip(xx,yy,zz))
        x_results=temp_zone.values('Xd*').as_numpy_array()[
                    temp_zone.values('Status').as_numpy_array()==status_key]
        if len(x_results)!=0:
            polarcap.values('Xd*')[i*n:(i+1)*n]=np.linspace(x_results.min(),
                                                            x_results.max(),
                                                            n)
        else:
            polarcap.values('Xd*')[i*n:(i+1)*n]=np.zeros(n)
        sphere_zone.dataset.delete_zones(temp_zone)
    #Re-interpolate the 1D zones values from the global zone
    tp.data.operate.interpolate_linear(polarcap,source_zones=[0])

def open_contour(sphere_zone, terminator_zone,*,
                 x='Xd *',y='Y *',z='rSigned*',status_key=2):
    """Function modifies the given zone to follow the open flux contour
    Inputs
        terminator_zone (Zone)- 1D tecplot Zone object
    Returns
        None (modifies given Zone object)
    """
    #Isolate values from the spherical zone and the 1D terminator curve
    terminator_x = terminator_zone.values(x).as_numpy_array()
    terminator_y = terminator_zone.values(y).as_numpy_array()
    terminator_z = terminator_zone.values(z).as_numpy_array()
    terminator_Status = terminator_zone.values('Status').as_numpy_array()

    sphere_x = sphere_zone.values(x).as_numpy_array()
    sphere_y = sphere_zone.values(y).as_numpy_array()
    sphere_z = sphere_zone.values(z).as_numpy_array()
    sphere_Status = sphere_zone.values('Status').as_numpy_array()
    #Get the points that have too hight of status (indicating closed flux)
    #x_bad_points = terminator_x[terminator_Status>status_limit]
    #y_bad_points = terminator_y[terminator_Status>status_limit]
    #z_bad_points = terminator_z[terminator_Status>status_limit]
    bad_points = terminator_Status>status_key
    for i, (xtest,ytest,ztest) in enumerate(zip(terminator_x,
                                                terminator_y,
                                                terminator_z)):
        #Adjust each of these points' x value to be the closest abs
        if bad_points[i]:
            xx = np.linspace(-1,1,300)
            yy = np.ones(300)*ytest
            zz = np.ones(300)*ztest
            temp_zone = tp.data.extract.extract_line(zip(xx,yy,zz))
            x_results = temp_zone.values('Xd*').as_numpy_array()[
                    temp_zone.values('Status').as_numpy_array()==status_key]
            if len(x_results)==0:
                if i!=0:
                    terminator_zone.values('Xd*')[i]=terminator_zone.values(
                                                                  'Xd*')[i-1]
            else:
                try:
                    terminator_zone.values('Xd*')[i]=x_results[abs(x_results)==
                                                       abs(x_results).min()]
                except:
                    #Probably not good practice here...
                    terminator_zone.values('Xd*')[i]=x_results[
                                      abs(x_results)==abs(x_results).min()][0]
            sphere_zone.dataset.delete_zones(temp_zone)
    #Re-interpolate the 1D zones values from the global zone
    tp.data.operate.interpolate_linear(terminator_zone,source_zones=[0])

def calc_ocflb_zone(name,source,**kwargs):
    """ Function extracts openclosed field line boundary from source and saves
        largest connected contour
    Inputs
        name (str) - for name of new zone
        source (Zone) - tecplot zone what we're extracting from
        kwargs:
            hemi (str)- default 'North', otw 'South'
            contour_offset (float) - offset value for contour level
    Returns
        keepzone (Zone) - newly create zone
    """
    # Read inputs
    hemi = kwargs.get('hemi','North')
    contour_offset = kwargs.get('contour_offset',0.4)
    if hemi=='North': contour_level = 2
    elif hemi=='South': contour_level = 1
    # Setup variable hooks and contours
    plot = tp.active_frame().plot()
    plot.show_contour = True
    plot.contour(0).variable_index = source.dataset.variable('Status').index
    plot.contour(0).levels.delete_range(-3,3)
    plot.contour(0).levels.add(contour_level+contour_offset)
    # Adjust the field map to show contour LINE so we can then extract
    fieldmap = plot.fieldmap(source.index)#NOTE assume fieldmap-zonelist match
    fieldmap.show = True
    fieldmap.contour.show = True
    fieldmap.contour.contour_type = ContourType.Lines
    zones = tp.macro.execute_command('$!CreateContourLineZones '+
                          'ContLineCreateMode = OneZonePerIndependentPolyline')
    toplength = 0
    for zone in source.dataset.zones():
        if str(contour_level+contour_offset) in zone.name:
            zonelength = zone.num_points
            if zonelength>toplength:
                keepzone = zone
                toplength = zonelength
    keepzone.name = name+'_'+hemi
    badzones = [z for z in source.dataset.zones('*'+str(contour_level+
                                                        contour_offset)+'*')]
    if len(badzones)>0:
        source.dataset.delete_zones(badzones)
        #NOTE deleting zones can f up the reference to keepzone so must
        #      re-establish the variable after the delete zone operation
    return source.dataset.zone(name+'_'+hemi)

def calc_terminator_zone(name, sp_zone, **kwargs):
    """ Function takes spherical zone and creates zones for the north and
        south 'terminators' (actually using forward/beind dipole)
    Inputs
        name (str)- for naming the new zone
        sp_zone (Zone)- tecplot zone, everything sourced from here
        kwargs:
            npoints (300)- how many points to include in 1D zone
            sp_rmax (3)- spherical radius, !!should match sp_zone!!
    Returns
        north,south (Zone)- tecplot zones of 1D objects in 3D
    """
    stat_var = kwargs.get('stat_var','Status')
    if kwargs.get('ionosphere',False):
        ## Change to 2D -> Xd,Y coordinates
        tp.active_frame().plot_type = PlotType.Cartesian2D
        plot = tp.active_frame().plot()
        plot.axes.x_axis.variable = sp_zone.dataset.variable('Xd *')
    else:
        ## Create a signed radius
        tp.data.operate.execute_equation('{rSigned}=sign({Zd [R]})*{r [R]}')
        ## Change XYZ -> Xd,Y,r
        plot = sp_zone.dataset.frame.plot()
        plot.axes.x_axis.variable = sp_zone.dataset.variable('Xd *')
        # No change in Y
        plot.axes.z_axis.variable = sp_zone.dataset.variable('rSigned')
    # Hide all zones that aren't the one passed here
    for fieldmap in plot.fieldmaps():
        if [z for z in fieldmap.zones][0] != sp_zone:
            fieldmap.show = False
        else:
            fieldmap.show = True

    if 'hemi' in kwargs:
        if kwargs.get('hemi')=='North':
            hemis = [('north',2)]
        else:
            hemis = [('south',1)]
    else:
        hemis = [('north',2),('south',1)]
    north,south = None,None
    for hemi,stat in hemis:
        ## Get Y+- limits
        status = sp_zone.values(stat_var).as_numpy_array()
        if '_cc' in stat_var:
            y = sp_zone.values('y_cc2').as_numpy_array()
        else:
            y = sp_zone.values('Y *').as_numpy_array()
        try:
            ymin = y[status==stat].min()
            ymax = y[status==stat].max()
        except ValueError as err:
            print('\nUnable to create '+name+'!')
            if kwargs.get('verbose',False):
                print('\n error: {}'.format(err))
            north = name+'north'
            south = name+'south'
        else:
            ## Extract line @ Xd=0, Y+- limits, r=r_sphere
            npoints = kwargs.get('npoints',300)
            xx = np.zeros(npoints)
            yy = np.linspace(ymin,ymax,npoints)
            if hemi=='north':
                if tp.active_frame().plot_type == PlotType.Cartesian2D:
                    points = zip(xx,yy)
                else:
                    zz = np.zeros(npoints)+kwargs.get('sp_rmax',3)
                    points = zip(xx,yy,zz)
                north = tp.data.extract.extract_line(points)
                north.name = name+hemi
                #open_contour(sp_zone,north,status_key=stat)
                #forced_polarcap(sp_zone,north,status_key=stat)
            else:
                if tp.active_frame().plot_type == PlotType.Cartesian2D:
                    points = zip(xx,yy)
                else:
                    zz = np.zeros(npoints)-kwargs.get('sp_rmax',3)
                    points = zip(xx,yy,zz)
                south = tp.data.extract.extract_line(points)
                south.name = name+hemi
                #open_contour(sp_zone,south,status_key=stat)
    ## Change XYZ back -> XYZ
    tp.active_frame().plot_type = PlotType.Cartesian3D
    plot = tp.active_frame().plot()
    plot.axes.x_axis.variable = sp_zone.dataset.variable('X *')
    # No change in Y
    plot.axes.z_axis.variable = sp_zone.dataset.variable('Z *')
    # Unhide all the fieldmaps
    for fieldmap in plot.fieldmaps():
        fieldmap.show = True
    return north, south

def calc_Jpar_state(mode, zones,  **kwargs):
    """Function creates equation for region of projected Jparallel (field
      aligned current density). Typically found on an inner coupling boundary
      between GMand IE.
    Inputs
        mode (str)- 'Jpar+' or 'Jpar-' looking for intense positive/neg values
        sourcezone (Zone)- zone which as field data
        kwargs
            j+sigma
            j-sigma
            projection_tol (float)- tolerance in deg for footmatch
    Returns
        zone
    """
    i_primary = kwargs.get('mainZoneIndex',0)
    #Initialize state variable
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    eq('{'+mode+'}=0', zones=zones)
    tol = kwargs.get('projection_tol', 2)
    state = zones[i_primary].values(mode).as_numpy_array()
    #pull footpoint values from entire domain into arrays
    #global_th1 = sourcezone.values('theta_1 *').as_numpy_array()
    #global_th2 = sourcezone.values('theta_2 *').as_numpy_array()
    #global_phi1 = sourcezone.values('phi_1 *').as_numpy_array()
    #global_phi2 = sourcezone.values('phi_2 *').as_numpy_array()

    #define zone that will be used to find current densities
    zone = zones[i_primary].dataset.zone('*inner*')
    lat = zone.values('theta *').as_numpy_array()
    Jpar = zone.values('J_par *').as_numpy_array()

    #split Jpar values by hemisphere
    ns_flag = [lat>0, lat<0]
    '''
    ns_indices = [np.where(lat>0), np.where(lat<0)]
    Jpar_dict = {'north':Jpar[ns_indices[0]],
                 'south':Jpar[ns_indices[1]]}
    '''
    thetas = kwargs.get('thetas',['theta_1 *','theta_2 *'])
    phis = kwargs.get('phis',['phi_1 *','phi_2 *'])
    #for hemisph in enumerate(['north','south']):
    for hemisph in enumerate(['north']):
        #Pull tecplot into numpy arrays
        th, phi = thetas[hemisph[0]], phis[hemisph[0]]
        latitude=zone.values(th).as_numpy_array()
        longitude=zone.values(phi).as_numpy_array()
        '''
        latitude=zone.values(th).as_numpy_array()[ns_indices[hemisph[0]]]
        longitude=zone.values(phi).as_numpy_array()[ns_indices[hemisph[0]]]
        '''

        #Calc indices of extreme values, based on +/- sign and hemisphere
        if '+' in mode:
            JparLim = extrema(Jpar[(Jpar>0) &
                                   (ns_flag[hemisph[0]]) &
                                   (latitude>0)],
                              kwargs.get('sigma',1))
            target_indices=((Jpar>JparLim) & (ns_flag[hemisph[0]])
                           &(latitude>0))
        elif '-' in mode:
            JparLim = extrema(Jpar[(Jpar<0) &
                                   (ns_flag[hemisph[0]]) &
                                   (latitude<0)],
                              kwargs.get('sigma',1))
            target_indices=((Jpar>JparLim) & (ns_flag[hemisph[0]])
                           &(latitude<0))

        #list of location coordinattes: [latitude, longitude]
        targets = np.reshape([latitude[target_indices],
                              longitude[target_indices]],
                             [len(latitude[target_indices]),2])
        '''
        #Define global registry of indexes that meet footpoint list
        global_index =np.linspace(0,len(state)-1,num=len(state),dtype=int)
        global_th = sourcezone.values(th).as_numpy_array()
        global_phi = sourcezone.values(phi).as_numpy_array()
        #global_feet = np.reshape([sourcezone.values(th).as_numpy_array(),
        #                          sourcezone.values(phi).as_numpy_array()],
        #                         [len(state),2])
        valid_indices =global_index[(global_th>0) & (global_phi>0)]
        #Calculate min distance for each point

        state = any([np.sqrt((global_th[i]-t[0])**2 +
                                    (global_phi[i]-t[1])**2) < tol
                                                   for t in targets]).real
        for i in valid_indices:
            state[i] = any([np.sqrt((global_th[i]-t[0])**2 +
                                    (global_phi[i]-t[1])**2) < tol
                                                   for t in targets]).real
        state[proj_indices] = 1
        sourcezone.values(mode)[::] = state

        '''
        th_str = th.split('*')[0]+'[deg]'
        phi_str= phi.split('*')[0]+'[deg]'
        xmax = 15
        ymax = 30
        zmax = 30
        xmin = -30
        ymin = -30
        zmin = -30
        plot = tp.active_frame().plot()
        plot.value_blanking.active = True
        #x
        xblank = plot.value_blanking.constraint(1)
        xblank.active = True
        xblank.variable = zone.dataset.variable('X *')
        xblank.comparison_operator = RelOp.LessThan
        xblank.comparison_value = xmin
        xblank = plot.value_blanking.constraint(2)
        xblank.active = True
        xblank.variable = zone.dataset.variable('X *')
        xblank.comparison_operator = RelOp.GreaterThan
        xblank.comparison_value = xmax
        #y
        yblank = plot.value_blanking.constraint(3)
        yblank.active = True
        yblank.variable = zone.dataset.variable('Y *')
        yblank.comparison_operator = RelOp.LessThan
        yblank.comparison_value = ymin
        yblank = plot.value_blanking.constraint(4)
        yblank.active = True
        yblank.variable = zone.dataset.variable('Y *')
        yblank.comparison_operator = RelOp.GreaterThan
        yblank.comparison_value = ymax
        #z
        zblank = plot.value_blanking.constraint(5)
        zblank.active = True
        zblank.variable = zone.dataset.variable('Z *')
        zblank.comparison_operator = RelOp.LessThan
        zblank.comparison_value = zmin
        zblank = plot.value_blanking.constraint(6)
        zblank.active = True
        zblank.variable = zone.dataset.variable('Z *')
        zblank.comparison_operator = RelOp.GreaterThan
        zblank.comparison_value = zmax
        for (latP,lonP) in targets:
            #Each iteration projects valid locations through domain
            eq('{'+mode+'}=if(('+
                'abs({'+th_str+'}-'+str(latP)+') < '+str(tol)+')&&('+
                'abs({'+phi_str+'}-'+str(lonP)+')<'+str(tol)+'),1,'+
                                                            '{'+mode+'})',
                                                        value_location=CC,
                                                        zones=zones)
    return zone.dataset.variable(mode).index

def calc_bs_state(sonicspeed, betastarblank, xtail, sourcezone, *,
                  mpexists=False):
    """Function creates equation for the bow shock region
    Inputs
        sonicspeed- gas dynamics speed of sound: sqrt(gamma*P/rho)[km/s]
        betastarblank- blanking condition for betastar
        mpexists(boolean)- used to define the back end condition
    Return
        index- index for the created variable
    """
    eq = tp.data.operate.execute_equation
    src=str(sourcezone.index+1)#Needs to be ref for non fixed variables XYZR
    '''
    if 'future' in sourcezone.name:
        state = 'future_ext_bs_Cs'
    else:
        state = 'ext_bs_Cs'
    '''
    state = 'ext_bs_Cs'
    if not mpexists:
        eq('{'+state+'}=if({beta_star}['+src+']>'+str(betastarblank)+
                             ',{Cs [km/s]}['+src+'],'+str(2*sonicspeed)+')',
                             zones=[0])
    else:
        #UPDATE
        eq('{'+state+'}=if({X [R]}=='+str(xtail)+','+str(sonicspeed)+','+
                       'if({beta_star}['+src+']>'+str(betastarblank)+'&&'+
                          '{X [R]}>'+str(xtail)+',{Cs [km/s]}['+src+'],0))',
                          zones=[0])
    return sourcezone.dataset.variable('ext_bs_Cs').index

def calc_bs_state2(deltaS, betastarblank, xtail, zones, *,
                   mpexists=False):
    """Function creates equation for the bow shock region
    Inputs
        deltaS- gas dynamics speed of sound: sqrt(gamma*P/rho)[km/s]
        betastarblank- blanking condition for betastar
        mpexists(boolean)- used to define the back end condition
    Return
        index- index for the created variable
    """
    i_primary = kwargs.get('mainZoneIndex',0)
    eq = tp.data.operate.execute_equation
    state = 'ext_bs_Ds'
    if True:
        eq('{'+state+'}=if({beta_star}>'+str(betastarblank)+
                  '&& ({s [Re^4/s^2kg^2/3]}/'+
                      '({1Ds [Re^4/s^2kg^2/3]}+1e-20)<'+str(deltaS)+
                            '),1,0)',zones=zones)
    return zones[i_primary].dataset.variable('ext_bs_Ds').index

def calc_xslice_state(varname,closed_name,zones):
    eq = tp.data.operate.execute_equation
    eq('{'+varname+'}=if({'+closed_name+'}==1&&{X [R]}==0,1,0)',zones=zones)
    return zones[0].dataset.variable(varname).index

def calc_ps_qDp_state(ps_qDp,closed_var,lshelllim,bxmax,zones,**kwargs):
    """Function creates equation for the plasmasheet or quasi diploar
        region within the confines of closed field line surface indicated
        by closed_var
    Inputs
        closed_var- variable name for magnetopause zone
        lshselllim- dipolar coord l shell limit
        bxmax- x comp B limit
    Return
        index- index for the created variable
    """
    i_primary = kwargs.get('mainZoneIndex',0)
    eq = tp.data.operate.execute_equation
    Lvar = kwargs.get('Lvar','Lshell')
    state = 'ms_'+ps_qDp+'_L>'
    if ps_qDp == 'ps':
        eq('{'+state+'} = if({'+closed_var+'}==1&&'+
                                    '{'+Lvar+'}>'+lshelllim+'&&'+
                            '{r [R]}>='+str(kwargs.get('inner_r',3))+'&&'+
                                   'abs({B_x [nT]})<'+bxmax+'&&'+
                                     '{X [R]}<0,1,0)',zones=zones)
    elif ps_qDp == 'qDp':
        eq('{'+state+'} = if({'+closed_var+'}>0&&'+
                                    '{'+Lvar+'}>'+lshelllim+'&&'+
                            '{r [R]}>='+str(kwargs.get('inner_r',3))+'&&'+
                                  '(abs({B_x [nT]})>'+bxmax+'||'+
                                    '{X [R]}>0),1,0)',zones=zones)
    elif ps_qDp == 'closed':
        eq('{'+state+'} = if({'+closed_var+'}>0&&'+
                                  '{'+Lvar+'}>='+lshelllim+'&&'+
              '{r [R]}>='+str(kwargs.get('inner_r',3))+',1,0)',zones=zones)
    return zones[i_primary].dataset.variable(state).index


def calc_rc_state(closed_var, lshellmax, zones, *,
                  Lvar='Lshell', **kwargs):
    """Function creates eq for region containing ring currents within the
        confines of closed field line surface indicated by closed_var
    Inputs
        closed_var- variable name for magnetopause zone
        lshsellmax- dipolar coord l shell limit
        kwargs:
            rmin(str)- minimum radius for inner boundary
    Return
        index- index for the created variable
    """
    i_primary = kwargs.get('mainZoneIndex',0)
    eq = tp.data.operate.execute_equation
    state = 'ms_rc_L='
    eq('{'+state+'} = if({'+closed_var+'}==1&&'+
                           '{r [R]}>='+str(kwargs.get('inner_r',3))+'&&'+
                                    '{'+Lvar+'}<'+lshellmax+',1,0)',
                                    zones=zones)
    return zones[i_primary].dataset.variable(state).index

def calc_plasmasheet_state(zones,xlims,ylims,zlims,**kwargs):
    """Function creates equation for the plasmasheet which is constricted to
        xyz limits
    Inputs
        closed_var
        zones
        xlims,ylims,zlims
        kwargs:
            None
    Return
        index- index of the created state variable
    """
    eq = tp.data.operate.execute_equation
    threshold = str(kwargs.get('plasmasheet_value',0.3))
    variable = kwargs.get('plasmasheet_var','curl_unitb_y')
    state = 'plasmasheet_'+threshold
    eqstr = ('{'+state+'}=if({'+variable+'}>'+threshold+
                      '&&{X [R]}>'+str(xlims[0])+' && {X [R]}<'+str(xlims[1])+
                      '&&{Y [R]}>'+str(ylims[0])+' && {Y [R]}<'+str(ylims[1])+
                      '&&{Z [R]}>'+str(zlims[0])+' && {Z [R]}<'+str(zlims[1])+
                          ',1,0)')
    print(eqstr)
    eq(eqstr, zones=zones)
    return zones[0].dataset.variable(state).index

def calc_lobe_state(mp_var, northsouth, zones, **kwargs):
    """Function creates equation for north or south lobe within the
        confines of magnetopause surface indicated by mp_var
    Inputs
        mp_var- variable name for magnetopause zone
        northsouth- which lobe to create equation for
        status- optional change to the status variable name
    Return
        index- index for the created variable
    """
    eq, cc = tp.data.operate.execute_equation, ValueLocation.CellCentered
    status = kwargs.get('status','Status')
    r = str(kwargs.get('inner_r',3))
    i_primary = kwargs.get('mainZoneIndex',0)
    #set state name
    if 'both' in northsouth: state = 'Lobe'
    else: state = northsouth[0].upper()+'Lobe'
    #calculate
    if northsouth == 'north':
        eqstr=('{'+state+'} =if(({'+mp_var+'}==1&&{r [R]}>='+r+')&&'+
                               '{'+status+'}==2,1,0)')
    elif northsouth == 'south':
        eqstr=('{'+state+'} =if(({'+mp_var+'}==1&&{r [R]}>='+r+')&&'+
                               '{'+status+'}==1,1,0)')
    else:
        eqstr=('{'+state+'} =if(({'+mp_var+'}==1&&{r [R]}>='+r+')&&'+
              '({'+status+'}==1 || {'+status+'}==2),1,0)')
    eq(eqstr, zones=zones)
    return zones[i_primary].dataset.variable(state).index

def calc_transition_rho_state(xmax, xmin, hmax, rhomax, rhomin, uBmin,
                              sourcezone):
    """Function creates equation in tecplot representing surface
    Inputs
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
    Outputs
        created variable index
    """
    ##OBSOLETE
    drho = rhomax-rhomin
    eq = tp.data.operate.execute_equation
    eq('{mp_rho_transition} = '+
        'IF({X [R]} >'+str(xmin)+' &&'+
        '{X [R]} <'+str(xmax)+' && {h} < '+str(hmax)+','+
           'IF({Rho [amu/cm^3]}<(atan({X [R]}+5)+pi/2)/pi*'+str(drho)+'+'+
            str(rhomin)+'||({uB [J/Re^3]}>'+str(uBmin)+'), 1, 0), 0)')
    return tp.active_frame().dataset.variable('mp_rho_transition').index

def calc_betastar_state(zonename, zones, **kwargs):
    """Function creates equation in tecplot representing surface
    Inputs
        zonename
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
        closed_zone- zone object, None to omit closed zone in equation
    Outputs
        created variable index
    """
    #Values needed to put into equation string passed to Tecplot
    #xmax = str(kwargs.get('x_subsolar','30'))
    xmin = str(kwargs.get('tail_cap',-20))
    core_r = str(kwargs.get('inner_r',3))
    betamax = str(kwargs.get('mpbetastar',0.7))
    closed_zone = kwargs.get('closed_zone')
    eq, cc = tp.data.operate.execute_equation, ValueLocation.CellCentered
    if kwargs.get('sunward_pole',False):
        #PALEO variant for dipole facing subsolar point
        eqstr=('{'+zonename+'}=IF({X [R]}>'+xmin+'&&'+'{r [R]}>='+core_r)
    else:
        eqstr=('{'+zonename+'}=IF({X [R]} >'+xmin+'&&'+'{r [R]} >='+core_r)
    if 'Status' in zones[kwargs.get('mainZone',0)].dataset.variable_names:
        eqstr+='&&{Status}>0'
    eqstr=(eqstr+',IF({beta_star}<'+betamax+',1,')
    if type(closed_zone) != type(None):
        eqstr =(eqstr+'IF({'+closed_zone.name+'} == 1,1,0))')
    else:
        eqstr =(eqstr+'IF({Status} == 3,1,0))')
        #eqstr =(eqstr+'0)')
    eqstr =(eqstr+',0)')
    eq(eqstr, zones=zones)
    return tp.active_frame().dataset.variable(zonename).index

def calc_delta_state(t0state, t1state):
    """Function creates equation representing volume between now and
        future magnetosphere
    Inputs
        t0state, t1state- names for use in equation for current and
                          future times
    """
    eq = tp.data.operate.execute_equation
    eq('{delta_'+t0state+'}=IF({'+t1state+'}==1 && {'+t0state+'}==0, 1,'+
                        'IF({'+t1state+'}==0 && {'+t0state+'}==1,-1, 0))',
                                zones=[0])

def calc_iso_rho_state(xmax, xmin, hmax, rhomax, rmin_north, rmin_south,
                       sourcezone):
    """Function creates equation in tecplot representing surface
    Inputs
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
    Outputs
        created variable index
    """
    ##OBSOLETE
    eq = tp.data.operate.execute_equation
    eq('{mp_rho_innerradius} = '+
        'IF({X [R]} >'+str(xmin-2)+'&&'+
        '{X [R]} <'+str(xmax)+'&& {h} < '+str(hmax)+','+
            'IF({Rho [amu/cm^3]}<'+str(rhomax)+', 1,'+
                'IF(({r [R]} <'+str(rmin_north)+'&&{Z [R]}>0)||'+
                   '({r [R]} <'+str(rmin_south)+'&&{Z [R]}<0),1,0)),0)')
    return tp.active_frame().dataset.variable('mp_rho_innerradius').index

def calc_shue_state(field_data, mode, x_subsolar,xtail,zones,*, dx=10):
    """Function creates state variable for magnetopause surface based on
        Shue emperical model
    Inputs
        field_data
        mode
        x_subsolar- if None will calculate with dayside fieldlines
        xtail- limit for how far to extend in negative x direction
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    state = mode
    for source in zones:
        src=str(source.index+1)#Needs to be ref for non fixed variables XYZR
        #Probe field data at x_subsolar + dx to find Bz and Dp
        Bz=tp.data.query.probe_at_position(x_subsolar+dx,0,0)[source.index][9]
        Dp_index = field_data.variable('Dp *').index
        Dp=tp.data.query.probe_at_position(x_subsolar+dx,0,0)[
                                                       source.index][Dp_index]
        #Get r0 and alpha based on IMF conditions
        if mode == 'shue97':
            r0, alpha = r0_alpha_1997(Bz, Dp)
        else:
            r0, alpha = r0_alpha_1998(Bz, Dp)
        eq = tp.data.operate.execute_equation
        eq('{theta} = IF({X [R]}>0,atan({h}/{X [R]}), pi-atan({h}/{X [R]}))',
           zones=[source])
        eq('{r'+state+'} = '+str(r0)+'*(2/(1+cos({theta})))**'+str(alpha),
           zones=[source])
        eq('{'+state+'} = IF(({r [R]} < {r'+state+'}) &&'+
                            '({X [R]} > '+str(xtail)+'), 1, 0)',zones=[source])
    return field_data.variable(mode).index

def calc_sphere_state(mode, xc, yc, zc, rmax, zones,*, rmin=0):
    """Function creates state variable for a simple box
    Inputs
        mode
        xc, yc, zc- locations for sphere center
        rmax- sphere radius
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq = tp.data.operate.execute_equation
    i_primary = kwargs.get('mainZoneIndex',0)
    '''
    if 'future' in sourcezone.name: state = 'future_'+mode
    else: state = mode
    '''
    state = mode+str(rmax)
    eq('{'+state+'} = IF(sqrt(({X [R]} -'+str(xc)+')**2 +'+
                            '({Y [R]} -'+str(yc)+')**2 +'+
                            '({Z [R]} -'+str(zc)+')**2) <'+
                             str(rmax)+'&&'+
                        'sqrt(({X [R]} -'+str(xc)+')**2 +'+
                            '({Y [R]} -'+str(yc)+')**2 +'+
                            '({Z [R]} -'+str(zc)+')**2) >'+
                            str(rmin)+',1, 0)',zones=zones)
    return zones[i_primary].dataset.variable(state).index

def calc_closed_state(statename, status_key,status_val,xmin,source,core_r):
    """Function creates state variable for the closed fieldline region
    Inputs
        status_key/val-string key and value used to denote closed fldlin
        xmin- minimum cuttoff value
        core_r- inner boundary cuttoff
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq, cc = tp.data.operate.execute_equation, ValueLocation.CellCentered
    eqstr = ('{'+statename+'}=IF({X [R]}>'+str(xmin)+'&&{r [R]}>='+str(core_r)+
                    ',IF({'+status_key+'}['+str(source+1)+']>=3,1,0), 0)')
    eq(eqstr,zones=[source])
    return tp.active_frame().dataset.variable(statename).index

def calc_box_state(mode, xmax, xmin, ymax, ymin, zmax, zmin,zones):
    """Function creates state variable for a simple box
    Inputs
        mode
        xmax,xmin...zmin- locations for box vertices
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq = tp.data.operate.execute_equation
    i_primary = kwargs.get('mainZoneIndex',0)
    state = mode
    eq('{'+state+'} = IF(({X [R]} >'+str(xmin)+') &&'+
                       '({X [R]} <'+str(xmax)+') &&'+
                       '({Y [R]} >'+str(ymin)+') &&'+
                       '({Y [R]} <'+str(ymax)+') &&'+
                       '({Z [R]} >'+str(zmin)+') &&'+
                       '({Z [R]} <'+str(zmax)+'), 1, 0)',zones=zones)
    return zones[i_primary].variable(state).index

def abs_to_timestamp(abstime):
    """Function converts absolute time in sec to a timestamp list
    Inputs
        abstime
    Outpus
        timestamp
    """
    secyear = 60*60*24*12*30
    year = np.floor(abstime/secyear)
    month = np.floor((abstime/secyear-year)*12)
    day = np.floor((((abstime/secyear-year)*12)-month)*30)
    hour = np.floor((((((abstime/secyear-year)*12)-month)*30)-day)*24)
    minute = np.floor((((((((abstime/secyear-year)*12)-month)*30)
                             -day)*24)-hour)*60)
    second = np.floor((((((((((abstime/secyear-year)*12)-month)*30)
                             -day)*24)-hour)*60)-minute)*60)
    timestamp = [year, month, day, hour, minute, second, abstime]
    return timestamp

def write_to_timelog(timelogname, time, data):
    """Function for writing the results from the current file to a file that contains time integrated data
    Inputs
        timelogname
        time- datetime object
        data- pandas DataFrame object that will be written into the file
    """
    #get the time entries for this file
    timestamp = [time.year, time.month, time.day, time.hour, time.minute, time.second]
    #write data to file
    with open(timelogname, 'a') as log:
        log.seek(0,2)
        log.write('\n')
        for entry in timestamp:
            log.write(str(entry)+', ')
        for num in data.values[0]:
            log.write(str(num)+',')
    return timestamp


# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()

    os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    DATAFILE = sys.argv[1]
    print('Processing '+DATAFILE)
    tp.new_layout()

    #Load .plt file
    log.info('loading .plt and reformatting')
    SWMF_DATA = tp.data.load_tecplot(DATAFILE)
    SWMF_DATA.zone(0).name = 'global_field'
    OUTPUTNAME = DATAFILE.split('e')[1].split('-000.')[0]+'done'
    print(SWMF_DATA)

    #Set parameters
    #DaySide
    N_AZIMUTH_DAY = 5
    AZIMUTH_MAX = 122
    R_MAX = 30
    R_MIN = 3.5
    ITR_MAX = 100
    TOL = 0.1
    AZIMUTH_RANGE = [np.deg2rad(-1*AZIMUTH_MAX), np.deg2rad(AZIMUTH_MAX)]
    PHI = np.linspace(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1], N_AZIMUTH_DAY)

    #Tail
    N_AZIMUTH_TAIL = 5
    RHO_MAX = 50
    RHO_STEP = 0.5
    X_TAIL_CAP = -20
    PSI = np.linspace(-pi*(1-pi/N_AZIMUTH_TAIL), pi, N_AZIMUTH_TAIL)

    #Cylindrical zone
    N_SLICE = 40
    N_ALPHA = 50

    with tp.session.suspend():
        tp.macro.execute_command("$!FrameName = 'main'")
        #Create R from cartesian coordinates
        tp.data.operate.execute_equation(
                    '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
        #Create Dayside Magnetopause field lines
        calc_dayside_mp(PHI, R_MAX, R_MIN, ITR_MAX, TOL)

        #Create Tail magnetopause field lines
        calc_tail_mp(PSI, X_TAIL_CAP, RHO_MAX, RHO_STEP)
        #Create Theta and Phi coordinates for all points in domain
        tp.data.operate.execute_equation(
                                   '{phi} = atan({Y [R]}/({X [R]}+1e-24))')
        tp.data.operate.execute_equation(
                                   '{theta} = acos({Z [R]}/{r [R]}) * '+
                                    '({X [R]}+1e-24) / abs({X [R]}+1e-24)')

        #port stream data to pandas DataFrame object
        STREAM_ZONE_LIST = np.linspace(2,SWMF_DATA.num_zones,
                                       SWMF_DATA.num_zones-2+1)

        STREAM_DF, X_MAX = dump_to_pandas(STREAM_ZONE_LIST, [1,2,3],
                                          'stream_points.csv')

