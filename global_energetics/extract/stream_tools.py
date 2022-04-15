#!/usr/bin/env python3
"""Functions for identifying surfaces from field data
"""
import logging as log
import os, warnings
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import scipy.spatial as space
#import spacepy
#from spacepy import coordinates as coord
#from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#Interpackage modules
from global_energetics.extract import shue
from global_energetics.extract.shue import (r_shue, r0_alpha_1997,
                                                    r0_alpha_1998)
from progress.bar import Bar

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
        rmin = field_data.zone(zone_name+'*').values('r *').min()
        if rmin < r_seed:
            isclosed = True
            return isclosed
        else:
            isclosed = False
            return isclosed
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

'''
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

def rotation(angle, axis):
    """Function returns rotation matrix given axis and angle
    Inputs
        angle
        axis
    Outputs
        matrix
    """
    if axis == 'x' or axis == 'X':
        matrix = [[1,           0,          0],
                  [0,  cos(angle), sin(angle)],
                  [0, -sin(angle), cos(angle)]]
    elif axis == 'y' or axis == 'Y':
        matrix = [[ cos(angle), 0, sin(angle)],
                  [0,           1,          0],
                  [-sin(angle), 0, cos(angle)]]
    elif axis == 'z' or axis == 'Z':
        matrix = [[ cos(angle), sin(angle), 0],
                  [-sin(angle), cos(angle), 0],
                  [0,           0,          1]]
    return matrix

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
def get_surf_geom_variables(zone):
    """Function calculates variables for new zone based only on geometry,
        independent of what analysis will be performed on surface
    Inputs
        zone(Zone)- tecplot zone to calculate variables
    """
    #Get grid dependent variables
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'GRIDKUNITNORMAL VALUELOCATION = '+
                                      'CELLCENTERED')
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'CELLCENTERED')
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    #Generate cellcentered versions of postitional variables
    for var in ['X [R]','Y [R]','Z [R]','r [R]',
                'B_x [nT]','B_y [nT]','B_z [nT]','Bdx','Bdy','Bdz']:
        if var in zone.dataset.variable_names:
            newvar = var.split(' ')[0].lower()+'_cc'
            eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                        zones=[zone.index])
    #Create a DataFrame for easy manipulations
    x_ccvalues =zone.values('x_cc').as_numpy_array()
    xnormals = zone.values('X GRID K Unit Normal').as_numpy_array()
    df = pd.DataFrame({'x_cc':x_ccvalues,'normal':xnormals})
    #Check that surface normals are pointing outward from surface
    #Spherical inner boundary surface case (want them to point inwards)
    if 'innerbound' in zone.name:
        if df[df['x_cc']==df['x_cc'].min()]['normal'].mean() < 0:
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
    elif 'bs' in zone.name:
        #Look at dayside max plane for bowshock
        if (len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']>0)]) <
            len(df[(df['x_cc']==df['x_cc'].max())&(df['normal']<0)])):
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
    else:
        #Look at tail cuttoff plane for other cases
        if (len(df[(df['x_cc']==df['x_cc'].min())&(df['normal']>0)]) >
            len(df[(df['x_cc']==df['x_cc'].min())&(df['normal']<0)])):
            eq('{surface_normal_x} = -1*{X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = -1*{Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = -1*{Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
        else:
            eq('{surface_normal_x} = {X Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_y} = {Y Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
            eq('{surface_normal_z} = {Z Grid K Unit Normal}',
               zones=[zone.index], value_location=CC)
    if ('mp' in zone.name) and ('innerbound' not in zone.name):
        #Store a helpful 'htail' value in aux data for potential later use
        xvals = zone.values('X *').as_numpy_array()
        hvals = zone.values('h').as_numpy_array()
        zone.aux_data.update({'hmin':
               hvals[np.where(np.logical_and(xvals<-5,xvals>-8))].min()})

def get_day_flank_tail(zone):
    """Function assigns variables to represent dayside, flank, and tail,
        typically for the magnetopause zone
    Inputs
        zone(Zone)- tecplot Zone object to do calculation
    """
    eq = tp.data.operate.execute_equation
    #Check that geometry variables have already been calculated
    tail_h = float(zone.dataset.zone('mp*').aux_data['hmin'])
    eq('{Day} = IF({X [R]}>0,1,0)', zones=[zone.index])
    eq('{Tail} = IF(({X [R]}<-5&&{h}<'+str(tail_h)+'*0.8)||'+
             '({X [R]}<-10&&{h}<'+str(tail_h)+'),1,0)',zones=[zone.index])
    eq('{Flank} = IF({Day}==0&&{Tail}==0,1,0)', zones=[zone.index])
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
    """
    eq = tp.data.operate.execute_equation
    #Check that geometry variables have already been calculated
    assert any([x!=0 for x in
        zone.values('surface_normal_x').as_numpy_array()]), ('Surface '+
                                              'geometry not calculated!')
    eq('{ux_cc}={U_x [km/s]}', value_location=ValueLocation.CellCentered)
    eq('{uy_cc}={U_y [km/s]}', value_location=ValueLocation.CellCentered)
    eq('{uz_cc}={U_z [km/s]}', value_location=ValueLocation.CellCentered)
    if 'energy' in analysis_type or analysis_type == 'all':
        eq('{Utot_cc}={Utot [J/Re^3]}',
                         value_location=ValueLocation.CellCentered)
    eq('{Bx_cc}={B_x [nT]}', value_location=ValueLocation.CellCentered)
    eq('{By_cc}={B_y [nT]}', value_location=ValueLocation.CellCentered)
    eq('{Bz_cc}={B_z [nT]}', value_location=ValueLocation.CellCentered)
    eq('{Status_cc}={Status}', value_location=ValueLocation.CellCentered)
    #eq('{W_cc}={W [km/s/Re]}', value_location=ValueLocation.CellCentered)
    eq('{W_cc}=0', value_location=ValueLocation.CellCentered)
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
        eq('{Bdx_cc}={Bdx}', value_location=ValueLocation.CellCentered)
        eq('{Bdy_cc}={Bdy}', value_location=ValueLocation.CellCentered)
        eq('{Bdz_cc}={Bdz}', value_location=ValueLocation.CellCentered)
        get_virials()
    ##Different prefixes allow for calculation of surface fluxes using 
    #   multiple sets of flowfield variables (denoted by the prefix)
    if 'energy' in analysis_type:
        prefixlist = ['']
        for add in prefixlist:
            ##############################################################
            #Velocity normal to the surface
            eq('{Unorm [km/s]}=sqrt(({U_x [km/s]}*{surface_normal_x})**2+'+
                                '({U_y [km/s]}*{surface_normal_y})**2+'+
                                '({U_z [km/s]}*{surface_normal_z})**2)',
                            value_location=ValueLocation.CellCentered,
                            zones=[zone.index])
            ##############################################################
            #Normal Poynting Flux
            eq('{'+add+'ExB_net [W/Re^2]} = {Bmag [nT]}**2/(4*pi*1e-7)*1e-9'+
                                            '*6371**2*('+
                                        '{U_x [km/s]}*{surface_normal_x}'+
                                        '+{U_y [km/s]}*{surface_normal_y}'+
                                        '+{U_z [km/s]}*{surface_normal_z})-'+
                '({B_x [nT]}*({U_x [km/s]})+'+
                '{B_y [nT]}*({U_y [km/s]})+'+
                '{B_z [nT]}*({U_z [km/s]}))'+
                                        '*({B_x [nT]}*{surface_normal_x}+'+
                                        '{B_y [nT]}*{surface_normal_y}+'+
                                        '{B_z [nT]}*{surface_normal_z})'+
                                            '/(4*pi*1e-7)*1e-9*6371**2',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            #Split into + and - flux
            eq('{'+add+'ExB_escape [W/Re^2]} = max({'+add+'ExB_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            eq('{'+add+'ExB_injection [W/Re^2]} = min({'+add+'ExB_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            ##############################################################
            #Normal Total Pressure Flux
            eq('{'+add+'P0_net [W/Re^2]} = (1/2*{Dp [nPa]}+2.5*{P [nPa]})'+
                                            '*6371**2*('+
                                        '{U_x [km/s]}*{surface_normal_x}'+
                                        '+{U_y [km/s]}*{surface_normal_y}'+
                                        '+{U_z [km/s]}*{surface_normal_z})',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            #Split into + and - flux
            eq('{'+add+'P0_escape [W/Re^2]} ='+
                        'max({'+add+'P0_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            eq('{'+add+'P0_injection [W/Re^2]} ='+
                        'min({'+add+'P0_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            ##############################################################
            #Normal Total Energy Flux
            eq('{'+add+'K_net [W/Re^2]}='+
                   '{P0_net [W/Re^2]}+{ExB_net [W/Re^2]}',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            #Split into + and - flux
            eq('{'+add+'K_escape [W/Re^2]}=max({'+add+'K_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])
            eq('{'+add+'K_injection [W/Re^2]} ='+
                         'min({'+add+'K_net [W/Re^2]},0)',
                value_location=ValueLocation.CellCentered,
                zones=[zone.index])


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
        oneD_data = oneD_data.append(pd.DataFrame([data],
                                     columns=field_data.variable_names),
                                     ignore_index=False)
    #Create new global variables
    for var in field_data.variable_names:
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

def get_dipole_field(auxdata, *, B0=31000):
    """Function calculates dipole field in given coordinate system based on
        current time in UTC
    Inputs
        auxdata- tecplot object containing key data pairs
        B0- surface magnetic field strength
    """
    #Determine dipole vector in given coord system
    theta_tilt = float(auxdata['BTHETATILT'])
    axis = [sin(deg2rad(theta_tilt)), 0, -1*cos(deg2rad(theta_tilt))]
    #Create dipole matrix
    ######################################
    #   (3x^2-r^2)      3xy         3xz
    #B =    3yx     (3y^2-r^2)      3yz    * vec{m}
    #       3zx         3zy     (3z^2-r^2)
    ######################################
    M11 = '(3*{X [R]}**2-{r [R]}**2)'
    M12 = '3*{X [R]}*{Y [R]}'
    M13 = '3*{X [R]}*{Z [R]}'
    M21 = M12
    M22 = '(3*{Y [R]}**2-{r [R]}**2)'
    M23 = '3*{Y [R]}*{Z [R]}'
    M31 = M13
    M32 = M23
    M33 = '(3*{Z [R]}**2-{r [R]}**2)'
    #Multiply dipole matrix by dipole vector
    d_x='{Bdx}='+str(B0)+'/{r [R]}**5*('+(M11+'*'+str(axis[0])+'+'+
                                            M12+'*'+str(axis[1])+'+'+
                                            M13+'*'+str(axis[2]))+')'
    d_y='{Bdy}='+str(B0)+'/{r [R]}**5*('+(M21+'*'+str(axis[0])+'+'+
                                            M22+'*'+str(axis[1])+'+'+
                                            M23+'*'+str(axis[2]))+')'
    d_z='{Bdz}='+str(B0)+'/{r [R]}**5*('+(M31+'*'+str(axis[0])+'+'+
                                            M32+'*'+str(axis[1])+'+'+
                                            M33+'*'+str(axis[2]))+')'
    #Return equation strings to be evaluated
    return d_x, d_y, d_z

def mag2cart(lat,lon,btheta):
    """
    """
    #find xyz_mag
    x_mag, y_mag, z_mag = sph_to_cart(1,lat,lon)
    #get rotation matrix
    rot = rotation(-btheta*pi/180,axis='y')
    #find new points by rotation
    return np.matmul(rot,[x_mag,y_mag,z_mag])


def get_global_variables(field_data, analysis_type, **kwargs):
    """Function calculates values for energetics tracing
    Inputs
        field_data- tecplot Dataset class containing 3D field data
        kwargs:
            is3D- if all 3 dimensions are present
    """
    ######################################################################
    eq = tp.data.operate.execute_equation
    #General equations
    if (any([var.find('J_')!=-1 for var in field_data.variable_names])and
         any([var.find('`mA')!=-1 for var in field_data.variable_names])):
        field_data.variable('J_x*').name = 'J_x [uA/m^2]'
        field_data.variable('J_y*').name = 'J_y [uA/m^2]'
        field_data.variable('J_z*').name = 'J_z [uA/m^2]'
    #if 'J_x [`mA/m^2]' in ds.variable_names:
    #Useful spatial variables
    if kwargs.get('is3D',True):
        eq('{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
        eq('{h} = sqrt({Y [R]}**2+{Z [R]}**2)')
        aux = field_data.zone('global_field').aux_data
        #Dipolar coordinate variables
        eq('{mXhat_x} = sin(('+aux['BTHETATILT']+'+90)*pi/180)')
        eq('{mXhat_y} = 0')
        eq('{mXhat_z} = -1*cos(('+aux['BTHETATILT']+'+90)*pi/180)')
        eq('{mZhat_x} = sin('+aux['BTHETATILT']+'*pi/180)')
        eq('{mZhat_y} = 0')
        eq('{mZhat_z} = -1*cos('+aux['BTHETATILT']+'*pi/180)')
        eq('{lambda}=asin(({mZhat_x}*{X [R]}+{mZhat_z}*{Z [R]})/{r [R]})')
        eq('{Lshell} = {r [R]}/cos({lambda})**2')
        eq('{theta [deg]} = -180/pi*{lambda}')
    else:
        if 'XY_zone_index' in kwargs:
            eq('{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2)',
                zones=[kwargs.get('XY_zone_index',1),
                       kwargs.get('XYTri_index',6)])
        else:
            eq('{r [R]} = sqrt({X [R]}**2 + {Z [R]}**2)',
                zones=[kwargs.get('XZ_zone_index',0),
                       kwargs.get('XZTri_index',2)])
    #Dynamic Pressure
    eq('{Dp [nPa]} = {Rho [amu/cm^3]}*1e6*1.6605e-27*'+
              '({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*1e6*1e9',
        value_location=ValueLocation.CellCentered)
    #Sonic speed
    eq('{Cs [km/s]} = sqrt(5/3*{P [nPa]}/{Rho [amu/cm^3]}/6.022)*10**3')
    #Plasma Beta
    eq('{beta}=({P [nPa]})/({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                '*(2*4*pi*1e-7)*1e9')
    #Plasma Beta* using total pressure
    eq('{beta_star}=({P [nPa]}+{Dp [nPa]})/'+
                          '({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                '*(2*4*pi*1e-7)*1e9')
    eq('{Bmag [nT]} =sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)',
        value_location=ValueLocation.CellCentered)
    ######################################################################
    #Fieldlinemaping
    if ('OCFLB' in analysis_type or analysis_type=='all') and kwargs.get('is3D',True):
        eq('{Xd [R]}= {mXhat_x}*({X [R]}*{mXhat_x}+{Z [R]}*{mXhat_z})')
        eq('{Zd [R]}= {mZhat_z}*({X [R]}*{mZhat_x}+{Z [R]}*{mZhat_z})')
        eq('{phi} = atan2({Y [R]}, {Xd [R]})')
        eq('{req} = 2.7/(cos({lambda})**2)')
        eq('{lambda2} =sqrt(acos(1/{req}))')
        eq('{X_r1project} = 1*cos({phi})*sin(pi/2-{lambda2})')
        eq('{Y_r1project} = 1*sin({phi})*sin(pi/2-{lambda2})')
        eq('{Z_r1project} = 1*cos(pi/2-{lambda2})')
    ######################################################################
    #Virial only intermediate terms
    if 'virial' in analysis_type or analysis_type=='all':
        #Density times velocity between now and next timestep
        eq('{rhoUx_cc}={Rho [amu/cm^3]}*{U_x [km/s]}',
                         value_location=ValueLocation.CellCentered)
        eq('{rhoUy_cc}={Rho [amu/cm^3]}*{U_y [km/s]}',
                         value_location=ValueLocation.CellCentered)
        eq('{rhoUz_cc}={Rho [amu/cm^3]}*{U_z [km/s]}',
                         value_location=ValueLocation.CellCentered)
        #Advection term
        eq('{rhoU_r [Js/Re^3]} = {Rho [amu/cm^3]}*1.6605e6*6.371**4*('+
                                              '{U_x [km/s]}*{X [R]}+'+
                                              '{U_y [km/s]}*{Y [R]}+'+
                                              '{U_z [km/s]}*{Z [R]})',
                          value_location=ValueLocation.CellCentered)
        #Dipole field (requires coordsys and UT information!!!)
        Bdx_eq,Bdy_eq,Bdz_eq = get_dipole_field(aux)
        eq(Bdx_eq); eq(Bdy_eq); eq(Bdz_eq)
        eq('{Bdmag [nT]} =sqrt({Bdx}**2+{Bdy}**2+{Bdz}**2)',
                          value_location=ValueLocation.CellCentered)
    ######################################################################
    #Volumetric energy terms
    #Total Magnetic Energy per volume
    eq('{uB [J/Re^3]} = {Bmag [nT]}**2'+
                        '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
        value_location=ValueLocation.CellCentered)
    #Thermal Pressure in Energy units
    eq('{Pth [J/Re^3]} = {P [nPa]}*6371**3',
        value_location=ValueLocation.CellCentered)
    #Kinetic energy per volume
    eq('{KE [J/Re^3]} = {Dp [nPa]}/2*6371**3',
        value_location=ValueLocation.CellCentered)
    if 'virial' in analysis_type or analysis_type=='all':
        #Dipole magnetic Energy
        eq('{uB_dipole [J/Re^3]} = {Bdmag [nT]}**2'+
                        '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
                          value_location=ValueLocation.CellCentered)
        #Disturbance Magnetic Energy per volume
        eq('{Virial Ub [J/Re^3]}= (({B_x [nT]}-{Bdx})**2+'+
                                  '({B_y [nT]}-{Bdy})**2+'+
                                  '({B_z [nT]}-{Bdz})**2)'+
                        '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
                          value_location=ValueLocation.CellCentered)
        #Special construction of hydrodynamic energy density for virial
        eq('{Virial 2x Uk [J/Re^3]} = 2*{KE [J/Re^3]}+{Pth [J/Re^3]}',
                          value_location=ValueLocation.CellCentered)
    #Hydrodynamic Energy Density
    eq('{uHydro [J/Re^3]} = ({P [nPa]}*1.5+{Dp [nPa]}/2)*6371**3',
                          value_location=ValueLocation.CellCentered)
    #Total Energy Density
    eq('{Utot [J/Re^3]} = {uHydro [J/Re^3]}+{uB [J/Re^3]}',
                          value_location=ValueLocation.CellCentered)
    if ('biotsavart' in analysis_type) or analysis_type=='all':
        eq('{dB_x [nT]} = -({Y [R]}*{J_z [uA/m^2]}-'+
                           '{Z [R]}*{J_y [uA/m^2]})*637.1/{r [R]}**3')
        eq('{dB_y [nT]} = -({Z [R]}*{J_x [uA/m^2]}-'+
                           '{X [R]}*{J_z [uA/m^2]})*637.1/{r [R]}**3')
        eq('{dB_z [nT]} = -({X [R]}*{J_y [uA/m^2]}-'+
                           '{Y [R]}*{J_x [uA/m^2]})*637.1/{r [R]}**3')
        eq('{dB [nT]} = {dB_x [nT]}*{mZhat_x}+{dB_z [nT]}*{mZhat_z}')

    ######################################################################
    if 'energy' in analysis_type or analysis_type=='all':
        #Energy Flux terms
        #Magnetic field unit vectors
        eq('{unitbx} ={B_x [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
                          value_location=ValueLocation.CellCentered)
        eq('{unitby} ={B_y [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
                          value_location=ValueLocation.CellCentered)
        eq('{unitbz} ={B_z [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
                          value_location=ValueLocation.CellCentered)
        #Field Aligned Current Magntitude
        eq('{J_par [uA/m^2]} = {unitbx}*{J_x [uA/m^2]} + '+
                              '{unitby}*{J_y [uA/m^2]} + '+
                              '{unitbz}*{J_z [uA/m^2]}',
                          value_location=ValueLocation.CellCentered)
        #Poynting Flux
        eq('{ExB_x [W/Re^2]} = {Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_x [km/s]})-{B_x [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
                          value_location=ValueLocation.CellCentered)
        eq('{ExB_y [W/Re^2]} = {Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_y [km/s]})-{B_y [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
                          value_location=ValueLocation.CellCentered)
        eq('{ExB_z [W/Re^2]} = {Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_z [km/s]})-{B_z [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
                          value_location=ValueLocation.CellCentered)
        #Total pressure Flux
        eq('{P0_x [W/Re^2]} = ({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_x [km/s]}',
                          value_location=ValueLocation.CellCentered)
        eq('{P0_y [W/Re^2]} = ({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_y [km/s]}',
                          value_location=ValueLocation.CellCentered)
        eq('{P0_z [W/Re^2]} = ({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_z [km/s]}',
                          value_location=ValueLocation.CellCentered)
        #Total Energy Flux
        eq('{K_x [W/Re^2]} = {P0_x [W/Re^2]}+{ExB_x [W/Re^2]}',
                          value_location=ValueLocation.CellCentered)
        eq('{K_y [W/Re^2]} = {P0_y [W/Re^2]}+{ExB_y [W/Re^2]}',
                          value_location=ValueLocation.CellCentered)
        eq('{K_z [W/Re^2]} = {P0_z [W/Re^2]}+{ExB_z [W/Re^2]}',
                          value_location=ValueLocation.CellCentered)
    ######################################################################
    if 'reconnect' in analysis_type:
        eq('{minus_uxB_x}=-({U_y [km/s]}*{B_z [nT]}-'+
                           '{U_z [km/s]}*{B_y [nT]})')
        eq('{minus_uxB_y}=-({U_z [km/s]}*{B_x [nT]}-'+
                           '{U_x [km/s]}*{B_z [nT]})')
        eq('{minus_uxB_z}=-({U_x [km/s]}*{B_y [nT]}-'+
                           '{U_y [km/s]}*{B_x [nT]})')
        eq('{E [uV/m]} = sqrt({minus_uxB_x}**2+'+
                             '{minus_uxB_y}**2+{minus_uxB_z}**2)')
        eq('{J [uA/m^2]} = sqrt({J_x [uA/m^2]}**2+'+
                               '{J_y [uA/m^2]}**2+{J_z [uA/m^2]}**2)')
        eq('{eta [m/S]} = IF({J [uA/m^2]}>0.002,'+
                            '{E [uV/m]}/({J [uA/m^2]}+1e-9),0)')
        eq('{eta [Re/S]} = {eta [m/S]}/(6371*1000)')
        tp.macro.execute_extended_command('CFDAnalyzer3',
                                          'CALCULATE FUNCTION = '+
                                          'CELLVOLUME VALUELOCATION = '+
                                          'CELLCENTERED')
        eq('{Cell Size [Re]} = {Cell Volume}**(1/3)')
        eq('{Reynolds_m_cell} = 4*pi*1e-4*'+
               'sqrt({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*'+
                                 '{Cell Size [Re]}/({eta [Re/S]}+1e-9)')
    ######################################################################
    if 'trackIM' in analysis_type:
        eq('{trackEth_acc [J/Re^3]} = {dp_acc [nPa]}*6371**3',
            value_location=ValueLocation.CellCentered)
        eq('{trackDp_acc [nPa]} = {drho_acc [amu/cm^3]}*1e6*1.6605e-27*'+
              '({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*1e6*1e9',
            value_location=ValueLocation.CellCentered)
        eq('{trackKE_acc [J/Re^3]} = {trackDp_acc [nPa]}*6371**3',
            value_location=ValueLocation.CellCentered)
        eq('{trackWth [W/Re^3]} = IF({dtime_acc [s]}>0,'+
                             '{trackEth_acc [J/Re^3]}/{dtime_acc [s]},0)',
            value_location=ValueLocation.CellCentered)
        eq('{trackWKE [W/Re^3]} = IF({dtime_acc [s]}>0,'+
                             '{trackKE_acc [J/Re^3]}/{dtime_acc [s]},0)',
            value_location=ValueLocation.CellCentered)
    if analysis_type=='all':
        eq('{KEpar [J/Re^3]} = {Rho [amu/cm^3]}/2 *'+
                                    '(({U_x [km/s]}*{unitbx})**2+'+
                                    '({U_y [km/s]}*{unitby})**2+'+
                                    '({U_z [km/s]}*{unitbz})**2) *'+
                                    '1e6*1.6605e-27*1e6*1e9*6371**3',
                          value_location=ValueLocation.CellCentered)
        eq('{KEperp [J/Re^3]} = {Rho [amu/cm^3]}/2 *'+
                   '(({U_y [km/s]}*{unitbz} - {U_z [km/s]}*{unitby})**2+'+
                    '({U_z [km/s]}*{unitbx} - {U_x [km/s]}*{unitbz})**2+'+
                    '({U_x [km/s]}*{unitby} - {U_y [km/s]}*{unitbx})**2)'+
                                       '*1e6*1.6605e-27*1e6*1e9*6371**3',
                          value_location=ValueLocation.CellCentered)
    #Terms with derivatives (experimental)
    if analysis_type=='development':
        #Vorticity
        eq('{W [km/s/Re]}=sqrt((ddy({U_z [km/s]})-ddz({U_y [km/s]}))**2+'+
                              '(ddz({U_x [km/s]})-ddx({U_z [km/s]}))**2+'+
                              '(ddx({U_y [km/s]})-ddy({U_x [km/s]}))**2)',
                          value_location=ValueLocation.CellCentered)

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

def setup_isosurface(iso_value, varindex, zonename, *,
                     contindex=7, isoindex=7, keep_condition=None,
                                              keep_cond_value=0,
                                              global_key='global_field',
                                              blankvar='',blankvalue=2,
                                              blankop=RelOp.LessThan):
    """Function creates an isosurface and then extracts and names the zone
    Inputs
        iso_value
        varindex, contindex, isoindex- storage locations on tecplot side
        zonename
        keep_condition- will keep connected region w/ max element & this
        keep_cond_value- value used for above condition
    Outputs
        newzone- primary zone created (w/ max elements)
        newzone2- secondary zone that meets keep condition
    """
    frame = tp.active_frame()
    plt = frame.plot()
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
                                    frame.dataset.variable(varindex).name,
                                    iso_value))
    orig_nzones = frame.dataset.num_zones
    #Check for blanking conditions
    if blankvar != '':
        plt.value_blanking.active = True
        blank = plt.value_blanking.constraint(1)
        blank.active = True
        blank.variable = frame.dataset.variable(blankvar)
        blank.comparison_operator = blankop
        blank.comparison_value = blankvalue
    try:
        tp.macro.execute_command('$!ExtractIsoSurfaces Group = {:d} '.format(
                                                                  isoindex+1)+
                                    'ExtractMode = OneZonePerConnectedRegion')
    except TecplotMacroError:
        print('Unable to create '+zonename+'!')
        return None, None
    iso.show = False
    #Turn off blanking
    if blankvar != '':
        plt.value_blanking.active = False
    #only keep zone with the highest number of elements, or meet condition
    nelements = 0
    for i in range(orig_nzones, frame.dataset.num_zones):
        if len(frame.dataset.zone(i).values('X *')) > nelements:
            nelements = len(frame.dataset.zone(i).values('X *'))
            keep_index = i
        if keep_condition == 'sphere':
            element_total = len(frame.dataset.zone(i).values('r *'))
            rvals = frame.dataset.zone(i).values('r *').as_numpy_array()
            elements_onsphere = len(np.where(
                                    abs(rvals-keep_cond_value)<0.5)[0])
            if (elements_onsphere/element_total > 0.9 and
                element_total >20):
                newzone2_key = frame.dataset.zone(i).name
                keep_alt = i
        else:
            keep_alt = None
    for i in reversed(range(orig_nzones, frame.dataset.num_zones)):
        if i != keep_index and i != keep_alt:
            frame.dataset.delete_zones(i)
        else:
            newzone_key = frame.dataset.zone(i).name
    newzone = frame.dataset.zone(newzone_key)
    newzone.name = zonename
    if keep_condition == None:
        return newzone, newzone
    else:
        newzone2 = frame.dataset.zone(newzone2_key)
        newzone2.name = zonename+'innerbound'
        return newzone, newzone2

def calc_state(mode, sourcezone, **kwargs):
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
    #       Simply create a function calc_MYNEWSURF_state and call it
    #       Recommend: zonename as input so variable name is automated
    #       See example use of 'assert' if any pre_recs are needed
    #####################################################################
    #Call calc_XYZ_state and return state_index and create zonename
    if 'iso_betastar' in mode:
        zonename = 'mp_'+mode
        if 'future' in sourcezone.name:
            zonename = 'future_'+mode
            closed_zone = kwargs.get('future_closed_zone')
        else:
            closed_zone = kwargs.get('closed_zone')
        state_index = calc_betastar_state(zonename, sourcezone.index,
                                          kwargs.get('x_subsolar'),
                                          kwargs.get('tail_cap', -20),
                                          50,kwargs.get('mpbetastar',0.7),
                                          kwargs.get('inner_r',3),
                                          closed_zone)
    elif mode == 'sphere':
        zonename = mode+str(kwargs.get('sp_rmax',3))
        state_index = calc_sphere_state(zonename, kwargs.get('sp_x',0),
                                        kwargs.get('sp_y',0),
                                        kwargs.get('sp_z',0),
                                        kwargs.get('sp_rmax',3),
                                        rmin=kwargs.get('sp_rmin',0))
    elif mode == 'box':
        zonename = mode
        state_index = calc_box_state(zonename,
                                     kwargs.get('box_xmax',-5),
                                     kwargs.get('box_xmin',-8),
                                     kwargs.get('box_ymax',5),
                                     kwargs.get('box_ymin',-5),
                                     kwargs.get('box_zmax',5),
                                     kwargs.get('box_zmin',-5))
    elif 'shue' in mode:
        if mode =='shue97':
            zonename = 'shu97'
        else:
            zonename = 'shue98'
        state_index = calc_shue_state(tp.active_frame().dataset, mode,
                                      kwargs.get('x_subsolar'),
                                      kwargs.get('tail_cap', -20))
    elif 'lcb' in mode:
        assert kwargs.get('do_trace',False) == False, (
                            "lcb mode only works with do_trace==False!")
        assert kwargs.get('closed_zone').index != None, (
                                    'No closed_zone present! Cant do lcb')
        if 'future' in sourcezone.name:
            zonename = 'future_mp_'+kwargs.get('closed_zone').name
            state_index = tp.active_frame().dataset.variable(
                              kwargs.get('future_closed_zone').name).index
        else:
            zonename = kwargs.get('closed_zone').name
            state_index = tp.active_frame().dataset.variable(
                                     kwargs.get('closed_zone').name).index
    elif 'lobe' in mode:
        mpvar = kwargs.get('mpvar', sourcezone.dataset.variable('mp*'))
        assert kwargs.get('do_trace',False) == False, (
                            "lobe mode only works with do_trace==False!")
        assert mpvar is not None,('magnetopause variable not found'+
                                  'cannot calculate lobe zone!')
        zonename = 'ms_'+mode
        if mode.lower().find('s') != -1:
            state_index = calc_lobe_state(mpvar.name, 'south')
        else:
            state_index = calc_lobe_state(mpvar.name, 'north')
    elif 'rc' in mode:
        assert type(kwargs.get('closed_zone')) != type(None), ('No'+
                                       ' closed_zone present! Cant do rc')
        zonename = 'ms_'+mode
        state_index = calc_rc_state(kwargs.get('closed_zone').name,
                                    str(kwargs.get('lshelllim',7)))
    elif 'ps' in mode:
        assert type(kwargs.get('closed_zone')) != type(None), ('No'+
                                       ' closed_zone present! Cant do ps')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('ps',
                                        kwargs.get('closed_zone').name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)))
    elif 'qDp' in mode:
        assert type(kwargs.get('closed_zone')) != type(None), ('No'+
                                      ' closed_zone present! Cant do qDp')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('qDp',
                                        kwargs.get('closed_zone').name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)))
    elif 'closed' in mode:
        assert type(kwargs.get('closed_zone')) != type(None), ('No'+
                                   ' closed_zone present! Cant do closed')
        zonename = 'ms_'+mode
        state_index = calc_ps_qDp_state('closed',
                                        kwargs.get('closed_zone').name,
                                        str(kwargs.get('lshelllim',7)),
                                        str(kwargs.get('bxmax',10)))
    elif 'bs' in mode:
        zonename = 'ext_'+mode
        state_index = calc_bs_state(kwargs.get('sonicspeed',34),
                                    kwargs.get('betastarblank',2),
                                    kwargs.get('tail_cap',-20),
                                    mpexists=('mpvar' in kwargs.keys()))
        #TEMPORARY REROUTE FOR BOW SHOCK MODE, SEEMS TO BE GRID RES ISSUE
        #state_index = tp.active_frame().dataset.variable('Cs *').index
        #zone, innerzone = setup_isosurface(kwargs.get('sonicspeed',34),
        #                    state_index, zonename, blankvar = 'beta_star',
        #                    blankvalue=kwargs.get('betastarblank',2))
        zone, innerzone = setup_isosurface(kwargs.get('sonicspeed',34),
                            state_index, zonename, blankvar = 'X *',
                            blankvalue=kwargs.get('tail_cap',-20))
        innerzone = None
        return zone, innerzone, state_index
    elif 'Jpar' in mode:
        #Make sure we have a place to find the regions of intense FAC's
        assert any(
                ['inner' in zn for zn in sourcezone.dataset.zone_names]),(
                                        'No inner boundary zone created, '
                                            +'unclear where to calc FAC!')
        #Warn user if there is multiple valid "inner" zone targets
        if (['inner' in zn for zn in
                           sourcezone.dataset.zone_names].count(True) >1):
            warnings.warn("multiple 'inner' zones found, "+
                                   "default to first listed!",UserWarning)
        zonename = 'ms_'+mode
        state_index = calc_Jpar_state(mode, sourcezone)
    else:
        assert False, ('mode not recognized!! Check "approved" list with'+
                       'available calc_state functions')
    if 'iso_betastar' not in mode:
        zone, innerzone = setup_isosurface(1, state_index, zonename)
        innerzone = None
    else:
        zone, innerzone = setup_isosurface(1, state_index, zonename,
                                           keep_condition='sphere',
                                  keep_cond_value=kwargs.get('inner_r',3))
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

def calc_Jpar_state(mode, sourcezone, **kwargs):
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
    #Initialize state variable
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    eq('{'+mode+'}=0', zones=[sourcezone])
    tol = kwargs.get('projection_tol', 2)
    state = sourcezone.values(mode).as_numpy_array()
    #pull footpoint values from entire domain into arrays
    #global_th1 = sourcezone.values('theta_1 *').as_numpy_array()
    #global_th2 = sourcezone.values('theta_2 *').as_numpy_array()
    #global_phi1 = sourcezone.values('phi_1 *').as_numpy_array()
    #global_phi2 = sourcezone.values('phi_2 *').as_numpy_array()

    #define zone that will be used to find current densities
    zone = sourcezone.dataset.zone('*inner*')
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

        from IPython import embed; embed()
        state = any([np.sqrt((global_th[i]-t[0])**2 +
                                    (global_phi[i]-t[1])**2) < tol
                                                   for t in targets]).real
        for i in valid_indices:
            state[i] = any([np.sqrt((global_th[i]-t[0])**2 +
                                    (global_phi[i]-t[1])**2) < tol
                                                   for t in targets]).real
        from IPython import embed; embed()
        state[proj_indices] = 1
        sourcezone.values(mode)[::] = state

        '''
        th_str = th.split('*')[0]+'[deg]'
        phi_str= phi.split('*')[0]+'[deg]'
        from IPython import embed; embed()
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
                                                        zones=[sourcezone])
        from IPython import embed; embed()
    return zone.dataset.variable(mode).index

def calc_bs_state(sonicspeed, betastarblank, xtail, mpexists=False):
    """Function creates equation for the bow shock region
    Inputs
        sonicspeed- gas dynamics speed of sound: sqrt(gamma*P/rho)[km/s]
        betastarblank- blanking condition for betastar
        mpexists(boolean)- used to define the back end condition
    Return
        index- index for the created variable
    """
    eq = tp.data.operate.execute_equation
    if not mpexists:
        eq('{ext_bs_Cs}=if({beta_star}>'+str(betastarblank)+',{Cs [km/s]}'
                                               +','+str(2*sonicspeed)+')')
    else:
        #UPDATE
        eq('{ext_bs_Cs}=if({X [R]}=='+str(xtail)+','+str(sonicspeed)+','+
                       'if({beta_star}>'+str(betastarblank)+'&&'+
                          '{X [R]}>'+str(xtail)+',{Cs [km/s]},0))')
    return tp.active_frame().dataset.variable('ext_bs_Cs').index

def calc_ps_qDp_state(ps_qDp,closed_var,lshelllim,bxmax,*,Lvar='Lshell'):
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
    eq = tp.data.operate.execute_equation
    if ps_qDp == 'ps':
        eq('{ms_ps_L>'+lshelllim+'} = if({'+closed_var+'}==1&&'+
                                        '{'+Lvar+'}>'+lshelllim+'&&'+
                                        '{r [R]}>3&&'+
                                        'abs({B_x [nT]})<'+bxmax+'&&'+
                                        '{X [R]}<0,1,0)')
        return tp.active_frame().dataset.variable(
                                               'ms_ps_L>'+lshelllim).index
    elif ps_qDp == 'qDp':
        eq('{ms_qDp_L>'+lshelllim+'} = if({'+closed_var+'}>0&&'+
                                         '{'+Lvar+'}>'+lshelllim+'&&'+
                                         '{r [R]}>3&&'+
                                         '(abs({B_x [nT]})>'+bxmax+'||'+
                                         '{X [R]}>0),1,0)')
        return tp.active_frame().dataset.variable(
                                              'ms_qDp_L>'+lshelllim).index
    elif ps_qDp == 'closed':
        eq('{ms_closed_L>'+lshelllim+'} = if({'+closed_var+'}>0&&'+
                                         '{'+Lvar+'}>'+lshelllim+'&&'+
                                         '{r [R]}>3,1,0)')
        return tp.active_frame().dataset.variable(
                                           'ms_closed_L>'+lshelllim).index


def calc_rc_state(closed_var, lshellmax, *, Lvar='Lshell', **kwargs):
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
    eq = tp.data.operate.execute_equation
    eq('{ms_rc_L='+lshellmax+'} = if({'+closed_var+'}==1&&'+
                                   '{r [R]}>'+kwargs.get('rmin','3')+'&&'+
                                    '{'+Lvar+'}<'+lshellmax+',1,0)')
    return tp.active_frame().dataset.variable('ms_rc_L='+lshellmax).index

def calc_lobe_state(mp_var, northsouth, *, status='Status'):
    """Function creates equation for north or south lobe within the
        confines of magnetopause surface indicated by mp_var
    Inputs
        mp_var- variable name for magnetopause zone
        northsouth- which lobe to create equation for
        status- optional change to the status variable name
    Return
        index- index for the created variable
    """
    eq = tp.data.operate.execute_equation
    if northsouth == 'north':
        eq('{NLobe} = if({'+mp_var+'}==1&&{'+status+'}==2,1,0)')
        return tp.active_frame().dataset.variable('NLobe').index
    else:
        eq('{SLobe} = if({'+mp_var+'}==1&&{'+status+'}==1,1,0)')
        return tp.active_frame().dataset.variable('SLobe').index

def calc_transition_rho_state(xmax, xmin, hmax, rhomax, rhomin, uBmin):
    """Function creates equation in tecplot representing surface
    Inputs
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
    Outputs
        created variable index
    """
    drho = rhomax-rhomin
    eq = tp.data.operate.execute_equation
    eq('{mp_rho_transition} = '+
        'IF({X [R]} >'+str(xmin)+' &&'+
        '{X [R]} <'+str(xmax)+' && {h} < '+str(hmax)+','+
           'IF({Rho [amu/cm^3]}<(atan({X [R]}+5)+pi/2)/pi*'+str(drho)+'+'+
            str(rhomin)+'||({uB [J/Re^3]}>'+str(uBmin)+'), 1, 0), 0)')
    return tp.active_frame().dataset.variable('mp_rho_transition').index

def calc_betastar_state(zonename, source, xmax, xmin, hmax, betamax,
                        coreradius, closed_zone):
    """Function creates equation in tecplot representing surface
    Inputs
        zonename
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
        closed_zone- zone object, None to omit closed zone in equation
    Outputs
        created variable index
    """
    #FOR LATER
    '''
    {d_Betastar0.7} = IF({beta_star}<2.8&&{beta_star}>0.7, {beta_star}, IF({beta_star}<0.7, 0.7, 2.8))
{dn_x} = ddx({d_Betastar0.7})
{dn_y} = ddy({d_Betastar0.7})
    '''
    eq = tp.data.operate.execute_equation
    eqstr = ('{'+zonename+'} = '+
        'IF({X [R]} >'+str(xmin-2)+'&&'+
        '{X [R]} <'+str(xmax)+'&& {r [R]} > '+str(coreradius))
    eqstr=(eqstr+',IF({beta_star}['+str(source+1)+']<'+str(betamax)+',1,')
    if type(closed_zone) != type(None):
        eqstr =(eqstr+'IF({'+closed_zone.name+'} == 1,1,0))')
    else:
        eqstr =(eqstr+'0)')
    eqstr =(eqstr+',0)')
    #eq(eqstr, zones=[0], value_location=ValueLocation.CellCentered)
    eq(eqstr, zones=[0])
    return tp.active_frame().dataset.variable(zonename).index

def calc_delta_state(t0state, t1state):
    """Function creates equation representing volume between now and
        future magnetosphere
    Inputs
        t0state, t1state- names for use in equation for current and
                          future times
    """
    eq = tp.data.operate.execute_equation
    eq('{delta_volume} = IF({'+t1state+'}==1 && {'+t0state+'}==0, 1,'+
                        'IF({'+t1state+'}==0 && {'+t0state+'}==1,-1, 0))',
                        value_location=ValueLocation.CellCentered)

def calc_iso_rho_state(xmax, xmin, hmax, rhomax, rmin_north, rmin_south):
    """Function creates equation in tecplot representing surface
    Inputs
        xmax, xmin, hmax, hmin, rmin- spatial bounds
        rhomax- density bound
    Outputs
        created variable index
    """
    eq = tp.data.operate.execute_equation
    eq('{mp_rho_innerradius} = '+
        'IF({X [R]} >'+str(xmin-2)+'&&'+
        '{X [R]} <'+str(xmax)+'&& {h} < '+str(hmax)+','+
            'IF({Rho [amu/cm^3]}<'+str(rhomax)+', 1,'+
                'IF(({r [R]} <'+str(rmin_north)+'&&{Z [R]}>0)||'+
                   '({r [R]} <'+str(rmin_south)+'&&{Z [R]}<0),1,0)),0)')
    return tp.active_frame().dataset.variable('mp_rho_innerradius').index

def calc_shue_state(field_data, mode, x_subsolar, xtail, *, dx=10):
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
    #Probe field data at x_subsolar + dx to find Bz and Dp
    Bz = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][9]
    Dp_index = field_data.variable('Dp *').index
    Dp = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][Dp_index]
    #Get r0 and alpha based on IMF conditions
    if mode == 'shue97':
        r0, alpha = r0_alpha_1997(Bz, Dp)
    else:
        r0, alpha = r0_alpha_1998(Bz, Dp)
    eq = tp.data.operate.execute_equation
    eq('{theta} = IF({X [R]}>0,atan({h}/{X [R]}), pi-atan({h}/{X [R]}))')
    eq('{r'+mode+'} = '+str(r0)+'*(2/(1+cos({theta})))**'+str(alpha))
    eq('{'+mode+'} = IF(({r [R]} < {r'+mode+'}) &&'+
                      '({X [R]} > '+str(xtail)+'), 1, 0)')
    return field_data.variable(mode).index

def calc_sphere_state(mode, xc, yc, zc, rmax, *, rmin=0):
    """Function creates state variable for a simple box
    Inputs
        mode
        xc, yc, zc- locations for sphere center
        rmax- sphere radius
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq = tp.data.operate.execute_equation
    eq('{'+mode+'} = IF(sqrt(({X [R]} -'+str(xc)+')**2 +'+
                            '({Y [R]} -'+str(yc)+')**2 +'+
                            '({Z [R]} -'+str(zc)+')**2) <'+
                             str(rmax)+'&&'+
                        'sqrt(({X [R]} -'+str(xc)+')**2 +'+
                            '({Y [R]} -'+str(yc)+')**2 +'+
                            '({Z [R]} -'+str(zc)+')**2) >'+
                            str(rmin)+',1, 0)')
    return tp.active_frame().dataset.variable(mode).index

def calc_closed_state(statename, status_key, status_val, xmin, source):
    """Function creates state variable for the closed fieldline region
    Inputs
        status_key/val-string key and value used to denote closed fldlin
        xmin- minimum cuttoff value
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq = tp.data.operate.execute_equation
    eq('{'+statename+'} = IF({X [R]} > '+str(xmin)+','+
                    'IF({'+status_key+'}['+str(source+1)+']=='+
                                            str(status_val)+',1,0), 0)',
                                                              zones=[0])
    return tp.active_frame().dataset.variable(statename).index

def calc_box_state(mode, xmax, xmin, ymax, ymin, zmax, zmin):
    """Function creates state variable for a simple box
    Inputs
        mode
        xmax,xmin...zmin- locations for box vertices
    Outputs
        state_var_index- index to find state variable in tecplot
    """
    eq = tp.data.operate.execute_equation
    eq('{'+mode+'} = IF(({X [R]} >'+str(xmin)+') &&'+
                       '({X [R]} <'+str(xmax)+') &&'+
                       '({Y [R]} >'+str(ymin)+') &&'+
                       '({Y [R]} <'+str(ymax)+') &&'+
                       '({Z [R]} >'+str(zmin)+') &&'+
                       '({Z [R]} <'+str(zmax)+'), 1, 0)')
    return tp.active_frame().dataset.variable(mode).index

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

