#!/usr/bin/env python3
"""Functions for identifying surfaces from field data
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import spacepy
from spacepy import coordinates as coord
from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd

def create_stream_zone(field_data, r_start, lat_start, lon_start,
                       zone_name, stream_type, *, cart_given=False):
    """Function to create a streamline, created in 2 directions from
       starting point
    Inputs
        field_data- Dataset class from tecplot with 3D field data
        r_start [R]- starting position for streamline
        lat_start [deg]
        lon_start [deg]
        zone_name
        stream_type- day, north or south for determining stream direction
        cart_given- optional input for giving cartesian coordinates
    """
    if cart_given==False:
        # Get starting position in cartesian coordinates
        [x_start, y_start, z_start] = sph_to_cart(r_start, lat_start,
                                                  lon_start)
    else:
        x_start = r_start
        y_start = lat_start
        z_start = lon_start
    # Create streamline
    tp.active_frame().plot().show_streamtraces = True
    field_line = tp.active_frame().plot().streamtraces
    if stream_type == 'south':
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Reverse)
    elif stream_type == 'north':
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Forward)
    else:
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Both)
    # Create zone
    field_line.extract()
    field_data.zone(-1).name = zone_name + '{}'.format(lon_start)
    # Delete streamlines
    field_line.delete_all()


def check_streamline_closed(field_data, zone_name, r_seed, stream_type):
    """Function to check if a streamline is open or closed
    Inputs
        field_data- tecplot Dataset class with 3D field data
        zone_name
        r_seed [R]- position used to seed field line
        stream_type- dayside, north or south from tail
    Outputs
        isclosed- boolean, True for closed
    """
    # Get starting and endpoints of streamzone
    r_values = field_data.zone(zone_name+'*').values('r *').as_numpy_array()
    if stream_type == 'north':
        r_end_n = r_values[-1]
        r_end_s = 0
        r_seed = 2
    elif stream_type == 'south':
        r_end_n = 0
        r_end_s = r_values[0]
        r_seed = 2
    elif stream_type == 'inner_mag':
        xmax, xmin = field_data.zone(zone_name+'*').values('X *').minmax()
        ymax, ymin = field_data.zone(zone_name+'*').values('Y *').minmax()
        zmax, zmin = field_data.zone(zone_name+'*').values('Z *').minmax()
        extrema = [[xmax, ymax, zmax],
                   [xmin, ymin, zmin]]
        bounds = [[-220,-126,-126],
                  [31.5, 126, 126]]
        isclosed = True
        #from IPython import embed; embed()
        for minmax in enumerate(extrema):
            for value in enumerate(minmax[1]):
                if abs(value[1])>0.9*abs(bounds[minmax[0]][value[0]]):
                    isclosed = False
                    return isclosed
        return isclosed
        '''
    elif stream_type == 'inner_mag':
        x_values = field_data.zone(zone_name+'*').values(
                                                    'X *').as_numpy_array()
        x_values = field_data.zone(zone_name+'*').values(
                                                    'X *').as_numpy_array()
        y_values = field_data.zone(zone_name+'*').values(
                                                    'Y *').as_numpy_array()
        z_values = field_data.zone(zone_name+'*').values(
                                                    'Z *').as_numpy_array()
        pos_bound = [40,128,128]
        neg_bound = [-256,-128,-128]
        x_max = abs(x_values.min())
        r_end_n, r_end_s = r_values[0], r_values[-1]
        '''
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

def mag2gsm(radius, latitude, longitude, time):
    """Function converts magnetic spherical coordinates to cartesian
        coordinates in GSM
    Inputs
        radius
        latitude- relative to magnetic dipole axis
        longitude- relative to magnetic dipole axis
        time- spacepy Ticktock object with time information
    Output
        xgsm, ygsm, zgsm- in form of 3 element np array
    """
    coordinates = coord.Coords([radius, latitude, longitude], 'SM', 'sph')
    coordinates.ticks = time
    return coordinates.convert('GSM', 'car').data[0][0:3]


def find_tail_disc_point(rho, psi_disc, x_pos):
    """Function find spherical coord of a point on a disc at a constant x
       position in the tail
    Inputs
        rho- radial position relative to the center of the disc
        psi_disc- angle relative to the axis pointing out from the center
        of the disc
        x_pos- x position of the disc
    Outputs
        [x_pos, y_pos, z_pos]- cart coord of the point relative
        to the global origin
    """
    y_pos = rho*sin(deg2rad(psi_disc))
    z_pos = rho*cos(deg2rad(psi_disc))
    return [x_pos, y_pos, z_pos]


def calc_dayside_mp(field_data, lon, r_max, r_min, itr_max, tolerance):
    """"Function to create zones that will makeup dayside magnetopause
    Inputs
        field_data- Dataset class from tecplot with 3D field data
        lon- set of longitudinal points in degrees
        r_max- maxium radial distance for equitorial search
        r_min
        itr_max
        tolerance- for searching algorithm
    """
    #Initialize objects that will be modified in creation loop
    r_eq_mid = np.zeros(int(len(lon)))
    itr = 0
    r_eq_max, r_eq_min = r_max, r_min

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('B_x*')
    plot.vector.v_variable = field_data.variable('B_y*')
    plot.vector.w_variable = field_data.variable('B_z*')


    #Create Dayside Magnetopause field lines
    stream_type = 'day'
    for i in range(int(len(lon))):
        #Create initial max min and mid field lines
        create_stream_zone(field_data, r_min, 0, lon[i],
                           'min_day_line', stream_type)
        create_stream_zone(field_data, r_max, 0, lon[i],
                           'max_day_line', stream_type)
        #Check that last closed is bounded
        min_closed = check_streamline_closed(field_data, 'min_day_line',
                                                  r_min, stream_type)
        max_closed = check_streamline_closed(field_data, 'max_day_line',
                                                  r_max, stream_type)
        field_data.delete_zones(field_data.zone('min_day*'),
                               field_data.zone('max_day*'))
        print('Day', i,'lon: {:.1f}, iters: {}, err: {}'.format(lon[i],
                                                  itr, r_eq_max-r_eq_min))
        if max_closed and min_closed:
            print('WARNING: field line closed at max {}R_e'.format(r_max))
            create_stream_zone(field_data, r_max, 0, lon[i],
                               'day_lon_', stream_type)
        elif not max_closed and not min_closed:
            print('WARNING: first field line open at {}R_e'.format(r_min))
            create_stream_zone(field_data, r_min, 0, lon[i],
                               'day_lon_', stream_type)
        else:
            r_eq_mid[i] = (r_max+r_min)/2
            itr = 0
            notfound = True
            r_eq_min, r_eq_max = r_min, r_max
            while(notfound and itr < itr_max):
                #This is a bisection root finding algorithm
                create_stream_zone(field_data, r_eq_mid[i], 0, lon[i],
                                   'temp_day_lon_', stream_type)
                mid_closed = check_streamline_closed(field_data,
                                                     'temp_day_lon_',
                                                     r_eq_mid[i],
                                                     stream_type)
                if mid_closed:
                    r_eq_min = r_eq_mid[i]
                else:
                    r_eq_max = r_eq_mid[i]
                if abs(r_eq_min - r_eq_max) < tolerance and mid_closed:
                    notfound = False
                    field_data.zone('temp*').name='day_lon_{:.1f}'.format(
                                                                   lon[i])
                else:
                    r_eq_mid[i] = (r_eq_max+r_eq_min)/2
                    field_data.delete_zones(field_data.zone('temp_day*'))
                itr += 1




def calc_tail_mp(field_data, psi, x_disc, rho_max, rho_step):
    """Function to create the zones that will become the tail magnetopause
    Inputs
        psi- set of disc azimuthal angles[degrees]
        x_disc- x position of the tail disc
        rho_max- outer radial bounds of the tail disc
        rho_step- radial distance increment for marching algorithm
    """
    #Initialize objects that will be modified in creation loop
    rho_tail = rho_max

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('B_x*')
    plot.vector.v_variable = field_data.variable('B_y*')
    plot.vector.w_variable = field_data.variable('B_z*')

    #Create Tail Magnetopause field lines
    for i in range(int(len(psi))):
        #Find global position based on seed point
        x_pos, y_pos, z_pos = find_tail_disc_point(rho_max,psi[i],x_disc)
        r_tail = sqrt(x_pos**2+y_pos**2+z_pos**2)
        #check if north or south attached
        if tp.data.query.probe_at_position(x_pos, y_pos, z_pos)[0][7] < 0:
            stream_type = 'south'
        else:
            stream_type = 'north'
        create_stream_zone(field_data, x_pos, y_pos, z_pos,
                          'temp_tail_line_', stream_type, cart_given=True)
        #check if closed
        tail_closed = check_streamline_closed(field_data,
                                              'temp_tail_line_', r_tail,
                                              stream_type)
        if tail_closed:
            print('WARNING: field line closed at RHO_MAX={}R_e'.format(
                                                                 rho_max))
            field_data.zone('temp_tail*').name='tail_field_{:.1f}'.format(
                                                                   psi[i])
        else:
            #This is a basic marching algorithm from outside in starting at
            #RHO_MAX
            rho_tail = rho_max
            notfound = True
            while notfound and rho_tail > rho_step:
                field_data.delete_zones(field_data.zone('temp_tail_line*'))
                rho_tail = rho_tail - rho_step
                x_pos, y_pos, z_pos = find_tail_disc_point(rho_tail,
                                                           psi[i],x_disc)
                r_tail = sqrt(x_pos**2+y_pos**2+z_pos**2)
                #check if north or south attached
                if tp.data.query.probe_at_position(x_pos,
                                                   y_pos, z_pos)[0][7] < 0:
                    stream_type = 'south'
                else:
                    stream_type = 'north'
                create_stream_zone(field_data, x_pos, y_pos, z_pos,
                                   'temp_tail_line_', stream_type,
                                   cart_given=True)
                tail_closed =check_streamline_closed(field_data,
                                                     'temp_tail_line_',
                                                    rho_tail, stream_type)
                if tail_closed:
                    field_data.zone('temp*').name='tail_field_{:.1f}'.format(
                                                        psi[i])
                    notfound = False
                    print('Tail', i,' rho{:.1f} psi{:.1f}'.format(rho_tail,
                                                           psi[i]))
                if rho_tail <= rho_step:
                    print('WARNING: not possible at psi={:.1f}'.format(
                                                                psi[i]))


def calc_plasmasheet(field_data, lat_max, lon_list, tail_cap,
                     max_iter, tolerance, time):
    """Function to create tecplot zones that define the plasmasheet boundary
    Inputs
        field_data- tecplot Dataset containing 3D field data
        lat_max- max latitude value corresponding to near pole
        lon_list- list containing longitudinal positions to be searched
        tail_cap
        max_iter- max iterations for bisection algorithm
        tolerance- tolerance on the bisection method
        time- spacepy Ticktock of current time
    """
    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('B_x*')
    plot.vector.v_variable = field_data.variable('B_y*')
    plot.vector.w_variable = field_data.variable('B_z*')
    seed_radius = 1

    #iterate through northside zones
    for lon in lon_list:
        print('longitude {:.1f}'.format(lon))
        sphcoor = coord.Coords([seed_radius, 85, lon], 'SM', 'sph')
        sphcoor.ticks = time
        #initialize latitude search bounds
        equat_lat = 45
        pole_lat = lat_max
        mid_lat = (pole_lat+equat_lat)/2

        """
        #Testing for XZ visulalizations
        map_index = 0
        for i in np.append(np.linspace(pi/18,lat_max, 20),
                           np.linspace(pi+pi/18, pi-lat_max, 20)):
            create_stream_zone(field_data, seed_radius, i,
                               phi, 'temp_'+str(rad2deg(i)), 'inner_mag')
            map_index +=1
            poleward_closed = check_streamline_closed(field_data,
                                                'temp_'+str(rad2deg(i)),
                                                abs(tail_cap),
                                                'inner_mag')
            if map_index > 18:
                plot.fieldmap(map_index).mesh.color=Color.Blue
            if not poleward_closed:
                plot.fieldmap(map_index).mesh.color=Color.Red
        """
        #Create bounding fieldlines
        [xgsm, ygsm, zgsm] = mag2gsm(seed_radius, pole_lat, lon, time)
        create_stream_zone(field_data, xgsm, ygsm, zgsm,
                           'temp_poleward_', 'inner_mag', cart_given=True)
        poleward_closed = check_streamline_closed(field_data,
                                                  'temp_poleward',
                                                  abs(tail_cap),
                                                  'inner_mag')
        [xgsm, ygsm, zgsm] = mag2gsm(seed_radius, equat_lat, lon, time)
        create_stream_zone(field_data, xgsm, ygsm, zgsm,
                        'temp_equatorward_', 'inner_mag', cart_given=True)
        equatorward_closed = check_streamline_closed(field_data,
                                                 'temp_equatorward',
                                                 abs(tail_cap),
                                                 'inner_mag')
        #check if both are open are closed to start
        if poleward_closed and equatorward_closed:
            print('\nWarning: high and low lat {:.2f}, {:.2f} '.format(
                                                      pole_lat, equat_lat)+
                  'closed at longitude {:.1f}\n'.format(lon))
            [xgsm, ygsm, zgsm] = mag2gsm(seed_radius, mid_lat, lon, time)
            create_stream_zone(field_data, xgsm, ygsm, zgsm,
                               'plasma_sheet_', 'inner_mag',
                               cart_given=True)
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
        elif not poleward_closed and (not equatorward_closed):
            print('Warning: high and low lat {:.2f}, {:.2f} open'.format(
                                                    pole_lat, equat_lat))
            [xgsm, ygsm, zgsm] = mag2gsm(seed_radius, mid_lat, lon, time)
            create_stream_zone(field_data, xgsm, ygsm, zgsm,
                               'plasma_sheet_', 'inner_mag',
                               cart_given=True)
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
        else:
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
            notfound = True
            itr = 0
            while (notfound and itr < max_iter):
                mid_lat = (pole_lat+equat_lat)/2
                [xgsm, ygsm, zgsm] = mag2gsm(seed_radius,mid_lat,lon,time)
                create_stream_zone(field_data, xgsm, ygsm, zgsm,
                                   'temp_ps_line_', 'inner_mag',
                                   cart_given=True)
                mid_lat_closed = check_streamline_closed(field_data,
                                                         'temp_ps_line_',
                                                         abs(tail_cap),
                                                         'inner_mag')
                if mid_lat_closed:
                    equat_lat = mid_lat
                else:
                    pole_lat = mid_lat
                if abs(pole_lat - equat_lat)<tolerance and (
                                                        mid_lat_closed):
                    notfound = False
                    field_data.zone('temp*').name = 'plasma_sheet_'.format(
                                                                 equat_lat)
                else:
                    field_data.delete_zones(field_data.zone('temp*'))
                itr += 1


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
    print('converting '+filename.split('.')[0]+' to DataFrame\n')
    os.system('touch '+filename)
    #Export 3D point data to csv file
    tp.macro.execute_extended_command(command_processor_id='excsv',
            command='VarNames:'+
                    'FrOp=1:'+
                    'ZnCount={:d}:'.format(len(zonelist))+
                    'ZnList=[{:d}-{:d}]:'.format(int(zonelist[0]),
                                                 int(zonelist[-1]))+
                    'VarCount={:d}:'.format(len(varlist))+
                    'VarList=[{:d}-{:d}]:'.format(int(varlist[0]),
                                                  int(varlist[-1]))+
                    'ValSep=",":'+
                    'FNAME="'+os.getcwd()+'/'+filename+'"')
    loc_data = pd.read_csv(filename)
    if any(col == 'X [R]' for col in loc_data.columns):
        loc_data = loc_data.drop(columns=['Unnamed: 3'])
        loc_data = loc_data.sort_values(by=['X [R]'])
        x_max = loc_data['X [R]'].max()
    else: x_max = []
    return loc_data, x_max

def create_cylinder(field_data, nx, nalpha, nfill, x_min, x_max,
                    zone_name):
    """Function creates empty cylindrical zone for loading of slice data
    Inputs
        field_data- tecplot Dataset class with 3D field data
        nx- number of x positions, same as n_slice
        nalpha- number of aximuthal points
        nfill- number of radial points
        x_min
        x_max
        zone_name
    """
    #use built in create zone function for verticle cylinder
    tp.macro.execute_command('''$!CreateCircularZone
                             IMax = {:d}
                             JMax = {:d}
                             KMax = {:d}
                             X = 0
                             Y = 0
                             Z1 = {:f}
                             Z2 = {:f}
                             Radius = 50'''.format(nfill, nalpha, nx,
                                                   x_min, x_max))

    #use built in function to rotate 90deg about y axis
    tp.macro.execute_command('''$!AxialDuplicate
                             ZoneList =  [{:d}]
                             Angle = 90
                             NumDuplicates = 1
                             XVar = 1
                             YVar = 2
                             ZVar = 3
                             UVarList =  [8]
                             VVarList =  [9]
                             WVarList =  [10]
                             NormalX = 0
                             NormalY = 1
                             NormalZ = 0'''.format(field_data.num_zones))

    #delete verticle cylinder
    field_data.delete_zones(field_data.zone('Circular zone'))
    field_data.zone('Circular*').name = zone_name
    print('empty zone created')

def load_cylinder(field_data, data, zonename, I, J, K):
    """Function to load processed slice data into cylindrial ordered zone
       I, J, K -> radial, azimuthal, axial
    Inputs
        field_data- tecplot Dataset class with 3D field data
        filename- path to .csv file for loading data
        zonename- name of cylindrical zone
        I- vector of I coordinates (0 to 1)
        J- vector of J coordinates (0 to num_alpha)
        K- vector of K coordinates (0 to num_slices)
    """
    print('cylindrical zone loading')
    mag_bound = field_data.zone(zonename)
    #copy data points
    xdata = data['X [R]'].values.copy()
    ydata = data['Y [R]'].values.copy()
    zdata = data['Z [R]'].values.copy()
    #initialize values
    mag_bound.values('X*')[0::I] = xdata
    mag_bound.values('Y*')[0::I] = np.zeros(J*K)
    mag_bound.values('Z*')[0::I] = np.zeros(J*K)
    ymean = np.zeros(len(ydata))
    zmean = np.zeros(len(zdata))
    #determine center point
    for k in range(0,K):
        ymean[k*J:(k+1)*J] = np.mean(ydata[k*J:(k+1)*J])
        zmean[k*J:(k+1)*J] = np.mean(zdata[k*J:(k+1)*J])
    for i in range(0,I):
        mag_bound.values('X*')[i::I] = xdata
        mag_bound.values('Y*')[i::I] = (ydata-ymean) * i/I + ymean
        mag_bound.values('Z*')[i::I] = (zdata-zmean) * i/I + zmean
    print('\nvalues loaded, check out how it looks\n')


def calculate_energetics(field_data, zone_name):
    """Function calculates values for energetics tracing
    Inputs
        field_data- tecplot Dataset class containing 3D field data
    """
    zone_index= field_data.zone(zone_name).index
    mu0 = 4*pi*1e-7
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'GRIDKUNITNORMAL VALUELOCATION = '+
                                      'CELLCENTERED')
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'GRIDIUNITNORMAL VALUELOCATION = '+
                                      'CELLCENTERED')
    eq = tp.data.operate.execute_equation

    #Surface normal vector components
    eq('{surface_normal_x} = IF(K==40||K==1,'+
                            '-1*{X Grid K Unit Normal},'+
                            '{X Grid I Unit Normal})')
    eq('{surface_normal_y} = IF(K==40||K==1,'+
                            '-1*{Y Grid K Unit Normal},'+
                            '{Y Grid I Unit Normal})')
    eq('{surface_normal_z} = IF(K==40||K==1,'+
                            '-1*{Z Grid K Unit Normal},'+
                            '{Z Grid I Unit Normal})')

    #Magnetic Energy per volume
    eq('{uB [J/km^3]} = ({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                        '/(2*4*3.14159*1e-7) * 1e-9')

    #Electric Field
    eq('{E_x [mV/km]} = ({U_z [km/s]}*{B_y [nT]}'+
                          '-{U_y [km/s]}*{B_z [nT]})')
    eq('{E_y [mV/km]} = ({U_x [km/s]}*{B_z [nT]}'+
                         '-{U_z [km/s]}*{B_x [nT]})')
    eq('{E_z [mV/km]} = ({U_y [km/s]}*{B_x [nT]}'+
                         '-{U_x [km/s]}*{B_y [nT]})')

    #Poynting Flux
    eq('{ExB_x [kW/km^2]} = -(1/1.25663706)*({E_z [mV/km]}*{B_y [nT]}'+
                                            '-{E_y [mV/km]}*{B_z [nT]})'+
                                            '*1e-6')
    eq('{ExB_y [kW/km^2]} = -(1/1.25663706)*({E_x [mV/km]}*{B_z [nT]}'+
                                            '-{E_z [mV/km]}*{B_x [nT]})'+
                                            '*1e-6')
    eq('{ExB_z [kW/km^2]} = -(1/1.25663706)*({E_y [mV/km]}*{B_x [nT]}'+
                                            '-{E_x [mV/km]}*{B_y [nT]})'+
                                            '*1e-6')
    #Total Energy Flux
    eq('{K_x [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/0.666667)'+
                               '+1e-3*{Rho [amu/cm^3]}/2*'+
                                   '({U_x [km/s]}**2+{U_y [km/s]}**2'+
                                   '+{U_z [km/s]}**2))'+
                          '*{U_x [km/s]}  +  {ExB_x [kW/km^2]}',
        zones=[zone_index])
    eq('{K_y [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/0.666667)'+
                               '+1e-3*{Rho [amu/cm^3]}/2*'+
                                   '({U_x [km/s]}**2+{U_y [km/s]}**2'+
                                   '+{U_z [km/s]}**2))'+
                          '*{U_y [km/s]}  +  {ExB_y [kW/km^2]}',
        zones=[zone_index])
    eq('{K_z [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/0.666667)'+
                               '+1e-3*{Rho [amu/cm^3]}/2*'+
                                   '({U_x [km/s]}**2+{U_y [km/s]}**2'+
                                   '+{U_z [km/s]}**2))'+
                          '*{U_z [km/s]}  +  {ExB_z [kW/km^2]}',
        zones=[zone_index])

    #Component Normal Flux
    eq('{Kn_x [kW/km^2]} = ({K_x [kW/km^2]}*{surface_normal_x}'+
                            '+{K_y [kW/km^2]}*{surface_normal_y}'+
                            '+{K_z [kW/km^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_x}',
       zones=[zone_index])
    eq('{Kn_y [kW/km^2]} = ({K_x [kW/km^2]}*{surface_normal_x}'+
                            '+{K_y [kW/km^2]}*{surface_normal_y}'+
                            '+{K_z [kW/km^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_y}',
        zones=[zone_index])
    eq('{Kn_z [kW/km^2]} = ({K_x [kW/km^2]}*{surface_normal_x}'+
                            '+{K_y [kW/km^2]}*{surface_normal_y}'+
                            '+{K_z [kW/km^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_z}',
        zones=[zone_index])

    #Magnitude Normal Flux
    eq('{K_out [kW/km^2]} = ({Kn_x [kW/km^2]}*{surface_normal_x}'+
                            '+{Kn_y [kW/km^2]}*{surface_normal_y}'+
                            '+{Kn_z [kW/km^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2 '+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)',
        zones=[zone_index])

    #Split into + and - flux
    eq('{K_out+} = max({K_out [kW/km^2]},0)', zones=[zone_index])
    eq('{K_out-} = min({K_out [kW/km^2]},0)', zones=[zone_index])


def display_variable_bar(oldframe, var_index, color, barid, newaxis):
    """Function to display bargraph of variable quantity in upper right
    Inputs
        frame- tecplot frame object
        var_index- index for displayed variable (4 for integrated qty)
        color
        barid- for multiple bars, setup to the right of the previous
        newaxis- True if plotting on new axis
    """
    oldframe.activate()
    tp.macro.execute_command('$!CreateNewFrame\n'+
            'XYPos{X='+str(1.25+0.25*barid)+'\n'
                  'Y=0}\n'+
            'Width = 1\n'+
            'Height = 3.7')
    frame = tp.active_frame()
    frame.show_border = False
    plt = frame.plot(PlotType.XYLine)
    frame.plot_type = PlotType.XYLine
    plt.linemap(0).show = False
    plt.linemap(2).show = True
    frame.transparent = True
    plt.show_bars = True
    plt.linemap(2).bars.show = True
    plt.linemap(2).bars.size = 16
    plt.linemap(2).bars.line_color = color
    plt.linemap(2).bars.fill_color = color
    plt.view.translate(x=-10,y=0)
    plt.axes.x_axis(0).show = False
    plt.axes.y_axis(0).title.title_mode = AxisTitleMode.UseText
    plt.axes.y_axis(0).title.text='Plasmasheet Power [kW]'
    plt.axes.y_axis(0).min = -1400
    plt.axes.y_axis(0).max = 1400
    if newaxis:
        plt.axes.y_axis(0).show = True
        plt.axes.y_axis(0).title.text='Magnetopause Power [kW]'
        plt.axes.y_axis(0).line.offset = -40
        plt.axes.y_axis(0).title.offset = 40
        if barid > 2:
            plt.axes.y_axis(0).min = -140
            plt.axes.y_axis(0).max = 140
            plt.axes.y_axis(0).line.alignment = AxisAlignment.WithGridMax
            plt.axes.y_axis(0).title.text='Plasmasheet Power [kW]'
            plt.axes.y_axis(0).line.offset = -20
            plt.axes.y_axis(0).title.offset = 20
            plt.axes.y_axis(0).tick_labels.offset = 5
            plt.view.translate(x=-25,y=0)
    else:
        plt.axes.y_axis(0).show = False
    oldframe.move_to_bottom()
    frame.activate()

def integrate_surface(var_index, zone_index, qtname, idimension,
                      kdimension, *, is_cylinder=True, frame_id='main'):
    """Function to calculate integral of variable on a 3D exterior surface
    Inputs
        var_index- variable to be integrated
        zone_index- index of the zone to perform integration
        qtname- integrated quantity will be saved as this name
        idimension- used to determine which planes are outer surface
        kdimension- used to determine which planes are outer surface
        is_cylinder- default true
        frame_id- frame name with the surface that integral is performed
    Output
        integrated_total
    """
    #Integrate total surface Flux
    frame=[fr for fr in tp.frames(frame_id)][0]
    frame.activate()
    page = tp.active_page()
    #validate name (special characters give tp a lot of problems
    qtname_abr = qtname.split('?')[0].split('[')[0].split('*')[0]+'*'
    if not is_cylinder:
        print("Warning: not cylindrical object, check surface integrals")
    surface_planes=["IRange={MIN =1 MAX = 0 SKIP ="+str(idimension-1)+"} "+
                    "JRange={MIN =1 MAX = 0 SKIP =1} "+
                    "KRange={MIN =1 MAX = 0 SKIP ="+str(kdimension-1)+"}"]

    integrate_command_I=("Integrate [{:d}] ".format(zone_index+1)+
                         "VariableOption='Scalar' "
                         "ScalarVar={:d} ".format(var_index+1)+
                         "XVariable=1 "+
                         "YVariable=2 "+
                         "ZVariable=3 "+
                         "ExcludeBlanked='T' "+
                         "IntegrateOver='IPlanes' "+
                         "IntegrateBy='Zones' "+
                         surface_planes[0]+
                         " PlotResults='T' "+
                         "PlotAs='"+qtname+"_I' "+
                         "TimeMin=0 TimeMax=0")
    integrate_command_K=("Integrate [{:d}] ".format(zone_index+1)+
                         "VariableOption='Scalar' "
                         "ScalarVar={:d} ".format(var_index+1)+
                         "XVariable=1 "+
                         "YVariable=2 "+
                         "ZVariable=3 "+
                         "ExcludeBlanked='T' "+
                         "IntegrateOver='KPlanes' "+
                         "IntegrateBy='Zones' "+
                         surface_planes[0]+
                         " PlotResults='T' "+
                         "PlotAs='"+qtname+"_K' "+
                         "TimeMin=0 TimeMax=0")


    #integrate over I planes
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command_I)
    tempframe = [fr for fr in tp.frames('Frame*')][-1]
    result = tempframe.dataset
    Ivalues = result.variable(qtname_abr).values('*').as_numpy_array()
    print('after I integration')
    for fr in tp.frames():
        print(fr.name)

    #delete frame and reinitialize frame structure
    page.delete_frame(tempframe)
    page.add_frame()
    frame.activate()
    print('after I cleanup')
    for fr in tp.frames():
        print(fr.name)

    #integrate over K planes
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command_K)
    tempframe = [fr for fr in tp.frames('Frame*')][-1]
    result = tempframe.dataset
    Kvalues = result.variable(qtname_abr).values('*').as_numpy_array()
    print('after K integration')
    for fr in tp.frames():
        print(fr.name)
    #page.delete_frame(tempframe)
    print('after K cleanup')
    for fr in tp.frames():
        print(fr.name)

    #sum all parts together
    integrated_total = sum(Ivalues)+sum(Kvalues)
    frame.activate()
    frame.move_to_top()
    return integrated_total

def integrate_volume(var_index, zone_index, qtname, *, frame_id='main',
                     tail_only=False):
    """Function to calculate integral of variable within a 3D volume
    Inputs
        var_index- variable to be integrated
        zone_index- index of the zone to perform integration
        qtname- integrated quantity will be saved as this name
        frame_id- frame name with the surface that integral is performed
        tail_only- default False, will integrate only tail if True
    Output
        integral
    """
    #Ensure that objects are initialized for tp operations
    page = tp.active_page()
    frame = [fr for fr in tp.frames(frame_id)][0]
    frame.activate()
    data = frame.dataset
    plt= frame.plot()

    #validate name (special characters give tp a lot of problems)
    qtname_abr = qtname.split('?')[0].split('[')[0].split('*')[0]+'*'

    if tail_only:
        #value blank domain around X>-2
        plt.value_blanking.active = True
        blank = plt.value_blanking.constraint(0)
        blank.active = True
        blank.variable = data.variable('X *')
        blank.comparison_operator = RelOp.GreaterThan
        blank.comparison_value = -2

    #Setup macrofunction command
    integrate_command=("Integrate [{:d}] ".format(zone_index+1)+
                       "VariableOption='Scalar' "
                       "ScalarVar={:d} ".format(var_index+1)+
                       "XVariable=1 "+
                       "YVariable=2 "+
                       "ZVariable=3 "+
                       "ExcludeBlanked='T' "+
                       "IntegrateOver='Cells' "+
                       "IntegrateBy='Zones' "+
                       "PlotResults='T' "+
                       "PlotAs='"+qtname+"' "+
                       "TimeMin=0 TimeMax=0")

    #Perform integration
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command)
    tempframe = [fr for fr in tp.frames('Frame*')][0]
    result = tempframe.dataset
    integral = result.variable(qtname_abr).values('*').as_numpy_array()

    #Delete created frame and turn off blanking
    page.delete_frame(tempframe)
    blank = plt.value_blanking.constraint(0)
    blank.active = False
    plt.value_blanking.active = False

    return integral

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

        #create and load cylidrical zone
        create_cylinder(N_SLICE, N_ALPHA, X_TAIL_CAP, X_MAX)
