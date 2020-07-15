#!/usr/bin/env python3
"""Functions for identifying surfaces from field data
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd

def create_stream_zone(field_data, r_start, theta_start, phi_start,
                       zone_name, stream_type):
    """Function to create a streamline, created in 2 directions from
       starting point
    Inputs
        field_data- Dataset class from tecplot with 3D field data
        r_start [R]- starting position for streamline
        theta_start [rad]
        phi_start [rad]
        zone_name
        stream_type- day, north or south for determining stream direction
    """
    # Get starting position in cartesian coordinates
    [x_start, y_start, z_start] = sph_to_cart(r_start,
                                              theta_start, phi_start)
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
    field_data.zone(-1).name = zone_name + '{}'.format(
                                                    rad2deg(phi_start))
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
        x_max = 0
    elif stream_type == 'south':
        r_end_n = 0
        r_end_s = r_values[0]
        r_seed = 2
        x_max = 0
    elif stream_type == 'inner_mag':
        x_values = field_data.zone(zone_name+'*').values(
                                                    'X *').as_numpy_array()
        x_max = abs(x_values.min())
        r_end_n, r_end_s = r_values[0], r_values[-1]
    else:
        r_end_n, r_end_s = r_values[0], r_values[-1]
        x_max = 0
        '''
        print('r values: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
                                         r_values[0],
                                         r_values[4],
                                         r_values[8],
                                         r_values[-8]))
        '''
    #check if closed
    if (r_end_n > r_seed) or (r_end_s > r_seed) or (x_max > r_seed):
        isclosed = False
    else:
        isclosed = True
    return isclosed

def sph_to_cart(radius, theta, phi):
    """Function converts spherical coordinates to cartesian coordinates
    Inputs
        radius- radial position
        theta
        phi
    Outputs
        [x_pos, y_pos, z_pos]- list of x y z_pos coordinates
    """
    x_pos = (radius * sin(theta) * cos(phi))
    y_pos = (radius * sin(theta) * sin(phi))
    z_pos = (radius * cos(theta))
    return [x_pos, y_pos, z_pos]

def find_tail_disc_point(rho, psi_disc, x_pos):
    """Function find spherical coord of a point on a disc at a constant x
       position in the tail
    Inputs
        rho- radial position relative to the center of the disc
        psi_disc- angle relative to the axis pointing out from the center
        of the disc
        x_pos- x position of the disc
    Outputs
        [radius, theta, phi_disc]- spherical coord of the point relative
        to the global origin
    """
    y_pos = rho*sin(psi_disc)
    z_pos = rho*cos(psi_disc)
    radius = sqrt(x_pos**2+rho**2)
    theta = pi/2 - np.arctan(z_pos/abs(x_pos))
    phi_disc = pi + np.arctan(y_pos/abs(x_pos))
    return [radius, theta, phi_disc]


def calc_dayside_mp(field_data, phi, r_max, r_min, itr_max, tolerance):
    """"Function to create zones that will makeup dayside magnetopause
    Inputs
        field_data- Dataset class from tecplot with 3D field data
        phi- set of phi angle points
        r_max- maxium radial distance for equitorial search
        r_min
        itr_max
        tolerance- for searching algorithm
    """
    #Initialize objects that will be modified in creation loop
    r_eq_mid = np.zeros(int(len(phi)))
    itr = 0
    r_eq_max, r_eq_min = r_max, r_min

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('B_x*')
    plot.vector.v_variable = field_data.variable('B_y*')
    plot.vector.w_variable = field_data.variable('B_z*')


    #Create Dayside Magnetopause field lines
    stream_type = 'day'
    for i in range(int(len(phi))):
        #Create initial max min and mid field lines
        create_stream_zone(field_data, r_min, pi/2, phi[i],
                           'min_field_line', stream_type)
        create_stream_zone(field_data, r_max, pi/2, phi[i],
                           'max_field_line', stream_type)
        #Check that last closed is bounded
        min_closed = check_streamline_closed(field_data, 'min_field_line',
                                                  r_min, stream_type)
        max_closed = check_streamline_closed(field_data, 'max_field_line',
                                                  r_max, stream_type)
        field_data.delete_zones(field_data.zone('min_field*'),
                               field_data.zone('max_field*'))
        print('Day', i,'phi: {:.1f}, iters: {}, err: {}'.format(
                                                  rad2deg(phi[i]),
                                                  itr, r_eq_max-r_eq_min))
        if max_closed and min_closed:
            print('WARNING: field line closed at max {}R_e'.format(r_max))
            create_stream_zone(field_data, r_max, pi/2, phi[i],
                               'field_phi_', stream_type)
        elif not max_closed and not min_closed:
            print('WARNING: first field line open at {}R_e'.format(r_min))
            create_stream_zone(field_data, r_min, pi/2, phi[i],
                               'field_phi_', stream_type)
        else:
            r_eq_mid[i] = (r_max+r_min)/2
            itr = 0
            notfound = True
            r_eq_min, r_eq_max = r_min, r_max
            while(notfound and itr < itr_max):
                #This is a bisection root finding algorithm
                create_stream_zone(field_data, r_eq_mid[i], pi/2, phi[i],
                                   'temp_field_phi_', stream_type)
                mid_closed = check_streamline_closed(field_data,
                                                     'temp_field_phi_',
                                                     r_eq_mid[i],
                                                     stream_type)
                if mid_closed:
                    r_eq_min = r_eq_mid[i]
                else:
                    r_eq_max = r_eq_mid[i]
                if abs(r_eq_min - r_eq_max) < tolerance and mid_closed:
                    notfound = False
                    field_data.zone('temp*').name ='field_phi_{:.1f}'.format(
                                                         rad2deg(phi[i]))
                else:
                    r_eq_mid[i] = (r_eq_max+r_eq_min)/2
                    field_data.delete_zones(field_data.zone('temp_field*'))
                itr += 1




def calc_tail_mp(field_data, psi, x_disc, rho_max, rho_step):
    """Function to create the zones that will become the tail magnetopause
    Inputs
        psi- set of disc azimuthal angles
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
        r_tail, theta_tail, phi_tail = find_tail_disc_point(rho_max, psi[i],
                                                            x_disc)
        #check if north or south attached
        x_pos, y_pos, z_pos = sph_to_cart(r_tail, theta_tail, phi_tail)
        if tp.data.query.probe_at_position(x_pos, y_pos, z_pos)[0][7] < 0:
            stream_type = 'south'
        else:
            stream_type = 'north'
        create_stream_zone(field_data, r_tail, theta_tail, phi_tail,
                           'temp_tail_line_', stream_type)
        #check if closed
        tail_closed = check_streamline_closed(field_data,
                                              'temp_tail_line_', r_tail,
                                              stream_type)
        if tail_closed:
            print('WARNING: field line closed at RHO_MAX={}R_e'.format(
                                                                 rho_max))
            field_data.zone('temp_tail*').name='tail_field_{:.1f}'.format(
                                                        rad2deg(psi[i]))
        else:
            #This is a basic marching algorithm from outside in starting at
            #RHO_MAX
            rho_tail = rho_max
            notfound = True
            while notfound and rho_tail > rho_step:
                field_data.delete_zones(field_data.zone('temp_tail_line*'))
                rho_tail = rho_tail - rho_step
                r_tail, theta_tail, phi_tail = find_tail_disc_point(
                                                rho_tail, psi[i], x_disc)
                #check if north or south attached
                x_pos, y_pos, z_pos = sph_to_cart(r_tail, theta_tail,
                                                  phi_tail)
                if tp.data.query.probe_at_position(x_pos,
                                                   y_pos, z_pos)[0][7] < 0:
                    stream_type = 'south'
                else:
                    stream_type = 'north'
                create_stream_zone(field_data, r_tail, theta_tail,
                                 phi_tail, 'temp_tail_line_', stream_type)
                tail_closed =check_streamline_closed(field_data,
                                                     'temp_tail_line_',
                                                    rho_tail, stream_type)
                if tail_closed:
                    field_data.zone('temp*').name='tail_field_{:.1f}'.format(
                                                        rad2deg(psi[i]))
                    notfound = False
                    print('Tail', i,' rho{:.1f} psi{:.1f}'.format(rho_tail,
                                                       rad2deg(psi[i])))
                if rho_tail <= rho_step:
                    print('WARNING: not possible at psi={:.1f}'.format(
                                                       rad2deg(psi[i])))


def calc_plasmasheet(field_data, theta_max, phi_list, tail_cap,
                     max_iter, tolerance):
    """Function to create tecplot zones that define the plasmasheet boundary
    Inputs
        field_data- tecplot Dataset containing 3D field data
        theta_max- max co-latitude value corresponding to near equator
        phi_list- list containing longitudinal positions to be searched
        tail_cap
        max_iter- max iterations for bisection algorithm
        tolerance
    """
    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('B_x*')
    plot.vector.v_variable = field_data.variable('B_y*')
    plot.vector.w_variable = field_data.variable('B_z*')
    seed_radius = 1.5

    #iterate through northside zones
    for phi in phi_list:
        print('phi {:.1f}'.format(rad2deg(phi)))
        #initialize latitude search bounds
        if theta_max > pi/2:
            pole_theta = pi-theta_max
        else:
            pole_theta = pi/18
        equat_theta = theta_max
        mid_theta = (pole_theta+equat_theta)/2

        """
        #Testing for XZ visulalizations
        map_index = 0
        for i in np.append(np.linspace(pi/18,theta_max, 20),
                           np.linspace(pi+pi/18, pi-theta_max, 20)):
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
        create_stream_zone(field_data, seed_radius, pole_theta, phi,
                           'temp_poleward_', 'inner_mag')
        poleward_closed = check_streamline_closed(field_data,
                                                  'temp_poleward',
                                                  abs(tail_cap),
                                                  'inner_mag')
        create_stream_zone(field_data, seed_radius, equat_theta, phi,
                           'temp_equatorward_', 'inner_mag')
        equatorward_closed = check_streamline_closed(field_data,
                                                 'temp_equatorward',
                                                 abs(tail_cap),
                                                 'inner_mag')
        #check if both are open are close to start
        if poleward_closed and equatorward_closed:
            print('Warning: high and low lat {:.2f}, {:.2f} '.format(
                                                  np.rad2deg(pole_theta),
                                                  np.rad2deg(equat_theta))+
                  'closed at lon {:.1f}'.format(np.rad2deg(phi)))
            create_stream_zone(field_data, seed_radius-0.5, mid_theta,
                               phi, 'plasma_sheet_', 'inner_mag')
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
        elif not poleward_closed and (not equatorward_closed):
            print('Warning: high and low lat {:.2f}, {:.2f} open'.format(
                        np.rad2deg(pole_theta), np.rad2deg(equat_theta)))
            create_stream_zone(field_data, seed_radius-0.5, mid_theta,
                               phi, 'plasma_sheet_', 'inner_mag')
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
        else:
            field_data.delete_zones(field_data.zone('temp*'))
            field_data.delete_zones(field_data.zone('temp*'))
            notfound = True
            itr = 0
            while (notfound and itr < max_iter):
                mid_theta = (pole_theta+equat_theta)/2
                create_stream_zone(field_data, seed_radius-0.5,
                                   mid_theta, phi,
                                   'temp_ps_line_', 'inner_mag')
                mid_lat_closed = check_streamline_closed(field_data,
                                                         'temp_ps_line_',
                                                         abs(tail_cap),
                                                         'inner_mag')
                if mid_lat_closed:
                    equat_theta = mid_theta
                else:
                    pole_theta = mid_theta
                if abs(pole_theta - equat_theta)<tolerance and (
                                                        mid_lat_closed):
                    notfound = False
                    field_data.zone('temp*').name = 'plasma_sheet_'.format(
                                                   np.rad2deg(equat_theta))
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

def create_cylinder(field_data, nx, nalpha, x_min, x_max, zone_name):
    """Function creates empty cylindrical zone for loading of slice data
    Inputs
        field_data- tecplot Dataset class with 3D field data
        nx- number of x positions, same as n_slice
        nalpha
        x_min
        x_max
        zone_name
    """
    #use built in create zone function for verticle cylinder
    tp.macro.execute_command('''$!CreateCircularZone
                             IMax = 2
                             JMax = {:d}
                             KMax = {:d}
                             X = 0
                             Y = 0
                             Z1 = {:f}
                             Z2 = {:f}
                             Radius = 50'''.format(nalpha,nx,x_min,x_max))

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
    #data = pd.read_csv(filename)
    xdata = data['X [R]'].values.copy()
    ydata = data['Y [R]'].values.copy()
    zdata = data['Z [R]'].values.copy()
    #ndata = np.meshgrid(xdata,ydata,zdata)
    mag_bound.values('X*')[1::2] = xdata
    mag_bound.values('Y*')[1::2] = ydata
    mag_bound.values('Z*')[1::2] = zdata


def calculate_energetics(field_data, zone_name):
    """Function calculates values for energetics tracing
    Inputs
        field_data- tecplot Dataset class containing 3D field data
    """
    zone_index= field_data.zone(zone_name).index
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'GRIDKUNITNORMAL VALUELOCATION = '+
                                      'CELLCENTERED')
    eq = tp.data.operate.execute_equation
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
    eq('{Kn_x [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{K_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{K_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2'+
                                  '+{Z Grid K Unit Normal}**2'+
                                  '+1e-25)'+
                          '* {X Grid K Unit Normal}',
       zones=[zone_index])
    eq('{Kn_y [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{K_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{K_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2'+
                                  '+{Z Grid K Unit Normal}**2'+
                                  '+1e-25)'+
                          '* {Y Grid K Unit Normal}',
        zones=[zone_index])
    eq('{Kn_z [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{K_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{K_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2'+
                                  '+{Z Grid K Unit Normal}**2'+
                                  '+1e-25)'+
                          '* {Z Grid K Unit Normal}',
        zones=[zone_index])

    #Magnitude Normal Flux
    eq('{K_in [kW/km^2]} = ({Kn_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{Kn_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{Kn_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2 '+
                                  '+{Z Grid K Unit Normal}**2'+
                                  '+1e-25)',
        zones=[zone_index])

    #Split into + and - flux
    eq('{K_in+} = max({K_in [kW/km^2]},0)', zones=[zone_index])
    eq('{K_in-} = min({K_in [kW/km^2]},0)', zones=[zone_index])


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

def integrate_surface(var_index, zone_index, qtname, *, frame_id='main'):
    """Function to calculate integral of variable on mp surface
    Inputs
        var_index- variable to be integrated
        zone_index- index of the zone to perform integration
        qtname- integrated quantity will be saved as this name
        frame_id- frame name with the surface that integral is performed
    Output
        newframe- created frame with integrated quantity
    """
    #Integrate total surface Flux
    frame=[fr for fr in tp.frames(frame_id)][0]
    frame.activate()
    print('\nframe selected: {}\n'.format(tp.active_frame().name))
    integrate_command=("Integrate [{:d}] ".format(zone_index+1)+
                       "ScalarVar={:d} ".format(var_index+1)+
                       "XVariable=1 "+
                       "YVariable=2 "+
                       "ZVariable=3 "+
                       "PlotResults='T' "+
                       "PlotAs='"+qtname+"' "+
                       "TimeMin=0 TimeMax=0")
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command)

    #Rename and hide newly created frame
    print('after integration')
    for fr in tp.frames():
        print(fr.name)
    newframe = [fr for fr in tp.frames('Frame*')][0]
    newframe.name = qtname
    #Create dummy frame as workaround, possibly version dependent issue??
    tp.macro.execute_command('$!CreateNewFrame')
    newframe.move_to_bottom()
    print('after dummy frame')
    for fr in tp.frames():
        print(fr.name)
    return newframe

def write_to_timelog(timelogname, sourcename, data):
    """Function for writing the results from the current file to a file that contains time integrated data
    Inputs
        timelogname
        sourcename
        data- pandas DataFrame object that will be written into the file
    """
    #get the time entries for this file
    from global_energetics.makevideo import get_time
    abstime = get_time(sourcename)
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
    #write data to file
    with open(timelogname, 'a') as log:
        log.seek(0,2)
        log.write('\n')
        for entry in timestamp:
            log.write(str(entry)+', ')
        for num in data.values[0]:
            log.write(str(num)+',')


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
