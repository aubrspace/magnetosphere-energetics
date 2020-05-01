#!/usr/bin/env python3
"""SWMF Energetics with Tecplot
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
import mpsurface_recon

log.basicConfig(level=log.INFO)
start_time = time.time()

def create_stream_zone(r_start, theta_start, phi_start,
                       zone_name, stream_type):
    """Function to create a streamline, created in 2 directions from
       starting point
    Inputs
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
    if stream_type == 'day':
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Both)
    elif stream_type == 'north':
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Forward)
    else:
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Reverse)
    # Create zone
    field_line.extract()
    SWMF_DATA.zone(-1).name = zone_name + '{}'.format(phi_start)
    # Delete streamlines
    field_line.delete_all()


def check_streamline_closed(zone_name, r_seed, r_cap, stream_type):
    """Function to check if a streamline is open or closed
    Inputs
        zone_name
        r_seed [R]- position used to seed field line
        r_cap [R]- radius of cap that determines if line is closed
        stream_type- dayside, north or south from tail
    Outputs
        isclosed- boolean, True for closed
        max(r_end_n, r_cap)- furthest out point at pole, for making smooth
        surface on the caps
        max(r_end_s, r_cap)
    """
    # Get starting and endpoints of streamzone
    r_values = SWMF_DATA.zone(zone_name+'*').values('r *').as_numpy_array()
    if stream_type == 'north':
        r_end_n = r_values[-1]
        r_end_s = 0
        r_seed = 2
    elif stream_type == 'south':
        r_end_n = 0
        r_end_s = r_values[0]
        r_seed = 2
    else:
        r_end_n, r_end_s = r_values[0], r_values[-1]
    #check if closed
    if (r_end_n > r_seed) or (r_end_s > r_seed):
        isclosed = False
    else:
        isclosed = True
    return isclosed, max(r_end_n, r_cap), max(r_end_s, r_cap)

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


def calc_dayside_mp(phi, r_max, r_min):
    """"Function to create zones that will makeup dayside magnetopause
    Inputs
        phi- set of phi angle points
        r_max- maxium radial distance for equitorial search
        r_min- min
    Outputs
        r_north- set of cuttoff points for northern hemisphere
        r_south- southern hemisphere
    """
    #Initialize objects that will be modified in creation loop
    r_eq_mid = np.zeros(int(len(phi)))
    r_north = np.zeros(int(len(phi)))
    r_south = np.zeros(int(len(phi)))
    itr = 0
    r_eq_max, r_eq_min = r_max, r_min

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = SWMF_DATA.variable('B_x*')
    plot.vector.v_variable = SWMF_DATA.variable('B_y*')
    plot.vector.w_variable = SWMF_DATA.variable('B_z*')


    #Create Dayside Magnetopause field lines
    stream_type = 'day'
    for i in range(int(len(phi))):
        #Create initial max min and mid field lines
        create_stream_zone(r_min, pi/2, phi[i], 'min_field_line',
                           stream_type)
        create_stream_zone(r_max, pi/2, phi[i], 'max_field_line',
                           stream_type)
        #Check that last closed is bounded
        min_closed, _, __ = check_streamline_closed('min_field_line',
                                                  r_min, R_CAP, stream_type)
        max_closed, _, __ = check_streamline_closed('max_field_line',
                                                  r_max, R_CAP, stream_type)
        SWMF_DATA.delete_zones(SWMF_DATA.zone('min_field*'),
                               SWMF_DATA.zone('max_field*'))
        print('Day', i,'phi: {:.1f}, iters: {}, err: {}'.format(
                                                    np.rad2deg(phi[i]),
                                                    itr, r_eq_max-r_eq_min))
        if max_closed and min_closed:
            print('WARNING: field line closed at max {}R_e'.format(r_max))
            create_stream_zone(r_max, pi/2, phi[i], 'field_phi_',
                               stream_type)
        elif not max_closed and not min_closed:
            print('WARNING: first field line open at {}R_e'.format(r_min))
            create_stream_zone(r_min, pi/2, phi[i], 'field_phi_',
                               stream_type)
        else:
            r_eq_mid[i] = (r_max+r_min)/2
            itr = 0
            notfound = True
            r_eq_min, r_eq_max = r_min, r_max
            while(notfound and itr < ITR_MAX):
                #This is a bisection root finding algorithm
                create_stream_zone(r_eq_mid[i], pi/2, phi[i],
                                   'temp_field_phi_', stream_type)
                mid_closed, r_north[i],r_south[i] = check_streamline_closed(
                                                         'temp_field_phi_',
                                                         r_eq_mid[i], R_CAP,
                                                         stream_type)
                if mid_closed:
                    r_eq_min = r_eq_mid[i]
                else:
                    r_eq_max = r_eq_mid[i]
                if abs(r_eq_min - r_eq_max) < TOL and mid_closed:
                    notfound = False
                    SWMF_DATA.zone('temp*').name ='field_phi_{:.1f}'.format(
                                                         np.rad2deg(phi[i]))
                else:
                    r_eq_mid[i] = (r_eq_max+r_eq_min)/2
                    SWMF_DATA.delete_zones(SWMF_DATA.zone('temp_field*'))
                itr += 1
    return r_north, r_south




def calc_tail_mp(psi, x_disc, rho_max, rho_step):
    """Function to create the zones that will become the tail magnetopause
    Inputs
        psi- set of disc azimuthal angles
        x_disc- x position of the tail disc
        rho_max- outer radial bounds of the tail disc
        rho_step- radial distance increment for marching algorithm
    """
    #Initialize objects that will be modified in creation loop
    rho_tail = rho_max
    r_north = np.zeros(int(len(psi)))
    r_south = np.zeros(int(len(psi)))

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = SWMF_DATA.variable('B_x*')
    plot.vector.v_variable = SWMF_DATA.variable('B_y*')
    plot.vector.w_variable = SWMF_DATA.variable('B_z*')

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
        create_stream_zone(r_tail, theta_tail, phi_tail,
                           'temp_tail_line_', stream_type)
        #check if closed
        tail_closed, r_north[i], r_south[i] = check_streamline_closed(
                                                  'temp_tail_line_', r_tail,
                                                  R_CAP, stream_type)
        if tail_closed:
            print('WARNING: field line closed at RHO_MAX={}R_e'.format(
                                                                   rho_max))
            SWMF_DATA.zone('temp_tail*').name = 'tail_field_{:.1f}'.format(
                                                        np.rad2deg(psi[i]))
        else:
            #This is a basic marching algorithm from outside in starting at
            #RHO_MAX
            rho_tail = rho_max
            notfound = True
            while notfound and rho_tail > rho_step:
                SWMF_DATA.delete_zones(SWMF_DATA.zone('temp_tail_line*'))
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
                create_stream_zone(r_tail, theta_tail, phi_tail,
                                   'temp_tail_line_', stream_type)
                tail_closed, r_north[i],r_south[i] =check_streamline_closed(
                                                         'temp_tail_line_',
                                                         rho_tail, R_CAP,
                                                         stream_type)
                if tail_closed:
                    SWMF_DATA.zone('temp*').name='tail_field_{:.1f}'.format(
                                                        np.rad2deg(psi[i]))
                    notfound = False
                    print('Tail', i,' rho{:.1f} psi{:.1f}'.format(rho_tail,
                                                       np.rad2deg(psi[i])))
                if rho_tail <= rho_step:
                    print('WARNING: not possible at psi={:.1f}'.format(
                                                       np.rad2deg(psi[i])))


def dump_to_pandas():
    """Function to hand zone data to pandas to do processing
    Inputs-
    Outputs-
        loc_data- DataFrame of stream zone data
        x_max
    """
    #Export 3D point data to csv file
    os.system('touch mp_points.csv | rm mp_points.csv')
    tp.macro.execute_extended_command(command_processor_id='excsv',
            command='VarNames:'+
                    'FrOp=1:'+
                    'ZnCount={:d}:'.format(SWMF_DATA.num_zones-1)+
                    'ZnList=[2-{:d}]:'.format(SWMF_DATA.num_zones)+
                    'VarCount=3:'+
                    'VarList=[1-3]:'+
                    'ValSep=",":'+
            'FNAME="/Users/ngpdl/Code/swmf-energetics/stream_points.csv"')
    loc_data = pd.read_csv('stream_points.csv')
    loc_data = loc_data.drop(columns=['Unnamed: 3'])
    loc_data = loc_data.sort_values(by=['X [R]'])
    x_max = loc_data['X [R]'].max()
    return loc_data, x_max

def create_cylinder(nx, nalpha, x_min, x_max):
    """Function creates empty cylindrical zone for loading of slice data
    Inputs-
        nx- number of x positions, same as n_slice
        nalpha
        x_min
        x_max
    """
    #use built in create zone function for verticle cylinder
    tp.macro.execute_command('''$!CreateCircularZone
                             IMax = 2
                             JMax = {:d}
                             KMax = {:d}
                             X = 0
                             Y = 0
                             Z1 = {:d}
                             Z2 = {:d}
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
                             NormalZ = 0'''.format(SWMF_DATA.num_zones))

    #delete verticle cylinder
    SWMF_DATA.delete_zones(SWMF_DATA.zone('Circular zone'))
    SWMF_DATA.zone('Circular*').name = 'mp_zone'

def load_cylinder(filename, zonename, I, J, K):
    """Function to load processed slice data into cylindrial ordered zone
       I, J, K -> radial, azimuthal, axial
    Inputs
        filename- path to .csv file for loading data
        zonename- name of cylindrical zone
        I- vector of I coordinates (0 to 1)
        J- vector of J coordinates (0 to num_alpha)
        K- vector of K coordinates (0 to num_slices)
    """
    mag_bound = SWMF_DATA.zone(zonename)
    data = pd.read_csv(filename)
    xdata = data['X'].values.copy()
    ydata = data['Y'].values.copy()
    zdata = data['Z'].values.copy()
    #ndata = np.meshgrid(xdata,ydata,zdata)
    mag_bound.values('X*')[1::2] = xdata
    mag_bound.values('Y*')[1::2] = ydata
    mag_bound.values('Z*')[1::2] = zdata

def calculate_energetics():
    """Function calculates values for energetics tracing
    """
    zone_index= SWMF_DATA.zone('mp_zone').index
    tp.macro.execute_extended_command('CFDAnalyzer3', 'CALCULATE FUNCTION = GRIDKUNITNORMAL VALUELOCATION = CELLCENTERED')
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
                                  '+{Z Grid K Unit Normal}**2)'+
                          '* {X Grid K Unit Normal}',
       zones=[zone_index])
    eq('{Kn_y [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{K_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{K_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2'+
                                  '+{Z Grid K Unit Normal}**2)'+
                          '* {Y Grid K Unit Normal}',
        zones=[zone_index])
    eq('{Kn_z [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{K_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{K_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2'+
                                  '+{Z Grid K Unit Normal}**2)'+
                          '* {Z Grid K Unit Normal}',
        zones=[zone_index])

    #Magnitude Normal Flux
    eq('{K_in [Kw/km^2]} = ({Kn_x [kW/km^2]}*{X Grid K Unit Normal}'+
                            '+{Kn_y [kW/km^2]}*{Y Grid K Unit Normal}'+
                            '+{Kn_z [kW/km^2]}*{Z Grid K Unit Normal})'+
                          '/ sqrt({X Grid K Unit Normal}**2'+
                                  '+{Y Grid K Unit Normal}**2 '+
                                  '+{Z Grid K Unit Normal}**2)',
        zones=[zone_index])


# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    if '-n' in sys.argv:
        os.system('open -a Tecplot\ 360\ EX\ 2019\ R1')
        tp.macro.execute_extended_command("TecUtilServer",'R"('+
                                          'AcceptRequests = Yes'+
                                          'ListenOnAddress = localhost'+
                                          'ListenOnPort = 7600)"')

    if '-c' in sys.argv:
        tp.session.connect()


    tp.new_layout()

    #Load .plt file, come back to this later for batching
    log.info('loading .plt and reformatting')
    SWMF_DATA = tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')
    SWMF_DATA.zone(0).name = 'global_field'
    print(SWMF_DATA)

    #Create R from cartesian coordinates
    tp.data.operate.execute_equation(
                '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')

    #Set the parameters for streamline seeding
    #DaySide
    N_AZIMUTH_DAY = 50
    AZIMUTH_MAX = 122
    AZIMUTH_RANGE = [np.deg2rad(-1*AZIMUTH_MAX), np.deg2rad(AZIMUTH_MAX)] #need to come back
    PHI = np.linspace(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1], N_AZIMUTH_DAY)
    R_MAX = 30
    R_MIN = 3.5

    #Tail
    N_AZIMUTH_TAIL = 50
    PSI = np.linspace(-pi*(1-pi/N_AZIMUTH_TAIL), pi, N_AZIMUTH_TAIL)
    RHO_MAX = 50
    RHO_STEP = 0.5
    X_TAIL_CAP = -20

    #Other
    R_CAP = 3.5
    ITR_MAX = 100
    TOL = 0.1
    SPARSENESS = 50

    with tp.session.suspend():
        #Create Dayside Magnetopause field lines
        calc_dayside_mp(PHI, R_MAX, R_MIN)

        #Create Tail magnetopause field lines
        calc_tail_mp(PSI, X_TAIL_CAP, RHO_MAX, RHO_STEP)

        #Create Theta and Phi coordinates for all points in domain
        tp.data.operate.execute_equation(
                                   '{phi} = atan({Y [R]}/({X [R]}+1e-24))')
        tp.data.operate.execute_equation(
                                   '{theta} = acos({Z [R]}/{r [R]}) * '+
                                    '({X [R]}+1e-24) / abs({X [R]}+1e-24)')

        #port stream data to pandas DataFrame object
        STREAM_DF, X_MAX = dump_to_pandas()

        #slice and construct XYZ data
        MP_MESH = mpsurface_recon.yz_slicer(STREAM_DF, X_TAIL_CAP, X_MAX,
                                         50, 50, False)

        #create and load cylidrical zone
        create_cylinder(50, 50, -20, 10)
        load_cylinder('slice_mesh.csv', 'mp_zone',
                      range(0,2), range(0,50), range(0,50))

        #interpolate field data to zone
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=SWMF_DATA.zone('mp_zone'),
                source_zones=SWMF_DATA.zone('global_field'))

        #calculate energetics
        calculate_energetics()

        #Show spatial axes labels
        #tp.active_frame().plot().fieldmap(0).show = False
        #tp.active_frame().plot().fieldmap(1).show = True
        #tp.active_frame().plot().show_mesh = True
        #tp.active_frame().plot().view.fit()
        #tp.active_frame().plot(PlotType.Cartesian3D).show_isosurfaces=True
        #tp.active_frame().plot().contour(0).variable_index=14
        #tp.active_frame().plot().isosurface(0).isosurface_values[0]=1
        #tp.active_frame().plot(PlotType.Cartesian3D).show_slices=True
        #tp.active_frame().plot().slice(0).origin=(X_TAIL_CAP,
        #                        tp.active_frame().plot().slice(0).origin[1],
        #                        tp.active_frame().plot().slice(0).origin[2])
        ltime = time.time()-start_time
        print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                               np.mod(ltime,60)))
#
#        -------------------------------------------------------------
#                 Function to check if a streamzone is open or closed
#                 Inputs |1| -> Streamzone number / ID
#                        |2| -> rPoint
#
#
# ============================================================================
# Create the dayside magnetopause zone
#
#
#
# ==================================================================================================
# Interpolate field data and calculate normal energy flux on magnetopause zone
#
# Interpolate data
#
# Create MP surface normal vector
#
# Calculate energy flux for all zones
#     Electric field (no resistivity)
#
#     Poynting flux
#
#     Total Energy flux
#
# Calculate orthogonal projection of energy flux through magnetopause
#     Component normal flux
#
#     Signed magnitude normal flux
