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
from progress.bar import Bar
from progress.spinner import Spinner
import spacepy
#from spacepy import coordinates as coord
from spacepy import time as spacetime
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd

def create_stream_zone(field_data, x1start, x2start, x3start,
                       zone_name, *, line_type=None, cart_given=False):
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
    """
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
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Reverse)
    elif line_type == 'north':
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Forward)
    else:
        field_line.add(seed_point=[x_start, y_start, z_start],
                       stream_type=Streamtrace.VolumeLine,
                       direction=StreamDir.Both)
    # Create zone
    field_line.extract()
    field_data.zone(-1).name = zone_name
    # Delete streamlines
    field_line.delete_all()


def check_streamline_closed(field_data, zone_name, r_seed, line_type):
    """Function to check if a streamline is open or closed
    Inputs
        field_data- tecplot Dataset class with 3D field data
        zone_name
        r_seed [R]- position used to seed field line
        line_type- dayside, north or south from tail
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
    elif line_type == 'inner_mag':
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
'''
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
    #from IPython import embed; embed()
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

def streamfind_bisection(field_data, method,
                         dimmax, nstream, rmax, rmin, itr_max, tolerance,
                         *, rcheck=5, time=None,
                         field_key_x='B_x*', global_key='global_field'):
    """Generalized function for generating streamlines for a 'last closed'          condition based on a bisection algorithm
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
    approved_methods = ['dayside', 'tail', 'flow', 'plasmasheet']
    reverse_if_flow = 0
    if all([test.find(method)==-1 for test in approved_methods]):
        print('WARNING: method for streamfind_bisection not recognized!!'+
              '\nno streamzones created')
        zonelist = None
        return zonelist

    #set streamline seed positions in the fixed dimension
    if method == 'dayside':
        #set of points in z=0 plane from + to - dimmax angle from +x ax
        positions = np.linspace(-1*dimmax, dimmax, nstream)
        disp_message = 'Finding Magnetopause Dayside Field Lines'
        lin_type = 'dayside'
        lineID = 'day_{:.1f}'
        l_max = 2*pi*15/nstream
    elif (method == 'tail') or (method == 'flow'):
        #set of points 360deg around disc in the tail at x=dimmax
        positions = np.linspace(-180*(1-1/nstream), 180, nstream)
        l_max = 2*pi*25/nstream
        if method == 'tail':
            disp_message = 'Finding Magnetopause Tail Field Lines'
            lineID = 'tail_{:.1f}'
        elif method == 'flow':
            disp_message = 'Finding Magnetopause Flow Lines'
            lin_type = 'flowline'
            lineID = 'flow_{:.1f}'
            reverse_if_flow = 1
    elif method == 'plasmasheet':
        #set of points on r=1Re at different longitudes N and S poles
        positions = np.append(np.linspace(-dimmax, -180,int(nstream/4)),
                              np.linspace(180, dimmax, int(nstream/4)))
        disp_message = 'Finding Plasmasheet Field Lines'
        lin_type = 'inner_mag'
        lineID = 'plasmasheet_{:.1f}'

    #set vector field
    field_key_y = field_key_x.split('x')[0]+'y'+field_key_x.split('x')[-1]
    field_key_z = field_key_x.split('x')[0]+'z'+field_key_x.split('x')[-1]
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable(field_key_x)
    plot.vector.v_variable = field_data.variable(field_key_y)
    plot.vector.w_variable = field_data.variable(field_key_z)

    bar = Bar(disp_message, max=len(positions))
    zonelist = []
    rold = 0
    aold = positions[0]
    for a_int in positions:
        #start at intial position
        a = a_int
        while True:
            notfound = True
            #Set getseed function based only on search variable r
            if method == 'dayside':
                #all getseed function return val: dim1, dim2, dim3,
                #                                 rcheck, lin_type
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
                    return [dimmax, r*cos(rad2deg(a)), r*sin(rad2deg(a)),
                            rcheck, lin_type]
                cartesian = True
            elif method == 'plasmasheet':
                def getseed(r):return sm2gsm_temp(1, r, a, time), None, lin_type
                cartesian = True
            #Create initial max/min to lines
            x1, x2, x3, rchk, lin_type = getseed(rmax)
            create_stream_zone(field_data, x1, x2, x3, 'max'+lineID.format(a),
                            cart_given=cartesian)
            x1, x2, x3, rchk, lin_type = getseed(rmin)
            create_stream_zone(field_data, x1, x2, x3, 'min'+lineID.format(a),
                            cart_given=cartesian)
            #Check that last closed is bounded, delete min/max
            max_closed = check_streamline_closed(field_data, 'max*', rchk,
                                                line_type=lin_type)
            min_closed = check_streamline_closed(field_data, 'min*', rchk,
                                                line_type=lin_type)
            field_data.delete_zones(field_data.zone('min*'),
                                    field_data.zone('max*'))
            if max_closed and min_closed:
                #print("WARNING:flowlines closed at {}R_e from YZ".format(rmax))
                x1, x2, x3, rchk, lin_type = getseed(rmax)
                create_stream_zone(field_data, x1, x2, x3, lineID.format(a),
                                cart_given=cartesian)
                notfound = False
                rmid = rmax
            elif not max_closed and not min_closed:
                #print("WARNING:flowlines good at {}R_e from YZ".format(rmin))
                x1, x2, x3, rchk, lin_type = getseed(rmin)
                create_stream_zone(field_data, x1, x2, x3, lineID.format(a),
                                cart_given=cartesian)
                notfound = False
                rmid = rmin
            else:
                rmid = (rmax+rmin)/2
                itr = 0
                rout = rmax
                rin = rmin
                while(notfound and itr < itr_max):
                    #create mid
                    x1, x2, x3, rchk, lin_type = getseed(rmid)
                    create_stream_zone(field_data, x1, x2, x3,
                                    'mid'+lineID.format(a),
                                    cart_given=cartesian)
                    #check midclosed
                    mid_closed = check_streamline_closed(field_data, 'mid*',
                                                    rchk, line_type=lin_type)
                    if mid_closed:
                        rin = rmid
                    else:
                        rout = rmid
                    if abs(rout-rin) < tolerance and (bool(
                                            int(mid_closed)-reverse_if_flow)):
                        #keep closed on boundary unless in 'flowline' method
                        notfound = False
                        field_data.zone('mid*').name=lineID.format(a)
                    else:
                        rmid = (rin+rout)/2
                        field_data.delete_zones(field_data.zone('mid*'))
                    itr += 1
            zonelist.append(field_data.zone(lineID.format(a)).index)
            #after adding to the list, check l_arc
            l_arc = abs(a-aold) * abs(rmid - rold)/2
            if l_arc < l_max:
                rold = rmid
                aold = a_int
                break
            #half the distance and loop again
            a = a - (a-aold)/2
            rold = rmid
            aold = a
        bar.next()
    bar.finish()
    return zonelist

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
    print('converting '+filename.split('.')[0]+' to DataFrame')
    os.system('touch '+filename)
    #Export 3D point data to csv file
    tp.macro.execute_extended_command(command_processor_id='excsv',
            command='VarNames:'+
                    'FrOp=1:'+
                    'ZnCount={:d}:'.format(len(zonelist))+
                    'ZnList=[{:d}-{:d}]:'.format(int(zonelist[0]+1),
                                                 int(zonelist[-1])+1)+
                    'VarCount={:d}:'.format(len(varlist))+
                    'VarList=[{:d}-{:d}]:'.format(int(varlist[0]),
                                                  int(varlist[-1]))+
                    'ValSep=",":'+
                    'FNAME="'+os.getcwd()+'/'+filename+'"')
    loc_data = pd.read_csv(filename)
    if any(col == 'X [R]' for col in loc_data.columns):
        loc_data = loc_data.drop(columns=['Unnamed: 3'])
        loc_data = loc_data.sort_values(by=['X [R]'])
        loc_data = loc_data.reset_index(drop=True)
        x_max = loc_data['X [R]'].max()
    else: x_max = []
    #Delete csv file
    os.system('rm '+os.getcwd()+'/'+filename)
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
    bar = Bar('Loading cylinder centers', max=K)
    for k in range(0,K):
        ymean[k*J:(k+1)*J] = np.mean(ydata[k*J:(k+1)*J])
        zmean[k*J:(k+1)*J] = np.mean(zdata[k*J:(k+1)*J])
        bar.next()
    bar.finish()
    bar = Bar('Loading I meshpoints', max=I)
    for i in range(0,I):
        mag_bound.values('X*')[i::I] = xdata
        mag_bound.values('Y*')[i::I] = (ydata-ymean) * i/I + ymean
        mag_bound.values('Z*')[i::I] = (zdata-zmean) * i/I + zmean
        bar.next()
    bar.finish()


def calculate_energetics(field_data, zone_name):
    """Function calculates values for energetics tracing
    Inputs
        field_data- tecplot Dataset class containing 3D field data
    """
    zone_index= field_data.zone(zone_name).index
    mu0 = 4*pi*1e-7
    Re = 6371
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

    #Magnetic field unit vectors
    eq('{bx} ={B_x [nT]}/sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)')
    eq('{by} ={B_y [nT]}/sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)')
    eq('{bz} ={B_z [nT]}/sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)')

    #Magnetic Energy per volume
    eq('{uB [J/Re^3]} = ({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                        '*0.205785')

    #Ram pressure energy per volume
    eq('{KEpar [J/Re^3]} = {Rho [amu/cm^3]}/2 *'+
                                    '(({U_x [km/s]}*{bx})**2+'+
                                    '({U_y [km/s]}*{by})**2+'+
                                    '({U_z [km/s]}*{bz})**2) * 4.2941e5')
    eq('{KEperp [J/Re^3]} = {Rho [amu/cm^3]}/2 *'+
                          '(({U_y [km/s]}*{bz} - {U_z [km/s]}*{by})**2+'+
                           '({U_z [km/s]}*{bx} - {U_x [km/s]}*{bz})**2+'+
                           '({U_x [km/s]}*{by} - {U_y [km/s]}*{bx})**2)'+
                                                            '*4.2941e5')

    #Motional Electric Field
    eq('{Em_x [mV/km]} = ({U_z [km/s]}*{B_y [nT]}'+
                          '-{U_y [km/s]}*{B_z [nT]})')
    eq('{Em_y [mV/km]} = ({U_x [km/s]}*{B_z [nT]}'+
                         '-{U_z [km/s]}*{B_x [nT]})')
    eq('{Em_z [mV/km]} = ({U_y [km/s]}*{B_x [nT]}'+
                         '-{U_x [km/s]}*{B_y [nT]})')

    #Hall Electric Field
    eq('{Eh_x [mV/km]} = ({J_z [`mA/m^2]}/{Rho [amu/cm^3]}*{B_y [nT]}'+
                          '-{J_y [`mA/m^2]}/{Rho [amu/cm^3]}*{B_z [nT]})'+
                          '*6.2415097523028e6')
    eq('{Eh_y [mV/km]} = ({J_x [`mA/m^2]}/{Rho [amu/cm^3]}*{B_z [nT]}'+
                         '-{J_z [`mA/m^2]}/{Rho [amu/cm^3]}*{B_x [nT]})'+
                          '*6.2415097523028e6')
    eq('{Eh_z [mV/km]} = ({J_y [`mA/m^2]}/{Rho [amu/cm^3]}*{B_x [nT]}'+
                         '-{J_x [`mA/m^2]}/{Rho [amu/cm^3]}*{B_y [nT]})'+
                          '*6.2415097523028e6')

    #Total Electric Field
    eq('{E_x [mV/km]} = ({Em_x [mV/km]}+{Eh_x [mV/km]})')
    eq('{E_y [mV/km]} = ({Em_y [mV/km]}+{Eh_y [mV/km]})')
    eq('{E_z [mV/km]} = ({Em_z [mV/km]}+{Eh_z [mV/km]})')

    #Electric Energy per volume
    eq('{uE [J/Re^3]} = ({E_x [mV/km]}**2+{E_y [mV/km]}**2+'+
                        '{E_z [mV/km]}**2)*1.143247989e-3')

    #Poynting Flux
    eq('{ExB_x [W/Re^2]} = -(3.22901e4)*({E_z [mV/km]}*{B_y [nT]}'+
                                       '-{E_y [mV/km]}*{B_z [nT]})')
    eq('{ExB_y [W/Re^2]} = -(3.22901e4)*({E_x [mV/km]}*{B_z [nT]}'+
                                       '-{E_z [mV/km]}*{B_x [nT]})')
    eq('{ExB_z [W/Re^2]} = -(3.22901e4)*({E_y [mV/km]}*{B_x [nT]}'+
                                       '-{E_x [mV/km]}*{B_y [nT]})')
    #Total Energy Flux
    eq('{K_x [W/Re^2]} = ({P [nPa]}*(1.666667/0.666667)*4.9430863e10'+
                                        '+4.2941e5*{Rho [amu/cm^3]}/2*'+
                                                    '({U_x [km/s]}**2'+
                                                    '+{U_y [km/s]}**2'+
                                                    '+{U_z [km/s]}**2))'+
                          '*1.5696123057605e-4*{U_x [km/s]}'+
                          '+  {ExB_x [W/Re^2]}')
    eq('{K_y [W/Re^2]} = ({P [nPa]}*(1.666667/0.666667)*4.9430863e10'+
                                        '+4.2941e5*{Rho [amu/cm^3]}/2*'+
                                                    '({U_x [km/s]}**2'+
                                                    '+{U_y [km/s]}**2'+
                                                    '+{U_z [km/s]}**2))'+
                          '*1.5696123057605e-4*{U_y [km/s]}'+
                          '+  {ExB_y [W/Re^2]}')
    eq('{K_z [W/Re^2]} = ({P [nPa]}*(1.666667/0.666667)*4.9430863e10'+
                                        '+4.2941e5*{Rho [amu/cm^3]}/2*'+
                                                    '({U_x [km/s]}**2'+
                                                    '+{U_y [km/s]}**2'+
                                                    '+{U_z [km/s]}**2))'+
                          '*1.5696123057605e-4*{U_z [km/s]}'+
                          '+  {ExB_z [W/Re^2]}')

    #Component Normal Flux
    eq('{Kn_x [W/Re^2]} = ({K_x [W/Re^2]}*{surface_normal_x}'+
                            '+{K_y [W/Re^2]}*{surface_normal_y}'+
                            '+{K_z [W/Re^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_x}',
       zones=[zone_index])
    eq('{Kn_y [W/Re^2]} = ({K_x [W/Re^2]}*{surface_normal_x}'+
                            '+{K_y [W/Re^2]}*{surface_normal_y}'+
                            '+{K_z [W/Re^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_y}',
        zones=[zone_index])
    eq('{Kn_z [W/Re^2]} = ({K_x [W/Re^2]}*{surface_normal_x}'+
                            '+{K_y [W/Re^2]}*{surface_normal_y}'+
                            '+{K_z [W/Re^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2'+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)'+
                          '* {surface_normal_z}',
        zones=[zone_index])

    #Magnitude Normal Flux
    eq('{K_out [W/Re^2]} = ({Kn_x [W/Re^2]}*{surface_normal_x}'+
                            '+{Kn_y [W/Re^2]}*{surface_normal_y}'+
                            '+{Kn_z [W/Re^2]}*{surface_normal_z})'+
                          '/ sqrt({surface_normal_x}**2'+
                                  '+{surface_normal_y}**2 '+
                                  '+{surface_normal_z}**2'+
                                  '+1e-25)',
        zones=[zone_index])

    #Split into + and - flux
    eq('{K_out+} = max({K_out [W/Re^2]},0)', zones=[zone_index])
    eq('{K_out-} = min({K_out [W/Re^2]},0)', zones=[zone_index])


def integrate_surface(var_index, zone_index, qtname, idimension,
                      kdimension, *, is_cylinder=True, frame_id='main',
                      VariableOption='Scalar'):
    """Function to calculate integral of variable on a 3D exterior surface
    Inputs
        var_index- variable to be integrated
        zone_index- index of the zone to perform integration
        qtname- integrated quantity will be saved as this name
        idimension- used to determine which planes are outer surface
        kdimension- used to determine which planes are outer surface
        is_cylinder- default true
        VariableOption- default scalar, can choose others
        frame_id- frame name with the surface that integral is performed
    Output
        integrated_total
    """
    #Integrate total surface Flux
    frame=[fr for fr in tp.frames(frame_id)][0]
    page = tp.active_page()
    #validate name (special characters give tp a lot of problems
    qtname_abr = qtname.split('?')[0].split('[')[0].split('*')[0]+'*'
    if not is_cylinder:
        print("WARNING: not cylindrical object, check surface integrals")
    surface_planes=["IRange={MIN =1 MAX = 0 SKIP ="+str(idimension-1)+"} "+
                    "JRange={MIN =1 MAX = 0 SKIP =1} "+
                    "KRange={MIN =1 MAX = 0 SKIP ="+str(kdimension-1)+"}"]

    integrate_command_I=("Integrate [{:d}] ".format(zone_index+1)+
                         "VariableOption="+VariableOption+" ")
    if VariableOption == 'Scalar':
        integrate_command_I = (integrate_command_I+
                         "ScalarVar={:d} ".format(var_index+1))
    integrate_command_I = (integrate_command_I+
                         "XVariable=1 "+
                         "YVariable=2 "+
                         "ZVariable=3 "+
                         "ExcludeBlanked='T' "+
                         "IntegrateOver='IPlanes' "+
                         "IntegrateBy='Zones' "+
                         surface_planes[0]+
                         " PlotResults='F' "+
                         "PlotAs='"+qtname+"_I' "+
                         "TimeMin=0 TimeMax=0")

    integrate_command_K=("Integrate [{:d}] ".format(zone_index+1)+
                         "VariableOption="+VariableOption+" ")
    if VariableOption == 'Scalar':
        integrate_command_K = (integrate_command_K+
                         "ScalarVar={:d} ".format(var_index+1))
    integrate_command_K = (integrate_command_K+
                         "XVariable=1 "+
                         "YVariable=2 "+
                         "ZVariable=3 "+
                         "ExcludeBlanked='T' "+
                         "IntegrateOver='KPlanes' "+
                         "IntegrateBy='Zones' "+
                         surface_planes[0]+
                         " PlotResults='F' "+
                         "PlotAs='"+qtname+"_K' "+
                         "TimeMin=0 TimeMax=0")


    #integrate over I planes
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command_I)
    filename = os.getcwd()+'/Out.txt'
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
        command='SaveIntegrationResults FileName="'+filename+'"')
    result = pd.read_table('./Out.txt', sep=':',index_col=0)
    Ivalue = result.iloc[-1].values[0]
    #integrate over K planes
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command_K)
    filename = os.getcwd()+'/Out.txt'
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
        command='SaveIntegrationResults FileName="'+filename+'"')
    result = pd.read_table('./Out.txt', sep=':',index_col=0)
    Kvalue = result.iloc[-1].values[0]
    #sum all parts together
    integrated_total = Ivalue+Kvalue
    return integrated_total

def integrate_volume(var_index, zone_index, qtname, *, frame_id='main',
                     subspace=None, VariableOption='Scalar'):
    """Function to calculate integral of variable within a 3D volume
    Inputs
        var_index- variable to be integrated
        zone_index- index of the zone to perform integration
        qtname- integrated quantity will be saved as this name
        frame_id- frame name with the surface that integral is performed
        subspace- used to integrate only partial volumes, default None
        VariableOption- default scalar, can choose others
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
    if subspace == 'tail':
        #value blank domain around X>-2
        plt.value_blanking.active = True
        blank = plt.value_blanking.constraint(0)
        blank.active = True
        blank.variable = data.variable('X *')
        blank.comparison_operator = RelOp.GreaterThan
        blank.comparison_value = -3
    #Setup macrofunction command
    integrate_command=("Integrate [{:d}] ".format(zone_index+1)+
                       "VariableOption={} ".format(VariableOption))
    if VariableOption == 'Scalar':
        integrate_command = (integrate_command+
                       "ScalarVar={:d} ".format(var_index+1))
    integrate_command = (integrate_command+
                       "XVariable=1 "+
                       "YVariable=2 "+
                       "ZVariable=3 "+
                       "ExcludeBlanked='T' "+
                       "IntegrateOver='Cells' "+
                       "IntegrateBy='Zones' "+
                       "PlotResults='F' "+
                       "PlotAs='"+qtname+"' "+
                       "TimeMin=0 TimeMax=0")
    #Perform integration and extract resultant value
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
                                      command=integrate_command)
    filename = os.getcwd()+'/Out.txt'
    tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
        command='SaveIntegrationResults FileName="'+filename+'"')
    result = pd.read_table('./Out.txt', sep=':',index_col=0)
    integral = result.iloc[-1].values[0]

    #Delete created file and turn off blanking
    os.system('rm Out.txt')
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
