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
from progress.bar import Bar
#interpackage modules
from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import volume_tools
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (streamfind_bisection,
                                                    dump_to_pandas,
                                                    create_cylinder,
                                                    load_cylinder,
                                                    abs_to_timestamp,
                                                    write_to_timelog)
from global_energetics.extract import shue
from global_energetics.extract.shue import (r_shue, r0_alpha_1997,
                                                    r0_alpha_1998)


def get_shue_mesh(field_data, year, nx, nphi, xtail,
                  *, x_subsolar=None, dx=5):
    """Function mesh of 3D volume points based on Shue 1997/8 model for
        magnetopause
    Inputs
        field_data
        year- 1997 or 1998 for which emperical model
        nx, nphi- 3D volume grid dimensions
        xtail- limit for how far to extend in negative x direction
        x_subsolar- default None, will calculate with dayside fieldlines
    Outputs
        mesh- pandas DataFrame with X,Y,Z locations of outer surface
        x_subsolar
    """
    if x_subsolar == None:
        x_subsolar = 0
        #Call get streamfind with limited settings to get x_subsolar
        frontzoneindicies = streamfind_bisection(field_data, 'dayside', 10,
                                                 5, 30, 3.5, 100, 0.1)
        #Find the max value from set of zones
        for index in frontzoneindicies:
            x_subsolar = max(x_subsolar,
                                field_data.zone(index).values('X *').max())
        print('x_subsolar found at {}'.format(x_subsolar))
    #Probe field data at x_subsolar + dx to find Bz and Pdyn
    Bz = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][9]
    rho = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][3]
    ux = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][4]
    uy = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][5]
    uz = tp.data.query.probe_at_position(x_subsolar+dx,0,0)[0][6]
    Pdyn = 1/2*rho*(ux**2+uy**2+uz**2)*1.67e-5
    #Get r0 and alpha based on IMF conditions
    if year == 1997:
        r0, alpha = r0_alpha_1997(Bz, Pdyn)
    else:
        r0, alpha = r0_alpha_1998(Bz, Pdyn)
    #Calculate the 2D r, theta curve
    thetalist = linspace(150, 0, 10000)
    h_curve = []
    x_curve = []
    for theta in thetalist:
        r = r_shue(r0, alpha, theta)
        h_curve.append(r*sin(deg2rad(theta)))
        x_curve.append(r*cos(deg2rad(theta)))
    plt.plot(x_curve,h_curve)
    plt.show()
    #Set volume grid points
    xlist = linspace(xtail, x_curve[-1], nx)
    hlist = []
    for x in xlist:
        hlist.append(np.interp(x, x_curve, h_curve))
    philist = linspace(-180, 180, nphi)
    #Fill in volume based on revolved 2D curve of points
    xyz = ['X [R]', 'Y [R]', 'Z [R]']
    mesh = pd.DataFrame(columns=xyz)
    for point in enumerate(xlist):
        x = point[1]
        h = hlist[point[0]]
        for phi in philist:
            y = h*cos(deg2rad(phi))
            z = h*sin(deg2rad(phi))
            mesh = mesh.append(pd.DataFrame([[x,y,z]],columns=xyz))
    return mesh, x_subsolar

def make_test_case():
    """Test case creates artifical set of flow and field streamlines for
    reliable testing
    Inputs
        None
    Outputs
        field_df, flow_df- data frames with 3D field data points
        x_subsolar- max xlocation from field_df
    """
    #define height at a given x location for flow and field
    def h_field(x):
        if x>=0:
            return 3*(10-x)**0.5
        else:
            return min((-0.01*(x+5)**3+3*sqrt(10))+1*sin(5*x/(2*pi)), 100)
    def h_flow(x):
        if x>=0:
            return 3*(12-x)**0.5
        else:
            return 3*sqrt(12)+1*sin(5*x/(2*pi))
    #define point depending on x and angle
    def p_field(x,a):return [x,h_field(x)*cos(a),h_field(x)*sin(a)]
    def p_flow(x,a): return [x, h_flow(x)*cos(a), h_flow(x)*sin(a)]
    xlist, dx = linspace(10, -60, 200, retstep=True)
    alist, da = linspace(-pi, pi, 142, retstep=True)
    xyz = ['X [R]', 'Y [R]', 'Z [R]']
    field_df = pd.DataFrame(columns=xyz)
    flow_df = pd.DataFrame(columns=xyz)
    #fill in a dataframe with many points
    bar = Bar('creating artificial field and flow points', max=len(xlist))
    for x in xlist:
        for a in alist:
            field_df = field_df.append(pd.DataFrame([p_field(x,a)],
                                                    columns=xyz),
                                       ignore_index=True)
            flow_df = flow_df.append(pd.DataFrame([p_flow(x,a)],
                                                  columns=xyz),
                                     ignore_index=True)
        bar.next()
    bar.finish()
    x_subsolar = field_df['X [R]'].max()
    '''
    #plot points using python
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(field_df['X [R]'], field_df['Y [R]'], field_df['Z [R]'],
               c='blue', label='synthetic fieldpoints')
    ax.scatter(flow_df['X [R]'], flow_df['Y [R]'], flow_df['Z [R]'],
               c='orange', label='synthetic flowpoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-40, 12])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])
    ax.legend()
    fig.savefig('synthetic.png')
    plt.show()
    '''
    return field_df, flow_df, x_subsolar

def inner_volume_df(df1, df2, upperbound, lowerbound, innerbound,
                    dim1, dim2, *, form='xcylinder',xkey='X [R]',
                                   quiet=True):
    """Function combines two dataframe sets of points representing volumes
        and keeping the interior points only based on form given
    Inputs
        df1, df2- pandas dataframe objects
        upperbound, lowerbound, innerbound- limits of search criteria
        dim1, dim2- dimensionality of search criteria of discrete vol elems
        form- default cylinder with axis on centerline
        xkey- string ID for x coordinate, y and z are assumed
        quiet- boolean for displaying points missing in 1 or more sets
    Returns
        df_combined
    """
    #get x, y, z variables
    ykey = xkey.split('X')[0]+'Y'+xkey.split('X')[-1]
    zkey = xkey.split('X')[0]+'Z'+xkey.split('X')[-1]
    xyz = [xkey, ykey, zkey]
    #establish volume elements for search according to form
    if form == 'xcylinder':
        #process dataframes according to upper,lower inner bounds
        df1 = df1[(df1[xkey]< upperbound) & (df1[xkey]>lowerbound) &
                  (df1[xkey]**2+df1[ykey]**2+df1[zkey]**2>innerbound**2)]
        df2 = df2[(df2[xkey]< upperbound) & (df2[xkey]>lowerbound) &
                  (df2[xkey]**2+df2[ykey]**2+df2[zkey]**2>innerbound**2)]
        #cylinder with axis on X axis, dim1=x slices, dim2=azimuth
        xmax = max(df1[xkey].max(), df2[xkey].max())
        xmin = min(df1[xkey].min(), df2[xkey].min())
        dim1list, dx1 = np.linspace(xmin, xmax, dim1, retstep=True)
        dim2list, dx2 = np.linspace(-pi, pi, dim2, retstep=True)
        #get height parameter
        def height(y,z): return np.sqrt(y**2+z**2)
        h1 = pd.DataFrame(height(df1[ykey],df1[zkey]), columns=['h'])
        h2 = pd.DataFrame(height(df2[ykey],df2[zkey]), columns=['h'])
        df1 = df1.combine(h1, np.minimum, fill_value=1000)
        df2 = df2.combine(h2, np.minimum, fill_value=1000)
        hkey = 'h'
        #remove points outside of hmax of the lower hmax
        hmax = min(h1['h'].max(), h2['h'].max())
        df1 = df1[(df1[hkey]< hmax)]
        df2 = df2[(df2[hkey]< hmax)]
        #set dim1key to x
        dim1key = xkey
        #get azimuth angle parameter
        def angle(y,z): return np.arctan2(z,y)
        a1 = pd.DataFrame(angle(df1[ykey],df1[zkey]), columns=['yz[rad]'])
        a2 = pd.DataFrame(angle(df2[ykey],df2[zkey]), columns=['yz[rad]'])
        df1 = df1.combine(a1, np.minimum, fill_value=1000)
        df2 = df2.combine(a2, np.minimum, fill_value=1000)
        dim2key = 'yz[rad]'
        #create placepoint function based on x1,x2,h general coordinates
        def placepoint(x1,x2,h):
            x = x1
            y = h*cos(x2)
            z = h*sin(x2)
            return x, y, z
    else:
        print('WARNING: form for combination of dataframes not recognized'+
              ' combining full set of points from each dataframe')
        df_combined = df1.append(df2)
        return df_combined.sort_values(by=[xkey])
    #loop through discretized volumes
    bar = Bar('combining dataframes:', max=len(dim1list)*len(dim2list))
    missing_points_list = []
    df_combined = pd.DataFrame(columns=xyz)
    df_flow = pd.DataFrame(columns=xyz)
    df_field = pd.DataFrame(columns=xyz)
    for x1 in dim1list:
        for x2 in dim2list:
            #get points within volume element
            tempdf1 = df1[(df1[dim1key]>x1-dx1/2)&(df1[dim1key]<x1+dx1/2) &
                          (df1[dim2key]>x2-dx2/2)&(df1[dim2key]<x2+dx2/2)]
            tempdf2 = df2[(df2[dim1key]>x1-dx1/2)&(df2[dim1key]<x1+dx1/2) &
                          (df2[dim2key]>x2-dx2/2)&(df2[dim2key]<x2+dx2/2)]
            #append a point at x1,x2 and h based on averages of df's
            if (not tempdf1.empty) | (not tempdf2.empty):
                hmax = min(tempdf1[hkey].max(), tempdf2[hkey].max())
                df_combined=df_combined.append(tempdf1[tempdf1[hkey]<hmax])
                df_combined=df_combined.append(tempdf2[tempdf2[hkey]<hmax])
            else:
                #assume location of point and record in missing list
                missing_points_list.append([x1,rad2deg(x2),
                                            tempdf1[hkey].mean(),
                                            tempdf2[hkey].mean()])
            bar.next()
    bar.finish()
    if not quiet and (len(missing_points_list)!=0):
        print('WARNING: Following points missing in both data sets')
        print('X    Theta(deg)      flow_h      field_h'+
            '\n--   ----------      ------      -------')
        for point in missing_points_list:
            print('{:.2f}  {:.0f}  {:.1f}  {:.1f}'.format(point[0],
                                                                  point[1],
                                                                  point[2],
                                                                  point[3]))
    return df_combined

def get_magnetopause(field_data, datafile, *, outputpath='output/',
                     mode='hybrid',
                     n_day=72, lon_max=90, rday_max=30, rday_min=3.5,
                     n_tail=72, rtail_max=125, rtail_min=0.5,
                     n_flow=72, flow_seed_x=-5, rflow_max=20, rflow_min=0,
                     shue=None,
                     itr_max=100, tol=0.1,
                     tail_cap=-40, innerbound=2,
                     nslice=60, nalpha=36, nfill=10,
                     integrate_surface=True, integrate_volume=True,
                     xyzvar=[1,2,3], zone_rename=None):
    """Function that finds, plots and calculates energetics on the
        magnetopause surface.
    Inputs
        General
            field_data- tecplot DataSet object with 3D field data
            datafile- field data filename, assumes .plt file
            outputpath- path for output of .csv of points
            mode- hybrid, fieldline, flowline, test or shue
        Dayside field lines
            n_day- number of streamlines generated for dayside algorithm
            lon_max- longitude limit of dayside algorithm for streams
            rday_max, rday_min- radial limits (in XY) for dayside algorithm
        Tail field lines
            n_tail- number of streamlines generated for tail algorithm
            rtail_max, rtail_min- tail disc min and max search radius
        Flow lines
            n_flow- number of flowlines generated
            flow_seed_x- x location to seed flowlines from
            rflow_min, rflow_max- boundaries for search criteria
        Shue
            shue- None, 1997, 1998 uses Shue empirical, read iff mode=shue
        Surface
            itr_max, tol- settings for bisection search algorithm
            tail_cap, innerbound- X position of tail cap and inner cuttoff
            nslice, nalpha, nfill- cyl points for surface reconstruction
            integrate_surface/volume- booleans for settings
            xyzvar- for X, Y, Z variables in field data variable list
            zone_rename- optional rename if calling multiple times
    """
    display = ('Analyzing Magnetopause with the following settings:\n'+
               '\tdatafile: {}\n'.format(datafile)+
               '\toutputpath: {}\n'.format(outputpath)+
               '\tmode: {}\n'.format(mode))
    if (mode == 'hybrid') or (mode == 'fieldline'):
        #field line settings
        display = (display+
               '\tn_day: {}\n'.format(n_day)+
               '\tlon_max: {}\n'.format(lon_max)+
               '\trday_max: {}\n'.format(rday_max)+
               '\tn_tail: {}\n'.format(n_tail)+
               '\trtail_max: {}\n'.format(rtail_max)+
               '\trtail_min: {}\n'.format(rtail_min))
    if (mode == 'hybrid') or (mode == 'flowline'):
        #flow line settings
        display = (display+
               '\tn_flow: {}\n'.format(n_flow)+
               '\tflow_seed_x: {}\n'.format(flow_seed_x)+
               '\trflow_max: {}\n'.format(rflow_max)+
               '\trflow_min: {}\n'.format(rflow_min))
    if mode == 'shue':
        #shue empirical settings
        display = (display+
               '\tshue: {}\n'.format(shue))
    #general surface settings
    display = (display+
               '\titr_max: {}\n'.format(itr_max)+
               '\ttol: {}\n'.format(tol)+
               '\ttail_cap: {}\n'.format(tail_cap)+
               '\tinnerbound: {}\n'.format(innerbound)+
               '\tnslice: {}\n'.format(nslice)+
               '\tnalpha: {}\n'.format(nalpha)+
               '\tnfill: {}\n'.format(nfill)+
               '\tintegrate_surface: {}\n'.format(integrate_surface)+
               '\tintegrate_volume: {}\n'.format(integrate_volume)+
               '\txyzvar: {}\n'.format(xyzvar)+
               '\tzone_rename: {}\n'.format(zone_rename))
    print(display)
    print(field_data)
    #get date and time info from datafile name
    time = get_time(datafile)

    #make unique outputname based on datafile string
    outputname = datafile.split('e')[1].split('-000.')[0]+'-mp'

    with tp.session.suspend():
        #get r, lon, lat if not already set
        if field_data.variable_names.count('r [R]') ==0:
            main_frame = tp.active_frame()
            main_frame.name = 'main'
            tp.data.operate.execute_equation(
                    '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
            tp.data.operate.execute_equation(
                    '{lat [deg]} = 180/pi*asin({Z [R]} / {r [R]})')
            tp.data.operate.execute_equation(
                    '{lon [deg]} = if({X [R]}>0,'+
                                     '180/pi*atan({Y [R]} / {X [R]}),'+
                                  'if({Y [R]}>0,'+
                                     '180/pi*atan({Y [R]}/{X [R]})+180,'+
                                     '180/pi*atan({Y [R]}/{X [R]})-180))')
        else:
            main_frame = [fr for fr in tp.frames('main')][0]
        #Get mesh points depending on mode setting
        ################################################################
        if mode == 'shue':
            mp_mesh, x_subsolar = get_shue_mesh(field_data, shue, nslice,
                                                nalpha, -40)
            zonename = 'mp_shue'+str(shue)
        ################################################################
        elif mode == 'test':
            field_df, flow_df, x_subsolar = make_test_case()
            zonename = 'mp_test'
        ################################################################
        elif (mode == 'hybrid') or (mode == 'flowline'):
            ###flowline points
            flowlist = streamfind_bisection(field_data, 'flow',flow_seed_x,
                                            n_flow, rflow_max, rflow_min,
                                            itr_max, tol,
                                            field_key_x='U_x*')
            flow_df, _ = dump_to_pandas(main_frame, flowlist, xyzvar,
                                        path_to_mesh+meshfile)
            for zone_index in reversed(flowlist):
                field_data.delete_zones(field_data.zone(zone_index))
            zonename = 'mp_flowline'
        ################################################################
        if (mode == 'hybrid') or (mode == 'fieldline'):
            ###tail points
            taillist = streamfind_bisection(field_data, 'tail', -20,
                                            n_tail,
                                            rtail_max, rtail_min,
                                            itr_max, tol)
            tail_df, _ = dump_to_pandas(main_frame, taillist,
                                xyzvar, path_to_mesh+meshfile)
            for zone_index in reversed(taillist):
                field_data.delete_zones(field_data.zone(zone_index))
            ###dayside points
            daysidelist = streamfind_bisection(field_data, 'dayside',
                                               lon_max, n_day,
                                               rday_max, rday_min,
                                               itr_max, tol)
            dayside_df, x_subsolar = dump_to_pandas(main_frame, daysidelist,
                            xyzvar, path_to_mesh+meshfile)
            for zone_index in reversed(daysidelist):
                field_data.delete_zones(field_data.zone(zone_index))
            ###combine dayside and tail into one set
            field_df = dayside_df.append(tail_df)
            zonename = 'mp_fieldline'
        ################################################################
        #Combine data sets if using hybrid mode
        if mode == 'hybrid':
            stream_df = inner_volume_df(flow_df, field_df, x_subsolar,
                                        tail_cap, innerbound,nslice,nalpha,
                                        quiet=False)
            #slice and construct XYZ data
            mp_mesh = surface_construct.ah_slicer(stream_df, -40,
                                              x_subsolar, nslice, nalpha,
                                              False)
            zonename = 'mp_hybrid'
        ################################################################
        if zone_rename != None:
            zonename = zone_rename
        '''
        #Quickrun
        flow_df = pd.read_csv('output/mp_flow_points.csv')
        flow_df = flow_df.drop(columns=['Unnamed: 3'])
        flow_df = flow_df.sort_values(by=['X [R]'])
        flow_df = flow_df.reset_index(drop=True)
        dayside_df = pd.read_csv('output/mp_dayside_points.csv')
        dayside_df = dayside_df.drop(columns=['Unnamed: 3'])
        dayside_df = dayside_df.sort_values(by=['X [R]'])
        dayside_df = dayside_df.reset_index(drop=True)
        x_subsolar = dayside_df['X [R]'].max()
        tail_df = pd.read_csv('output/mp_tail_points.csv')
        tail_df = tail_df.drop(columns=['Unnamed: 3'])
        tail_df = tail_df.sort_values(by=['X [R]'])
        tail_df = tail_df.reset_index(drop=True)
        field_df = dayside_df.append(tail_df)
        '''
        #save mesh to file
        path_to_mesh = outputpath+'/'+mode+'/'
        os.system('mkdir '+path_to_mesh)
        meshfile = datafile.split('.')[0]+'_'+mode+'.csv'
        mesh.to_csv(path_to_mesh+meshfile)

        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, tail_cap,
                        x_subsolar, zonename)
        load_cylinder(field_data, mp_mesh, zonename, I=nfill, K=nslice,
                      J=nalpha)

        #interpolate field data to zone
        print('interpolating field data to magnetopause')
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=field_data.zone(zonename),
                source_zones=field_data.zone('global_field'))

        #perform integration for surface and volume quantities
        magnetopause_powers = pd.DataFrame([[0,0,0]],
                                      columns=['no_mp_surf1',
                                               'no_mp_surf2',
                                               'no_mp_surf3'])
        mp_magnetic_energy = pd.DataFrame([[0]],
                                      columns=['mp_vol_not_integrated'])

        if integrate_surface:
            magnetopause_powers = surface_analysis(field_data, zonename,
                                                  nfill, nslice)
            print('Magnetopause Power Terms')
            print(magnetopause_powers)
        if integrate_volume:
            mp_energies = volume_analysis(field_data, zonename)
            print('Magnetopause Energy Terms')
            print(mp_energies)
        if integrate_surface or integrate_power:
            write_to_timelog(outputpath+'mp_integral_log.csv', time.UTC[0],
                             magnetopause_powers.combine(mp_energies,
                                                         np.maximum,
                                                         fill_value=-1e12))



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
