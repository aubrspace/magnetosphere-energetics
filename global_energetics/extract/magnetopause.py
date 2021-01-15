#!/usr/bin/env python3
"""Extraction routine for magnetopause surface
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, linspace
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
                     nstream_day=36, lon_max=122,
                     rday_max=30,rday_min=3.5, dayitr_max=100, daytol=0.1,
                     nstream_tail=72, rho_max=125,rho_min=0.5,tail_cap=-20,
                     nslice=60, nalpha=36, nfill=10,
                     integrate_surface=True, integrate_volume=True,
                     varlist=[1,2,3], use_test=False):
    """Function that finds, plots and calculates energetics on the
        magnetopause surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        datafile- field data filename, assumes .plt file
        outputpath- path for output of .csv of points
        nstream_day- number of streamlines generated for dayside algorithm
        lon_max- longitude limit of dayside algorithm for streams
        rday_max, rday_min- radial limits (in XY) for dayside algorithm
        dayitr_max, daytol- settings for bisection search algorithm
        nstream_tail- number of streamlines generated for tail algorithm
        rho_max, rho_step- tail disc maximium radius and step (in YZ)
        tail_cap- X position of tail cap
        nslice, nalpha, nfill- cylindrical point for surface reconstruction
        integrate_surface/volume- booleans for settings
        varlist- for X, Y, Z variables in field data variable list
        use_test- boolean overrides and uses a test case of stream data
    """
    print('Analyzing Magnetopause with the following settings:\n'+
            '\tdatafile: {}\n'.format(datafile)+
            '\toutputpath: {}\n'.format(outputpath)+
            '\tnstream_day: {}\n'.format(nstream_day)+
            '\tlon_max: {}\n'.format(lon_max)+
            '\trday_max: {}\n'.format(rday_max)+
            '\tdayitr_max: {}\n'.format(dayitr_max)+
            '\tnstream_tail: {}\n'.format(nstream_tail)+
            '\trho_max: {}\n'.format(rho_max)+
            '\ttail_cap: {}\n'.format(tail_cap)+
            '\tnslice: {}\n'.format(nslice)+
            '\tnalpha: {}\n'.format(nalpha)+
            '\tnfill: {}\n'.format(nfill)+
            '\tintegrate_surface: {}\n'.format(integrate_surface)+
            '\tintegrate_volume: {}\n'.format(integrate_volume)+
            '\tvarlist: {}\n'.format(varlist)+
            '\tuse_test: {}\n'.format(use_test))
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
        if use_test:
            field_df, flow_df, x_subsolar = make_test_case()
        else:
            ###flowline points
            flowlist = streamfind_bisection(field_data, 'flow', -5, 72,
                                            20, 0,
                                            dayitr_max, daytol,
                                            field_key_x='U_x*')
            flow_df, _ = dump_to_pandas(main_frame, flowlist, varlist,
                                        outputpath+'mp_flow_points.csv')
            for zone_index in reversed(flowlist):
                field_data.delete_zones(field_data.zone(zone_index))
            ###tail points
            taillist = streamfind_bisection(field_data, 'tail', -20,
                                            nstream_tail,
                                            rho_max, rho_min,
                                            dayitr_max, daytol)
            tail_df, _ = dump_to_pandas(main_frame, taillist,
                                varlist, outputpath+'mp_tail_points.csv')
            for zone_index in reversed(taillist):
                field_data.delete_zones(field_data.zone(zone_index))
            ###dayside points
            daysidelist = streamfind_bisection(field_data, 'dayside', lon_max,
                                            nstream_day,
                                            rday_max, rday_min,
                                            dayitr_max, daytol)
            dayside_df, x_subsolar = dump_to_pandas(main_frame, daysidelist,
                            varlist, outputpath+'mp_dayside_points.csv')
            for zone_index in reversed(daysidelist):
                field_data.delete_zones(field_data.zone(zone_index))
            ###combine dayside and tail into one set
            field_df = dayside_df.append(tail_df)
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

        stream_df = inner_volume_df(flow_df, field_df, x_subsolar,
                                        -40, 2, 40, 36, quiet=False)
        #slice and construct XYZ data
        mp_mesh = surface_construct.ah_slicer(stream_df, -40,
                                              x_subsolar, nslice, nalpha,
                                              False)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, -40,
                        x_subsolar, 'mp_zone')
        load_cylinder(field_data, mp_mesh, 'mp_zone', I=nfill, J=nslice,
                      K=nalpha)
        main_frame.activate()

        #interpolate field data to zone
        print('interpolating field data to magnetopause')
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=field_data.zone('mp_zone'),
                source_zones=field_data.zone('global_field'))

        #perform integration for surface and volume quantities
        magnetopause_powers = pd.DataFrame([[0,0,0]],
                                      columns=['no_mp_surf1',
                                               'no_mp_surf2',
                                               'no_mp_surf3'])
        mp_magnetic_energy = pd.DataFrame([[0]],
                                      columns=['mp_vol_not_integrated'])

        if integrate_surface:
            magnetopause_powers = surface_analysis(field_data, 'mp_zone',
                                                  nfill, nslice)
            print('Magnetopause Power Terms')
            print(magnetopause_powers)
        if integrate_volume:
            mp_energies = volume_analysis(field_data, 'mp_zone')
            print('Magnetopause Energy Terms')
            print(mp_energies)
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
