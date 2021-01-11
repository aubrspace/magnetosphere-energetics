#!/usr/bin/env python3
"""Extraction routine for magnetopause surface
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
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

def inner_volume_df(df1, df2, dim1, dim2, *,form='xcylinder',xkey='X [R]'):
    """Function combines two dataframe sets of points representing volumes
        and keeping the interior points only based on form given
    Inputs
        df1, df2- pandas dataframe objects
        dim1, dim2- dimensionality of search criteria of discrete vol elems
        form- default cylinder with axis on centerline
        xkey- string ID for x coordinate, y and z are assumed
    Returns
        df_combined
    """
    #get x, y, z variables
    ykey = xkey.split('X')[0]+'Y'+xkey.split('X')[-1]
    zkey = xkey.split('X')[0]+'Z'+xkey.split('X')[-1]
    #establish volume elements for search according to form
    if form == 'xcylinder':
        #cylinder with axis on X axis, dim1=x slices, dim2=azimuth
        xmax = max(df1[xkey].max(), df2[xkey].max())
        xmin = min(df1[xkey].min(), df2[xkey].min())
        dim1list = np.linspace(xmax, xmin, dim1, endpoint=False)
        dim2list = np.linspace(-pi, pi, dim2, endpoint=False)
        #get height parameter
        h1 = pd.DataFrame(np.sqrt(df1[zkey]**2+df1[ykey]**2), columns=['h'])
        h2 = pd.DataFrame(np.sqrt(df2[zkey]**2+df2[ykey]**2), columns=['h'])
        df1 = df1.combine(h1, np.minimum, fill_value=1000)
        df2 = df2.combine(h2, np.minimum, fill_value=1000)
        hkey = 'h'
        #set dim1key to x
        dim1key = xkey
        #get azimuth angle parameter
        a1 = pd.DataFrame(np.arctan2(df1[zkey], df1[ykey]),
                          columns=['yz[rad]'])
        a2 = pd.DataFrame(np.arctan2(df2[zkey], df2[ykey]),
                          columns=['yz[rad]'])
        df1 = df1.combine(a1, np.minimum, fill_value=1000)
        df2 = df2.combine(a2, np.minimum, fill_value=1000)
        dim2key = 'yz[rad]'
    else:
        print('WARNING: form for combination of dataframes not recognized'+
              ' combining full set of points from each dataframe')
        df_combined = df1.append(df2)
        return df_combined.sort_values(by=[xkey])
    #loop through discretized volumes
    dx1 = abs(dim1list[1]-dim1list[0])
    dx2 = abs(dim2list[1]-dim2list[0])
    bar = Bar('combining dataframes:', max=len(dim1list)*len(dim2list))
    for x1 in dim1list:
        for x2 in dim2list:
            #get points within volume element
            tempdf1 = df1[(df1[dim1key]>x1) & (df1[dim1key]<x1+dx1) &
                          (df1[dim2key]>x2) & (df1[dim2key]<x2+dx2)]
            tempdf2 = df2[(df2[dim1key]>x1) & (df2[dim1key]<x1+dx1) &
                          (df2[dim2key]>x2) & (df2[dim2key]<x2+dx2)]
            #get average of included points for each df
            if tempdf1[hkey].mean() > tempdf2[hkey].mean():
                #drop df1 points based on indicies
                df1 = df1.drop(tempdf1.index)
            else:
                #drop df2 points based on indicies
                df2 = df2.drop(tempdf2.index)
            bar.next()
    df_combined = df1.append(df2)
    bar.finish()
    return df_combined.sort_values(by=[xkey])

def get_magnetopause(field_data, datafile, *, outputpath='output/',
                     nstream_day=36, lon_max=122,
                     rday_max=30,rday_min=3.5, dayitr_max=100, daytol=0.1,
                     nstream_tail=36, rho_max=125,rho_min=0.5,tail_cap=-20,
                     nslice=40, nalpha=36, nfill=10,
                     integrate_surface=True, integrate_volume=True,
                     varlist=[1,2,3]):
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
        nslice, nalpha- cylindrical points used for surface reconstruction
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
            '\tnalpha: {}\n'.format(nalpha))
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
        ###tail points
        taillist = streamfind_bisection(field_data, 'tail', -20,
                                        nstream_tail,
                                        rho_max, rho_min,
                                        dayitr_max, daytol)
        tail_df, _ = dump_to_pandas(main_frame, taillist,
                            varlist, outputpath+'mp_dayside_points.csv')
        #for zone_index in reversed(taillist):
        #    field_data.delete_zones(field_data.zone(zone_index))
        ###dayside points
        daysidelist = streamfind_bisection(field_data, 'dayside', 90,
                                        36,
                                        rday_max, rday_min,
                                        dayitr_max, daytol)
        dayside_df, x_subsolar = dump_to_pandas(main_frame, daysidelist,
                            varlist, outputpath+'mp_dayside_points.csv')
        #for zone_index in reversed(daysidelist):
        #    field_data.delete_zones(field_data.zone(zone_index))
        ###flowline points
        flowlist = streamfind_bisection(field_data, 'flow', -5, 72, 20, 0,
                                        dayitr_max, daytol,
                                        field_key_x='U_x*')
        flow_df, _ = dump_to_pandas(main_frame, flowlist, varlist,
                                    outputpath+'mp_flow_points.csv')
        #for zone_index in reversed(flowlist):
        #    field_data.delete_zones(field_data.zone(zone_index))
        ###combine portions into a single dataframe
        stream_df = inner_volume_df(flow_df, dayside_df.append(tail_df),
                                    25, 18)
        '''
        stream_df = flow_df[(flow_df['X [R]'] < -8)].append(
                    dayside_df[(dayside_df['X [R]'] > -10)])
        '''
        #slice and construct XYZ data
        mp_mesh = surface_construct.ah_slicer(stream_df, tail_cap,
                                              x_subsolar, nslice, nalpha,
                                              False)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, tail_cap,
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
        magnetopause_power = pd.DataFrame([[0,0,0]],
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
