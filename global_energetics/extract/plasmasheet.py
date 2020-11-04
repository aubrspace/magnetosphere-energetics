#!/usr/bin/env python3
"""Extraction routine for plasmasheet surface
"""
import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import pandas as pd
import spacepy
from spacepy import coordinates as coord
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
#custom packages
from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import stream_tools
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract.stream_tools import (calc_plasmasheet,
                                                    dump_to_pandas,
                                                    create_cylinder,
                                                    load_cylinder,
                                                    abs_to_timestamp,
                                                    write_to_timelog)

def get_plasmasheet(field_data, datafile, *, pltpath='./', laypath='./',
                     pngpath='./', nstream=50, lat_max=89,
                     lon_limit=90, rday_max=30,rday_min=3.5,
                     itr_max=100, searchtol=pi/90, tail_cap=-20,
                     nslice=40, nalpha=15, nfill=5):
    """Function that finds, plots and calculates energetics on the
        plasmasheet surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        datafile- field data input, assumes .plt file
        pltpath, laypath, pngpath- path for output of .plt,.lay,.png files
        nstream- number of streamlines generated for algorithm
        lat_max- latitude limit of algorithm for field line seeding
        itr_max, searchtol- settings for bisection search algorithm
        tail_cap- X position of tail cap
        nslice, nalpha- cylindrical points used for surface reconstruction
    """
    #get date and time info from datafile name
    time = get_time(datafile)

    #set unique outputname
    outputname = datafile.split('e')[1].split('-000.')[0]+'-cps'
    print(field_data)


    #set parameters
    lon_set = np.append(np.linspace(-lon_limit, -180, int(nstream/4)),
                          np.linspace(180, lon_limit, int(nstream/4)))
    with tp.session.suspend():
        main_frame = tp.active_frame()
        main_frame.name = 'main'
        if field_data.variable_names.count('r [R]') ==0:
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

        #Create plasmasheet field lines
        print('\nfinding north hemisphere boundary')
        calc_plasmasheet(field_data, lat_max, lon_set,
                         3/2*tail_cap, itr_max, searchtol, time)
        print('\nfinding south hemisphere boundary')
        calc_plasmasheet(field_data, -lat_max, lon_set,
                         3/2*tail_cap, itr_max, searchtol, time)
        #port stream data to pandas DataFrame object
        stream_zone_list = []
        for zone in range(field_data.num_zones):
            if field_data.zone(zone).name.find('plasmasheet') != -1:
                stream_zone_list.append(zone.index)
        print(stream_zone_list)
        stream_zone_list = np.linspace(3,field_data.num_zones,
                                       field_data.num_zones-3+1)
        stream_df, max_x = dump_to_pandas(main_frame, stream_zone_list,
                                          [1,2,3], 'cps_stream_points.csv')
        xstd = stream_df.std()['X [R]']
        xmean = stream_df.mean()['X [R]']
        min_x = xmean-3.5*xstd
        print(stream_df)
        print(min_x)
        #slice and construct XYZ data
        print('max x: {:.2f}, set to x=-5\nmin x: {:.2f}'.format(max_x,
                                                                min_x))
        cps_mesh = surface_construct.yz_slicer(stream_df, min_x,
                                               -5, nslice, nalpha,
                                               True)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, min_x+5, -1,
                        'cps_zone')
        load_cylinder(field_data, cps_mesh, 'cps_zone',
                      nfill, nalpha, nslice)

        #interpolate field data to zone
        print('interpolating field data to plasmasheet')
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=field_data.zone('cps_zone'),
                source_zones=field_data.zone('global_field'))
        plasmasheet_power = surface_analysis(field_data,'cps_zone',
                                             nfill, nslice)
        print(plasmasheet_power)
        write_to_timelog('cps_integral_log.csv', time.UTC[0],
                         plasmasheet_power)

        #delete stream zones
        main_frame.activate()
        '''
        for zone in reversed(range(field_data.num_zones)):
            tp.active_frame().plot().fieldmap(zone).show=True
            if (field_data.zone(zone).name.find('cps_zone') == -1 and
                field_data.zone(zone).name.find('global_field') == -1 and
                field_data.zone(zone).name.find('mp_zone') == -1):
                field_data.delete_zones(field_data.zone(zone))
        '''


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
