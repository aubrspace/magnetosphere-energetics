#!/usr/bin/env python3
"""Extraction routine for plasmasheet surface
"""
#import logging as log
import os
import sys
import time
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import pandas as pd
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
#custom packages
#from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import volume_tools
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract import tec_tools
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract.tec_tools import (streamfind_bisection,
                                                    dump_to_pandas,
                                                    abs_to_timestamp,
                                                    write_to_timelog)

def get_plasmasheet(field_data, datafile, *, outputpath='output/',
                     nstream=64, lat_max=89,
                     lon_limit=30, rday_max=30,rday_min=3.5,
                     itr_max=100, searchtol=0.01, tail_cap=-20,
                     nslice=20, nalpha=36, nfill=5,
                     integrate_surface=True, integrate_volume=True):
    """Function that finds, plots and calculates energetics on the
        plasmasheet surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        datafile- field data input, assumes .plt file
        outputpath- path for output files
        nstream- number of streamlines generated for algorithm
        lat_max- latitude limit of algorithm for field line seeding
        itr_max, searchtol- settings for bisection search algorithm
        tail_cap- X position of tail cap
        nslice, nalpha- cylindrical points used for surface reconstruction
    """
    #get date and time info from datafile name
    #time = get_time(datafile)
    #set unique outputname
    outputname = datafile.split('e')[1].split('-000.')[0]+'-cps'
    print(field_data)
    #set parameters
    lon_set = np.append(np.linspace(-lon_limit, -180, int(nstream/4)),
                          np.linspace(180, lon_limit, int(nstream/4)))
    with tp.session.suspend():
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
            if field_data.zone(zone).name.find('plasma_sheet_') != -1:
                stream_zone_list.append(field_data.zone(zone).index+1)
        stream_df, max_x = dump_to_pandas(main_frame, stream_zone_list,
                                          [1,2,3],
                                        outputpath+'cps_stream_points.csv')
        min_x = stream_df['X [R]'].min()
        max_x = stream_df['X [R]'].max()
        #slice and construct XYZ data
        print('max x: {:.2f}, set to x=-3\nmin x: {:.2f}'.format(max_x,
                                                                min_x))
        cps_mesh = surface_construct.ah_slicer(stream_df, min_x,
                                              -3, nslice, nalpha,
                                              False)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, min_x, -3,
                        'cps_zone')
        load_cylinder(field_data, cps_mesh, 'cps_zone',
                      nfill, nalpha, nslice)

        #interpolate field data to zone
        print('interpolating field data to plasmasheet')
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=field_data.zone('cps_zone'),
                source_zones=field_data.zone('global_field'))
        if integrate_surface:
            plasmasheet_powers = surface_analysis(field_data,'cps_zone',
                                             nfill, nslice)
            print('Plasmasheet Power Quantities')
            print(plasmasheet_powers)
        if integrate_volume:
            plasmasheet_energies = volume_analysis(field_data, 'cps_zone')
            print('Plasmasheet Energy Quantities')
            print(plasmasheet_energies)
        write_to_timelog(outputpath+'cps_integral_log.csv', time.UTC[0],
                         plasmasheet_powers.combine(plasmasheet_energies,
                                                    np.maximum,
                                                    fill_value=-1e12))

        #delete stream zones
        main_frame.activate()
        for zone in reversed(range(field_data.num_zones)):
            tp.active_frame().plot().fieldmap(zone).show=True
            if (field_data.zone(zone).name.find('cps_zone') == -1 and
                field_data.zone(zone).name.find('global_field') == -1 and
                field_data.zone(zone).name.find('mp_zone') == -1):
                field_data.delete_zones(field_data.zone(zone))


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
