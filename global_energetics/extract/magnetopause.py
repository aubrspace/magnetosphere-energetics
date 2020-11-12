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
#interpackage modules
from global_energetics.makevideo import get_time
from global_energetics.extract import surface_construct
#from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract import surface_tools
from global_energetics.extract.surface_tools import surface_analysis
from global_energetics.extract import volume_tools
from global_energetics.extract.volume_tools import volume_analysis
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (calc_dayside_mp,
                                                    calc_tail_mp,
                                                    dump_to_pandas,
                                                    create_cylinder,
                                                    load_cylinder,
                                                    abs_to_timestamp,
                                                    write_to_timelog)

def get_magnetopause(field_data, datafile, *, pltpath='./', laypath='./',
                     pngpath='./', nstream_day=36, lon_max=122,
                     rday_max=30,rday_min=3.5, dayitr_max=100, daytol=0.1,
                     nstream_tail=36, rho_max=50,rho_min=0.5,tail_cap=-20,
                     nslice=40, nalpha=36, nfill=2,
                     integrate_surface=True, integrate_volume=True):
    """Function that finds, plots and calculates energetics on the
        magnetopause surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        datafile- field data filename, assumes .plt file
        pltpath, laypath, pngpath- path for output of .plt,.lay,.png files
        nstream_day- number of streamlines generated for dayside algorithm
        lon_max- longitude limit of dayside algorithm for streams
        rday_max, rday_min- radial limits (in XY) for dayside algorithm
        dayitr_max, daytol- settings for bisection search algorithm
        nstream_tail- number of streamlines generated for tail algorithm
        rho_max, rho_step- tail disc maximium radius and step (in YZ)
        tail_cap- X position of tail cap
        nslice, nalpha- cylindrical points used for surface reconstruction
    """
    #get date and time info from datafile name
    time = get_time(datafile)

    #make unique outputname based on datafile string
    outputname = datafile.split('e')[1].split('-000.')[0]+'-mp'
    print(field_data)

    #set parameters
    lon_set = np.linspace(-1*lon_max, lon_max, nstream_day)
    psi = np.linspace(-180*(1-1/nstream_tail), 180, nstream_tail)
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
        #Create Dayside Magnetopause field lines
        calc_dayside_mp(field_data, lon_set, rday_max, rday_min, dayitr_max,
                        daytol)
        #Create Tail magnetopause field lines
        tail_cap_mod = calc_tail_mp(field_data, psi, tail_cap, rho_max,
                                    rho_min, dayitr_max, daytol)
        #go into loop modifiying tail cap placement if no mp points found
        if tail_cap_mod != tail_cap:
            temp_tail = 0
            while temp_tail != tail_cap_mod:
                print('\nSetting tail cap to {}'.format(tail_cap_mod))
                temp_tail = tail_cap_mod
                tail_cap_mod = calc_tail_mp(field_data, psi, tail_cap_mod,
                                            rho_max, rho_min, dayitr_max,
                                            daytol)
        #port stream data to pandas DataFrame object
        stream_zone_list = np.linspace(2,field_data.num_zones,
                                       field_data.num_zones-2+1)
        stream_df, x_subsolar = dump_to_pandas(main_frame,
                                               stream_zone_list, [1,2,3],
                                               'mp_stream_points.csv')
        #slice and construct XYZ data
        mp_mesh = surface_construct.ah_slicer(stream_df, tail_cap_mod,
                                              x_subsolar, nslice, nalpha,
                                              False)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, nfill, tail_cap_mod,
                        x_subsolar, 'mp_zone')
        load_cylinder(field_data, mp_mesh, 'mp_zone', I=nfill, J=nslice,
                      K=nalpha)

        '''
        #delete stream zones
        main_frame.activate()
        for zone in reversed(range(field_data.num_zones)):
            tp.active_frame().plot().fieldmap(zone).show=True
            if (field_data.zone(zone).name.find('cps_zone') == -1 and
                field_data.zone(zone).name.find('global_field') == -1 and
                field_data.zone(zone).name.find('mp_zone') == -1):
                field_data.delete_zones(field_data.zone(zone))
        '''

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
            magnetopause_power = surface_analysis(field_data, 'mp_zone',
                                                  nfill, nslice)
            print(magnetopause_power)
        if integrate_volume:
            mp_magnetic_energy = volume_analysis(field_data, 'mp_zone')
            print(mp_magnetic_energy)
        '''
        #clean extra frames generated
        page = tp.active_page()
        for frame in tp.frames('Frame*'):
            page.delete_frame(frame)
        '''
        write_to_timelog('output/mp_integral_log.csv', time.UTC[0],
                          magnetopause_power.combine(mp_magnetic_energy,
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
