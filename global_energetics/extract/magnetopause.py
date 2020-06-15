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
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#custom packages
from global_energetics.extract import surface_construct
from global_energetics.extract.view_set import display_magnetopause
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (calc_dayside_mp,
                                                    calc_tail_mp,
                                                    dump_to_pandas,
                                                    create_cylinder,
                                                    load_cylinder,
                                                    calculate_energetics,
                                                    integrate_surface,
                                                    write_to_timelog)

def magnetopause_analysis(field_data, colorbar):
        """Function to calculate energy flux at magnetopause surface
        Inputs
            field_data- tecplot Dataset object with 3D field data and mp
            colorbar- settings for the color to be displayed on frame
        Outputs
            mp_power- power, or energy flux at the magnetopause surface
        """
        #calculate energetics
        calculate_energetics(field_data)
        #initialize objects for main frame
        main = tp.active_frame()
        mp_index = int(field_data.zone('mp_zone').index)
        Knet_index = int(field_data.variable('K_in *').index)
        Kplus_index = int(field_data.variable('K_in+*').index)
        Kminus_index = int(field_data.variable('K_in-*').index)
        #adjust main frame settings
        display_magnetopause(main, mp_index, Knet_index, colorbar, False)
        #integrate k flux
        integrate_surface(Kplus_index, mp_index,
                          'Total K_out [kW]', barid=0)
        main.activate()
        integrate_surface(Knet_index, mp_index,
                          'Total K_net [kW]', barid=1)
        main.activate()
        integrate_surface(Kminus_index, mp_index,
                          'Total K_in [kW]', barid=2)
        main.activate()
        for frames in tp.frames('Total K_in*'):
            influx = frames
        for frames in tp.frames('Total K_net*'):
            netflux = frames
        for frames in tp.frames('Total K_out*'):
            outflux = frames
        outflux.move_to_top()
        netflux.move_to_top()
        influx.move_to_top()
        outflux.activate()
        outflux_df, _ = dump_to_pandas([1],[4],'outflux.csv')
        netflux.activate()
        netflux_df, _ = dump_to_pandas([1],[4],'netflux.csv')
        influx.activate()
        influx_df, _ = dump_to_pandas([1],[4],'influx.csv')
        mp_power = outflux_df.combine(netflux_df, np.minimum,
                                     fill_value=1e12).combine(
                                    influx_df, np.minimum,
                                    fill_value=1e12).drop(
                                            columns=['Unnamed: 1'])
        main.activate()
        return mp_power


def get_magnetopause(field_data, datafile, *, pltpath='./', laypath='./',
                     pngpath='./', nstream_day=15, phi_max=122,
                     rday_max=30,rday_min=3.5, dayitr_max=100, daytol=0.1,
                     nstream_tail=15, rho_max=50,rho_step=0.5,tail_cap=-20,
                     nslice=40, nalpha=50,
                     rcolor=2.5):
    """Function that finds, plots and calculates energetics on the
        magnetopause surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        datafile- field data filename, assumes .plt file
        pltpath, laypath, pngpath- path for output of .plt,.lay,.png files
        nstream_day- number of streamlines generated for dayside algorithm
        phi_max- azimuthal (sun=0) limit of dayside algorithm for streams
        rday_max, rday_min- radial limits (in XY) for dayside algorithm
        dayitr_max, daytol- settings for bisection search algorithm
        nstream_tail- number of streamlines generated for tail algorithm
        rho_max, rho_step- tail disc maximium radius and step (in YZ)
        tail_cap- X position of tail cap
        nslice, nalpha- cylindrical points used for surface reconstruction
        rcolor- colorbar range, symmetrical about zero
    """
    #make unique outputname based on datafile string
    outputname = datafile.split('e')[1].split('-000.')[0]+'done'
    print(field_data)

    #set parameters
    phi = np.linspace(np.deg2rad(-1*phi_max),np.deg2rad(phi_max),
                      nstream_day)
    psi = np.linspace(-pi*(1-pi/nstream_tail), pi, nstream_tail)
    colorbar = np.linspace(-1*rcolor,rcolor,int(4*rcolor+1))
    with tp.session.suspend():
        tp.macro.execute_command("$!FrameName = 'main'")
        tp.data.operate.execute_equation(
                    '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')

        #Create Dayside Magnetopause field lines
        calc_dayside_mp(field_data, phi, rday_max, rday_min, dayitr_max,
                        daytol)
        #Create Tail magnetopause field lines
        calc_tail_mp(field_data, psi, tail_cap, rho_max, rho_step)
        #Create Theta and Phi coordinates for all points in domain
        tp.data.operate.execute_equation(
                                   '{phi} = atan({Y [R]}/({X [R]}+1e-24))')
        tp.data.operate.execute_equation(
                                   '{theta} = acos({Z [R]}/{r [R]}) * '+
                                    '({X [R]}+1e-24) / abs({X [R]}+1e-24)')
        #port stream data to pandas DataFrame object
        stream_zone_list = np.linspace(2,field_data.num_zones,
                                       field_data.num_zones-2+1)
        stream_df, x_subsolar = dump_to_pandas(stream_zone_list, [1,2,3],
                                          'stream_points.csv')
        #slice and construct XYZ data
        mp_mesh = surface_construct.yz_slicer(stream_df, tail_cap,
                                              x_subsolar, nslice, nalpha,
                                              False)
        #create and load cylidrical zone
        create_cylinder(field_data, nslice, nalpha, tail_cap, x_subsolar)
        load_cylinder(field_data, mp_mesh, 'mp_zone',
                      range(0,2), range(0,nslice), range(0,nalpha))

        #interpolate field data to zone
        print('interpolating field data to magnetopause')
        tp.data.operate.interpolate_inverse_distance(
                destination_zone=field_data.zone('mp_zone'),
                source_zones=field_data.zone('global_field'))
        magnetopause_power = magnetopause_analysis(field_data, colorbar)
        print(magnetopause_power)
        write_to_timelog('integral_log.csv',outputname, magnetopause_power)

        #delete stream zones
        for zone in reversed(range(field_data.num_zones)):
            tp.active_frame().plot().fieldmap(zone).show=True
            if (field_data.zone(zone).name != 'mp_zone' and
                field_data.zone(zone).name != 'global_field'):
                field_data.delete_zones(field_data.zone(zone))

        #write .plt and .lay files
        #tp.data.save_tecplot_plt(pltpath+outputname+'.plt')
        #tp.save_layout(laypath+outputname+'.lay')
        tp.export.save_png(pngpath+outputname+'.png')


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
