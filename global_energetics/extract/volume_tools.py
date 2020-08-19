#!/usr/bin/env python3
"""Functions for analyzing surfaces from field data
"""
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
#interpackage modules, different path if running as main to test
from global_energetics.extract.stream_tools import (integrate_surface,
                                                      calculate_energetics,
                                                      dump_to_pandas)

def volume_analysis(field_data, zone_name):
    """Function to calculate forms of total energy inside magnetopause or
    other zones
    Inputs
        field_data- tecplot Dataset object with 3d field data and mp
        zone_name
    Outputs
        magnetic_energy- volume integrated magnetic energy B2/2mu0
    """
    if [var.name for var in field_data.variables()][::].count('K_in+')==0:
        calculate_energetics(field_data, zone_name)
    #initialize objects for main frame
    main_frame = [fr for fr in tp.frames('main')][0]
    volume_name = zone_name.split('_')[0]
    zone_index = int(field_data.zone(zone_name).index)
    uB_index = int(field_data.variable('uB *').index)
    #integrate magnetic energy
    uB_frame = integrate_surface(uB_index, zone_index,
                                   volume_name+' uB [J]')
    #Identify and delete dummy frame as workaround
    dummy_frame = [fr for fr in tp.frames('Frame*')][0]
    dummy_frame.move_to_bottom()
    #port data to pandas dataframes
    uB_df, _ = dump_to_pandas(uB_frame, [1], [4],
                                volume_name+'.csv')
    uB_df = uB_df*6371**3
    magnetic_energy = uB_df.drop(columns=['Unnamed: 1'])

    return magnetic_energy


# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"
# Run as main to test script functionality, will need valid .plt file that
# can handle the dummy circular zone

if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()
    os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    tp.active_frame().name = 'main'
    #Give valid test dataset here!
    DATASET = tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')
    #Create small test zone
    tp.macro.execute_command('''$!CreateCircularZone
                             IMax = 2
                             JMax = 20
                             KMax = 5
                             X = 0
                             Y = 0
                             Z1 = 0
                             Z2 = 5
                             Radius = 5''')
    POWER = surface_analysis(DATASET, 'Circular zone', [-5,0,5])
    print(POWER)
