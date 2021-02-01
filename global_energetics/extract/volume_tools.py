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
                                                    integrate_volume,
                                                      calculate_energetics,
                                                      dump_to_pandas)

def volume_analysis(field_data, zone_name, *, voluB=True, volKEpar=True,
                    volKEperp=True, volEth=True, volume=True, cuttoff=-20):
    """Function to calculate forms of total energy inside magnetopause or
    other zones
    Inputs
        field_data- tecplot Dataset object with 3d field data and mp
        zone_name
        cuttoff- X position to stop integration
    Outputs
        magnetic_energy- volume integrated magnetic energy B2/2mu0
    """
    if [var.name for var in field_data.variables()][::].count('K_out+')==0:
        calculate_energetics(field_data, zone_name)
    #initialize objects for main frame
    main_frame = [fr for fr in tp.frames('main')][0]
    volume_name = zone_name.split('_')[0]
    zone_index = int(field_data.zone(zone_name).index)
    #Blank X < X_cuttoff
    main_frame.plot().value_blanking.active = True
    xblank = main_frame.plot().value_blanking.constraint(1)
    xblank.active = True
    xblank.variable = field_data.variable('X *')
    xblank.comparison_operator = RelOp.LessThan
    xblank.comparison_value = cuttoff
    keys = []
    data = []
    keys = []
    data = []
    if voluB:
        #integrate magnetic energy
        keys.append(volume_name+' uB [J]')
        uB_index = int(field_data.variable('uB *').index)
        uB = integrate_volume(uB_index, zone_index, volume_name+' uB [J]',
                              subspace='tail')
        print('{} uB integration done'.format(zone_name))
        data.append(uB)
    if volKEpar:
        #integrate parallel KE
        keys.append(volume_name+' KEpar [J]')
        KEpar_index = int(field_data.variable('KEpar *').index)
        KEpar = integrate_volume(KEpar_index, zone_index,
                                 volume_name+' KEpar [J]',
                                 subspace='tail')
        print('{} KEparallel integration done'.format(zone_name))
        data.append(KEpar)
    if volKEperp:
        #integrate perp KE
        keys.append(volume_name+' KEperp [J]')
        KEperp_index = int(field_data.variable('KEperp *').index)
        KEperp = integrate_volume(KEperp_index, zone_index,
                                 volume_name+' KEperp [J]',
                                 subspace='tail')
        print('{} KEperp integration done'.format(zone_name))
        data.append(KEperp)
    if volEth:
        #integrate thermal energy
        keys.append(volume_name+' Etherm [J]')
        Eth_index = int(field_data.variable('P *').index)
        Eth = integrate_volume(Eth_index, zone_index,
                                 volume_name+' Etherm [J]',
                                 subspace='tail')
        print('{} Ethermal integration done'.format(zone_name))
        data.append(Eth)
    if volume:
        #integrate thermal energy
        keys.append(volume_name+' Volume [R^3]')
        Vol = integrate_volume(None, zone_index,
                               volume_name+' Volume [R^3]',
                               VariableOption='LengthAreaVolume')
        print('{} Volume integration done'.format(zone_name))
        data.append(Vol)
    volume_energies = pd.DataFrame([data], columns=keys)
    volume_energies = volume_energies * 6370**3
    #Turn blanking back off
    xblank.active = False
    main_frame.plot().value_blanking.active = False

    return volume_energies


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
