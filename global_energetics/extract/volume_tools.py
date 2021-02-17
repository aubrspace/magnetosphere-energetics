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
                                                      dump_to_pandas)

def volume_analysis(frame, state_variable_name, *,
                    voluB=True, voluE=True, volKEpar=True, volKEperp=True,
                    volEth=True, volume=True,
                    cuttoff=-20):
    """Function to calculate forms of total energy inside magnetopause or
    other zones
    Inputs
        frame- tecplot frame that contains the field data dataset
        zone_name
        cuttoff- X position to stop integration
    Outputs
        magnetic_energy- volume integrated magnetic energy B2/2mu0
    """
    #initialize objects for main frame
    field_data = frame.dataset
    volume_name = state_variable_name
    zone_index = int(field_data.zone('global_field').index)
    if len([var for var in field_data.variables('K_x *')]) < 1:
        print('Global variables not setup! Cannot perform integration')
        return None
    #Blank X < X_cuttoff
    frame.plot().value_blanking.active = True
    xblank = frame.plot().value_blanking.constraint(1)
    xblank.active = True
    xblank.variable = field_data.variable('X *')
    xblank.comparison_operator = RelOp.LessThan
    xblank.comparison_value = cuttoff
    keys = []
    data = []
    keys = []
    data = []
    eq = tp.data.operate.execute_equation
    if voluB:
        #integrate magnetic energy
        eq('{uB temp} = IF({'+state_variable_name+'}<1, 0, {uB [J/Re^3]})')
        keys.append(volume_name+' uB [J]')
        uB_index = int(field_data.variable('uB temp').index)
        uB = integrate_volume(uB_index, zone_index)
        print('{} uB integration done'.format(volume_name))
        data.append(uB)
    if voluE:
        #integrate electric energy
        eq('{uE temp} = IF({'+state_variable_name+'}<1, 0, {uE [J/Re^3]})')
        keys.append(volume_name+' uE [J]')
        uE_index = int(field_data.variable('uE temp').index)
        uE = integrate_volume(uE_index, zone_index)
        print('{} uE integration done'.format(volume_name))
        data.append(uE)
    if volKEpar:
        #integrate parallel KE
        eq(
         '{KEpar temp}=IF({'+state_variable_name+'}<1,0,{KEpar [J/Re^3]})')
        keys.append(volume_name+' KEpar [J]')
        KEpar_index = int(field_data.variable('KEpar temp').index)
        KEpar = integrate_volume(KEpar_index, zone_index)
        print('{} KEparallel integration done'.format(volume_name))
        data.append(KEpar)
    if volKEperp:
        #integrate perp KE
        eq('{KEperp temp} =IF({'+state_variable_name+'}<1,0,'+
                                                      '{KEperp [J/Re^3]})')
        keys.append(volume_name+' KEperp [J]')
        KEperp_index = int(field_data.variable('KEperp temp').index)
        KEperp = integrate_volume(KEperp_index, zone_index)
        print('{} KEperp integration done'.format(volume_name))
        data.append(KEperp)
    if volEth:
        #integrate thermal energy
        eq('{Etherm temp} =IF({'+state_variable_name+'}<1,0,'+
                                                      '{P [nPa]}*6371**3)')
        keys.append(volume_name+' Etherm [J]')
        Eth_index = int(field_data.variable('Etherm temp').index)
        Eth = integrate_volume(Eth_index, zone_index)
        print('{} Ethermal integration done'.format(volume_name))
        data.append(Eth)
    if volume:
        #integrate thermal energy
        eq('{Volume temp} =IF({'+state_variable_name+'}<1,0,'+
                                                          '{Cell Volume})')
        keys.append(volume_name+' Volume [Re^3]')
        Vol_index = int(field_data.variable('Volume temp').index)
        Vol = integrate_volume(Vol_index, zone_index)
        print('{} Volume integration done'.format(volume_name))
        data.append(Vol)
    volume_energies = pd.DataFrame([data], columns=keys)
    #Turn blanking back off
    xblank.active = False
    frame.plot().value_blanking.active = False

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
