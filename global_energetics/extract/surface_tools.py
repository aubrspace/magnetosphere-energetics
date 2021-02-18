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
                                                    get_surface_variables,
                                                    dump_to_pandas)

def surface_analysis(frame, zone_name, *,
                     calc_K=True, calc_ExB=True, calc_P0=True,
                     surface_area=True, cuttoff=-20):
    """Function to calculate energy flux at magnetopause surface
    Inputs
        field_data- tecplot Dataset object with 3D field data and mp
        zone_name
        calc_ - boolean for performing that integration
        cuttoff- used to blank tail end of surface
    Outputs
        surface_power- power, or energy flux at the magnetopause surface
    """
    #get surface specific variables
    field_data = frame.dataset
    get_surface_variables(field_data, zone_name)
    #initialize objects for frame
    zone_index = int(field_data.zone(zone_name).index)
    #Blank X < X_cuttoff
    frame.plot().value_blanking.active = True
    xblank = frame.plot().value_blanking.constraint(1)
    xblank.active = True
    xblank.variable = field_data.variable('X *')
    xblank.comparison_operator = RelOp.LessThan
    xblank.comparison_value = cuttoff
    keys = []
    data = []
    ######################################################################
    #integrate Poynting flux
    if calc_ExB:
        #ESCAPE
        keys.append(zone_name+' ExB_escape [W]')
        ExBesc_index = int(field_data.variable('ExB_escape*').index)
        ExBesc = integrate_surface(ExBesc_index, zone_index)
        data.append(ExBesc)
        #NET
        keys.append(zone_name+' ExB_net [W]')
        ExBnet_index = int(field_data.variable('ExB_net *').index)
        ExBnet = integrate_surface(ExBnet_index, zone_index)
        data.append(ExBnet)
        #INJECTION
        keys.append(zone_name+' ExB_injection [W]')
        ExBinj_index = int(field_data.variable('ExB_injection*').index)
        ExBinj = integrate_surface(ExBinj_index, zone_index)
        data.append(ExBinj)
        print('{} ExB integration done'.format(zone_name))
    ######################################################################
    #integrate P0 flux
    if calc_P0:
        #ESCAPE
        keys.append(zone_name+' P0_escape [W]')
        P0esc_index = int(field_data.variable('P0_escape*').index)
        P0esc = integrate_surface(P0esc_index, zone_index)
        data.append(P0esc)
        #NET
        keys.append(zone_name+' P0_net [W]')
        P0net_index = int(field_data.variable('P0_net *').index)
        P0net = integrate_surface(P0net_index, zone_index)
        data.append(P0net)
        #INJECTION
        keys.append(zone_name+' P0_injection [W]')
        P0inj_index = int(field_data.variable('P0_injection*').index)
        P0inj = integrate_surface(P0inj_index, zone_index)
        data.append(P0inj)
        print('{} P0 integration done'.format(zone_name))
    ######################################################################
    #integrate K flux
    if calc_K:
        #ESCAPE
        keys.append(zone_name+' K_escape [W]')
        kesc_index = int(field_data.variable('K_escape*').index)
        kesc = integrate_surface(kesc_index, zone_index)
        data.append(kesc)
        #NET
        keys.append(zone_name+' K_net [W]')
        knet_index = int(field_data.variable('K_net *').index)
        knet = integrate_surface(knet_index, zone_index)
        data.append(knet)
        #INJECTION
        keys.append(zone_name+' K_injection [W]')
        kinj_index = int(field_data.variable('K_injection*').index)
        kinj = integrate_surface(kinj_index, zone_index)
        data.append(kinj)
        print('{} K integration done'.format(zone_name))
    ######################################################################
    #integrate area
    if surface_area:
        keys.append(zone_name+' Area [Re^2]')
        area_index = None
        SA = integrate_surface(area_index, zone_index,
                                   VariableOption='LengthAreaVolume')
        data.append(SA)
        print('{} area integration done'.format(zone_name))
    ######################################################################
    #average K flux
    if calc_K and surface_area:
        #ESCAPE
        keys.append(zone_name+' Average K_escape [W/Re^2]')
        kesc_average = kesc/SA
        data.append(kesc_average)
        #NET
        keys.append(zone_name+' Average K_net [W/Re^2]')
        knet_average = knet/SA
        data.append(knet_average)
        #INJECTION
        keys.append(zone_name+' Average K_injection [W/Re^2]')
        kinj_average = kinj/SA
        data.append(kinj_average)
    ######################################################################
    #Collect and report surface integrated quantities
    surface_power = pd.DataFrame([data],columns=keys)
    #Turn blanking back off
    xblank.active = False
    frame.plot().value_blanking.active = False
    return surface_power


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
