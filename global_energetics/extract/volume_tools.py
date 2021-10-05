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

def volume_analysis(frame, state_variable_name, do_1Dsw, do_cms, rblank, *,
                    voluB=True, voluE=True, volKE=True, volKEpar=True,
                    volKEperp=True, volEth=True, volTotal=True,
                    volume=True, findS=True, virial=True, dt=60,
                    cuttoff=-20, blank=True, tail_h=15):
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
    if len([var for var in field_data.variables('beta_star')]) < 1:
        print('Global variables not setup! Cannot perform integration')
        return None
    if blank:
        #Blank X < X_cuttoff
        frame.plot().value_blanking.active = True
        rblank = frame.plot().value_blanking.constraint(2)
        rblank.active = True
        rblank.variable = field_data.variable('r *')
        rblank.comparison_operator = RelOp.LessThan
        rblank.comparison_value = rblank
        rblank.blanking.cell_mode = ValueBlankCellMode.AllCorners
    keys = []
    data = []
    eq = tp.data.operate.execute_equation
    ##Different prefixes allow for calculation of surface fluxes using 
    #   multiple sets of flowfield variables (denoted by the prefix)
    prefixlist = ['']
    if do_1Dsw:
        prefixlist.append('1D')
    for add in prefixlist:
        if voluB:
            #integrate total magnetic energy
            eq('{uBtot temp} = IF({'+state_variable_name+'}<1, 0, '+
                                   '{'+add+'uB [J/Re^3]})')
            keys.append(add+'uBtot [J]')
            uB_index = int(field_data.variable('uBtot temp').index)
            uB = integrate_volume(uB_index, zone_index)
            print(add+'{} uBtot integration done'.format(volume_name))
            data.append(uB)
            #integrate dipole magnetic energy
            eq('{uB_dipole temp} = IF({'+state_variable_name+'}<1, 0, '+
                                   '{'+add+'uB_dipole [J/Re^3]})')
            keys.append(add+'uB_dipole [J]')
            uB_index = int(field_data.variable('uB_dipole temp').index)
            uB = integrate_volume(uB_index, zone_index)
            print(add+'{} uB dipole integration done'.format(volume_name))
            data.append(uB)
            #integrate disturbance magnetic energy
            eq('{uB_dist temp} = IF({'+state_variable_name+'}<1, 0, '+
                                   '{'+add+'delta_uB [J/Re^3]})')
            keys.append(add+'uB_dist [J]')
            uB_index = int(field_data.variable('uB_dist temp').index)
            uB = integrate_volume(uB_index, zone_index)
            print(add+'{} uB disturb integration done'.format(volume_name))
            data.append(uB)
        if volKE:
            #integrate KE
            eq('{KE temp}=IF({'+state_variable_name+'}<1,0,'+
                               '{'+add+'KE [J/Re^3]})')
            keys.append(add+'KE [J]')
            KE_index = int(field_data.variable('KE temp').index)
            KE = integrate_volume(KE_index, zone_index)
            print(add+'{} KE integration done'.format(volume_name))
            data.append(KE)
        if volKEpar:
            #integrate parallel KE
            eq('{KEpar temp}=IF({'+state_variable_name+'}<1,0,'+
                               '{'+add+'KEpar [J/Re^3]})')
            keys.append(add+'KEpar [J]')
            KEpar_index = int(field_data.variable('KEpar temp').index)
            KEpar = integrate_volume(KEpar_index, zone_index)
            print(add+'{} KEparallel integration done'.format(volume_name))
            data.append(KEpar)
        if volKEperp:
            #integrate perp KE
            eq('{KEperp temp} =IF({'+state_variable_name+'}<1,0,'+
                                 '{'+add+'KEperp [J/Re^3]})')
            keys.append(add+'KEperp [J]')
            KEperp_index = int(field_data.variable('KEperp temp').index)
            KEperp = integrate_volume(KEperp_index, zone_index)
            print(add+'{} KEperp integration done'.format(volume_name))
            data.append(KEperp)
        if volEth:
            #integrate thermal energy
            eq('{Etherm temp} =IF({'+state_variable_name+'}<1,0,'+
                                                '{P [nPa]}*6371**3*1.5)')
            keys.append(add+'Etherm [J]')
            Eth_index = int(field_data.variable('Etherm temp').index)
            Eth = integrate_volume(Eth_index, zone_index)
            print(add+'{} Ethermal integration done'.format(volume_name))
            data.append(Eth)
        if volTotal:
            #integrate total energy
            eq('{Total temp} =IF({'+state_variable_name+'}<1,0,'+
                                                '{'+add+'Utot [J/Re^3]})')
            keys.append(add+'Total [J]')
            Total_index = int(field_data.variable('Total temp').index)
            Total = integrate_volume(Total_index, zone_index)
            print(add+'{} Total integration done'.format(volume_name))
            data.append(Total)
        if volume:
            #integrate volume size
            eq('{Volume temp} =IF({'+state_variable_name+'}<1,0,1)')
            keys.append('Volume [Re^3]')
            Vol_index = int(field_data.variable('Volume temp').index)
            Vol = integrate_volume(Vol_index, zone_index)
            print('{} Volume integration done'.format(volume_name))
            data.append(Vol)
            volume = False
        #Energy density
        keys.append(add+'Energy Density [J/Re^3]')
        energy_density = Total/Vol
        data.append(energy_density)
        if virial:
            #Virial kinetic energy
            keys.append('Virial 2x Uk [J]')
            data.append(2*KE+2*Eth)
            keys.append('Virial 2x Uk [nT]')
            data.append((2*KE+2*Eth)*(-3/2)/(8e13))
            #Virial differential magnetic field energy
            keys.append('Virial Ub [J]')
            data.append(uB)
            keys.append('Virial Ub [nT]')
            data.append(uB*(-3/2)/(8e13))
        if (do_cms) and (dt!=0):
            ##Volume change
            dVol_index = field_data.variable('delta_volume').index
            dVol = integrate_volume(dVol_index, zone_index)
            keys.append('dVolume [Re^3]')
            print('{} Volume integration done'.format(volume_name))
            data.append(dVol)
            #Temporary DayFlankTail designations
            eq('{DayTemp} = IF({X [R]}>0,1,0)', zones=[zone_index])
            eq('{TailTemp} = IF(({X [R]}<-5&&{h}<'+str(tail_h)+'*0.8)||'+
                               '({X [R]}<-10&&{h}<'+str(tail_h)+'),1,0)',
                                                zones=[zone_index])
            eq('{FlankTemp} = IF({DayTemp}==0&&{TailTemp}==0,1,0)',
                                                    zones=[zone_index])
            for qty in['uB [J/Re^3]','uHydro [J/Re^3]','Utot [J/Re^3]']:
                name = qty.split(' ')[0]
                eq('{'+name+'Day}={DayTemp}*{'+qty+'}',zones=[zone_index])
                eq('{'+name+'Flank}={FlankTemp}*{'+qty+'}',
                                                       zones=[zone_index])
                eq('{'+name+'Tail}={TailTemp}*{'+qty+'}',zones=[zone_index])
            ##Integrate acquired/forfeited flux
            for qty in['uB [J/Re^3]','uHydro [J/Re^3]','Utot [J/Re^3]',
                       'uBDay','uHydroDay','UtotDay',
                       'uBFlank','uHydroFlank','UtotFlank',
                       'uBTail','uHydroTail','UtotTail']:
                temp=qty.split(' ')[0]
                eq('{'+temp+'acqu} =IF({delta_volume}== 1,-1*{'+qty+'},0)')
                eq('{'+temp+'forf} =IF({delta_volume}==-1,   {'+qty+'},0)')
                eq('{'+temp+'net} =    {delta_volume}   * -1*{'+qty+'}   ')
                acqu_index = field_data.variable(temp+'acqu').index
                forf_index = field_data.variable(temp+'forf').index
                net_index  = field_data.variable(temp+'net').index
                acqu = integrate_volume(acqu_index, zone_index)
                forf = integrate_volume(forf_index, zone_index)
                net  = integrate_volume(net_index, zone_index)
                keys.append(qty.split(' ')[0]+'_acquired [W]')
                keys.append(qty.split(' ')[0]+'_forfeited [W]')
                keys.append(qty.split(' ')[0]+'_net [W]')
                print('{} Volume integration done'.format(qty))
                data.append(acqu/dt)
                data.append(forf/dt)
                data.append(net/dt)
    volume_energies = pd.DataFrame([data], columns=keys)
    if blank:
        #Turn blanking back off
        rblank.active = False
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
