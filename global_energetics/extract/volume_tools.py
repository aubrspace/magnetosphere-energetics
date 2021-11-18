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

def volume_analysis(frame, state_var, analysis_type, do_1Dsw, do_cms,
                    dt, cuttoff, tail_h):
    """Function to calculate forms of total energy inside magnetopause or
    other zones
    Inputs
        frame- tecplot frame that contains the field data dataset
        zone_name
        cuttoff- X position to stop integration
    Outputs
        magnetic_energy- volume integrated magnetic energy B2/2mu0
    """
    #initalize everything to False
    voluB=False
    voluE=False
    volKE=False
    volKEpar=False
    volKEperp=False
    volEth=False
    volTotal=False
    volume=False
    virial=False
    biotsavart=False
    if 'energy' in analysis_type or analysis_type=='all':
        voluB=True
        voluE=True
        volKE=True
        volEth=True
        volTotal=True
        volume=True
    if 'virial' in analysis_type or analysis_type=='all':
        voluB=True
        volKE=True
        volEth=True
        virial=True
        volume=True
        voluB=True
        volKE=True
        volEth=True
    if 'biotsavart' in analysis_type or analysis_type=='all':
        biotsavart=True
    if analysis_type=='all':
        volKEpar=True
        volKEperp=True
    #initialize objects for main frame
    field_data = frame.dataset
    volume_name = state_var
    zone_index = int(field_data.zone('global_field').index)
    if len([var for var in field_data.variables('beta_star')]) < 1:
        print('Global variables not setup! Cannot perform integration')
        return None
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
            eq('{uBtot '+state_var+'} = IF({'+state_var+'}<1, 0, '+
                                   '{'+add+'uB [J/Re^3]})')
            keys.append(add+'uBtot [J]')
            uB_index = int(field_data.variable('uBtot '+state_var).index)
            uB = integrate_volume(uB_index, zone_index)
            print(add+'{} uBtot integration done'.format(volume_name))
            data.append(uB)
            if 'virial' in analysis_type or analysis_type=='all':
                #integrate dipole magnetic energy
                eq('{uB_dipole temp} = IF({'+state_var+'}<1, 0, '+
                                   '{'+add+'uB_dipole [J/Re^3]})')
                keys.append(add+'uB_dipole [J]')
                uB_index = int(field_data.variable('uB_dipole temp').index)
                uB = integrate_volume(uB_index, zone_index)
                print(add+'{} uB dipole integration done'.format(volume_name))
                data.append(uB)
                #integrate disturbance magnetic energy
                eq('{uB_dist temp} = IF({'+state_var+'}<1, 0, '+
                                   '{'+add+'delta_uB [J/Re^3]})')
                keys.append(add+'uB_dist [J]')
                uB_index = int(field_data.variable('uB_dist temp').index)
                uB = integrate_volume(uB_index, zone_index)
                print(add+'{} uB disturb integration done'.format(volume_name))
                data.append(uB)
        if volKE:
            #integrate KE
            eq('{KE '+state_var+'}=IF({'+state_var+'}<1,0,'+
                               '{'+add+'KE [J/Re^3]})')
            keys.append(add+'KE [J]')
            KE_index = int(field_data.variable('KE '+state_var).index)
            KE = integrate_volume(KE_index, zone_index)
            print(add+'{} KE integration done'.format(volume_name))
            data.append(KE)
        if volKEpar:
            #integrate parallel KE
            eq('{KEpar '+state_var+'}=IF({'+state_var+'}<1,0,'+
                               '{'+add+'KEpar [J/Re^3]})')
            keys.append(add+'KEpar [J]')
            KEpar_index = int(field_data.variable('KEpar '+state_var).index)
            KEpar = integrate_volume(KEpar_index, zone_index)
            print(add+'{} KEparallel integration done'.format(volume_name))
            data.append(KEpar)
        if volKEperp:
            #integrate perp KE
            eq('{KEperp '+state_var+'} =IF({'+state_var+'}<1,0,'+
                                 '{'+add+'KEperp [J/Re^3]})')
            keys.append(add+'KEperp [J]')
            KEperp_index = int(field_data.variable('KEperp '+state_var).index)
            KEperp = integrate_volume(KEperp_index, zone_index)
            print(add+'{} KEperp integration done'.format(volume_name))
            data.append(KEperp)
        if volEth:
            #integrate thermal energy
            eq('{Etherm '+state_var+'} =IF({'+state_var+'}<1,0,'+
                                                '{P [nPa]}*6371**3*1.5)')
            keys.append(add+'Etherm [J]')
            Eth_index = int(field_data.variable('Etherm '+state_var).index)
            Eth = integrate_volume(Eth_index, zone_index)
            print(add+'{} Ethermal integration done'.format(volume_name))
            data.append(Eth)
        if volTotal:
            #integrate total energy
            eq('{Total '+state_var+'} =IF({'+state_var+'}<1,0,'+
                                                '{'+add+'Utot [J/Re^3]})')
            keys.append(add+'Total [J]')
            Total_index = int(field_data.variable('Total '+state_var).index)
            Total = integrate_volume(Total_index, zone_index)
            print(add+'{} Total integration done'.format(volume_name))
            data.append(Total)
        if volume:
            #integrate volume size
            eq('{Volume '+state_var+'} =IF({'+state_var+'}>0,'+
                                                    '{'+state_var+'},0)')
            keys.append('Volume [Re^3]')
            Vol_index = int(field_data.variable('Volume '+state_var).index)
            Vol = integrate_volume(Vol_index, zone_index)
            print('{} Volume integration done'.format(volume_name))
            data.append(Vol)
            volume = False
        if volTotal and volume:
            #Energy density
            keys.append(add+'Energy Density [J/Re^3]')
            energy_density = Total/Vol
            data.append(energy_density)
        if biotsavart:
            #evaluate Biot Savart integral for magnetosphere and fulldomain
            eq('{BioS '+state_var+'}=IF({'+state_var+'}<1,0,'+'{dB [nT]})')
            keys.append(add+'BioS '+state_var)
            bioS_ms_index=int(field_data.variable('BioS '+state_var).index)
            bioS_ms = integrate_volume(bioS_ms_index, zone_index)
            print('{} BioS ms integration done'.format(volume_name))
            data.append(bioS_ms)
            keys.append(add+'BioS full')
            bioS_index=int(field_data.variable('dB *').index)
            bioS = integrate_volume(bioS_index, zone_index)
            print('{} BioS full integration done'.format(volume_name))
            data.append(bioS)
        if virial:
            #integrate r^2 weighted mass (will look @ 2nd derivative)
            eq('{rho r^2 '+state_var+'} = IF({'+state_var+'}<1,0,'+
                                                 '{rho r^2 [kgm^2/Re^3]})')
            keys.append(add+'Mr^2 [kgm^2]')
            rhor2_index = int(field_data.variable('rho r^2 '+state_var).index)
            rhor2 = integrate_volume(rhor2_index, zone_index)
            print('{} Mr^2 integration done'.format(volume_name))
            data.append(rhor2)
            #Virial kinetic energy
            keys.append('Virial 2x Uk [J]')
            data.append(2*KE+2*Eth)
            keys.append('Virial 2x Uk [nT]')
            data.append((2*KE+2*Eth)/(-8e13))
            #Virial differential magnetic field energy
            keys.append('Virial Ub [J]')
            data.append(uB)
            keys.append('Virial Ub [nT]')
            data.append(uB/(-8e13))
            #Virial volumetric subtotal
            keys.append('Virial Volume Total [J]')
            data.append(uB+2*KE+2*Eth)
            keys.append('Virial Volume Total [nT]')
            data.append((uB+2*KE+2*Eth)/(-8e13))
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
