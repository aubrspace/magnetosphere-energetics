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
from global_energetics.extract.surface_tools import (energy_to_dB,
                                                     calc_integral)
from global_energetics.extract.stream_tools import (integrate_tecplot,
                                                    get_day_flank_tail,
                                                      dump_to_pandas)

def virial_post_integr(results):
    """Creates dictionary of key:value combos of existing results
    Inputs
        results(dict{str:[value]})
    Outputs
        results(dict{str:[value]})
    """
    newresults = {}
    #Virial volume terms
    newresults.update(
     {'Virial 2x Uk [J]':[2*results['KE [J]'][0]+results['Pth [J]'][0]/1.5],
      'Virial Ub [J]':[results['delta_uB [J]'][0]]})
    newresults.update({
              'Virial Volume Total [J]':[newresults['Virial 2x Uk [J]'][0]+
                                         newresults['Virial Ub [J]'][0]]})
    #Convert from J to nT
    for newterm in newresults.copy().items():
        newresults.update(energy_to_dB(newterm))
    return newresults
def get_biotsavart_integrands(state_var):
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
        zone(Zone)- tecplot zone object used to decide which terms included
    Outputs
        virialdict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    state, biotsavart_dict = state_var.name, {}
    eq = tp.data.operate.execute_equation
    eq('{BioS '+state_var+'}=IF({'+state_var+'}<1,0,'+'{dB [nT]})')
    return {'BioS '+state_var:'bioS_ms [nT]',
            'dB [nT]':'bioS_full [nT]'}

def get_virial_integrands(state_var):
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
        zone(Zone)- tecplot zone object used to decide which terms included
    Outputs
        virialdict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    state, virialdict = state_var.name, {}
    eq = tp.data.operate.execute_equation
    existing_variables = state_var.dataset.variable_names
    #Integrands
    integrands = ['uB [J/Re^3]', 'KE [J/Re^3]', 'Pth [J/Re^3]',
                  'delta_uB [J/Re^3]', 'rho r^2 [kgm^2/Re^3]']
    for term in integrands:
        name = term.split(' [')[0]
        if name+state not in existing_variables:
            eq('{'+name+state+'}=IF({'+state+'}<1, 0, {'+term+'})')
            if 'rho' in name:
                virialdict.update({name+state:name+' [kgm^2]'})
            else:
                virialdict.update({name+state:name+' [J]'})
    return virialdict

def get_energy_integrands(state_var):
    """Creates dictionary of terms to be integrated for energy analysis
    Inputs
        state_var(Variable)- variable used to determine spatial limits
    Outputs
        energydict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    state, energydict  = state_var.name, {}
    eq = tp.data.operate.execute_equation
    existing_variables = state_var.dataset.variable_names
    #Integrands
    integrands = ['uB [J/Re^3]', 'KE [J/Re^3]', 'Pth [J/Re^3]',
                  'uHydro [J/Re^3]', 'Utot [J/Re^3]']
    for term in integrands:
        name = term.split(' ')[0]
        if name+state not in existing_variables:
            #Create variable for integrand that only exists in isolated zone
            if 'Pth' in term:
                eq('{'+name+state+'}=IF({'+state+'}<1, 0, 1.5*{'+term+'})')
            else:
                eq('{'+name+state+'}=IF({'+state+'}<1, 0, {'+term+'})')
            energydict.update({name+state:name+' [J]'})
    return energydict

def get_dft_integrands(zone, integrands):
    """Creates dictionary of terms to be integrated for energy analysis
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    dft_dict, eq = {}, tp.data.operate.execute_equation
    #May need to get Day flank tail designations first
    if not any(['Tail' in n for n in zone.dataset.variable_names]):
        get_day_flank_tail(zone)
    for term in integrands.items():
        name = term[0].split(' ')[0]
        if not any([n in name for n in ['Closed','Open']]):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Day}={Day}*{'+term[0]+'}',zones=[zone])
            eq('{'+name+'Flank}={Flank}*{'+term[0]+'}',zones=[zone])
            eq('{'+name+'Tail}={Tail}*{'+term[0]+'}',zones=[zone])
            dft_dict.update({name+'Day':name+'Day '+units,
                             name+'Flank':name+'Flank '+units,
                             name+'Tail':name+'Tail '+units})
    return dft_dict

def get_open_close_integrands(zone, integrands):
    """Creates dictionary of terms to be integrated for energy analysis
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    openClose_dict, eq = {}, tp.data.operate.execute_equation
    #May need to get Day flank tail designations first
    if not any(['Tail' in n for n in zone.dataset.variable_names]):
        get_day_flank_tail(zone)
    for term in integrands.items():
        name = term[0].split(' ')[0]
        if not any([n in name for n in ['Day','Flank','Tail']]):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Closed}=IF({Status}==3&&{Tail}==0,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenN}=IF({Status}==2&&{Tail}==0,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenS}=IF({Status}==1&&{Tail}==0,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            openClose_dict.update({name+'Closed':name+'Closed '+units,
                                   name+'OpenN':name+'OpenN '+units,
                                   name+'OpenS':name+'OpenS '+units})
    return openClose_dict

def get_mobile_integrands(zone, integrands, tdelta):
    """Creates dict of integrands for surface motion effects
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
        tdelta(float)- timestep between output files
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    mobiledict, td, eq = {}, str(tdelta), tp.data.operate.execute_equation
    for term in integrands:
        name = term.split(' ')[0]
        eq('{'+name+'acqu}=IF({delta_volume}==1,  -1*{'+term+'}/'+td+',0)',
                                                              zones=[zone])
        eq('{'+name+'forf}=IF({delta_volume}==-1,    {'+term+'}/'+td+',0)',
                                                              zones=[zone])
        eq('{'+name+'net} =    {delta_volume}   * -1*{'+term+'}/'+td,
                                                              zones=[zone])
        mobiledict.update({name+'acqu':name+'acqu [W]',
                           name+'forf':name+'forf [W]',
                           name+'net':name+'net [W]'})
    return mobiledict

def volume_analysis(state_var, **kwargs):
    """Function to calculate forms of total energy inside magnetopause or
    other zones
    Inputs
        state_var(Variable)- tecplot variable used to isolate vol in zone
        kwargs:
            doVolume(bool)- True, to calculate the volume of the zone
            customTerms(dict(str:str))- integrand:result name pairs
    Outputs
        magnetic_energy- volume integrated magnetic energy B2/2mu0
    """
    #Check for global variables
    assert 'r [R]' in state_var.dataset.variable_names, ('Need to'+
                                       'calculate global variables first!')
    if 'analysis_type' in kwargs:
        analysis_type = kwargs.pop('analysis_type')
    if 'mp' not in state_var.name:
        ''.join(analysis_type.split('energy'))
    #initialize empty dictionary that will make up the results of calc
    integrands, results, eq = {}, {}, tp.data.operate.execute_equation
    global_zone = state_var.dataset.zone(0)
    ###################################################################
    #Core integral terms
    if 'virial' in analysis_type:
        integrands.update(get_virial_integrands(state_var))
    if 'energy' in analysis_type:
        integrands.update(get_energy_integrands(state_var))
    if 'biotsavart' in analysis_type:
        integrands.update(get_biotsavart_integrands(state_var))
    integrands.update(kwargs.get('customTerms', {}))
    ###################################################################
    #Integral bounds modifications
    integrands.update(get_dft_integrands(global_zone,integrands))
    integrands.update(get_open_close_integrands(global_zone,integrands))
    if kwargs.get('do_cms', False):
        integrands.update(get_cms_energy_integrands(global_zone,integrands))
    ###################################################################
    #Evaluate integrals
    for term in integrands.items():
        results.update(calc_integral(term, global_zone))
        if kwargs.get('verbose',False):
            print(stat_var.name+term[1]+' integration done')
    ###################################################################
    #Non scalar integrals (empty integrands)
    if kwargs.get('doVolume', True):
        eq('{Volume '+state_var.name+'}=IF({'+state_var.name+'}<1, 0,1)')
        results.update(calc_integral((' ','Volume '+state_var.name),
                           global_zone, VariableOption='LengthAreaVolume'))
        if kwargs.get('do_cms',False):
            results.update(calc_integral((' ','delta_volume'),
                           global_zone, VariableOption='LengthAreaVolume'))
        if kwargs.get('verbose',False):
            print(stat_var.name+' Volume integration done')
    ###################################################################
    #Post integration manipulations
    if 'virial' in analysis_type:
        results.update(virial_post_integr(results))
    return pd.DataFrame(results)


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
