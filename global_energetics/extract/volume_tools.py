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
                                        get_open_close_integrands,
                                               get_dft_integrands,
                                                    calc_integral)
from global_energetics.extract.stream_tools import (integrate_tecplot,
                                                    get_day_flank_tail,
                                                      dump_to_pandas)

def energy_post_integr(results, **kwargs):
    """Creates dictionary of key:value combos of existing results
    Inputs
        results(dict{str:[value]})
        kwargs:
            do_cms
    Outputs
        newterms(dict{str:[value]})
    """
    newterms = {}
    #Virial volume terms
    newterms.update({'uHydro [J]':[results['Eth [J]'][0]+
                                        results['KE [J]'][0]]})
    newterms.update({'Utot [J]':[newterms['uHydro [J]'][0]+
                                      results['uB [J]'][0]]})
    if kwargs.get('do_cms', False):
        for direction in ['_acqu', '_forf', '_net']:
            newterms.update({'uHydro'+direction+' [W]':
                                      [results['Eth'+direction+' [W]'][0]+
                                       results['KE'+direction+' [W]'][0]]})
            newterms.update({'Utot'+direction+'[W]':
                                  [newterms['uHydro'+direction+' [W]'][0]+
                                    results['uB'+direction+' [W]'][0]]})
            for loc in['Day', 'Flank', 'Tail', 'OpenN', 'OpenS', 'Closed']:
                newterms.update({'uHydro'+direction+loc+' [W]':
                                [results['Eth'+direction+loc+' [W]'][0]+
                                 results['KE'+direction+loc+' [W]'][0]]})
                newterms.update({'Utot'+direction+'[W]':
                                [newterms['uHydro'+direction+loc+' [W]'][0]+
                                 results['uB'+direction+loc+' [W]'][0]]})
    return newterms

def virial_post_integr(results):
    """Creates dictionary of key:value combos of existing results
    Inputs
        results(dict{str:[value]})
    Outputs
        newterms(dict{str:[value]})
    """
    newterms = {}
    #Virial volume terms
    newterms.update({'Virial Volume Total [J]':
                                          [results['Virial 2x Uk [J]'][0]+
                                            results['Virial Ub [J]'][0]]})
    #Convert from J to nT
    for newterm in newterms.copy().items():
        newterms.update(energy_to_dB(newterm))
    return newterms

def get_biotsavart_integrands(state_var):
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
        zone(Zone)- tecplot zone object used to decide which terms included
    Outputs
        biots_dict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    state, biots_dict = state_var.name, {}
    eq = tp.data.operate.execute_equation
    eq('{BioS '+state_var.name+'}=IF({'+state_var.name+'}<1,0,{dB [nT]})')
    biots_dict.update({'BioS '+state_var.name:'bioS [nT]'})
    if 'mp' in state_var.name:
        eq('{BioS_full'+state_var.name+'}=IF({r [R]}<3,0,{dB [nT]})')
        biots_dict.update({'BioS_full'+state_var.name:'bioS_full [nT]'})
    return biots_dict

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
    integrands = ['Virial Ub [J/Re^3]','Virial 2x Uk [J/Re^3]',
                  'rhoU_r [Js/Re^3]', 'Pth [J/Re^3]']
    #Debug:
    '''
    integrands = ['Virial Ub [J/Re^3]','Virial 2x Uk [J/Re^3]',
                  'rhoU_r [Js/Re^3]','uB [J/Re^3]', 'uB_dipole [J/Re^3]']
    '''

    for term in integrands:
        name = term.split(' [')[0]
        units = '['+term.split('[')[1].split('/Re^3')[0]+']'
        if name+state not in existing_variables:
            eq('{'+name+state+'}=IF({'+state+'}<1, 0, {'+term+'})')
            virialdict.update({name+state:name+' '+units})
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
    integrands = ['uB [J/Re^3]', 'KE [J/Re^3]', 'Pth [J/Re^3]']
    for term in integrands:
        name = term.split(' ')[0]
        if 'Pth' in term:
            eq('{Eth'+state+'}=IF({'+state+'}<1, 0, 1.5*{'+term+'})')
            energydict.update({'Eth'+state:'Eth [J]'})
        elif name+state not in existing_variables:
        #Create variable for integrand that only exists in isolated zone
            eq('{'+name+state+'}=IF({'+state+'}<1, 0, {'+term+'})')
            energydict.update({name+state:name+' [J]'})
    return energydict

def get_mobile_integrands(zone, integrands, tdelta, analysis_type):
    """Creates dict of integrands for surface motion effects
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
        tdelta(float)- timestep between output files
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    mobiledict, td, eq = {}, str(tdelta), tp.data.operate.execute_equation
    for term in integrands.items():
        if ('energy' in analysis_type) or ('rhoU_r' in term[0]):
            name = term[0].split(' [')[0]
            outputname = term[1].split(' [')[0]
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            if 'Js' in units:
                units = '[J]'
            else:
                units = '[W]'
            eq('{'+name+'_acqu}=IF({delta_volume}==1,'+
                                            '-1*{'+term[0]+'}/'+td+',0)',
                                                             zones=[zone])
            eq('{'+name+'_forf}=IF({delta_volume}==-1,'+
                                                '{'+term[0]+'}/'+td+',0)',
                                                             zones=[zone])
            eq('{'+name+'_net} ={delta_volume} * -1*{'+term[0]+'}/'+td,
                                                             zones=[zone])
            mobiledict.update({name+'_acqu':outputname+'_acqu '+units,
                            name+'_forf':outputname+'_forf '+units,
                            name+'_net':outputname+'_net '+units})
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
        kwargs.update({'do_cms':False})
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
    #Debug
    ###################################################################
    #Integral bounds modifications THIS ACCOUNTS FOR SURFACE MOTION
    if kwargs.get('do_cms', False) and (('virial' in analysis_type) or
                                        ('energy' in analysis_type)):
        mobile_terms = get_mobile_integrands(global_zone,integrands,
                                             kwargs.get('deltatime',60),
                                             analysis_type)
        if 'mp' in state_var.name:
            #Integral bounds for spatially parsing results
            mobile_terms.update(get_dft_integrands(global_zone,
                                                   mobile_terms))
            mobile_terms.update(get_open_close_integrands(global_zone,
                                                          mobile_terms))
        integrands.update(mobile_terms)
    ###################################################################
    #Evaluate integrals
    for term in integrands.items():
        results.update(calc_integral(term, global_zone))
        if kwargs.get('verbose',False):
            print(stat_var.name+term[1]+' integration done')
    #Debug
    ###################################################################
    #Non scalar integrals (empty integrands)
    if kwargs.get('doVolume', True):
        eq('{Volume '+state_var.name+'}=IF({'+state_var.name+'}<1, 0,1)')
        results.update(calc_integral(('Volume '+state_var.name,
                                      'Volume [Re^3]'), global_zone))
                                  #**{'VariableOption':'LengthAreaVolume'}))
        if kwargs.get('do_cms',False):
            results.update(calc_integral(('delta_volume','dVolume [Re^3]'),
                                         global_zone))
        if kwargs.get('verbose',False):
            print(stat_var.name+' Volume integration done')
    ###################################################################
    #Post integration manipulations
    if 'virial' in analysis_type:
        results.update(virial_post_integr(results))
    if 'energy' in analysis_type:
        results.update(energy_post_integr(results, **kwargs))
        if kwargs.get('do_cms', False):
            for direction in ['_acqu', '_forf','_net']:
                results.pop('Eth'+direction+' [W]')
                results.pop('KE'+direction+' [W]')
                for loc in['Day','Flank','Tail','OpenN','OpenS','Closed']:
                    results.pop('Eth'+direction+loc+' [W]')
                    results.pop('KE'+direction+loc+' [W]')
    return pd.DataFrame(results)

