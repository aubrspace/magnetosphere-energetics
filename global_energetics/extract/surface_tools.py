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
from global_energetics.extract.stream_tools import (integrate_tecplot,
                                                    get_surface_variables,
                                                    get_day_flank_tail,
                                                  get_surf_geom_variables,
                                            get_surface_velocity_estimate,
                                                    dump_to_pandas)
from global_energetics.extract.view_set import variable_blank

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
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        if ('Open' not in name) and ('Closed' not in name):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Day}={Day}*{'+term[0]+'}',zones=[zone])
            eq('{'+name+'Flank}={Flank}*{'+term[0]+'}',zones=[zone])
            eq('{'+name+'Tail}={Tail}*{'+term[0]+'}',zones=[zone])
            dft_dict.update({name+'Day':outputname+'Day '+units,
                             name+'Flank':outputname+'Flank '+units,
                             name+'Tail':outputname+'Tail '+units})
    return dft_dict

def get_low_lat_integrands(zone, integrands):
    """Creates dictionary of terms to integrated for virial of inner
        surface to indicate portion found at low latitude
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    lowlat_dict, eq = {}, tp.data.operate.execute_equation
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        if not any([n in name for n in ['Open','Closed']]):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Lowlat}=IF({Lshell}<7,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            lowlat_dict.update({name+'Lowlat':outputname+'Lowlat '+units})
    return lowlat_dict

def get_open_close_integrands(zone, integrands):
    """Creates dictionary of terms to be integrated for energy analysis
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    openClose_dict, eq = {}, tp.data.operate.execute_equation
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        if not any([n in name for n in ['Day','Flank','Tail','Lowlat']]):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Closed}=IF({Status}==3,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenN}=IF({Status}==2,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenS}=IF({Status}==1,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            openClose_dict.update({name+'Closed':outputname+'Closed '+units,
                                  name+'OpenN':outputname+'OpenN ' +units,
                                  name+'OpenS':outputname+'OpenS ' +units})
    return openClose_dict

def energy_to_dB(energy, *, conversion=-8e13):
    """Function converts energy term to magnetic perturbation term w factor
    Inputs
        energy(tuple(tuple)- (str,[float]) term to be converted
        virial_conversion(float)- -8e13, conversion factor based on dipole
    Outputs
        (dict{str:float})
    """
    return {'[nT]'.join(energy[0].split('[J]')):[energy[1][0]/conversion]}

def calc_integral(term, zone, **kwargs):
    """Calls tecplot integration for term
    Inputs
        term(tuple(str:str))- name pre:post integration
        zone(Zone)- tecplot zone object where integration is performed
        kwargs:
            VariableOption(str)- 'Scalar', alt is 'LengthAreaVolume'
    Outputs
        result(dict{str:float})
    """
    if kwargs.get('VariableOption','Scalar')=='Scalar':
        variable = zone.dataset.variable(term[0].split('[')[0]+'*')
    else:
        variable = None
    value = integrate_tecplot(variable, zone,
                      VariableOption=kwargs.get('VariableOption','Scalar'))
    result = {term[1]:[value]}
    return result

def get_energy_dict():
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
    Outputs
        energy_dict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    flux_suffixes = ['_escape','_net', '_injection']
    units = ' [W/Re^2]'
    postunits = ' [W]'
    energy_dict = {}
    for direction in flux_suffixes:
        energy_dict.update({'ExB'+direction+units:'ExB'+direction+postunits,
                            'P0'+direction+units:'P0'+direction+postunits,
                            'K'+direction+units:'K'+direction+postunits})
    return energy_dict

def get_virial_dict(zone):
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
        zone(Zone)- tecplot zone object used to decide which terms included
    Outputs
        virialdict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    virial_dict = {'virial_Fadv [J/Re^2]':'Virial Fadv [J]',
                   'virial_Floz [J/Re^2]':'Virial Floz [J]',
                   'Virial Ub [J/Re^3]':'Virial b^2 [J]',
                   'virial_surfTotal [J/Re^2]':'Virial Surface Total [J]'}
    #Debug:
    '''
    virial_dict = {'virial_Fadv [J/Re^2]':'Virial Fadv [J]',
                   'virial_Floz [J/Re^2]':'Virial Floz [J]',
                   'Virial Ub [J/Re^2]':'Virial b^2 [J]',
                   'virial_surfTotal [J/Re^2]':'Virial Surface Total [J]',
                   'example_surface [#/Re^2]':'example_surface [#]'}
    '''
    return virial_dict

def surface_analysis(zone, **kwargs):
    """Function to calculate energy flux at magnetopause surface
    Inputs
        zone(Zone)- tecplot zone object
        kwargs:
            analysis_type(str)- 'energy', determines which terms to include
            do_cms(bool)- False, moving surface integration in development
            dt(float)- time in seconds for moving surface velocity
            do_1Dsw(bool)- False, optional 1D calculation
            blank(bool)- False, tecplot Blanking used to isolate dataset
            blank_variable(str)- 'r *', tecplot Variable name
            blank_value(float)- 3, blanking values used in condition
            blank_operator(RelOp)- RelOp.LessThan, tecplot constant obj
            customTerms(dict{str:str})- any one-off integrations
    Outputs
        surface_power- power, or energy flux at the magnetopause surface
    """
    #Calculate needed surface variables for integrations
    if 'analysis_type' in kwargs:
        analysis_type = kwargs.pop('analysis_type')
    if ('innerbound' in zone.name) and (len(zone.aux_data.as_dict())==0):
        get_surf_geom_variables(zone)
    get_surface_variables(zone, analysis_type, **kwargs)
    #initialize empty dictionary that will make up the results of calc
    integrands, results, eq = {}, {}, tp.data.operate.execute_equation
    ###################################################################
    #Core integral terms
    if 'virial' in analysis_type:
        integrands.update(get_virial_dict(zone))
    if 'energy' in analysis_type:
        integrands.update(get_energy_dict())
    integrands.update(kwargs.get('customTerms', {}))
    #Debug:
    ###################################################################
    #Integral bounds modifications spatially parsing results
    if 'innerbound' not in zone.name:
        integrands.update(get_dft_integrands(zone, integrands))
    else:
        integrands.update(get_low_lat_integrands(zone, integrands))
    integrands.update(get_open_close_integrands(zone, integrands))
    ###################################################################
    #Evaluate integrals
    for term in integrands.items():
        results.update(calc_integral(term, zone))
        if kwargs.get('verbose',False):
            print(zone.name+term[1]+' integration done')
    #Debug:
    ###################################################################
    #Non scalar integrals (empty integrands)
    if kwargs.get('doSurfaceArea', True):
        results.update(calc_integral((' ','Area [Re^2]'), zone,
                        VariableOption='LengthAreaVolume'))
        if kwargs.get('verbose',False):
            print(zone.name+' Surface Area integration done')
    ###################################################################
    #Post integration manipulations
    if 'virial' in analysis_type:
        for term in [{key:pair} for (key,pair) in results.items()
                                                       if 'Virial' in key]:
            results.update(energy_to_dB([t for t in term.items()][0]))
    return pd.DataFrame(results)
