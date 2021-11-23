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
                                            get_surface_velocity_estimate,
                                                    dump_to_pandas)
from global_energetics.extract.view_set import variable_blank

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
    units = ' [W]'
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
    virial_dict = {'virial_scalarPth':'Virial ScalarPth [J]',
                   'virial_scalaruB':'Virial ScalarPmag [J]',
                   'virial_scalaruB_dipole':'Virial ScalarPdip [J]',
                   'virial_advect1':'Virial Advection d/dt [J]',
                   'virial_advect2':'Virial Advection 2 [J]',
                   'virial_MagB_':'Virial B Stress [J]',
                   'virial_MagBd':'Virial Bd Stress [J]',
                   'virial_BBd':'Virial BBd Stress [J]',
                   'virial_surfTotal':'Virial Surface Total [J]'}
    if 'innerbound' in zone.name:
        virial_dict.update({'virial_scalardelta_uB':
                                                  'Virial ScalarPbin [J]'})
        virial_dict.update({'virial_Magb':'Virial b Stress [J]'})
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
    get_surface_variables(zone, analysis_type, **kwargs)
    #initialize empty dictionary that will make up the results of calc
    results = {}
    if kwargs.get('do_1Dsw', False):
        ##Different prefixes allow for calculation of surface fluxes using 
        #   multiple sets of flowfield variables (denoted by the prefix)
        prefix = '1D'
        #OUTDATED! Need to update to be compatible with refactoring
    if kwargs.get('do_cms', False):
        pass #currently all handled in volume integrations
    ###################################################################
    if 'virial' in analysis_type:
        virialdict = get_virial_dict(zone)
        for virialterm in virialdict.items():
            results.update(calc_integral(virialterm, zone))
            results.update(energy_to_dB((virialterm[1],
                                         results[virialterm[1]])))
            if kwargs.get('verbose',False):
                print(zone.name+' '+virialterm[1]+' integration done')
    ###################################################################
    if 'energy' in analysis_type:
        energydict = get_energy_dict()
        for energyterm in energydict.items():
            results.update(calc_integral(energyterm, zone))
            if kwargs.get('verbose',False):
                print(zone.name+' '+energyterm[1]+' integration done')
    ###################################################################
    if kwargs.get('doSurfaceArea', True):
        results.update(calc_integral((' ','Area [Re^2]'), zone,
                        VariableOption='LengthAreaVolume'))
        if kwargs.get('verbose',False):
            print(zone.name+' Surface Area integration done')
    ###################################################################
    if len(kwargs.get('customTerms', {}))!=0:
        for term in kwargs.get('customTerms', {}):
            results.update(calc_integral(term, zone))
            if kwargs.get('verbose',False):
                print(zone.name+' '+term[1]+' integration done')
    ###################################################################
    #Package in pandas DataFrame
    surface_power = pd.DataFrame(results)
    #Turn off blanking
    frame = kwargs.get('frame', tp.active_frame())
    frame.plot().value_blanking.active = False
    return surface_power
