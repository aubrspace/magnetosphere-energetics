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
            eq('{'+name+'Closed}=IF(Round({Status})==3,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenN}=IF(Round({Status})==2,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            eq('{'+name+'OpenS}=IF(Round({Status})==1,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            openClose_dict.update({name+'Closed':outputname+'Closed '+units,
                                  name+'OpenN':outputname+'OpenN ' +units,
                                  name+'OpenS':outputname+'OpenS ' +units})
    return openClose_dict

def conditional_mod(integrands,conditions,savename,**kwargs):
    """Constructer function for common integrand modifications
    Inputs
        integrands(dict{str:str})- (static) in/output terms to calculate
        conditions (list[str,str,...])- keys for conditions will be AND
        savename (str)- name for output ie:'Flank','AOP','Downtail_lobes'
    Outputs
        interfaces(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    mods, eq = {}, tp.data.operate.execute_equation
    #Condition options: 'open', 'closed', 'tail','on_innerbound', '<L7','>L7'
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        units = '['+term[1].split('[')[1].split(']')[0]+']'
        new_eq = '{'+name+savename+'} = IF('
        if ('open' in conditions) or ('closed' in conditions):
            if (('not' in conditions) and ('open' in conditions)) or(
                                                      'closed' in conditions):
                new_eq+='({Status}==3) &&'#closed
            else:
                new_eq+='({Status}==2 || {Status}==1) &&'#open
        if 'tail' in conditions:
            if 'not' in conditions:
                new_eq+='({Tail}==0) &&'
            else:
                new_eq+='({Tail}==1) &&'
        if 'on_innerbound' in conditions:
            if 'not' in conditions:
                new_eq+='({r [R]}=='+str(kwargs.get('inner_r',3))+') &&'
            else:
                new_eq+='({r [R]}!='+str(kwargs.get('inner_r',3))+') &&'
        if 'L7' in conditions:
            if '<' in conditions:
                new_eq+='({Lshell}<7) &&'
            elif '>' in conditions:
                new_eq+='({Lshell}>7) &&'
            elif '=' in conditions:
                new_eq+='({Lshell}==7) &&'
        if any([c in ['open','closed','tail','on_innerbound','L7'] for
                                                           c in conditions]):
            #chop hanging && and close up condition
            new_eq = '&&'.join(new_eq.split('&&')[0:-1])+',{'+term[0]+'},0)'
            #TODO: check if the equation has already been calculated
            mods.update({name+savename:outputname+savename+units})
    return mods

def get_interface_integrands(zone,integrands,**kwargs):
    """Creates dictionary of terms to be integrated for energy analysis
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        interfaces(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    #May need to identify the 'tail' portion of the surface
    if ((('mp'in zone.name and 'inner' not in zone.name) or
        ('lobe' in zone.name) or
        ('close' in zone.name))and
     (not any(['Tail' in n for n in zone.dataset.variable_names]))):
        get_day_flank_tail(zone)

    interfaces, eq = {}, tp.data.operate.execute_equation
    #Depending on the zone we'll have different interfaces
    ##Magnetopause
    if ('mp' in zone.name) and ('inner' not in zone.name):
        #Flank
        interfaces.update(conditional_mod(integrands,
                                          ['open','not tail'],'Flank'))
        #Tail(lobe)
        interfaces.update(conditional_mod(integrands,
                                          ['open','tail'],'Tail_lobe'))
        #Tail(closed/NearEarthrXnLine)
        interfaces.update(conditional_mod(integrands,
                                          ['closed','tail'],'Tail_close'))
        #TODO: find and store the NearEarthNeutralLine Xloc like subsolar
        #Dayside
        interfaces.update(conditional_mod(integrands,
                                          ['closed','not tail'],'Dayside'))
    ##InnerBoundary
    if 'inner' in zone.name:
        #Poles
        interfaces.update(conditional_mod(integrands,['open'],'Poles'))
        #MidLatitude
        interfaces.update(conditional_mod(integrands,['L>7'],'MidLat'))
        #LowLatitude
        interfaces.update(conditional_mod(integrands,['L<7'],'LowLat'))
    ##Lobes
    if 'lobe' in zone.name:
        #Flank- but not really bc it's hard to infer
        #Poles
        interfaces.update(conditional_mod(integrands,
                    ['on_innerbound'],'Poles',inner_r=kwargs.get('inner_r',3)))
        #Tail(lobe)
        interfaces.update(conditional_mod(integrands,['tail'],'Tail_lobe'))
        #AuroralOvalProjection- Very hard to infer, will save for post
    ##Closed
    if 'close' in zone.name:
        #Dayside- again letting magnetopause lead here
        #L7
        interfaces.update(conditional_mod(integrands,['L=7'],'L7'))
        #AuroralOvalProjection- skipped
        #MidLatitude
        interfaces.update(conditional_mod(integrands,
                   ['on_innerbound'],'MidLat',inner_r=kwargs.get('inner_r',3)))
        #Tail(closed/NearEarthrXnLine)
        interfaces.update(conditional_mod(integrands,['tail'],'Tail_close'))
    ##RingCurrent
    if 'rc' in zone.name:
        #LowLatitude
        interfaces.update(conditional_mod(integrands,
                   ['on_innerbound'],'LowLat',inner_r=kwargs.get('inner_r',3)))
        #L7
        interfaces.update(conditional_mod(integrands,
               ['not on_innerbound'],'LowLat',inner_r=kwargs.get('inner_r',3)))
    from IPython import embed; embed()
    time.sleep(3)

    return interfaces


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
    flux_suffixes = ['_escape','_injection']#net will be calculated in post
    units = ' [W/Re^2]'
    postunits = ' [W]'
    energy_dict = {}
    for direction in flux_suffixes:
        energy_dict.update({'ExB'+direction+units:'ExB'+direction+postunits,
                            'P0'+direction+units:'P0'+direction+postunits})
        #Total flux K calculated in post
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

    ###################################################################
    # These integrals are like sandwhiches, pick and chose what to
    #  include from each category:
    #       Core: primary integrands (turkey club or meatball sub)
    #       SpatialMods: proliferates core results (6" or footlong)
    #       Nonscalars: usually just surface area (lettuce)
    #       customTerms: anything else? see 'equations' for options!
    ###################################################################
    """
    #TODO: 1.Develop a way to tell while you're here if a certain interface is
    #         already covered.
    #      2.Add Downtail/NEXL, AOP, Poles, Midlat dictionaries for integrands
    #      3.Put subzone specific if's for adding to integrands
    if'analysis_type' in kwargs: analysis_type = kwargs.pop('analysis_type')
    #Find needed surface variables for integrations
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
    ###################################################################
    #Integral bounds modifications spatially parsing results
    if kwargs.get('do_interfacing',False):
        integrands.update(get_interface_integrands(zone,integrands))
    else:
        if 'innerbound' not in zone.name and kwargs.get('doDFT',False):
            integrands.update(get_dft_integrands(zone, integrands))
        if 'innerbound' in zone.name:
            integrands.update(get_low_lat_integrands(zone, integrands))
        integrands.update(get_open_close_integrands(zone, integrands))
    ###################################################################
    #Evaluate integrals
    if 'rc' in zone.name: from IPython import embed; embed()
    for term in integrands.items():
        results.update(calc_integral(term, zone))
        if kwargs.get('verbose',False):
            print(zone.name+term[1]+' integration done')
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
