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
from global_energetics.extract.stream_tools import get_daymapped_nightmapped
from global_energetics.extract.surface_tools import (energy_to_dB,
                                        get_open_close_integrands,
                                         get_interface_integrands,
                                               get_dft_integrands,
                                                  conditional_mod,
                                                    calc_integral)

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
    df = pd.DataFrame(results)
    #Combine MAG and HYDRO back into total energy
    uB = df[[k for k in df.keys() if ('uB' in k)and('uB_dipole'not in k)]]
    ub = df[[k for k in df.keys() if 'u_db' in k]]
    uH = df[[k for k in df.keys() if 'Hydro' in k]]
    u1_values = uB.values + uH.values #including dipole field
    u2_values = ub.values + uH.values #disturbance energy
    u1_keys = ['Utot'.join(k.split('uB')) for k in df.keys()if('uB' in k)and
                                                         ('uB_dipole'not in k)]
    u2_keys=['Utot2'.join(k.split('u_db'))for k in df.keys()if'u_db'in k]
    for k in enumerate(u1_keys):df[k[1]]=u1_values[0][k[0]]
    for k in enumerate(u2_keys):df[k[1]]=u2_values[0][k[0]]
    if kwargs.get('do_cms', False):
        #Combine 'acquisitions' and 'forfeitures' back into net
        acqus = df[[k for k in df.keys()if 'acqu' in k]]
        forfs = df[[k for k in df.keys()if 'forf' in k]]
        net_values = acqus.values+forfs.values
        net_keys = ['_net'.join(k.split('_acqu')) for k in df.keys()
                    if 'acqu' in k]
        for k in enumerate(net_keys):df[k[1]]=net_values[0][k[0]]
    for key in df.keys(): newterms[key] = [df[key].values[0]]
    return newterms

def virial_post_integr(results,**kwargs):
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
    if kwargs.get('do_cms', False):
        df = pd.DataFrame(results)
        #Combine 'acquisitions' and 'forfeitures' back into net
        acqus = df[[k for k in df.keys()if 'acqu' in k]]
        forfs = df[[k for k in df.keys()if 'forf' in k]]
        net_values = acqus.values+forfs.values
        net_keys = ['_net'.join(k.split('_acqu')) for k in df.keys()
                    if 'acqu' in k]
        for k in enumerate(net_keys):df[k[1]]=net_values[0][k[0]]
    #Convert from J to nT
    for newterm in newterms.copy().items():
        newterms.update(energy_to_dB(newterm))
    return newterms

def get_imtrack_integrands(state_var):
    """Creates dictionary of terms to be integrated for IM track analysis
    Inputs
        zone(Zone)- tecplot zone object used to decide which terms included
    Outputs
        imtrack(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    state, imtrack = state_var.name, {}
    eq = tp.data.operate.execute_equation
    existing_variables = state_var.dataset.variable_names
    #Integrands
    integrands = ['trackEth_acc [J/Re^3]','trackKE_acc [J/Re^3]',
                  'trackWth [W/Re^3]', 'trackWKE [W/Re^3]']
    for term in integrands:
        name = term.split(' [')[0]
        units = '['+term.split('[')[1].split('/Re^3')[0]+']'
        if name+state not in existing_variables:
            eq('{'+name+state+'}=IF({'+state+'}<1, 0, {'+term+'})')
            imtrack.update({name+state:name+' '+units})
    return imtrack

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
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    existing_variables = state_var.dataset.variable_names
    #source_index = str(zone.index+1)
    #future_index = str(zone.dataset.zone('future*').index+1)
    #Integrands
    #integrands = ['uB [J/Re^3]', 'KE [J/Re^3]', 'Pth [J/Re^3]']
    #integrands = ['uB [J/Re^3]','uB_dipole [J/Re^3]','u_db [J/Re^3]',
    #              'uHydro [J/Re^3]']
    integrands = ['Utot [J/Re^3]']
    for term in integrands:
        name = term.split(' ')[0]
        if 'Pth' in term:
            eq('{Eth'+state+'}=IF({'+state+'}<1, 0, 1.5*{'+term+'})')
            energydict.update({'Eth'+state:'Eth [J]'})
        elif name+state not in existing_variables:
        #Create variable for integrand that only exists in isolated zone
            eq('{'+name+state+'}=IF({'+state+'}[1]<1, 0, {'+term+'}[1])',
                                                  zones=[0],value_location=CC)
            '''
            #Add the statevariableonly version of ex.(Utot) to the FUTURE
            # dataset specifically in the cells which have already been
            # determined to be "acquired"
            #equ = ('{'+name+state+'}=IF({delta_'+state+'}[1]==1,{'+
            #       term+'}[2],0)')
            eq('{'+name+state+'}=IF({delta_'+state+'}[1]==1,{'+term+'}[2],0)',
                                                  zones=[1],value_location=CC)
            '''
            energydict.update({name+state:name+' [J]'})
    return energydict

def get_mobile_integrands(zone,state_var,integrands,**kwargs):
    """Creates dict of integrands for surface motion effects
    Inputs
        zone(Zone)- tecplot Zone
        state_var(Variable)- tecplot variable used to isolate vol in zone
        integrands(dict{str:str})- (static) in/output terms to calculate
        tdelta(float)- timestep between output files
        kwargs:
            cms_simple(bool)- don't put both acqu and forf but just net
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    tdelta = kwargs.get('tdelta',60)
    analysis_type = kwargs.get('analysis_type','')
    mobiledict, td, eq = {}, str(tdelta), tp.data.operate.execute_equation
    CC = ValueLocation.CellCentered
    dstate = 'delta_'+str(state_var.name)
    source_index = str(zone.index+1)
    future_index = str(zone.dataset.zone('future*').index+1)
    #Don't proliferate IM tracking variables
    integrand_VarOut_pairs = [(t,d) for t,d in integrands.items()
                                                       if 'track' not in t]
    for variablename,outputname in integrand_VarOut_pairs:
        if ('energy' in analysis_type) or ('rhoU_r' in variablename):
            new_variablename = variablename.split(' [')[0]
            new_outputname = outputname.split(' [')[0]
            units = '['+outputname.split('[')[1].split(']')[0]+']'
            if 'Js' in units:
                units = '[J]'
            elif 'J' in units:
                units = '[W]'
            else:
                units = '[Re^3'+units+'/s]'
            if dstate in zone.dataset.variable_names:
                #NOTE Undo naming convention previously created where volume
                # energy is only INTERIOR to state volume, the delta volume
                # (acquired specifically) is strictly OUTSIDE state volume.
                if 'J' in outputname:
                    basevar = outputname.replace('J]','J/Re^3]')
                elif 'Area' in outputname:
                    basevar = variablename

                print(basevar)
                #Assign the base variable value
                #  from CURRENT time for forfeited volume
                #  from FUTURE time for acquired volume
                #TODO values seem entirely too low!!Could be too strict arguments?
                eq('{'+new_variablename+'_acqu}='+
                        'IF({'+dstate+'}['+source_index+']>0,'+
                        '{'+dstate+'}['+source_index+']*'+
                #            '-1*({'+basevar+'}['+future_index+'])'+
                            '-1*({'+basevar+'}['+future_index+']+'+
                                '{'+basevar+'}['+source_index+'])/2'+
                                                '/'+td+',0)',zones=[zone],
                                                        value_location=CC)
                eq('{'+new_variablename+'_forf}='+
                        'IF({'+dstate+'}['+source_index+'],'+
                        '{'+dstate+'}['+source_index+']*'+
                               '({'+basevar+'}['+future_index+']+'+
                                '{'+basevar+'}['+source_index+'])/2'+
                                                '/'+td+',0)',zones=[zone],
                                                        value_location=CC)
                '''
                eq('{'+new_variablename+'_acqu}='+
                        'IF({'+dstate+'}['+source_index+']==1,'+
                            '-1*({'+basevar+'}['+future_index+'])'+
                                                '/'+td+',0)',zones=[zone],
                                                        value_location=CC)
                eq('{'+new_variablename+'_forf}='+
                        'IF({'+dstate+'}['+source_index+']==-1,'+
                               '({'+basevar+'}['+source_index+'])'+
                                                '/'+td+',0)',zones=[zone],
                                                        value_location=CC)
                '''
                mobiledict.update({new_variablename+'_acqu':
                                            new_outputname+'_acqu '+units,
                                       new_variablename+'_forf':
                                            new_outputname+'_forf '+units})
    return mobiledict

def get_lshell_integrands(zone,state_var,integrands,**kwargs):
    """Creates dict of integrands for lshell distribution of integrated
        quantities
    Inputs
        zone(Zone)- tecplot Zone
        state_var(Variable)- tecplot variable used to isolate vol in zone
        integrands(dict{str:str})- (static) in/output terms to calculate
        kwargs
            lshell_vars(array/list)- lshell cuttoff limits, lower and upper
                                 bounds with be added inclusively
                                 eg. [7,9,11] -> <7, 7-9, 9-11, >11
            split_dayNight(bool) -True
    Returns
        lshelldict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    lshelldict, aplist, eq = {}, {}, tp.data.operate.execute_equation
    lvar =['{'+name+'}' for name in zone.dataset.variable_names
                                            if 'lshell' in name.lower()][0]
    #only do Lshell splitting for limitted set of quantities
    approved = kwargs.get('lshell_vars',['test'])
    for t,d in integrands.items():
        for ap in approved:
            if ap in t: aplist.update({t:d})
    lshells = kwargs.get('lshells',[7,9,11])
    lshells.append('end')
    for term in aplist.items():
        for i,l in enumerate(lshells):
            if kwargs.get('split_dayNight',True):
                sectors=['_day','_night']
            else:
                sectors=['']
            for dayNight in sectors:
                if i==0:
                    tag='<'+str(l)
                    cond=lvar+tag
                elif l=='end':
                    tag='>'+str(lshells[i-1])
                    cond=lvar+tag
                else:
                    tag = str(lshells[i-1])+'-'+str(l)
                    cond= lvar+'<'+str(l)+'&&'+lvar+'>'+str(lshells[i-1])
                tag+=dayNight
                if 'day' in dayNight:
                    cond+='&&{X [R]}>0'
                elif 'night' in dayNight:
                    cond+='&&{X [R]}<0'
                name = term[0].split(' [')[0]+tag
                outputname = term[1].split(' [')[0]+'_l'+tag
                units = ' ['+term[1].split('[')[1].split(']')[0]+']'
                #Construct new variable and add it to the dictionary
                eq('{'+name+'}=IF('+cond+',{'+term[0]+'},0)',zones=[zone])
                lshelldict.update({name:outputname+units})
    return lshelldict

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
        analysis_type = kwargs.get('analysis_type')
    #initialize empty dictionary that will make up the results of calc
    integrands, results, eq = {}, {}, tp.data.operate.execute_equation
    global_zone = state_var.dataset.zone(0)
    #Seems to change values by ~3-10%, mostly affects differential qtys
    if False:
        plt = tp.active_frame().plot()
        plt.value_blanking.active = True
        blank = plt.value_blanking.constraint(1)
        blank.active = True
        blank.variable = state_var
        blank.comparison_operator = RelOp.EqualTo
        blank.comparison_value = 0
    ###################################################################
    #Core integral terms
    if 'virial' in analysis_type:
        integrands.update(get_virial_integrands(state_var))
    if 'energy' in analysis_type:
        integrands.update(get_energy_integrands(state_var))
    if 'biotsavart' in analysis_type:
        integrands.update(get_biotsavart_integrands(state_var))
    if 'trackIM' in analysis_type:
        integrands.update(get_imtrack_integrands(state_var))
    #integrands.update(kwargs.get('customTerms', {}))
    ###################################################################
    #Integral bounds modifications THIS ACCOUNTS FOR SURFACE MOTION
    if kwargs.get('do_cms', False) and (('virial' in analysis_type) or
                                        ('energy' in analysis_type) or
                                        (kwargs.get('customTerms',{})!={})):
        mobile_terms = get_mobile_integrands(global_zone,state_var,
                                             integrands,
                                             **kwargs)
        if kwargs.get('do_interfacing',False):
        #and'mp' not in state_var.name:
            #get_daymapped_nightmapped(global_zone,**kwargs,
            #                          state_var=state_var)
            interface_terms = get_interface_integrands(global_zone,
                                                     mobile_terms,**kwargs,
                                                       state_var=state_var)
            mobile_terms.update(interface_terms)
            '''
            daymapped_terms = conditional_mod(global_zone,mobile_terms,
                                              ['daymapped'],'DayMapped')
            nightmapped_terms = conditional_mod(global_zone,mobile_terms,
                                              ['nightmapped'],'NightMapped')
            mobile_terms.update(daymapped_terms)
            mobile_terms.update(nightmapped_terms)
            '''
        elif not kwargs.get('do_interfacing',False):
            if 'mp' in state_var.name:
                #Integral bounds for spatially parsing results
                #mobile_terms.update(get_dft_integrands(global_zone,
                #                                       mobile_terms))
                mobile_terms.update(get_open_close_integrands(global_zone,
                                                             mobile_terms))
        integrands.update(mobile_terms)
    if ('Lshell' in analysis_type) and ('closed' in state_var.name):
        integrands.update(get_lshell_integrands(global_zone,state_var,
                                                integrands,**kwargs))
    ###################################################################
    #Evaluate integrals
    if kwargs.get('verbose',False):
        print('{:<20}{:<25}{:<9}'.format('Volume','Term','Value'))
        print('{:<20}{:<25}{:<9}'.format('******','****','*****'))
    for term in integrands.items():
        results.update(calc_integral(term, global_zone))
        if kwargs.get('verbose',False):
            print('{:<20}{:<25}{:>.3}'.format(
                      state_var.name,term[1],results[term[1]][0]))
    pieces = np.sum([results[k] for k in results.keys() if 'K' in k])
    if 'mp' in state_var.name and kwargs.get('do_interfacing',False):
        pieces-=(results['Utot_acquK1 [W]'][0]+
                 results['Utot_forfK1 [W]'][0]+
                 results['Utot_acquK5 [W]'][0]+
                 results['Utot_forfK5 [W]'][0])
    print('{:<20}{:<25}{:>.3}'.format(state_var.name,'error',
               ((results['Utot_acqu [W]'][0]+
                 results['Utot_forf [W]'][0])-pieces)))
    ###################################################################
    #Non scalar integrals (empty integrands)
    if kwargs.get('doVolume', True):
        results.update(calc_integral((state_var.name,
                                          'Volume [Re^3]'),global_zone))
                                  #**{'VariableOption':'LengthAreaVolume'}))
        if kwargs.get('verbose',False):
            print('{:<20}{:<25}{:>.3}'.format(state_var.name,
                               'Volume [Re^3]',results['Volume [Re^3]'][0]))
        if kwargs.get('do_cms',False):
            if'delta_'+str(state_var.name)in state_var.dataset.variable_names:
                eq('{dVolume '+state_var.name+'}={delta_'+state_var.name+'}',
                                    value_location=ValueLocation.CellCentered)
                results.update(calc_integral(('delta_'+state_var.name,
                                          'dVolume [Re^3]'),global_zone))
            if kwargs.get('verbose',False):
                print('{:<20}{:<25}{:>.3}'.format(state_var.name,
                             'dVolume [Re^3]',results['dVolume [Re^3]'][0]))
    ###################################################################
    #Post integration manipulations
    if 'virial' in analysis_type:
        results.update(virial_post_integr(results))
    if 'energy' in analysis_type:
        results.update(energy_post_integr(results, **kwargs))
    #blank.active = False
    return pd.DataFrame(results)

