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

def post_proc_interface(results,**kwargs):
    """Interface matching section of post processing to save integrations
    Inputs
        results ()
    Returns
        results (MODIFIED)
    """
    flank = pd.DataFrame()
    tail_l = pd.DataFrame()
    tail_c = pd.DataFrame()
    day = pd.DataFrame()
    day_in = pd.DataFrame()
    poles = pd.DataFrame()
    midlat = pd.DataFrame()
    lowlat = pd.DataFrame()
    l7 = pd.DataFrame()
    #Find the non-empty interfaces
    for name, df in results.items():
        if flank.empty:flank=df[[k for k in df.keys()if'Flank'in k]].copy()
        if tail_l.empty:
            tail_l=df[[k for k in df.keys()if'Tail_lobe'in k]].copy()
        if tail_c.empty:
            tail_c=df[[k for k in df.keys()if'Tail_close'in k]].copy()
        if day.empty: day=df[[k for k in df.keys() if'Dayside_reg'in k]].copy()
        if day_in.empty:
            day_in = df[[k for k in df.keys() if 'Dayside_inner' in k]].copy()
        if poles.empty:poles=df[[k for k in df.keys() if'Poles'in k]].copy()
        if midlat.empty:
            midlat =df[[k for k in df.keys()if'MidLat'in k]].copy()
        if lowlat.empty:
            lowlat=df[[k for k in df.keys()if'LowLat'in k]].copy()
        if l7.empty:
            l7=df[[k for k in df.keys()if'L7'in k]].copy()
    #Fill the empty interfaces with opposite non-empty copy
    #   CAVEAT: for subzone surfaces (dayside,flank,tail)
    #           the surfaces are identical for the magnetopause
    #           subsurface and the subvolume surface interface
    #           so these will NOT be multiplied by -1
    for name, df in results.items():
        #Magnetopause: Flank+Tail_l+Tail_c+Dayside+Dayside_inner
        if ('mp' in name) and ('inner' not in name):
            if df[[k for k in df.keys() if 'Flank' in k]].empty:
                for k in flank.keys(): df[k]=flank[k]
            if df[[k for k in df.keys() if 'Tail_lobe'in k]].empty:
                for k in tail_l.keys(): df[k]=tail_l[k]
            if df[[k for k in df.keys() if'Tail_close'in k]].empty:
                for k in tail_c.keys(): df[k]=tail_c[k]
            if df[[k for k in df.keys() if'Dayside_reg'in k]].empty:
                for k in day.keys(): df[k]=day[k]
            if df[[k for k in df.keys() if'Dayside_inner'in k]].empty:
                for k in day_in.keys(): df[k]=day_in[k]
        ##InnerBoundary: Poles+MidLat+LowLat
        if 'inner' in name:
            if df[[k for k in df.keys() if 'Poles' in k]].empty:
                for k in poles.keys(): df[k]=poles[k]*-1
            if df[[k for k in df.keys() if 'MidLat' in k]].empty:
                for k in midlat.keys(): df[k]=midlat[k]*-1
            if df[[k for k in df.keys() if 'LowLat' in k]].empty:
                for k in lowlat.keys(): df[k]=lowlat[k]*-1
        ##Lobes: Flank+Poles+Tail_l+PlasmaSheetBoundaryLayer
        if 'lobe' in name:
            if df[[k for k in df.keys() if 'Flank' in k]].empty:
                for k in flank.keys(): df[k]=flank[k]
            if df[[k for k in df.keys() if 'Poles' in k]].empty:
                for k in poles.keys(): df[k]=poles[k]*-1
            if df[[k for k in df.keys() if 'Tail_lobe' in k]].empty:
                for k in tail_l.keys(): df[k]=tail_l[k]
        ##Closed: Dayside+L7+PSB+MidLat+Tail_c
        if 'close' in name:
            if df[[k for k in df.keys() if 'Dayside_reg' in k]].empty:
                for k in day.keys(): df[k]=day[k]
            if df[[k for k in df.keys() if 'L7' in k]].empty:
                for k in l7.keys(): df[k]=l7[k]*-1
            if df[[k for k in df.keys() if 'MidLat' in k]].empty:
                for k in midlat.keys(): df[k]=midlat[k]*-1
            if df[[k for k in df.keys()if'Tail_close' in k]].empty:
                for k in tail_c.keys(): df[k]=tail_c[k]
        ##RingCurrent: LowLat+L7
        if 'rc' in name:
            if df[[k for k in df.keys() if 'LowLat' in k]].empty:
                for k in lowlat.keys(): df[k]=lowlat[k]*-1
            if df[[k for k in df.keys() if 'L7' in k]].empty:
                for k in l7.keys(): df[k]=l7[k]*-1
            if df[[k for k in df.keys() if'Dayside_inner'in k]].empty:
                for k in day_in.keys(): df[k]=day_in[k]
        ##Reverse 'injection' 'escape' to stay with consistent conventions
        true_esc = [(k,v.values[0]) for k,v in df.items()
                    if ('injection' in k) and (v.values[0]>0)]
        true_inj = [(k,v.values[0]) for k,v in df.items()
                    if ('escape' in k) and (v.values[0]<0)]
        for n,(kinj,vesc) in enumerate(true_esc):
            df[kinj] = true_inj[n][1]
            df[true_inj[n][0]] = vesc
    ##Finally, calculate PSB
    for name, df in results.items():
        if ('lobe' in name) or ('close' in name) or ('rc' in name):
            whole_keys=[k for k in df.keys()if ('_injection 'in k
                                                or  '_escape 'in k
                                                    or  '_net 'in k
                                                      or 'TestArea ' in k)]
            for k in whole_keys:
                units = '['+k.split('[')[1].split(']')[0]+']'
                psb = k.split(' [')[0]+'PSB '+units
                fl = k.split(' [')[0]+'Flank '+units
                pl = k.split(' [')[0]+'Poles '+units
                tl_l = k.split(' [')[0]+'Tail_lobe '+units
                tl_c = k.split(' [')[0]+'Tail_close '+units
                dy = k.split(' [')[0]+'Dayside_reg '+units
                dyi = k.split(' [')[0]+'Dayside_inner '+units
                l_7 = k.split(' [')[0]+'L7 '+units
                ml = k.split(' [')[0]+'MidLat '+units
                ll = k.split(' [')[0]+'LowLat '+units
                if 'lobe' in name:
                    df[psb] = (df[k]-df[fl]-df[pl]-df[tl_l])
                elif 'close' in name:
                    df[psb] = (df[k]-df[dy]-df[l_7]-df[ml]-df[tl_c])
                elif 'rc' in name:
                    df[psb] = (df[k]-df[ll]-df[l_7]-df[dyi])
    return results

def post_proc(results,**kwargs):
    """Simple post integration processing, especially for summed terms and
        other sides of interface values
    Inputs
        results ()
        kwargs:
            do_interfacing (bool)
    Returns
        results (MODIFIED)
    """
    for name, df in results.items():
        #Combine 'injections' and 'escapes' back into net
        injections = df[[k for k in df.keys() if 'injection' in k]]
        escapes = df[[k for k in df.keys() if 'escape' in k]]
        net_values = injections.values+escapes.values
        net_keys = ['_net'.join(k.split('_injection')) for k in df.keys()
                    if 'injection' in k]
        for k in enumerate(net_keys):df[k[1]]=net_values[0][k[0]]

        #Combine P0 and ExB into K (total flux)
        p0 = df[[k for k in df.keys() if 'P0' in k]]
        ExB = df[[k for k in df.keys() if 'ExB' in k]]
        K_values = p0.values+ExB.values
        K_keys = ['K'.join(k.split('P0')) for k in df.keys()if 'P0' in k]
        for k in enumerate(K_keys):df[k[1]]=K_values[0][k[0]]

    if kwargs.get('do_interfacing',False):
        #Combine north and south lobes into single 'lobes'
        #North
        if 'ms_nlobe_surface' in results.keys():
            n = results.pop('ms_nlobe_surface')
            t = n['Time [UTC]']
        elif 'ms_slobe_surface' in results.keys():
            n = pd.DataFrame(columns=results['ms_slobe_surface'].keys())
            t = results['ms_slobe_surface']['Time [UTC]']
        else:
            results = post_proc_interface(results)
            return results
        #South
        if 'ms_slobe_surface' in results.keys():
            s = results.pop('ms_slobe_surface')
        elif 'ms_nlobe_surface' in results.keys():
            s = pd.DataFrame(columns=results['ms_nlobe_surface'].keys())
        lobes=n.drop(columns=['Time [UTC]'])+s.drop(columns=['Time [UTC]'])
        lobes['Time [UTC]'] = t
        results.update({'ms_lobes_surface':lobes})
        results = post_proc_interface(results)
    return results

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

def get_low_lat_integrands(zone, integrands, **kwargs):
    """Creates dictionary of terms to integrated for virial of inner
        surface to indicate portion found at low latitude
    Inputs
        zone(Zone)- tecplot Zone
        integrands(dict{str:str})- (static) in/output terms to calculate
    Outputs
        mobiledict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    lowlat_dict, eq = {}, tp.data.operate.execute_equation
    #LowLatitude
    lowlat_dict.update(conditional_mod(zone,integrands,['<L7'],'LowLat',
                                          L=kwargs.get('lshelllim')))
    '''
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        if not any([n in name for n in ['Open','Closed']]):
            units = '['+term[1].split('[')[1].split(']')[0]+']'
            eq('{'+name+'Lowlat}=IF({Lshell}<7,'+
                                           '{'+term[0]+'},0)',zones=[zone])
            lowlat_dict.update({name+'Lowlat':outputname+'Lowlat '+units})
    '''
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

def conditional_mod(zone,integrands,conditions,modname,**kwargs):
    """Constructer function for common integrand modifications
    Inputs
        integrands(dict{str:str})- (static) in/output terms to calculate
        conditions (list[str,str,...])- keys for conditions will be AND
        modname (str)- name for output ie:'Flank','PSB','Downtail_lobes'
    Outputs
        interfaces(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    mods, eq = {}, tp.data.operate.execute_equation
    variables = tp.active_frame().dataset.variable_names
    #Condition options:'open','closed','tail','on_innerbound','<L7','>L7'

    #Check that this interface hasn't already been done by another sz
    #if [i.split(' ')[0] for i in integrands][0]+modname in variables:
    #    return mods
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        units = ' ['+term[1].split('[')[1].split(']')[0]+']'
        new_eq = '{'+name+modname+'} = IF('
        if ('open' in conditions) or ('closed' in conditions):
            if ('not open' in conditions) or('closed' in conditions):
                new_eq+='({Status}==3) &&'#closed
            else:
                new_eq+='({Status}==2 || {Status}==1) &&'#open
        if any(['tail' in c for c in conditions]):
            if 'not tail' in conditions:
                new_eq+='({Tail}==0) &&'
            else:
                new_eq+='({Tail}==1) &&'
        if any(['on_innerbound' in c for c in conditions]):
            if 'not on_innerbound' in conditions:
                new_eq+=['(abs({r [R]}-'+str(kwargs.get('inner_r',3))+
                                          ')>{Cell Size [Re]}*0.75) &&'][0]
            else:
                new_eq+=['(abs({r [R]}-'+str(kwargs.get('inner_r',3))+
                                          ')<{Cell Size [Re]}*0.75) &&'][0]
        if any(['L7' in c for c in conditions]):
            if '<L7' in conditions:
                new_eq+='({Lshell}<'+str(kwargs.get('L',7))+') &&'
            elif '>L7' in conditions:
                new_eq+='({Lshell}>'+str(kwargs.get('L',7))+') &&'
            elif '=L7' in conditions:
                new_eq+=['(abs({Lshell}-'+str(kwargs.get('L',7))+')<'+
                                              '{Cell Size [Re]}*1)&&'][0]
                #NOTE 0.75*cell size worked for everyone but L7,
                #       which was optimized at 1*cell size
        if any([a in c for c in conditions for a in
                           ['open','closed','tail','on_innerbound','L7']]):
        #if any([c in ['open','closed','tail','on_innerbound','L7'] for
        #                                                 c in conditions]):
            #chop hanging && and close up condition
            new_eq='&&'.join(new_eq.split('&&')[0:-1])+',{'+term[0]+'},0)'
            try:
                eq(new_eq,zones=[zone])
                mods.update({name+modname:outputname+modname+units})
            except TecplotLogicError:
                print('Equation eval failed!\n',new_eq,'\n')
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
    if (('mp'in zone.name and 'inner' not in zone.name) or
        ('lobe' in zone.name) or ('close' in zone.name)):
        get_day_flank_tail(zone)

    interfaces, test_i = {}, [i.split(' ')[0] for i in integrands][0]
    variables = zone.dataset.variable_names
    #Depending on the zone we'll have different interfaces
    ##Magnetopause
    if ('mp' in zone.name) and ('inner' not in zone.name):
        #Flank
        interfaces.update(conditional_mod(zone,integrands,
                                          ['open','not tail'],'Flank'))
        #Tail(lobe)
        interfaces.update(conditional_mod(zone,integrands,
                                          ['open','tail'],'Tail_lobe'))
        #Tail(closed/NearEarthXLine)
        interfaces.update(conditional_mod(zone,integrands,
                                          ['closed','tail'],'Tail_close'))
        #Dayside
        interfaces.update(conditional_mod(zone,integrands,
                                    ['closed','not tail','>L7'],'Dayside_reg',
                                          L=kwargs.get('lshelllim',7)))
        #Dayside_Inner- special case when closed pushed into 'ring current'
        interfaces.update(conditional_mod(zone,integrands,
                              ['closed','not tail','<L7'],'Dayside_inner',
                                          L=kwargs.get('lshelllim',7)))
    ##InnerBoundary
    if 'inner' in zone.name:
        #Poles
        interfaces.update(conditional_mod(zone,integrands,['open'],'Poles'))
        #MidLatitude
        interfaces.update(conditional_mod(zone,integrands,
                                          ['>L7','closed'],'MidLat',
                                          L=kwargs.get('lshelllim',7)))
        #LowLatitude
        interfaces.update(conditional_mod(zone,integrands,['<L7'],'LowLat',
                                          L=kwargs.get('lshelllim',7)))
    ##Lobes
    if 'lobe' in zone.name:
        #Flank- but not really bc it's hard to infer
        #Poles
        interfaces.update(conditional_mod(zone,integrands,
                ['on_innerbound'],'Poles',inner_r=kwargs.get('inner_r',3)))
        #Tail(lobe)
        interfaces.update(conditional_mod(zone,integrands,['tail'],
                                          'Tail_lobe'))
        #PlasmaSheetBoundaryLayer- Very hard to infer, will save for post
    ##Closed
    if 'close' in zone.name:
        #Dayside- again letting magnetopause lead here
        #L7
        interfaces.update(conditional_mod(zone,integrands,['=L7'],'L7',
                                          L=kwargs.get('lshelllim',7)))
        #PlasmaSheetBoundaryLayer- skipped
        #MidLatitude
        interfaces.update(conditional_mod(zone,integrands,
               ['on_innerbound'],'MidLat',inner_r=kwargs.get('inner_r',3)))
        #Tail(closed/NearEarthXLine)
        interfaces.update(conditional_mod(zone,integrands,['tail'],
                                          'Tail_close'))
    ##RingCurrent
    if 'rc' in zone.name:
        #Dayside_Inner- will be 0 at many points as interface only occurs when
        #               dayside closed field is < L7
        #               *but yet again, it's much easier to let MP find this
        #PlasmaSheetBoundaryLayer- same as above, only sometimes, *skipped
        #LowLatitude
        interfaces.update(conditional_mod(zone,integrands,
               ['on_innerbound'],'LowLat',inner_r=kwargs.get('inner_r',3)))
        #L7
        interfaces.update(conditional_mod(zone,integrands,
           ['not on_innerbound','=L7'],'L7',inner_r=kwargs.get('inner_r',3),
                                              L=kwargs.get('lshelllim',7)))
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

def get_mass_dict():
    """Creates dictionary of terms to be integrated for virial analysis
    Inputs
    Outputs
        energy_dict(dict{str:str})- dictionary of terms w/ pre:post integral
    """
    flux_suffixes = ['_escape','_injection']#net will be calculated in post
    units = ' [kg/s/Re^2]'; postunits = ' [kg/s]'; mass_dict = {}
    for direction in flux_suffixes:
        mass_dict.update({'RhoU'+direction+units:'M'+direction+postunits})
    return mass_dict

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
    if 'mass' in analysis_type:
        integrands.update(get_mass_dict())
    integrands.update(kwargs.get('customTerms', {}))
    ###################################################################
    #Integral bounds modifications spatially parsing results
    if kwargs.get('do_interfacing',False):
        integrands.update(get_interface_integrands(zone,integrands,**kwargs))
    else:
        if 'innerbound' not in zone.name and kwargs.get('doDFT',False):
            integrands.update(get_dft_integrands(zone, integrands))
        if 'innerbound' in zone.name:
            integrands.update(get_low_lat_integrands(zone, integrands,
                                                     **kwargs))
        #integrands.update(get_open_close_integrands(zone, integrands))
    ###################################################################
    #Evaluate integrals
    for term in integrands.items():
        results.update(calc_integral(term, zone))
        if kwargs.get('verbose',False):
            print(zone.name+term[1]+' integration done')
    if 'closed' in zone.name:
        for k in [k for k in results.keys() if 'PSB' in k]: print(results[k])
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
