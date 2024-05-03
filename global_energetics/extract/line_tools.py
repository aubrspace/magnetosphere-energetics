#!/usr/bin/env python3
"""Functions for analyzing 1D objects from field data
"""
import numpy as np
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
from scipy.integrate import trapezoid as trap
#interpackage modules, different path if running as main to test
from global_energetics.extract.tec_tools import (get_surf_geom_variables)
#from global_energetics.extract.tec_tools import (get_surface_variables)

def get_mag_dict():
    """Function returns dictionary in the form {(str)Input:(str)Output}
    """
    return {'Br [Wb/Re^2]':'dPhidt [Wb/s]'}

def get_x(zone,**kwargs):
    """Function determines 'x' for form I = int( f(x)*dx )
    Inputs
        zone(Zone)- tecplot zone object
        kwargs:
    Returns
        xvar (np arr)- numpy array of values
    """
    #Find or create needed flux variables or integration
    if 'terminator' in zone.name:
        xvar = 'Y *'
    elif 'ocflb' in zone.name:
        eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
        get_surf_geom_variables(zone)
        #TODO make sure were not wrecking the geom variables for iono zones
        # Perp projection of the 'surface normal' to be // to the sphere surf
        #   Get normal proj to the radial direction
        eq('{nradial_x} = ({surface_normal_x}*{x_cc}+'+
                          '{surface_normal_y}*{y_cc}+'+
                          '{surface_normal_z}*{z_cc})*{x_cc}/'+
                          '({x_cc}**2+{y_cc}**2+{z_cc}**2)',
                          zones=[zone],value_location=CC)
        eq('{nradial_y} = ({surface_normal_x}*{x_cc}+'+
                          '{surface_normal_y}*{y_cc}+'+
                          '{surface_normal_z}*{z_cc})*{y_cc}/'+
                          '({x_cc}**2+{y_cc}**2+{z_cc}**2)',
                          zones=[zone],value_location=CC)
        eq('{nradial_z} = ({surface_normal_x}*{x_cc}+'+
                          '{surface_normal_y}*{y_cc}+'+
                          '{surface_normal_z}*{z_cc})*{z_cc}/'+
                          '({x_cc}**2+{y_cc}**2+{z_cc}**2)',
                          zones=[zone],value_location=CC)
        #   Subtract this part to obtain the vector parallel to radial dir
        eq('{npar_x} = {surface_normal_x}-{nradial_x}',
                          zones=[zone],value_location=CC)
        eq('{npar_y} = {surface_normal_y}-{nradial_y}',
                          zones=[zone],value_location=CC)
        eq('{npar_z} = {surface_normal_z}-{nradial_z}',
                          zones=[zone],value_location=CC)
        #   Calculate tangential velocity along this curve normal direction
        eq('{U_tnorm} = {U_txd}*{npar_x}+{U_ty}*{npar_y}+{U_tzd}*{npar_z}',
                          zones=[zone],value_location=CC)
        # Construct the 'y' value as a path length from 0-fullpath
        x = zone.values('x_cc').as_numpy_array()
        y = zone.values('y_cc').as_numpy_array()
        #   Get the ID of the subsolar (polar cap) point
        #   start at the ID for the subsolar point and go towards +Y
        #   count backwards in the other direction for the remaining points
        cell_length = zone.values('Cell Area').as_numpy_array()
        idvals = np.array([i for i in range(1,zone.num_elements+1)])
        #i_start = idvals[x==x[abs(y)<0.01].max()][0]
        i_start = 0
        id_shifted = (idvals[-1]+idvals-i_start)%idvals[-1]+1
        length = 0
        length_arr = np.zeros(len(idvals))
        for k in (idvals-1)[i_start::]:
            length += 0.5*cell_length[k]+0.5*cell_length[k-1]
            length_arr[k] = length
        #   Get the full length
        length = np.sum(cell_length)
        for k in reversed((idvals-1)[0:i_start-1]):
            length -= 0.5*cell_length[k]+0.5*cell_length[k+1]
            length_arr[k] = length
        zone.dataset.add_variable('ID',locations=CC)
        zone.dataset.add_variable(zone.name+'_length',locations=CC)
        zone.values('ID')[::] = id_shifted
        zone.values(zone.name+'_length')[::] = length_arr
        xvar = 'Cell Area'
    return zone.values(xvar).as_numpy_array()

def get_fx(zone,integrands,**kwargs):
    """Function determines 'f(x)' for form I = int( f(x)*dx )
    Inputs
        zone(Zone)- tecplot zone object
        kwargs:
    Returns
        ydict dict(integrand output:np arr)- numpy array of values
    """
    ydict, eq = {}, tp.data.operate.execute_equation
    CC = ValueLocation.CellCentered
    for inpt,outpt in integrands.items():
        #Split off the units
        base_in = inpt.split(' ')[0].replace('{','')
        u_in = inpt.split(' ')[-1].replace('}','')
        base_out = outpt.split(' ')[0].replace('{','')
        u_out = outpt.split(' ')[-1].replace('}','')
        #Find or create needed flux variables or integration
        if 'terminator' in zone.name:
            ybase = base_in+'U_txd'
            y_u = u_in.replace('/Re^2','/sRe')
            ## Net flux
            ynet =  '{'+ybase+'_net '+y_u+'}'
            eq(ynet+' = if({Status}==1||{Status}==2,'+
                           'abs({'+inpt+'})*{U_txd}/6371,0)',zones=[zone])
            ydict[base_out+'_net '+u_out] = zone.values(
                                              ybase+'_net*').as_numpy_array()
            ## Day->Night
            yday2night =  '{'+ybase+'_day2night '+y_u+'}'
            eq(yday2night+' = if({Status}==1||{Status}==2,'+
                                 'abs({'+inpt+'})*min({U_txd},0)/6371,0)',
                                 zones=[zone])
            ydict[base_out+'_day2night '+u_out] = zone.values(
                                             ybase+'_day2*').as_numpy_array()
            ## Day<-Night
            ynight2day =  '{'+ybase+'_night2day '+y_u+'}'
            eq(ynight2day+' = if({Status}==1||{Status}==2,'+
                                'abs({'+inpt+'})*max({U_txd},0)/6371,0)',
                                zones=[zone])
            ydict[base_out+'_night2day '+u_out] = zone.values(
                                           ybase+'_night2*').as_numpy_array()
        elif 'ocflb' in zone.name:
            ybase = base_in+'U_tnorm'
            y_u = u_in.replace('/Re^2','/sRe')
            ## Net flux
            ynet = '{'+ybase+'_net '+y_u+'}'
            eq(ynet+' = abs({'+inpt+'})*{U_tnorm}/6371',
                                              zones=[zone],value_location=CC)
            ydict[base_out+'_net '+u_out] = zone.values(
                                              ybase+'_net*').as_numpy_array()
            ## Injected flux
            yinj = '{'+ybase+'_injection '+y_u+'}'
            eq(yinj+' = min(0,'+ynet+')',zones=[zone],value_location=CC)
            ydict[base_out+'_injection '+u_out] = zone.values(
                                        ybase+'_injection*').as_numpy_array()
            ## Escaped flux
            yesc = '{'+ybase+'_escape '+y_u+'}'
            eq(yesc+' = max(0,'+ynet+')',zones=[zone],value_location=CC)
            ydict[base_out+'_escape '+u_out] = zone.values(
                                           ybase+'_escape*').as_numpy_array()
    return ydict

def get_daynight_integrands(zone,integrands):
    daynight_dict, eq = {}, tp.data.operate.execute_equation
    for term in integrands.items():
        name = term[0].split(' [')[0]
        outputname = term[1].split(' [')[0]
        units = '['+term[1].split('[')[1].split(']')[0]+']'
        eq('{'+name+'Day}=if({daynight}>0,{'+term[0]+'},0)',zones=[zone])
        eq('{'+name+'Night}=if({daynight}<0,{'+term[0]+'},0)',zones=[zone])
        daynight_dict.update({name+'Day':outputname+'Day '+units,
                              name+'Night':outputname+'Night '+units})
    return daynight_dict

def line_analysis(zone, **kwargs):
    """Function to calculate fluxes through 1D objects in the magnetosphere
    Inputs
        zone(Zone)- tecplot zone object
        kwargs:
            analysis_type(str)- 'energy', determines which terms to include
            customTerms(dict{str:str})- any one-off integrations
    Outputs
        results- integrated flux across the 1D line/curve/path
    """
    if'analysis_type' in kwargs: analysis_type = kwargs.pop('analysis_type')
    else: analysis_type==''
    #initialize empty dictionary that will make up the results of calc
    integrands, results = {}, {}
    ###################################################################
    ## Set integrands
    if 'mag' in analysis_type:
        integrands.update(get_mag_dict())
    if True:
        integrands.update(get_daynight_integrands(zone, integrands))
    integrands.update(kwargs.get('customTerms', {}))
    ###################################################################
    ## Setup integrals
    x = get_x(zone,**kwargs)
    ydict = get_fx(zone,integrands,**kwargs)
    ###################################################################
    ## Evaluate integrals
    if kwargs.get('verbose',False):
        print('{:<30}{:<35}{:<9}'.format('Line','Term','Value'))
        print('{:<30}{:<35}{:<9}'.format('****','****','*****'))
    for name,y in ydict.items():
        if 'terminator' in zone.name:
            results[name] = [trap(y,x)]
        elif 'ocflb' in zone.name:
            results[name] = [np.sum(y*x)]
        if kwargs.get('verbose',False):
            print('{:<30}{:<35}{:>.3}'.format(zone.name,name,results[name][0]))
    ###################################################################

    return pd.DataFrame(results)
