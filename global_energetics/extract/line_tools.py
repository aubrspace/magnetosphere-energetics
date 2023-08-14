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
#from global_energetics.extract.stream_tools import (get_surface_variables)

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
    return ydict

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
        results[name] = [trap(y,x)]
        if kwargs.get('verbose',False):
            print('{:<30}{:<35}{:>.3}'.format(zone.name,name,results[name][0]))
    ###################################################################

    return pd.DataFrame(results)
