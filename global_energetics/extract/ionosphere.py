#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
#import logging as log
import os
import sys
import time
import glob
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#interpackage modules
from global_energetics.makevideo import get_time, time_sort
from global_energetics.extract.stream_tools import integrate_tecplot

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def read_idl_ascii(infile,**kwargs):
    """Function reads .idl file that is ascii formatted
    Inputs
        infile
        kwargs:
    Returns
        north,south (DataFrame)- data for north and south hemispheres
        aux (dict)- dictionary of other information
    """
    headers = ['VALUES','TIME','SIMULATION','DIPOLE']
    aux = {}
    variables = []
    i=0
    with open(infile,'r') as f:
        for line in f:
            if 'TITLE' in line:
                title = f.readline()
                i+=1
            elif 'VARIABLE' in line:
                value = f.readline()
                i+=1
                isblank = all([c==''for c in value.split('\n')[0].split(' ')])
                while not isblank:
                    name= '_'.join([c.split('\n')[0] for c in value.split(' ')
                                                    if c!='\n' and c!=''][1::])
                    variables.append(name)
                    value = f.readline()
                    i+=1
                    isblank=all([c==''for c in value.split('\n')[0].split(' ')])
            elif any([h in line for h in headers]):
                value = f.readline()
                i+=1
                isblank = all([c==''for c in value.split('\n')[0].split(' ')])
                while not isblank:
                    qty = [c for c in value.split(' ') if c!='\n' and c!=''][0]
                    name= '_'.join([c.split('\n')[0] for c in value.split(' ')
                                                    if c!='\n' and c!=''][1::])
                    if qty.isnumeric(): qty = int(qty)
                    elif isfloat(qty): qty = float(qty)
                    aux[name] = qty
                    value = f.readline()
                    i+=1
                    isblank=all([c==''for c in value.split('\n')[0].split(' ')])
            elif 'BEGIN NORTH' in line:
                i_ns, northstart = i, line
            elif 'BEGIN SOUTH' in line:
                i_ss, southstart = i, line
            i+=1
    north = pd.read_csv(infile,sep='\s+',skiprows=i_ns+1,names=variables,
                        nrows=i_ss)
    south = pd.read_csv(infile,sep='\s+',skiprows=i_ss+1,names=variables)
    return north,south,aux

def calc_shell_variables(ds, **kwargs):
    """Calculates helpful variables such as cell area (labeled as volume)
    """
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'CELLCENTERED')
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    #Generate cellcentered versions of postitional variables
    for var in ['X [R]','Y [R]','Z [R]','JouleHeat [mW/m^2]']:
        if var in zone.dataset.variable_names:
            newvar = var.split(' ')[0].lower()+'_cc'
            eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                        zones=[zone.index])

def get_ionosphere_zone(eventdt, datapath):
    """Function to find the correct ionosphere datafile and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of event time
        datapath- str path to ionosphere data
    """
    eventstring = str(eventdt)
    #parse event string (will update later)
    yr = eventstring.split('-')[0][-2::]
    mn = eventstring.split('-')[1]
    dy = eventstring.split('-')[2].split(' ')[0]
    hr = eventstring.split(' ')[1].split(':')[0]
    minute = eventstring.split(':')[1]
    datafile = 'it'+yr+mn+dy+'_'+hr+minute+'00_000.tec'
    #load file matching description
    if os.path.exists(datapath+datafile):
        ie_data = tp.data.load_tecplot(datapath+datafile)
        north_iezone = ie_data.zone('IonN *')
        south_iezone = ie_data.zone('IonS *')
        return north_iezone, south_iezone
    else:
        print('no ionosphere data found!')
        return None, None

def get_ionosphere(field_data, *, show=False,
                   comp1=True, comp2=True, comp3=True,
                   local_integrate=False):
    """Function that finds, plots and calculates energetics on the
        ionosphere surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        ndatafile, sdatafile- ionosphere data, assumes .csv files
        show- boolean for display
        comp1, comp2- boolean for which comp to include
        local_integrate- boolean for integration in python or pass to tec
    """
    #get date and time info from datafiles name

    #Read .csv files into dataframe objects

    #Call calc_components for each hemisphere

    if show:
        #display component values
        pass

    if local_integrate:
        #integrate in python using pandas
        comp1var, comp2var, comp3var = [],[],[]
        compvars = [comp1var, comp2var, comp3var]
        compstrings = ['comp1', 'comp2', 'comp3']
        for comp in enumerate([comp1, comp2, comp3]):
            if comp[1]:
                compvar[comp[0]] = (northdf[compstrings[comp[0]]].sum()+
                                   southdf[compstrings[comp[0]]].sum())[0]
        [comp1var, comp2var, comp3var] = compvars
    else:
        #pass data to tecplot
        load_ionosphere_tecplot(northdf, southdf)

def save_tofile(infile,timestamp,filetype='hdf',outputdir='localdbug/ie',
                hdfkey='ie',**values):
    """Function saves data to file
    Inputs
        infile (str)- input filename
        timestamp (datettime)
        filetype (str)- only hdf5 supported
        hdfkey (str)
        values:
            dict(list of values)- typically single valued list
    """
    df = pd.DataFrame(values)
    df.index = [timestamp]
    #output info
    outfile = '/'+infile.split('/')[-1].split('it')[-1].split('.')[0]
    if 'hdf' in filetype:
        df.to_hdf(outputdir+outfile+'.h5', key='ie')
    if 'ascii' in filetype:
        df.to_csv(outputdir+outfile+'.dat',sep=' ',index=False)



# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()
    #datapath = ('/Users/ngpdl/Code/swmf-energetics/localdbug/ie/')
    datapath = ('/nfs/solsticedisk/tuija/ccmc/2019-05-13/run/IE/ionosphere/')
    filelist = sorted(glob.glob(datapath+'it*.idl'), key=time_sort)
    for file in filelist[0:1]:
        #get timestamp
        timestamp = get_time(file)
        #read data
        north,south,aux = read_idl_ascii(file)
        #Integrate
        #save data
        #save_tofile(file,timestamp,
        #            nJouleHeat_W=[njoul],sJouleHeat_W=[sjoul],
        #            nEFlux_W=[nenergy],sEFlux_W=[senergy])
