#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
import logging as log
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
    datapath = ('/nfs/solsticedisk/tuija/starlink/IE/ionosphere/')
    filelist = sorted(glob.glob(datapath+'it*.tec'), key=time_sort)
    for file in filelist[0:1]:
        field_data = tp.data.load_tecplot(file)
        #get timestamp
        timestamp = get_time(file)
        #setup zones
        north = field_data.zone('IonN *')
        south = field_data.zone('IonS *')
        joule = field_data.variable('JouleHeat *')
        #integrate
        conversion = 6371**2*1e3 #mW/m^2*Re^2 -> W
        nint = integrate_tecplot(joule,north)*conversion
        sint = integrate_tecplot(joule,south)*conversion
        #save data
        save_tofile(file,timestamp,nJouleHeat_W=[nint],sJouleHeat_W=[sint])
