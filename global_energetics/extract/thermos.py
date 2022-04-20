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
import pandas as pd
#interpackage modules
from global_energetics.makevideo import get_time, time_sort

def get_averaged_values(data,**kwargs):
    """Function
    Input
        data (GitmBin)- spacepy data object
        kwargs:
            valuekey (str)- typically 'Rho'
            alt (float)- altitude in km
            dalt (float)- delta altitude in km
    Return
        result (dict{mean(float),limits([min,max]),std(float)})
    """
    #organize kwargs
    ave_alt = str(kwargs.get('alt',210))
    altmax = kwargs.get('alt',210)+kwargs.get('dalt',20)
    altmin = kwargs.get('alt',210)-kwargs.get('dalt',20)
    #split into pieces
    lat = data['Latitude']
    lon = data['Longitude']
    alt = data['Altitude']
    value = data[kwargs.get('valuekey','Rho')]
    ##average across longitude
    df = pd.DataFrame()
    for l in enumerate(lat[0,:,0]):
        col=str(np.rad2deg(l[1]))
        avg = np.zeros(len(alt[0,l[0],:]))
        for a in enumerate(alt[0,0,:]):
            avg[a[0]] = np.mean(value[:,l[0],a[0]])
        df[col] = avg
    #use altitude (km) for index
    df.index = alt[0,0,:]/1000
    #get specific layer value at altitude
    layer = df[(df.index < altmax) & (df.index > altmin)].mean()
    result = {kwargs.get('valuekey','Rho')+'_mean'+ave_alt:layer.mean(),
              kwargs.get('valuekey','Rho')+'_max'+ave_alt:layer.max(),
              kwargs.get('valuekey','Rho')+'_min'+ave_alt:layer.min(),
              kwargs.get('valuekey','Rho')+'_std'+ave_alt:layer.std()}
    return result

def save_tofile(infile,timestamp,filetype='hdf',outputdir='localdbug/ua',
                hdfkey='ua',**values):
    """Function saves data to file
    Inputs
        infile (str)- input filename
        timestamp (datettime)
        filetype (str)- only hdf5 supported
        hdfkey (str)
        values:
            dict(list of values)- typically single valued list
    """
    df = pd.DataFrame(values,index=[timestamp])
    #output info
    outfile = '/'+infile.split('_t')[-1].split('.')[0]
    if 'hdf' in filetype:
        df.to_hdf(outputdir+outfile+'.h5', key=hdfkey)
    if 'ascii' in filetype:
        df.to_csv(outputdir+outfile+'.dat',sep=' ',index=False)



# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    from spacepy.pybats import gitm
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()
    datapath = ('/home/aubr/Code/GITM2/run/data/')
    filelist = sorted(glob.glob(datapath+'3DALL*.bin'), key=time_sort)
    for file in filelist[0:1]:
        #get timestamp
        timestamp = get_time(file)
        ##read single file
        dat = gitm.GitmBin(file)
        density = get_averaged_values(dat)
        #save data
        save_tofile(file,timestamp,**density)
