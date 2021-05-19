#/usr/bin/env python
"""module for writing and displaying progress for energetics data
"""
import sys
import os
import time
import glob
import numpy as np
import datetime as dt
import pandas as pd
import spacepy as sp
import tecplot as tp

def write_mesh(filename, zonename, timedata, mesh):
    """Function writes out 3D mesh data to hdf5 file
    Inputs
        filename- for output
        zonename- serves as key to store data under in hdf5 structure
        timedata- pandas series with time information
        mesh- pandas dataframe of mesh data
    """
    pathstring = ''
    for lvl in filename.split('/')[0:-1]:
        pathstring = pathstring+lvl+'/'
    if not os.path.exists(pathstring):
            os.system('mkdir '+pathstring)
    with pd.HDFStore(filename) as store:
        store[zonename] = mesh
        store['Time_UTC'] = timedata

def write_to_hdf(filename, zonename, *, mp_energies=None, mp_powers=None,
                                        mp_inner_powers=None,
                                        ie_energies=None, ie_powers=None,
                                        im_energies=None, im_powers=None):
    """Function writes pandas data to hdf5 file
    Inputs
        filename- for output
        zonename- serves as key to store data under in hdf5 structure
        {}_energies- volume integrated energy quantities, pandas DataFrame
        {}_powers- surface integrated power quantities, pandas DataFrame
    """
    pathstring = ''
    for lvl in filename.split('/')[0:-1]:
        pathstring = pathstring+lvl+'/'
    if not os.path.exists(pathstring):
            os.system('mkdir '+pathstring)
    energetics = pd.DataFrame()
    cols = energetics.keys()
    #Combine all dataframes that are passed
    for df in [mp_energies, mp_powers, mp_inner_powers,
               ie_energies, ie_powers,
               im_energies, im_powers]:
        if type(df) != type(None):
            cols = cols.append(df.keys())
            energetics = pd.DataFrame(columns=cols, data=[np.append(
                                      energetics.values, df.values)])
    #Remove duplicate time columns
    #TBD
    if not energetics.empty:
        #Write data to hdf5 file
        with pd.HDFStore(filename) as store:
            if any([key == '/'+zonename for key in store.keys()]):
                energetics = store[zonename].append(energetics,
                                                ignore_index=True)
            store[zonename] = energetics

def display_progress(meshfile, integralfile, zonename):
    """Function displays current status of hdf5 files
    Inputs
        meshfile- full path to meshfile
        integralfile- full path to integral file
        zonename- zonename of current addition
    """
    meshpath, integralpath = '', ''
    for lvl in meshfile.split('/')[0:-1]:
        meshpath = meshpath+lvl+'/'
    for lvl in integralfile.split('/')[0:-1]:
        integralpath = integralpath+lvl+'/'
    #Display result from this step
    result = ('Result\n'+
               '\tmeshdatafile: {}\n'.format(meshfile)+
               '\tmeshfilecount: {}\n'.format(
                                        len(glob.glob(meshpath+'*.h5'))))
    result = (result+
               '\tintegralfile: {}\n'.format(integralfile)+
               '\tzonename_added: {}\n'.format(zonename)+
               '\tintegralfilecount: {}\n'.format(
                                      len(glob.glob(integralpath+'*.h5'))))
    '''
    with pd.HDFStore(integralfile) as store:
        result = result+'\tmp_energetics:\n'
        for key in store.keys():
            result = (result+
            '\t\tkey={}\n'.format(key)+
            '\t\t\tn_values: {}\n'.format(len(store[key])))
    '''
    print('**************************************************************')
    print(result)
    print('**************************************************************')

def combine_hdfs(datapath, outputpath):
    """Function combines all .h5 files at the given datapath, cleans and
        sorts data
    Inputs
        datapath
    """
    filelist = glob.glob(datapath+'/*.h5')
    with pd.HDFStore(filelist[0]) as store:
        anykey = store.keys()[0]
        blank = pd.DataFrame(columns=store[anykey].keys())
        keylist = store.keys()
    for key in keylist:
        energetics = blank
        for hdffile in filelist:
            print(hdffile)
            with pd.HDFStore(hdffile) as store:
                if any([localkey == key for localkey in store.keys()]):
                    energetics = energetics.append(store.get(key),
                                                ignore_index=True)
        energetics = energetics.sort_values(by=['Time [UTC]'])
        energetics = energetics.reset_index(drop=True)
        print(energetics)
        with pd.HDFStore(outputpath+'/energetics.h5') as store:
            store[key] = energetics
    display_progress(outputpath+'/meshdata/*.h5',
                     outputpath+'/energetics.h5', 'Combined_zones')

if __name__ == "__main__":
    combine_hdfs('output/testoutput/energeticsdata', 'output/testoutput')
