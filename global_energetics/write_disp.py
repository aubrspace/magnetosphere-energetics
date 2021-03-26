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
                                        ie_energies=None, ie_powers=None,
                                        im_energies=None, im_powers=None):
    """Function writes pandas data to hdf5 file
    Inputs
        filename- for output
        zonename- serves as key to store data under in hdf5 structure
        {}_energies- volume integrated energy quantities, pandas DataFrame
        {}_powers- surface integrated power quantities, pandas DataFrame
    """
    energetics = pd.DataFrame()
    cols = energetics.keys()
    #Combine all dataframes that are passed
    for df in [mp_energies, mp_powers, ie_energies, ie_powers,
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
    meshpath = ''
    for lvl in meshfile.split('/')[0:-1]:
        meshpath = meshpath+lvl+'/'
    #Display result from this step
    result = ('Result\n'+
               '\tmeshdatafile: {}\n'.format(meshfile)+
               '\tmeshfilecount: {}\n'.format(
                                        len(glob.glob(meshpath+'*.h5'))))
    result = (result+
               '\tintegralfile: {}\n'.format(integralfile)+
               '\tzonename_added: {}\n'.format(zonename))
    with pd.HDFStore(integralfile) as store:
        result = result+'\tmp_energetics:\n'
        for key in store.keys():
            result = (result+
            '\t\tkey={}\n'.format(key)+
            '\t\t\tn_values: {}\n'.format(len(store[key])))
    print('**************************************************************')
    print(result)
    print('**************************************************************')

if __name__ == "__main__":
    pass
