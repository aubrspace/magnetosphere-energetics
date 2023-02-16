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
#import spacepy as sp
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
    os.makedirs(pathstring, exist_ok=True)
    with pd.HDFStore(filename) as store:
        store[zonename] = mesh
        store['Time_UTC'] = timedata

def write_to_hdf(filename, data):
    """Function writes pandas data to hdf5 file
    Inputs
        filename- for output
        data (Dict of DataFrames)- dictionary with name and associated df
    """
    pathstring = ''
    for lvl in filename.split('/')[0:-1]:
        pathstring = pathstring+lvl+'/'
    os.makedirs(pathstring, exist_ok=True)
    #Combine all dataframes that are passed
    with pd.HDFStore(filename) as store:
        for key,df in data.items():
            if type(df) != type(pd.DataFrame()):
                raise TypeError ('write_to_hdf expects Dict of DataFrames')
            #if '/'+key in store.keys():
            #    updated_df = pd.concat([store[key],df],ignore_index=True)
            #store['/'+key] = data[key]
            store[key] = data[key]

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
               '\tzones_added: {}\n'.format(zonename)+
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

def merge_hdfs(datapath, outputpath, *, combo_name='energetics.h5',
                                          progress=True):
    """Function combines all .h5 files with different headers and same time
    Inputs
        datapath
    """
    filelist = glob.glob(datapath+'/*.h5')
    keylength=0
    for file in filelist:
        with pd.HDFStore(file) as store:
            anykey = store.keys()[0]
            blank = pd.DataFrame(columns=store[anykey].keys())
            temp_keylist = store.keys()
        if len(temp_keylist)>keylength:
            keylist = temp_keylist
            keylength = len(keylist)
    for key in keylist:
        energetics = blank
        for hdffile in filelist:
            print(hdffile)
            with pd.HDFStore(hdffile) as store:
                if any([localkey == key for localkey in store.keys()]):
                    #for localkey in energetics.keys():
                    #    if all(energetics[localkey].isna()):
                    #        if localkey in store.get(key).keys():
                    #            energetics[localkey]=store.get(key)[localkey]
                    for localkey in store.get(key).keys():
                        energetics[localkey] = store.get(key)[localkey]
                    #energetics = energetics.append(store.get(key),
                    #                            ignore_index=False)
        timekey=[key for key in energetics.keys()if'time' in key.lower()][0]
        energetics = energetics.sort_values(by=[timekey])
        energetics = energetics.reset_index(drop=True)
        print(energetics)
        with pd.HDFStore(outputpath+'/'+combo_name) as store:
            store[key] = energetics
    '''
    for key in keylist:
        energetics = blank
        for hdffile in filelist:
            print(hdffile)
            with pd.HDFStore(hdffile) as store:
                if any([localkey == key for localkey in store.keys()]):
                    energetics = energetics.append(store.get(key),
                                                ignore_index=True)
        timekey=[key for key in energetics.keys()if'time' in key.lower()][0]
        energetics = energetics.sort_values(by=[timekey])
        energetics = energetics.reset_index(drop=True)
        print(energetics)
        with pd.HDFStore(outputpath+'/'+combo_name) as store:
            store[key] = energetics
    if progress:
        display_progress(outputpath+'/meshdata/*.h5',
                        outputpath+'/energetics.h5', 'Combined_zones')
    '''

def combine_hdfs2(datapath, outputpath, *, combo_name='energetics.h5',
                                           progress=True):
    filelist = glob.glob(os.path.join(datapath,'*.h5'))
    output_data = {}
    for i,infile in enumerate(filelist):
        if progress:
            print('{:>4}/{:<4}\t{:<25}'.format(i+1,len(filelist),infile))
        with pd.HDFStore(infile) as input_data:
            for key in input_data.keys():
                input_dataframe = input_data[key]
                if key in output_data:
                    output_data[key] = pd.concat([input_dataframe,
                                                  output_data[key]],
                                                 ignore_index=True)
                else:
                    output_data[key] = input_dataframe
    with pd.HDFStore(outputpath+'/'+combo_name) as output:
       for key in output_data.keys():
            output[key] = output_data[key]

def combine_hdfs(datapath, outputpath, *, combo_name='energetics.h5',
                                          progress=True):
    """Function combines all .h5 files at the given datapath, cleans and
        sorts data
    Inputs
        datapath
    """
    filelist = glob.glob(datapath+'/*.h5')
    with pd.HDFStore(filelist[1]) as store:
        anykey = store.keys()[0]
        blank = pd.DataFrame(columns=store[anykey].keys())
        keylist = store.keys()
        #see if index contains data we care about (usually time)
        if type(store[anykey].index[0]) != type(0):
            doignore=False
        else:
            doignore=True
    for key in keylist:
        energetics = blank
        for hdffile in filelist:
            print(hdffile)
            with pd.HDFStore(hdffile) as store:
                if any([localkey == key for localkey in store.keys()]):
                    energetics = energetics.append(store.get(key),
                                                ignore_index=doignore)
        if doignore: #index values
            timekey=[key for key in energetics.keys()if'time'in key.lower()][0]
            energetics = energetics.sort_values(by=[timekey])
            energetics = energetics.reset_index(drop=True)
        else:
            energetics.sort_index(inplace=True)
        print(energetics)
        with pd.HDFStore(outputpath+'/'+combo_name) as store:
            store[key] = energetics
    if progress:
        display_progress(outputpath+'/meshdata/*.h5',
                        outputpath+'/energetics.h5', 'Combined_zones')

if __name__ == "__main__":
    DATA = sys.argv[1]
    OPATH = sys.argv[2]
    combine_hdfs2(DATA, OPATH)
    #merge_hdfs(DATA, OPATH)
