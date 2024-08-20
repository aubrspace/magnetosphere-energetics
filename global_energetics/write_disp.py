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
from global_energetics import makevideo

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
            if 'time' in data[key].attrs.keys():
                #from IPython import embed; embed()
                store.get_storer(key).attrs.time = data[key].attrs['time']

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
    if meshfile=='NoMesh':
        result = ''
    else:
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

def combine_to_multi_index(datapath,outputpath,**kwargs):
    """Function combines .h5 files and stores common dataframes within the
        store in a multiIndex dataframe within an output .h5 file. This is
        useful for data which multi-dimensional and not sparse. For instance
        2500+ timesteps with 6 fluxes across ~90000 flux points.
    Inputs
        datapath
        outpath
    """
    combo_name = 'test.h5'
    filelist = sorted(glob.glob(os.path.join(datapath,'*.h5')),key=makevideo.time_sort)
    #TODO
    time_dict = {}
    value_dict = {}
    point_dict = {}
    index_dict = {}
    # Get the number of dataframes for the final output array
    with pd.HDFStore(filelist[0]) as input_store:
        dataframe_keys = input_store.keys()
        for key in input_store.keys():
            df = input_store[key]
            #NOTE assuming each dataframe in the store has the same cols
            columns = np.array(df.keys())
            time_dict[key] = np.array([])
            point_dict[key] = np.array([])
            index_dict[key] = np.array([])
            value_dict[key] = {}
            for col in columns:
                value_dict[key][col] = np.array([])
    for i,infile in enumerate(filelist):
        if kwargs.get('progress',True):
            print('{:>4}/{:<4}\t{:<25}'.format(i+1,len(filelist),infile))
        with pd.HDFStore(infile) as input_store:
            for key in input_store.keys():
                df = input_store[key]
                # Get time entry
                if 'time' in input_store.get_storer(key).attrs:
                    time = input_store.get_storer(key).attrs.time
                else:
                    time = f'time{i}'
                # Update time array
                time_dict[key] = np.append(time_dict[key],
                                           np.array([time]*len(df)))
                #columns = np.array([c.replace('/','') for c in df.keys()])
                # Update the point array depending on the number of points
                point_dict[key] = np.append(point_dict[key],
                                            list(range(len(df))))
                # Add this files values to the value array
                for col in columns: #NOTE assuming all have same columns
                    value_dict[key][col] = np.append(value_dict[key][col],
                                                     df[col].values)
    with pd.HDFStore(outputpath+'/'+combo_name) as output:
        # Set the index array with points + time entry
        for key in dataframe_keys:
            index_dict[key] = [time_dict[key],point_dict[key]]
            # Smoosh the values from dict ->  (points) x (columns) array
            values = np.array(list(value_dict[key].values())).T
            output[key.replace('/','')] = pd.DataFrame(values,
                                                       index=index_dict[key],
                                                       columns=columns)
    '''
                input_dataframe = input_data[key]
                if key in output_data:
                    output_data[key] = pd.concat([input_dataframe,
                                                  output_data[key]],
                                                 ignore_index=True)
                else:
                    output_data[key] = input_dataframe
    with pd.HDFStore(outputpath+'/'+combo_name) as output:
       for key in output_data.keys():
           if 'Time [UTC]' in output_data[key].keys():
               load_df = output_data[key].sort_values(by='Time [UTC]')
               load_df.index = load_df['Time [UTC]']
               load_df.drop(columns=['Time [UTC]'],inplace=True)
           else:
               load_df = output_data[key]
           output[key] = load_df
    pass
    '''

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
           if 'Time [UTC]' in output_data[key].keys():
               load_df = output_data[key].sort_values(by='Time [UTC]')
               load_df.index = load_df['Time [UTC]']
               load_df.drop(columns=['Time [UTC]'],inplace=True)
           else:
               load_df = output_data[key]
           output[key] = load_df

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
    PATH = sys.argv[1]
    OPATH = sys.argv[2]
    kwargs = {}
    if '-c' in sys.argv:
        kwargs['combo_name'] = sys.argv[sys.argv.index('-c')+1]
    if '-multi' in sys.argv:
        combine_to_multi_index(PATH,OPATH,**kwargs)
    else:
        combine_hdfs2(PATH, OPATH,**kwargs)
