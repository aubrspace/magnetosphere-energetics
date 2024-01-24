#/usr/bin/env python
import sys
import os
import glob
import warnings
import pandas as pd
import numpy as np
import time
warnings.simplefilter(action='ignore',
                      category=pd.errors.PerformanceWarning)

def prRed(thing): return "\033[91m {:<20}\033[00m".format(thing)
def prGreen(thing): return "\033[92m {:<20}\033[00m".format(thing)
def prCyan(thing): return "\033[96m {:<5}\033[00m".format(thing)
def trueColor(istrue):
    if istrue: return prGreen(istrue)
    else: return prRed(istrue)

def store_str_sort(store):
    fullpath = store.filename
    filename = fullpath.split('/')[-1]
    first_letter = filename[0]
    return first_letter

def timesort(store,*,timekey='Time [UTC]'):
    """Function sorts each DataFrame in an HDFStore by some time column
    Inputs
        store (HDFStore)
        timekey (str)
    Returns
        None
    """
    print('sorting '+store.filename)
    for key in [k for k in store.keys() if k!=timekey]:
        print('\t'+key)
        df = store[key]
        df = df.sort_values(by=timekey,ignore_index=True)
        df.index = df[timekey].values
        store[key] = df

def combine_hdfs(storelist,outputname):
    storelist[0].copy(outputname)#Start with the first store in the list
    outstore = pd.HDFStore(outputname)
    #Get a full list of all DataFrames between all stores
    alldfs = outstore.keys()
    for store in storelist[1::]:
        store.open()
        alldfs = np.union1d(alldfs,store.keys())
        store.close()
    for dfname in alldfs:
        #Check the lower level columns of the dataframe
        if dfname in storelist[0]:
            outdf = storelist[0][dfname]
        else:
            outdf = pd.DataFrame()
        print('Checking '+dfname+' in '+
            storelist[0].filename.split('/')[-1]+'...')
        for nextstore in storelist[1::]:
            nextstore.open()
            if dfname in nextstore.keys():
                nextdf = nextstore[dfname]
                #Create full set of all columns
                allcols = np.union1d(outdf.keys(),nextdf.keys())
                #Iterate over full set and replace if necessary
                print('{:<35}{:<20}{:<20}'.format('Term',
                            nextstore.filename.split('/')[-1],
                        storelist[0].filename.split('/')[-1]))
                for col in allcols:
                    innext = col in nextdf.keys()
                    innow = col in outdf.keys()
                    if ((col in nextdf.keys()) and
                        (col not in outdf.keys())):
                        print('{:<35}'.format(col)+trueColor(innext)
                                                      +prCyan('->')
                                                      +trueColor(innow))
                        outdf[col] = nextstore[dfname][col]
                    else:
                        print('{:<35}'.format(col)+trueColor(innext)
                                                      +prCyan('')
                                                      +trueColor(innow))
                #Put the combined DataFrame back in the HDF file
                outstore[dfname] = outdf
            nextstore.close()
    outstore.close()

def read_args(args):
    """Function reads input arguments and updates settings
    Inputs
        args (list[str])- list of strings input from user
    Returns
        settings (dict)- dictionary of settings
        storelist (list[HDFStore])- list of 'Store' objects (see pandas)
    """
    settings = {}
    storelist = []
    i_file_start = 1#index where files begin
    for i,arg in enumerate(args):
        if '-' in arg:
            #Adjust the depth if requested
            if '-d' in arg or '--depth' in arg:
                settings['depth'] = args[i+1]
            #Assign a name if requested
            elif '-n' in arg or '--name' in arg:
                settings['name'] = args[i+1]
            else:
                warnings.warn('Unrecognized setting '+arg+'!',UserWarning)
            i_file_start = i+2
        if i>=i_file_start:
            if os.path.isdir(arg):
                searchpath = os.path.join(arg,'*.h5')
                for infile in glob.glob(searchpath):
                    storelist.append(pd.HDFStore(infile))
            else:
                storelist.append(pd.HDFStore(arg))
    storelist = sorted(storelist,key=store_str_sort)
    print(storelist)
    return settings, storelist

if __name__ == "__main__":
    #Help message
    if '-h' in sys.argv or '--help' in sys.argv:
        print("""
        Combines two or more hdf files
        Usage: python combo_hdf.py [OPTION] [file1/path1] [file2/path2] ..

        Options:
            -h --help   prints this message then exit
            -n --name   give a name to the output (default output.h5)
            -d --depth  number of levels to search

        Example:
            This will combine two files
                python combo_hdf.py lobe_results.h5 all_volumes.h5

            This will combine .h5 files files in 'energeticsdata'
                python combo_hdf.py output_5min/energeticsdata

             """)
        exit()
    #Read input arguments from user
    settings, storelist = read_args(sys.argv)
    print('combining ',[s.filename.split('/')[-1] for s in storelist],
          ' -> ', settings.get('name','output.h5'))
    combine_hdfs(storelist, settings.get('name','output.h5'))
