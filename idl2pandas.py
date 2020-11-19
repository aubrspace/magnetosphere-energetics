#!/usr/bin/env python3
"""Functions for converting SWMF IE .idl file to pandas data
"""
import sys
import time
import numpy as np
from numpy import deg2rad, cos, sin
import pandas as pd

def read2pandas(filename, *, varname_start=7, varname_end=23, varnamelen=2,
                             num_start=2, num_end=6, northfirst=True,
                             data_start=42, between_data=1, radius=1.075,
                             theta_searchlist=['theta','Theta','THETA'],
                             phi_searchlist=['phi','Phi','PHI'],
                             include_cart=False, verbose=False,
                             outpath='./'):
    """Function takes .idl datafile and extracts the data into a pandas
    DataFrame object for further processing
    Inputs
        filename
        varname_start, varname_end- optional start/end variable names
        varnamelen- length of the variable entries (number of terms)
        num_start, num_end- optional start/end of datashape info
        northfirst- boolean, optional
        data_start- location of first datapoint
        between_data- number of non-empty rows between datasets
        radius- ionoshpere shell radius in Re
        theta_searchlist, phi_searchlist- list of terms to id ntheta
    Outputs
        ntheta, nphi- number of theta/phi points of grid
        northdf, southdf- pandas dataframe object north/south hemisphere
    """
    #get date and time info
    time = filename.split('it')[-1].split('.')[0]
    #get number of phi and theta points information from the file
    datashape = pd.read_table(filename, skiprows=lambda x: (x<num_start
                                                         or x>num_end))
    for entry in datashape.values:
        string = entry[0].split(' ')[-1]
        value = int(entry[0].split(' ')[-2])
        for key in theta_searchlist:
            if string.find(key)==1:
                ntheta = value
                thetastring = string
        for key in phi_searchlist:
            if string.find(key)==1:
                nphi = value
    if verbose:
        print('ntheta: {}'.format(ntheta))
        print('nphi: {}'.format(nphi))
    #get variable header information from the file
    varnames = pd.read_table(filename, skiprows=lambda x: (x<varname_start
                                                         or x>varname_end))
    headers=[]
    for entry in varnames.values:
        headers.append(' '.join(entry[0].split(' ')[-varnamelen:]))
    if verbose:
        print(headers)
    #put north and south data into dataframe object
    northdata = pd.read_table(filename, names=headers,
                              skiprows=lambda x: (x<data_start+1
                                           or x>data_start+ntheta*nphi))
    southdata = pd.read_table(filename, names=headers,
                              skiprows=lambda x: (x<data_start+2+
                                                 between_data+ntheta*nphi))
    if northfirst == False:
        holddata = northdata
        northdata = southdata
        southdata = holddata
    northdf = pd.DataFrame(columns=headers)
    print('copying north hemisphere data:')
    for k in range(0, len(northdata)):
        row = []
        for val in northdata['Theta [deg]'].values[k].split(' '):
            if val != '':
                row.append(float(val))
        northdf = northdf.append(pd.DataFrame([row], columns=headers),
                                              ignore_index=True)
        print('row: {} complete'.format(k))
    southdf = pd.DataFrame(columns=headers)
    print('copying south hemisphere data:')
    for k in range(0, len(southdata)):
        row = []
        for val in southdata['Theta [deg]'].values[k].split(' '):
            if val != '':
                row.append(float(val))
        southdf = southdf.append(pd.DataFrame([row], columns=headers),
                                              ignore_index=True)
        print('row: {} complete'.format(k))
    if include_cart:
        #calculate local cartesian coordinates
        #North
        xion = pd.DataFrame((radius*
                             cos(deg2rad(northdf['Theta [deg]'].values))
                            *cos(deg2rad(northdf['Psi [deg]'].values))),
                            columns=['X_iono [Re]'])
        headers.append('X_iono [Re]')
        yion = pd.DataFrame((radius*
                             cos(deg2rad(northdf['Theta [deg]'].values))
                            *sin(deg2rad(northdf['Psi [deg]'].values))),
                            columns=['Y_iono [Re]'])
        headers.append('Y_iono [Re]')
        zion = pd.DataFrame((radius*
                             sin(deg2rad(northdf['Theta [deg]'].values))),
                            columns=['Z_iono [Re]'])
        headers.append('Z_iono [Re]')
        #combine all newly calculated columns
        northdf = northdf.reindex(columns=headers,fill_value=999)
        for df in [xion, yion, zion]:
            northdf = northdf.combine(df,np.minimum,fill_value=1000)
        #South
        xion = pd.DataFrame((radius*
                             cos(deg2rad(southdf['Theta [deg]'].values))
                            *cos(deg2rad(southdf['Psi [deg]'].values))),
                            columns=['X_iono [Re]'])
        yion = pd.DataFrame((radius*
                             cos(deg2rad(southdf['Theta [deg]'].values))
                            *sin(deg2rad(southdf['Psi [deg]'].values))),
                            columns=['Y_iono [Re]'])
        zion = pd.DataFrame((radius*
                             sin(deg2rad(southdf['Theta [deg]'].values))),
                            columns=['Z_iono [Re]'])
        #combine all newly calculated columns
        southdf = southdf.reindex(columns=headers,fill_value=999)
        for df in [xion, yion, zion]:
            southdf = southdf.combine(df,np.minimum,fill_value=1000)
    if verbose:
        print('North data')
        print(northdf)
        print('South data')
        print(southdf)
    #write df to csv file
    northdf.to_csv(outpath+'iono_north_'+time+'.csv', index=False)
    southdf.to_csv(outpath+'iono_south_'+time+'.csv', index=False)
    return ntheta, nphi


if __name__ == "__main__":
    start_time = time.time()
    if '-v' in sys.argv:
        VERBOSE = True
    else:
        VERBOSE = False
    NTHETA, NPHI = read2pandas(sys.argv[1], verbose=VERBOSE,
                               include_cart=True,
                               outpath=sys.argv[2])
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
