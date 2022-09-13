#!/usr/bin/env python3
"""Extracting data related to ground based magnetometers (obs and virtual)
"""
import glob
import numpy as np
import datetime as dt
import pandas as pd
#import spacepy
#from spacepy import coordinates as coord
#from spacepy import time as spt
from global_energetics.extract.stream_tools import(mag2gsm,mag2cart)
from global_energetics.makevideo import(get_time)

def datetimeparser(datetimestring):
    #NOTE copy!! should consolidate this
    try:
        return dt.datetime.strptime(datetimestring,'%Y %m %d %H %M %S %f')
    except TypeError:
        print('TypeError!')

def datetimeparser2(instring):
    return dt.datetime.strptime(instring,'%Y-%m-%dT%H:%M:%S')

def read_station_locations(*,file_in='stations.loc'):
    """Function reads in station locations (magLat/Lon), file should be
        included with swmf-energetics dist
    Inputs
        file_in (str)- file path
    Returns
        stations
    """
    stations = pd.read_csv(file_in,sep='\s+',header=[1],comment='#')
    stations.index = stations['IAGA']
    stations.drop(columns='IAGA',inplace=True)
    return stations

def lon2mlt(t):
    """Function takes a datetime stamp and returns the associated local
        timethat is pointing toward 0
    Inputs
        t (datetime)-
    Returns
        lon (float)- in degrees
    """
    return (t.hour+t.minute/60+t.second/3600)%24*180/12

def where_stations_now(nowtime,**kwargs):
    """Function returns location in GSM of stations
    Inputs
        nowtime (datetime)- which time to find in file
        kwargs:
            tilt (float)- current dipole tilt (GSM) default 0
    Returns
        stations (DataFrame)- data with all stations with coords in GSM
    """
    stations = read_station_locations()
    #Convert longitude to MLT shift (rel to 0:UTC London)
    stations['MLTshift'] = stations['MAGLON']*12/180
    stations['LONnow'] = (stations['MLTshift']+nowtime.hour+nowtime.minute/60+
                                               nowtime.second/3600)%24*180/12
    #Get theta tilt
    tilt = kwargs.get('tilt',0)
    x = np.zeros(len(stations))
    y = np.zeros(len(stations))
    z = np.zeros(len(stations))
    for i,(lat,lon) in enumerate(stations[['MAGLAT','LONnow']].values):
        x[i],y[i],z[i] = mag2cart(lat,lon,tilt)
    stations['X'] = x
    stations['Y'] = y
    stations['Z'] = z
    return stations

def read_simstations(file_in,*,cordsys='GSM'):
    """Function reads in simulation station data
    Inputs
        file_in (str)- location of file
    Returns
        stations (list[str])- list of strings with station ID's
        station_df (DataFrame)- data from the file
    """
    #Read first line which contains station ID's
    with open(file_in,'r')as f:
        stations = f.readline()
    #Parse string into a list with just the 3letter tags
    stations = stations.split(' ')[4::]
    stations[-1] = stations[-1].split('\n')[0]
    #Read the rest of the data into DataFrame with date parsing
    station_df = pd.read_csv(file_in,sep='\s+',skiprows=1,
                             parse_dates={'times':
                                  ['year','mo','dy','hr','mn','sc','msc']},
                             date_parser=datetimeparser,
                             infer_datetime_format=True,keep_date_col=True)
    #Set index to datetime, drop time column so non index consistent dtypes
    station_df.index=station_df['times']
    station_df.drop(columns=['times','year','mo','dy','hr','mn','sc','msc'],
                    inplace=True)
    #Assume that data is stored in order of stations repeating at each time
    test_station = station_df[station_df['station']==1]
    station_df['station'] = stations*len(test_station)
    return stations, station_df

def get_stations_now(file_in,nowtime,**kwargs):
    """Function gets station data from file for the given time in cordsys
    Inputs
        file_in (str)- location of file
        nowtime (datetime)- which time to find in file
        kwargs:
            tilt (float)- current dipole tilt (GSM) default 0
    Returns
        stations (list[str])- list of strings with station ID's
        station_df (DataFrame)- data from the file
    """
    #Read simulation data
    stations, alltimes = read_simstations(file_in)
    #Get specific time instance and reindex on station ID
    station_df = alltimes[alltimes.index==nowtime]
    station_df.index = station_df['station']
    station_df.drop(columns=['X','Y','Z'],inplace=True)
    #Simulation output locations seem suspect so recalculate from MAGLAT/LON
    station_xyz = where_stations_now(nowtime,**kwargs)
    #Update XYZ columns
    station_df[['X','Y','Z']] = station_xyz.loc[:,['X','Y','Z']]
    #Read in extra data
    aux_data_path = kwargs.get('aux_path',
                      '/'.join(file_in.split('/')[0:-1])+'/station_data/')
    #read_station_values(aux_data_path, station_df, nowtime)
    return stations, station_df

#Function that calculates error
def read_station_values(data_path,station_df,now):
    """Function appends Error or other columns found in aux data files
    Inputs
        data_path (str)- where files are located
        station_df (DataFrame)- data for all stations at this time
    Return
        station_df (modified)
    """
    filelist = glob.glob(data_path+'*.txt')
    for file in filelist:
        #scrape station name from file name
        #read in the file
        for key in aux.keys():
            if key not in station_df.keys():
                #Add the column if not already present
                station_df[key] = [0]*len(station_df)
        #set the station row's values
        pass
    from IPython import embed; embed()
    return station_df
#Function that calculates RSD index
#Function that projects value from XYZ_gsm into domain

if __name__ == "__main__":
    #Read the first station location
    #file_in = ('/home/aubr/Code/swmf-energetics/febstorm/'+
    #           'magnetometers_e20140218-060000.mag')
    file_in = ('localdbug/febstorm/magnetometers_e20140218-060000.mag')
    aux_data_path = 'localdbug/febstorm/station_data/'
    test_time = get_time('localdbug/febstorm/3d__var_1_e20140218-060400-033.plt')
    IDs, station_df = get_stations_now(file_in,test_time,tilt=20.9499)
    #TODO use these to write a testing function
    '''
            r = np.sqrt(x**2+y**2+z**2)
            mXhat_x = sin(deg2rad(btilt+90))
            mXhat_y = 0
            mXhat_z = -1*cos(deg2rad(btilt+90))
            mZhat_x = sin(deg2rad(btilt))
            mZhat_y = 0
            mZhat_z = -1*cos(deg2rad(btilt))
            lambda_ = np.arcsin(((mZhat_x*x+mZhat_z*z)/r)-
                        np.trunc((mZhat_x*x+mZhat_z*z)/r))
            theta = -180/pi*lambda_
    '''
