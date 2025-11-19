#!/usr/bin/env python3
"""Functions for handling data from swmf only
"""
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *

def swmf_read_time(*, zoneindex=0, key='TIMEEVENT',**kwargs):
    """Function looks in auxillary data for time information
    Inputs
        zoneindex- 0 based index for which swmf field data is found
        key- typically is TIMEEVENT, can update here
    Outputs
        datetime object of the event time
    """
    if 'auxfile' in kwargs:
        from global_energetics.extract.shared_tools import read_aux
        aux_data = read_aux(kwargs.get('auxfile'))
    else:
        #make dict of aux data
        aux_data = tp.active_frame().dataset.zone(zoneindex).aux_data
        aux_data = aux_data.as_dict()
    if aux_data != {}:
        timestring = aux_data[key]
        year = int(timestring.split('/')[0])
        month = int(timestring.split('/')[1])
        day = int(timestring.split('/')[2].split(' ')[0])
        hour = int(timestring.split('/')[2].split(' ')[-1].split(':')[0])
        minute = int(timestring.split('/')[2].split(' ')[-1].split(':')[1])
        second = int(timestring.split(
                        '/')[2].split(' ')[-1].split(':')[2].split('.')[0])
        return dt.datetime(year,month,day,hour,minute,second)
    else:
        return dt.datetime(2000,1,1,0,0,0)
