#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
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

def add_units(var):
    """Function adds appropriate units to field data variable
    Inputs
        var
    """
    if var =='X' or var=='Y' or var=='Z':
        return var+' [R]'
    if var =='Ux' or var=='Uy' or var=='Uz':
        return 'U_'.join(var.split('U'))+' [km/s]'
    if var =='Bx' or var=='By' or var=='Bz':
        return 'B_'.join(var.split('B'))+' [nT]'
    if var =='jx' or var=='jy' or var=='jz':
        return 'J_'.join(var.split('j'))+' [`mA/m^2]'
    if var =='Rho':
        return var + ' [amu/cm^3]'
    if var =='P':
        return var + ' [nPa]'
    if var =='status':
        return 'Status'
    else:
        return var

def load_swmf_sat(filename, field_variables):
    """Function loads satellite file based on SWMF PARAM.in settings
    Inputs
        filename
        field_variables
    """
    with open(filename,'r') as satfile:
        header = satfile.readline()
        satname = header.split('/')[-1].split('.')[0]
        variables = satfile.readline().split('\n')[0].split(' ')[1::]
    varstring = ''
    u_variables = []
    for var in enumerate(variables):
        w_units = add_units(var[1])
        u_variables.append(w_units)
        varstring = varstring+w_units+','
    varstring = varstring.rstrip(',')
    varnamelist = ''
    for field_var in enumerate(field_variables):
        for var in variables:
            if field_var[1] == add_units(var):
                varnamelist = (varnamelist +
                               '\"{}\";\"{}\" '.format(field_var[1], var))
                found = True
        if not found:
            varnamelist = (varnamelist + '\"{}\" '.format(field_var[1]))
        found = False
    for var in variables:
        if not any([varname == add_units(var)
                    for varname in field_variables]):
            varnamelist = (varnamelist + '\"{}\" '.format(var))

    readDataCmd = ("'"+'\"'+os.getcwd()+'/'+filename+'\" '+
    '\"VERSION=100 FILEEXT=\\\"*.txt\\\" '+
    'FILEDESC=\\\"General Text\\\" '+
    '\"+\"\"+\"TITLE{SEARCH=LINE  '+
                    'NAME=\\\"'+satname+'\\\" '+
                    'LINE=1 '+
                    'DELIMITER=AUTO '+
                    'WIDTH=10 }'+
    '\"+\"\"+\"VARIABLES{\"+\"SEARCH=NONE  '+
                             'NAMES = \\"'+varstring+'\\"\"+\" '+
                             'LOADED= All  '+
                             'STARTLINE=1 '+
                             'ENDLINE=1 '+
                             'DELIMITER=AUTO '+
                             'WIDTH=10 }'+
    '\"+\"\"+\"DATA\"+\"{\"+\"IGNORENONNUMERICTOKENS=TRUE '+
                             'IMPORT\"+\"{\"+\"STARTID=LINE '+
                                        '{\"+\"LINE=3 }'+
                                    '\"+\"\"+\"ENDID=EOF '+
                                        '{\"+\"LINE=1 }\"+\"\"+\"'+
                             'FORMAT=IJKPOINT '+
                             'DELIMITER=AUTO '+
                             'WIDTH=10 }'+
    '\"+\"\"+\"DIMENSION\"+\"{\"+\"AUTO=TRUE '+
                             'CREATEMULTIPLEZONES=FALSE }'+
    '\"+\"}\"+\"GLOBALFILTERS{\"+\"USEBLANKCELLVALUE=FALSE '+
                                  'BLANKCELLVALUE=0.000000 }\"'+"' ")
    readDataCmd = ("""$!ReadDataSet  """+readDataCmd+"""
                   DataSetReader = 'General Text Loader'
                   VarNameList = '"""+varnamelist+"""'
                   ReadDataOption = Append
                   ResetStyle = No
                   AssignStrandIDs = No
                   InitialPlotType = XYLine
                   InitialPlotFirstZoneOnly = No
                   AddZonesToExistingStrands = No
                   VarLoadMode = ByName""")
    tp.macro.execute_command(readDataCmd)
    tp.active_frame().dataset.zone('Zone 1').name = satname
    return satname

def add_currentlocation(zonenames, field_data):
    #Get corresponding satellite fieldmap variable indices
    satindices = []
    for name in zonenames:
        satindices.append(int(field_data.zone(name).index))
        if len([zn for zn in field_data.zones('loc_'+name)]) > 0:
            loc_satzone = field_data.zone('loc_'+name)
        else:
            #create local sat zone with sat current position
            loc_satzone =  field_data.add_ordered_zone('loc_'+name, [1,1,1])
            #get the current position of the satellite based on aux data
            eventstring =field_data.zone('global_field').aux_data['TIMEEVENT']
            startstring=field_data.zone('global_field').aux_data[
                                                          'TIMEEVENTSTART']
            eventdt = dt.datetime.strptime(eventstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            startdt = dt.datetime.strptime(startstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            deltadt = eventdt-startdt
            tvals = field_data.zone(name).values('t').as_numpy_array()
            xvals = field_data.zone(name).values('X *').as_numpy_array()
            yvals = field_data.zone(name).values('Y *').as_numpy_array()
            zvals = field_data.zone(name).values('Z *').as_numpy_array()
            svals = field_data.zone(name).values('Status').as_numpy_array()
            xpos = xvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            ypos = yvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            zpos = zvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            status = svals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            loc_satzone.values('X *')[0] = xpos
            loc_satzone.values('Y *')[0] = ypos
            loc_satzone.values('Z *')[0] = zpos
            loc_satzone.values('Status')[0] = status

def get_satellite_zones(eventdt, datapath, field_data, *, coordsys='GSM'):
    """Function to find satellite trace data (if avail) and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of the event
        datapath- str path to ionosphere data
        coordsys- coordinate system of the field data
    """
    eventstring = str(eventdt)
    satfiles = glob.glob(datapath+'/*.sat')
    satzones = []
    if satfiles == []:
        print('No satellite data at {}!'.format(datapath))
        return []
    else:
        variables = []
        for var in field_data.variables():
            variables.append(var.name)
        for satfile in satfiles[0::]:
            print('reading: {}'.format(satfile))
            satzonename = load_swmf_sat(satfile, variables)
            satzones.append(satzonename)
    #reset variable names
    for var in field_data.variables('*\n*'):
        var.name = var.name.split('\n')[0]
    #add specific location data
    add_currentlocation(satzones, field_data)
    return satzones

if __name__ == "__main__":
    pass
