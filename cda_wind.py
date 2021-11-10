#/usr/bin/env python
"""accesses WIND satellite data from CDA and creates IMF.dat for SWMF input
"""
import pandas as pd
import os
import sys
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from cdasws import CdasWs
cdas = CdasWs()

def toTimestamp(d):
    return d.timestamp()

def cda_to_np(c_dict):
    """Function converts items in dict like object from cdaWeb to dict of
        nparrays
    Inputs
        c_dict (dictlike)- from CDAWeb, see documentation
    Returns
        dict of np.array objects
    """
    return dict((key, np.array(val)) for key, val in c_dict.items())

#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    start = dt.datetime(2021,11,3,12,0)
    end = dt.datetime(2021,11,5,12,0)
    #end = dt.datetime.now()
    outpath = './'
    #Use "cdas.get_variables('WI_H1_SWE')" to see options,
    #  K0=>realtime, SWE=>solarwindexperiment, MFI=>magneticfieldinstrument
    #######################################################################
    #Get data
    #print(cdas.get_variables('WI_K0_SWE'))
    swvars = ['V_GSM', 'THERMAL_SPD', 'Np','SC_pos_GSM','QF_V']
    magvars = ['BGSMc']
    swstatus, swdata = cdas.get_data('WI_K0_SWE', swvars, start, end)
    magstatus, magdata = cdas.get_data('WI_K0_MFI', magvars, start, end)
    sw = cda_to_np(swdata)
    mag = cda_to_np(magdata)
    #from IPython import embed; embed()
    #Unpack/rename/calculate to get terms for IMF.dat columns
    #B
    for comp in enumerate(['bx','by','bz']):
        Binterp=np.interp([t.timestamp() for t in sw['Epoch']],
                          [t.timestamp() for t in mag['Epoch']],
                          mag['BGSMc'][:,comp[0]])
        sw.update({comp[1]:Binterp})
    #V
    for comp in enumerate(['vx', 'vy', 'vz']):
        sw.update({comp[1]:sw['V_GSM'][:,comp[0]]})
    #Rho
    sw.update({'density':sw['Np']})
    #Temperature
    sw.update({'temperature':np.array([vth*500*1.6726/1.3808
                                           for vth in sw['THERMAL_SPD']])})
    #Time
    sw.update({'times':sw['Epoch']})
    #Process data based on quality flag
    imf_keys=['times','bx','by','bz','vx','vy','vz','density','temperature']
    '''
    assert any(sw['QF_V']!=0), ('No quality data present!!')
    for key in imf_keys:
        sw.update({key:sw[key][(sw['QF_V']==0)]})
    '''
    for key in imf_keys:
        sw.update({key:sw[key][(sw['temperature']>0)]})
    #Shift data based on time to X=+10Re
    shift = (np.average(np.array(
                             [-1*x+63710 for x in sw['SC_pos_GSM'][:,0]]))/
                             np.average(sw['vx']))
    sw.update({'times':np.array(
                    [t+dt.timedelta(seconds=shift) for t in sw['times']])})
    #Remove all non IMF.dat items in dictionary
    for key in sw.copy():
        if key not in imf_keys:
            sw.pop(key)
    #Write to file
    from swmfpy.io import write_imf_input
    write_imf_input(sw, coords='GSM')
    if os.path.exists('IMF.dat'):
        print('Successfully wrote to IMF.dat\n\tfrom:{} to:{}'.format(
                                           sw['times'][0],sw['times'][-1]))
        print('\twith timeshift of {}'.format(dt.timedelta(seconds=shift)))
