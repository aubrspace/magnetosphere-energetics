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
from swmfpy.io import write_imf_input

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

def obtain_vGSM(solarwind,**kwargs):
    """Function backs out the GSE->GSM rotation matrix since we have it in the
        magnetic field, then puts in the missing VyVz GSM values
    Inputs
        solarwind (dict{str:1Darr})
        kwargs:
            doTest (bool)- default False
    Returns
        solarwind (modified)
    """
    # Get unit vectors for the b field in each coordinate
    A = [v for v in zip(solarwind['bx']/solarwind['b'],
                        solarwind['by_gse']/solarwind['b'],
                        solarwind['bz_gse']/solarwind['b'])]
    B = [v for v in zip(solarwind['bx']/solarwind['b'],
                        solarwind['by']/solarwind['b'],
                        solarwind['bz']/solarwind['b'])]
    # Initialize our new dictionary values with the correct shape
    solarwind['vx'] = solarwind['vx_gse']
    solarwind['vy'] = solarwind['vy_gse']*0
    solarwind['vz'] = solarwind['vz_gse']*0
    if kwargs.get('doTest',False):
        test_x = solarwind['bx']*0
        test_y = solarwind['by']*0
        test_z = solarwind['bz']*0
        max_x = 0
        max_y = 0
        max_z = 0
    # Solving system a = Rb for R
    for i,(a,b) in enumerate(zip(A,B)):
        R = [[np.dot(a,b),                 -np.linalg.norm(np.cross(a,b)),0],
             [np.linalg.norm(np.cross(a,b)),np.dot(a,b),                  0],
             [0,                       0,                                 1]]
        vx,vy,vz = np.matmul(R,[solarwind['vx_gse'][i],
                                solarwind['vy_gse'][i],
                                solarwind['vz_gse'][i]])
        solarwind['vy'][i] = vy
        solarwind['vz'][i] = vz
        # Optional Test
        if kwargs.get('doTest',False):
            bx,by,bz = np.matmul(R,[solarwind['bx'][i],
                                    solarwind['by_gse'][i],
                                    solarwind['bz_gse'][i]])
            test_x[i] = bx-solarwind['bx'][i]
            test_y[i] = by-solarwind['by'][i]
            test_z[i] = bz-solarwind['bz'][i]
            max_x = max(max_x,abs(test_x[i]))
            max_y = max(max_y,abs(test_y[i]))
            max_z = max(max_z,abs(test_z[i]))
    if kwargs.get('doTest',False):
        print('max errors, {},{},{}'.format(max_x,max_y,max_z))
    return solarwind


#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    start = dt.datetime(2000,6,24,2,0)
    end = dt.datetime(2000,6,24,8,0)
    #end = dt.datetime.now()
    outpath = './'
    use_cda = False
    #Use "cdas.get_variables('WI_H1_SWE')" to see options,
    #  K0=>realtime, SWE=>solarwindexperiment, MFI=>magneticfieldinstrument
    #######################################################################
    #Get data
    #print(cdas.get_variables('WI_K0_SWE'))
    if use_cda:
        from cdasws import CdasWs
        cdas = CdasWs()
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
        imf_keys=['times','bx','by','bz','vx','vy','vz','density',
                  'temperature']
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

    else:
        # load omni data
        from swmfpy.web import get_omni_data
        sw = get_omni_data(start,end)
        # back out rotation matrix from b_gse and b_gsm
        sw = obtain_vGSM(sw,doTest=False)
    #Write to file
    write_imf_input(sw, coords='GSM')
    if os.path.exists('IMF.dat'):
        print('Successfully wrote to IMF.dat\n\tfrom:{} to:{}'.format(
                                           sw['times'][0],sw['times'][-1]))
        if use_cda:
            print('\twith timeshift of {}'.format(dt.timedelta(seconds=shift)))
        else:
            print('\tused omni data')
