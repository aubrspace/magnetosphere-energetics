#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import logging
import numpy as np
from numpy import pi
import datetime as dt
import spacepy
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
#import global_energetics
from global_energetics.extract import magnetosphere
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics.write_disp import write_to_hdf

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    #pass in arguments
    starlink = ('localdbug/starlink/3d__var_1_e20220203-114000-000.plt',
                'localdbug/starlink/3d__var_1_e20220203-115000-000.plt')
    trackim = ('localdbug/trackim/3d__var_1_e20140219-020000-000.plt',
               'localdbug/trackim/3d__var_1_e20140219-020100-000.plt')
    paleo=('/home/aubr/Code/paleo/3d__var_4_e20100320-030000-000_40125_kya.plt')
    ccmc  = ('output/CCMC/3d__var_1_e20130713-204700-037.plt',
             'output/CCMC/3d__var_1_e20130713-204700-037.plt')

    '''
    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''

    for inputs in [starlink]:
        tp.new_layout()
        mhddatafile = inputs[0]
        OUTPUTNAME = mhddatafile.split('e')[-1].split('.')[0]
        #python objects
        field_data = tp.data.load_tecplot(inputs)
        field_data.zone(0).name = 'global_field'
        if len(field_data.zone_names)>1:
            field_data.zone(1).name = 'future'
        main = tp.active_frame()
        main.name = 'main'

        #Perform data extraction
        with tp.session.suspend():
            mesh, data = magnetosphere.get_magnetosphere(field_data,
                                                    outputpath='babyrun/',
                                                    integrate_volume=False,
                                                    do_interfacing=True,
                                            modes=['iso_betastar','nlobe'],
                                                    verbose=True,
                                                   analysis_type='energy')
            #modes=['iso_betastar','rc'],
    #with tp.session.suspend():
    #    if True:#manually switch on or off
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
