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
    #Nice condition
    #starlink = ('localdbug/starlink/3d__var_1_e20220203-114000-000.plt',
    #            'localdbug/starlink/3d__var_1_e20220203-115000-000.plt')
    starlink = ('starlink/3d__var_1_e20220204-223000-000.plt',
                'starlink/3d__var_1_e20220204-224000-000.plt')
    #Current fails
    #starlink = ('starlink/3d__var_1_e20220202-050300-000.plt',
    #            'starlink/3d__var_1_e20220202-050400-000.plt')
    #Future fails
    #starlink = ('starlink/3d__var_1_e20220202-050200-000.plt',
    #            'starlink/3d__var_1_e20220202-050300-000.plt')
    #Some other fail
    #starlink = ('starlink/3d__var_1_e20220202-061500-000.plt',
    #            'starlink/3d__var_1_e20220202-063000-000.plt')
    febstorm = ('localdbug/feb2014/3d__var_1_e20140218-060300-037.plt',
                'localdbug/feb2014/3d__var_1_e20140218-060400-033.plt')
    feb_asym = ('febstorm/3d__var_1_e20140219-130000-000.plt',
                'febstorm/3d__var_1_e20140219-130100-010.plt')
    trackim = ('localdbug/trackim/3d__var_1_e20140219-020000-000.plt',
               'localdbug/trackim/3d__var_1_e20140219-020100-000.plt')
    paleo=('/home/aubr/Code/paleo/3d__var_4_e20100320-030000-000_40125_kya.plt')
    ccmc  = ('output/CCMC/3d__var_1_e20130713-204700-037.plt',
             'output/CCMC/3d__var_1_e20130713-204700-037.plt')
    ccmc2  = (
            'ccmc_2019-08-30/3d__var_1_e20190830-165000-001.plt',
            'ccmc_2019-08-30/3d__var_1_e20190830-165100-032.plt')
    ccmc3  = (
            'ccmc_2019-05-13/3d__var_1_e20190513-225800-010.plt',
            'ccmc_2019-05-13/3d__var_1_e20190513-225900-036.plt')
    ccmc4  = (
            'ccmc_2019-05-13/3d__var_1_e20190514-025600-028.plt',
            'ccmc_2019-05-13/3d__var_1_e20190514-025700-023.plt')

    '''
    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''

    for inputs in [ccmc4]:
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
                                      do_cms=True,
                                      analysis_type='energymagmass',
                                      modes=['nlobe','slobe'],
                                      do_interfacing=True,
                                      integrate_surface=True,
                                      integrate_volume=False,
                                      verbose=True,
                                      extract_flowline=False,
                                      outputpath='babyrun/',
                                      customTerms={'test':'TestArea [Re^2]'})
                                      #analysis_type='energymassmag',
    #with tp.session.suspend():
    if False:#manually switch on or off
        #adjust view settings
        #proc = 'Multi Frame Manager'
        #cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
        #tp.macro.execute_extended_command(command_processor_id=proc,
        #                                  command=cmd)
        mode = ['iso_day', 'other_iso', 'iso_tail', 'hood_open_north']
        zone_hidekeys = ['sphere', 'box','shue','future','innerbound',
                         'lcb','nlobe','slobe','closed','rc']
        timestamp=True
        for n, frame in enumerate(tp.frames()):
            #frame[1].activate()
            if n==0:
                legend = False
                timestamp = True
                doslice = False#
                slicelegend = False
                fieldlegend = True
                fieldline=False
            if n==1:
                legend = True
                timestamp = False
                doslice = True
                slicelegend = False
                fieldlegend = False
                fieldline=True
            if n==2:
                legend = False
                timestamp = False
                doslice = True
                slicelegend = True
                fieldlegend = False
                fieldline=False
            if n==3:
                legend = True
                save = True
                timestamp = False
                doslice = False
                slicelegend = False
                fieldlegend = False
                fieldline=True
                zone_hidekeys = ['sphere', 'box','shue','future','lcb']
            view_set.display_single_iso(frame, mhddatafile,
                                        mode=mode[n],
                                        show_contour=False,
                                        show_fieldline=fieldline,
                                        show_legend=legend,
                                        show_slegend=slicelegend,
                                        show_flegend=fieldlegend,
                                        show_slice=doslice,
                                        timestamp_pos=[4,5],
                                        zone_hidekeys=zone_hidekeys,
                                        show_timestamp=timestamp)
            view_set.add_fieldlines(tp.active_frame(),mhddatafile,showleg=True,
                                    mode='allstations',
                                    station_file=
                         'ccmc_2019-08-30/magnetometers_e20190830-161300.mag')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
