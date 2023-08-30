#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import logging
import glob
import numpy as np
from numpy import pi
import pandas as pd
import datetime as dt
#import spacepy
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
#import global_energetics
from global_energetics.extract import magnetosphere
from global_energetics.extract import ionosphere
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import tec_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics.write_disp import write_to_hdf
from global_energetics import makevideo

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        pass
    # Set file paths/individual file
    all_main_phase = glob.glob('ccmc_2022-02-02/3d*')
    all_times = sorted(glob.glob('pc2000_run/GM/IO2/3d__var_1_*.plt'),
                                key=makevideo.time_sort)[0::]

    oggridfile = 'ccmc_2022-02-02/3d__volume_e20220202.plt'

    i=0
    for k,f in enumerate(all_main_phase[1200:1201]):
        filetime = makevideo.get_time(f)
        OUTPUTNAME = f.split('e')[-1].split('.')[0]
        if True:
            print('('+str(i)+') ',filetime)
            i+=1
            tp.new_layout()
            mhddatafile = f
            iedatafile=('pc2000_run/IE/ionosphere/'+
                    'it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      filetime.year-2000,
                      filetime.month,
                      filetime.day,
                      filetime.hour,
                      filetime.minute,
                      filetime.second))
            #python objects
            field_data = tp.data.load_tecplot(mhddatafile)
            field_data.zone(0).name = 'global_field'
            if len(field_data.zone_names)>1:
                field_data.zone(1).name = 'future'
            main = tp.active_frame()
            main.name = 'main'

            #Perform data extraction
            with tp.session.suspend():
                # GM data
                _,results = magnetosphere.get_magnetosphere(field_data,
                                                        save_mesh=False,
                                                        write_data=True,
                                    verbose=True,
                                    debug=False,
                                    do_cms=False,
                                    do_central_diff=False,
                                    analysis_type='energy_mass_mag',
                                    modes=['iso_betastar','closed',
                                           'nlobe','slobe'],
                                    do_interfacing=True,
                                    integrate_surface=True,
                                    integrate_volume=False,
                                    integrate_line=False,
                                    do_1Dsw=True,
                                    #truegridfile=oggridfile,
                                    outputpath='fte_test/',
                                    )
                '''
                # IE data
                tp.data.load_tecplot(iedatafile,
                                 read_data_option=ReadDataOption.Append)
                field_data.zone(-2).name = 'ionosphere_north'
                field_data.zone(-1).name = 'ionosphere_south'
                ionosphere.get_ionosphere(field_data,
                                          verbose=True,
                                          hasGM=True,
                                          eventtime=filetime,
                                          analysis_type='mag',
                                          integrate_surface=True,
                                          integrate_line=True,
                                          do_interfacing=True,
                                          outputpath='polarcap2000/analysis/')
                '''
            '''
            # Create and save an image
            northsheet = '/home/aubr/Code/swmf-energetics/north_pc.sty'
            southsheet = '/home/aubr/Code/swmf-energetics/south_pc.sty'
            tp.macro.execute_extended_command(
                            command_processor_id='Multi Frame Manager',
                            command='MAKEFRAMES3D ARRANGE=TILE SIZE=50')
            for i,frame in enumerate(tp.frames()):
                if i==0:
                    frame.load_stylesheet(northsheet)
                elif i==1:
                    frame.load_stylesheet(southsheet)
                elif i>1:
                    tp.layout.active_page().delete_frame(frame)
            tp.layout.active_page().tile_frames(mode=TileMode.Rows)
            tp.save_png(os.path.join('polarcap2000','figures','vis',
                                     OUTPUTNAME+'.png'),width=1600)
            '''

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
