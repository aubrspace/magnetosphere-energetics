#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
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

    # Set file paths/individual file
    #file_path = 'ccmc_2022-02-02/'
    file_path = 'ideal_run/GM/IO2/'
    #file_path = 'error_cases/ideal_conserve/GM/IO2/'
    #file_path = 'run_GMonly/GM/IO2/'
    filekey = '3d__var_1_*.plt'
    all_times = sorted(glob.glob(os.path.join(file_path,filekey)),
                                key=makevideo.time_sort)

    #oggridfile = 'ccmc_2022-02-02/3d__volume_e20220202.plt'
    #oggridfile = 'ideal_run/GM/IO2/3d__volume__n00000800.plt'
    oggridfile = glob.glob(file_path+'3d*volume*.plt')[0]

    i=0
    for k,f in enumerate(all_times[1050:1051]):
        filetime = makevideo.get_time(f)
        OUTPUTNAME = f.split('_1_')[-1].split('.')[0][1::]
        outpath = os.path.join('error_results',file_path.split('/')[0])
        os.makedirs(outpath,exist_ok=True)
        print('('+str(i)+') ',filetime, outpath)
        i+=1
        tp.new_layout()

        #python objects
        field_data = tp.data.load_tecplot(f)
        field_data.zone(0).name = 'global_field'
        if len(field_data.zone_names)>1:
            field_data.zone(1).name = 'future'
        main = tp.active_frame()
        main.name = 'main'

        #TODO: test out basic version with full energy analysis + 4Re sphere
        #       Then combine with combo_hdf.py script to put together
        #       Then call plotting script in line with analysis
        #       Test out so it all works together

        #TODO: Make a separate plot in the plotting script to just show errors

        #TODO: Run short restart tests with 2sec ouput for a total of 1min
        #       1. Before the IMF rotation
        #       2. With Conservation criteria turned on everywhere
        #       3. With IM turned off
        #       4. With IM coupling rate doubled
        #       5. With IE coupling rate doubled


        #Perform data extraction
        with tp.session.suspend():
            # GM data
            _,results = magnetosphere.get_magnetosphere(field_data,
                                    verbose=True,
                                    do_cms=False,
                                    do_central_diff=False,
                                    analysis_type='wave_energy',
                                    modes=['sphere'],
                                    sp_rmax=10,
                                    sp_rmin=3,
                                    #sp_x=-13,
                                    keep_zones='all',
                                    do_interfacing=False,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    integrate_line=False,
                                    truegridfile=oggridfile,
                                    outputpath=outpath
                                    )

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
