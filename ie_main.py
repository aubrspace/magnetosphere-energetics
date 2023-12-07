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

def save_ie_image(ie_stylehead_north, ie_stylehead_south,
                  outpath):
    # Create and save an image
    northsheet = os.getcwd()+'/'+ie_stylehead_north
    southsheet = os.getcwd()+'/'+ie_stylehead_south
    if os.path.exists(northsheet) or os.path.exists(southsheet):
        tp.macro.execute_extended_command(
                            command_processor_id='Multi Frame Manager',
                            command='MAKEFRAMES3D ARRANGE=TILE SIZE=50')
        for i,frame in enumerate(tp.frames()):
            if i==0 and os.path.exists(northsheet):
                frame.load_stylesheet(northsheet)
            elif i==1 and os.path.exists(southsheet):
                frame.load_stylesheet(southsheet)
            elif i>1:
                tp.layout.active_page().delete_frame(frame)
        tp.layout.active_page().tile_frames(mode=TileMode.Rows)
        tp.save_png(os.path.join(outpath,'png',
                                     OUTPUTNAME+'.png'),width=1600)
    else:
        print('NO STYLESHEETS!!')

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        pass
    # Set file paths/individual file
    inpath = 'localdbug/parameter_study/ie/'
    outpath = 'localdbug/parameter_study/ie/'
    #inpath = 'ideal_recon_tim/IE/'
    #outpath = 'polarcap_rxn/'
    head = 'it*'
    ie_stylehead_north, ie_stylehead_south = 'north_Jr_status.sty','south_Jr_status.sty'

    # Search to find the full list of files
    filelist = sorted(glob.glob(os.path.join(inpath,head)),
                                key=makevideo.time_sort)[0::]

    i=0
    for k,f in enumerate(filelist[0::]):
        filetime = makevideo.get_time(f)
        OUTPUTNAME = f.split('it')[-1].split('.')[0]
        if True:
            print('('+str(i)+') ',filetime)
            i+=1
            tp.new_layout()
            iedatafile = f
            #python objects
            main = tp.active_frame()
            main.name = 'main'

            #Perform data extraction
            with tp.session.suspend():
                # IE data
                field_data = tp.data.load_tecplot(iedatafile,
                                    read_data_option=ReadDataOption.Append)
                if len(field_data.zone_names)<2:
                    pass
                else:
                    field_data.zone(0).name = 'ionosphere_north'
                    field_data.zone(1).name = 'ionosphere_south'
                    ionosphere.get_ionosphere(field_data,
                                          verbose=True,
                                          hasGM=False,
                                          eventtime=filetime,
                                          analysis_type='mag',
                                          integrate_surface=True,
                                          integrate_line=False,
                                          integrate_contour=True,
                                          do_interfacing=False,
                        outputpath=outpath)
                    if True:
                        save_ie_image(ie_stylehead_north, ie_stylehead_south,
                                  outpath)

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
