#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os, warnings
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

def find_IE_matched_file(path,filetime):
    """Function returns the IE file at a specific time, if it exists
    Inputs
        path (str)
        filetime (datetime)
    Returns
        iedatafile (str)
        success (bool)
    """
    iedatafile = (path+
                  'it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      filetime.year-2000,
                      filetime.month,
                      filetime.day,
                      filetime.hour,
                      filetime.minute,
                      filetime.second))
    if not os.path.exists(iedatafile):
        print(iedatafile,'does not exist!')
        success = False
    else:
        success = True
    return iedatafile, success

def save_image(stylehead, outpath,filetime):
    # Create and save an image
    stylesheet = os.getcwd()+'/'+stylehead
    if os.path.exists(stylesheet):
        frame = tp.active_frame()
        frame.load_stylesheet(stylesheet)
        t0 = dt.datetime(2022,6,6,0,0)#NOTE
        reltime = (filetime-t0).days*24*3600+(filetime-t0).seconds
        phase = int(np.floor((reltime/3600)/2))
        text2 =tp.active_frame().add_text('Test Phase: '+str(phase))
        if tp.active_frame().background_color == Color.Black:
            text2.color = Color.White
        else:
            text2.color = Color.Black
        text2.position = (2,3)
        # Save
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
    inpath = 'localdbug/parameter_study/MEDHIGH/'
    outpath = 'localdbug/parameter_study/MEDHIGH/'
    gmhead = '3d__var_1_e'
    ie_stylehead_north = 'north_Jr_status.sty'
    ie_stylehead_south = 'south_Jr_status.sty'

    # Search to find the full list of files
    filelist = sorted(glob.glob(os.path.join(inpath,gmhead+'*')),
                                key=makevideo.time_sort)[0::]

    i=0
    for k,f in enumerate(filelist[0::]):
        # NonTecplot stuff
        filetime = makevideo.get_time(f)
        OUTPUTNAME = f.split(gmhead)[-1].split('.')[0]
        print('('+str(i)+') ',filetime)
        iedatafile, ie_exists = find_IE_matched_file(inpath,filetime)
        #Tecplot objects
        tp.new_layout()
        dataset = tp.data.load_tecplot(f)
        dataset.zone(0).name = 'global_field'
        main = tp.active_frame()
        main.name = 'main'

        #Perform data extraction
        with tp.session.suspend():
            # GM data
            _,results = magnetosphere.get_magnetosphere(dataset,
                                    analysis_type='',
                                    modes=['closed'],
                                    inner_r=4,
                                    customTerms={'test':'TestArea [Re^2]'},
                                    do_interfacing=True,
                                    tail_cap=-100,
                                    integrate_surface=False,
                                    integrate_volume=False,
                                    truegridfile='',
                                    debug=True,
                                    verbose=True,
                                    outputpath=outpath)
            save_image('daynight_closed.sty',outpath,filetime)
            # IE data
            if False:
                dataset = tp.data.load_tecplot(iedatafile,
                                    read_data_option=ReadDataOption.Append)
                if dataset.zone('IonN*') is not None:
                    dataset.zone('IonN*').name = 'ionosphere_north'
                    do_north = True
                if dataset.zone('IonS*') is not None:
                    dataset.zone('IonS*').name = 'ionosphere_south'
                    do_south = True
            #if do_north*do_south:
            #    pass
                '''
                ionosphere.get_ionosphere(dataset,
                                          verbose=True,
                                          hasGM=True,
                                          eventtime=filetime,
                                          analysis_type='mag',
                                          integrate_surface=True,
                                          integrate_line=False,
                                          integrate_contour=True,
                                          do_interfacing=False,
                                          outputpath=outpath)
                '''

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
