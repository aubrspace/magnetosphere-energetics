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

def save_ie_image(ie_stylehead_north, ie_stylehead_south):
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
        tp.save_png(os.path.join('polarcap2000','figures','vis',
                                     OUTPUTNAME+'.png'),width=1600)
    else:
        print('NO STYLESHEETS!!')

def save_gm_multi(gm_style_list,outpath,OUTPUTNAME,filetime):
    # Quickly duplicate date across 4 frames
    tp.macro.execute_extended_command(
                        command_processor_id='Multi Frame Manager',
                        command='MAKEFRAMES3D ARRANGE=TILE SIZE=50')
    # Load a list of premade style sheets
    for i,frame in enumerate(tp.frames()):
        if i<(len(gm_style_list)):
            frame.load_stylesheet(gm_style_list[i])
        else:
            tp.layout.active_page().delete_frame(frame)
    # Regroup into a grid if not using 4 frames
    tp.layout.active_page().tile_frames(mode=TileMode.Grid)
    # Stamp one of the frames with the 'test phase'
    t0 = dt.datetime(2022,6,6,0,0)#NOTE
    reltime = (filetime-t0).days*24*3600+(filetime-t0).seconds
    phase = int(np.floor((reltime/3600)%2))
    text =tp.active_frame().add_text('Test Phase: '+str(phase))
    if tp.active_frame().background_color == Color.Black:
        text.color = Color.White
    else:
        text.color = Color.Black
    text.position = (2,2)
    # Save
    tp.save_png(os.path.join(outpath,'data/png/',
                             OUTPUTNAME+'.png'),width=1600)

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        pass
    # Set file paths/individual file
    inpath = 'localdbug/parameter_study/'
    #inpath = 'run_HIGHnHIGHu/GM/IO2/'
    outpath = 'parameter_study/'
    head = '3d__var_1_*'
    ie_stylehead_north, ie_stylehead_south = 'north_pc.sty','south_pc.sty'
    #gm_stylehead = 'simple_vis.sty'
    #gm_stylehead = 'ffj_vis.sty'
    #gm_stylehead = 'twopanel_status.sty'

    # Search to find the full list of files
    filelist = sorted(glob.glob(os.path.join(inpath,head)),
                                key=makevideo.time_sort)[0::]
    #oggridfile = glob.glob(os.path.join(inpath,'3d*volume*.plt'))[0]
    oggridfile = ''

    i=0
    for k,f in enumerate(filelist[0:1]):
    #k=0
    #f='febstorm/3d__var_1_e20140219-055500-008.plt'
    #if True:
        filetime = makevideo.get_time(f)
        OUTPUTNAME = f.split('e')[-1].split('.')[0]
        if True:
            print('('+str(i)+') ',filetime)
            i+=1
            tp.new_layout()
            mhddatafile = f
            '''
            iedatafile=(inpath.replace('/GM/IO2/','/IE/ionosphere/')+
                    'it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      filetime.year-2000,
                      filetime.month,
                      filetime.day,
                      filetime.hour,
                      filetime.minute,
                      filetime.second))
            '''
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
                                                        disp_result=True,
                                    verbose=True,
                                    debug=False,
                                    do_cms=False,
                                    do_central_diff=False,
                                    do_1Dsw=False,
                                    analysis_type='energy',
                                    modes=['iso_betastar','closed',
                                           'nlobe','slobe'],
                                    inner_r=4,
                                    customTerms={'test':'TestArea [Re^2]'},
                                    do_interfacing=True,
                                    integrate_line=False,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    truegridfile=oggridfile,
                                    extract_flowline=False,
                                    outputpath=outpath)
                '''
                if os.path.exists(iedatafile) and False:
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
                    if True:
                        save_ie_image(ie_stylehead_north, ie_stylehead_south)
                '''
                #if os.path.exists(os.getcwd()+'/'+gm_stylehead)and False:
                if True:
                    save_gm_multi(['front_iso_status.sty',
                                   'tail_iso_status.sty',
                                   'front_iso_Knet.sty',
                                   'north_eq_Kx.sty'],outpath,OUTPUTNAME,
                                   filetime)
                    '''
                    # split into multiple frames
                    main.load_stylesheet(os.getcwd()+'/'+gm_stylehead)
                    tp.save_png(os.path.join(outpath,'data/png/',
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
