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
import datetime as dt
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

def save_gm_multi(gm_style_list,outpath,OUTPUTNAME,filetime):
    # Quickly duplicate date across 4 frames
    tp.macro.execute_command('$!LoadColorMap  '+
                 '"'+os.path.join(os.getcwd(),'cosmetic/energetics.map')+'"')
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
    #t0 = dt.datetime(2015,5,18,3,0)#NOTE
    reltime = (filetime-t0).days*24*3600+(filetime-t0).seconds
    phase = int(np.floor((reltime/3600)/2))
    text1 = tp.active_frame().add_text('&(AUXZONE[1]:TIMEEVENT)')
    text2 =tp.active_frame().add_text('Test Phase: '+str(phase))
    if tp.active_frame().background_color == Color.Black:
        text1.color = Color.White
        text2.color = Color.White
    else:
        text1.color = Color.Black
        text2.color = Color.Black
    text1.position = (2,7)
    text2.position = (2,3)
    # Save
    tp.save_png(os.path.join(outpath,'png',
                             OUTPUTNAME+'.png'),width=1600)

def parse_infiles(inpath,outpath):
    # Get the set of data files to be processed (solution times)
    all_solution_times = sorted(glob.glob(inpath+'/3d__var_*.plt'),
                                key=makevideo.time_sort)[0::]
    head = all_solution_times[0].split('/')[-1].split('e')[0]
    # Prune any repeat times
    times = [makevideo.get_time(f) for f in all_solution_times]
    _,unique_i = np.unique(times,return_index=True)
    all_solution_times = [all_solution_times[i] for i in unique_i]
    #Pick up only the files that haven't been processed
    if os.path.exists(os.path.join(outpath,'energeticsdata')):
        parseddonelist, parsednotdone = [], []
        donelist = glob.glob(outpath+'/png/*.png')
        parseddonelist = [p.split('/')[-1].split('.')[0] for p in donelist]
        parsednotdonelist = [p.split('/')[-1].split('e')[-1].split('.')[0] for
                            p in all_solution_times if
                            p.split('/')[-1].split('e')[-1].split('.')[0]
                            not in parseddonelist]
        solution_times=[os.path.join(inpath,head+'e'+item+'.plt')for item
                        in parsednotdonelist]
    else:
        solution_times = all_solution_times
    print('files remaining: ',len(solution_times))
    return solution_times, all_solution_times

def energetics_analysis(infiles,outpath):
    #Reset session
    tp.new_layout()
    #python objects
    gm_stylehead = 'twopanel_status.sty'
    field_data = tp.data.load_tecplot(infiles)
    filetime = makevideo.get_time(infiles[0])
    outputname = infiles[0].split('e')[-1].split('.plt')[0]
    field_data.zone(0).name = 'global_field'
    if len(field_data.zone_names)>1:
        field_data.zone(1).name = 'future'
    main = tp.active_frame()
    main.name = 'main'

    #Perform data extraction
    # GM
    mesh, data = magnetosphere.get_magnetosphere(field_data,
                                      save_mesh=False,
                                      write_data=True,
                                      disp_result=True,
                                      verbose=True,
                                      do_cms=False,
                                      analysis_type='energy_mass',
            #### COMPARING THE FOLLOWING #####
                #modes=['perfectellipsoid','perfectsphere','ellipsoid'],
                modes=['iso_betastar','closed','nlobe','slobe','plasmasheet'],
                inner_r=3,
                tail_cap=-120,
                mpbetastar=0.7,
                                      customTerms={'test':'TestArea [Re^2]'},
                                      do_interfacing=False,
                                      integrate_surface=True,
                                      save_surface_flux_dist=True,#NOTE
                                      integrate_volume=False,
                        #truegridfile='starlink2/IO2/3d__volume_e20220202.plt',
                        truegridfile='starlink_volume_grab/GM/IO2/3d__volume.plt',
                                      outputpath=outpath)
    if False:
        save_gm_multi([
                       'cosmetic/energy_closed2.sty',
                       'cosmetic/energy_forward2.sty',
                       'cosmetic/dBn_night_north.sty',
                       'cosmetic/longtail_Bz.sty'],
        #save_gm_multi(['cosmetic/stretched_ux_plasmasheet.sty'],
        #'cosmetic/status_forward.sty',
        #               'cosmetic/energy_forward.sty',
        #               'cosmetic/daynight_closed_side.sty',
        #               'cosmetic/north_pc_rxn_busy.sty'],
                       outpath,outputname,filetime)
    else:
        with open(os.path.join(outpath,'png',outputname+'.png'),'wb') as png:
            png.close()

if __name__ == "__main__":
    start_time = time.time()
    ##Parse input flags
    # Input files
    if '-i' in sys.argv:
        inpath = sys.argv[sys.argv.index('-i')+1]
    elif '--idir' in sys.argv:
        inpath = sys.argv[sys.argv.index('--idir')+1]
    else:
        inpath = 'test_inputs/'
    if not os.path.exists(inpath):
        print('input path "'+inpath+'" not found!')
        exit()
    # Output path files
    if '-o' in sys.argv:
        outpath = sys.argv[sys.argv.index('-o')+1]
    elif '--odir' in sys.argv:
        outpath = sys.argv[sys.argv.index('--odir')+1]
    else:
        outpath = 'test_outputs/'
    ########################################
    #make directories for output
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(outpath+'/figures', exist_ok=True)
    os.makedirs(outpath+'/indices', exist_ok=True)
    os.makedirs(outpath+'/png', exist_ok=True)
    ########################################

    # Get the whole file list remaining
    file_list, full_list = parse_infiles(inpath,outpath)
    print('Full list = ',len(full_list))
    #[print(str(f)+'\n') for f in file_list]
    # If just a single file is requested
    if '-f' in sys.argv or '--file' in sys.argv:
        if '-f' in sys.argv:
            infile = sys.argv[sys.argv.index('-f')+1]
        elif '--file' in sys.argv:
            infile = sys.argv[sys.argv.index('--file')+1]
        nowfile = os.path.join(inpath,infile)
        if nowfile not in full_list:
            print('DIST '+nowfile+' not in list....')
            exit()
        elif nowfile not in file_list:
            print('DIST '+nowfile+' already done....')
            exit()

        try:
            nextfile = full_list[full_list.index(nowfile)+1].split('/')[-1]
            nextfile_mirror = os.path.join(inpath,nextfile)
            #nextfile_mirror = os.path.join(inpath,'mirror',nextfile)
            previousfile = os.path.join(inpath,
                              full_list[full_list.index(nowfile)-1].split('/')[-1])
            #previousfile_mirror = os.path.join(inpath,'mirror',previousfile)
        except IndexError:
            print(nowfile+' is the end of the list!')
            if full_list.index(nowfile)==0:
                previousfile = nowfile
            else:
                previousfile = os.path.join(inpath,
                               full_list[full_list.index(nowfile)-1].split('/')[-1])
                nextfile_mirror = nowfile
        print('MAIN previous: ',previousfile)
        print('MAIN now: ',nowfile)
        print('MAIN next: ',nextfile)
        #energetics_analysis([nowfile,nextfile_mirror],outpath)
        energetics_analysis([nowfile],outpath)
        #energetics_analysis([previousfile,nextfile_mirror],outpath)
        #Test message
        '''
        print('Processing: ',previousfile,'\n\twith\n',nextfile_mirror,
                '\ncurrent time:\t',makevideo.get_time(nowfile))
        '''
    else:
        # Process the whole list
        for i,nowfile in enumerate(file_list[0:2]):
            if i!=len(file_list):
                energetics_analysis(file_list[i:i+2],outpath)
            else:
                pass
                #energetics_analysis([nowfile,nowfile],outpath)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    exit()
