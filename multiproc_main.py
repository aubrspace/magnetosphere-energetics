#/usr/bin/env python
"""Example from the pytecplot documentation
"""
import time
import logging
import atexit, os, multiprocessing, sys
import numpy as np
from numpy import pi
import datetime as dt
import glob
import spacepy
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetopause
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics import write_disp, makevideo

def init(rundir, mhddir, iedir, imdir, scriptdir, outputpath, pngpath,
         all_solution_times):
    # !!! IMPORTANT !!!
    # Must register stop at exit to ensure Tecplot cleans
    # up all temporary files and does not create a core dump
    atexit.register(tp.session.stop)
    #globalize variables for each worker
    global RUNDIR, MHDDIR, IEDIR, IMDIR, SCRIPTDIR, OUTPUTPATH, PNGPATH, ID
    global ALL_SOLUTION_TIMES
    RUNDIR = rundir
    MHDDIR = mhddir
    IEDIR = iedir
    IMDIR = imdir
    SCRIPTDIR = scriptdir
    OUTPUTPATH = outputpath
    PNGPATH = pngpath
    ALL_SOLUTION_TIMES = all_solution_times
    ID = id(multiprocessing.current_process())
    os.system('mkdir '+MHDDIR+'/'+str(ID))
    os.system('cp '+MHDDIR+'/*.sat '+MHDDIR+'/'+str(ID)+'/')

def work(mhddatafile):
    # Load data and change zone name
    print(mhddatafile)
    for sol in enumerate(ALL_SOLUTION_TIMES):
        if mhddatafile == sol[1]:
            if sol[0]!=len(ALL_SOLUTION_TIMES)-1:
                nextmhdfile = ALL_SOLUTION_TIMES[sol[0]+1]
            else:
                nextmhdfile = mhddatafile
    tp.new_layout()
    unzipdatafile = mhddatafile.split('.gz')[0]
    unzipnextfile = (MHDDIR+'/'+str(ID)+
                     nextmhdfile.split('.gz')[0].split('/')[-1])
    os.system('gunzip -c '+mhddatafile+' > '+unzipdatafile)
    os.system('gunzip -c '+nextmhdfile+' > '+unzipnextfile)
    mhddatafile = unzipdatafile
    nextmhdfile = unzipnextfile
    field_data = tp.data.load_tecplot([mhddatafile, nextmhdfile])
    field_data.zone(0).name = 'global_field'
    field_data.zone(1).name = 'future'
    OUTPUTNAME = mhddatafile.split('e')[-1].split('.plt')[0]
    #Caclulate surfaces
    magnetopause.get_magnetopause(field_data, mhddatafile,
                                  zone_rename='mp_full',
                                  outputpath=OUTPUTPATH)
    '''
    magnetopause.get_magnetopause(field_data, mhddatafile,
                                  zone_rename='mp_w100',
                                  do_blank=True,blank_value=100,
                                  outputpath=OUTPUTPATH)
    magnetopause.get_magnetopause(field_data, mhddatafile,
                                  zone_rename='mp_w200',
                                  do_blank=True,blank_value=200,
                                  outputpath=OUTPUTPATH)
    magnetopause.get_magnetopause(field_data, mhddatafile,
                                  zone_rename='mp_w300',
                                  do_blank=True,blank_value=300,
                                  outputpath=OUTPUTPATH)
    '''
    #get supporting module data for this timestamp
    eventstring =field_data.zone('global_field').aux_data['TIMEEVENT']
    startstring =field_data.zone('global_field').aux_data['TIMEEVENTSTART']
    eventdt = dt.datetime.strptime(eventstring,'%Y/%m/%d %H:%M:%S.%f')
    startdt = dt.datetime.strptime(startstring,'%Y/%m/%d %H:%M:%S.%f')
    deltadt = eventdt-startdt
    #satzones = satellites.get_satellite_zones(eventdt, MHDDIR+'/'+str(ID),
    #                                          field_data)
    satzones = []
    #adjust view settings
    #tile
    proc = 'Multi Frame Manager'
    cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
    tp.macro.execute_extended_command(command_processor_id=proc,
                                          command=cmd)
    bot_right = [frame for frame in tp.frames('main')][0]
    bot_right.name = 'inside_from_tail'
    frame1 = [frame for frame in tp.frames('Frame 001')][0]
    frame2 = [frame for frame in tp.frames('Frame 002')][0]
    frame3 = [frame for frame in tp.frames('Frame 003')][0]
    view_set.display_single_iso(bot_right,
                                'KSurf_net *', mhddatafile,
                                show_contour=True,
                                show_slice=False, energyrange=9e10,
                                show_legend=False, mode='inside_from_tail',
                                pngpath=PNGPATH, energy_contourmap=4,
                                plot_satellites=False, satzones=satzones,
                                outputname=OUTPUTNAME, save_img=False,
                                zone_hidekeys=['box', 'lcb', '30', '40'])
    frame1.activate()
    frame1.name = 'isodefault'
    view_set.display_single_iso(frame1,
                                'KSurf_net *', mhddatafile,
                                show_contour=True,
                                show_slice=True, show_legend=False,
                                pngpath=PNGPATH,
                                plot_satellites=False, satzones=satzones,
                                outputname=OUTPUTNAME, save_img=False,
                                show_timestamp=False,
                                zone_hidekeys=['box', 'lcb', '30', '40'])
    frame2.activate()
    frame2.name = 'alternate_iso'
    view_set.display_single_iso(frame2,
                                'KSurf_net *', mhddatafile,
                                show_contour=True,
                                show_slice=True,
                                pngpath=PNGPATH, add_clock=True,
                                plot_satellites=False, satzones=satzones,
                                outputname=OUTPUTNAME, save_img=False,
                                mode='other_iso',
                                show_timestamp=False,
                                zone_hidekeys=['box', 'lcb', '30', '40'])
    frame3.activate()
    frame3.name = 'tail_iso'
    view_set.display_single_iso(frame3,
                                'KSurf_net *', mhddatafile,
                                show_contour=True,
                                show_slice=True, show_legend=False,
                                pngpath=PNGPATH, transluc=60,
                                plot_satellites=False, satzones=satzones,
                                outputname=OUTPUTNAME,
                                mode='iso_tail',
                                show_timestamp=False,
                                zone_hidekeys=['box', 'lcb', '30', '40'])
    os.system('rm '+mhddatafile)
    os.system('rm '+nextmhdfile)

if __name__ == '__main__':
    start_time = time.time()
    if sys.version_info < (3, 5):
        raise Exception('This script requires Python version 3.5+')
    if tp.session.connected():
        raise Exception('This script must be run in batch mode')
    ########################################
    ### SET GLOBAL INPUT PARAMETERS HERE ###
    RUNDIR = 'Energetics1'
    MHDDIR = os.path.join(RUNDIR, 'GM', 'IO2','partB')
    IEDIR = os.path.join(RUNDIR)
    IMDIR = os.path.join(RUNDIR)
    SCRIPTDIR = './'
    OUTPUTPATH = os.path.join(SCRIPTDIR, 'output')
    PNGPATH = os.path.join(OUTPUTPATH, 'png')
    ########################################
    #make directories for output
    os.system('mkdir '+OUTPUTPATH)
    os.system('mkdir '+OUTPUTPATH+'/figures')
    os.system('mkdir '+OUTPUTPATH+'/indices')
    os.system('mkdir '+OUTPUTPATH+'/png')
    ########################################
    ########### MULTIPROCESSING ###########
    #Pytecplot requires spawn method
    multiprocessing.set_start_method('spawn')

    # Get the set of data files to be processed (solution times)
    all_solution_times = sorted(glob.glob(MHDDIR+'/*.plt.gz'),
                                key=makevideo.time_sort)
    #Pick up only the files that haven't been processed
    if os.path.exists(OUTPUTPATH+'/energeticsdata'):
        parseddonelist, parsednotdone = [], []
        donelist = glob.glob(OUTPUTPATH+'/png/*')
        for png in donelist:
            parseddonelist.append(png.split('/')[-1].split('.')[0])
        for plt in all_solution_times:
            parsednotdone.append(plt.split('e')[-1].split('.')[0])
        solution_times = [MHDDIR+'/3d__var_1_e'+item+'.plt.gz' for item
                    in parsednotdone if item not in parseddonelist]
    else:
        solution_times = all_solution_times
    print(solution_times)
    print(len(solution_times))
    numproc = multiprocessing.cpu_count()-1

    # Set up the pool with initializing function and associated arguments
    num_workers = min(numproc, len(solution_times))
    pool = multiprocessing.Pool(num_workers, initializer=init,
            initargs=(RUNDIR, MHDDIR, IEDIR, IMDIR, SCRIPTDIR, OUTPUTPATH,
                      PNGPATH, all_solution_times))
    try:
        # Map the work function to each of the job arguments
        pool.map(work, solution_times)
    finally:
        # Join the process pool before exit so Tec cleans up & no core dump
        pool.close()
        pool.join()
        os.system('rm -r '+MHDDIR+'/*/')
    ########################################

    #Combine and delete individual energetics files
    write_disp.combine_hdfs(os.path.join(OUTPUTPATH,'energeticsdata'),
                            OUTPUTPATH)
    os.system('rm -r '+OUTPUTPATH+'/energeticsdata')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
