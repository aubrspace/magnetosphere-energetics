#/usr/bin/env python
"""Example from the pytecplot documentation
"""
import time
import logging
import atexit, os, multiprocessing, sys
import numpy as np
import pandas as pd
from numpy import pi
import datetime as dt
import glob
import gzip
import shutil
#import spacepy
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetosphere
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics.extract.view_set import twodigit
from global_energetics import write_disp, makevideo

def init(rundir, mhddir, iedir, imdir, scriptdir, outputpath, pngpath,
         all_solution_times):
    '''Initialization function for each new spawn
    Inputs
        rundir, mhddir, etc. - filepaths for input/output
        all_solution_times - list of all files for finding adjacents
    '''
    # !!! IMPORTANT !!!
    # Must register stop at exit to ensure Tecplot cleans
    # up all temporary files and does not create a core dump
    atexit.register(tp.session.stop)
    #globalize variables for each worker
    global CONTEXT
    CONTEXT = {
            'RUNDIR' : rundir,
            'MHDDIR' : mhddir,
            'IEDIR' : iedir,
            'IMDIR' : imdir,
            'SCRIPTDIR' : scriptdir,
            'OUTPUTPATH' : outputpath,
            'PNGPATH' : pngpath,
            'ALL_SOLUTION_TIMES' : all_solution_times,
            'id' : id(multiprocessing.current_process()),
            }
    os.makedirs(mhddir+'/'+str(CONTEXT['id']), exist_ok=True)

def work(mhddatafile):
    print(str(CONTEXT['id'])+'working on: '+mhddatafile)
    #Unzip copies to spawn's local folder
    for sol in enumerate(CONTEXT['ALL_SOLUTION_TIMES']):
        if mhddatafile == sol[1]:
            if sol[0]!=len(CONTEXT['ALL_SOLUTION_TIMES'])-1:
                nextmhdfile = CONTEXT['ALL_SOLUTION_TIMES'][sol[0]+1]
            else:
                nextmhdfile = mhddatafile
    temp_cfile = CONTEXT['MHDDIR']+'/'+str(CONTEXT['id'])+'/current.plt'
    temp_nfile = CONTEXT['MHDDIR']+'/'+str(CONTEXT['id'])+'/next.plt'
    temp_files = [temp_cfile, temp_nfile]
    for f in enumerate([mhddatafile, nextmhdfile]):
        with gzip.open(f[1],'r')as fin,open(temp_files[f[0]],'wb')as fout:
            shutil.copyfileobj(fin, fout)
    #Load data into tecplot and setup field zone names
    tp.new_layout()
    field_data = tp.data.load_tecplot(temp_files)
    field_data.zone(0).name = 'global_field'
    field_data.zone(1).name = 'future'
    OUTPUTNAME = mhddatafile.split('e')[-1].split('.plt')[0]
    #Caclulate surfaces
    magnetosphere.get_magnetosphere(field_data,save_mesh=False,
                                    do_cms=True,
                                    analysis_type='virial_biotsavart',
                                    outputpath=CONTEXT['OUTPUTPATH'])
    #get supporting module data for this timestamp
    #satzones = satellites.get_satellite_zones(field_data,
    #                              CONTEXT['MHDDIR']+'/'+str(CONTEXT['id']))
    if False:#manually switch on or off
        #adjust view settings
        proc = 'Multi Frame Manager'
        cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
        tp.macro.execute_extended_command(command_processor_id=proc,
                                          command=cmd)
        mode = ['iso_day', 'other_iso', 'iso_tail', 'inside_from_tail']
        zone_hidekeys = ['sphere', 'box','lcb','shue','future',
                         'mp_iso_betastar']
        timestamp=True
        for frame in enumerate(tp.frames()):
            frame[1].activate()
            if frame[0]==0:
                pass
            if frame[0]==1:
                pass
            if frame[0]==2:
                pass
            if frame[0]==3:
                save = True
                timestamp = True
            view_set.display_single_iso(frame[1], mhddatafile,
                                        mode=mode[frame[0]],
                                        pngpath=CONTEXT['PNGPATH'],
                                        outputname=OUTPUTNAME,
                                        IDstr=str(CONTEXT['id']),
                                        show_contour=False,
                                        timestamp_pos=[4,20],
                                        zone_hidekeys=zone_hidekeys,
                                        show_timestamp=timestamp)
    else:
        with open(CONTEXT['PNGPATH']+'/'+OUTPUTNAME+'.png','wb') as png:
            png.close()
    #Remove unzipped copies now that work is done for that file
    for f in temp_files:
        os.remove(f)

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
    os.makedirs(OUTPUTPATH, exist_ok=True)
    os.makedirs(OUTPUTPATH+'/figures', exist_ok=True)
    os.makedirs(OUTPUTPATH+'/indices', exist_ok=True)
    os.makedirs(OUTPUTPATH+'/png', exist_ok=True)
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
        donelist = glob.glob(OUTPUTPATH+'/png/*.png')
        #donelist = glob.glob(OUTPUTPATH+'/energeticsdata/*.h5')
        for png in donelist:
            parseddonelist.append(png.split('/')[-1].split('.')[0])
            #yr,mo,dy,hr,mn = png.split('-')
            #yr = yr.split('_')[-1]
            #mn = mn.split('.')[0]
            #parsed = (yr+twodigit(int(mo))+twodigit(int(dy))+'-'+
            #                           twodigit(int(hr))+twodigit(int(mn)))
            #parseddonelist.append(parsed)
        for plt in all_solution_times:
            parsednotdone.append(plt.split('e')[-1].split('.')[0])
            #parsednotdone.append(plt.split('e')[-1].split('.')[0].split(
            #                                                     '00-')[0])
        solution_times = [MHDDIR+'/3d__var_1_e'+item+'.plt.gz' for item
                    in parsednotdone if item not in parseddonelist]
    else:
        solution_times = all_solution_times
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
        for f in glob.glob(MHDDIR+'/*'):
            if os.path.isdir(f):
                os.removedirs(f)
    ########################################

    #Combine and delete individual energetics files
    if os.path.exists(OUTPUTPATH+'/energeticsdata'):
        write_disp.combine_hdfs(os.path.join(OUTPUTPATH,'energeticsdata'),
                                OUTPUTPATH)
        shutil.rmtree(OUTPUTPATH+'/energeticsdata/')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
