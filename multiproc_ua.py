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
from spacepy.pybats import gitm
import global_energetics
from global_energetics.preplot import load_hdf5_data, IDL_to_hdf5
from global_energetics.extract import thermos as therm
from global_energetics.extract import view_set
from global_energetics import write_disp, makevideo

def init(eventpath, outputpath,all_solution_times):
    '''Initialization function for each new spawn
    Inputs
        rundir, mhddir, etc. - filepaths for input/output
        all_solution_times - list of all files for finding adjacents
    '''
    #globalize variables for each worker
    global CONTEXT
    CONTEXT = {
            'EVENTPATH' : eventpath,
            'OUTPUTPATH' : outputpath,
            'ALL_SOLUTION_TIMES' : all_solution_times,
            'id' : id(multiprocessing.current_process()),
            }

def work(file):
    print(str(CONTEXT['id'])+'working on: '+file)
    #get timestamp
    timestamp = makevideo.get_time(file)
    ##read single file
    dat = gitm.GitmBin(file)
    density = {}
    for altitude in [210,310,410,510]:
        density.update(therm.get_averaged_values(dat,alt=altitude))
    #save data
    therm.save_tofile(file,timestamp,
                      outputdir=CONTEXT['OUTPUTPATH'],**density)

    ##Display

    ##Clean up

def single_event_run(eventpath, **kwargs):
    """Runs full analysis on single event
    Inputs
        eventdir (str)
        kwargs:
            stormdir
            outname
    """
    #EXAMPLE file paths to explain naming convention
    #   eventpath (inputs)
    #   -> it220202_081000_000.tec
    #   -> ...
    #   outputdir
    #   -> energetics (outputpath)
    #   -> png
    #   -> figures

    ##Setup paths
    outputdir = os.path.join(kwargs.get('outputdir','ua_output'))
    outputpath = os.path.join(outputdir,'energetics')

    ##Make directories for output
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(outputpath, exist_ok=True)

    ##Run
    multiprocess(eventpath, outputpath)

    ##Combine HDF5 files
    write_disp.combine_hdfs(outputpath,outputdir, progress=False,
             combo_name=kwargs.get('outname','ua_energetics.h5'))

    ##Make a video
    pass

    ##Clean up
    shutil.rmtree(outputpath)

    ##Report
    print('\nCOMPLETE '+time.ctime()+'\n')

def multiprocess(eventpath, outputpath,**kwargs):
    """Calls multiprocesssing modules to perform analysis
    inputs
        eventpath
        outputpath
        kwargs:
            checkfiles (bool)
    """
    # Get the set of data files to be processed (solution times)
    all_solution_times = sorted(glob.glob(eventpath+'/3DALL*.bin'),
                                key=makevideo.time_sort)
    solution_times = all_solution_times
    print(len(solution_times))
    numproc = multiprocessing.cpu_count()-1

    # Set up the pool with initializing function and associated arguments
    num_workers = min(numproc, len(solution_times))
    pool = multiprocessing.Pool(num_workers, initializer=init,
            initargs=(eventpath, outputpath, all_solution_times))
    try:
        # Map the work function to each of the job arguments
        pool.map(work, solution_times)
    finally:
        # Join the process pool before exit so Tec cleans up & no core dump
        pool.close()
        pool.join()

def create_video(inputdir,outputdir,**kwargs):
    """Compiles video found at inpath and saves to outpath
    inputs
        inputdir
        kwargs:
            vid_resolution
            vid_framerate
    """
    #Combine and delete individual hdf5 files
    if os.path.exists(inputdir):
        ##Create videos from images
        #Video settings
        vid_resolution = 400
        vid_framerate = 16
        for cut in ['y','z']:
            folder = inputdir+'/'+cut+'figures/'
            frame_loc = makevideo.set_frames(folder)
            makevideo.vid_compile(frame_loc, outputdir, vid_framerate,
                                  '2Dmotion_'+cut)


if __name__ == '__main__':
    start_time = time.time()
    if sys.version_info < (3, 5):
        raise Exception('This script requires Python version 3.5+')
    #Still using spawn method although there are no tecplot calls
    multiprocessing.set_start_method('spawn')

    ########################################
    ### SET GLOBAL INPUT PARAMETERS HERE ###
    nJOULEdir = '/home/aubr/Code/GITM2/run/UA/joule_run'
    nENERGYdir = '/home/aubr/Code/GITM2/run/UA/nEnergy_run'
    NOHPI = '/home/aubr/Code/GITM2/run/UA/no_hpi'
    ########################################

    single_event_run(nJOULEdir, outname='ua_njoule.h5')
    single_event_run(nENERGYdir, outname='ua_nenergy.h5')
    single_event_run(NOHPI, outname='ua_nohpi.h5')

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
