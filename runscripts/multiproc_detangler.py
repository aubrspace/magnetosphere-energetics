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
from global_energetics import write_disp

def init(rundir, mhddir, iedir, imdir, scriptdir, outputpath, pngpath):
    # !!! IMPORTANT !!!
    # Must register stop at exit to ensure Tecplot cleans
    # up all temporary files and does not create a core dump
    atexit.register(tp.session.stop)
    #globalize variables for each worker
    global RUNDIR, MHDDIR, IEDIR, IMDIR, SCRIPTDIR, OUTPUTPATH, PNGPATH, ID
    RUNDIR = rundir
    MHDDIR = mhddir
    IEDIR = iedir
    IMDIR = imdir
    SCRIPTDIR = scriptdir
    OUTPUTPATH = outputpath
    PNGPATH = pngpath
    '''
    ID = id(multiprocessing.current_process())
    os.system('mkdir '+MHDDIR+'/'+str(ID))
    os.system('cp '+MHDDIR+'/*.sat '+MHDDIR+'/'+str(ID)+'/')
    '''

def work(mhddatafiles):
    # Load data and change zone name
    print(mhddatafiles)
    tp.new_layout()
    savedates=[]
    field_data = tp.data.load_tecplot(mhddatafiles)
    for zone in field_data.zones():
        datestr = field_data.zone(zone).aux_data['SAVEDATE']
        date = datestr.split(' ')[2]
        time = datestr.split(' ')[4]
        datetime = ' '.join([date,time])
        savedates.append(dt.datetime.strptime(datetime,'%Y/%m/%d %H:%M:%S'))
    maxdate_index = 0
    for date in enumerate(savedates):
        if date[1]>savedates[maxdate_index]:
            maxdate_index=date[0]
    mhddatafiles.pop(maxdate_index)
    for mhdfile in mhddatafiles:
        os.system('rm '+mhdfile)

if __name__ == '__main__':
    start_time = time.time()
    if sys.version_info < (3, 5):
        raise Exception('This script requires Python version 3.5+')
    if tp.session.connected():
        raise Exception('This script must be run in batch mode')
    ########################################
    ### SET GLOBAL INPUT PARAMETERS HERE ###
    RUNDIR = './'
    MHDDIR = os.path.join(RUNDIR)
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
    filelist = glob.glob(MHDDIR+'/*.plt')
    numproc = multiprocessing.cpu_count()-1
    print(filelist)
    solution_times = []
    for afile in filelist:
        match = '-'.join(afile.split('-')[0:-1])
        pairs = [mhd for mhd in filelist if mhd.startswith(match)]
        #if (len(pairs)>1):
        if (len(pairs)>1) and (solution_times.count(pairs)==0):
            solution_times.append(pairs)
            print(solution_times.count(pairs))
    print(solution_times)

    # Set up the pool with initializing function and associated arguments
    num_workers = min(numproc, len(solution_times))
    pool = multiprocessing.Pool(num_workers, initializer=init,
            initargs=(RUNDIR, MHDDIR, IEDIR, IMDIR, SCRIPTDIR, OUTPUTPATH,
                      PNGPATH))
    try:
        # Map the work function to each of the job arguments
        pool.map(work, solution_times)
    finally:
        # Join the process pool before exit so Tec cleans up & no core dump
        pool.close()
        pool.join()
    ########################################

    '''
    #Combine and delete individual energetics files
    write_disp.combine_hdfs(os.path.join(OUTPUTPATH,'energeticsdata'),
                            OUTPUTPATH)
    os.system('rm -r '+OUTPUTPATH+'/energeticsdata')
    '''
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
