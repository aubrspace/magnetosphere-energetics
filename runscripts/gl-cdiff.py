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

def parse_infiles(inpath,outpath):
    # Get the set of data files to be processed (solution times)
    all_solution_times = sorted(glob.glob(inpath+'/3d__var_1*.plt'),
                                key=makevideo.time_sort)[0::]
    # Prune any repeat times
    times = [makevideo.get_time(f) for f in all_solution_times]
    _,unique_i = np.unique(times,return_index=True)
    all_solution_times = [all_solution_times[i] for i in unique_i]
    #Pick up only the files that haven't been processed
    if os.path.exists(os.path.join(outpath,'energeticsdata')):
        parseddonelist, parsednotdone = [], []
        donelist = glob.glob(outpath+'/png/*.png')
        for png in donelist:
            parseddonelist.append(png.split('/')[-1].split('.')[0])
        for plt in all_solution_times:
            parsednotdone.append(plt.split('e')[-1].split('.')[0])
        solution_times=[os.path.join(inpath,'3d__var_1_e'+item+'.plt')for item
                        in parsednotdone if item not in parseddonelist]
    else:
        solution_times = all_solution_times
    print('files remaining: ',len(solution_times))
    return solution_times, all_solution_times

def energetics_analysis(infiles,outpath):
    #Reset session
    tp.new_layout()
    #python objects
    #oggridfile = 'ideal_conserve/GM/IO2/3d__volume.plt'
    oggridfile = ''
    field_data = tp.data.load_tecplot(infiles)
    filetime = makevideo.get_time(infiles[0])
    futuretime = makevideo.get_time(infiles[1])
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
                                      disp_result=False,
                                      do_cms=True,
                                      do_central_diff=True,
                                      do_1Dsw=False,
                                      analysis_type='energy_mass_mag',
                                      #modes=['sphere'],
                                      #sp_rmax=10,
                                      #sp_rmin=3,
                                      #keep_zones='all',
                                      modes=['iso_betastar','closed',
                                             'nlobe','slobe'],
                                      #modes=['xslice'],
                                      inner_r=4,
                                      customTerms={'test':'TestArea [Re^2]'},
                                      do_interfacing=True,
                                      integrate_line=False,
                                      integrate_surface=False,
                                      integrate_volume=True,
                                      truegridfile=oggridfile,
                                      verbose=False,
                                      extract_flowline=False,
                                      outputpath=outpath)
    # IE data
    inpath = '/'.join([f for f in infiles[0].split('/')][0:-3])+'/IE/ionosphere/'
    iedatafile, success = find_IE_matched_file(inpath,filetime)
    future_iefile, _ = find_IE_matched_file(inpath,futuretime)
    if os.path.exists(iedatafile):
        dataset = tp.data.load_tecplot(iedatafile,
                                    read_data_option=ReadDataOption.Append)
        if dataset.zone('IonN*') is not None:
            dataset.zone('IonN*').name = 'ionosphere_north'
            do_north = True
        if dataset.zone('IonS*') is not None:
            dataset.zone('IonS*').name = 'ionosphere_south'
            do_south = True
        dataset = tp.data.load_tecplot(future_iefile,
                        read_data_option=ReadDataOption.Append)
        if dataset.zone('IonN*') is not None:
            dataset.zone('IonN*').name = 'future_ionosphere_north'
            do_north = True
        if dataset.zone('IonS*') is not None:
            dataset.zone('IonS*').name = 'future_ionosphere_south'
            do_south = True
        if do_north*do_south:
            ionosphere.get_ionosphere(dataset,
                                              verbose=False,
                                              hasGM=True,
                                              eventtime=filetime,
                                              analysis_type='mag',
                                              integrate_surface=True,
                                              integrate_line=False,
                                              integrate_contour=False,
                                              do_interfacing=False,
                                              do_cms=True,
                                              do_central_diff=True,
                                              outputpath=outpath)
    print(os.path.join(outpath,'png',outputname+'.png'))
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
    #[print(str(f)+'\n') for f in file_list]
    # If just a single file is requested
    if '-f' in sys.argv or '--file' in sys.argv:
        if '-f' in sys.argv:
            infile = sys.argv[sys.argv.index('-f')+1]
        elif '--file' in sys.argv:
            infile = sys.argv[sys.argv.index('--file')+1]
        nowfile = os.path.join(inpath,infile)

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
        print('CDIFF previous: ',previousfile)
        print('CDIFF now: ',nowfile)
        print('CDIFF next: ',nextfile)
        if previousfile not in file_list:
            print('CDIFF '+previousfile+' already done....')
            exit()
        #energetics_analysis([nowfile,nextfile_mirror],outpath)
        energetics_analysis([previousfile,nextfile_mirror],outpath)
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
