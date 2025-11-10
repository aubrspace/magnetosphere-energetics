#### standard
import os,sys
import time
import glob
import numpy as np
import datetime as dt
#### paraview
import paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
#### Custom packages #####
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import global_energetics
from global_energetics.makevideo import time_sort, get_time
from global_energetics.extract.pv_magnetosphere import (setup_pipeline,
                                                        generate_surfaces,
                                                        generate_volumes,
                                                        merge_times,
                                                        update_merge)
from global_energetics.extract.pv_input_tools import (read_tecplot,read_aux)
from global_energetics.extract.pv_tools import create_globe
from global_energetics.extract.pv_surface_tools import (
                                                    get_numpy_surface_analysis)
from global_energetics.extract.pv_volume_tools import get_numpy_volume_analysis

global FILTER
FILTER = paraview.vtk.vtkAlgorithm # Generic "filter" object

def initial_processing(infiles:list) -> [dict,dt.datetime]:
    print('INITALIZING SURFACES & VOLUMES: ...')
    # Read aux data
    aux = read_aux(infiles[1].replace('.dat','.aux'))
    # Get time information
    localtime = get_time(infiles[1])
    # Create a representation of Earth updated with the coord system
    #earth = create_globe(localtime,coord='gsm')

    # Setup the pipeline for 3 time steps
    settings = {'doEnergyFlux':True,
                'doVolumeEnergy':True,
                'do_daynight':False,
                'surfaces':['mp','closed','lobes','inner'],
                'tail_x':-60,
                'aux':aux}
    past    = setup_pipeline(infiles[0],**settings)
    present = setup_pipeline(infiles[1],**settings)
    future  = setup_pipeline(infiles[2],**settings)

    # Merge 3 times into one for just the volume data
    merged  = merge_times([past['field'],present['field'],future['field']])

    # Generate surfaces from the present time, and volumes from merged time
    pipeline = present
    surfaces = generate_surfaces(present['field'],
                                 surfaces=settings['surfaces'])
    #volumes  = generate_volumes(merged,surfaces=settings['surfaces'])
    volume   = merged

    return pipeline,surfaces,volume,localtime

def perform_integrations(surfaces:dict,volume:object,
                           tstamp:dt.datetime) -> None:
    # Perform calculations
    surface_results = get_numpy_surface_analysis(surfaces)
    volume_results = get_numpy_volume_analysis(volume,
                                        volume_list=['mp','closed','lobes'])

    # Set output filename
    outfile = ('energetics_'+tstamp.isoformat().replace(':',''
                                              ).replace('-',''
                                              ).replace('T','')+'.npz')

    # Pack up and save
    combined_results = {}
    combined_results['time'] = np.array(np.datetime64(tstamp),
                                        dtype='datetime64[s]')
    combined_results.update(surface_results)
    combined_results.update(volume_results)
    np.savez_compressed(f"{OUTPATH}/{outfile}",allow_pickle=False,
                        **combined_results)
    print(f"\033[92m Saved\033[00m {OUTPATH}/{outfile}")

def main() -> None:
    # Locate files
    filelist = sorted(glob.glob(f'{INPATH}/*paraview*.dat'),key=time_sort)

    # Initialize variables
    tstart = get_time(filelist[0])# for relative timestamping
    renderView1 = GetActiveViewOrCreate('RenderView')# for view hooks
    energetics = {}# output data -> .npz file

    # If we have a state ready, load it, otw do initial processing
    if True:
        pipeline,surfaces,volume,localtime = initial_processing(filelist)
        perform_integrations(surfaces,volume,localtime)

    '''
    for ifile,infile in enumerate(filelist):
        # Set output file name
        outfile='t'+str(ifile)+infile.split('_4_e')[-1].replace('.dat','.png')
        # Read aux data
        aux = read_aux(infile.replace('.dat','.aux'))
        # Get time information
        localtime = get_time(infile)
    '''


if True:
    start_time = time.time()

    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    global INPATH,OUTPATH

    herepath=os.getcwd()

    INPATH  = os.path.join(herepath,'localdbug/weak_dipole')
    OUTPATH = os.path.join(herepath,'localdbug/weak_dipole/analysis')

    main()

    # rudimentary timing
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
