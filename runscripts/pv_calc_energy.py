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
from global_energetics.extract.pv_magnetosphere import setup_pipeline
from global_energetics.extract.pv_input_tools import (read_tecplot,read_aux)
from global_energetics.extract.pv_tools import create_globe
from global_energetics.extract.pv_surface_tools import (
                                                    get_numpy_surface_analysis)
from global_energetics.extract.pv_volume_tools import get_numpy_volume_analysis

global FILTER
FILTER = paraview.vtk.vtkAlgorithm # Generic "filter" object

def initial_processing(infile:str) -> [dict,dt.datetime]:
    print('INITALIZING SURFACES & VOLUMES: ...')
    # Read aux data
    aux = read_aux(infile.replace('.dat','.aux'))
    # Get time information
    localtime = get_time(infile)
    # Create a representation of Earth updated with the coord system
    #earth = create_globe(localtime,coord='gsm')

    # Setup the pipeline
    pipeline_dict = setup_pipeline(infile,doEnergyFlux=True,
                                          doVolumeEnergy=True,
                                          do_daynight=False,
                                     surfaces=['mp','closed','lobes','inner'],
                                          tail_x=-60,
                                          aux=aux,
                                          path=OUTPATH)

    '''
    surf_colors = {'mp':np.array([245,245,245])/255,#white
                   'closed':np.array([235,70,7])/255,#orange-red
                   'lobes':np.array([7,235,229])/255,#cyan
                   'inner':np.array([225,235,7])/255}#yellow
    renderView = GetActiveView()# for view hooks
    for surf,filt in pipeline_dict['surfaces'].items():
        display = Show(filt,renderView,'GeometryRepresentation')
        ColorBy(display, None)# solid color, no variable contour
        display.AmbientColor = surf_colors[surf]
        display.DiffuseColor = surf_colors[surf]
        display.Opacity = 0.3
    '''
    return pipeline_dict,localtime

def perform_integrations(pipeline_dict:dict,tstamp:dt.datetime) -> None:
    # Perform calculations
    surface_results = get_numpy_surface_analysis(pipeline_dict['surfaces'])
    volume_results = get_numpy_volume_analysis(pipeline_dict['volumes'])

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
        pipeline_dict,localtime = initial_processing(filelist[0])
        perform_integrations(pipeline_dict,localtime)

    #TODO finish this once the initial processing is setup and the state is saved
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
