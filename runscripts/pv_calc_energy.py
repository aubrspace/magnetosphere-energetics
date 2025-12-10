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
from global_energetics.extract.pv_input_tools import (read_tecplot)
from global_energetics.extract.shared_tools import (read_aux)
from global_energetics.extract.pv_tools import create_globe
from global_energetics.extract.pv_surface_tools import (
                                                    get_numpy_surface_analysis)
from global_energetics.extract.pv_volume_tools import get_numpy_volume_analysis

global FILTER
FILTER = paraview.vtk.vtkAlgorithm # Generic "filter" object

def initial_processing(infiles:list) -> [dict,dt.datetime]:
    print('INITALIZING SURFACES & VOLUMES: ...')
    # Read aux data
    if '.dat' in infiles[1]:
        aux = read_aux(infiles[1].replace('.dat','.aux'))
    elif '.plt' in infiles[1]:
        aux = read_aux(infiles[1].replace('.plt','.aux'))
    # Get time information
    localtime = get_time(infiles[1])
    # Create a representation of Earth updated with the coord system
    #earth = create_globe(localtime,coord='gsm')

    # Setup the pipeline for 3 time steps
    settings = {'doEnergyFlux':True,
                'doVolumeEnergy':True,
                'doEntropy':True,
                'do_daynight':False,
                'surfaces':['mp','closed','lobes','inner','sheath'],
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
                           tstamp:dt.datetime,**kwargs) -> None:
    # Perform calculations
    surface_results = get_numpy_surface_analysis(surfaces,**kwargs)
    volume_results  = get_numpy_volume_analysis(volume,
                                 volume_list=['mp','closed','lobes','sheath'],
                                               **kwargs)

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
    #print(f"\033[92m Saved\033[00m {OUTPATH}/{outfile}")
    print(f"\033[92m Saved\033[00m {outfile}")

def main() -> None:
    # Locate files
    filelist = sorted(glob.glob(f'{INPATH}/*paraview*.plt'),key=time_sort)

    # Initialize variables
    tstart = get_time(filelist[0])# for relative timestamping
    renderView = GetActiveViewOrCreate('RenderView')# for view hooks

    # If we have a state ready, load it, otw do initial processing
    if True:
        # Load
        #LoadState(os.path.join(INPATH,'magnetopause_and_sheath.pvsm'),
        #          data_directory=INPATH)
        #LoadState(os.path.join(os.getcwd(),'cosmetic/magnetopause_and_sheath.pvsm'))
        LoadState(os.path.join(os.getcwd(),'cosmetic/sheath-mp-iso2.pvsm'))
        # Get view
        renderView = GetActiveView()
        # Set the heads of the pipeline
        old_past_head   = FindSource('3d__paraview_1_e20190513-195600-016')
        old_present_head= FindSource('3d__paraview_1_e20190513-195700-032')
        old_future_head = FindSource('3d__paraview_1_e20190513-195800-008')
        # Set the tails where the processing takes over
        surfaces = {'mp'    :FindSource('mp'),
                    'closed':FindSource('closed'),
                    'lobes' :FindSource('lobes'),
                    'inner' :FindSource('inner'),
                    'sheath':FindSource('sheath')}
        volume = FindSource('merged')
    else:
        pipeline,surfaces,volume,localtime = initial_processing(filelist)
        perform_integrations(surfaces,volume,localtime)
        old_past_head   = FindSource(filelist[0].split('/')[-1].split('.')[0])
        old_present_head= FindSource(filelist[1].split('/')[-1].split('.')[0])
        old_future_head = FindSource(filelist[2].split('/')[-1].split('.')[0])

    #for ifile,infile in enumerate(filelist[1:-1]):
    for ifile,infile in enumerate(filelist[1:2]):
        # Set output file name
        outfile=infile.split('_1_e')[-1].replace('.plt','.png')
        if os.path.exists(OUTPATH.replace('analysis','png')+outfile):
            pass# Skip
        else:
            print(f"{infile.split('/')[-1]}")
            localtime = get_time(infile)
            '''
            # Read aux data
            aux = read_aux(infile.replace('.plt','.aux'))
            # Get time information
            localtime = get_time(infile)
            # Update time
            timestamp = FindSource('time')
            timestamp.Text = str(localtime)

            # Update the pipeline
            new_data = read_tecplot(filelist[ifile+1])
            old_data = old_past_head.Input
            # Set old_future_head to new_data
            #     old_present_head to old_future_file
            #     old_past_head to old_present_file
            #     delete old_past_file
            old_future_head.Input = new_data
            old_present_head.Input= FindSource(filelist[ifile].split('/')[-1])
            old_past_head.Input = FindSource(filelist[ifile-1].split('/')[-1])
            Delete(old_data)
            del old_data

            '''
            # Update
            renderView.Update()
            for surf_name,surface in surfaces.items():
                surface.UpdatePipeline()
            volume.UpdatePipeline()

            # Crunch the numbers
            perform_integrations(surfaces,volume,localtime)

            # Save screenshot
            SaveScreenshot(f"{OUTPATH.replace('analysis','png')}/{outfile}",
                           GetLayout())
            print(f"\033[36m Saved \033[00m {outfile}")

if True:
    start_time = time.time()

    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    global INPATH,OUTPATH

    herepath=os.getcwd()

    #INPATH  = os.path.join(herepath,'weakdip_50_katus/GM/')
    #OUTPATH = os.path.join(herepath,'weakdip_50_katus/GM/analysis')
    #INPATH  = os.path.join(herepath,'localdbug/weak_dipole/')
    #OUTPATH = os.path.join(herepath,'localdbug/weak_dipole/')
    #INPATH   = os.path.join(herepath,'data/large/GM/IO2/')
    #OUTPATH  = os.path.join(herepath,'data/analysis/')
    INPATH   = os.path.join(herepath,'run_may2019/GM/IO2/')
    OUTPATH   = os.path.join(herepath,'outputs_may2019/')

    main()

    # rudimentary timing
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
