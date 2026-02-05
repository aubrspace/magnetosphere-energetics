#### standard
import os,sys
import paraview
import time
import glob
import numpy as np
import datetime as dt
#import pandas as pd
#### paraview
from paraview.simple import *
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10
from paraview.vtk.numpy_interface import dataset_adapter as dsa
#### Custom packages ####
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from global_energetics.makevideo import (get_time, time_sort)
from global_energetics.extract import pv_ionosphere
from global_energetics.extract.pv_tools import (create_globe,
                                                create_stations,
                                                load_maggrid)

def initial_processing(infile:str) -> dict:
    print(f"INITIAL PROCESSING ON {infile.split('/')[-1]} ...")
    pvstate = {}

    # Define helpful variables
    renderView1 = GetActiveViewOrCreate('RenderView')
    localtime = get_time(infile)
    magfile = (f"{GMPATH}/mag_grid_global_e"+
                                   localtime.strftime('%Y%m%d-%H%M%S')+'.out')
    outfile = infile.split('_1_')[-1].split('.')[0]+'.png'

    # Setup the pipeline by calling in individual elements
    pvstate['ionosphere'] = pv_ionosphere.load_ie(infile)
    pvstate['earth']      = create_globe(localtime,coord='mag',
                              blue_marble_loc=f'{INPATH}/bluemarble_map.png')
    pvstate['stations']   = create_stations(localtime,coord='mag')
    pvstate['maggrid']    = load_maggrid(magfile,coord='mag')

    return pvstate

def main() -> None:
    # Find input files to process
    filelist = sorted(glob.glob(f"{IEPATH}/*.tec"),key=time_sort)

    global T0 #define a global variable for the earliest file
    T0 = get_time(filelist[0])

    if True:# Starting from scratch
        pvstate = initial_processing(filelist[0])
    else:# OR load a state that already been done
        pass

    for ifile,infile in enumerate(filelist):
        pass

if True:
    start_time = time.time()
    global HERE,INPATH,IEPATH,GMPATH,OUTPATH

    HERE    = os.getcwd()
    INPATH  = f"{HERE}/data"
    GMPATH  = f"{INPATH}/large/GM/IO2"
    IEPATH  = f"{INPATH}/large/IE/ionosphere"
    OUTPATH = f"{HERE}/outputs"

    main()

    # rudimentary timing
    dtime = time.time()-start_time
    print('DONE')
    print(f'--- {int(dtime/60):d}min {np.mod(dtime,60):.2f}s ---')
    """

    filelist = sorted(glob.glob(IEpath+'*.tec'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    #FAC_all = pd.DataFrame()
    for i,infile in enumerate(filelist[0:1]):
        localtime = get_time(infile)
        print(localtime)
        # magfile
        t = localtime
        magfile = (f"{inpath}GM/IO2/mag_grid_global_e"+
                   f"{t.year}{t.month:02d}{t.day:02d}-"+
                   f"{t.hour:02d}{t.minute:02d}{t.second:02d}.out")
        outfile = 't'+str(i)+infile.split('_1_')[-1].split('.')[0]+'.png'
        iono = pv_ionosphere.load_ie(infile)
        earth = create_globe(localtime,coord='mag')
        stations = create_stations(localtime,coord='mag')
        maggrid = load_maggrid(magfile,coord='mag')
        '''
        FAC  = pv_ionosphere.integrate_Jr(iono)
        FAC2 = pv_ionosphere.id_R1R2_currents(iono)
        for key in FAC2:
            FAC[key] = FAC2[key]
        oval = pv_ionosphere.id_RIM_oval(iono)
        for key in oval:
            FAC[key] = oval[key]
        ocflb = pv_ionosphere.id_ocflb(iono)
        for key in ocflb:
            FAC[key] = ocflb[key]
        FAC['time'] = [localtime]
        FAC_all = pd.concat([FAC_all,pd.DataFrame(data=FAC)])
        #ResetSession()
    # Sort and clean up results for export as hdf5 file
    print(FAC_all)
    FAC_all.index = FAC_all['time']
    FAC_all.drop(columns=['time'],inplace=True)
    FAC_all.to_csv(outpath+'integrated_currents.csv')
        '''
    """
