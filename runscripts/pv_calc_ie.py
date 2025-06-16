import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')

import time
import glob
import numpy as np
import datetime as dt
import pandas as pd
#### import the simple module from paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from pv_tools import create_globe, create_stations, load_maggrid
import pv_ionosphere
from makevideo import (get_time, time_sort)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    herepath=os.getcwd()
    #inpath = os.path.join(herepath,'gannon-storm/data/large/')
    #IEpath = os.path.join(inpath,'IE/ionosphere/')
    #outpath= os.path.join(herepath,'gannon-storm/outputs/vis/')
    inpath = os.path.join(herepath,'Starlink_Pleiades/')
    IEpath = os.path.join(inpath,'IE/')
    outpath= os.path.join(herepath,'outputs_starlink_hires/ie/')

    filelist = sorted(glob.glob(IEpath+'*.tec'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    FAC_all = pd.DataFrame()
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
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
