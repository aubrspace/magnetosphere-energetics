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
import pv_ionosphere
from pv_input_tools import (read_tecplot)
from makevideo import (get_time, time_sort)

def calc_oval(ie:dict,itime:int) -> [float,float]:
    # Define variables at this time index
    jr = ie['JR [`mA/m^2]'][itime,:].reshape(181,91)
    colat = ie['Theta [deg]'][itime,:].reshape(181,91)
    lon = ie['Psi [deg]'][itime,:].reshape(181,91)

    ## Perform calculations
    # mean colatitude
    Jmax = np.max(jr,axis=1) #max along each lon
    colat_max = colat[0,:][np.argmax(jr,axis=1)]
    colat_mean = np.sum(Jmax*colat_max)/np.sum(Jmax)

    # dcolat
    Jnoon = Jmax[0]
    Jmidnight = Jmax[int(len(Jmax)/2)-1]
    colat_noon = colat_max[0]
    colat_midnight = colat_max[int(len(Jmax)/2)-1]
    dcolat = (Jnoon*(colat_noon-colat_mean)-
              Jmidnight*(colat_midnight-colat_mean))/(Jnoon+Jmidnight)
    return colat_mean,dcolat

#if __name__ == "__main__":
if True:
    start_time = time.time()
    ie_north = dict(np.load(
                       'gannon-storm/data/large/IE/ionosphere/compiled_N.npz',
                    allow_pickle=True))
    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'gannon-storm/data/large/')
    IEpath = os.path.join(inpath,'IE/ionosphere/')
    outpath= os.path.join(herepath,'gannon-storm/outputs/vis/regions/')

    filelist = sorted(glob.glob(IEpath+'*.tec'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    # Load master state
    print(f'Saving images at {outpath}:')
    LoadState(outpath+'../cpcp_vis_state2.pvsm')
    for i,infile in enumerate(filelist):
        localtime = get_time(infile)
        outfile = infile.split('/')[-1].replace('.tec','.png')
        # Find corresponding IE file
        iehead = FindSource('MergeBlocks')
        oldIE = iehead.Input
        # Read in new data and feed into the pipe, delete old data
        newIE = read_tecplot(infile,binary=False)
        iehead.Input = newIE
        Delete(oldIE)
        del oldIE
        # Find new up/down FAC isovolume levels + R1R2 defintions
        FAC2 = pv_ionosphere.id_R1R2_currents(FindSource('names'))
        # Update oval location
        oval = FindSource('Oval')
        colat_mean,dcolat = calc_oval(ie_north,filelist.index(infile))
        oval.Function = (
            f"({colat_mean}+({dcolat})*cos(3.14159/180*Psi_deg))-Theta_deg")
        # Update time
        timestamp = FindSource('time')
        timestamp.Text = str(localtime)

        # Save an image
        layout = GetLayout()
        SaveScreenshot(outpath+outfile,layout,
                       SaveAllViews=1,ImageResolution=[2560,1440])
        print(f'\t{outfile}')
    #TODO
    #   Find a timestamp from the video which has a bad fit
    #       Try a couple more checks
    #       Rank the areas of the IDs and ignore those with areas X smaller
    #           than the largest?
    #       Validate the ordering of the sign of Jr
    #           basically, if the outermost is in the +Y sector and has +
    #                      then the inner most +Y sector should be -

    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
