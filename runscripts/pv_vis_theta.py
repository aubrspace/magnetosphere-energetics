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
#### import the simple module from paraview
from paraview.simple import *
import pv_magnetopause
from pv_magnetopause import (setup_pipeline)
from pv_ionosphere import (load_ie)
from makevideo import (get_time, time_sort)
from pv_tools import (update_mag2gsm,create_globe,project_to_iono)
from pv_input_tools import (read_aux, read_tecplot)
#import pv_surface_tools
from pv_visuals import (display_visuals)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    # Set the paths NOTE cwd will be where paraview OR pvbatch is launched
    herepath=os.getcwd()
    inpath = os.path.join(herepath,'theta_aurora/outputs/data/')
    outpath= os.path.join(herepath,'theta_aurora/outputs/figures/unfiled/')

    filelist = sorted(glob.glob(inpath+'*paraview*.plt'),
                      key=time_sort)
    tstart = get_time(filelist[0])
    renderView1 = GetActiveViewOrCreate('RenderView')

    for i,infile in enumerate(filelist[-1::]):
        # Read GM File
        aux = read_aux(infile.replace('.plt','.aux'))
        localtime = get_time(infile)
        outfile = 't'+str(i)+infile.split('_1_')[-1].split('.')[0]+'.png'
        # Add in an Earth for reference
        create_globe(localtime)
        # Setup process pipeline for GM
        oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(
                                                       infile,
                                                       tail_x=-120,
                                                       dimensionless=False,
                                                       localtime=localtime,
                                                       path=herepath,
                                                       repair_status=True,
                                                       ffj=True,
                                                       doEnergyFlux=True)

        # Read IE File
        iedatafile = (inpath+
                  'it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      localtime.year-1900,
                      localtime.month,
                      localtime.day,
                      localtime.hour,
                      localtime.minute,
                      localtime.second))
        if not os.path.exists(iedatafile):
            print(iedatafile,'does not exist!')
        else:
            # Setup process pipeline for IE
            ie = load_ie(iedatafile,coord='GSM',tevent=localtime,
                         tilt=aux['BTHETATILT'])

        # Pull the closed field region
        closed = Contour(registrationName='closed', Input=field)
        closed.ComputeNormals = 1
        closed.Isosurfaces = 3
        closed.PointMergeMethod = 'Uniform Binning'
        # Project GM solution down to ionosphere
        #gm_iono = project_to_iono(closed,localtime)
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
