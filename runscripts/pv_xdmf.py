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
from pv_magnetopause import *
#from pv_magnetopause import (get_time, time_sort, read_aux, setup_pipeline,
#                             display_visuals,update_rotation,read_tecplot,
#                             get_dipole_field,tec2para,update_fluxVolume,
#                             update_fluxResults,export_datacube,
#                             update_datacube, fix_names,get_surface_flux)
import magnetometer
from magnetometer import(get_stations_now,update_stationHead)

#if __name__ == "__main__":
if True:
    start_time = time.time()
    if 'Users' in os.getcwd():
        path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
        outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
        herepath=os.getcwd()
    elif 'aubr' in os.getcwd():
        path='/home/aubr/Code/testSWMF/run_test/GM/IO2/'
        outpath='/home/aubr/Code/swmf-energetics/xdmftest/'
        herepath=os.getcwd()
    elif os.path.exists('/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'):
        path='/Users/ngpdl/Code/swmf-energetics/localdbug/vis/'
        outpath='/Users/ngpdl/Code/swmf-energetics/vis_com_pv/'
        herepath='/Users/ngpdl/Code/swmf-energetics/'
    elif os.path.exists('/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'):
        path='/home/aubr/Code/swmf-energetics/ccmc_2022-02-02/copy_paraview/'
        outpath='/home/aubr/Code/swmf-energetics/output_hyperwall3_redo/'
        herepath='/home/aubr/Code/swmf-energetics/'
    print('path: ',path)
    print('outpath: ',outpath)
    print('herepath: ',herepath)
    #filelist = sorted(glob.glob(path+'*.xmf'),
    #                  key=pv_magnetopause.time_sort)
    filelist = glob.glob(path+'*.xmf')
    renderView1 = GetActiveViewOrCreate('RenderView')

    for infile in filelist:
        print(infile)
        head = Xdmf3ReaderS(FileName=infile)
        head.CellArrays = ['B_x [nT]', 'B_y [nT]', 'B_z [nT]',
                           'J_x [`mA/m^2]', 'J_y [`mA/m^2]', 'J_z [`mA/m^2]',
                           'P [nPa]', 'Rho [amu/cm^3]', 'Status',
                           'U_x [km/s]','U_y [km/s]','U_z [km/s]',
                           'theta_1 [deg]','theta_2 [deg]',
                           'phi_1 [deg]','phi_2 [deg]']
        #headDisplay = Show(head, renderView1, 'UnstructuredGridRepresentation')
        # create a new 'Calculator'
        #calculator1 = Calculator(registrationName='Calculator1', Input=head)
        #calculator1.AttributeType = 'Cell Data'
        #calculator1.Function = 'sqrt("B_x [nT]"^2+"B_y [nT]"^2+"B_z [nT]"^2)'
        #calculator1.ResultArrayName = 'Bmag'

# Properties modified on calculator1
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
