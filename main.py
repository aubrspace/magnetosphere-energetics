#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import logging
import numpy as np
from numpy import pi
import datetime as dt
import spacepy
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
#import global_energetics
from global_energetics.extract import magnetosphere
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics.write_disp import write_to_hdf

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    #pass in arguments
    mhddatafile = '3d__var_1_e20140219-031700-018.plt'
    future = '3d__var_1_e20140219-031800-036.plt'
    OUTPATH = 'temp/'
    PNGPATH = 'temp/'
    OUTPUTNAME = 'testoutput1.png'

    '''
1947849 2014 02 19 03 17 00 018  3.93901E-002  2.56463E+000 -1.70966E-015  1.69500E-017 -2.75944E-018  1.04916E-002  3.64284E+000 -4.80667E-001 -9.16607E+000  1.42366E-005  1.81455E+001 -4.43327E+001 -4.90226E+001  1.18495E+002  9.75989E+001

    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''

    #python objects
    field_data = tp.data.load_tecplot([mhddatafile,future])
    field_data.zone(0).name = 'global_field'
    field_data.zone(1).name = 'future'
    main = tp.active_frame()
    main.name = 'main'

    #Caclulate initial surface
    #for mode in ['iso_betastar', 'ps','qDp','rc','nlobe','slobe']:
    with tp.session.suspend():
        mesh, data = magnetosphere.get_magnetosphere(field_data,
                                    outputpath=OUTPATH,
                                 analysis_type='energy_virial',
                                    tail_cap=-20,
                                    save_mesh=False,
                                    integrate_surface=True,
                                    integrate_volume=True)
    if True:#manually switch on or off
        #adjust view settings
        proc = 'Multi Frame Manager'
        cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
        #tp.macro.execute_extended_command(command_processor_id=proc,
        #                                  command=cmd)
        #mode = ['iso_day', 'other_iso', 'iso_tail', 'inside_from_tail']
        mode = ['iso_day']
        save=False
        zone_hidekeys = ['sphere', 'box','lcb','shue','future',
                         'mp_iso_betastar']
        for frame in enumerate(tp.frames()):
            frame[1].activate()
            if frame[0]==0:
                pass
            if frame[0]==1:
                pass
            if frame[0]==2:
                pass
            if frame[0]==3:
                #save = True
                timestamp = True
            view_set.display_single_iso(frame[1], mhddatafile,
                                        mode=mode[frame[0]],
                                        save_img=save,
                                        verbose=True,
                                        zone_hidekeys=zone_hidekeys,
                                        show_timestamp=True,
                                        show_contour=False)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
