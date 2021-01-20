#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import numpy as np
from numpy import pi
import datetime as dt
import spacepy
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetopause
from global_energetics.extract import plasmasheet
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set

if __name__ == "__main__":
    print('\nProcessing {pltfile}\n'.format(pltfile=sys.argv[1]))
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    #pass in arguments
    datafile = sys.argv[1].split('/')[-1]
    #nameout = datafile.split('e')[1].split('-000.')[0]+'-mp'
    nameout = datafile.split('.')[0]
    print('nameout:'+nameout)
    OUTPATH = sys.argv[2]
    PNGPATH = sys.argv[3]
    PLTPATH = sys.argv[4]
    #OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-a'
    OUTPUTNAME = datafile.split('.')[0]
    main = tp.active_frame()
    main.name = 'main'

    #python objects
    field_data=tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'

    '''
    #Caclulate surfaces
    magnetopause.get_magnetopause(field_data, datafile, mode='hybrid')
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline')
    magnetopause.get_magnetopause(field_data, datafile, mode='shue',
                                  shue=1998)
    #plasmasheet.get_plasmasheet(field_data, datafile)
    '''

    #adjust view settings
    '''
    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              'K_out *', datafile, plasmasheet=False,
                              pngpath=PNGPATH, pltpath=PLTPATH,
                              show_contour=False, outputname=nameout)
    '''
    view_set.display_single_iso([frame for frame in tp.frames('main')][0],
                                'K_out *', datafile)
    main.plot().fieldmap(0).show = False
    tp.macro.execute_extended_command(command_processor_id='Multi Frame Manager',
    command='MAKEFRAMES3D ARRANGE=TOP SIZE=50')
    main.plot().fieldmap(0).show = True

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
