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
from global_energetics.mpdynamics_analysis import proc_spatial
from global_energetics.mpdynamics_analysis import proc_temporal

if __name__ == "__main__":
    #print('\nProcessing {pltfile}\n'.format(pltfile=sys.argv[1]))
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    #pass in arguments
    datafile = sys.argv[1].split('/')[-1]
    OUTPATH = sys.argv[2]
    PNGPATH = sys.argv[3]
    PLTPATH = sys.argv[4]
    OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-a'
    main = tp.active_frame()
    main.name = 'main'

    #python objects
    field_data=tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'

    #Caclulate surfaces
    magnetopause.get_magnetopause(field_data, datafile, mode='box',
                                  box_xmax= 7, box_xmin=-7)

    '''
    #adjust view settings
    view_set.display_single_iso([frame for frame in tp.frames('main')][0],
                                'K_net *', datafile, show_contour=True,
                                pngpath=PNGPATH, pltpath=PLTPATH,
                                outputname=OUTPUTNAME)
    '''

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
