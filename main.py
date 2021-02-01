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
    nameout = datafile.split('e')[1].split('-000.')[0]+'-mp'
    print('nameout:{}'.format(nameout))
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
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline',
                                  cuttoff=-10,zone_rename='mp_flowline_10')
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline',
                                  cuttoff=-15,zone_rename='mp_flowline_15')
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline',
                                  cuttoff=-20,zone_rename='mp_flowline_20')
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline',
                                  cuttoff=-30,zone_rename='mp_flowline_30')
    magnetopause.get_magnetopause(field_data, datafile, mode='flowline',
                                  cuttoff=-40,zone_rename='mp_flowline_40')
    magnetopause.get_magnetopause(field_data, datafile, mode='shue',
                                  shue=1998, cuttoff=-15,
                                  zone_rename='mp_shue98_15')
    magnetopause.get_magnetopause(field_data, datafile, mode='shue',
                                  shue=1998, cuttoff=-30,
                                  zone_rename='mp_shue98_30')
    #plasmasheet.get_plasmasheet(field_data, datafile)

    #adjust view settings
    view_set.display_single_iso([frame for frame in tp.frames('main')][0],
                                'K_out *', datafile, show_contour=False,
                                pngpath=PNGPATH, pltpath=PLTPATH,
                                outputname=OUTPUTNAME)

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
