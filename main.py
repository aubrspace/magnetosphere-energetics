#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import numpy as np
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetopause
from global_energetics.extract import plasmasheet
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import view_set

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    #datafile = '3d__mhd_2_e20140219-123000-000.plt'
    datafile = os.environ["file"]
    field_data=tp.data.load_tecplot(datafile)
    field_data.zone(0).name = 'global_field'

    magnetopause.get_magnetopause(field_data, datafile)
    plasmasheet.get_plasmasheet(field_data, datafile)

    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              field_data.variable('K_in *').index)
    view_set.integral_display('mp')
    view_set.integral_display('cps', left_aligned=False)

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
