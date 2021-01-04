#/usr/bin/env python
"""prototype for visualizing upstream flowlines for dev mp surface alt
"""
import sys
import os
import time
import numpy as np
import spacepy
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
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
    print('nameout:'+nameout)
    OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-a'

    #python objects
    field_data=tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'

    #set frame object and radial coordinates
    main_frame = tp.active_frame()
    main_frame.name = 'main'
    tp.data.operate.execute_equation(
            '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
    tp.data.operate.execute_equation(
            '{lat [deg]} = 180/pi*asin({Z [R]} / {r [R]})')
    tp.data.operate.execute_equation(
            '{lon [deg]} = if({X [R]}>0,'+
                                '180/pi*atan({Y [R]} / {X [R]}),'+
                            'if({Y [R]}>0,'+
                                '180/pi*atan({Y [R]}/{X [R]})+180,'+
                                '180/pi*atan({Y [R]}/{X [R]})-180))')

    #adjust view settings
    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              'Rho *', datafile, plasmasheet=False,
                              magnetopause=False, save_img=False,
                              show_contour=False, show_slice=False,
                              show_fieldline=False, do_blanking=False)

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
