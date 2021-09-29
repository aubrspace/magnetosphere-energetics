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
from global_energetics.extract import magnetopause
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
    _,mp_powers, mp_energies = magnetopause.get_magnetopause(field_data,
                                                             mhddatafile,
                                                             do_cms=False,
                                                        outputpath=OUTPATH)

    """
    #adjust view settings
    #tile
    #proc = 'Multi Frame Manager'
    #cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
    #tp.macro.execute_extended_command(command_processor_id=proc,
    #                                      command=cmd)
    bot_right = [frame for frame in tp.frames('main')][0]
    bot_right.name = 'inside_from_tail'
    view_set.display_single_iso(bot_right,
                                'beta_star',mhddatafile, show_contour=False,
                                energyrange=20, transluc=40,
                                show_slice=True, show_legend=True,
                                show_fieldline=True,
                                pngpath=PNGPATH,
                                plot_satellites=False,
                                outputname=OUTPUTNAME, save_img=False,
                                show_timestamp=False)
    '''
    frame1 = [frame for frame in tp.frames('Frame 001')][0]
    frame2 = [frame for frame in tp.frames('Frame 002')][0]
    frame3 = [frame for frame in tp.frames('Frame 003')][0]
    frame1.activate()
    frame1.name = 'isodefault'
    view_set.display_single_iso(frame1,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True, show_legend=False,
                                pngpath=PNGPATH,
                                plot_satellites=False,
                                outputname=OUTPUTNAME, save_img=False,
                                show_timestamp=False)
    frame2.activate()
    frame2.name = 'alternate_iso'
    view_set.display_single_iso(frame2,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True,
                                pngpath=PNGPATH, add_clock=True,
                                plot_satellites=False,
                                outputname=OUTPUTNAME, save_img=False,
                                mode='other_iso',
                                show_timestamp=False)
    frame3.activate()
    frame3.name = 'tail_iso'
    view_set.display_single_iso(frame3,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True, show_legend=False,
                                pngpath=PNGPATH, transluc=60,
                                plot_satellites=False,
                                outputname=OUTPUTNAME,
                                mode='iso_tail',
                                show_timestamp=False, save_img=True)
    bot_right.activate()
    view_set.display_single_iso(bot_right,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=False,
                                show_legend=False, mode='inside_from_tail',
                                pngpath=PNGPATH,
                                plot_satellites=False,
                                show_timestamp=True, transluc=40,
                                outputname=OUTPUTNAME, save_img=False)
    '''
    """
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    from IPython import embed; embed()
