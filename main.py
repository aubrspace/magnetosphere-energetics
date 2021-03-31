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
from global_energetics.extract import satellites
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
        tp.new_layout()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    #pass in arguments
    mhddatafile = sys.argv[1].split('/')[-1]
    MHDPATH = '/'.join(sys.argv[1].split('/')[0:-1])+'/'
    IEPATH = sys.argv[2]
    IMPATH = sys.argv[3]
    OUTPATH = sys.argv[4]
    PNGPATH = sys.argv[5]
    OUTPUTNAME = mhddatafile.split('e')[1].split('.plt')[0]

    #python objects
    field_data = tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'
    main = tp.active_frame()
    main.name = 'main'


    #Caclulate surfaces
    magnetopause.get_magnetopause(field_data, mhddatafile)

    #get supporting module data for this timestamp
    eventstring =field_data.zone('global_field').aux_data['TIMEEVENT']
    startstring =field_data.zone('global_field').aux_data['TIMEEVENTSTART']
    eventdt = dt.datetime.strptime(eventstring,'%Y/%m/%d %H:%M:%S.%f')
    startdt = dt.datetime.strptime(startstring,'%Y/%m/%d %H:%M:%S.%f')
    deltadt = eventdt-startdt
    satzones = satellites.get_satellite_zones(eventdt, MHDPATH, field_data)
    print('\nmade past call return!\n')
    #satzones = []
    '''
    north_iezone, south_iezone = get_ionosphere_zone(eventdt, IEPATH)
    im_zone = get_innermag_zone(deltadt, IMPATH)
    '''
    #adjust view settings
    bot_right = [frame for frame in tp.frames('main')][0]
    view_set.display_single_iso(bot_right,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=False, show_fieldline=True,
                                pngpath=PNGPATH,
                                plot_satellites=True, satzones=satzones,
                                outputname=OUTPUTNAME, save_img=True)
    '''
    #tile
    proc = 'Multi Frame Manager'
    cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
    tp.macro.execute_extended_command(command_processor_id=proc,
                                          command=cmd)
    bot_right = [frame for frame in tp.frames('main')][0]
    bot_right.name = 'inside_from_tail'
    frame1 = [frame for frame in tp.frames('Frame 001')][0]
    frame2 = [frame for frame in tp.frames('Frame 002')][0]
    frame3 = [frame for frame in tp.frames('Frame 003')][0]
    view_set.display_single_iso(bot_right,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=False, show_fieldline=False,
                                pngpath=PNGPATH,
                                outputname=OUTPUTNAME, save_img=False,
                                mode='inside_from_tail')
    frame1.activate()
    frame1.name = 'isodefault'
    view_set.display_single_iso(frame1,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True, show_fieldline=True,
                                pngpath=PNGPATH,
                                outputname=OUTPUTNAME, save_img=False,
                                show_timestamp=False)
    frame2.activate()
    frame2.name = 'alternate_iso'
    view_set.display_single_iso(frame2,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True, show_fieldline=True,
                                pngpath=PNGPATH,
                                outputname=OUTPUTNAME, save_img=False,
                                mode='other_iso',
                                show_timestamp=False)
    frame3.activate()
    frame3.name = 'tail_iso'
    view_set.display_single_iso(frame3,
                                'K_net *', mhddatafile, show_contour=True,
                                show_slice=True, show_fieldline=True,
                                pngpath=PNGPATH,
                                outputname=OUTPUTNAME,
                                mode='iso_tail',
                                show_timestamp=False)
    '''
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
