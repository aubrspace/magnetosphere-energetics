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
    files1 = ('3d__var_1_e20140219-090000-000.plt',
              '3d__var_1_e20140219-090100-023.plt')

    files2 = ('3d__var_1_e20140219-023400-027.plt',
              '3d__var_1_e20140219-023500-037.plt')

    files3 = ('3d__var_1_e20140219-041500-002.plt',
              '3d__var_1_e20140219-041600-027.plt')

    files4 = ('3d__var_1_e20140219-060000-000.plt',
              '3d__var_1_e20140219-060100-014.plt')

    files5 = ('3d__var_1_e20140218-060700-011.plt',
              '3d__var_1_e20140218-060900-002.plt')
    files5 = ('3d__var_1_e20220202-051000-000.plt',
              '3d__var_1_e20220202-050500-000.plt')
    #files5 = ('output/CCMC/3d__var_1_e20130713-204700-037.plt',
    #         'output/CCMC/3d__var_1_e20130713-204700-037.plt')

    OUTPATH = 'temp/'
    PNGPATH = 'temp/'
    OUTPUTNAME = 'testoutput1.png'

    '''
    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''
    mhddatafile = files5[0]

    #for point in [files1, files2, files3, files4]:
    for point in [files5]:
        #python objects
        field_data = tp.data.load_tecplot([point[0],point[1]])
        field_data.zone(0).name = 'global_field'
        field_data.zone(1).name = 'future'
        main = tp.active_frame()
        main.name = 'main'

        #Caclulate initial surface
        #for mode in ['iso_betastar', 'ps','qDp','rc','nlobe','slobe']:
        with tp.session.suspend():
            mesh, data = magnetosphere.get_magnetosphere(field_data,
                                                    outputpath=OUTPATH,
                                    analysis_type='energy',
                                                    do_cms=False,
                                                    mpbetastar=0.6,
                                                    tail_cap=-20,
                                                    save_mesh=False,
                                                    integrate_surface=True,
                                                    integrate_volume=False)
        """
        vol = data['mp_iso_betastar_volume']
        surf = data['mp_iso_betastar_surface']
        inner = data['mp_iso_betastar_inner_surface']
        for key in vol.keys():
            if '[J]' in key:
                vol[key.split('[J]')[0]+'[nT]'] = vol[key]/(-8e13)
        for key in surf.keys():
            if '[J]' in key:
                surf[key.split('[J]')[0]+'[nT]'] = surf[key]/(-8e13)
        total = (vol['Virial 2x Uk [nT]']+ vol['Virial Ub [nT]'] +
               surf['Virial Surface Total [nT]']+inner['Virial Fadv [nT]']+
                inner['Virial b^2 [nT]'])
        biot = vol['bioS [nT]']
        print('2xUk: {}'.format(vol['Virial 2x Uk [nT]']))
        print('Ub: {}'.format(vol['Virial Ub [nT]']))
        print('Surf: {}'.format(surf['Virial Surface Total [nT]']))
        print('InnerSurf: {}'.format(inner['Virial Fadv [nT]']+
                                     inner['Virial b^2 [nT]']))
        print('Total: {}'.format(total))
        print('Scaled: {}'.format(total*1.5))
        print('BioS: {}'.format(biot))
        print('Ratio: {}'.format(biot/total))
        """
    #with tp.session.suspend():
    if True:
        if True:#manually switch on or off
            #adjust view settings
            proc = 'Multi Frame Manager'
            cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
            tp.macro.execute_extended_command(command_processor_id=proc,
                                              command=cmd)
            mode = ['iso_day', 'other_iso', 'iso_tail', 'hood_open_north']
            zone_hidekeys = ['sphere', 'box','shue','future',
                             'lcb']
            timestamp=True
            for frame in enumerate(tp.frames()):
                frame[1].activate()
                if frame[0]==0:
                    legend = False
                    timestamp = True
                    doslice = True
                    slicelegend = False
                    fieldlegend = True
                    fieldline=True
                if frame[0]==1:
                    legend = True
                    timestamp = False
                    doslice = True
                    slicelegend = False
                    fieldlegend = False
                    fieldline=True
                if frame[0]==2:
                    legend = False
                    timestamp = False
                    doslice = True
                    slicelegend = True
                    fieldlegend = False
                    fieldline=True
                if frame[0]==3:
                    legend = True
                    save = True
                    timestamp = False
                    doslice = False
                    slicelegend = False
                    fieldlegend = False
                    fieldline=True
                view_set.display_single_iso(frame[1], mhddatafile,
                                            mode=mode[frame[0]],
                                            outputname=OUTPUTNAME,
                                            show_fieldline=fieldline,
                                            show_legend=legend,
                                            show_slegend=slicelegend,
                                            show_flegend=fieldlegend,
                                            show_slice=doslice,
                                            timestamp_pos=[4,5],
                                            zone_hidekeys=zone_hidekeys,
                                            show_timestamp=timestamp)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
