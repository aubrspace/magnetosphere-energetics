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

    OUTPATH = 'temp/'
    PNGPATH = 'temp/'
    OUTPUTNAME = 'testoutput1.png'

    '''
1947849 2014 02 19 03 17 00 018  3.93901E-002  2.56463E+000 -1.70966E-015  1.69500E-017 -2.75944E-018  1.04916E-002  3.64284E+000 -4.80667E-001 -9.16607E+000  1.42366E-005  1.81455E+001 -4.43327E+001 -4.90226E+001  1.18495E+002  9.75989E+001

    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''

    #for point in [files1, files2, files3, files4]:
    for point in [files1]:
        #python objects
        field_data = tp.data.load_tecplot([point[0],point[1]])
        field_data.zone(0).name = 'global_field'
        field_data.zone(1).name = 'future'
        main = tp.active_frame()
        main.name = 'main'

        #Caclulate initial surface
        #for mode in ['iso_betastar', 'ps','qDp','rc','nlobe','slobe']:
        #with tp.session.suspend():
        mesh, data = magnetosphere.get_magnetosphere(field_data,
                                        outputpath=OUTPATH,
                                    analysis_type='virial_biotsavart',
                                        do_cms=True,
                                        save_mesh=False,
                                        integrate_surface=True,
                                        integrate_volume=True)
        '''
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
        '''
        if False:#manually switch on or off
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
        tp.new_layout()
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
