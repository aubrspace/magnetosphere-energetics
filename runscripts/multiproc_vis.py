#/usr/bin/env python
"""Multiprocessing BATSRUS data primarily for visualization
"""
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import time
import logging
import multiprocessing, warnings
import numpy as np
import datetime as dt
import glob
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
from global_energetics.extract import view_set
from global_energetics.makevideo import get_time
from global_energetics.extract import magnetosphere

if __name__ == '__main__':
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
    #master = '/home/aubr/Desktop/paleo_ms_volumes.lay'
    #master = '/home/aubr/Desktop/paleo_jpar2.lay'
    #master = '/home/aubr/Desktop/paleo_closeup_gap.lay'
    #master = 'egu_video/recovery_loop.lay'
    #master = 'egu_video/basic_loop.lay'
    #master = 'egu_video/balance_loop.lay'
    #master = '/home/aubr/Desktop/lobe_rangers.lay'
    #master = '/home/aubr/Desktop/magnetometer_vis/prelim.lay'
    master = '/home/aubr/Desktop/convection_movies/north_convection.lay'
    mainsheet = '/home/aubr/Desktop/convection_movies/north_pc_flow_term.sty'
    #TODO: use tecplot 'stylesheet' instead of loading the source data eachtime
    #for infile in glob.glob('ccmc_2019-05-13/*.plt')[9*60+45:9*60+90:4]:
    for infile in glob.glob(
                         '/nfs/solsticedisk/tuija/polarcap2000/run/GM/IO2/*'):
        intime = get_time(infile)
        if True:
            marktime = time.time()
            outlabel = infile.split('var_1_')[-1].split('.plt')[0]
            print(infile)
            #tempfile = './tempfile.plt'
            tp.new_layout()
            ds = tp.data.load_tecplot(infile)
            ds.zone(0).name = 'global_field'
            magnetosphere.get_magnetosphere(ds,save_mesh=False,
                                            analysis_type='mag',
                                            modes=['sphere','terminator'],
                                            sp_rmax=2.65,
                                                do_interfacing=True,
                                                do_cms=False,
                                                integrate_volume=False,
                                                integrate_surface=True,
                                                integrate_line=True,
                                                extract_flowline=False,
                                                    write_data=False,
                                                    disp_result=False,
                                                    verbose=False)
            tp.active_frame().load_stylesheet(mainsheet)
            tp.save_png('/home/aubr/Desktop/convection_movies/'+outlabel+'.png',
                        width=1600)
            ltime = time.time()-marktime
            print(infile+' DONE')
            print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                            np.mod(ltime,60)))
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
