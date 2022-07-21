#/usr/bin/env python
"""Multiprocessing BATSRUS data primarily for visualization
"""
import time
import logging
import os, multiprocessing, sys, warnings
import numpy as np
import datetime as dt
import glob
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
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
    master = '/home/aubr/Desktop/lobe_rangers.lay'
    #for epoch in glob.glob('/home/aubr/Code/paleo/*.plt'):
    #for infile in glob.glob('starlink/*.plt'):
    #TODO: use tecplot 'stylesheet' instead of loading the source data eachtime
    for infile in glob.glob('febstorm/*.plt'):
        intime = get_time(infile)
        #if (intime>dt.datetime(2022,2,4,22,0) and
        #    intime<dt.datetime(2022,2,5,6,30)):
        if not (intime>dt.datetime(2014,2,19,2,15) and
                intime<dt.datetime(2014,2,19,8,16)):
            marktime = time.time()
            outlabel = infile.split('var_1_')[-1].split('.plt')[0]
            print(infile)
            tempfile = './tempfile.plt'
            tp.new_layout()
            #path = 'egu_video/balance.map'
            #tp.macro.execute_command('$!LOADCOLORMAP "'+path+'"')
            ds = tp.data.load_tecplot(infile)
            ds.zone(0).name = 'global_field'
            magnetosphere.get_magnetosphere(ds,save_mesh=False,
                                                    outputpath='babyrun/',
                                                    do_interfacing=False,
                                                    do_cms=False,tshift=45,
                                                    integrate_volume=False,
                                                    integrate_surface=False,
                                                    verbose=False,
                                                    extract_flowline=False,
                                              analysis_type='energy',
                      modes=['iso_betastar','nlobe','slobe','closed','rc'])
            tp.data.save_tecplot_plt(tempfile)
            newvars = ds.variable_names
            tp.new_layout()
            tp.load_layout(master)
            tp.data.load_tecplot(tempfile,
                                read_data_option=ReadDataOption.Replace,
                                reset_style=False)
            oldvars = tp.active_frame().dataset.variable_names
            if not all([any(n in o for o in oldvars) for n in newvars]):
                warnings.warn("Variable lists don't match!!",
                              UserWarning)
            '''
            f1,f2 = [f for f in tp.frames()]
            f1.activate()
            f1.plot().contour(0).levels.reset_levels(np.linspace(-1*3e9,3e9,11))
            f1.plot().contour(0).colormap_filter.distribution=ColorMapDistribution.Banded
            f1.plot().contour(0).variable = f1.dataset.variable('K_net *')
            f2.activate()
            f2.plot().contour(0).levels.reset_levels(np.linspace(-1*3e9,3e9,11))
            f2.plot().contour(0).colormap_filter.distribution=ColorMapDistribution.Banded
            f2.plot().contour(0).variable = f2.dataset.variable('K_net *')
            '''

            tp.save_png('/home/aubr/Desktop/lobe_dist/'+outlabel+'.png',
                        width=1600)
            ltime = time.time()-marktime
            print(infile+' DONE')
            print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                            np.mod(ltime,60)))
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
