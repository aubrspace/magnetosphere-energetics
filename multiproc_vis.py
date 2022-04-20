#/usr/bin/env python
"""Multiprocessing BATSRUS data primarily for visualization
"""
import time
import logging
import os, multiprocessing, sys, warnings
import numpy as np
import glob
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
from global_energetics.extract import magnetosphere

if __name__ == '__main__':
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
    master = '/home/aubr/Desktop/paleo_ms_volumes.lay'
    for epoch in glob.glob('/home/aubr/Code/paleo/*.plt'):
        if '4080' in epoch:
            marktime = time.time()
            outlabel = epoch.split('-000_')[-1].split('.plt')[0]
            print(epoch)
            tempfile = './tempfile.plt'
            tp.new_layout()
            ds = tp.data.load_tecplot(epoch)
            ds.zone(0).name = 'global_field'
            magnetosphere.get_magnetosphere(ds,save_mesh=False,
                                            tail_cap=-40,
                                            do_cms=False,
                                            integrate_volume=False,
                                            integrate_surface=False,
                                            analysis_type='energy',
                                            inner_r=2,sunward_pole=True,
                                            mpbetastar=0.7)
            tp.data.save_tecplot_plt(tempfile)
            newvars = ds.variable_names
            tp.load_layout(master)
            tp.data.load_tecplot(tempfile,
                                read_data_option=ReadDataOption.Replace,
                                reset_style=False)
            oldvars = tp.active_frame().dataset.variable_names
            if not all([any(n in o for o in oldvars) for n in newvars]):
                warnings.warn("Variable lists don't match!!",
                              UserWarning)
            tp.save_png('/home/aubr/Code/paleo/'+outlabel+'.png',
                        width=1600)
            ltime = time.time()-marktime
            print(epoch+' DONE')
            print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                            np.mod(ltime,60)))
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
