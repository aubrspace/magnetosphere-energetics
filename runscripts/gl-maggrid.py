#/usr/bin/env python
"""script for calculating quanties from the simulation virtual mag grid
"""
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import time
import numpy as np
import pandas as pd
from global_energetics.extract import magnetometer

if __name__ == "__main__":
    start_time = time.time()
    ##Parse input flags
    # Input files
    if '-i' in sys.argv:
        inpath = sys.argv[sys.argv.index('-i')+1]
    elif '--idir' in sys.argv:
        inpath = sys.argv[sys.argv.index('--idir')+1]
    else:
        inpath = 'test_inputs/'
    if not os.path.exists(inpath):
        print('input path "'+inpath+'" not found!')
        exit()
    # Output path files
    if '-o' in sys.argv:
        outpath = sys.argv[sys.argv.index('-o')+1]
    elif '--odir' in sys.argv:
        outpath = sys.argv[sys.argv.index('--odir')+1]
    else:
        outpath = 'test_outputs/'
    if '-b' in sys.argv:
        filetype = 'real4'
    else:
        filetype = 'ascii'
    ########################################
    #make directories for output
    os.makedirs(outpath, exist_ok=True)
    ########################################
    MGL = magnetometer.read_MGL(inpath,type=filetype)
    mgl_file = os.path.join(outpath,'MGL.h5')
    MGL.to_hdf(mgl_file,key='gridMin')
    print('Created ',mgl_file)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
    exit()
