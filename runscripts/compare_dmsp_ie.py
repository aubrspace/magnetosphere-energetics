#!/usr/bin/env python3
""" Comparing DMSP with SWMF IE output
"""
import os,sys
sys.path.append(os.getcwd().split('magnetosphere-energetics')[0]+
                                      'magnetosphere-energetics/')
import time
import numpy as np
import glob
#
from global_energetics import makevideo
from global_energetics.extract.ionosphere import (create_ie_npz,
                                                  read_ie_npz,read_dmsp,
                                                  integrate_E)


def main() -> None:
    inpath = 'run_mothersday_ne/IE/ionosphere/'
    dmsp_path = 'DMSP_passes_F18'
    # (Option) combine all IE files into 1 compressed .npz
    if not any(['.npz' in f for f in glob.glob(inpath+'*.tec')]):
        npz_file = create_ie_npz(inpath)
    else:
        npz_file = glob.glob(inpath+'*.npz')[0]
    # Read ie .npz file
    ie = read_npz_file(npz_file)
    # Read dmsp trajectory files
    if not any(['.npz' in f for f in glob.glob(dmsp_path)]):
        dmsp = {}
        for sat in ['F16','F17','F18']:
            sat_data = read_dmsp(f"{dmsp_path}/DMSP_passes_{sat}")
            dmsp[sat] = sat_data
    else:
        dmsp = np.load(glob.glob(dmsp_path+'*.npz')[0])

if __name__ == "__main__":
    start_time = time.time()

    main()

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
