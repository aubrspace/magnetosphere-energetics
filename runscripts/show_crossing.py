#/usr/bin/env python
import os,sys,time
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import glob
import numpy as np
# Tecplot
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
# Custom
from global_energetics import makevideo
from global_energetics.extract import magnetosphere
from global_energetics.extract import satellites

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    # Set file paths/individual file
    inpath = 'magEx/data/starlink/'
    outpath = 'magEx/figures/'
    head = '3d__var_1_*'
    # Search to find the full list of files
    filelist = sorted(glob.glob(os.path.join(inpath,head)),
                      key=makevideo.time_sort)
    for k,f in enumerate(filelist):
        filetime = makevideo.get_time(f)
        tp.new_layout()
        #python objects
        field_data = tp.data.load_tecplot(f)
        field_data.zone(0).name = 'global_field'
        main = tp.active_frame()
        main.name = 'main'
        #Perform data extraction
        with tp.session.suspend():
            _,results = magnetosphere.get_magnetosphere(field_data,
                                                        write_data=False,
                                                        verbose=True,
                                                        do_cms=False,
                                    analysis_type='energy_mass_mag_plasmoid',
                                    modes=['iso_betastar','closed',
                                           'nlobe','slobe','plasmasheet'],
                                    do_interfacing=True,
                                    tail_cap=-120,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    #3truegridfile=oggridfile,
                                    outputpath=outpath)
        satellites.get_satzone_fromHdf5(field_data,inpath+'mms_pos.h5')
        satellites.add_currentlocation(['mms1'],field_data)

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
