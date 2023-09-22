#/usr/bin/env python
"""script for passing dvol [R]^3 variable forward to next time file
"""
import time,sys,os
import glob
import numpy as np
import tecplot as tp
#
from global_energetics import makevideo

def clean_and_save(dataset,filename):
    """Function deletes all but the last zone (should be the toZone(newfile))
        and then saves it, overwritting the newfile
    """
    dataset.delete_zones(dataset.zone(0))
    tp.data.save_tecplot_plt(filename)
    print('\033[92m Updated\033[00m',filename)

def long_pass(fromZone,toZone):
    """Function linearly interpolates volume fromZone -> toZone *should*
        always work
    """
    tp.data.operate.interpolate_linear(toZone,source_zones=[fromZone],
                                       variables=[
                                        fromZone.dataset.variable('dVol *')])

def short_pass(fromZone,toZone):
    """Function directly ports volume fromZone -> toZone, only works if the
        connectivity is identical!
    """
    eq =tp.data.operate.execute_equation
    eq('{dvol [R]^3}={dvol [R]^3}['+str(fromZone.index+1)+']',zones=[toZone])

def check_volume_match(zone1,zone2):
    """Function integrates volume in each zone and returns comparison (bool)
    """
    x1 = zone1.values('X *').as_numpy_array()
    x2 = zone2.values('X *').as_numpy_array()
    volume1 = np.sum(zone1.values('dvol *').as_numpy_array()[(x1<30)&(x1>-30)])
    volume2 = np.sum(zone2.values('dvol *').as_numpy_array()[(x2<30)&(x2>-30)])
    print('\nVolume1: ',volume1)
    print('Volume2: ',volume2,'\n')
    return volume1==volume2

def volume_pass(pastfile,newfile):
    """Loads two files and ensures that the past file passes volume to new
        file if possible
    """
    tp.new_layout() #fresh tecplot environment
    # Load past file and make sure it's okay
    ds = tp.data.load_tecplot(pastfile)
    if 'dvol [R]^3' not in ds.variable_names:
        print(pastfile,' does not have volume! Unable to pass!!')
        return -1
    # Load new file and check if it is already good
    tp.data.load_tecplot(newfile)
    if ds.zone(1).values('dvol *').max()>0:
        success = check_volume_match(ds.zone(0),ds.zone(1))
        if success:
            print(newfile,' already good to go!')
            return 1
    # Try short pass first
    short_pass(ds.zone(0),ds.zone(1))
    if check_volume_match(ds.zone(0),ds.zone(1)):
        print(newfile,' short pass successful!')
        return 2
    else:
        # Hail mary
        long_pass(ds.zone(0),ds.zone(1))
        if check_volume_match(ds.zone(0),ds.zone(1)):
            return 3

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()
    file_path = 'run_GMonly/GM/IO2/'
    filekey = '3d__var_1_*.plt'
    all_times = sorted(glob.glob(os.path.join(file_path,filekey)),
                                key=makevideo.time_sort)
    oggridfile = glob.glob(file_path+'3d*volume*.plt')[0]
    result = volume_pass(oggridfile,all_times[0])
    assert result>=0
    if result>1:
        clean_and_save(tp.active_frame().dataset,all_times[0])
    for k,f in enumerate(all_times[1::]):
        print(all_times[k],' passing to ->\t',f)
        result = volume_pass(all_times[k],f)
        if result<0: break
        elif result==1:
            pass
        else:
            clean_and_save(tp.active_frame().dataset,f)
