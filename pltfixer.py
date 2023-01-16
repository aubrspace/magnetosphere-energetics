#/usr/bin/env python
import sys,os
import glob
import datetime as dt
import tecplot as tp
from global_energetics.makevideo import get_time

def standardize_name(filename):
    """Function reads filename and changes it to the
     '3d__VAR_1_eYYYYMMDD-HHMMSS-mmm.plt' format
    Inputs
        filename(str)
    return
        newname (str)
    """
    path = '/'.join(filename.split('/')[0:-1])
    infile = '/'.join(filename.split('/')[-1::])
    filetime = get_time(infile)
    return '3d__paraview_1_e'+filetime.strftime('%Y%m%d-%H%M%S-000')+'.plt'

if __name__ == '__main__':
    header = 'localdbug/fte/test/'
    #header = '/nfs/solsticedisk/tuija/amr_fte/secondtry/GM/IO2/'
    #header = '/nfs/solsticedisk/tuija/ccmc/2022-02-02/2022-02-02/run/GM/IO2/copy_paraview/'
    #header = '/home/aubr/Code/swmf-energetics/localdbug/fte/pieces/'
    #header = '/home/aubr/Code/swmf-energetics/localdbug/feb2014/'
    if '-c' in sys.argv:
        tp.session.connect()
    for i,infile in enumerate(glob.glob(header+'*_var*.plt')):
        outfile = 'paraview'.join(infile.split('/')[-1].split('var'))
        outfile_name = standardize_name(outfile)
        if os.path.exists(header+outfile_name):
            print(str(i)+' already found '+outfile+'....')
            os.remove(infile)
        else:
            try:
                print(str(i)+' fixing into '+outfile+'....')
                ds = tp.data.load_tecplot(infile)
                ds.variable('X*').name = 'x'
                ds.variable('Y*').name = 'y'
                ds.variable('Z*').name = 'z'
                aux = ds.zone(0).aux_data.as_dict()
                with open(header+outfile_name.split('.plt')[0]+'.aux',
                          'w')as f:
                    for key,value in aux.items():
                        f.write('%s:%s\n' % (key,value))
                ds.zone(0).aux_data.clear()
                tp.data.save_tecplot_plt(header+outfile_name,dataset=ds)
                tp.data.save_tecplot_plt('test.plt',dataset=ds)
                tp.new_layout()
                os.remove(infile)
            except tp.exception.TecplotLogicError:
                print('FAILED!!')
                pass
    print('DONE')
