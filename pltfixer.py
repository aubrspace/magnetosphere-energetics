#/usr/bin/env python
"""Modifies tecplot .plt files to be save for the VisIt reader in Paraview
"""
import sys,os
import glob
import datetime as dt
import tecplot as tp
#Interpackage imports
try:
    from global_energetics.makevideo import get_time
    DOSTANDARDIZE = True
except ImportError:
    print('global_energetics not found, unable to standardize names')
    DOSTANDARDIZE = False

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

def fix_plt_files(pathtofiles,**kwargs):
    """Function modifies all .plt files at pathtofiles by splitting off
        aux data and renaming x,y,z variable names
    Inputs
        pathtofiles (str)
        kwargs:
            verbose (bool)- default False, True will print TecplotErrors
            standarize (bool)- default True, standardize the file names
            keep (bool)- default False, have only modified one remaining
    """
    for i,infile in enumerate(glob.glob(pathtofiles+'*_var*.plt')):
        outfile = 'paraview'.join(infile.split('/')[-1].split('var'))
        if kwargs.get('standardize',False):
            outfile_name = standardize_name(outfile)
        else:
            outfile_name = outfile
        if os.path.exists(pathtofiles+outfile_name):
            print(str(i)+' already found '+outfile+'....')
            if not kwargs.get('keep',False):
                os.remove(infile)
        else:
            try:
                print(str(i)+' fixing '+infile+' into '+outfile_name+'....')
                ds = tp.data.load_tecplot(infile)
                ds.variable('X*').name = 'x'
                ds.variable('Y*').name = 'y'
                ds.variable('Z*').name = 'z'
                aux = ds.zone(0).aux_data.as_dict()
                with open(pathtofiles+outfile_name.split('.plt')[0]+'.aux',
                          'w')as f:
                    for key,value in aux.items():
                        f.write('%s:%s\n' % (key,value))
                ds.zone(0).aux_data.clear()
                tp.data.save_tecplot_plt(pathtofiles+outfile_name,
                                         dataset=ds)
                tp.new_layout()
                if not kwargs.get('keep',False):
                    os.remove(infile)
            except tp.exception.TecplotLogicError as err:
                if kwargs.get('verbose',False):
                    print('FAILED!!\n'+str(err))
                else:
                    print('FAILED!!')

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print("""
    Script to modify tecplot .plt files to be read by VisIt Paraview reader

    Requirements:
        python packages: sys,os,glob,datetime
                          optional- global_energetics.makevideo and depends
                                    (for time sorting function)

    Usage: python pltfixer.py [-flags] PATHTOFILES

    Options:
        -h  --help      prints this message then exit
        -v  --verbose   prints TecplotErrors if any
        -c              uses pytecplot in 'connected' mode,
                            see pytecplot for more details
        -k  --keep      keeps both original and modified .plt files
        -s  --standard  standardizes the name to a common type

    Example:
        python pltfixer.py -c localdbug/fte/
            this will modify all the .plt files in localdbug/fte/
              using the 'connected' mode

        multiPytec pltfixer.py localdbug/fte/
            this will modify all the .plt files in localdbug/fte/
              using 'multiPytec' which is an alias that sets the
              correct tec-env so that it can be run in batch mode
              for more information see:
              https://www.tecplot.com/docs/pytecplot/install.html

        """)
        exit()
    ###########################################################
    # Read in arguments and flags
    pathtofiles = sys.argv[-1]
    if not os.path.exists(pathtofiles):
        print('Path not found please try again')
        exit()
    if pathtofiles=='pltfixer.py' or len(sys.argv)==1:
        print('No path to files given!! use -h or --help for more info')
        exit()
    if '-v' in sys.argv:
        doVerbose=True
    else:
        doVerbose=False
    if '-c' in sys.argv:
        tp.session.connect()
    if '-k' in sys.argv or '--keep' in sys.argv:
        keepboth = True
    else:
        keepboth = False
    if (DOSTANDARDIZE and '-s' not in sys.argv and
                          '--standard' not in sys.argv):
        DOSTANDARDIZE=False

    # Modify files
    fix_plt_files(pathtofiles,verbose=doVerbose,
                  standardize=DOSTANDARDIZE,
                  keep=keepboth)
    print('DONE')
