#/usr/bin/env python
"""Modifies tecplot .plt files to be save for the VisIt reader in Paraview
"""
import sys,os,glob,time
sys.path.append(os.getcwd().split('magnetosphere-energetics')[0]+
                                      'magnetosphere-energetics/')
import datetime as dt
import tecplot as tp
#Interpackage imports
try:
    from global_energetics.makevideo import get_time
    from global_energetics.extract import magnetosphere
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

def fix_dat_files(pathtofiles:str,**kwargs:dict) -> None:
    """Function modifies all .plt files at pathtofiles by splitting off
        aux data and renaming x,y,z variable names
    Inputs
        pathtofiles (str)
        kwargs:
            verbose (bool)- default False, True will print TecplotErrors
            standarize (bool)- default True, standardize the file names
            keep (bool)- default False, have only modified one remaining
    """
    for i,infile in enumerate(glob.glob(pathtofiles+'*d__var*.dat')):
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
            print(str(i)+' fixing '+infile+' into '+outfile_name+'....')
            done = False
            aux = {}
            # Read the original file
            with open(infile,'rt') as fin:
                lines = fin.readlines()
            # Write a new copy with some edits
            write_lines = lines.copy()
            pop_buffer = 0
            for iline,line in enumerate(lines):
                # Change the name of XYZ so the reader can pick it up
                if line.startswith('VARIABLES'):
                    write_lines[iline] = line.replace('X [R]','X'
                                            ).replace('Y [R]','Y'
                                            ).replace('Z [R]','Z')
                # Pop out the aux data to its own file
                if line.startswith('AUXDATA'):
                    auxline = write_lines.pop(iline-pop_buffer)
                    pop_buffer+=1
                    key = auxline.replace('AUXDATA ','').split('=')[0]
                    value = ''.join(
                               auxline.replace('AUXDATA ','').split('=')[1::])
                    aux[key] = value.replace('"','')
            # Write the new file
            with open(pathtofiles+outfile,'wt') as fout:
                fout.writelines(write_lines)
            # Write the aux data
            with open(pathtofiles+outfile_name.replace('.dat','.aux'),
                                                                 'wt')as faux:
                for key,value in aux.items():
                    faux.write('%s:%s\n' % (key,value))
            # Delete the old file
            if not kwargs.get('keep',False):
                os.remove(infile)

    return

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
                ds.zone(0).name = 'global_field'
                if kwargs.get('dimensionalize',False):
                    magnetosphere.todimensional(ds)
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
    return

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
        -d  --dat       works on a .dat file instead of a .plt file

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
    if '-c' in sys.argv and '-a' not in sys.argv:
        tp.session.connect()
    if '-k' in sys.argv or '--keep' in sys.argv:
        keepboth = True
    else:
        keepboth = False
    if '-a' in sys.argv or '--ascii' in sys.argv:
        doDatfix = True
    else:
        doDatfix = False
    if (DOSTANDARDIZE and '-s' not in sys.argv and
                          '--standard' not in sys.argv):
        DOSTANDARDIZE=False
    if DOSTANDARDIZE and('-d' in sys.argv or '--dimensionalize' in sys.argv):
        dimension = True
    else:
        dimension = False

    # Modify files
    if not doDatfix:
        filelist = glob.glob(pathtofiles+'*_var*.plt')
        if len(filelist)>0:
            fix_plt_files(pathtofiles,verbose=doVerbose,
                          standardize=DOSTANDARDIZE,
                          dimensionalize=dimension,
                          keep=keepboth)
        else:
            print('Couldnt find any .plt files, trying .dat files instead')
            doDatfix = True
    if doDatfix:
            fix_dat_files(pathtofiles,verbose=doVerbose,
                          standardize=DOSTANDARDIZE,
                          dimensionalize=dimension,
                          keep=keepboth)
    print('DONE')
