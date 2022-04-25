#/usr/bin/env python
"""Python script will find preplot tool in pytecplot library (for macOS)
and run on set of .dat files +
   Also load "IDL" format and synthesize connectivity if possible
"""

import glob
import os,warnings
import sys
from subprocess import check_output as read_out
import tecplot as tp
from tecplot.constant import *

def IDL_to_hdf5(filepath, **kwargs):
    """Function uses spacepy package to open and write 'IDL' formatted swmf
        output to hdf5 where it can be read by tecplot
    Inputs
        filepath
    Return
        hdf5file
    """
    if os.path.exists(filepath):
        #import spacepy
        from spacepy import pybats as bats
        fil = bats.IdlFile(filepath)
        hdffile = filepath.split('/')[-1].split('.out')[0]+'.h5'
        fil.toHDF5(os.getcwd()+'/'+hdffile)
        return hdffile
    else:
        warnings.warn(filepath+' Does not exist!',UserWarning)
        return None

def load_hdf5_data(filepath, **kwargs):
    """Function reads in 'IDL' data given variable list and modifies
        variable names
    Inputs
        filepath
        kwargs:
                in_variables- what is contained in the file
                variable_name_list- new total set of variables
                readertype- what loader is used default is HDF5
                initial_plottype- default Cartesian2D
    """
    default_vars= ['/x','/z','/Bx','/By','/Bz','/jx','/jy','/jz','/P',
                        '/Rho','/Ux','/Uy','/Uz','/b1x','/b1y','/b1z']
    command=("""$!ReadDataSet  '\"-F\" \"1\" """+'\"'+filepath+'" \"-D\" \"'
             +str(len(kwargs.get('in_variables',default_vars)))+'\" ')
    for variable in kwargs.get('in_variables',default_vars):
        command = command+'\"'+variable+'\" '
    command = command+"""\"-K\" \"1\" \"1\" \"1\"'"""
    command = (command+"""
  DataSetReader = '"""+kwargs.get('readertype','HDF5 Loader')+"'")
    if 'variable_name_list' in kwargs:
        command = (command+"""
  VarNameList = '""")
    for var in kwargs.get('variable_name_list',[]):
        command = command+'\"'+var+'\" '
    if 'variable_name_list' in kwargs:
        command = ' '.join(command.split(' '))[0:-1]+"'"
    command = (command+"""
  ReadDataOption = Append
  ResetStyle = Yes
  AssignStrandIDs = No
  InitialPlotType = """+kwargs.get('initial_plottype','Cartesian2D')+"""
  InitialPlotFirstZoneOnly = No
  AddZonesToExistingStrands = No
  VarLoadMode = ByName""")

    tp.macro.execute_command(command)

def unzip_files(path):
    for filename in glob.glob(path+'*.dat.gz'):
        zip_cmd = 'gunzip '+filename
        os.system(zip_cmd)

def preplot_files(path,pltfolder):
    for filename in glob.glob(path+'*.dat'):
        #get path to preplot tool and put into nice str format
        preplot_path = str(read_out('which preplot',shell=True))
        preplot_path = preplot_path.split("'")[1].split('\\')[0]
        #use preplot tool
        preplot_cmd = (preplot_path+' '+filename+' '+pltfolder
                       +filename.split('/')[1].split('.dat')[0]+'.plt')
        os.system(preplot_cmd)

#Main program
if __name__ == '__main__':
    PATH = sys.argv[1]
    print('unzipping')
    unzip_files(PATH)
    print('running preplot')
    preplot_files(PATH, 'plt/')
