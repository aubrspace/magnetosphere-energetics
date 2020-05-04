#/usr/bin/env python

import glob
import os
from subprocess import check_output as read_out

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
    print('unzipping')
    unzip_files('./')
    print('running preplot')
    preplot_files('./', 'plt/')
