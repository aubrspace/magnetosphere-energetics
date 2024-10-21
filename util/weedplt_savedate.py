#/usr/bin/env python
"""Modifies tecplot .plt files to be save for the VisIt reader in Paraview
"""
import sys,os
import glob
import datetime as dt
import tecplot as tp
#Interpackage imports
try:
    from global_energetics.makevideo import time_sort
except ImportError:
    print('global_energetics not found, unable to standardize names')

def weed_savedate(pathtofiles,keytime,keep_side,*,verbose):
    """
    #TODO
    """
    # Glob for the files at this path
    filelist = sorted(glob.glob(f'{pathtofiles}/3d*var*.plt'),
                      key=time_sort)
    if verbose:
        print(filelist,'\n')
    keeps = [[]]*len(filelist)
    savedates = [[]]*len(filelist)
    for i,infile in enumerate(filelist):
        if verbose:
            print(f'\topening {infile.split("/")[-1]} ...')
        # Read file with tecplot and scrape the aux data
        tp.new_layout()
        ds = tp.data.load_tecplot(infile)
        aux = ds.zone(0).aux_data.as_dict()
        savedate = dt.datetime.strptime(aux['SAVEDATE'],
                                             'Save Date: %Y/%m/%d at %H:%M:%S')
        savedates[i]=savedate
        # Determine keep or delete based on save date relative to key time
        isafter = (savedate-keytime).total_seconds()>0
        if keep_side == 'before':
            keeps[i] = not isafter
        else:
            keeps[i] = isafter
    # Display the result
    if verbose:
        print(f'\033[92m KEEP \033[00m')
        print(f'\033[95m DELETE \033[00m')
        print('\tFile\t\tSAVEDATE')
        print('***************************')
        for i,infile in enumerate(filelist):
            if keeps[i]:
                color = 92
            else:
                color = 95
            print(f'\t\033[{color}m {infile.split("/")[-1]}\t'
                  f' {savedates[i]}\033[00m')
    else:
        print(f'\033[95m DELETE \033[00m')
        print('***************************')
        for i,infile in enumerate(filelist):
            if not keeps[i]:
                print(f'\t\033[95m {infile} \033[00m')
    if not all(keeps):
        # Ask user to confirm deletion
        done = False
        i==0
        while not done:
            i+=1
            if i==1:
                delete_entry = input('\nDelete these files? [y/n]\n')
            if (delete_entry!='y') and (delete_entry!='n'):
                delete_entry = input('\noops, try again. Delete? [y/n]\n')
            elif delete_entry=='y':
                doDelete = True
                done = True
            elif delete_entry=='n':
                doDelete = False
                done = True
            if i>3:
                done = True
                doDelete = False
        if doDelete:
            for i,infile in enumerate(filelist):
                if not keeps[i]:
                    os.remove(infile)
                    print(f'\tdeleted: {infile.split("/")[-1]}')
    else:
        print('Nothing to delete, all files pass')
    # init:
    #   savedates
    # for each file
    #   get aux
    #   add aux to savedates array
    # if verbose:
    #   print all file sorted by save date
    #   color by delete or keep
    #   print a key as first line
    # else:
    #   print dates which are after savedate

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print("""
    Script to check tecplot .plt files savedate and delete accordingly

    Requirements:
        python packages: sys,os,glob,datetime
                          optional- global_energetics.makevideo and depends
                                    (for time sorting function)

    Usage: python pltfixer.py [-flags] PATHTOFILES

    Options:
        -h  --help      prints this message then exit
        -v              prints TecplotErrors if any
        -c              uses pytecplot in 'connected' mode,
                            see pytecplot for more details
        -k              keep side, "after" or "before"
        -d              deletes files with savedate after given argument
                            arg in the form: "yyyy-mo-dy hr:mn:sc"

    Example:
        python pltfixer.py -c -d "2022-11-3 12:00:00" localdbug/fte/
            this will check all the .plt files in localdbug/fte/
              using the 'connected' mode then delete files saved after the date

        multiPytec pltfixer.py localdbug/fte/
            this will check all the .plt files in localdbug/fte/
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
    if pathtofiles=='weedplt_savedate.py' or len(sys.argv)==1:
        print('No path to files given!! use -h or --help for more info')
        exit()
    if '-v' in sys.argv:
        doVerbose=True
    else:
        doVerbose=False
    if '-c' in sys.argv:
        tp.session.connect()
    if '-k' in sys.argv:
        keepside = sys.argv[sys.argv.index('-k')+1]
        if (keepside.lower()!='after') and (keepside.lower()!='before'):
            print('Bad argument after -k, not sure which side to keep!!',
                  ' (before, or after)')
            exit
    else: keepside = 'after'
    if '-d' in sys.argv:
        keytime = sys.argv[sys.argv.index('-d')+1]
        keytime = dt.datetime.strptime(keytime,'%Y-%m-%d %H:%M:%S')
    else:
        keytime = None
    if doVerbose:
        print(f'\nKeeping {keepside.lower()} "{keytime}"')

    # Modify files
    weed_savedate(pathtofiles,keytime,keepside.lower(),verbose=doVerbose)
    print('DONE')
