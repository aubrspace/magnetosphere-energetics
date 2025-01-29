#/usr/bin/env python
"""Modifies tecplot .plt files to be save for the VisIt reader in Paraview
"""
import sys,os,glob,time
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import datetime as dt
import tecplot as tp
#Interpackage imports
try:
    from global_energetics.makevideo import time_sort,get_time
    nosort = False
except ImportError:
    print('global_energetics not found cant sort files!')
    nosort = True

def weed_savedate(pathtofiles,keytime,keep_side,*,verbose):
    """
    #TODO
    """
    # Glob for the files at this path
    if nosort:
        filelist = glob.glob(f'{pathtofiles}/3d*var*.plt')
    else:
        filelist = sorted(glob.glob(f'{pathtofiles}/3d*var*.plt'),
                          key=time_sort)
    times = [get_time(f) for f in filelist]
    if verbose:
        print(filelist,'\n')
    kills = []
    skip = 0
    for i,infile in enumerate(filelist):
        if skip>0:
            skip-=1
            continue
        if verbose:
            print(f'\t{i}/761 opening {infile.split("/")[-1]} ...')
        # Check that this timestamp is only included once
        ftime = get_time(infile)
        count = times.count(ftime)
        if count>1:
            skip = count-1
            savedates = [[]]*count
            for k in range(0,count):
                # Read file with tecplot and scrape the aux data
                tp.new_layout()
                ds = tp.data.load_tecplot(filelist[i+k])
                aux = ds.zone(0).aux_data.as_dict()
                savedate = dt.datetime.strptime(aux['SAVEDATE'],
                                             'Save Date: %Y/%m/%d at %H:%M:%S')
                savedates[k] = savedate
            # keep only the one that was created most recently
            savevalues = [(s-savedates[0]).total_seconds() for s in savedates]
            for k,v in enumerate(savevalues):
                if v!=min(savevalues):
                    kills.append(filelist[i+k])
                    if verbose:
                        print(f'Flagging {i+k}: {filelist[i+k]}')
    if len(kills)>0:
        os.makedirs(os.path.join(pathtofiles,'trash'),exist_ok=True)
    for f in kills:
        # Move file to a trash file
        os.rename(f,f.replace('3d__','trash/3d__'))

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
