#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
import sys,os
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import time
import glob
import numpy as np
import datetime as dt
import pandas as pd
import tecplot as tp
from tecplot.constant import ReadDataOption
from global_energetics.makevideo import get_time, time_sort

def add_units(var):
    """Function adds appropriate units to field data variable
    Inputs
        var
    """
    if var =='X' or var=='Y' or var=='Z':
        return var+' [R]'
    if var =='Ux' or var=='Uy' or var=='Uz':
        return 'U_'.join(var.split('U'))+' [km/s]'
    if var =='Bx' or var=='By' or var=='Bz':
        return 'B_'.join(var.split('B'))+' [nT]'
    if var =='jx' or var=='jy' or var=='jz':
        return 'J_'.join(var.split('j'))+' [`mA/m^2]'
    if var =='Rho':
        return var + ' [amu/cm^3]'
    if var =='P':
        return var + ' [nPa]'
    if var =='status':
        return 'Status'
    else:
        return var

def load_swmf_sat(filename, field_variables):
    """Function loads satellite file based on SWMF PARAM.in settings
    Inputs
        filename
        field_variables
    """
    with open(filename,'r') as satfile:
        header = satfile.readline()
        satname = header.split('/')[-1].split('.')[0]
        variables = satfile.readline().split('\n')[0].split(' ')[1::]
    varstring = ''
    u_variables = []
    for var in enumerate(variables):
        w_units = add_units(var[1])
        u_variables.append(w_units)
        varstring = varstring+w_units+','
    varstring = varstring.rstrip(',')
    varnamelist = ''
    for field_var in enumerate(field_variables):
        for var in variables:
            if field_var[1] == add_units(var):
                varnamelist = (varnamelist +
                               '\"{}\";\"{}\" '.format(field_var[1], var))
                found = True
        if not found:
            varnamelist = (varnamelist + '\"{}\" '.format(field_var[1]))
        found = False
    for var in variables:
        if not any([varname == add_units(var)
                    for varname in field_variables]):
            varnamelist = (varnamelist + '\"{}\" '.format(var))

    readDataCmd = ("'"+'\"'+os.getcwd()+'/'+filename+'\" '+
    '\"VERSION=100 FILEEXT=\\\"*.txt\\\" '+
    'FILEDESC=\\\"General Text\\\" '+
    '\"+\"\"+\"TITLE{SEARCH=LINE  '+
                    'NAME=\\\"'+satname+'\\\" '+
                    'LINE=1 '+
                    'DELIMITER=AUTO '+
                    'WIDTH=10 }'+
    '\"+\"\"+\"VARIABLES{\"+\"SEARCH=NONE  '+
                             'NAMES = \\"'+varstring+'\\"\"+\" '+
                             'LOADED= All  '+
                             'STARTLINE=1 '+
                             'ENDLINE=1 '+
                             'DELIMITER=AUTO '+
                             'WIDTH=10 }'+
    '\"+\"\"+\"DATA\"+\"{\"+\"IGNORENONNUMERICTOKENS=TRUE '+
                             'IMPORT\"+\"{\"+\"STARTID=LINE '+
                                        '{\"+\"LINE=3 }'+
                                    '\"+\"\"+\"ENDID=EOF '+
                                        '{\"+\"LINE=1 }\"+\"\"+\"'+
                             'FORMAT=IJKPOINT '+
                             'DELIMITER=AUTO '+
                             'WIDTH=10 }'+
    '\"+\"\"+\"DIMENSION\"+\"{\"+\"AUTO=TRUE '+
                             'CREATEMULTIPLEZONES=FALSE }'+
    '\"+\"}\"+\"GLOBALFILTERS{\"+\"USEBLANKCELLVALUE=FALSE '+
                                  'BLANKCELLVALUE=0.000000 }\"'+"' ")
    readDataCmd = ("""$!ReadDataSet  """+readDataCmd+"""
                   DataSetReader = 'General Text Loader'
                   VarNameList = '"""+varnamelist+"""'
                   ReadDataOption = Append
                   ResetStyle = No
                   AssignStrandIDs = No
                   InitialPlotType = XYLine
                   InitialPlotFirstZoneOnly = No
                   AddZonesToExistingStrands = No
                   VarLoadMode = ByName""")
    tp.macro.execute_command(readDataCmd)
    tp.active_frame().dataset.zone('Zone 1').name = satname
    return satname

def add_currentlocation(zonenames, field_data):
    t0 = dt.datetime(2000,1,1,0)
    #Get corresponding satellite fieldmap variable indices
    satindices = []
    for name in zonenames:
        satindices.append(int(field_data.zone(name).index))
        if len([zn for zn in field_data.zones('loc_'+name)]) > 0:
            loc_satzone = field_data.zone('loc_'+name)
        else:
            #create local sat zone with sat current position
            loc_satzone =  field_data.add_ordered_zone('loc_'+name, [1,1,1])
            #get the current position of the satellite based on aux data
            eventstring =field_data.zone('global_field').aux_data['TIMEEVENT']
            startstring=field_data.zone('global_field').aux_data[
                                                          'TIMEEVENTSTART']
            eventdt = dt.datetime.strptime(eventstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            startdt = dt.datetime.strptime(startstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            deltadt = eventdt-startdt
            tvals = field_data.zone(name).values('t').as_numpy_array()
            times = [dt.datetime(2000,1,1,0)+dt.timedelta(seconds=int(t))
                                                                for t in tvals]
            dtimes = [(abs(t-eventdt)).total_seconds() for t in times]
            i_time = np.where(dtimes==np.min(dtimes))[0][0]
            xvals = field_data.zone(name).values('X *').as_numpy_array()
            yvals = field_data.zone(name).values('Y *').as_numpy_array()
            zvals = field_data.zone(name).values('Z *').as_numpy_array()
            svals = field_data.zone(name).values('Status').as_numpy_array()
            xpos = xvals[i_time]
            ypos = yvals[i_time]
            zpos = zvals[i_time]
            #status = svals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            loc_satzone.values('X *')[::] = xpos
            loc_satzone.values('Y *')[::] = ypos
            loc_satzone.values('Z *')[::] = zpos
            #loc_satzone.values('Status')[0] = status

def get_satzone_fromHdf5(field_data,hdffile,**kwargs):
    t0 = dt.datetime(2000,1,1,0)
    with pd.HDFStore(hdffile) as store:
        for key in store.keys():
            df = store[key]
            keyzone = field_data.add_ordered_zone(key.replace('/',''),len(df))
            keyzone.values('X *')[::] = df['x']
            keyzone.values('Y *')[::] = df['y']
            keyzone.values('Z *')[::] = df['z']
            #keyzone.values('Status')[::] = df['status']
            if 't' not in field_data.variable_names:
                field_data.add_variable('t')
            keyzone.values('t')[::]=[(t-t0).total_seconds()for t in df.index]
            print(f'Created new zone from {key}')
    return

def get_satellite_zones(field_data, datapath, *, coordsys='GSM'):
    """Function to find satellite trace data (if avail) and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of the event
        datapath- str path to ionosphere data
        coordsys- coordinate system of the field data
    """
    satfiles = glob.glob(datapath+'/*.sat')
    satzones = []
    if satfiles == []:
        print('No satellite data at {}!'.format(datapath))
        return []
    else:
        variables = []
        for var in field_data.variables():
            variables.append(var.name)
        for satfile in satfiles[0::]:
            print('reading: {}'.format(satfile))
            satzonename = load_swmf_sat(satfile, variables)
            satzones.append(satzonename)
    #reset variable names
    for var in field_data.variables('*\n*'):
        var.name = var.name.split('\n')[0]
    #add specific location data
    add_currentlocation(satzones, field_data)
    return satzones

def txt_to_hdf5(infile,**kwargs):
    """converts a text file into hd5 for easier working w pandas
    Inputs
        infile
        kwargs:
            skiprows- DEFAULT 65!!!
    Returns
        outfile
    """
    #NOTE could use pandas parsing arguments in 'read_csv' but it was finicky
    #     Added the '#' characters to the .txt file myself to denote skips
    outfile = 'satout.h5'
    data = pd.read_csv('mysatdata.txt',comment='#',header=None,sep='\s+',
                       skiprows=kwargs.get('skiprows',65))
    timevalues = []
    parser_str = kwargs.get('parser_str','%y/%m/%d %H:%M:%S')
    for values in data.values:
        timevalues.append(dt.datetime.strptime(' '.join(values[0:2]),
                                               parser_str))
    data['time'] = timevalues
    data.drop(columns=[0,1],inplace=True)
    data.columns = ['sat','Xgsm','Ygsm','Zgsm','time']
    store = pd.HDFStore(outfile)
    for sat in data.sat.unique():
        satdata = data.copy(deep=True)[data['sat']==sat]
        satdata.reset_index(drop=True,inplace=True)
        store[sat] = satdata
    store.close()
    print('Created ',outfile,'!')
    return outfile

def interpolate_satellite_loc(satfile,source_files,ofilepath):
    os.makedirs(ofilepath, exist_ok=True)
    # Read satellite location file for available satellites
    locfile = pd.HDFStore(satfile)
    # glob for the available 3d files and get source_times
    source_times=[get_time(f) for f in source_files]
    i = 1
    nsat = len(locfile.keys())
    ntimes = len(source_times)
    total = nsat*ntimes
    interp_dict = {}
    # For each satellite:
    for sat in locfile.keys():
        print('{:.1%} Interpolating '.format(i/nsat),sat)
        # Initiate DataFrame for this satellite output
        vsat_df = pd.DataFrame()
        vsat_df['time'] = source_times
        # Interpolate locations and loc_times to source_times
        loc_times = locfile[sat].index
        xtimes = [float(t) for t in loc_times.values]
        ytimes = [pd.Timestamp(t).value for t in source_times]
        vsat_df['Xgsm'] = np.interp(ytimes,xtimes,locfile[sat]['x_gsm'].values)
        vsat_df['Ygsm'] = np.interp(ytimes,xtimes,locfile[sat]['y_gsm'].values)
        vsat_df['Zgsm'] = np.interp(ytimes,xtimes,locfile[sat]['z_gsm'].values)
        interp_dict[sat] = vsat_df
        i+=1
    locfile.close()
    print(f'Creating {os.path.join(ofilepath,satfile.split("/")[-1])}')
    outfile = pd.HDFStore(os.path.join(ofilepath,satfile.split('/')[-1]))
    for sat in interp_dict.keys():
        outfile[sat] = interp_dict[sat]
        print(f'\tWriting {sat} to file')
    '''
    i=1
    from IPython import embed; embed()
    for sourcefile in source_files:
        eventtime = get_time(sourcefile)
        print('{:.1%} Packaging '.format(i/ntimes),eventtime)
        # Create 'virtual_sat_out' file
        datestring = (str(eventtime.year)+'-'+str(eventtime.month)+'-'+
                          str(eventtime.day)+'-'+str(eventtime.hour)+'-'+
                          str(eventtime.minute))
        outfile = pd.HDFStore(os.path.join(ofilepath,sat+
                                           '_'+datestring+'.h5'))
        for sat in interp_dict.keys():
            save_df = interp_dict[sat][interp_dict[sat]['time']==eventtime]
            outfile[sat] = save_df
        outfile.close()
        i+=1
    '''
    return ofilepath

def extract_satellite(sourcefile,satfile):
    """Loads source file,returns {sat:DataFrame} at corresponding vsat locs
    Inputs
        sourcefile
        satfile
        outfile
    Returns
        vsatdict {sat(str):data(DataFrame)}
    """
    # Create 'vsat' dict
    vsatdict = {}
    # Read satellite location file for available satellites
    locfile = pd.HDFStore(satfile)
    sourcetime = get_time(sourcefile)
    i = 1
    nsat = len(locfile.keys())
    tp.new_layout()
    tp.data.load_tecplot(sourcefile)
    variable_names = tp.active_frame().dataset.variable_names
    for sat in locfile.keys():
        #X,Y,Z = locfile[sat][locfile[sat]['time']==sourcetime][[
        #                                      'Xgsm','Ygsm','Zgsm']].values[0]
        t,X,Y,Z = locfile[sat].values[0]
        #X,Y,Z = locfile[sat][locfile[sat].index==sourcetime][[
        #                                      'x','y','z']].values[0]
        # Load tecplot source data file at the timestamp and extract
        print('{:.1%} Extracting '.format(i/nsat),sat,' at ',sourcetime)
        probe = tp.data.query.probe_at_position(X,Y,Z)[0]
        snapshot = pd.DataFrame(data=[probe],columns=variable_names)
        snapshot['time'] = sourcetime
        vsatdict[sat] = snapshot
        i+=1
    locfile.close()
    return vsatdict

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()
    if False:
        # Convert observed .txt file to hdf
        #hdffile = txt_to_hdf5('mysatdata.txt')
        for infile in glob.glob('inputs/*.txt'):
            hdf = txt_to_hdf5(infile)
    else:
        hdffile = 'obssatloc.h5'
    if True:
        # Convert observed .h5 file to interpolated (event specific)
        MHDDIR = 'run_mothersday/GM/IO2/'
        # glob for the available 3d files and get source_times
        source_files = sorted(glob.glob(MHDDIR+'3d__var*.plt'),
                              key=time_sort)
        for satfile in glob.glob('mothersday_sats/*.h5'):
            print(f'satellite: {satfile}')
            interpolate_satellite_loc(satfile,source_files,
                                              'mothersday_sats/interp/')
    else:
        ofilepath = 'star2satloc'

    #results = extract_satellite(source_files[0],ofilepath)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
