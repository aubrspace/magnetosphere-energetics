#/usr/bin/env python
"""accesses WIND data from nasa CDA and creates IMF_new.dat for SWMF input
"""
#import pandas as pd
import os
import sys
#import time
import datetime as dt
import numpy as np
from numpy import cos, sin, pi, matmul, deg2rad
import pandas as pd
from cdasws import CdasWs
cdas = CdasWs()
from sscws.sscws import SscWs
ssc = SscWs()
from sscws.coordinates import CoordinateSystem as coordsys
import swmfpy
from geopack import geopack as gp

def read_MFI_SWE_WIND(filename):
    """Reads specific type of file output https://wind.nasa.gov/mfi_swe_plot.php
    Inputs
        filename
    """
    print("""
    pd.read_csv(filename,sep='\s+',header=1,skiprows=[2],
                       parse_dates={'Time_UTC':['Year','DOY','millisecs']},
                       date_parser=datetimeparser2,
                       infer_datetime_format=True, keep_date_col=True)
          """)
    #from spacepy import time as spacetime
    df = pd.read_csv(filename,sep='\s+',header=1,skiprows=[2])
    month, day = spacetime.doy2date(df['Year'],df['DOY'])
    df['mn'] = month; df['dy'] = day
    hrs, mins, secs = [], [], []
    for msec in (df['millisecs']/1000).values:
        hr,min,sec = spacetime.sec2hms(msec)
        hrs.append(hr); mins.append(min); secs.append(sec)
    df['hr'] = hrs; df['min'] = mins; df['sec'] = secs
    dates = []
    for index in df.index:
        datestamp =dt.datetime(df['Year'].iloc[index],df['mn'].iloc[index],
                               df['dy'].iloc[index],df['hr'].iloc[index],
                               df['min'].iloc[index],df['sec'].iloc[index])
        dates.append(datestamp)
    df['Time_UTC'] = dates
    df['yr'] = df['Year']
    df['msec'] = 0
    return df

def datetimeparser(datetimestring):
    return dt.datetime.strptime(datetimestring,'%Y %m %d %H %M %S %f')
def datetimeparser2(datetimestring):
    return dt.datetime.strptime(datetimestring,'%Y %-j %f')

def read_SWMF_IMF(filename):
    """Function takes data from IMF.dat (or whatever SW file is called) and
        returns pandas dataframe with the data
    Inputs
        filename
    Outputs
        df
    """
    if os.path.exists(filename):
        print('reading '+filename)
        with open(filename,'r') as IMF:
            skip = []
            for line in enumerate(IMF):
                if line[1].find('START')!=-1:
                    skip.append(line[0])
                    break
                else:
                    if line[0]!= 0:
                        skip.append(line[0])
        return pd.read_csv(filename,sep='\s+',header=0,skiprows=skip,
                           parse_dates={'Time_UTC':['yr','mn','dy','hr',
                                                    'min','sec','msec']},
                           date_parser=datetimeparser,
                           infer_datetime_format=True, keep_date_col=True)

    else:
        print('missing file '+filename)
        return pd.DataFrame()

def add_swmf_vars(df):
    """Function renames/adds variables needed for SWMF input file
    Inputs
        df
    Outputs
        df
    """
    for key in enumerate(df.keys()):
        #IMF
        if key[1].lower().find('bx')!=-1:
            df = df.rename(columns={key[1]:'bx'})
        elif key[1].lower().find('by')!=-1:
            df = df.rename(columns={key[1]:'by'})
        elif key[1].lower().find('bz')!=-1:
            df = df.rename(columns={key[1]:'bz'})
        #Velocity
        elif key[1].lower().find('vx')!=-1:
            df = df.rename(columns={key[1]:'vx'})
        elif key[1].lower().find('vy')!=-1:
            df = df.rename(columns={key[1]:'vy'})
        elif key[1].lower().find('vz')!=-1:
            df = df.rename(columns={key[1]:'vz'})
        #Density
        elif key[1].lower().find('np')!=-1:
            df = df.rename(columns={key[1]:'dens'})
        #Temperature
        elif ((key[1].lower().find('w')!=-1) or
              (key[1].lower().find('th')!=-1)):
            #temp = vth^2 * m_sw/2 / k, m = 1.04*mp
            df['temp'] = df[key[1]]**2*1.04*1.6726e2/1.3807/2
    ###Time
    #Calculate average shift, use 15Re as rough average bowshock loc
    d = df['x [km]']-15*6371
    dave = d.mean()
    #dave = 238-15#distance from L1=238Re to 15Re just upstream of shock
    vxave = df['vx'].mean()
    shift = dt.timedelta(minutes=-1*dave/vxave/60)
    print('Propagation calculated: '+str(shift)+' min')
    df['Time_UTC'] = df['Time_UTC']+shift

    #Split into specific columns that will go into the swmf input file
    yr,mm,dy,hr,minu,sec,msec = [],[],[],[],[],[],[]
    for entry in df['Time_UTC']:
        yr.append(entry.year)
        mm.append(entry.month)
        dy.append(entry.day)
        hr.append(entry.hour)
        minu.append(entry.minute)
        sec.append(entry.second)
        msec.append(entry.microsecond/1000)
    df['year'] = yr
    df['month'] = mm
    df['day'] = dy
    df['hour'] = hr
    df['min'] = minu
    df['sec'] = sec
    df['msec'] = msec
    return df

def toIMFdict(df, **kwargs):
    """Function converts pandas DataFrame back to dictionary
    Inputs
        df (DataFrame)
        kwargs
            keep_keys (list[str])
    Returns
        imf_dict (dict)
    """
    imf_dict = {}
    #df['times'] = df['Time_UTC']
    for key in kwargs.get('keep_keys',['year','month','day','hour',
                                       'min','sec', 'msec',
                                       'bx','by','bz','vx','vy','vz',
                                       'dens', 'temp']):
        imf_dict[key] = df[key].values
    imf_dict['times'] = swmfpy.io.gather_times(imf_dict)
    imf_dict['density'] = imf_dict['dens']
    imf_dict['temperature'] = imf_dict['temp']
    return imf_dict

def save_to_csv(df, coordinates, outpath):
    """Function saves data from df to csv in correct SWMF input format
    Inputs
        df
        outpath
    """
    data = df[['year','month','day','hour','min', 'sec', 'msec',
               'bx','by','bz','vx','vy','vz', 'dens', 'temp']]
    data.to_csv('IMF.dat', sep=' ', index=False)
    with open('IMF.dat', 'r') as base:
        header = base.readline()
        basefile = base.read()
    with open('IMF.dat', 'w') as final:
        final.write(header+'#COOR\n'+coordinates+'\n\n#START\n'+
                    basefile)
    print('File created, saved to '+outpath+'IMF.dat')

def plot_comparison(ori_df, df, outpath):
    """Function plots original and new solar wind data
    Inputs
        ori_df, df
        outpath
    """
    import matplotlib.pyplot as plt
    figname = 'SolarWind'
    timekeys = ['yr', 'mn', 'dy', 'hr', 'min', 'sec', 'msec', 'Time_UTC']
    num_ax = len(ori_df.keys())-len(timekeys)
    fig, axes = plt.subplots(nrows=num_ax, ncols=1, sharex=True,
                             figsize=[18,12])
    axcount = 0
    for key in ori_df.keys():
        if not any([match.find(key) != -1 for match in timekeys]):
            if not ori_df.empty:
                axes[axcount].plot(ori_df[timekeys[-1]],ori_df[key],
                                label='original')
            axes[axcount].plot(df[timekeys[-1]],df[key],
                               label='NewData(WIND SWE)')
            axes[axcount].set_xlabel(timekeys[-1])
            axes[axcount].set_ylabel(key)
            axes[axcount].legend()
            axcount += 1
    fig.savefig(outpath+'{}.png'.format(figname))

def clean_data(df, **checks):
    """Function cleans up obvious non physical values
    Inputs
        df (DataFrame)- dataset of values
        checks:
            checks (dict{str:float})- eg. {'vx':-5e4}
    Returns
        df (DataFrame)- same format, but modified
    """
    #Time be in the index, lets make a column instead so we can reset
    df['Time_UTC'] = df.index
    df.reset_index(drop=True,inplace=True)

    #Defaults
    checks['vx'] = checks.get('vx',5e4)
    checks['vy'] = checks.get('vy',250)
    checks['vz'] = checks.get('vz',250)
    checks['bx'] = checks.get('bx',1e5)
    checks['by'] = checks.get('by',1e5)
    checks['bz'] = checks.get('bz',1e5)
    checks['Np'] = checks.get('Np',50)
    checks['Vth'] =checks.get('Vth',np.sqrt(600000*2*1.3807/1.6726e2/1.04))
    for (checkvar,checkval) in checks.items():
        assert checkvar in df.keys(), (checkvar+
                    ' not in data but being used as check for clean data!')
        #Check data against threshold
        for loc in df[abs(df[checkvar])>checkval].index:
            if loc != 0:
                df.iloc[loc]=df.iloc[loc-1] #Keep last value
            else: #Find the first good value (hopefully there are lots!)
                first_good_loc = df[abs(df[checkvar])<checkval].index[0]
                df.iloc[loc] = df.iloc[first_good_loc]
    return df


def collect_wind(start, end, **kwargs):
    """Function calls CDAweb with specifics for wind satellite to obtain
        solar wind data
    Inputs
        start, end (datetime.datetime)- start and end times of collection
        kwargs:
            mfilist (list(str))- magnetic field instrument variables
            swelist (list(str))-
            varlist (list(str))- variable list for specific measurements
    Return
        df (DataFrame)- collected and processed data
    """

    #Instruments keys
    dat_key = 'WI_H1_SWE'
    mfi_key = 'WI_H0_MFI'
    swe_key = 'WI_K0_SWE'

    #Use "cdas.get_variables('WI_H1_SWE')" to see options
    #Set up variables needed
    mfilist = kwargs.get('mfilist',['BGSM'])
    swelist = kwargs.get('swelist',
                              ['V_GSM', 'THERMAL_SPD', 'Np', 'SC_pos_GSM'])
    #Query the data using CDAS, padtime because a shift will happen
    status,mfi =cdas.get_data(mfi_key,mfilist,
                      start-dt.timedelta(minutes=kwargs.get('padtime',360)),
                      end  +dt.timedelta(minutes=kwargs.get('padtime',360)))
    swestatus,swe =cdas.get_data(swe_key,swelist,
                      start-dt.timedelta(minutes=kwargs.get('padtime',360)),
                      end  +dt.timedelta(minutes=kwargs.get('padtime',360)))

    #Store data in pandas data frames bc types are weird otherwise
    df_mfi = pd.DataFrame(mfi['BGSM'], columns=['bx','by','bz'],
                          index=mfi['Epoch'])
    df_swe = pd.DataFrame(swe['V_GSM'], columns=['vx','vy','vz'],
                          index=swe['Epoch'])
    df_swe['Vth'] = swe['THERMAL_SPD']
    df_swe['Np'] = swe['Np']
    df_swe['x [km]'] = swe['SC_pos_GSM'][:,0]

    #Interpolate so two streams can combine, seems mfi has nicer cadence
    for v in df_swe.keys():
        df_mfi[v] = np.interp(df_mfi.index,df_swe.index,df_swe[v])

    #Call functions to manipulate raw variables into SWMF variables
    df_clean = clean_data(df_mfi)
    df = add_swmf_vars(df_clean)

    #Trim data for on the original period of interest
    df = df[(df['Time_UTC']>start) & (df['Time_UTC']<end)]

    #Save to a file, first  convert to dict, then use swmfpy
    imf_dict = toIMFdict(df,**kwargs)
    swmfpy.io.write_imf_input(imf_dict,coords='GSM')
    print('File created at IMF.dat')
    assert 'gsm' in mfilist[0].lower(), ('File created with GSM'+
                        'coordinates, but mfi is not in GSM coordinates!!')
    return df

def rotate_gse_gsm(angles,gsedata):
    rot_matrix = np.array([[[1,                0,                0,],
                            [0, cos(deg2rad(th)),-sin(deg2rad(th)),],
                            [0, sin(deg2rad(th)), cos(deg2rad(th)) ]]
                                                    for th in angles])
    #gse_shaped = np.array([[v] for v in gsedata])
    #from IPython import embed; embed()
    #time.sleep(3)
    return [matmul(m,gse) for m,gse in zip(rot_matrix,gsedata)]

def collect_geotail(start, end, **kwargs):
    """Function pulls geotail trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    # Get tilt from somewhere else bc apparently it's not included here...
    print('Getting dipole tilt')
    mms_instrument = 'MMS1_MEC_SRVY_L2_EPHT89D'
    tilt_key = 'mms1_mec_dipole_tilt'
    status,tiltdata = cdas.get_data(mms_instrument,[tilt_key],start,end)
    xtime = [pd.Timestamp(t).value for t in tiltdata['Epoch']]

    # Position
    print('Gathering Position Data')
    positions = {}
    print('\tgeotail')
    df = pd.DataFrame()
    pos_instrument = 'GE_OR_DEF'
    gsm_key = 'GSM_POS'
    epoch_key = 'Epoch'
    status,posdata = cdas.get_data(pos_instrument,[gsm_key],start,end)
    df[['x','y','z']] = posdata[gsm_key]/6371 # from km to Re
    df.index = posdata[epoch_key]
    positions['geotail'] = df

    # Magnetic Field Instrument
    print('Gathering Bfield Data')
    bfield = {}
    print('\tgeotail')
    df = pd.DataFrame()
    mag_instrument = 'GE_K0_MGF'
    bvec_key = 'IB_vector'
    epoch_key = 'Epoch'
    status,magdata = cdas.get_data(mag_instrument,[bvec_key],start,end)
    ytime = [pd.Timestamp(t).value for t in magdata[epoch_key]]
    dipole_angles = np.interp(ytime,xtime,tiltdata[tilt_key])
    bgsm=rotate_gse_gsm(dipole_angles,magdata[bvec_key])
    df[['bx','by','bz']] = bgsm
    df.index=magdata[epoch_key]
    bfield['geotail'] = df

    # Comprehensive Plasma Instrument 
    print('Gathering Plasma Data')
    plasma = {}
    print('\tgeotail')
    df = pd.DataFrame()
    plasma_instrument = 'GE_K0_CPI'
    sw_n_ion_key = 'SW_P_Den'  #solar wind number density
    sw_u_ion_key = 'SW_V'      #solar wind bulk velocity
    sw_e_ion_key = 'SW_P_AVGE' #solar wind average energy
    #hp_n_ion_key = 'HP_P_Den' #hot plasma number density
    #hp_u_ion_key = 'HP_V'     #hot plasma bulk velocity
    hp_p_key = 'W'             #hot plasma plasma pressure
    epoch_key = 'Epoch'
    status,plasmadata = cdas.get_data(plasma_instrument,[sw_n_ion_key,
                                                         sw_u_ion_key,
                                                         sw_e_ion_key,
                                                         #hp_n_ion_key,
                                                         #hp_u_ion_key,
                                                         hp_p_key],
                                                            start,end)
    df['n'] = plasmadata[sw_n_ion_key] # n/cc
    vgsm=rotate_gse_gsm(dipole_angles,plasmadata[sw_u_ion_key])
    df[['vx','vy','vz']] = vgsm # km/s
    df['energy'] = plasmadata[sw_e_ion_key] # eV
    df.index = plasmadata[epoch_key]
    plasma['geotail'] = df
    if kwargs.get('writeData',True):
        ofilename = kwargs.get('ofilename','geotail')
        # Position
        posfile = pd.HDFStore(ofilename+'_pos.h5')
        for key in positions.keys():
            posfile[key] = positions[key]
        posfile.close()
        print('Created ',ofilename+'_pos.h5 output file')
        # B Field
        bfile = pd.HDFStore(ofilename+'_bfield.h5')
        for key in bfield.keys():
            bfile[key] = bfield[key]
        bfile.close()
        print('Created ',ofilename+'_bfield.h5 output file')
        # Plasma
        plasmafile = pd.HDFStore(ofilename+'_plasma.h5')
        for key in plasma.keys():
            plasmafile[key] = plasma[key]
        plasmafile.close()
        print('Created ',ofilename+'_plasma.h5 output file')
    return positions, bfield, plasma

def collect_themis(start, end, **kwargs):
    """Function pulls themis trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    # Position
    print('Gathering Position Data')
    positions = {}
    for num in ['A','D','E']:
        print('\tthemis',num)
        df = pd.DataFrame()
        pos_instrument = 'TH'+num+'_OR_SSC'
        gsm_key = 'XYZ_GSM'
        bx_key = 'MAG_X'
        by_key = 'MAG_Y'
        bz_key = 'MAG_Z'
        epoch_key = 'Epoch'
        status,posdata = cdas.get_data(pos_instrument,
                                       [gsm_key,bx_key,by_key,bz_key],
                                       start,end)
        if posdata:
            df.index = posdata[epoch_key]
            df[['x','y','z']] = posdata[gsm_key] # Re
        positions['themis'+num] = df
    # Flux Gate Magnetometer
    print('Gathering Bfield Data')
    bfield = {}
    for num in ['A','D','E']:
        print('\tthemis',num)
        df = pd.DataFrame()
        fgm_instrument = 'TH'+num+'_L2_FGM'
        bvec_key = 'th'+num.lower()+'_fgs_gsm'
        epoch_key = 'th'+num.lower()+'_fgs_epoch'
        status,fgmdata = cdas.get_data(fgm_instrument,[bvec_key],start,end)
        if fgmdata:
            df.index=fgmdata[epoch_key]
            df[['bx','by','bz']] = fgmdata[bvec_key]
        bfield['themis'+num] = df
    # On board plasma moments 
    print('Gathering Plasma Data')
    plasma = {}
    for num in ['A','D','E']:
        print('\tthemis',num)
        df = pd.DataFrame()
        plasma_instrument = 'TH'+num+'_L2_MOM'
        n_ion_key = 'th'+num.lower()+'_peim_density'
        u_ion_key = 'th'+num.lower()+'_peim_velocity_gsm'
        p_ion_key = 'th'+num.lower()+'_peim_ptot'
        epoch_key = 'th'+num.lower()+'_peim_epoch'
        status,plasmadata = cdas.get_data(plasma_instrument,[n_ion_key,
                                                             u_ion_key,
                                                             p_ion_key],
                                                            start,end)
        if plasmadata:
            df.index = plasmadata[epoch_key]
            df['n'] = plasmadata[n_ion_key] # n/cc
            df[['vx','vy','vz']] = plasmadata[u_ion_key] # km/s
            df['p'] = plasmadata[p_ion_key] # eV/cc
        plasma['themis'+num] = df
    if kwargs.get('writeData',True):
        ofilename = kwargs.get('ofilename','themis')
        # Position
        posfile = pd.HDFStore(ofilename+'_pos.h5')
        for key in positions.keys():
            posfile[key] = positions[key]
        posfile.close()
        print('Created ',ofilename+'_pos.h5 output file')
        # B Field
        bfile = pd.HDFStore(ofilename+'_bfield.h5')
        for key in bfield.keys():
            bfile[key] = bfield[key]
        bfile.close()
        print('Created ',ofilename+'_bfield.h5 output file')
        # Plasma
        plasmafile = pd.HDFStore(ofilename+'_plasma.h5')
        for key in plasma.keys():
            plasmafile[key] = plasma[key]
        plasmafile.close()
        print('Created ',ofilename+'_plasma.h5 output file')
    return positions, bfield, plasma

def collect_mms(start, end, **kwargs):
    """Function pulls mms trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    # Position
    print('Gathering Position Data')
    positions = {}
    mode = kwargs.get('eph_mode','srvy')
    for num in ['1','2','3','4']:
        print('\tmms',num)
        skip = False
        df = pd.DataFrame()
        pos_instrument = 'MMS'+num+'_MEC_'+mode.upper()+'_L2_EPHT89D'
        tilt_key = 'mms'+num+'_mec_dipole_tilt'
        gsm_key = 'mms'+num+'_mec_r_gsm'
        field_status_key = 'mms'+num+'_mec_fieldline_type'
        bfield_key = 'mms'+num+'_mec_bsc_gsm'
        epoch_key = 'Epoch'
        status,posdata = cdas.get_data(pos_instrument,[gsm_key,bfield_key,
                                                       field_status_key,
                                                       tilt_key],
                                                      start,end)
        if posdata:
            df[['x','y','z']] = posdata[gsm_key]/6371 # km->Re
            #df[['bx','by','bz','b']] = posdata[bfield_key] # nT
            df['status'] = posdata[field_status_key]
            df['tilt'] = posdata[tilt_key]
            df.index = posdata[epoch_key]
        positions['mms'+num] = df
    # Flux Gate Magnetometer
    print('Gathering Bfield Data')
    bfield = {}
    mode = kwargs.get('fgm_mode','srvy')
    for num in ['1','2','3','4']:
        print('\tmms',num)
        df = pd.DataFrame()
        fgm_instrument = 'MMS'+num+'_FGM_'+mode.upper()+'_L2'
        bvec_key = 'mms'+num+'_fgm_b_gsm_'+mode+'_l2_clean'
        epoch_key = 'Epoch'
        status,fgmdata = cdas.get_data(fgm_instrument,[bvec_key],start,end)
        if fgmdata:
            df[['bx','by','bz','b']] = fgmdata[bvec_key]
            df.index=fgmdata[epoch_key]
        bfield['mms'+num] = df
    # Dual Ion Spectrometer (distribution moments)
    print('Gathering Plasma Data')
    plasma = {}
    mode = kwargs.get('fpi_mode','fast')
    for num in ['1','2','3','4']:
        print('\tmms',num)
        df = pd.DataFrame()
        plasma_instrument = 'MMS'+num+'_FPI_'+mode.upper()+'_L2_DIS-MOMS'
        n_ion_key = 'mms'+num+'_dis_numberdensity_'+mode
        u_ion_key = 'mms'+num+'_dis_bulkv_gse_'+mode
        tpar_ion_key = 'mms'+num+'_dis_temppara_'+mode
        tperp_ion_key = 'mms'+num+'_dis_tempperp_'+mode
        epoch_key = 'Epoch'
        status,plasmadata = cdas.get_data(plasma_instrument,[n_ion_key,
                                                             u_ion_key,
                                                             tpar_ion_key,
                                                             tperp_ion_key],
                                                            start,end)
        if plasmadata and posdata:
            df['n'] = plasmadata[n_ion_key] # n/cc
            """
            for gse,gse_keys in [[[df['vx_gse'],df['vy_gse'],df['vz_gse']],
                                  ['vx','vy','vz']]]:
                x = np.zeros(len(gse[0]))
                y = np.zeros(len(gse[0]))
                z = np.zeros(len(gse[0]))
                for i,t in enumerate(df[epoch_key]):
                    ut = (t-t0).total_seconds()
                    gp.recalc(ut)
                    gsm = gp.gsmgse(gse[0][i],gse[1][i],gse[2][i],-1)
                    x[i],y[i],z[i] = gsm
            """
            # Convert gse velocity to gsm
            xtime = [t.value for t in positions['mms'+num].index]
            ytime = [pd.Timestamp(t).value for t in plasmadata[epoch_key]]
            dipole_angles = np.interp(ytime,xtime,posdata[tilt_key])
            vgsm=rotate_gse_gsm(dipole_angles,plasmadata[u_ion_key])
            df[['vx','vy','vz']] = vgsm # km/s
            df['tpar'] = plasmadata[tpar_ion_key] # eV
            df['tperp'] = plasmadata[tperp_ion_key] # eV
            df.index = plasmadata[epoch_key]
        plasma['mms'+num] = df
    # Hot Plasma Composition Analyzer (other ion species moments)
    # TODO if we need to grab 'MMS'+num+'_HPCA_'+mode.upper()+'_L2_DIS-MOMS'
    #   has H+
    #       He+
    #       He++
    #       O+
    #       Spin averaged B
    if kwargs.get('writeData',True):
        ofilename = kwargs.get('ofilename','mms')
        # Position
        posfile = pd.HDFStore(ofilename+'_pos.h5')
        for key in positions.keys():
            posfile[key] = positions[key]
        posfile.close()
        print('Created ',ofilename+'_pos.h5 output file')
        # B Field
        bfile = pd.HDFStore(ofilename+'_bfield.h5')
        for key in bfield.keys():
            bfile[key] = bfield[key]
        bfile.close()
        print('Created ',ofilename+'_bfield.h5 output file')
        # Plasma
        plasmafile = pd.HDFStore(ofilename+'_plasma.h5')
        for key in plasma.keys():
            plasmafile[key] = plasma[key]
        plasmafile.close()
        print('Created ',ofilename+'_plasma.h5 output file')
    return positions, bfield, plasma

def collect_cluster(start, end, **kwargs):
    """Function pulls cluster trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    # Position
    print('Gathering Position Data')
    positions = {}
    pos_instrument = kwargs.get('instrument','CL_SP_AUX')
    gsmgse_key = kwargs.get('gsmgse_key','gse_gsm__CL_SP_AUX')
    gseref_key = kwargs.get('gseref_key','sc_r_xyz_gse__CL_SP_AUX')
    gse_keys =['sc_dr'+str(num)+'_xyz_gse__CL_SP_AUX' for num in [1,2,3,4]]
    status,posdata = cdas.get_data(pos_instrument,gse_keys.append(gseref_key),
                                   start,end)
    for sc in gse_keys[0:-1]:#drop appended ref key
        num = sc.split('sc_dr')[1].split('_')[0]
        scpos_gse = (posdata[sc]+posdata[gseref_key])/6371 #units are km
        scpos = rotate_gse_gsm(posdata[gsmgse_key],scpos_gse)
        df = pd.DataFrame(scpos, columns=['x','y','z'],
                          index=posdata['Epoch__CL_SP_AUX'])
        positions['cluster'+num] = df
    # Flux Gate Magnetometer
    print('Gathering Bfield Data')
    bfield = {}
    for num in ['1','2','3','4']:
        df = pd.DataFrame()
        fgm_instrument = 'C'+num+'_CP_FGM_SPIN'
        bvec_key = 'B_vec_xyz_gse__C'+num+'_CP_FGM_SPIN'
        epoch_key = 'Epoch__C'+num+'_CP_FGM_SPIN'
        status,fgmdata = cdas.get_data(fgm_instrument,[bvec_key],start,end)
        columns = ['bx','by','bz']
        if fgmdata == None:
            fgm_instrument = 'C'+num+'_UP_FGM'
            bvec_key = 'B_xyz_gse__C'+num+'_UP_FGM'
            status_key = 'Status__C'+num+'_UP_FGM'
            status,fgmdata=cdas.get_data(fgm_instrument,[bvec_key,status_key],
                                         start,end)
            df[['status1','status2','status3','status4']]=fgmdata[status_key]
            epoch_key = 'Epoch__C'+num+'_UP_FGM'
        xtime = [t.value for t in positions['cluster'+num].index]
        ytime = [pd.Timestamp(t).value for t in fgmdata[epoch_key]]
        dipole_angles = np.interp(ytime,xtime,posdata[gsmgse_key])
        bgsm = rotate_gse_gsm(dipole_angles,fgmdata[bvec_key])
        df[['bx','by','bz']] =  bgsm
        df.index = fgmdata[epoch_key]
        bfield['cluster'+num] = df
    # Cluster Ion Spectrometry (CIS)
    print('Gathering Plasma Data')
    plasma = {}
    for num in ['1','2','3','4']:
        df = pd.DataFrame()
        # Variable keys
        cis_instrument = 'C'+num+'_PP_CIS'
        n_proton_key = 'N_p__C'+num+'_PP_CIS'# n/cc
        u_proton_key = 'V_p_xyz_gse__C'+num+'_PP_CIS' #km/s
        Tpar_proton_key = 'T_p_par__C'+num+'_PP_CIS' #MK
        Tperp_proton_key = 'T_p_perp__C'+num+'_PP_CIS' #MK
        epoch_key = 'Epoch__C'+num+'_PP_CIS'
        # Get Data
        status,plasmadata = cdas.get_data(cis_instrument,[n_proton_key,
                                                          u_proton_key,
                                                          Tpar_proton_key,
                                                          Tperp_proton_key],
                                                          start,end)
        if plasmadata!=None:
            if len(plasmadata[epoch_key])<10:
                plasmadata = None
                plasma['cluster'+num] = pd.DataFrame()
            else:
                df['n'] = plasmadata[n_proton_key]
                xtime = [t.value for t in positions['cluster'+num].index]
                ytime = [pd.Timestamp(t).value for t in plasmadata[epoch_key]]
                dipole_angles = np.interp(ytime,xtime,posdata[gsmgse_key])
                vgsm=rotate_gse_gsm(dipole_angles,plasmadata[u_proton_key])
                df[['vx','vy','vz']] = vgsm
                df['Tpar'] = plasmadata[Tpar_proton_key]
                df['Tperp'] = plasmadata[Tperp_proton_key]
                df.index = plasmadata[epoch_key]
                plasma['cluster'+num] = df
        else:
            plasma['cluster'+num] = pd.DataFrame()
    if kwargs.get('writeData',True):
        ofilename = kwargs.get('ofilename','cluster')
        # Position
        posfile = pd.HDFStore(ofilename+'_pos.h5')
        for key in positions.keys():
            posfile[key] = positions[key]
        posfile.close()
        print('Created ',ofilename+'_pos.h5 output file')
        # B Field
        bfile = pd.HDFStore(ofilename+'_bfield.h5')
        for key in bfield.keys():
            bfile[key] = bfield[key]
        bfile.close()
        print('Created ',ofilename+'_bfield.h5 output file')
        # Plasma
        plasmafile = pd.HDFStore(ofilename+'_plasma.h5')
        for key in plasma.keys():
            plasmafile[key] = plasma[key]
        plasmafile.close()
        print('Created ',ofilename+'_plasma.h5 output file')
    return positions, bfield, plasma

def collect_orbits(start,end,sourcelist):
    gsm = coordsys.GSM
    for source in sourcelist:
        pass
    #TODO I have no idea why it won't let me specify what I want out of the "get_locations" method, it won't accept GSM as a coordinate system is local time is requested i guess??
    #from IPython import embed; embed()
    pass

#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    path_to_ori_file = None
    #start = dt.datetime(2019,5,13,12,0)
    #end = dt.datetime(2019,5,15,12,0)
    #start = dt.datetime(2014,2,18,4,0)
    #end = dt.datetime(2014,2,25,0,0)
    start = dt.datetime(2022,2,2,5)
    end = dt.datetime(2022,2,5,12)
    outpath = './'
    plot_data = True
    #######################################################################

    #wind = collect_wind(start, end)
    #cluster_pos, cluster_b,cluster_plasma = collect_cluster(start, end)
    #mms_pos, mms_b,mms_plasma = collect_mms(start, end)
    #themis_pos, themis_b,themis_plasma = collect_themis(start, end)
    #geotail_pos, geotail_b, geotail_plasma = collect_geotail(start,end)

    ## For a list of all observatories for location plots use:
    #   observatories = ssc.get_observatories()
    #   obslist = [o['Name'] for o in observatories['Observatory']]                          
    sourcelist = ['cluster1','cluster2','cluster3','cluster4',
                  'geotail',
                  'mms1','mms2','mms3','mms4',
                  'themisa','themisd','themise']
    #collect_orbits(start,end,sourcelist)
    #omni = swmfpy.web.get_omni_data(start,end)

    #Additional options
    if path_to_ori_file is not None:
        ori_df = read_SWMF_IMF(path_to_ori_file)
        if not ori_df.empty:
            plot_comparison(ori_df, df, outpath)
    if plot_data:
        try:
            from util.plotSW import plot_solarwind as plotsw
        except ModuleNotFoundError:
            print("Unable to plot, can't find plotSW.py!")
        else:
            plotsw(WIND=wind, CLUSTER=cluster, OMNI=omni)
            #plotsw(WIND=wind)
            #plt.show()
            plt.figure(1).savefig('IMF.png')
            print('Figure saved to IMF.png')
