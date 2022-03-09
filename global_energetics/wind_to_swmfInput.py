#/usr/bin/env python
"""accesses WIND data from nasa CDA and creates IMF_new.dat for SWMF input
"""
import pandas as pd
import os
import sys
import time
import datetime as dt
import numpy as np
from numpy import cos, sin, pi, matmul, deg2rad
import matplotlib.pyplot as plt
from cdasws import CdasWs
cdas = CdasWs()
import swmfpy

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
    from spacepy import time as spacetime
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

def collect_cluster(start, end, **kwargs):
    """Function pulls cluster trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    #Instrument keys
    instrument = kwargs.get('instrument','CL_SP_AUX')
    gsmgse_key = kwargs.get('gsmgse_key','gse_gsm__CL_SP_AUX')
    gseref_key = kwargs.get('gseref_key','sc_r_xyz_gse__CL_SP_AUX')
    gse_keys =['sc_dr'+str(num)+'_xyz_gse__CL_SP_AUX' for num in [1,2,3,4]]
    status,data = cdas.get_data(instrument,gse_keys.append(gseref_key),
                                start,end)
    #th = data[gsmgse_key]
    rot_matrix = [[[1,                0,                0,],
                   [0, cos(deg2rad(th)),-sin(deg2rad(th)),],
                   [0, sin(deg2rad(th)), cos(deg2rad(th)) ]]
                                                for th in data[gsmgse_key]]
    for sc in gse_keys[0:-1]:#drop appended ref key
        scpos_gse = (data[sc]+data[gseref_key])/6371 #confirm units are km
        scpos=[matmul(m[1],scpos_gse[m[0]]) for m in enumerate(rot_matrix)]
        df = pd.DataFrame(scpos, columns=['x','y','z'],
                          index=data['Epoch__CL_SP_AUX'])
    return df

def collect_themis(start, end, **kwargs):
    """Function pulls themis trajectory and orbit data from cdaweb
    Inputs
        satkey (str)- themis, cluster, geotail, etc.
        start, end (datetime.datetime)- start and end times of collection
    Returns
        df (DataFrame)- collected and processed data
    """
    #Probe keys
    probe_keys = kwargs.get('probe_keys',['THA_','THB_','THC_',
                                                            'THD_','THE_'])
    '''
    #Instrument keys
    instrument = kwargs.get('instrument','CL_SP_AUX')
    gsmgse_key = kwargs.get('gsmgse_key','gse_gsm__CL_SP_AUX')
    gseref_key = kwargs.get('gseref_key','sc_r_xyz_gse__CL_SP_AUX')
    gse_keys =['sc_dr'+str(num)+'_xyz_gse__CL_SP_AUX' for num in [1,2,3,4]]
    status,data = cdas.get_data(instrument,gse_keys.append(gseref_key),
                                start,end)
    #th = data[gsmgse_key]
    rot_matrix = [[[1,                0,                0,],
                   [0, cos(deg2rad(th)),-sin(deg2rad(th)),],
                   [0, sin(deg2rad(th)), cos(deg2rad(th)) ]]
                                                for th in data[gsmgse_key]]
    for sc in gse_keys[0:-1]:#drop appended ref key
        scpos_gse = (data[sc]+data[gseref_key])/6371 #confirm units are km
        scpos=[matmul(m[1],scpos_gse[m[0]]) for m in enumerate(rot_matrix)]
        df = pd.DataFrame(scpos, columns=['x','y','z'],
                          index=data['Epoch__CL_SP_AUX'])
    '''
    return df


#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    path_to_ori_file = None
    #start = dt.datetime(2019,5,13,12,0)
    #end = dt.datetime(2019,5,15,12,0)
    start = dt.datetime(2014,2,18,4,0)
    end = dt.datetime(2014,2,25,0,0)
    outpath = './'
    plot_data = True
    #######################################################################

    wind = collect_wind(start, end)
    cluster = collect_cluster(start, end)
    omni = swmfpy.web.get_omni_data(start,end)

    #Additional options
    if path_to_ori_file is not None:
        ori_df = read_SWMF_IMF(path_to_ori_file)
        if not ori_df.empty:
            plot_comparison(ori_df, df, outpath)
    if plot_data:
        try:
            from plotSW import plot_solarwind as plotsw
        except ModuleNotFoundError:
            print("Unable to plot, can't find plotSW.py!")
        else:
            plotsw(WIND=wind, CLUSTER=cluster, OMNI=omni)
            #plotsw(WIND=wind)
            #plt.show()
            plt.figure(1).savefig('IMF.png')
            print('Figure saved to IMF.png')
