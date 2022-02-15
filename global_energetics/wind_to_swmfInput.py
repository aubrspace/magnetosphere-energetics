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
from spacepy import time as spacetime

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
            df['temp'] = df[key[1]]**2*1.6726e2/1.3807/2
        #Time
        elif key[1].lower().find('epoch')!=-1:
            yr,mm,dy,hr,minu,sec,msec = [],[],[],[],[],[],[]
            for entry in df[key[1]]:
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
            df = df.rename(columns={key[1]:'Time_UTC'})
    #Calculate average shift
    #d = df['xgse']-15
    #dave = d.mean()
    dave = 238-15#distance from L1=238Re to 15Re just upstream of shock
    vxave = df['vx'].mean()
    shift = dt.timedelta(minutes=-1*dave*6371/vxave/60)
    print('Propagation calculated: '+str(shift)+' min')
    df['Time_UTC'] = df['Time_UTC']+shift
    return df

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
    mfilist = kwargs.get('mfilist',['BGSM'])
    swelist = kwargs.get('swelist',
                              ['V_GSM', 'THERMAL_SPD', 'Np', 'SC_pos_GSM'])
    varlist = kwargs.get('varlist',
               ['Proton_VX_moment', 'Proton_VY_moment', 'Proton_VZ_moment',
                'BX', 'BY', 'BZ', 'Proton_Np_moment', 'Proton_W_moment',
                'xgse'])#note xGSE = xGSM
    status,data =cdas.get_data(dat_key,varlist,start-dt.timedelta(1),end)
    status,mfi =cdas.get_data(mfi_key,mfilist,start-dt.timedelta(1),end)
    swestatus,swe =cdas.get_data(swe_key,swelist,start-dt.timedelta(1),end)
    df_mfi = pd.DataFrame(mfi['BGSM'], columns=['bx','by','bz'],
                          index=mfi['Epoch'])
    df_mfi.resample('60S').asfreq()
    df_mfi = df_mfi.interpolate()
    df_swe = pd.DataFrame(swe['V_GSM'], columns=['vx','vy','vz'],
                          index=swe['Epoch'])
    df = pd.DataFrame(data)
    df = add_swmf_vars(df)
    df = df[(df['Time_UTC']>start) & (df['Time_UTC']<end)]
    save_to_csv(df, 'GSM', outpath)
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


#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    path_to_ori_file = None
    #start = dt.datetime(2019,5,13,12,0)
    #end = dt.datetime(2019,5,15,12,0)
    start = dt.datetime(2022,2,10,0,0)
    end = dt.datetime(2022,2,11,23,0)
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
            #plt.show()
            plt.figure(1).savefig('IMF.png')
            print('Figure saved to IMF.png')
