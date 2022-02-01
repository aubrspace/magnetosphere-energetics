#/usr/bin/env python
"""accesses WIND data from nasa CDA and creates IMF_new.dat for SWMF input
"""
import pandas as pd
import os
import sys
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from cdasws import CdasWs
cdas = CdasWs()
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
            df['yr'] = yr
            df['mn'] = mm
            df['dy'] = dy
            df['hr'] = hr
            df['min'] = minu
            df['sec'] = sec
            df['msec'] = msec
            df = df.rename(columns={key[1]:'Time_UTC'})
    #Calculate average shift
    #d = df['xgse']-15
    #dave = d.mean()
    dave = 238-15
    vxave = df['vx'].mean()
    shift = dt.timedelta(minutes=-1*dave*6371/vxave/60)
    print(shift)
    #shift = dt.timedelta(minutes=15)
    print(df['Time_UTC'])
    #quarter shift matches the best with previous run!
    df['Time_UTC'] = df['Time_UTC']+shift
    print(df['Time_UTC'])
    return df

def save_to_csv(df, coordinates, outpath):
    """Function saves data from df to csv in correct SWMF input format
    Inputs
        df
        outpath
    """
    data = df[['yr','mn','dy','hr','min', 'sec', 'msec',
               'bx','by','bz','vx','vy','vz', 'dens', 'temp']]
    data.to_csv('IMF_new.dat', sep=' ', index=False)
    with open('IMF_new.dat', 'r') as base:
        header = base.readline()
        basefile = base.read()
    with open('IMF_new.dat', 'w') as final:
        final.write(header+'#COOR\n'+coordinates+'\n\n#START\n'+
                    basefile)

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


#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    path_to_ori_file = './IMF.dat'
    start = dt.datetime(2013,9,17,0,0)
    end = dt.datetime(2013,9,20,0,0)
    outpath = './'
    #Use "cdas.get_variables('WI_H1_SWE')" to see options
    mfilist = ['BGSE']
    swelist = ['V_GSE', 'THERMAL_SPD', 'Np', 'SC_pos_gse']
    varlist = ['xgse',
               'Proton_VX_moment', 'Proton_VY_moment', 'Proton_VZ_moment',
               'BX', 'BY', 'BZ', 'Proton_Np_moment', 'Proton_W_moment']
    #######################################################################

    ori_df = read_SWMF_IMF(path_to_ori_file)
    status,data =cdas.get_data('WI_H1_SWE',varlist,start-dt.timedelta(1),
                                           end)
    status,mfi =cdas.get_data('WI_H0_MFI',mfilist,start-dt.timedelta(1),
                                           end)
    swestatus,swe =cdas.get_data('WI_K0_SWE',swelist,start-dt.timedelta(1),
                                           end)
    df_mfi = pd.DataFrame(mfi['BGSE'], columns=['bx','by','bz'],
                          index=mfi['Epoch'])
    df_mfi.resample('60S').asfreq()
    df_mfi = df_mfi.interpolate()
    df_swe = pd.DataFrame(swe['V_GSE'], columns=['vx','vy','vz'],
                          index=swe['Epoch'])
    df2 = read_MFI_SWE_WIND('scratch_WIND.dat')
    df = pd.DataFrame(data)
    df = add_swmf_vars(df2)
    df = df[(df['Time_UTC']>start) & (df['Time_UTC']<end)]
    print(ori_df)
    print(df)
    save_to_csv(df, 'GSE', outpath)
    plot_comparison(ori_df, df, outpath)
