""" load .cdf files and then re-export as .npz
"""
import numpy as np
import datetime as dt
from glob import glob
from tqdm import tqdm
from spacepy import datamodel

def fill_data_gaps(data:dict,**kwargs:dict) -> None:#NOTE will modify data
    flux_key   = kwargs.get('flux_key','FESA')
    energy_key = kwargs.get('energy_key','FESA_Energy')
    time_key   = kwargs.get('time_key','Epoch')
    nan_value  = kwargs.get('nan_value',-9.9999998e+30)

    flux = data[flux_key]
    energy = data[energy_key]

    print(f"Filling data w linear interp ...")
    # Mark the nan values
    flux[flux == nan_value]= np.nan
    # Set a time array, in seconds for interpolation
    t = (data[time_key]-data[time_key][0])/1e6
    # Will be removing all columns which are completely nan's
    poplist = []
    for itime in tqdm(range(0,len(t))):
        # Per timestep, interpolating across energy channels
        flux_now = flux[itime,:]
        x_new  = energy
        x_old  = energy[~np.isnan(flux_now)]
        if len(x_old)==0:
            poplist.append(itime)
        else:
            fx_old = flux_now[~np.isnan(flux_now)]
            flux[itime,:] = np.interp(x_new,x_old,fx_old)
    keeplist = list(range(0,len(t)))
    for ibadtime in poplist[::-1]:
        keeplist.pop(ibadtime)

    for key in data:
        if len(data[key]) == len(t):
            data[key]   = data[key][keeplist]
    # NOTE this will change the 'data' dict passed


def time_from_filename(infile:str) -> None:
    if 'L2_' in infile:
        datestring = infile.split('/')[-1].split('_v')[0].split('L2_')[-1]
        date = dt.datetime.strptime(datestring,'%Y%m%d')

    elif 'L3_' in infile:
        datestring = infile.split('/')[-1].split('_v')[0].split('L3_')[-1]
        date = dt.datetime.strptime(datestring,'%Y%m%d')

    return (date - dt.datetime(1970,1,1,0)).total_seconds()

def main() -> None:
    # Get the list of .cdf files
    cdf_filelist = sorted(glob(f"{INPATH}/*.cdf"),key=time_from_filename)

    non_timeseries = ['FESA_Energy',
                      'FESA_Energy_DELTA_plus',
                      'FESA_Energy_DELTA_minus',
                      'FESA_Instrument']

    for ifile,cdffile in enumerate(tqdm(cdf_filelist)):
        # Use spacepy to read the cdf file
        dm = datamodel.fromCDF(cdffile)
        # Convert the datamodel type to a dict of numpy arrays
        indata = {}
        for key in dm:
            indata[key] = np.array(dm[key])
        # Fix the time column to numpy type
        indata['Epoch']= np.array([np.datetime64(t) for t in indata['Epoch']])
        if ifile == 0:
            # Set the output data
            data = indata
        else:
            # Append to the output data
            time_length = len(indata['Epoch'])
            for key in [k for k in indata if k not in non_timeseries]:
                data[key] = np.concatenate([data[key],indata[key]])

    from IPython import embed; embed()
    # Clean data
    #fill_data_gaps(data) #NOTE this modifies data

    # Save as an npz file
    #np.savez_compressed(f"{OUTPATH}/{OUTNAME}",**data,allow_pickle=False)
    #print('\033[92m Created\033[00m ',f"{OUTPATH}/{OUTNAME.split('/')[-1]}")


if __name__=='__main__':

    global INPATH,OUTPATH,OUTNAME

    INPATH  = '/Users/ambrenne/Code/rb_project_plotting/combined_rbsp'
    OUTPATH = '/Users/ambrenne/Code/rb_project_plotting/combined_rbsp'
    OUTNAME = '2018_rbspA_combined.npz'

    main()
