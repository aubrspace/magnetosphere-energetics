#/usr/bin/env python
"""accesses data from nasa CDAWeb
"""
import numpy as np
import datetime as dt
from cdasws import CdasWs
cdas = CdasWs()
from geopack import geopack as gp
from tqdm import tqdm
from global_energetics.cdaweb_static import *

def gse_to_gsm(times:np.ndarray,
             vec_gse:np.ndarray) -> np.ndarray:
    """Converts a vector [n x 3] form GSE to GSM coords
    Inputs
        times (as datetimes)
        vec_gse (n x 3) vector
    Returns
        vec_gsm (same shape as input)
    """
    t1970 = dt.datetime(1970,1,1,0)
    vec_gsm = np.zeros_like(vec_gse)
    for it, t in enumerate(tqdm(times)):
        ut = (t-t1970).total_seconds()
        gp.recalc(ut)
        vec_gsm[it,:] = gp.gsmgse(*np.array(vec_gse[it,:]).astype(float),-1)
    return vec_gsm

def get_satellite_positions(sat_name:str,
                               start:dt.datetime,
                                 end:dt.datetime,
                            **kwargs:dict) -> dict:
    """Calls CDAWeb to get the position data from the relevant instrument
    Inputs
        sat_name (str) -  name of the spacecraft or constellation
        start,end (datetime)
        kwargs:
            spacecraft_dict: list of keys identifying sc in the constellation
    """
    print(f'Loading {sat_name} Position Data ...')
    # NOTE Set a time buffer bc for some reason it can miss our target
    buff = kwargs.get('buff',dt.timedelta(hours=12))
    # Set spacecraft IDs and time key
    spacecraft_list = kwargs.get('spacecraft_list',spacecraft_IDs[sat_name])
    #time_key = kwargs.get('time_key',pos_time_key_dict[sat_name])

    # Set spacecraft IDs, time key and unit conversion
    pos_instrument = kwargs.get('pos_instrument',pos_instrument_dict[sat_name])
    pos_variables = kwargs.get('pos_variables',
                               pos_variable_keys_dict[sat_name])
    unit_conversion = kwargs.get('unit_conversion',
                                 unit_conversion_dict[sat_name])

    all_position = {}
    for satID in spacecraft_list:
        if satID =='':
            position = all_position
        else:
            print(f'\t {sat_name}-{satID}')
            all_position[satID] = {}
            position = all_position[satID]
        # Set orbit position instrument key and variable key
        pos_instrument = pos_instrument.replace('*',satID)
        pos_variables = [v.replace('*',satID) for v in pos_variables]
        # Call CDAWeb to get the data
        status,posdata = cdas.get_data(pos_instrument,pos_variables,
                                           start-buff,end+buff)
        # Convert units, if necessary
        posdata[pos_variables[0]] = posdata[pos_variables[0]]*unit_conversion

        # Define the time array, and trim back from our buffer
        time_key = [k for k in posdata.keys()if 'epoch' in k.lower()][0]
        times = np.array([np.datetime64(t) for t in posdata[time_key]])
        interv = (times>start)&(times<end)

        # Rotate coordinate system, if necessary
        if kwargs.get('needs_rotation',needs_rotation[sat_name]):
            xGSE = posdata[pos_variables[0]][interv]
            position['x_gse'] = np.array(xGSE[:,0])
            position['y_gse'] = np.array(xGSE[:,1])
            position['z_gse'] = np.array(xGSE[:,2])
            print('\t\t Rotating vectors GSE->GSM')
            xGSM = gse_to_gsm(times[interv],xGSE)
        else:
            xGSM = posdata[pos_variables[0]][interv]

        # Load the data into a dictionary of numpy arrays
        position['time']  = times[interv]
        position['x_gsm'] = np.array(xGSM[:,0])
        position['y_gsm'] = np.array(xGSM[:,1])
        position['z_gsm'] = np.array(xGSM[:,2])

    return all_position

def get_satellite_bfield(sat_name:str,
                            start:dt.datetime,
                              end:dt.datetime,
                            **kwargs:dict) -> dict:
    """Calls CDAWeb to get the B field data from the relevant instrument
    Inputs
        sat_name (str) -  name of the spacecraft or constellation
        start,end (datetime)
        kwargs:
            spacecraft_dict: list of keys identifying sc in the constellation
    """
    print(f'Loading {sat_name} B Field Data ...')
    # NOTE Set a time buffer bc for some reason it can miss our target
    buff = kwargs.get('buff',dt.timedelta(hours=12))

    # Set spacecraft IDs and time key
    spacecraft_list = kwargs.get('spacecraft_list',spacecraft_IDs[sat_name])
    #time_key = kwargs.get('time_key',bfield_time_key_dict[sat_name])

    all_bfield = {}
    for satID in spacecraft_list:
        if satID =='':
            bfield = all_bfield
        else:
            print(f'\t {sat_name}-{satID}')
            all_bfield[satID] = {}
            bfield = all_bfield[satID]
        # Set magnetic field instrument key and variable key
        bfield_instrument = kwargs.get('bfield_instrument',
                          bfield_instrument_dict[sat_name].replace('*',satID))
        bfield_variables = [kwargs.get('bfield_variables',
                      bfield_variable_keys_dict[sat_name].replace('*',satID))]

        # Call CDAWeb to get the data
        status,bfielddata = cdas.get_data(bfield_instrument,bfield_variables,
                                           start-buff,end+buff)

        # Define the time array, and trim back from our buffer
        # This part is awful :(
        time_key = [k for k in bfielddata.keys()
                       if 'epoch' in k.replace('time_tag','epoch').lower()][0]
        times = np.array([np.datetime64(t) for t in bfielddata[time_key]])
        interv = (times>start)&(times<end)

        # Rotate coordinate system, if necessary
        if kwargs.get('needs_rotation',needs_rotation[sat_name]):
            b_GSE = bfielddata[bfield_variables[0]][interv]
            print('\t\t Rotating vectors GSE->GSM')
            b_GSM = gse_to_gsm(times[interv],b_GSE)
        else:
            b_GSM = bfielddata[bfield_variables[0]][interv]

        # Load the data into a dictionary of numpy arrays
        bfield['time']  = times[interv]
        bfield['bx_gsm'] = np.array(b_GSM[:,0])
        bfield['by_gsm'] = np.array(b_GSM[:,1])
        bfield['bz_gsm'] = np.array(b_GSM[:,2])

    return all_bfield

def get_satellite_plasma(sat_name:str,
                            start:dt.datetime,
                              end:dt.datetime,
                            **kwargs:dict) -> dict:
    """Calls CDAWeb to get the plasma data from the relevant instrument
    Inputs
        sat_name (str) -  name of the spacecraft or constellation
        start,end (datetime)
        kwargs:
            spacecraft_dict: list of keys identifying sc in the constellation
    """
    print(f'Loading {sat_name} Plasma Data ...')
    # NOTE Set a time buffer bc for some reason it can miss our target
    buff = kwargs.get('buff',dt.timedelta(hours=12))

    # Set spacecraft IDs and time key
    spacecraft_list = kwargs.get('spacecraft_list',spacecraft_IDs[sat_name])
    #time_key = kwargs.get('time_key',plasma_time_key_dict[sat_name])

    all_plasma = {}
    for satID in spacecraft_list:
        if satID =='':
            plasma = all_plasma
        else:
            print(f'\t {sat_name}-{satID}')
            all_plasma[satID] = {}
            plasma = all_plasma[satID]
        # Set magnetic field instrument key and variable key
        # NOTE for plasma data there may be multiple instruments to combine
        plasma_instruments = kwargs.get('plasma_instrument',
                                         plasma_instrument_dict[sat_name])
        plasma_instruments = [i.replace('*',satID)for i in plasma_instruments]
        for instrument in plasma_instruments:
            # Get the variables for this particular instrument
            variable_dict = kwargs.get('plasma_variables',
                                        plasma_variable_keys_dict[sat_name])
            if satID != '':
                variables = variable_dict[instrument.replace(satID,'*')]
            else:
                variables = variable_dict[instrument]

            # Call CDAWeb to get the data
            status,plasmadata = cdas.get_data(instrument,variables,
                                              start-buff,end+buff)

            if plasmadata: #successfully found the data
                print(f'\t\t {instrument} - (SUCCESS)')
                # Define the time array, and trim back from our buffer
                interv = (times>start)&(times<end)
                for key in plasmadata:
                    if 'epoch' in key.lower():
                        # If its time, then convert dt -> np.datetime64
                        times = np.array([np.datetime64(t)
                                          for t in plasmadata[key]])
                        plasma[f"{instrument}_time"] = times[interv]
                    elif key in variables:
                        # If its timeseries data, trim the buffer
                        plasma[f"{instrument}_{key}"] = np.array(
                                                      plasmadata[key][interv])
                    else:
                        # Anything else just load as is
                        plasma[f"{instrument}_{key}"] = np.array(
                                                              plasmadata[key])
            else:
                print(f'\t\t {instrument} - (FAIL)')

                '''
        # Rotate coordinate system, if necessary
        if kwargs.get('needs_rotation',needs_rotation[sat_name]):
            print('\t\t Rotating vectors GSE->GSM')
            plasma[variables[0]] = gse_to_gsm(times[interv],
                                      plasmadata[plasma_variables[0]][interv])
                # Load the data into a dictionary of numpy arrays
                plasma['time']  = times[interv]
                plasma['bx_gsm'] = np.array(
                                 plasmadata[plasma_variables[0]][:,0][interv])
                plasma['by_gsm'] = np.array(
                                 plasmadata[plasma_variables[0]][:,1][interv])
                plasma['bz_gsm'] = np.array(
                                 plasmadata[plasma_variables[0]][:,2][interv])
                '''


    return all_plasma
