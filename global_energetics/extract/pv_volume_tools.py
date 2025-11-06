import numpy as np
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy

def get_numpy_volume_analysis(volumes_dict:dict,*,
                      integrands:list=['Utot_J_Re3','uHydro_J_Re3','uB_J_Re3'],
                                   skip_keys:list=[]) -> dict:
    """Staging function that will take the volume_dict and pass what is needed
        for each calculation one at a time, compiling the results
    Inputs
        volume_dict {str:object}
        kwargs:
            integrands (list) -  which things are being integrated
            skip_keys (list) - default empty
    Returns
        results_dict {str:np.ndarray}
    """
    integral_translation = {'Utot_J_Re3':('Utot','_PJ'),# gives ID tag, unit
                          'uHydro_J_Re3':('uHydro','_PJ'),
                              'uB_J_Re3':('uB','_PJ')}
    results = {}
    print('ANALYZING VOLUME(S): ...')
    for vol,source in volumes_dict.items():
        print(f'\t{vol}')
        # Extract the volume state from VTKArray -> np.ndarray
        np_volume = {}
        data = servermanager.Fetch(source)
        data = dsa.WrapDataObject(data)
        for variable in data.PointData.keys():
            np_volume[variable] = vtk_to_numpy(data.PointData[variable])
        # Get conditional volume representing partial integral bounds
        #conditions = map_volume_to_interfaces(vol,np_volume)
        # Calculate each partial integral
        for integrand in integrands:
            print(f'\t\t{integrand}')
            # pull out the integrand as an array
            integrand_values = np_volume[integrand]
            # adjust post-integration units using a dict
            integral_name, units = integral_translation[integrand]
            # Calculate the total (no partial bounds)
            entry_name = vol+'_'+integral_name+units
            #TODO - check the magnitudes or the conversion somewhere ...
            results[entry_name] = np.sum(integrand_values
                                                  *np_volume['dvol_R3'])/1e12
            #for condition_name,cond in conditions.items():
            #TODO put in the dVolume calcs
            #    pass
    return results

def extract_volume(source:object,
                 variable:str,
              volume_name:str,**kwargs:dict) -> object:
    # Create threshold filter
    volume = Threshold(registrationName=volume_name+'_init', Input=source)
    volume.Scalars = ['POINTS', variable]
    volume.ThresholdMethod = kwargs.get('threshold_method',
                                        'Above Upper Threshold')
    volume.UpperThreshold = kwargs.get('upper_threshold',0.99)
    if ('Between' in volume.ThresholdMethod or
                                           'Lower' in volume.ThresholdMethod):
        volume.LowerThreshold = kwargs.get('lower_threshold',0)

    if kwargs.get('clean_volume',True):
        volume_clean = Connectivity(registrationName=volume_name,
                                    Input=volume)
        return volume_clean
    else:
        return volume

def add_fluxVolume(field,**kwargs):
    """Function integrates volume flux values
    Inputs
        field- source to find the flux volumes
        kwargs:
            station_file
            localtime
            tilt
    Returns
        None
    """
    #Integrate total
    total_int=IntegrateVariables(registrationName='totalInt',Input=field)

    fluxVolume_hits=ProgrammableFilter(registrationName='fluxVolume_hits',
                                       Input=field)
    fluxVolume_hits.Script=update_fluxVolume(**kwargs)
    # create a new 'Threshold'
    fluxVolume = Threshold(registrationName='fluxVolume',
                           Input=fluxVolume_hits)
    fluxVolume.Scalars = ['POINTS', 'projectedVol']
    fluxVolume.UpperThreshold = 0.95
    fluxVolume.ThresholdMethod = 'Above Upper Threshold'
    #Set a view
    view = GetActiveViewOrCreate('RenderView')
    #Adjust view settings
    volumeDisplay = Show(fluxVolume, view, 'GeometryRepresentation')
    volumeDisplay.AmbientColor = [0.33, 1.0, 0.0]
    volumeDisplay.DiffuseColor = [0.33, 1.0, 0.0]
    volumeDisplay.Opacity = 0.20
    ColorBy(volumeDisplay, None)
    #Integrate projected flux
    flux_int=IntegrateVariables(registrationName='fluxInt',
                                Input=fluxVolume)
    #Grab what you like
    vals = update_fluxResults(flux_int,total_int)
    return fluxVolume, vals

def update_fluxResults(flux_int,total_int):
    """Function extracts specific data from the flux_int filter results
    Inputs
        flux_int (pipeline) - the IntegrateVariables filter
        total_int (pipeline)
    Returns
        vals (dict{str:float}) - results that are of interest
    """
    #TODO make this more general so it can be better used
    #Grab what you like
    vals = {}
    vals['flux_volume'] = flux_int.CellData['Volume'].GetRange()[0]
    vals['flux_Umag'] = flux_int.PointData['uB_J_Re3'].GetRange()[0]
    vals['flux_Udb'] = flux_int.PointData['u_db_J_Re3'].GetRange()[0]
    vals['total_volume'] = total_int.CellData['Volume'].GetRange()[0]
    vals['total_Umag'] = total_int.PointData['uB_J_Re3'].GetRange()[0]
    vals['total_Udb'] = total_int.PointData['u_db_J_Re3'].GetRange()[0]
    return vals

#TODO: put in a numpy volume integrated function (maybe merge with tecplot
#                                                 version??)
