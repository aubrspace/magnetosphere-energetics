import numpy as np
import paraview
from paraview.simple import *
paraview.compatibility.major = 5
paraview.compatibility.minor = 12
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy

def get_diff_volume_integrals(volume:str,np_volume:dict,
                              dt:float) -> dict:
    """ Conditional arrays for the differential bounds (integrand eval @ t0)
        NOTE - if point will be acquired (future - past > 0) this is energy
                ADDED to the system, therfore for sign convention we multiply
                by -1, so it matches with surface flux
    """
    conditions = {}
    if volume=='mp':
        #M  -  [not mp]  <-> [mp]
        #M1 -  [not mp]  <-> [lobes]
        #M5 -  [not mp]  <-> [closed]
        conditions['M'] = -1*(
                np_volume['FUTUREmp']*(1-np_volume['PASTmp']) -
                np_volume['PASTmp']*  (1-np_volume['FUTUREmp'])
                              )/dt
        conditions['M1'] = -1*(
                np_volume['FUTURElobes']*(1-np_volume['PASTmp']) -
                np_volume['PASTlobes']*  (1-np_volume['FUTUREmp'])
                              )/dt
        conditions['M5'] = -1*(
                np_volume['FUTUREclosed']*(1-np_volume['PASTmp']) -
                np_volume['PASTclosed']*  (1-np_volume['FUTUREmp'])
                              )/dt
    if volume=='closed':
        #M2 -  [lobes]   <-> [closed]
        conditions['M2'] = -1*(
                np_volume['FUTUREclosed']*  (1-np_volume['PASTlobes'])-
                np_volume['PASTclosed']*    (1-np_volume['FUTURElobes'])
                              )/dt
    if volume=='lobes':
        #M2 -  [closed]  <-> [lobes]
        pass
    if volume=='sheath':
        #M0 -  [solar wind]  <-> [sheath]
        #M15 -  [mp]         <-> [sheath]
        conditions['M0'] = -1*(
                np_volume['FUTUREsheath']*((1-np_volume['PASTmp'])*
                                           (1-np_volume['PASTsheath'])) -
                np_volume['PASTsheath']*((1-np_volume['FUTUREmp'])*
                                         (1-np_volume['FUTUREsheath']))
                              )/dt
        conditions['M15'] = -1*(
                np_volume['FUTUREsheath']* np_volume['PASTmp'] -
                np_volume['PASTsheath']*   np_volume['FUTUREmp']
                               )/dt
    return conditions

def get_numpy_volume_analysis(source:object,*,
                         volume_list:list,
                      integrands:list=['Utot_J_Re3','uHydro_J_Re3','uB_J_Re3'],
                                skip_keys:list=[],**kwargs:dict) -> dict:
    """Staging function that will take the volume_dict and pass what is needed
        for each calculation one at a time, compiling the results
    Inputs
        source (Filter) - the whole solution including state variables used to
                           calculate partial volume integrals
        kwargs:
            volume_list (list) - which volumes are being integrated
            integrands (list) -  which things are being integrated
            skip_keys (list) - default empty
    Returns
        results_dict {str:np.ndarray}
    """
    integral_translation = {'Utot_J_Re3':('Utot','_J'),# gives ID tag, unit
                          'uHydro_J_Re3':('uHydro','_J'),
                              'uB_J_Re3':('uB','_J')}
    results = {}
    if kwargs.get('verbose',False):
        print('ANALYZING VOLUME(S): ...')
    # Extract the volume state from VTKArray -> np.ndarray
    np_volume = {}
    data = servermanager.Fetch(source)
    data = dsa.WrapDataObject(data)
    for variable in data.PointData.keys():
        np_volume[variable] = vtk_to_numpy(data.PointData[variable])
    for volume in volume_list:
        if kwargs.get('verbose',False):
            print(f'\t{volume}')
        # for volumes the primary (only?) condition is the subvolume itself
        conditions = {'':np_volume[volume]}
        if 'FUTUREUtot_J_Re3' in np_volume.keys():
            conditions.update(get_diff_volume_integrals(volume,np_volume,1800))
        # Calculate each partial integral
        for integrand in integrands:
            if kwargs.get('verbose',False):
                print(f'\t\t{integrand}')
            # pull out the integrand as an array
            integrand_values = np_volume[integrand]
            # adjust post-integration units using a dict
            integral_name, units = integral_translation[integrand]
            #TODO - check the magnitudes or the conversion somewhere ...
            for condition_name,cond in conditions.items():
                if condition_name=='':
                    entry_name = volume+'_'+integral_name+condition_name+'_J'
                else:
                    entry_name = volume+'_'+integral_name+condition_name+'_W'
                results[entry_name] = np.sum(integrand_values*cond*
                                                    np_volume['dvol_R3'])
            if 'FUTUREUtot_J_Re3' in np_volume.keys():
                entry_name = volume+'_'+integral_name+'ddt'+'_W'
                results[entry_name] = np.sum((np_volume['PAST'+integrand]-
                                              np_volume['FUTURE'+integrand])*
                           np_volume[volume]*np_volume['dvol_R3'])/(1800)
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
