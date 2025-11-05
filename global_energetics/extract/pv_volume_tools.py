from paraview.simple import *

def get_numpy_volume_analysis():
    #TODO
    # Staging function that will take the volume_dict and pass what is needed
    #   for each calculation one at a time, compiling the results
    # Inputs
    #   volume_dict, *, skip_keys=[]
    # Returns
    #   results_dict
    #   
    #   results_dict = {}
    #   for vol,source in volume_dict.items():
    #       TODO- figure out the level of specificity here
    #       could do a number of very specific numpy domain functions, one
    #       for each instance of the volume object?
    #           or
    #       Could have a generic function with some key word arguments
    #        controling what is calculated and what is not ...
    #       
    #       results_dict[vol] = somecall(vol,source)
    #   return results_dict
    pass

def extract_volume(source:object,
                 variable:str,
              volume_name:str,**kwargs:dict) -> object:
    #TODO flesh this out
    #   call threshold filter
    #       set variable
    #       set threshold type (gt,lt,inbetween)
    #       set threshold level (1)
    #       set name (given)
    #   if not skipped:
    #       call connectivity filter
    #   update pipeline
    #   return pipeline endpoint
    pass

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
