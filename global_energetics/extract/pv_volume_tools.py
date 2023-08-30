from paraview.simple import *

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
