import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from paraview
from paraview.simple import *
import pv_input_tools
from pv_tools import mag_to_gsm

def load_ie(infile,**kwargs):
    """Function loads IE file into a paraview environment
    Inputs
        infile
    Returns
        ie_pipe
    """
    r_iono = kwargs.get('r_iono',1+300/6371)
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Read input file
    sourcedata = pv_input_tools.read_tecplot(infile,binary=False)
    # Merge blocks
    merge = MergeBlocks(registrationName='MergeBlocks',
                               Input=sourcedata)
    merge.MergePoints=0
    pipeline = merge
    ###Establish the XYZ position variables
    position = Calculator(registrationName='iePosition', Input=pipeline)
    position.ResultArrayName = 'Position'
    position.Function = f'{r_iono}*("X [R]"*iHat+"Y [R]"*jHat+"Z [R]"*kHat)'
    position.CoordinateResults = 1
    ###Adjust some names for consistency
    pipeline = pv_input_tools.fix_ie_names(position,**kwargs)
    if kwargs.get('coord','MAG')=='GSM':
        ### Rotate into GSM coordinates
        #tevent = kwargs.get('tevent')
        #pipeline = mag_to_gsm(pipeline,tevent)
        pipeline = rotate2GSM(pipeline,kwargs.get('tilt',0))
        #NOTE  TODO need to follow up, above has the coord filter included
        # Set XYZ GSM to be used as the new coordinates
        pipeline = Calculator(registrationName='XYZgsm',Input=pipeline)
        pipeline.Function = "x_gsm*iHat+y_gsm*jHat+z_gsm*kHat"
        pipeline.ResultArrayName = 'XYZgsm'
        pipeline.CoordinateResults = 1

    return pipeline
