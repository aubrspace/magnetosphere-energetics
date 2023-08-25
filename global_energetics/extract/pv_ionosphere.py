import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from paraview
from paraview.simple import *
import pv_input_tools

def load_ie(infile,**kwargs):
    """Function loads IE file into a paraview environment
    Inputs
        infile
    Returns
        ie_pipe
    """
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Read input file
    sourcedata = pv_input_tools.read_tecplot(infile,binary=False)
    pipeline = sourcedata
    ###Establish the XYZ position variables
    position = Calculator(registrationName='iePosition', Input=pipeline)
    position.ResultArrayName = 'Position'
    position.Function = '"X [R]"*iHat+"Y [R]"*jHat+"Z [R]"*kHat'
    position.CoordinateResults = 1
    ###Adjust some names for consistency
    pipeline = pv_input_tools.fix_ie_names(position,**kwargs)
