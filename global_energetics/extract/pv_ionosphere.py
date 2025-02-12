import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from paraview
from paraview.simple import *
import pv_input_tools
from pv_tools import mag_to_gsm, rotate2GSM

def load_ie(infile,**kwargs):
    """Function loads IE file into a paraview environment
    Inputs
        infile
        kwargs:
            coord (str) - coordinate system to represent data
            tilt  (float)- dipole tilt value
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
        '''
        #NOTE  TODO need to follow up, above has the coord filter included
        # Set XYZ GSM to be used as the new coordinates
        pipeline = Calculator(registrationName='XYZgsm',Input=pipeline)
        pipeline.Function = "x_gsm*iHat+y_gsm*jHat+z_gsm*kHat"
        pipeline.ResultArrayName = 'XYZgsm'
        pipeline.CoordinateResults = 1
        '''

    return pipeline

def integrate_Jr(ionosphere,**kwargs):
    """Function integrates upward/downward/net FAC for each hemisphere
    Inputs
        ionosphere (filter) - end of the pipeline that contains IE data,
                              assuming IE data is POINT data
        kwargs:
            None
    Returns
        FAC (dict) - {'up_N_mA':(float),'down_N_mA':(float), ... etc.}
    """
    FAC = {}
    # Point data to cell data
    iono_cc = PointDatatoCellData(registrationName='Point2Cell',
                                  Input=ionosphere)
    iono_cc.ProcessAllArrays = 1
    # Find the cell area
    area    = CellSize(registrationName='Area',Input=iono_cc)
    # Get combinations of Jr weighted by Area
    jr_weights = ProgrammableFilter(registrationName='jr_weighted',
                                    Input=area)
    jr_weights.Script = update_Jr_pieces()
    # Take integral as sum(Area*J_r_cc)
    integral = IntegrateVariables(registrationName='integral',
                                  Input=jr_weights)
    integral.DivideCellDataByVolume = 0
    for key in integral.CellData.keys():
        FAC[key] = integral.CellData[key].GetRange()[0]
    return FAC

def update_Jr_pieces():
    return """
    import numpy as np
    data = inputs[0]
    # Get base variables
    Jr   = data.CellData['J_R_uA_m^2']
    Z    = data.CellData['z']
    Area = data.CellData['Area']

    # Calc N/S up/down/net combinations
    net_north  = Jr*Area*(np.sign(Z)+1)/2*6371**2
    net_south  = Jr*Area*-1*(np.sign(Z)-1)/2*6371**2
    up_north   = net_north*(np.sign(net_north)+1)/2
    up_south   = net_south*(np.sign(net_south)+1)/2
    down_north = net_north*(np.sign(net_north)-1)/2
    down_south = net_south*(np.sign(net_south)-1)/2

    # Encode into output data
    output.CellData.append(net_north,'net_north_A')
    output.CellData.append(net_south,'net_south_A')
    output.CellData.append(up_north,'up_north_A')
    output.CellData.append(up_south,'up_south_A')
    output.CellData.append(down_north,'down_north_A')
    output.CellData.append(down_south,'down_south_A')
    """

#TODO
#   create a function that calculates Jr positions
#   . Find the effective zenith that Jr is connecting into
#   . Find the overall conductance that Jr is connecting into
#     - could also look at overall (pederson) conductance
#   . Find the auroral oval, using the defintion from the paper
#     - find the relative size
#     - find the relative position

