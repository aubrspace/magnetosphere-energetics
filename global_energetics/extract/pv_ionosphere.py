import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
from numpy import sqrt,arctan2,sin,cos,deg2rad
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
        FAC[key] = integral.CellData[key].GetRange()[0]*6371**2*1e-6
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
    net_north  = Jr*(np.sign(Z)+1)/2
    net_south  = Jr*-1*(np.sign(Z)-1)/2
    up_north   = net_north*(np.sign(net_north)+1)/2
    up_south   = net_south*(np.sign(net_south)+1)/2
    down_north = net_north*(np.sign(net_north)-1)/2
    down_south = net_south*(np.sign(net_south)-1)/2

    # Encode into output data
    output.CellData.append(net_north,'net_north_MA')
    output.CellData.append(net_south,'net_south_MA')
    output.CellData.append(up_north,'up_north_MA')
    output.CellData.append(up_south,'up_south_MA')
    output.CellData.append(down_north,'down_north_MA')
    output.CellData.append(down_south,'down_south_MA')
    """

def vincenty_formula_dist(theta1,psi1,theta2,psi2):
    # Copied from https://en.wikipedia.org/wiki/Great-circle_distance
    #print(f'Inputs: th1:{theta1}, p1:{psi1}, th2:{theta2}, p2:{psi2}')
    th1 = np.deg2rad(90-theta1)
    th2 = np.deg2rad(90-theta2)
    p1  = np.deg2rad(psi1)
    p2  = np.deg2rad(psi2)

    theta_m = 0.5*(th1+th2)
    dtheta  = th1-th2
    dpsi    = p1-p2

    dsigma = arctan2(sqrt((cos(th2)*sin(dpsi))**2+
                          (cos(th1)*sin(th2)-sin(th1)*cos(th2)*cos(dpsi))**2),
                    sin(th1)*sin(th2)+cos(th1)*cos(th2)*cos(dpsi))
    #print('dsigma: ',dsigma)
    return dsigma

def id_R1R2_currents(ionosphere,**kwargs):
    """
    """
    Jdata = servermanager.Fetch(ionosphere)
    Jdata = dsa.WrapDataObject(Jdata)
    J = Jdata.PointData['J_R_uA_m^2']
    if not FindSource('Jr_plus'):# making new things
        # Create isovolume filter with the positive and negative currents
        Jr_plus = IsoVolume(registrationName='Jr_plus',Input=ionosphere)
        Jr_plus.InputScalars=['POINTS','J_R_uA_m^2']
        Jr_plus.ThresholdRange = [0.15*J.max(),2*J.max()]

        Jr_minus = IsoVolume(registrationName='Jr_minus',Input=ionosphere)
        Jr_minus.InputScalars=['POINTS','J_R_uA_m^2']
        Jr_minus.ThresholdRange = [2*J.min(),0.15*J.min()]
        # Create connectivity filter which assigns IDs to each piece
        ID_plus = Connectivity(registrationName='ID_plus',Input=Jr_plus)
        ID_minus = Connectivity(registrationName='ID_minus',Input=Jr_minus)
        # Point data to cell data
        CC_plus = PointDatatoCellData(registrationName='CC_plus',Input=ID_plus)
        CC_plus.ProcessAllArrays = 1

        CC_minus = PointDatatoCellData(registrationName='CC_minus',
                                       Input=ID_minus)
        CC_minus.ProcessAllArrays = 1
        # Find the cell area
        area_plus = CellSize(registrationName='Area_plus',Input=CC_plus)
        area_minus = CellSize(registrationName='Area_minus',Input=CC_minus)
    else: # updating existing things
        Jr_plus = FindSource('Jr_plus')
        Jr_minus = FindSource('Jr_minus')
        Jr_plus.ThresholdRange = [0.15*J.max(),2*J.max()]
        Jr_minus.ThresholdRange = [2*J.min(),0.15*J.min()]

        area_plus = FindSource('Area_plus')
        area_minus = FindSource('Area_minus')
    # Map region IDs to an R1/R2 designation
    regions = calc_average_lats(area_plus,area_minus)#NOTE python-side
    if not FindSource('Regions_plus'):
    # Call a prog filter to make a variable
        Regions_plus = ProgrammableFilter(registrationName='Regions_plus',
                                          Input=area_plus)
        Regions_minus = ProgrammableFilter(registrationName='Regions_minus',
                                          Input=area_minus)
    else:
        Regions_plus = FindSource('Regions_plus')
        Regions_minus = FindSource('Regions_minus')
    # Either way, need to update it
    Regions_plus.Script = update_R1R2(regions,0)
    Regions_minus.Script = update_R1R2(regions,100)
    FAC = integrate_regional_currents(area_plus,area_minus,regions)
    # Get distance between + - potentials
    th = Jdata.PointData['Theta_deg']
    psi = Jdata.PointData['Psi_deg']
    V_N = Jdata.PointData['PHI_kV'][th<90]
    V_S = Jdata.PointData['PHI_kV'][th>90]

    th_Vmax_N  = th[th<90][ V_N==V_N.max()][0]
    psi_Vmax_N = psi[th<90][V_N==V_N.max()][0]
    th_Vmin_N  = th[th<90][ V_N==V_N.min()][0]
    psi_Vmin_N = psi[th<90][V_N==V_N.min()][0]
    FAC['d_North']  = vincenty_formula_dist(th_Vmin_N,psi_Vmin_N,
                                            th_Vmax_N,psi_Vmax_N)

    th_Vmax_S  = th[th>90][ V_S==V_S.max()][0]
    psi_Vmax_S = psi[th>90][V_S==V_S.max()][0]
    th_Vmin_S  = th[th>90][ V_S==V_S.min()][0]
    psi_Vmin_S = psi[th>90][V_S==V_S.min()][0]
    FAC['d_South']  = vincenty_formula_dist(th_Vmin_S,psi_Vmin_S,
                                            th_Vmax_S,psi_Vmax_S)
    return FAC

def calc_average_lats(plus,minus,**kwargs):
    """ Copies info from connected regions and returns centroid latitues of
        each piece in a dictionary
    Inputs
        plus, minus
        kwargs:
            offset (100)
    Returns
        regions {id:Region}
    """
    distances = {}
    Ys = {}
    #As = {}
    Polarity = {}
    for i,source in enumerate([plus,minus]):
        data = servermanager.Fetch(source)
        data = dsa.WrapDataObject(data)
        Y  = data.CellData['y']
        Th = data.CellData['Theta_deg']
        Area = data.CellData['Area']
        ID_data = data.CellData['RegionId']+100*i
        IDs = np.unique(ID_data)
        for ID in IDs:
            Polarity[ID] = i*-2+1
            if len(Area[ID_data==ID])>0:
                distances[ID] = np.sum(Th[ID_data==ID]*
                                  Area[ID_data==ID])/np.sum(Area[ID_data==ID])
                Ys[ID] = np.sum(Y[ID_data==ID]*
                                  Area[ID_data==ID])/np.sum(Area[ID_data==ID])
                #As[ID] = np.sum(Area[ID_data==ID])
            else:
                distances[ID] = 0
                Ys[ID] = -1
                print('WARNING: bad case found!')
    regions = {}
    #Athresh = np.array([As[i] for i in As.keys()]).max()/4
    #north_dusk_lats = np.array([distances[i] for i in distances.keys()
    #                        if distances[i]<90 and Ys[i]>0] and As[i]>Athresh)
    #north_dawn_lats = np.array([distances[i] for i in distances.keys()
    #                        if distances[i]<90 and Ys[i]<0] and As[i]>Athresh)
    #south_dusk_lats = np.array([distances[i] for i in distances.keys()
    #                        if distances[i]>90 and Ys[i]>0] and As[i]>Athresh)
    #south_dawn_lats = np.array([distances[i] for i in distances.keys()
    #                        if distances[i]>90 and Ys[i]<0] and As[i]>Athresh)

    #north_dusk_Polarity = [Polarity[i] for i in Polarity.keys()
    #                                if distances[i]==north_dusk_lats.max()][0]
    #south_lats = np.array([distances[i] for i in distances.keys()
    #                      if distances[i]>90])
    #south_theta_limit = (south_lats.min()+south_lats.max())/2
    for ID in distances.keys():
        if distances[ID]<90 and Ys[ID]>0:
            if Polarity[ID]==1:
                regions[ID] = 1
            else:
                regions[ID] = 2
        elif distances[ID]<90 and Ys[ID]<0:
            if Polarity[ID]==1:
                regions[ID] = 2
            else:
                regions[ID] = 1
        elif distances[ID]>90 and Ys[ID]>0:
            if Polarity[ID]==1:
                regions[ID] = 1
            else:
                regions[ID] = 2
        elif distances[ID]>90 and Ys[ID]<0:
            if Polarity[ID]==1:
                regions[ID] = 2
            else:
                regions[ID] = 1
        '''
        if ((distances[ID]<north_theta_limit) or
            (distances[ID]>south_theta_limit)):
            regions[ID] = 1 #region 1 closer to poles (0 and 180deg)
        else:
            regions[ID] = 2 #region 2 closter to equator (90)
        '''
    return regions

def update_R1R2(regions,offset):
    return f"""
    import numpy as np
    ID = inputs[0].CellData['RegionId']+{offset}
    regions = {regions}
    Region = np.array([regions[id] for id in ID])
    output.CellData.append(Region,'Region')
    """

def integrate_regional_currents(plus,minus,region_dict):
    FAC = {}
    for i,source in enumerate([plus,minus]):
        # Pull down data from the server
        data = servermanager.Fetch(source)
        data = dsa.WrapDataObject(data)
        Jr = data.CellData['J_R_uA_m^2']
        Z = data.CellData['z']
        Area = data.CellData['Area']
        ID_data = data.CellData['RegionId']+100*i
        Region = np.array([region_dict[id] for id in ID_data])
        # integrate
        if i==0:
            FAC['DOWN_R1_N'] = np.sum(Jr[(Z>0) & (Region==1)]*
                                    Area[(Z>0) & (Region==1)])*6371**2*1e-6
            FAC['DOWN_R2_N'] = np.sum(Jr[(Z>0) & (Region==2)]*
                                    Area[(Z>0) & (Region==2)])*6371**2*1e-6

            FAC['UP_R1_S']   = np.sum(Jr[(Z<0) & (Region==1)]*
                                    Area[(Z<0) & (Region==1)])*6371**2*1e-6
            FAC['UP_R2_S']   = np.sum(Jr[(Z<0) & (Region==2)]*
                                    Area[(Z<0) & (Region==2)])*6371**2*1e-6
        elif i==1:
            FAC['UP_R1_N']   = np.sum(Jr[(Z>0) & (Region==1)]*
                                    Area[(Z>0) & (Region==1)])*6371**2*1e-6
            FAC['UP_R2_N']   = np.sum(Jr[(Z>0) & (Region==2)]*
                                    Area[(Z>0) & (Region==2)])*6371**2*1e-6

            FAC['DOWN_R1_S'] = np.sum(Jr[(Z<0) & (Region==1)]*
                                    Area[(Z<0) & (Region==1)])*6371**2*1e-6
            FAC['DOWN_R2_S'] = np.sum(Jr[(Z<0) & (Region==2)]*
                                    Area[(Z<0) & (Region==2)])*6371**2*1e-6
    return FAC

#TODO
#   create a function that calculates Jr positions
#   . Find the effective zenith that Jr is connecting into
#   . Find the overall conductance that Jr is connecting into
#     - could also look at overall (pederson) conductance
#   . Find the auroral oval, using the defintion from the paper
#     - find the relative size
#     - find the relative position

