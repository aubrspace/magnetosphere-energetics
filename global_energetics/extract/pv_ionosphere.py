import paraview
#### import the simple module from paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
from numpy import sqrt,arctan2,sin,cos,deg2rad,pi
from global_energetics.extract import pv_input_tools
from global_energetics.extract.pv_tools import mag_to_gsm, rotate2GSM

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
    names = pv_input_tools.fix_ie_names(position,**kwargs)
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
    ###Create some generally nice to have contours
    lats = Contour(registrationName='Lats',Input=names)
    lats.ContourBy = ['POINTS','Theta_deg']
    lats.Isosurfaces = [10,20,30,40,50,   130,140,150,160,170]

    lons = Contour(registrationName='Lons',Input=names)
    lons.ContourBy = ['POINTS','Psi_deg']
    lons.Isosurfaces = [0.1,90,180,270]

    ocflb = Contour(registrationName='Ocflb',Input=names)
    ocflb.ContourBy = ['POINTS','RT_1_B_1_T']
    ocflb.Isosurfaces = [-1e5]

    pipeline = ocflb

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
    Vs = {}
    Polarity = {}
    regions = {}
    for i,source in enumerate([plus,minus]):
        data = servermanager.Fetch(source)
        data = dsa.WrapDataObject(data)
        Jr = data.CellData['J_R_uA_m^2']
        Y  = data.CellData['y']
        V  = data.CellData['PHI_kV']
        Th = data.CellData['Theta_deg']
        Area = data.CellData['Area']
        ID_data = data.CellData['RegionId']+100*i
        IDs = np.unique(ID_data)
        # Get the lats of the max/min in each hemi
        Jr_north = Jr[Th<90]
        ID_north = ID_data[Th<90]
        Jr_south = Jr[Th>90]
        ID_south = ID_data[Th>90]
        if i==0:
            dusk_north_maxID = ID_north[Jr_north==Jr_north.max()][0]
            dusk_south_maxID = ID_south[Jr_south==Jr_south.max()][0]
        else:
            dawn_north_minID = ID_north[Jr_north==Jr_north.min()][0]
            dawn_south_minID = ID_south[Jr_south==Jr_south.min()][0]
        for ID in IDs:
            Polarity[ID] = i*-2+1
            if len(Area[ID_data==ID])>0:
                distances[ID] = np.sum(Th[ID_data==ID]*
                                  Area[ID_data==ID])/np.sum(Area[ID_data==ID])
                Ys[ID] = np.sum(Y[ID_data==ID]*
                                  Area[ID_data==ID])/np.sum(Area[ID_data==ID])
                Vs[ID] = np.sum(V[ID_data==ID]*
                                  Area[ID_data==ID])/np.sum(Area[ID_data==ID])
                regions[ID] = 0
            else:
                distances[ID] = 0
                Ys[ID] = -1
                print('WARNING: bad case found!')
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
    # Get the R1 regions first, since the 0 potential (throat?) works well
    for ID in distances.keys():
        if distances[ID]<90 and (
           distances[ID]>distances[dusk_north_maxID] and Ys[ID]>0) or (
           distances[ID]<=distances[dusk_north_maxID] and Vs[ID]<0):
            # North, dusk: +Jr is up, R1 is up, R2 is down
            if Polarity[ID]==1:
                regions[ID] = 1
            else:
                regions[ID] = 2
        if distances[ID]<90 and (
           distances[ID]>distances[dawn_north_minID] and Ys[ID]<0) or (
           distances[ID]<=distances[dawn_north_minID] and Vs[ID]>0):
            # North, dawn: +Jr is up, R1 is down, R2 is up
            if Polarity[ID]==-1:
                regions[ID] = 1
            else:
                regions[ID] = 2
        if distances[ID]>90 and (
           distances[ID]<distances[dusk_south_maxID] and Ys[ID]>0) or (
           distances[ID]>=distances[dusk_south_maxID] and Vs[ID]<0):
            # South, dusk: +Jr is up, R1 is up, R2 is down
            if Polarity[ID]==1:
                regions[ID] = 1
            else:
                regions[ID] = 2
        if distances[ID]>90 and (
           distances[ID]<distances[dawn_south_minID] and Ys[ID]<0) or (
           distances[ID]>=distances[dawn_south_minID] and Vs[ID]>0):
            # South, dawn: +Jr means Up, R1 is down, R2 is up
            if Polarity[ID]==-1:
                regions[ID] = 1
            else:
                regions[ID] = 2

        '''
        if distances[ID]<90 and Vs[ID]<0 and distances[ID]<:
            if Polarity[ID]==1:
                regions[ID] = 1
        elif distances[ID]<90 and Vs[ID]>0:
            if Polarity[ID]==-1:
                regions[ID] = 1
        elif distances[ID]>90 and Vs[ID]<0:
            if Polarity[ID]==1:
                regions[ID] = 1
        elif distances[ID]>90 and Vs[ID]>0:
            if Polarity[ID]==-1:
                regions[ID] = 1
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

def find_oval_dims(J:np.array,th:np.array,psi:np.array,hemi:str) -> dict:
    oval = {}
    # Get theta mean from Jmax along each longitude
    th_max_array = np.zeros(len(np.unique(psi)))
    J_max_array = np.zeros(len(np.unique(psi)))
    for i,psi_search in enumerate(np.unique(psi)):
        J_max_array[i] = J[psi==psi_search].max()
        th_max_array[i] = th[(psi==psi_search) & (J==J_max_array[i])][0]
    th_mean = np.sum(th_max_array*J_max_array)/np.sum(J_max_array)
    # Get the day/night shift from noon/midnight values
    J_noon = J[psi==0].max()
    J_midnight = J[psi==180].max()
    th_noon = th[(psi==0) & (J==J_noon)][0]
    th_midnight = th[(psi==180) & (J==J_midnight)][0]

    dtheta = (J_noon*(th_noon-th_mean)-J_midnight*(th_midnight-th_mean))/(
                                    J_noon+J_midnight)

    oval['dtheta'] = dtheta
    if hemi=='N':
        oval['theta_mean'] = np.max([th_mean,15])
        oval['theta_day'] = oval['theta_mean']+oval['dtheta']+2.5
        oval['theta_night'] = oval['theta_mean']-oval['dtheta']-2.5
        oval['Aoval'] = 2*pi*1*(1-cos(deg2rad(oval['theta_mean'])))
    else:
        oval['theta_mean'] = np.min([th_mean,165])
        oval['theta_day'] = oval['theta_mean']-oval['dtheta']-2.5
        oval['theta_night'] = oval['theta_mean']+oval['dtheta']+2.5
        oval['Aoval'] = 2*pi*1*(1-cos(pi-deg2rad(oval['theta_mean'])))
    return oval


def id_RIM_oval(ionosphere:Calculator) -> dict:
    oval = {}
    iono = servermanager.Fetch(ionosphere)
    iono = dsa.WrapDataObject(iono)
    J = iono.PointData['J_R_uA_m^2']
    th = iono.PointData['Theta_deg']
    psi = iono.PointData['Psi_deg']
    # Split hemispheres
    J_N = J[th<90]
    th_N = th[th<90]
    psi_N = psi[th<90]
    oval_N = find_oval_dims(J_N,th_N,psi_N,'N')
    for key in oval_N:
        oval[key+'_N'] = oval_N[key]

    J_S = J[th>90]
    th_S = th[th>90]
    psi_S = psi[th>90]
    oval_S = find_oval_dims(J_S,th_S,psi_S,'S')
    for key in oval_S:
        oval[key+'_S'] = oval_S[key]

    return oval

def save_ocflb_stats(polarcap:Threshold,hemi:str) -> dict:
    stats = {}
    ## Get the total area
    # point data to cell data 
    pc_cell=PointDatatoCellData(registrationName=f"cell{hemi}",Input=polarcap)
    pc_cell.PassPointData = 1
    # reveal cell areas
    pc_area = CellSize(registrationName=f"area{hemi}",Input=pc_cell)
    # fetch the data
    pc_data = servermanager.Fetch(pc_area)
    pc_data = dsa.WrapDataObject(pc_data)
    area_cc = pc_data.CellData['Area']
    area_total = np.sum(area_cc)

    ## Get the max/min latitudes
    th_point = pc_data.PointData['Theta_deg']
    psi_point = pc_data.PointData['Psi_deg']
    lat_max = th_point.max()
    lat_min = th_point.min()
    if hemi=='N':
        lat_noon = th_point[psi_point==0].max()
        if any(psi_point==180):
            lat_midnight = th_point[psi_point==180].max()
        else:
            lat_midnight = th_point[psi_point==0].min()
    elif hemi=='S':
        lat_noon = th_point[psi_point==0].min()
        if any(psi_point==180):
            lat_midnight = th_point[psi_point==180].min()
        else:
            lat_midnight = th_point[psi_point==0].max()

    ## Load output dict
    stats['open_area'] = area_total
    stats['theta_max'] = lat_max
    stats['theta_min'] = lat_min
    stats['theta_noon'] = lat_noon
    stats['theta_midnight'] = lat_midnight

    return stats



def id_ocflb(ionosphere:Calculator) -> dict:
    ocflb = {}
    # Separate hemispheres
    north_iono = Threshold(registrationName='North',Input=ionosphere)
    north_iono.Scalars = 'Theta_deg'
    north_iono.ThresholdMethod = 1 # 0- between, 1- below, 2- above
    north_iono.LowerThreshold = 90

    south_iono = Threshold(registrationName='South',Input=ionosphere)
    south_iono.Scalars = 'Theta_deg'
    south_iono.ThresholdMethod = 2 # 0- between, 1- below, 2- above
    south_iono.UpperThreshold = 90
    # Threshold the variable from #TRACEIE
    north_cap = Threshold(registrationName='NorthCap',Input=north_iono)
    north_cap.Scalars = 'RT_1_B_1_T'
    north_cap.ThresholdMethod = 1
    north_cap.LowerThreshold = -1e5

    south_cap = Threshold(registrationName='SouthCap',Input=south_iono)
    south_cap.Scalars = 'RT_1_B_1_T'
    south_cap.ThresholdMethod = 1
    south_cap.LowerThreshold = -1e5
    # Save some quick statistics
    north_stats = save_ocflb_stats(north_cap,'N')
    for key in north_stats:
        ocflb[key+'N'] = north_stats[key]
    south_stats = save_ocflb_stats(south_cap,'S')
    for key in south_stats:
        ocflb[key+'S'] = south_stats[key]

    return ocflb


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
            FAC['UP_R1_N'] = np.sum(Jr[(Z>0) & (Region==1)]*
                                    Area[(Z>0) & (Region==1)])*6371**2*1e-6
            FAC['UP_R2_N'] = np.sum(Jr[(Z>0) & (Region==2)]*
                                    Area[(Z>0) & (Region==2)])*6371**2*1e-6

            FAC['UP_R1_S']   = np.sum(Jr[(Z<0) & (Region==1)]*
                                    Area[(Z<0) & (Region==1)])*6371**2*1e-6
            FAC['UP_R2_S']   = np.sum(Jr[(Z<0) & (Region==2)]*
                                    Area[(Z<0) & (Region==2)])*6371**2*1e-6
        elif i==1:
            FAC['DOWN_R1_N']   = np.sum(Jr[(Z>0) & (Region==1)]*
                                    Area[(Z>0) & (Region==1)])*6371**2*1e-6
            FAC['DOWN_R2_N']   = np.sum(Jr[(Z>0) & (Region==2)]*
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

