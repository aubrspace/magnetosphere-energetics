import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *

def find_IE_matched_file(path,filetime):
    """Function returns the IE file at a specific time, if it exists
    Inputs
        path (str)
        filetime (datetime)
    Returns
        iedatafile (str)
    """
    iedatafile = (path+
                  'it{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}_000.tec'.format(
                      filetime.year-2000,
                      filetime.month,
                      filetime.day,
                      filetime.hour,
                      filetime.minute,
                      filetime.second))
    return iedatafile

def read_aux(infile):
    """Reads in auxillary data file stripped from a .plt tecplot file
    Inputs
        infile (str)- full path to file location
    Returns
        data (dict)- dictionary of all data contained in the file
    """
    data = {}
    with open(infile,'r') as f:
        for line in f.readlines():
            if 'TIME' in line:
                data[line.split(':')[0]] = ':'.join(line.split(':')[1::])
            else:
                data[line.split(':')[0]]=line.split(':')[-1].replace(
                                                 ' ','').replace('\n','')
    return data

def read_tecplot(infile,**kwargs):
    """Function reads tecplot binary file
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
    Returns
        sourcedata (pvpython object)- python object attached to theVTKobject
                                      for the input data
    """
    if kwargs.get('binary',True):
        # create a new 'VisItTecplotBinaryReader'
        sourcedata = VisItTecplotBinaryReader(FileName=[infile],
                                   registrationName=infile.split('/')[-1])
        # Call Mesh
        sourcedata.MeshStatus
        # Call Point arrays- we want to load everything
        status = sourcedata.GetProperty('PointArrayInfo')
        #listed as ['thing','1','thing2','0'] where '0' and '1' are 
        #   unloaded and loaded respectively
        arraylist = [s for s in status if s!='0' and s!='1']
        sourcedata.PointArrayStatus = arraylist
    else:
        # create a new 'TecplotReader'
        sourcedata = TecplotReader(registrationName=infile.split('/')[-1],
                                   FileNames=[infile])
    return sourcedata

def prepend_names(pipeline,prepend,**kwargs):
    names = ProgrammableFilter(registrationName='prepends', Input=pipeline)
    names.Script = """
        #Get upstream data
        data = inputs[0]
        #These are the variables names that cause issues
        for key in data.PointData.keys():
            values = data.PointData[key]
            output.PointData.append(values,'"""+prepend+"""'+key)
    """
    pipeline = names
    return pipeline

def fix_ie_names(pipeline,**kwargs):
    names = ProgrammableFilter(registrationName='names', Input=pipeline)
    #NOTE JR is mislabeled and is in mA not microA!!!! (flips table)
    names.Script = """
        #Get upstream data
        data = inputs[0]
        #These are the variables names that cause issues
        new_name = {"Ave-E [keV]":'Ave-E_keV',
                    "conjugate dLat [deg]":"conjugate_dLat_deg",
                    "conjugate dLon [deg]":"conjugate_dLon_deg",
                    "E-Flux [W_m^2]":"E-Flux_W_m^2",
                    "Ex [mV_m]":"E_x_mV_m",
                    "Ey [mV_m]":"E_y_mV_m",
                    "Ez [mV_m]":"E_z_mV_m",
                    "IonNumFlux [_cm^2_s]":"IonNumFlux__cm^2_s",
                    "JouleHeat [mW_m^2]":"JouleHeat_mW_m^2",
                    "JR [`mA_m^2]":"J_R_uA_m^2",
                    "Jx [`mA_m^2]":"J_x_uA_m^2",
                    "Jy [`mA_m^2]":"J_y_uA_m^2",
                    "Jz [`mA_m^2]":"J_z_uA_m^2",
                    "PHI [kV]":"PHI_kV",
                    "Psi [deg]":"Psi_deg",
                    "RT 1_B [1_T]":"RT_1_B_1_T",
                    "RT P [Pa]":"RT_P_Pa",
                    "RT Rho [kg_m^3]":"RT_Rho_kg_m^3",
                    "SigmaH [S]":"SigmaH_S",
                    "SigmaP [S]":"SigmaP_S",
                    "Theta [deg]":"Theta_deg",
                    "Ux [km_s]":"U_x_km_s",
                    "Uy [km_s]":"U_y_km_s",
                    "Uz [km_s]":"U_z_km_s",
                    "X [R]":"x",
                    "Y [R]":"y",
                    "Z [R]":"z"}
        #for var in varlist:
        #    if var in data.PointData.keys():
        #        print(var)
        for var in data.PointData.keys():
            output.PointData.append(data.PointData[var],new_name[var])
        #rho = data.PointData["Rho_amu_cm^3"]
        #jx = data.PointData["J_x_`mA_m^2"]
        #jy = data.PointData["J_y_`mA_m^2"]
        #jz = data.PointData["J_z_`mA_m^2"]
        #Copy input to output so we don't lose any data
        #output.ShallowCopy(inputs[0].VTKObject)#maintaining other variables
        #Now append the copies of the variables with better names
        #output.PointData.append(rho,'Rho_amu_cm3')
        #output.PointData.append(jx,'J_x_uA_m2')
        #output.PointData.append(jy,'J_y_uA_m2')
        #output.PointData.append(jz,'J_z_uA_m2')
    """
    pipeline = names
    return pipeline

def fix_names(pipeline,**kwargs):
    names = ProgrammableFilter(registrationName='names', Input=pipeline)
    names.Script = """
        #Get upstream data
        data = inputs[0]
        name_dict = {"Rho_amu_cm^3":"Rho_amu_cm3",
                     "dvol_R^3":"dvol_R3",
                     "J_x_`mA_m^2":"J_x_uA_m2",
                     "J_y_`mA_m^2":"J_y_uA_m2",
                     "J_z_`mA_m^2":"J_z_uA_m2"}
        for var in data.PointData.keys():
            if var not in name_dict.keys():
                output.PointData.append(data.PointData[var],var)
            else:
                output.PointData.append(data.PointData[var],name_dict[var])
    """
    pipeline = names
    return pipeline

def status_repair(pipeline,**kwargs):
    status_repair = ProgrammableFilter(registrationName='status_repair',
                                       Input=pipeline)
    status_repair.Script = """
    #Get upstream data
    data = inputs[0]
    status = data.PointData['Status']
    theta1 = data.PointData['theta_1_deg']
    theta2 = data.PointData['theta_2_deg']
    closed = ((theta1>=0)&(theta1<=90)
                  &
              (theta2<=0)&(theta2>=-90)).astype(int)
    open_north = ((theta1>=0)&(theta1<=90)
                  &
                  (theta2<=-90)).astype(int)
    open_south = ((theta1<=0)
                  &
                  (theta2<=0)&(theta2>=-90)).astype(int)
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(closed,'closed')
    output.PointData.append(open_north,'open_north')
    output.PointData.append(open_south,'open_south')
                         """
    return status_repair

def todimensional(pipeline, **kwargs):
    """Function modifies dimensionless variables -> dimensional variables
    Inputs
        dataset (frame.dataset)- tecplot dataset object
        kwargs:
    """
    proton_mass = 1.6605e-27
    cMu = np.pi*4e-7
    #SWMF sets up two conversions to go between
    # No (nondimensional) Si (SI) and Io (input output),
    # what we want is No2Io = No2Si_V / Io2Si_V
    #Found these conversions by grep'ing for 'No2Si_V' in ModPhysics.f90
    No2Si = {'X':6371*1e3,                             #planet radius
             'Y':6371*1e3,
             'Z':6371*1e3,
             'Rho':1e6*proton_mass,                    #proton mass
             'U_x':6371*1e3,                           #planet radius
             'U_y':6371*1e3,
             'U_z':6371*1e3,
             'P':1e6*proton_mass*(6371*1e3)**2,        #Rho * U^2
             'B_x':6371*1e3*np.sqrt(cMu*1e6*proton_mass), #U * sqrt(M*rho)
             'B_y':6371*1e3*np.sqrt(cMu*1e6*proton_mass),
             'B_z':6371*1e3*np.sqrt(cMu*1e6*proton_mass),
             'J_x':(np.sqrt(cMu*1e6*proton_mass)/cMu),    #B/(X*cMu)
             'J_y':(np.sqrt(cMu*1e6*proton_mass)/cMu),
             'J_z':(np.sqrt(cMu*1e6*proton_mass)/cMu)
            }
    #Found these conversions by grep'ing for 'Io2Si_V' in ModPhysics.f90
    Io2Si = {'X':6371*1e3,                  #planet radius
             'Y':6371*1e3,                  #planet radius
             'Z':6371*1e3,                  #planet radius
             'Rho':1e6*proton_mass,         #Mp/cm^3
             'U_x':1e3,                     #km/s
             'U_y':1e3,                     #km/s
             'U_z':1e3,                     #km/s
             'P':1e-9,                      #nPa
             'B_x':1e-9,                    #nT
             'B_y':1e-9,                    #nT
             'B_z':1e-9,                    #nT
             'J_x':1e-6,                    #microA/m^2
             'J_y':1e-6,                    #microA/m^2
             'J_z':1e-6,                    #microA/m^2
             }#'theta_1':pi/180,              #degrees
             #'theta_2':pi/180,              #degrees
             #'phi_1':pi/180,                #degrees
             #'phi_2':pi/180                 #degrees
            #}
    units = {'X':'R','Y':'R','Z':'R',
             'Rho':'amu_cm3',
             'U_x':'km_s','U_y':'km_s','U_z':'km_s',
             'P':'nPa',
             'B_x':'nT','B_y':'nT','B_z':'nT',
             'J_x':'uA_m2','J_y':'uA_m2','J_z':'uA_m2',
             'theta_1':'deg',
             'theta_2':'deg',
             'phi_1':'deg',
             'phi_2':'deg'
            }
    points = pipeline.PointData
    var_names = points.keys()
    for var_in in var_names:
        if 'future_' in var_in:
            var = var_in.split('future_')[-1]
            prepend = 'future_'
        else:
            var = var_in
            prepend = ''
        if 'status' not in var.lower() and '[' not in var:
            if var in units.keys():
                unit = units[var]
                #NOTE default is no modification
                conversion = No2Si.get(var,1)/Io2Si.get(var,1)
                print('Changing '+prepend+var+' by multiplying by '+
                      str(conversion))
                #eq('{'+var+' '+unit+'} = {'+var+'}*'+str(conversion))
                eq = Calculator(registrationName=prepend+var+'_'+unit,
                                Input=pipeline)
                eq.Function = prepend+var+'*'+str(conversion)
                eq.ResultArrayName = prepend+var+'_'+unit
                pipeline=eq
                #print('Removing '+prepend+var)
                #dataset.delete_variables([dataset.variable(prepend+var).index])
            else:
                print('unable to convert '+prepend+var+' not in dictionary!')
        elif 'status' in var.lower():
            pass
    #Reset XYZ variables so 3D plotting makes sense
    #dataset.frame.plot().axes.x_axis.variable = dataset.variable('X *')
    #dataset.frame.plot().axes.y_axis.variable = dataset.variable('Y *')
    #dataset.frame.plot().axes.z_axis.variable = dataset.variable('Z *')
    return pipeline

def merge_sources(source1,source2,*,prepend1='',prepend2=''):
    """Function creates a filter to combine two sources
    Inputs
        source(1/2) - source to be combined
    Returns
        merged_source
    """
    if prepend1 != '':
        new_source1 = prepend_names(source1,prepend1)
    else:
        new_source1 = source1
    if prepend2 != '':
        new_source2 = prepend_names(source2,prepend2)
    else:
        new_source2 = source2
    merged_source = AppendAttributes(registrationName='DataMerge',
                                     Input=[new_source1, new_source2])
    return merged_source

def interpolate_data(source,name,origin,scale):
    """Function creates a pointVolumeInterpolator given the bounds
    Inputs
        source
        name
    Returns
        interpolated_source
    """
    interpolated = PointVolumeInterpolator(registrationName='interpolated',
                                       Input=source, Source='Bounded Volume')
    interpolated.Kernel = 'VoronoiKernel'
    interpolated.Locator = 'Static Point Locator'
    interpolated.Source.Origin = origin
    interpolated.Source.Scale = scale
    interpolated.Source.RefinementMode = 'Use cell-size'
    interpolated.Source.CellSize = 0.125
    return interpolated

def prepare_data(infile,**kwargs):
    source_now = read_tecplot(infile)
    if kwargs.get('do_motion',False) and 'futurefile' in kwargs:
        source_future = read_tecplot(kwargs.get('futurefile'))
        #source1 = MergeBlocks(registrationName='ExtractNow',
        #                       Input=source_now)
        #source2 = MergeBlocks(registrationName='ExtractFuture',
        #                       Input=source_future)
        pipeline = merge_sources(source_now,source_future,prepend2='future_')
    else:
        pipeline = MergeBlocks(registrationName='MergeBlocks',
                               Input=source_now)
    if 'dimensionless' in kwargs:
        pipeline = todimensional(pipeline,**kwargs)
    else:
        ###Rename some tricky variables
        pipeline = fix_names(pipeline,**kwargs)
    pipeline = interpolate_data(pipeline,'interpolated',[-20,-40,-40],
                                                        [40,80,80])
    return pipeline
