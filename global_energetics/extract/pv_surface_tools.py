import numpy as np
import paraview
#### import the simple module from paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy
#NOTE 'object' = stand in for paraview filter-like object in type hints

def construct_condition(description:list[str],
                       surface_data:dict,
                           **kwargs:dict) -> []:
    """ Takes a description in the form of list of str, considered AND of all
         and returns a numpy array of bools which represents all surface
         elements meeting this condition
    Inputs
        description (list[str]) - ex. ['closed','daymapped']
        surface_data {str:np.ndarrays} - surface solution/state
        kwargs:
    Returns
        condition (np.ndarray(dtype=bool)) - True/False for each element
    """
    # Start will all true
    condition = np.ones(len(surface_data['x']),dtype=bool)
    # Pull some helpful variables out
    status = surface_data['Status']
    x = surface_data['x']
    z = surface_data['z']
    r = surface_data['r_R']
    for descriptor in description:
        if descriptor=='open':
            condition = condition*(status>0)*(status<3)
        if descriptor=='closed':
            condition = condition*(status==3)
        if descriptor=='tail':
            tail_x = x.min()
            condition = condition*(abs(x-tail_x)<1)#TODO come back to this
        if descriptor=='no_tail':
            tail_x = x.min()
            condition = condition*(abs(x-tail_x)>1)#TODO come back to this
        if descriptor=='on_innerbound':
            inner_r = r.min()
            condition = condition*(abs(r-inner_r)<0.25)#TODO come back to this
        if descriptor=='north':
            condition = condition*(z>0)
        if descriptor=='north':
            condition = condition*(z>0)
    return condition

def map_surface_to_interfaces(surface_name:str,
                              surface_data:dict[np.ndarray],
                                  **kwargs:dict) -> dict[str:np.ndarray]:
    """ Takes a surface name and dictionary of arrays and gets the condition
         arrays which represent the partial integral bounds of each required
         calculation
    Inputs
        surface_name (str) - name
        surface_data {str:np.ndarray} - solution or state of the surface
        kwargs:
    Returns
        conditions {str:np.ndarray(dytype=bool)} - dict of conditions
    """
    conditions = {}
    if surface_name == 'mp':
        # K1        - open, no tail
        # K5        - closed, no tail
        conditions['K1']=construct_condition(['open','no_tail'],surface_data)
        conditions['K5']=construct_condition(['closed','no_tail'],surface_data)
    if surface_name == 'lobes':
        # K3        - on innerbound
        # K4        - tail
        conditions['K3'] = construct_condition(['on_innerbound'],surface_data)
        conditions['K4'] = construct_condition(['tail'],surface_data)
    if surface_name == 'closed':
        # K6        - tail
        # K7        - on innerbound
        conditions['K6'] = construct_condition(['tail'],surface_data)
        conditions['K7'] = construct_condition(['on_innerbound'],surface_data)
    if surface_name == 'inner':
        # K3n       - open, north
        # K3s       - open, south
        # K7        - closed
        conditions['K3n'] = construct_condition(['open','north'],surface_data)
        conditions['K3s'] = construct_condition(['open','south'],surface_data)
        conditions['K7'] = construct_condition(['closed'],surface_data)
    return conditions

def get_numpy_surface_analysis(surfaces_dict:dict,*,
                            integrands:list=['K_W_Re2','ExB_W_Re2','P0_W_Re2'],
                                   skip_keys:list=[]) -> dict:
    """Staging function that will take the surface_dict and pass what is needed
        for each calculation one at a time, compiling the results
    Inputs
        surface_dict {str:object}
        kwargs:
            integrands (list) -  which things are being integrated
            skip_keys (list) - default empty
    Returns
        results_dict {str:np.ndarray}
    """
    integral_translation = {'K_W_Re2':('K','_TW'),# gives ID tag, unit
                          'ExB_W_Re2':('ExB','_TW'),
                           'P0_W_Re2':('P0','_TW')}
    results = {}
    print('ANALYZING SURFACE(S): ...')
    for surf,source in surfaces_dict.items():
        print(f'\t{surf}')
        # Extract the surface state from VTKArray -> np.ndarray
        np_surface = {}
        data = servermanager.Fetch(source)
        data = dsa.WrapDataObject(data)
        for variable in data.CellData.keys():
            np_surface[variable] = vtk_to_numpy(data.CellData[variable])
        # Get conditional areas representing partial integral bounds
        conditions = map_surface_to_interfaces(surf,np_surface)
        # Calculate each partial integral
        for integrand in integrands:
            print(f'\t\t{integrand}')
            # pull out the integrand as an array
            integrand_values = np_surface[integrand]
            # adjust post-integration units using a dict
            integral_name, units = integral_translation[integrand]
            # Calculate the total (no partial bounds)
            entry_name = surf+'_'+integral_name+units
            results[entry_name] = np.sum(np.sum(
                                integrand_values*np_surface['Normals'],axis=1)
                                                      *np_surface['Area'])/1e12
            for condition_name,cond in conditions.items():
                # Set the specific entry given the surf+condition combo
                entry_name = surf+'_'+integral_name+condition_name+units
                # Use double np.sum for dot product with Normals and Area
                results[entry_name] = np.sum(np.sum(
                    integrand_values[cond]*np_surface['Normals'][cond],axis=1)
                                                *np_surface['Area'][cond])/1e12
        ##################################################################
        #DEBUG
        #np.savez_compressed(surf+'.npz',allow_pickle=False,**np_surface)
        #print(f"Saved {surf}.npz")
        ##################################################################
    return results

def get_surface_flux(source:object,
                     variable:str,name:str,
                     **kwargs:dict         ) -> object:
    #First find out if our variable lives on points or cell centers
    #NOTE if on both lists (bad practice) we'll use the cell centered one
    cc = variable in source.CellData.keys()
    if not cc:
        assert variable in source.PointData.keys(), "Bad variable name!"
        vartype = 'Point Data'
    else:
        vartype = 'Cell Data'
    #Create calculator filter that is flux
    flux = Calculator(registrationName=name,Input=source)
    flux.AttributeType = vartype
    flux.Function = 'dot('+variable+',Normals)'
    flux.ResultArrayName = name
    # create a new 'Integrate Variables'
    result=IntegrateVariables(registrationName=name+'_integrated',Input=flux)
    return result

def create_magnetopause_state(pipeline:object,**kwargs:dict) -> object:
    """Function calculates a magnetopause variable, NOTE:will still need to
        process variable into iso surface then cleanup iso surface!
    Inputs
        inputsource (filter/source)- upstream that calculator will process
        kwargs:
            verbose_pipeline (bool) - default False
            betastar_max (float)- default 0.7
            status_closed (float)- default 3
            tail_x (float)- default -20
            inner_r (float)- default 3.0
    Returns
        outputsource (filter)- last filter applied keeping a straight pipeline
            or
        string (script)- just the text of the prog filter to be combined
    """
    betastar_max = kwargs.get('betastar_max',0.7)
    closed_value = kwargs.get('status_closed',3)
    tail_x = kwargs.get('tail_x',-20)
    inner_r = kwargs.get('inner_r',3.0)
    script = ''
    if kwargs.get('verbose_pipeline',False):
        script +="""
# Get input
beta_star = inputs[0].PointData['beta_star']
Status = inputs[0].PointData['Status']
x = inputs[0].PointData['x']
r_R = inputs[0].PointData['r_R']"""
    script +="""
#Compute magnetopause as logical combination
mp = ((Status>=1)&(beta_star<"""+str(betastar_max)+""")&
          (x>"""+str(tail_x)+""")&(r_R>="""+str(inner_r)+""")
                |
            (Status=="""+str(closed_value)+""")&
          (x>"""+str(tail_x)+""")&(r_R>="""+str(inner_r)+""")).astype(int)
output.PointData.append(mp,'mp')"""

    if kwargs.get('verbose_pipeline',False):
        script+="""
#Assign to output
output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
output.PointData.append(mp,'mp')"""
        #Must have the following conditions met first
        assert 'beta_star' in inputsource.PointData.keys(), 'No Beta*!'
        mp = ProgrammableFilter(registrationName='mp',
                                      Input=inputsource)
        mp.Script = script
        return mp
    else:
        return script

def create_closed_state(inputsource:object, **kwargs:dict) -> object:
    """Function creates closed field state
    Inputs
        inputsource (filter/source)- upstream that calculator will process
        kwargs:
            verbose_pipeline (bool) - default False
            status_closed (float)- default 3
            tail_x (float)- default -20
            inner_r (float)- default 3.0
    Returns
        outputsource (filter)- last filter applied keeping a straight pipeline
            or
        string (script)- just the text of the prog filter to be combined
    """
    closed_value = kwargs.get('status_closed',3)
    tail_x = kwargs.get('tail_x',-20)
    inner_r = kwargs.get('inner_r',3.0)
    script = ''
    if kwargs.get('verbose_pipeline',False):
        script +="""
# Get input
Status = inputs[0].PointData['Status']
x = inputs[0].PointData['x']
r_R = inputs[0].PointData['r_R']"""
    script +="""
#Compute closed surface as logical combination
closed = ((Status=="""+str(closed_value)+""")&
          (x>"""+str(tail_x)+""")&(r_R>="""+str(inner_r)+""")).astype(int)
output.PointData.append(closed,'closed')"""

    if kwargs.get('verbose_pipeline',False):
        script+="""
#Assign to output
output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
output.PointData.append(closed,'closed')"""
        #Must have the following conditions met first
        assert 'Status' in pipeline.PointData.keys(), 'No Status variable!'
        closed = ProgrammableFilter(registrationName='closed',
                                      Input=inputsource)
        closed.Script = script
        return closed
    else:
        return script

def create_lobes_state(inputsource:object, **kwargs:dict) -> object:
    """Function creates lobes field state
    Inputs
        inputsource (filter/source)- upstream that calculator will process
        kwargs:
            verbose_pipeline (bool) - default False
            betastar_max (float) - default 0.7
            status_open ([float,float])- default [1,2]
            tail_x (float)- default -20
            inner_r (float)- default 3.0
    Returns
        outputsource (filter)- last filter applied keeping a straight pipeline
            or
        string (script)- just the text of the prog filter to be combined
    """
    betastar_max = kwargs.get('betastar_max',0.7)
    open_values = kwargs.get('status_open',[1.0,2.0])
    tail_x = kwargs.get('tail_x',-20)
    inner_r = kwargs.get('inner_r',3.0)
    script = ''
    if kwargs.get('verbose_pipeline',False):
        script +="""
# Get input
Status = inputs[0].PointData['Status']
x = inputs[0].PointData['x']
r_R = inputs[0].PointData['r_R']"""
    script +="""
#Compute lobes surface as logical combination
lobes = (((Status=="""+str(open_values[0])+""")|
          (Status=="""+str(open_values[1])+"""))&
          (beta_star<"""+str(betastar_max)+""")&
          (x>"""+str(tail_x)+""")&(r_R>="""+str(inner_r)+""")).astype(int)
output.PointData.append(lobes,'lobes')"""

    if kwargs.get('verbose_pipeline',False):
        script+="""
#Assign to output
output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
output.PointData.append(lobes,'lobes')"""
        #Must have the following conditions met first
        assert 'Status' in pipeline.PointData.keys(), 'No Status variable!'
        lobes = ProgrammableFilter(registrationName='lobes',
                                      Input=inputsource)
        lobes.Script = script
        return lobes
    else:
        return script

def create_inner_state(inputsource:object,**kwargs:dict) -> object:
    """Function calculates a magnetopause variable, NOTE:will still need to
        process variable into iso surface then cleanup iso surface!
    Inputs
        inputsource (filter/source)- upstream that calculator will process
        kwargs:
            inner_r (float)- default 3.0
    Returns
        outputsource (filter)- last filter applied keeping a straight pipeline
            or
        string (script)- just the text of the prog filter to be combined
    """
    inner_r = kwargs.get('inner_r',3.0)
    script = ''
    if kwargs.get('verbose_pipeline',False):
        script +="""
# Get input
r = inputs[0].PointData['r_R']"""
    script +="""
#Compute lobes surface as logical combination
inner = ().astype(int)
output.PointData.append(lobes,'lobes')"""

    if kwargs.get('verbose_pipeline',False):
        script+="""
#Assign to output
output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
output.PointData.append(lobes,'lobes')"""
        #Must have the following conditions met first
        assert 'Status' in pipeline.PointData.keys(), 'No Status variable!'
        lobes = ProgrammableFilter(registrationName='lobes',
                                      Input=inputsource)
        lobes.Script = script
        return lobes
    else:
        return script
    pass


def create_iso_surface(inputsource, variable, name, **kwargs):
    """Function creates iso surface from variable
    Inputs
        inputsource (filter/source)- what data is used as input
        variable (str)- name of variable for contour
        name (str)- registration name for new object (filter)
        kwargs:
            iso_value (float)- default 1
            contourtyps (str)- default 'POINTS'
            mergemethod (str)- default 'Uniform Binning'
            trim_regions (bool)- default True, will keep largest connected
    Returns
        outputsource (filter)- filter applied so things can easily attach
    """
    # Create iso surface
    iso = Contour(registrationName=name+'_init', Input=inputsource)
    iso.ContourBy = ['POINTS', variable]
    iso.ComputeNormals = 0#NOTE if comptuted now, seem to cause trouble
    iso.Isosurfaces = [kwargs.get('iso_value',1.0)]
    iso.PointMergeMethod = kwargs.get('mergemethod','Uniform Binning')

    # Convert point data -> cell data to ease integration
    point2cell = PointDatatoCellData(registrationName=name+'_p2c',
                                     Input=iso)
    point2cell.ProcessAllArrays = 1
    outputsource = point2cell

    #Trim any small floating regions
    if kwargs.get('trim_regions',True):
        #assert FindSource('MergeBlocks1')!=None
        # Keep only the largest connected region
        trim = Connectivity(registrationName=name+'_trim', Input=outputsource)
        trim.ExtractionMode = 'Extract Largest Region'
        trim.ColorRegions = 0
        outputsource = trim

    #Generate normals now that the surface is fully constructed
    if kwargs.get('calc_normals',True):
        if paraview.__version__ == '6.0.0':
            normals = SurfaceNormals(registrationName=name+'_normals',
                                     Input=outputsource)
        else:
            normals = GenerateSurfaceNormals(registrationName=name+'_normals',
                                             Input=outputsource)
        normals.ComputeCellNormals = 1
        normals.NonManifoldTraversal = 0
        outputsource = normals

    # Calculate Cell Area
    cellsize = CellSize(registrationName=name,
                        Input=outputsource)
    cellsize.ComputeVertexCount = 0
    cellsize.ComputeLength = 0
    cellsize.ComputeArea = 1
    cellsize.ComputeVolume = 0
    cellsize.ComputeSum = 0
    outputsource = cellsize

    outputsource.UpdatePipeline()
    return outputsource
