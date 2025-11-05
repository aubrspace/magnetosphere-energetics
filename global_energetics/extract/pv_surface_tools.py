import paraview
#### import the simple module from paraview
from paraview.simple import *
#NOTE 'object' = stand in for paraview filter-like object in type hints

def get_numpy_surface_analysis():
    #TODO
    # Staging function that will take the surface_dict and pass what is needed
    #   for each calculation one at a time, compiling the results
    # Inputs
    #   surface_dict, *, skip_keys=[]
    # Returns
    #   results_dict
    #   
    #   results_dict = {}
    #   for surf,source in surface_dict.items():
    #       TODO- figure out the level of specificity here
    #       could do a number of very specific numpy domain functions, one
    #       for each instance of the suface object?
    #           or
    #       Could have a generic function with some key word arguments
    #        controling what is calculated and what is not ...
    #       
    #       results_dict[surf] = somecall(surf,source)
    #   return results_dict
    pass

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
    iso1 = Contour(registrationName=name, Input=inputsource)
    iso1.ContourBy = ['POINTS', variable]
    iso1.ComputeNormals = 1#NOTE if comptuted now, seem to cause trouble
    iso1.Isosurfaces = [kwargs.get('iso_value',1.0)]
    iso1.PointMergeMethod = kwargs.get('mergemethod','Uniform Binning')
    outputsource = iso1

    #Trim any small floating regions
    if kwargs.get('trim_regions',True):
        #assert FindSource('MergeBlocks1')!=None
        # Keep only the largest connected region
        RenameSource(name+'_hits', outputsource)
        iso2 = Connectivity(registrationName=name, Input=outputsource)
        iso2.ExtractionMode = 'Extract Largest Region'
        outputsource = iso2

    #Generate normals now that the surface is fully constructed
    if kwargs.get('calc_normals',True):
        RenameSource(name+'_beforeNormals', outputsource)
        if paraview.__version__ == '6.0.0':
            iso3 = SurfaceNormals(registrationName=name,Input=outputsource)
        else:
            iso3 = GenerateSurfaceNormals(registrationName=name,
                                      Input=outputsource)
        iso3.ComputeCellNormals = 1
        iso3.NonManifoldTraversal = 0
        outputsource = iso3

    outputsource.UpdatePipeline()
    return outputsource
