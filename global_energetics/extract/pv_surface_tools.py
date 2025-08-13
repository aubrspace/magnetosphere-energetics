import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0
import numpy as np
#### import the simple module from paraview
from paraview.simple import *

def get_surface_flux(source,variable,name,**kwargs):
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
    iso1.Isosurfaces = [kwargs.get('iso_value',1)]
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

    return outputsource
