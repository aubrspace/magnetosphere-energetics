# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

import glob
import time
import numpy as np
#### import the simple module from paraview
from paraview.simple import *
def read_tecplot(infile):
    """Function reads tecplot binary file
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
    Returns
        sourcedata (pvpython object)- python object attached to theVTKobject
                                      for the input data
    """
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
    return sourcedata

def get_vectors(pipeline,**kwargs):
    """Function sets up calculator filters to turn components into vector
        objects (this will allow field tracing and other things)
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            vector_comps (dict)- defaults below, change if needed
                               {'B_nT':['B_x_nT','B_y_nT','B_z_nT'],
                                'U_km_s':['U_x_km_s','U_y_km_s','U_z_km_s'],
            '"J_`mA_m^2"':['"J_x_`mA_m^2"','"J_y_`mA_m^2"','"J_z_`mA_m^2"']}
    Return
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    ###Get Vectors from field variable components
    vector_comps = kwargs.get('vector_comps',
                   {'B_nT':['B_x_nT','B_y_nT','B_z_nT'],
                    'U_km_s':['U_x_km_s','U_y_km_s','U_z_km_s'],
            '"J_`mA_m^2"':['"J_x_`mA_m^2"','"J_y_`mA_m^2"','"J_z_`mA_m^2"']})
    for vecName in vector_comps.keys():
        vector = Calculator(registrationName=vecName.split('_')[0],
                            Input=pipeline)
        vector.Function = [vector_comps[vecName][0]+'*iHat+'+
                           vector_comps[vecName][1]+'*jHat+'+
                           vector_comps[vecName][2]+'*kHat'][0]
        vector.ResultArrayName = vecName
        pipeline=vector
    return pipeline

def get_pressure_gradient(pipeline,**kwargs):
    """Function calculates a pressure gradient variable
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            point_data_name (str)- default \'P_nPa\'
            new_name (str)- default "GradP_nPa_Re"
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    gradP = ProgrammableFilter(registrationName='gradP', Input=pipeline)
    P_name = kwargs.get('point_data_name','\'P_nPa\'')
    gradP_name = kwargs.get('new_name','GradP_nPa_Re')
    gradP.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]

    #Compute gradient
    gradP = algs.gradient(data.PointData"""+str(P_name)+"""])
    data.PointData.append(gradP,"""+str(gradP_name)+""")

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(gradP,"""+str(gradP_name)+""")
    """
    pipeline = gradP
    return pipeline

def get_magnetopause_filter(pipeline,**kwargs):
    """Function calculates a magnetopause variable, NOTE:will still need to
        process variable into iso surface then cleanup iso surface!
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            betastar_max (float)- default 0.7
            status_closed (float)- default 3
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    #Must have the following conditions met first
    assert FindSource('betastar') != None
    betastar_max = kwargs.get('betastar_max',0.7)
    closed_value = kwargs.get('status_closed',3)
    mp_state =ProgrammableFilter(registrationName='mp_state',Input=pipeline)
    mp_state.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]
    betastar = data.PointData['betastar']
    status = data.PointData['Status']

    #Compute magnetopause as logical combination
    mp_state = ((betastar<"""+str(betastar_max)+""")|
                (status=="""+str(closed_value)+""")).astype(int)

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(mp_state,'mp_state')
    """
    pipeline = mp_state
    return pipeline

def setup_outline(source,**kwargs):
    """Once variables are set make an outline view of the domain
    Inputs
        source (filter/source)- what will be displayed
        kwargs:
            variablename (str)- what color to represent
    """
    # get active view
    renderView = GetActiveViewOrCreate('RenderView')
    # show data in view
    sourceDisplay = Show(source, renderView,
                        'UnstructuredGridRepresentation')

    # change representation type
    sourceDisplay.SetRepresentationType('Outline')
    # get color transfer function/color map
    variableLUT =GetColorTransferFunction(kwargs.get('variable','betastar'))
    # Hide the scalar bar for this color map if not used.
    HideScalarBarIfNotNeeded(variableLUT, renderView)

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
    if kwargs.get('trim_regions',True):
        name2 = name
        name = name+'_hits'
    # Create iso surface
    iso1 = Contour(registrationName=name, Input=inputsource)
    iso1.ContourBy = ['POINTS', variable]
    iso1.Isosurfaces = [kwargs.get('iso_value',1)]
    iso1.PointMergeMethod = kwargs.get('mergemethod','Uniform Binning')

    if kwargs.get('trim_regions',True):
        assert FindSource('MergeBlocks1')!=None
        # Keep only the largest connected region
        iso2 = Connectivity(registrationName=name2, Input=iso1)
        iso2.ExtractionMode = 'Extract Largest Region'
        outputsource = iso2
    else:
        outputsource = iso

    return outputsource

def display_visuals(field,mp,renderView,**kwargs):
    """Function standin for separate file governing visual representations
    Inputs
        field (source/filter)- data w all variables for streams, slices etc.
        mp (source/filter)- finalized magnetopause data
        renderView (View)- where to show things
        kwargs:
            mpContourBy
            doSlice
            sliceContourBy
            sliceContourLog
            doFieldLines
    returns
        TBD
    """
    # show outline of field
    setup_outline(field)
    # show iso surface
    mpDisplay = Show(mp, renderView, 'GeometryRepresentation')

    '''
    # get color transfer function/color map for 'Status'
    statusLUT = GetColorTransferFunction('Status')
    # Apply a preset using its name. Note this may not work as expected
    #   when presets have duplicate names.
    statusLUT.ApplyPreset('Rainbow Uniform', True)
    # set scalar coloring
    ColorBy(mpDisplay, ('POINTS', 'Status'))
    '''
    # change solid color
    ColorBy(mpDisplay, None)
    mpDisplay.AmbientColor = [0.0, 1.0, 1.0]
    mpDisplay.DiffuseColor = [0.0, 1.0, 1.0]

    # Properties modified on mpDisplay.DataAxesGrid
    mpDisplay.DataAxesGrid.GridAxesVisibility = 1
    # Properties modified on slice1Display
    mpDisplay.Opacity = 0.4


    ###Slice
    # create a new 'Slice'
    slice1 = Slice(registrationName='Slice1', Input=field)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = [0.0, 0.0, 0.0]
    slice1.SliceType.Normal = [0.0, 1.0, 0.0]

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice1.HyperTreeGridSlicer.Origin = [-94.25, 0.0, 0.0]
    # show data in view
    slice1Display = Show(slice1, renderView, 'GeometryRepresentation')
    # Properties modified on slice1Display
    slice1Display.Opacity = 0.6
    # set scalar coloring
    ColorBy(slice1Display, ('POINTS', 'Rho_amu_cm^3'))
    # get color transfer function/color map for 'Rho_amu_cm3'
    rho_amu_cm3LUT = GetColorTransferFunction('Rho_amu_cm3')

    # get opacity transfer function/opacity map for 'Rho_amu_cm3'
    rho_amu_cm3PWF = GetOpacityTransferFunction('Rho_amu_cm3')

    # convert to log space
    rho_amu_cm3LUT.MapControlPointsToLogSpace()

    # Properties modified on rho_amu_cm3LUT
    rho_amu_cm3LUT.UseLogScale = 1

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    rho_amu_cm3LUT.ApplyPreset('Inferno (matplotlib)', True)
    # show color bar/color legend
    slice1Display.SetScalarBarVisibility(renderView, True)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1600, 1600)

    # camera placement for renderView
    renderView.CameraPosition = [123.8821359932328, 162.9578433260544, 24.207094682916125]
    renderView.CameraFocalPoint = [-56.49901724813269, -32.56582457808803, -12.159020395552512]
    renderView.CameraViewUp = [-0.06681529228180919, -0.12253235709025494, 0.9902128751855344]
    renderView.CameraParallelScale = 218.09415971089186

def setup_pipeline(infile,**kwargs):
    """Function takes single data file and builds pipeline to find and
        visualize magnetopause
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
        kwargs:
            get_gradP (bool)- default false, will add additional variable
    Returns
        source (pvpython object)- python object attached to the VTKobject
                                  for the input data
        pipelinehead (pypython filter)- top level filter which starts the
                                        pipeline processing
        field (source/filter)- where the dataset has finished creating new
                               variables
        mp (source/filter)- final version of magnetopause
    """
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Read input file
    sourcedata = read_tecplot(infile)

    # apply 'Merge Blocks' so 'Connectivity' can be used
    mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1',
                               Input=sourcedata)

    ##Set the head of the pipeline, we will want to return this!
    pipelinehead = mergeBlocks1
    pipeline = mergeBlocks1

    ###Get Vectors from field variable components
    pipeline = get_vectors(pipeline)

    ###Build functions up to betastar
    #TODO: generalize similar to tecplot equations fnc in stream_tools.py
    # Dynamic Pressure
    pdyn = Calculator(registrationName='Pdyn', Input=pipeline)
    pdyn.Function = '"Rho_amu_cm^3"*1.6605*10^-6*mag(U_km_s)^2'
    pdyn.ResultArrayName = 'Pdyn_nPa'
    pipeline = pdyn

    # Beta*
    betastar = Calculator(registrationName='betastar', Input=pipeline)
    betastar.Function = '(P_nPa+Pdyn_nPa)/mag(B_nT)^2*8*3.14159*100'
    betastar.ResultArrayName = 'betastar'
    pipeline = betastar


    ###Programmable filters
    # Pressure gradient, optional variable
    if kwargs.get('doGradP',False):
        pipeline = get_pressure_gradient(pipeline)

    # Magnetopause
    pipeline = get_magnetopause_filter(pipeline)

    ###Now that last variable is done set 'field' for visualizer and a View
    field = pipeline

    ###Contour (iso-surface) of the magnetopause
    mp = create_iso_surface(pipeline, 'mp_state', 'mp')
    """How to calculate fluxes at surface use 'PerpendicularScale'
    # create a new 'Surface Vectors'
    surfaceVectors1 = SurfaceVectors(registrationName='SurfaceVectors1', Input=contour1)
    surfaceVectors1.SelectInputVectors = ['POINTS', '"J_`mA_m^2"']

    # Properties modified on surfaceVectors1
    surfaceVectors1.SelectInputVectors = ['POINTS', 'GradP_nPa_Re']
    surfaceVectors1.ConstraintMode = 'Perpendicular'
    """

    return sourcedata, pipelinehead, field, mp

if __name__ == "__main__":
    start_time = time.time()
    path = '/home/aubr/Code/swmf-energetics/febstorm/copy_paraview_plt/'
    filelist = glob.glob(path+'*.plt')
    for infile in filelist[0:1]:
        print('processing '+infile.split('/')[-1]+'...')
        oldsource,pipelinehead,field,mp=setup_pipeline(infile,do_gradP=True)
        renderView = GetActiveViewOrCreate('RenderView')
        display_visuals(field,mp,renderView)

        # Render and save screenshot
        RenderAllViews()
        path = 'output_pv_magnetosphere/'
        SaveScreenshot(path+infile.split('/')[-1].split('.plt')[0]+'.png',
                       renderView)
    for infile in filelist[1::]:
        print('processing '+infile.split('/')[-1]+'...')
        #Read in new file unattached to current pipeline
        SetActiveSource(None)
        newsource = read_tecplot(infile)

        #Attach pipeline to the new source file and delete the old
        pipelinehead.Input = newsource
        Delete(oldsource)

        # Render and save screenshot
        RenderAllViews()
        SaveScreenshot(path+infile.split('/')[-1].split('.plt')[0]+'.png',
                       renderView)

        # Set the current source to be replaced on next loop
        oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
