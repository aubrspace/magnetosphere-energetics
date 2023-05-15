import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

import os
import time
import glob
import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *
import magnetometer
from magnetometer import(get_stations_now, read_station_paraview)
import pv_input_tools
import pv_equations
import pv_surface_tools
#import (equations, eqeval, get_dipole_field,
#                          create_iso_surface)

def get_vectors(pipeline,**kwargs):
    """Function sets up calculator filters to turn components into vector
        objects (this will allow field tracing and other things)
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            vector_comps (dict)- default empty, will try to detect some
    Return
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    ###Get Vectors from field variable components
    vector_comps = kwargs.get('vector_comps',{})
    #Dig for the variable names so all variables can be vectorized
    points = pipeline.PointData
    var_names = points.keys()
    #info = pipeline.GetPointDataInformation()
    #n_arr = info.GetNumberOfArrays()
    #var_names = ['']*n_arr
    #for i in range(0,n_arr):
    #    var_names[i] = info.GetArray(i).Name
    deconlist=dict([(v.split('_')[0],'_'+'_'.join(v.split('_')[2::]))
                          for v in var_names if('_x' in v or '_y' in v or
                                                             '_z' in v)])
    for (base,tail) in deconlist.items():
        if tail=='_': tail=''
        vector = Calculator(registrationName=base,Input=pipeline)
        vector.Function = [base+'_x'+tail+'*iHat+'+
                           base+'_y'+tail+'*jHat+'+
                           base+'_z'+tail+'*kHat'][0]
        vector.ResultArrayName = base+tail
        pipeline=vector
    return pipeline

def mltset(pipeline,nowtime,**kwargs):
    """Simple filter to get MAG LON given a time
    """
    ###Might not always use the same nomenclature for column headers
    maglon = kwargs.get('maglon','MAGLON')
    ###'MLT' shift based on MAG longitude
    mltshift = 'MAGLON*12/180'
    ###'MLT' shift based on local time
    strtime = str(nowtime.hour+nowtime.minute/60+nowtime.second/3600)
    func = 'mod('+mltshift+'+'+strtime+',24)*180/12'
    lonNow = Calculator(registrationName='LONnow',Input=pipeline)
    lonNow.Function = func
    lonNow.ResultArrayName = 'LONnow'
    return lonNow

def rMag(pipeline,**kwargs):
    """Simple filter to set xyz points from table of LON/LAT
    """
    ###Might not always use the same nomenclature for column headers
    lon = kwargs.get('lon','LONnow')#accounts for local time shift
    lat = kwargs.get('maglat','MAGLAT')
    radius = kwargs.get('r',1)
    d2r = str(np.pi/180)+'*'
    x_func = str(radius)+'* cos('+d2r+lat+') * cos('+d2r+lon+')'
    y_func = str(radius)+'* cos('+d2r+lat+') * sin('+d2r+lon+')'
    z_func = str(radius)+'* sin('+d2r+lat+')'
    ###X
    x = Calculator(registrationName='xMag',Input=pipeline)
    x.Function = x_func
    #x.AttributeType = 'Point Data'
    x.ResultArrayName = 'xMag'
    ###Y
    y = Calculator(registrationName='yMag',Input=x)
    y.Function = y_func
    #y.AttributeType = 'Point Data'
    y.ResultArrayName = 'yMag'
    ###Z
    z = Calculator(registrationName='zMag',Input=y)
    z.Function = z_func
    #z.AttributeType = 'Point Data'
    z.ResultArrayName = 'zMag'
    ###Table2Points filter because I don't know how to access 'Table' values
    #   directly w/ the programmable filters
    points = TableToPoints(registrationName='TableToPoints', Input=z)
    points.KeepAllDataArrays = 1
    points.XColumn = 'xMag'
    points.YColumn = 'yMag'
    points.ZColumn = 'zMag'
    return points

def rotate2GSM(pipeline,tilt,**kwargs):
    """Function rotates MAG points to GSM points based on the time
    Inputs
        pipeline
        tilt
        kwargs:
    Return
        pipeline
    """
    rotationFilter = ProgrammableFilter(registrationName='rotate2GSM',
                                       Input=pipeline)
    rotationFilter.Script = update_rotation(tilt)
    ###Probably extraenous calculator to export the xyz to the actual
    #   coordinate values bc I don't know how to do that in the progfilt
    rGSM = Calculator(registrationName='stations',Input=rotationFilter)
    rGSM.Function = 'x*iHat+y*jHat+z*kHat'
    rGSM.AttributeType = 'Point Data'
    rGSM.ResultArrayName = 'rGSM'
    rGSM.CoordinateResults = 1
    return rGSM

def update_rotation(tilt):
    return"""
    import numpy as np
    data = inputs[0]
    angle = """+str(-tilt*np.pi/180)+"""
    x_mag = data.Points[:,0]
    y_mag = data.Points[:,1]
    z_mag = data.Points[:,2]

    rot = [[ np.cos(angle), 0, np.sin(angle)],
           [0,              1,             0],
           [-np.sin(angle), 0, np.cos(angle)]]

    x,y,z = np.matmul(rot,[x_mag,y_mag,z_mag])
    output.PointData.append(x,'x')
    output.PointData.append(y,'y')
    output.PointData.append(z,'z')"""

def magPoints2Gsm(pipeline,localtime,tilt,**kwargs):
    """Function creates a run of filters to convert a table of MAG coord vals
    into 3D points in GSM space
    Inputs
        pipeline
        kwargs:
            maglon
            maglat
    Returns
        pipeline
        time_hooks
    """
    ###Set the Longitude in MAG coordinates based on the time
    #mltMag = mltset(pipeline, localtime, **kwargs)
    ###Create x,y,z values based on R=1Re w/ these mag values
    #magPoints = rMag(mltMag, **kwargs)
    ###Rotation about the dipole axis generating new xyz values
    gsmPoints = rotate2GSM(pipeline, tilt, **kwargs)
    return gsmPoints

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

def get_ffj_filter1(pipeline,**kwargs):
    """Function to calculate the 'fourfieldjunction' to indicate a rxn site
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            status_opts (list)- default is 0-sw, 1-n, 2-s, 3-closed
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    #Must have the following conditions met first
    ffj =ProgrammableFilter(registrationName='ffj1',Input=pipeline)
    ffj.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]
    status = data.PointData['Status']
    m1 = (status==0).astype(int)
    m2 = (status==1).astype(int)
    m3 = (status==2).astype(int)
    m4 = (status==3).astype(int)
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(m1,'m1')
    output.PointData.append(m2,'m2')
    output.PointData.append(m3,'m3')
    output.PointData.append(m4,'m4')
    """
    pipeline = ffj
    return pipeline

def get_ffj_filter2(pipeline,**kwargs):
    #Must have the following conditions met first
    ffj =ProgrammableFilter(registrationName='ffj2',Input=pipeline)
    ffj.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]

    m1cc = data.CellData['m1']
    m2cc = data.CellData['m2']
    m3cc = data.CellData['m3']
    m4cc = data.CellData['m4']

    ffj = ((m1cc>0)&
           (m2cc>0)&
           (m3cc>0)&
           (m4cc>0)).astype(int)
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.CellData.append(ffj,'ffj_state')
    """
    pipeline = ffj
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
    assert FindSource('beta_star') != None
    betastar_max = kwargs.get('betastar_max',0.7)
    closed_value = kwargs.get('status_closed',3)
    mp_state =ProgrammableFilter(registrationName='mp_state',Input=pipeline)
    mp_state.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]
    beta_star = data.PointData['beta_star']
    status = data.PointData['Status']
    x = data.PointData['x']

    #Compute magnetopause as logical combination
    mp_state = ((status>=1)&(beta_star<"""+str(betastar_max)+""")&(x>-20)
                |
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

def point2cell(inputsource, fluxes):
    point2cell = PointDatatoCellData(registrationName='surface_p2cc',
                                     Input=inputsource)
    point2cell.ProcessAllArrays = 0
    convert_list = [f[0] for f in fluxes]
    convert_list.append('Normals')
    point2cell.PointDataArraytoprocess = convert_list
    return point2cell

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

def setup_table(**kwargs):
    """Function sets up a table (spreadsheet view) so data can be exported
    Inputs
        kwargs
            layout_name
            view_name
    Returns
        tableLayout
        tableView
    """
    # create new layout object 'Layout #2'
    tableLayout=CreateLayout(name=kwargs.get('layout_name','tableLayout'))
    # set active view
    SetActiveView(None)

    # Create a new 'SpreadSheet View'
    tableView = CreateView('SpreadSheetView')
    tableView.ColumnToSort = ''
    tableView.BlockSize = 1024

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=tableView, layout=tableLayout, hint=0)
    return tableLayout, tableView

def save_table_data(source, view, path, table_name):
    """Function exports tabular data from a given source to output file
    Inputs
        source
        view
        path
        table_name
    Returns
        None
    """
    # show data in view
    display = Show(source, view, 'SpreadSheetRepresentation')
    # export view
    ExportView(path+table_name+'.csv', view=view)

def export_datacube(pipeline,**kwargs):
    if 'path' not in kwargs:
        kwargs['path'] = '/home/aubr/Code/swmf-energetics/localdbug/fte/'
    datacubeFilter = ProgrammableFilter(registrationName='datacube',
                                        Input=pipeline)
    datacubeFilter.Script = update_datacube(**kwargs)
    return datacubeFilter

def update_datacube(**kwargs):
    return """
    # Get input
    data = inputs[0]
    p = data.PointData['P_nPa']
    rho = data.PointData['Rho_amu_cm3']
    bx = data.PointData['B_x_nT']
    by = data.PointData['B_y_nT']
    bz = data.PointData['B_z_nT']
    ux = data.PointData['U_x_km_s']
    uy = data.PointData['U_y_km_s']
    uz = data.PointData['U_z_km_s']
    status = data.PointData['Status']
    pdyn = data.PointData['Dp_nPa']
    bs = data.PointData['beta_star']
    mp = data.PointData['mp_state']
    ffj = data.PointData['ffj_state']

    # Get data statistic info
    extents = data.GetExtent()
    bounds = data.GetBounds()
    # format: [nx0,nxlast, ny0, nylast, ...]
    shape_xyz = [extents[1]+1,
                 extents[3]+1,
                 extents[5]+1]
    # Reshape coordinates based on range/extent
    X = numpy.linspace(bounds[0],bounds[1],shape_xyz[0])
    Y = numpy.linspace(bounds[2],bounds[3],shape_xyz[1])
    Z = numpy.linspace(bounds[4],bounds[5],shape_xyz[2])
    P = numpy.reshape(p,shape_xyz)
    RHO = numpy.reshape(rho,shape_xyz)
    BX = numpy.reshape(bx,shape_xyz)
    BY = numpy.reshape(by,shape_xyz)
    BZ = numpy.reshape(bz,shape_xyz)
    UX = numpy.reshape(ux,shape_xyz)
    UY = numpy.reshape(uy,shape_xyz)
    UZ = numpy.reshape(uz,shape_xyz)
    STATUS = numpy.reshape(status,shape_xyz)
    PDYN = numpy.reshape(pdyn,shape_xyz)
    BS = numpy.reshape(bs,shape_xyz)
    MP = numpy.reshape(mp,shape_xyz)
    FFJ = numpy.reshape(ffj,shape_xyz)

    # Set output file
    outpath = '"""+kwargs.get('path','')+"""'
    outname = '"""+kwargs.get('filename','test_cube.npz')+"""'

    # Save data
    numpy.savez(outpath+outname,x=X,y=Y,z=Z,
                                p=P,rho=rho,
                                bx=bx,by=by,bz=bz,
                                ux=ux,uy=uy,uz=uz,
                                status=status,
                                pdyn=pdyn,
                                betastar=BS,mp=MP,ffj=FFJ,
                                dims=shape_xyz)
                    """
def load_fte(pipeline,**kwargs):
    filtername = 'fte_state'
    # Usually we have Point Data so if reading Cell Data need to filter
    if kwargs.get('isCellData',True):
        xyzCell = PointDatatoCellData(registrationName='xyzCell',
                                      Input=pipeline)
        xyzCell.ProcessAllArrays = 0
        xyzCell.PointDataArraytoprocess = ['x', 'y', 'z']
        xyzCell.PassPointData = 1
        filtername = 'fteCell'
        pipeline = xyzCell

    # Progammable filter will read things in *hopefully* fast enough
    if 'file' not in kwargs:
        kwargs['file'] = '/home/aubr/Downloads/FTE020500.npz'
    fteFilter=ProgrammableFilter(registrationName=filtername,Input=pipeline)
    fteFilter.Script = update_fte(**kwargs)
    pipeline = fteFilter

    # Now change that Cell Data back to Point Data
    if kwargs.get('isCellData',True):
        ftePoint = CellDatatoPointData(registrationName='fte_state',
                                       Input=fteFilter)
        ftePoint.ProcessAllArrays = 0
        ftePoint.CellDataArraytoprocess = ['fte']
        ftePoint.PassCellData = 0
        pipeline = ftePoint

    return pipeline

def update_fte(**kwargs):
    return"""
        data = inputs[0]
        #Read .npz file
        fte = numpy.load('"""+kwargs.get('file')+"""')

        #Reconfigure XYZ data
        X = data.CellData['x']
        Y = data.CellData['y']
        Z = data.CellData['z']

        #Create new variable
        fte_indata = numpy.zeros(len(X))

        source_arr = numpy.zeros((len(X),3))
        source_arr[:,0] = X
        source_arr[:,1] = Y
        source_arr[:,2] = Z
        #Isolate just the points within the bounds of the fte_extent
        town = numpy.where((X<=fte['ftex'].max())&
                           (X>=fte['ftex'].min())&
                           (Y<=fte['ftey'].max())&
                           (Y>=fte['ftey'].min())&
                           (Z<=fte['ftez'].max())&
                           (Z>=fte['ftez'].min()))

        village = town[0][::]
        #Change all the matching points to have value=1
        Xunique = numpy.unique(X[village])
        Yunique = numpy.unique(Y[village])
        for Xi in Xunique:
            if Xi in fte['ftex']:
                for Yi in Yunique:
                    if Yi in fte['ftey']:
                        fte_sub=fte['ftez'][numpy.where((fte['ftex']==Xi)&
                                                       (fte['ftey']==Yi))[0]]
                        if len(fte_sub)>0:
                            extrusion = numpy.where((X[village]==Xi)&
                                                    (Y[village]==Yi))[0]
                            Z_extrusion = Z[village[extrusion]]
                            street = numpy.where((Z_extrusion<fte_sub.max())&
                                                 (Z_extrusion>fte_sub.min()))
                            for address in street[0]:
                                if Z_extrusion[address] in fte_sub:
                                    fte_indata[village[extrusion[address]]]=1

        #Copy input to output
        output.ShallowCopy(inputs[0].VTKObject)

        #Add our new array
        output.CellData.append(fte_indata,'fte')
    """

def display_visuals(field,mp,renderView,**kwargs):
    """Function standin for separate file governing visual representations
    Inputs
        field (source/filter)- data w all variables for streams, slices etc.
        mp (source/filter)- finalized magnetopause data
        renderView (View)- where to show things
        kwargs:
            mpContourBy
            contourMin,contourMax
            cmap
            doSlice
            sliceContourBy
            sliceContourLog
            doFieldLines
    returns
        TBD
    """
    # show outline of field
    #setup_outline(field)
    # show iso surfaces
    if kwargs.get('show_mp', True):
        mpDisplay = Show(mp, renderView, 'GeometryRepresentation')

        if 'mpContourBy' in kwargs:
            # set scalar coloring
            ColorBy(mpDisplay, ('POINTS', kwargs.get('mpContourBy')))
            # get color & opacity transfer functions'
            mpLUT = GetColorTransferFunction(kwargs.get('mpContourBy'))
            mpPWF = GetOpacityTransferFunction(kwargs.get('mpContourBy'))
            # Set limits, default both equal
            mpLUT.RescaleTransferFunction(kwargs.get('contourMin',-10),
                                        kwargs.get('contourMax',10))
            mpPWF.RescaleTransferFunction(kwargs.get('contourMin',-10),
                                        kwargs.get('contourMax',10))
            # Apply a preset using its name. Note this may not work as expected
            #   when presets have duplicate names.
            mpLUT.ApplyPreset(kwargs.get('cmap','Cool to Warm (Extended)'),
                            True)
            # Show contour legend
            mpDisplay.SetScalarBarVisibility(renderView,True)
            mpLUTColorBar = GetScalarBar(mpLUT, renderView)
            mpLUTColorBar.WindowLocation = 'Upper Right Corner'

        else:
            # change solid color
            ColorBy(mpDisplay, None)
            #mpDisplay.AmbientColor = [0.0, 1.0, 1.0]
            #mpDisplay.DiffuseColor = [0.0, 1.0, 1.0]
            mpDisplay.AmbientColor = [0.5764705882352941, 0.5764705882352941, 0.5764705882352941]
            mpDisplay.DiffuseColor = [0.5764705882352941, 0.5764705882352941, 0.5764705882352941]

        # Properties modified on mpDisplay.DataAxesGrid
        mpDisplay.DataAxesGrid.GridAxesVisibility = 1
        mpDisplay.DataAxesGrid.XTitleFontSize = 20
        mpDisplay.DataAxesGrid.YTitleFontSize = 20
        mpDisplay.DataAxesGrid.ZTitleFontSize = 20
        mpDisplay.DataAxesGrid.XTitle = '      X Axis'
        mpDisplay.DataAxesGrid.ZTitle = '  Z Axis'

        # Properties modified on mpDisplay.DataAxesGrid
        mpDisplay.DataAxesGrid.XLabelFontSize = 15
        mpDisplay.DataAxesGrid.YLabelFontSize = 15
        mpDisplay.DataAxesGrid.ZLabelFontSize = 15
        mpDisplay.DataAxesGrid.ShowGrid = 0
        '''
        mpDisplay.DataAxesGrid.XAxisUseCustomLabels = 1
        mpDisplay.DataAxesGrid.YAxisUseCustomLabels = 1
        mpDisplay.DataAxesGrid.ZAxisUseCustomLabels = 1
        mpDisplay.DataAxesGrid.XAxisLabels = np.linspace(10,-30,9)
        mpDisplay.DataAxesGrid.YAxisLabels = np.linspace(20,-20,9)
        mpDisplay.DataAxesGrid.ZAxisLabels = np.linspace(20,-20,9)
        '''

        # Properties modified on slice1Display
        mpDisplay.Opacity = 1
        #mpDisplay.Opacity = 0.3

    if kwargs.get('show_fte', False):
        fteDisplay = Show(FindSource('fte'), renderView,
                          'GeometryRepresentation')
        if 'fteContourBy' in kwargs:
            # set scalar coloring
            ColorBy(fteDisplay, ('POINTS', kwargs.get('fteContourBy')))
            # get color & opacity transfer functions'
            fteLUT = GetColorTransferFunction(kwargs.get('fteContourBy'))
            ftePWF = GetOpacityTransferFunction(kwargs.get('fteContourBy'))
            # Set limits, default both equal
            fteLUT.RescaleTransferFunction(kwargs.get('ftecontourMin',-10),
                                        kwargs.get('ftecontourMax',10))
            ftePWF.RescaleTransferFunction(kwargs.get('ftecontourMin',-10),
                                        kwargs.get('ftecontourMax',10))
            # Apply a preset using its name. Note this may not work as expected
            #   when presets have duplicate names.
            fteLUT.ApplyPreset(kwargs.get('cmap','Cool to Warm (Extended)'),
                               True)
            # Show contour legend
            fteDisplay.SetScalarBarVisibility(renderView,True)
            fteLUTColorBar = GetScalarBar(fteLUT, renderView)
            fteLUTColorBar.WindowLocation = 'Lower Right Corner'


    if kwargs.get('doFFJ',False):
        isoFFJ = Contour(registrationName='FFJ', Input=field)
        isoFFJ.ContourBy = ['POINTS', 'ffj_state']
        isoFFJ.ComputeNormals = 0
        isoFFJ.Isosurfaces = [0.1]
        isoFFJ.PointMergeMethod = 'Uniform Binning'
        isoFFJdisplay = Show(isoFFJ, renderView, 'GeometryRepresentation')
        ColorBy(isoFFJdisplay, None)
        isoFFJdisplay.AmbientColor = [0.0, 1.0, 0.0]
        isoFFJdisplay.DiffuseColor = [0.0, 1.0, 0.0]
        isoFFJdisplay.Opacity = 0.4

    if kwargs.get('timestamp',True):
        simtime = kwargs.get('localtime')-kwargs.get('tstart')
        #timestamp
        stamp1 = Text(registrationName='tstamp')
        stamp1.Text = str(kwargs.get('localtime'))
        stamp1Display = Show(stamp1,renderView,
                                  'TextSourceRepresentation')
        stamp1Display.FontSize = kwargs.get('fontsize')
        stamp1Display.WindowLocation = 'Any Location'
        stamp1Display.Position = [0.845, 0.20]
        stamp1Display.Color = [0.652, 0.652, 0.652]
        #simulation runtime
        stamp2 = Text(registrationName='tsim')
        stamp2.Text = 'tsim: '+str(simtime)
        stamp2Display = Show(stamp2,renderView,
                                  'TextSourceRepresentation')
        stamp2Display.FontSize = kwargs.get('fontsize')
        stamp2Display.WindowLocation = 'Any Location'
        stamp2Display.Position = [0.845, 0.16]
        stamp2Display.Color = [0.652, 0.652, 0.652]

    if 'n' in kwargs and kwargs.get('station_tag',False):
        #Tag station results to the page
        station_tag = Text(registrationName='station_tag')
        station_tag.Text = '# of Stations: '
        station_tagDisplay = Show(station_tag,renderView,
                                  'TextSourceRepresentation')
        station_tagDisplay.FontSize = kwargs.get('fontsize')
        station_tagDisplay.WindowLocation = 'Any Location'
        if kwargs.get('doFluxVol',False):
            station_tagDisplay.Position = [0.01, 0.95555553]
        else:
            station_tagDisplay.Position = [0.86, 0.95555553]

        #Tag station results to the page
        station_num = Text(registrationName='station_num')
        station_num.Text = str(kwargs.get('n',379))
        station_numDisplay = Show(station_num,renderView,
                                  'TextSourceRepresentation')
        station_numDisplay.FontSize = kwargs.get('fontsize')
        station_numDisplay.WindowLocation = 'Any Location'
        if kwargs.get('doFluxVol',False):
            station_numDisplay.Position = [0.11, 0.95555553]
            station_numDisplay.Color = [0.0925, 1.0, 0.0112]
        else:
            station_numDisplay.Position = [0.963, 0.95555553]
            station_numDisplay.Color = [0.096, 0.903, 0.977]


    if kwargs.get('doFluxVol',False):
        results = kwargs.get('fluxResults')
        ####Tag volume header to the page
        vol_tag = Text(registrationName='volume_tag')
        vol_tag.Text = 'Volume :'
        vol_tagDisplay = Show(vol_tag,renderView,
                                  'TextSourceRepresentation')
        vol_tagDisplay.FontSize = kwargs.get('fontsize')
        vol_tagDisplay.WindowLocation = 'Any Location'
        vol_tagDisplay.Position = [0.85, 0.95555553]
        #Tag volume results to the page
        vol_num = Text(registrationName='volume_num')
        vol_num.Text = '{:.2f}%'.format(results['flux_volume']/
                                        results['total_volume']*100)
        vol_numDisplay = Show(vol_num,renderView,
                              'TextSourceRepresentation')
        vol_numDisplay.WindowLocation = 'Any Location'
        vol_numDisplay.Position = [0.92, 0.95555553]
        vol_numDisplay.FontSize = kwargs.get('fontsize')
        vol_numDisplay.Color = [0.0383, 1.0, 0.0279]

        #####Tag Bflux header to the page
        bflux_tag = Text(registrationName='bflux_tag')
        bflux_tag.Text = 'MagEnergy :'
        bflux_tagDisplay = Show(bflux_tag,renderView,
                                  'TextSourceRepresentation')
        bflux_tagDisplay.FontSize = kwargs.get('fontsize')
        bflux_tagDisplay.WindowLocation = 'Any Location'
        bflux_tagDisplay.Position = [0.82, 0.90]
        #Tag Bflux results to the page
        bflux_num = Text(registrationName='bflux_num')
        bflux_num.Text = '{:.2f}%'.format(results['flux_Umag']/
                                        results['total_Umag']*100)
        bflux_numDisplay = Show(bflux_num,renderView,
                              'TextSourceRepresentation')
        bflux_numDisplay.WindowLocation = 'Any Location'
        bflux_numDisplay.Position = [0.92, 0.90]
        bflux_numDisplay.FontSize = kwargs.get('fontsize')
        bflux_numDisplay.Color = [0.0383, 1.0, 0.0279]

        #####Tag diff Bflux header to the page
        dbflux_tag = Text(registrationName='dbflux_tag')
        dbflux_tag.Text = 'Disturbance MagEnergy :'
        dbflux_tagDisplay = Show(dbflux_tag,renderView,
                                  'TextSourceRepresentation')
        dbflux_tagDisplay.FontSize = kwargs.get('fontsize')
        dbflux_tagDisplay.WindowLocation = 'Any Location'
        dbflux_tagDisplay.Position = [0.725, 0.85555553]
        #Tag diff Bflux results to the page
        dbflux_num = Text(registrationName='dbflux_num')
        dbflux_num.Text = '{:.2f}%'.format(results['flux_Udb']/
                                        results['total_Udb']*100)
        dbflux_numDisplay = Show(dbflux_num,renderView,
                              'TextSourceRepresentation')
        dbflux_numDisplay.WindowLocation = 'Any Location'
        dbflux_numDisplay.Position = [0.92, 0.85555553]
        dbflux_numDisplay.FontSize = kwargs.get('fontsize')
        dbflux_numDisplay.Color = [0.0383, 1.0, 0.0279]

    if kwargs.get('doSlice',False):
        ###Slice
        # create a new 'Slice'
        slice1 = Slice(registrationName='Slice1', Input=field)
        if 'Plane' in kwargs.get('slicetype','yPlane'):
            slice1.SliceType = 'Plane'
            slice1.HyperTreeGridSlicer = 'Plane'
            slice1.SliceOffsetValues = [0.0]

            if 'y' in kwargs.get('slicetype','yPlane'):
                # init the 'Plane' selected for 'SliceType'
                slice1.SliceType.Origin = [0.0, 0.0, 0.0]
                slice1.SliceType.Normal = [0.0, 1.0, 0.0]

                # init the 'Plane' selected for 'HyperTreeGridSlicer'
                slice1.HyperTreeGridSlicer.Origin = [-94.25, 0.0, 0.0]
            elif 'z' in kwargs.get('slicetype','yPlane'):
                # init the 'Plane' selected for 'SliceType'
                slice1.SliceType.Origin = [0.0, 0.0, 0.0]
                slice1.SliceType.Normal = [0.0, 0.0, 1.0]

                # init the 'Plane' selected for 'HyperTreeGridSlicer'
                slice1.HyperTreeGridSlicer.Origin = [-94.25, 0.0, 0.0]
                #TODO
                pass
        # show data in view
        slice1Display = Show(slice1, renderView, 'GeometryRepresentation')
        # Properties modified on slice1Display
        slice1Display.Opacity = 1
        # set scalar coloring
        ColorBy(slice1Display, ('POINTS', 'u_db_J_Re3'))
        # get color transfer function/color map for 'Rho_amu_cm3'
        bzLUT = GetColorTransferFunction('u_db_J_Re3')
        #bzLUT.RescaleTransferFunction(-10.0, 10.0)
        #bzLUT.ApplyPreset('Gray and Red', True)
        #bzLUT.InvertTransferFunction()

        # get opacity transfer function/opacity map for 'Rho_amu_cm3'
        #bzPWF = GetOpacityTransferFunction('B_z_nT')


        # convert to log space
        #bzLUT.MapControlPointsToLogSpace()

        # Properties modified on rho_amu_cm3LUT
        #rho_amu_cm3LUT.UseLogScale = 1

        # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
        #bzLUT.ApplyPreset('Inferno (matplotlib)', True)
        # show color bar/color legend
        slice1Display.SetScalarBarVisibility(renderView, True)

    '''
    # camera placement for renderView
    renderView.CameraPosition = [123.8821359932328, 162.9578433260544, 24.207094682916125]
    renderView.CameraFocalPoint = [-56.49901724813269, -32.56582457808803, -12.159020395552512]
    renderView.CameraViewUp = [-0.06681529228180919, -0.12253235709025494, 0.9902128751855344]
    renderView.CameraParallelScale = 218.09415971089186
    '''
    # Edit background properties
    renderView.UseColorPaletteForBackground = 0
    renderView.Background = [0.0, 0.0, 0.0]

    # Zoomed dayside +Y side magnetosphere
    '''
    renderView.CameraPosition = [33.10, 56.51, 14.49]
    renderView.CameraFocalPoint = [-33.19, -15.11, -3.46]
    renderView.CameraViewUp = [-0.10, -0.15, 0.98]
    renderView.CameraParallelScale = 66.62
    '''
    if False:
        # Rotating earth sciVis panel 1
        '''
        renderView.CameraPosition = [29.32, 37.86, 7.61]
        renderView.CameraFocalPoint = [-25.45, -21.32, -7.22]
        renderView.CameraViewUp = [-0.100, -0.150, 0.983]
        renderView.CameraParallelScale = 66.62
        '''
        renderView.CameraPosition = [32.565256893942504, 33.961411032755066, 11.182203304141801]
        renderView.CameraFocalPoint = [-22.2047431060575, -25.218588967244937, -3.6477966958581973]
        renderView.CameraViewUp = [-0.10006060505009504, -0.15009090757514254, 0.9835957476424342]
        renderView.CameraParallelScale = 66.62
        '''
        renderView.CameraPosition = [57.984013893942524, 61.426849032755115, 18.064806304141808]
        renderView.CameraFocalPoint = [-22.2047431060575, -25.218588967244937, -3.6477966958581973]
        renderView.CameraViewUp = [-0.10006060505009504, -0.15009090757514254, 0.9835957476424342]
        renderView.CameraParallelScale = 66.62
        '''
    elif False:
        # Rotating earth sciVis panel 1 ZOOMED for topleft screen
        renderView.CameraPosition = [2.9511799712866575, 3.374731655415902, 0.7958056194894831]
        renderView.CameraFocalPoint = [-22.599429242800465, -24.2331751053961, -6.122498829020866]
        renderView.CameraViewUp = [-0.10006060505009504, -0.15009090757514254, 0.9835957476424342]
        renderView.CameraParallelScale = 66.62

    elif False:
        # Flux increasing sciVis panel 3
        renderView.CameraPosition = [-70.58912364537356, -15.750308500254196, 48.517414160762996]
        renderView.CameraFocalPoint = [17.05613736727104, 12.94876210057961, -28.3603263872939]
        renderView.CameraViewUp = [0.5899444617072703, 0.25266438976703015, 0.7668938898208627]
        renderView.CameraParallelScale = 66.62

    elif True:
        # Head on looking for evidence of FTE's and the FFJ pattern
        renderView.CameraPosition = [54.575688611894364, -0.03966433288374481, 3.952625747389735]
        renderView.CameraFocalPoint = [47.90558157105895, 0.33132763449067765, 3.5946634254149736]
        renderView.CameraViewUp = [-0.05356990972544573, 0.0003538349672490677, 0.9985640387941195]

def add_fluxVolume(field,**kwargs):
    """Function adds field lines to current view
    Inputs
        field- source to find the flux volumes
        kwargs:
            station_file
            localtime
            tilt
    Returns
        None
    """
    #Integrate total
    total_int=IntegrateVariables(registrationName='totalInt',Input=field)

    fluxVolume_hits=ProgrammableFilter(registrationName='fluxVolume_hits',
                                       Input=field)
    fluxVolume_hits.Script=update_fluxVolume(**kwargs)
    # create a new 'Threshold'
    fluxVolume = Threshold(registrationName='fluxVolume',
                           Input=fluxVolume_hits)
    fluxVolume.Scalars = ['POINTS', 'projectedVol']
    fluxVolume.UpperThreshold = 0.95
    fluxVolume.ThresholdMethod = 'Above Upper Threshold'
    #Set a view
    view = GetActiveViewOrCreate('RenderView')
    #Adjust view settings
    volumeDisplay = Show(fluxVolume, view, 'GeometryRepresentation')
    volumeDisplay.AmbientColor = [0.33, 1.0, 0.0]
    volumeDisplay.DiffuseColor = [0.33, 1.0, 0.0]
    volumeDisplay.Opacity = 0.20
    ColorBy(volumeDisplay, None)
    #Integrate projected flux
    flux_int=IntegrateVariables(registrationName='fluxInt',
                                Input=fluxVolume)
    #Grab what you like
    vals = update_fluxResults(flux_int,total_int)
    return fluxVolume, vals

def update_fluxResults(flux_int,total_int):
    #Grab what you like
    vals = {}
    vals['flux_volume'] = flux_int.CellData['Volume'].GetRange()[0]
    vals['flux_Umag'] = flux_int.PointData['uB_J_Re3'].GetRange()[0]
    vals['flux_Udb'] = flux_int.PointData['u_db_J_Re3'].GetRange()[0]
    vals['total_volume'] = total_int.CellData['Volume'].GetRange()[0]
    vals['total_Umag'] = total_int.PointData['uB_J_Re3'].GetRange()[0]
    vals['total_Udb'] = total_int.PointData['u_db_J_Re3'].GetRange()[0]
    return vals

def update_fluxVolume(**kwargs):
    nowtime = kwargs.get('localtime')
    tshift = str(nowtime.hour+nowtime.minute/60+nowtime.second/3600)
    return """
    from vtk.numpy_interface import algorithms as algs
    from vtk.numpy_interface import dataset_adapter as dsa
    import numpy as np

    n = """+str(kwargs.get('n',379))+"""
    data = inputs[0]
    beta_star = data.PointData['beta_star']
    status = data.PointData['Status']
    x = data.PointData['x']
    th1 = data.PointData['theta_1']
    th2 = data.PointData['theta_2']
    ph1 = data.PointData['phi_1']
    ph2 = data.PointData['phi_2']

    # assuming stations.csv is a CSV file with the 1st row being
    # the names names for the columns
    stations = np.genfromtxt('"""+os.path.join(kwargs.get('path',''),
                            kwargs.get('file_in','stations.csv'))+"""',
                             dtype=None, names=True,
                             delimiter=',', autostrip=True)
    tshift = """+tshift+"""
    hits = th1*0
    #for ID,_,__,lat,lon in [stations[282]]:
    #for ID,_,__,lat,lon in [stations[39]]:
    #for ID,_,__,lat,lon in [stations[26],stations[27]]:
    for ID,_,__,lat,lon in stations[0:n]:
        #print(ID)
        lon = ((lon*12/180)+tshift)%24*180/12
        if lat>0:
            theta = th1
            phi = ph1
        else:
            theta = th2
            phi = ph2
        if isinstance(theta, vtk.numpy_interface.dataset_adapter.VTKNoneArray):
            lat_adjust = 1
            print(ID)
        else:
            lat_adjust = np.sqrt(abs(np.cos(theta/180*np.pi)))
            th_tol = 150 * (180/np.pi/6371) / lat_adjust
            phi_tol = 150 * (180/np.pi/6371) / lat_adjust
            footcond1=((theta<(lat+th_tol))&(theta>(lat-th_tol))).astype(int)
            footcond2 = ((phi>(lon-phi_tol))&(phi<(lon+phi_tol))).astype(int)
            #mp_state = ((beta_star<0.7)|
            #        (status==3)).astype(int)
            mp_state = ((status!=0)&(beta_star<0.7)&(x>-30)
                        |
                        (status==3)).astype(int)
            hits = np.maximum.reduce([hits, footcond1*footcond2*mp_state])

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(hits,'projectedVol')
    """

def add_fieldlines(head,**kwargs):
    """Function adds field lines to current view
    Inputs
        head- source to generatet the streamlines
        kwargs:
            station_file
            localtime
            tilt
    Returns
        None
    """
    '''
    #Blank inside the earth
    clip1 = Clip(registrationName='Clip1', Input=head)
    clip1.ClipType = 'Sphere'
    clip1.Invert = 0
    clip1.ClipType.Center = [0.0, 0.0, 0.0]
    clip1.ClipType.Radius = 0.99
    #Blank outside the magnetosphere (as in 0.7 beta*)
    clip2 = Clip(registrationName='Clip2', Input=clip1)
    clip2.ClipType = 'Scalar'
    clip2.Scalars = ['POINTS', 'beta_star']
    clip2.Value = 0.7
    '''
    #Set a view
    view = GetActiveViewOrCreate('RenderView')
    stations = FindSource('stations')
    # create a new 'Stream Tracer With Custom Source'
    trace = StreamTracerWithCustomSource(registrationName='station_field',
                                         Input=head, SeedSource=stations)
    trace.Vectors = ['POINTS', 'B_nT']
    trace.MaximumStreamlineLength = 252.0
    traceDisplay = Show(trace, view, 'GeometryRepresentation')
    traceDisplay.AmbientColor = [0.0, 0.66, 1.0]
    traceDisplay.DiffuseColor = [0.0, 0.66, 1.0]
    ColorBy(traceDisplay, None)

def setup_pipeline(infile,**kwargs):
    """Function takes single data file and builds pipeline to find and
        visualize magnetopause
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
        kwargs:
            aux (dict)- optional to include dictionary with other data
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
    sourcedata = pv_input_tools.read_tecplot(infile)

    # apply 'Merge Blocks' so 'Connectivity' can be used
    mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1',
                               Input=sourcedata)

    ##Set the head of the pipeline, we will want to return this!
    pipelinehead = mergeBlocks1
    pipeline = mergeBlocks1
    #pipelinehead = sourcedata
    #pipeline = sourcedata

    ###Check if unitless variables are present
    if 'dimensionless' in kwargs:
        pipeline = pv_input_tools.todimensional(pipeline,**kwargs)

    else:
        ###Rename some tricky variables
        pipeline = pv_input_tools.fix_names(pipeline,**kwargs)
    ###Build functions up to betastar
    alleq = pv_equations.equations(**kwargs)
    pipeline = pv_equations.eqeval(alleq['basic3d'],pipeline)
    pipeline = pv_equations.eqeval(alleq['basic_physics'],pipeline)
    if 'aux' in kwargs:
        pipeline = pv_equations.eqeval(alleq['dipole_coord'],pipeline)
    ###Energy flux variables
    if kwargs.get('doEnergyFlux',False):
        pipeline = pv_equations.eqeval(alleq['energy_flux'],pipeline)
    if kwargs.get('doVolumeEnergy',False):
        pipeline = pv_equations.eqeval(alleq['dipole'],pipeline)
        pipeline = pv_equations.eqeval(alleq['volume_energy'],pipeline)
    ###Get Vectors from field variable components
    pipeline = get_vectors(pipeline)

    ###Programmable filters
    # Pressure gradient, optional variable
    if kwargs.get('doGradP',False):
        pipeline = get_pressure_gradient(pipeline)
    if kwargs.get('ffj',False):
        ffj1 = get_ffj_filter1(pipeline)
        ffj2 = PointDatatoCellData(registrationName='ffj_interp1',
                                   Input=ffj1)
        ffj2.ProcessAllArrays = 1
        ffj3 = get_ffj_filter2(ffj2)
        #pipeline=ffj3
        ffj4 = CellDatatoPointData(registrationName='ffj_interp2',
                                   Input=ffj3)
        pipeline = ffj4

    ###Read satellite trajectories
    if kwargs.get('doSat',False):
        satfiles = kwargs.get('satfiles')
        for satin in satfiles:
            name = satin.split('.csv')[0]
            csv = CSVReader(registrationName=name+'_in',
                  FileName=os.path.join(kwargs.get('path'),satin))
            points = TableToPoints(registrationName=name,
                                   Input=csv)
            points.XColumn = 'x'
            points.YColumn = 'y'
            points.ZColumn = 'z'
            renderView = GetActiveViewOrCreate('RenderView')
            pointsDisplay=Show(points,renderView,'GeometryRepresentation')
            colors = {
                    'cl1':[0.9,0.9,0.9],
                    'cl2':[0.9,0.9,0.9],
                    'cl3':[0.9,0.9,0.9],
                    'cl4':[0.9,0.9,0.9],
                    'thA':[0.9,0.9,0.9],
                    'thB':[0.9,0.9,0.9],
                    'thC':[0.9,0.9,0.9],
                    'thD':[0.9,0.9,0.9],
                    'thE':[0.9,0.9,0.9],
                    'geo':[0.9,0.9,0.9],
                    'mms1':[0.9,0.9,0.9],
                    'mms2':[0.9,0.9,0.9],
                    'mms3':[0.9,0.9,0.9],
                    'mms4':[0.9,0.9,0.9]
                    }
    # FTE
    pipeline = load_fte(pipeline)

    # Magnetopause
    pipeline = get_magnetopause_filter(pipeline)

    ###Now that last variable is done set 'field' for visualizer and a View
    field = pipeline

    ###Contour (iso-surface) of the magnetopause
    mp = pv_surface_tools.create_iso_surface(pipeline, 'mp_state', 'mp')

    if kwargs.get('fte',False):
        ###Contour (iso-surface) of the magnetopause
        fte = pv_surface_tools.create_iso_surface(pipeline, 'fte', 'fte')

    ###Field line seeding or Field line projected flux volumes
    fluxResults = None
    if(kwargs.get('doFieldlines',False)or kwargs.get('doFluxVol',False)):
        station_MAG, success = read_station_paraview(
                                    kwargs.get('localtime'),
                                    n=kwargs.get('n',379),
                                    path=kwargs.get('path'))
        if success and 'localtime' in kwargs and 'tilt' in kwargs:
            stations = magPoints2Gsm(station_MAG,kwargs.get('localtime'),
                                     kwargs.get('tilt'))
            renderView = GetActiveViewOrCreate('RenderView')
            stationsDisplay = Show(stations, renderView,
                                   'GeometryRepresentation')
            stationsDisplay.AmbientColor = [1.0, 1.0, 0.0]
            stationsDisplay.DiffuseColor = [1.0, 1.0, 0.0]
            #Blank inside the earth
            clip1 = Clip(registrationName='Clip1', Input=pipeline)
            clip1.ClipType = 'Sphere'
            clip1.Invert = 0
            clip1.ClipType.Center = [0.0, 0.0, 0.0]
            clip1.ClipType.Radius = 0.99
            #Blank outside the magnetosphere (as in 0.7 beta*)
            clip2 = Clip(registrationName='Clip2', Input=clip1)
            clip2.ClipType = 'Scalar'
            clip2.Scalars = ['POINTS', 'mp_state']
            clip2.Value = 1
            clip2.Invert = 0
            #clip2=clip1
            if kwargs.get('doFieldlines',False):
                add_fieldlines(clip2)
            if kwargs.get('doFluxVol',False):
                clip1.ClipType.Radius = 2.5
                obj, fluxResults = add_fluxVolume(clip2,**kwargs)

    return sourcedata, pipelinehead, field, mp, fluxResults

if __name__ == "__main__":
#if True:
    #from pv_input_tools import time_sort, read_aux, read_tecplot
    start_time = time.time()
    ######################################################################
    # USER INPUTS
    ######################################################################
    #path='/Users/ngpdl/Code/swmf-energetics/localdbug/fte/'
    #path='/home/aubr/Code/swmf-energetics/localdbug/fte/'
    path='/nfs/solsticedisk/tuija/amr_fte/secondtry/GM/IO2/'
    outpath = 'output7_fte_pv/'
    #outpath = '/Users/ngpdl/Code/swmf-energetics/localdbug/fte/output5_fte_pv/'
    ######################################################################

    #Make the paths if they don't already exist
    os.makedirs(path, exist_ok=True)
    #os.makedirs(outpath, exist_ok=True)

    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=time_sort)
    #filelist = ['/home/aubr/Code/swmf-energetics/localdbug/fte/3d__paraview_1_e20140610-010000-000.plt']
    #filelist = ['/Users/ngpdl/Code/swmf-energetics/localdbug/febstorm/3d__paraview_1_e20140219-024500-000.plt']
    for infile in filelist[-1::]:
        print('processing '+infile.split('/')[-1]+'...')
        aux = read_aux(infile.replace('.plt','.aux'))
        oldsource,pipelinehead,field,mp=setup_pipeline(infile,aux=aux,
                                                       doEnergy=False)
        ###Surface flux on magnetopause
        get_surface_flux(mp, 'B_nT','Bnormal_net')
        mp_Bnorm = FindSource('Bnormal_net')
        #decide which values to calculate (will need to make cell data)
        #fluxes = [('K_W_Re2','k_flux'),('P0_W_Re2','h_flux'),
        #          ('ExB_W_Re2','p_flux')]
        #mp_cc = point2cell(mp,fluxes)#mp object with cell centered data
        #mp_K_flux = get_surface_flux(mp, 'K_W_Re2','k_flux')
        #mp_S_flux = get_surface_flux(mp_cc, 'ExB_W_Re2','s_net_flux')
        renderView1 = GetActiveViewOrCreate('RenderView')
        #TODO find how to limit integration variables and group all together
        #tableLayout, tableView = setup_table()
        #save_table_data(mp_S_flux, tableView, outpath,'s_net_flux')
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,
                        mpContourBy='B_x_nT',contourMin=-10,contourMax=10,
                        **kwargs)

        # Create a new 'Render View'
        layout = GetLayout()
        layout.SplitVertical(0, 0.5)
        renderView2 = CreateView('RenderView')
        # assign view to a particular cell in the layout
        AssignViewToLayout(view=renderView2, layout=layout, hint=2)
        display_visuals(field,mp_Bnorm,renderView2,doSlice=True,
                        mpContourBy='Bnormal_net',
                        contourMin=-10,contourMax=10,
                        cmap='Cool to Warm')

        # Render and save screenshot
        RenderAllViews()
        # layout/tab size in pixels
        layout.SetSize(2162, 1079)
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
    for infile in filelist[0:-1]:
        print('processing '+infile.split('/')[-1]+'...')
        outfile=outpath+infile.split('/')[-1].split('.plt')[0]+'.png'
        if os.path.exists(outfile):
            print(outfile+' already exists, skipping')
        else:
            #Read in new file unattached to current pipeline
            SetActiveSource(None)
            newsource = read_tecplot(infile)

            #Attach pipeline to the new source file and delete the old
            pipelinehead.Input = newsource
            Delete(oldsource)

            # Render and save screenshot
            RenderAllViews()
            # layout/tab size in pixels
            layout.SetSize(2162, 1079)
            SaveScreenshot(outpath+
                        infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                        SaveAllViews=1,ImageResolution=[2162,1079])

            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
