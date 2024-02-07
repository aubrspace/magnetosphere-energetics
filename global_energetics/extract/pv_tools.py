# An eclectic  mix of functions ranging from critical to mildly useful
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
import numpy as np
from numpy import sin,cos,pi
#### import the simple module from paraview
from paraview.simple import *
from geopack import geopack as gp
#from equations import rotation

def tec2para(instr):
    """Function takes a tecplot function evaluation string and makes it
        readable by paraview using string manipulations
    Inputs
        instr (str)
    Returns
        outstr (str)
    """
    badchars = ['{','}','[',']']
    replacements = {' [':'_','**':'^','1e':'10^','pi':'3.14159',#generic
          '/Re':'_Re', 'amu/cm':'amu_cm','km/s':'km_s','/m':'_m',#specific
                    'm^':'m','e^':'e','^3':'3'}#very specific, for units only
            #'/':'_',
    coords = {'X_R':'x','Y_R':'y','Z_R':'z'}
    outstr = instr
    for was_,is_ in replacements.items():
        outstr = is_.join(outstr.split(was_))
    for was_,is_ in coords.items():
        outstr = is_.join(outstr.split(was_))
        #NOTE this order to find "{var [unit]}" cases (space before [unit])
        #TODO see .replace
    for bc in badchars:
        outstr = ''.join(outstr.split(bc))
    #print('WAS: ',instr,'\tIS: ',outstr)
    return outstr

def eqeval(eqset,pipeline,**kwargs):
    """Function creates a calculator object which evaluates the given function
    Inputs
        eqset (dict{lhs:rhs}) - ported from tecplot format, NOTE all strings
                                must be converted with 'tec2para' before they
                                can be evaluated!!
        pipeline (pipeline) - where attach the new filter
        kwargs:
    Returns
        pipeline (pipeline) - a new endpoint of the pipeline
    """
    for lhs_tec,rhs_tec in eqset.items():
        lhs = tec2para(lhs_tec)
        rhs = tec2para(rhs_tec)
        var = Calculator(registrationName=lhs, Input=pipeline)
        var.Function = rhs
        var.ResultArrayName = lhs
        pipeline = var
    return pipeline

def get_sphere_filter(pipeline,**kwargs):
    """Function calculates a sphere variable, NOTE:will still need to
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
    assert FindSource('r_R') != None
    radius = kwargs.get('radius',3)
    r_state =ProgrammableFilter(registrationName='r_state',Input=pipeline)
    r_state.Script = """
    # Get input
    data = inputs[0]
    r = data.PointData['r_R']

    #Compute sphere as logical for discrete points
    r_state = (abs(r-"""+str(radius)+""")<0.2).astype(int)

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(r_state,'r_state')
    """
    pipeline = r_state
    return pipeline

def rotate_vectors(pipeline,angle,**kwargs):
    """Rotates the coordinate variables by multiplying by a rotation matrix
    Inputs
        pipeline
        angle
        kwargs:
            xbase- (default 'x')
            coordinates- (default False)
    Returns
        new_position
    """
    # Contruct the rotation matrix
    mXhat_x = str(sin((-angle+90)*pi/180))
    mXhat_y = str(0)
    mXhat_z = str(-1*cos((-angle+90)*pi/180))
    mZhat_x = str(sin(-angle*pi/180))
    mZhat_y = str(0)
    mZhat_z = str(-1*cos(-angle*pi/180))
    # Save old values
    xbase = kwargs.get('xbase','x')
    Xd = xbase.replace('x','xd')
    Y = xbase.replace('x','y')
    Zd = xbase.replace('x','zd')
    pipeline = Calculator(registrationName=Xd, Input=pipeline)
    pipeline.ResultArrayName = Xd
    pipeline.Function = xbase
    pipeline = Calculator(registrationName=Zd, Input=pipeline)
    pipeline.ResultArrayName = Zd
    pipeline.Function = xbase.replace('x','z')
    # Create the paraview calculator filter 
    Xd = 'xd'
    Zd = 'zd'
    if kwargs.get('coordinates',False):
        new_position = Calculator(registrationName='rotation', Input=pipeline)
        new_position.ResultArrayName = 'rotatedPosition'
        new_position.Function = (
                    mXhat_x+'*('+Xd+'*'+mXhat_x+'+'+Zd+'*'+mXhat_z+')*iHat+'+
                    Y+'*jHat+'+
                    mZhat_z+'*('+Xd+'*'+mZhat_x+'+'+Zd+'*'+mZhat_z+')*kHat')
        new_position.CoordinateResults = 1
        pipeline = new_position
    # X
    new_x = Calculator(registrationName=xbase, Input=pipeline)
    new_x.ResultArrayName = xbase
    new_x.Function = mXhat_x+'*('+Xd+'*'+mXhat_x+'+'+Zd+'*'+mXhat_z+')'
    pipeline = new_x
    # Z
    new_z = Calculator(registrationName=xbase.replace('x','z'),Input=pipeline)
    new_z.ResultArrayName = xbase.replace('x','z')
    new_z.Function = mZhat_x+'*('+Xd+'*'+mZhat_x+'+'+Zd+'*'+mZhat_z+')'
    return new_z

def export_datacube(pipeline,**kwargs):
    """Function creates the prog filter object and calls the script to populate
    Inputs
        pipelilne (pipeline) - where to attach the filter
    Returns
        pipelilne (pipeline) - a newly created enpoint in the pipeline
    """
    if 'path' not in kwargs:
        kwargs['path'] = '/home/aubr/Code/swmf-energetics/localdbug/fte/'
    datacubeFilter = ProgrammableFilter(registrationName='datacube',
                                        Input=pipeline)
    datacubeFilter.Script = update_datacube(**kwargs)
    return datacubeFilter

def update_datacube(**kwargs):
    """Function the script in the prog filter for 'export_datacube'
    Inputs
        kwargs:
            path
            filename
    Returns
        (str) NOTE this is really a small self-contained python function!!
    """
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

def extract_field(pipeline,**kwargs):
    extractFilter = ProgrammableFilter(registrationName='extract_field',
                                        Input=pipeline)
    extractFilter.Script = update_extraction(**kwargs)
    return extractFilter

def update_extraction(**kwargs):
    """Function the script in the prog filter for 'extract_field'
    Inputs
        kwargs:
            path
            filename
    Returns
        (str) NOTE this is really a small self-contained python function!!
    """
    return """
    for var in """+str(kwargs.get('varlist',['x','y','z']))+""":
        #Assign to output
        output.PointData.append(inputs[0].PointData[var],var)
    """

def point2cell(inputsource, fluxes):
    """Function creates a point to cell conversion filter
    Inputs
        inputsource (pipeline)
        fluxes (list) - variable names to convert
    Returns
        point2cell (pipeline) - a new endpoint to the pipeline
    """
    point2cell = PointDatatoCellData(registrationName='surface_p2cc',
                                     Input=inputsource)
    point2cell.ProcessAllArrays = 0
    convert_list = [f[0] for f in fluxes]
    convert_list.append('Normals')
    point2cell.PointDataArraytoprocess = convert_list
    return point2cell

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

def gsm_to_eci(pipeline,ut):
    gp.recalc(ut)
    # Do the coord transform in a programmable filter
    eci =ProgrammableFilter(registrationName='eci',Input=pipeline)
    eci.Script = """
    import numpy as np
    from geopack import geopack as gp
    # Get input
    data = inputs[0]
    # Pull the XYZgsm,Bxyz coordinates from the paraview objects
    x = data.PointData['x']
    y = data.PointData['y']
    z = data.PointData['z']
    bx = data.PointData['B_x_nT']
    by = data.PointData['B_y_nT']
    bz = data.PointData['B_z_nT']
    # Convert GSM -> ECI using geopack functions
    x_eci = np.zeros(len(x))
    y_eci = np.zeros(len(y))
    z_eci = np.zeros(len(z))
    bx_eci = np.zeros(len(bx))
    by_eci = np.zeros(len(by))
    bz_eci = np.zeros(len(bz))
    for i,(xi,yi,zi,bxi,byi,bzi) in enumerate(zip(x,y,z,bx,by,bz)):
        geopos = gp.geogsm(xi,yi,zi,-1)
        geipos = gp.geigeo(*geopos,-1)
        x_eci[i],y_eci[i],z_eci[i] = geipos
        # B field
        geoB = gp.geogsm(bxi,byi,bzi,-1)
        geiB = gp.geigeo(*geoB,-1)
        bx_eci[i],by_eci[i],bz_eci[i] = geiB
    # Load the data back into the output of the Paraview filter
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(x_eci,'x_eci')
    output.PointData.append(y_eci,'y_eci')
    output.PointData.append(z_eci,'z_eci')
    output.PointData.append(bx_eci,'Bx_eci')
    output.PointData.append(by_eci,'By_eci')
    output.PointData.append(bz_eci,'Bz_eci')
    """
    return eci

def gse_to_gsm(pipeline,ut):
    gp.recalc(ut)
    # Do the coord transform in a programmable filter
    gsm =ProgrammableFilter(registrationName='gsm',Input=pipeline)
    gsm.Script = update_gse_to_gsm()
    return gsm

def update_gse_to_gsm():
    return """
    import numpy as np
    from geopack import geopack as gp
    # Get input
    data = inputs[0]
    done = []
    # Loop through all variables that need to be converted
    for var in data.PointData.keys():
        if ((var=='x' or var=='y' or var=='z' or
             '_x' in var or '_y' in var or '_z' in var) and
             var not in done):
            if '_x' in var:
                gse_xvar = var
                gse_yvar = var.replace('_x','_y')
                gse_zvar = var.replace('_x','_z')
            elif '_y' in var:
                gse_xvar = var.replace('_y','_x')
                gse_yvar = var
                gse_zvar = var.replace('_y','_z')
            elif '_z' in var:
                gse_xvar = var.replace('_z','_x')
                gse_yvar = var.replace('_z','_y')
                gse_zvar = var
            else:
                break #Means its not a directional variable
            # Convert GSE -> GSM using geopack functions
            x = data.PointData[gse_xvar]
            y = data.PointData[gse_yvar]
            z = data.PointData[gse_zvar]
            gsm_x = np.zeros(len(x))
            gsm_y = np.zeros(len(y))
            gsm_z = np.zeros(len(z))
            for i,(xi,yi,zi) in enumerate(zip(x,y,z)):
                gsm_x[i],gsm_y[i],gsm_z[i] = gp.gsmgse(xi,yi,zi,-1)
            # Overwrite the gse variable names with the gsm data
            data.PointData[gse_xvar] = gsm_x
            data.PointData[gse_yvar] = gsm_y
            data.PointData[gse_zvar] = gsm_z
            # Mark all three as done so we dont duplicate effort
            done.append(gse_xvar)
            done.append(gse_yvar)
            done.append(gse_zvar)
    # Load the data back into the output of the Paraview filter
    output.ShallowCopy(data.VTKObject)#So rest of inputs flow
    """
