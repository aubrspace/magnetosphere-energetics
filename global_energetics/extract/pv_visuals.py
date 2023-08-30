from paraview.simple import *

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
            mpLUTColorBar = GetScalarBar(mpLUT, renderView)
            mpLUTColorBar.WindowLocation = 'Upper Right Corner'
            mpDisplay.SetScalarBarVisibility(renderView,
                                             kwargs.get('show_legend',True))

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

