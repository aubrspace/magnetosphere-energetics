#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import os
import tecplot as tp
import time as realtime
from tecplot.constant import *
import numpy as np
from numpy import deg2rad, linspace
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
#from global_energetics.makevideo import get_time
from global_energetics.makevideo import get_time
from global_energetics.extract.swmf_access import swmf_read_time
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import (abs_to_timestamp,
                                                    mag2cart)

def add_IMF_clock(frame, clockangle, coordsys, bmag, pdyn, position, size,
                  strID):
    """Adds a clock with current IMF orientation
    Inputs
        frame- tecplot frame to add clock to
        clockangle- IMF angle, angle from +Y towards +Z
        coordsys- GSM, GSE etc
        bmag- magnitude of B at X=+31.5, Y=0, Z=0
        outputpath- where to save image (will be deleted)
        position, size- image settings
    """
    plt.rc('xtick', labelsize=30, color='white')
    plt.rc('ytick', labelsize=30, color='white')
    fig = plt.figure(figsize=(12, 6*1.2), facecolor='gray')

    ##Create a clock plot in python
    arrowsize = 4*bmag/15
    # make a square figure
    ax = fig.add_subplot(122, polar=True, facecolor='gray', frame_on=False)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids([0,90,180,270], ['+Z','+Y','-Z','-Y'])
    ax.set_rticks([])

    r = np.arange(0, 3.0, 0.01)
    ax.set_rmax(2.0)
    # arrow at 45 degree
    arrfwd = plt.arrow(clockangle/180.*np.pi, 0, 0, 1, alpha=0.5,
                       width=0.0375*arrowsize, edgecolor='black',
                       facecolor='cyan', lw=6, zorder=5)
    arrback = plt.arrow((clockangle+180)/180.*np.pi, 0, 0, 1, alpha=0.5,
                        width = 0.0375*arrowsize, edgecolor='black',
                        facecolor='cyan', lw=6, zorder=5, head_width=0)
    ax.set_title('\nIMF ({})\n'.format(coordsys),fontsize=40,color='white')
    ##Create a dynamic pressure bar graph
    bar = fig.add_subplot(121, facecolor='gray', frame_on=False)
    bar.bar(0,pdyn,color='magenta')
    bar.set_xlim([-0.5,0.5])
    bar.set_ylim([0,10])
    bar.set_title('\nPdyn (nPa)\n'.format(coordsys), fontsize=40,
                  color='white')
    bar.tick_params(axis='x',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks
                   bottom=False,      # ticks along the bottom edge
                   top=False,         # ticks along the top edge
                   labelbottom=False) # labels along the bottom edge
    fig.tight_layout(pad=1)
    #Save plot
    figname = (os.getcwd()+'/temp_imfclock'+strID+'.png')
    #           str(np.random.rand()).split('.')[-1]+'.png')
    fig.savefig(figname, facecolor='gray', edgecolor='gray')
    #Load plot onto current frame
    img = frame.add_image(figname, position, size)
    #Delete plot image file
    os.remove(figname)

def add_shade_legend(frame, entries, location, markersize):
    """Adds box with colored squares and text for legend of shaded surfaces
    Inputs
        frame
        entries- list of strings for which surfaces are present
        location- legend box location
        markersize
    """
    boxsize = (30, 8*len(entries))
    legendbox = frame.add_rectangle(location, boxsize, CoordSys.Frame)
    legendbox.color = Color.Grey
    legendbox.fill_color = Color.Grey
    for entry in enumerate(entries):
        #fill in text
        entrytext = frame.add_text(entry[1])
        entrytext.position = [location[0]+0.3*boxsize[0],
                    location[1]+boxsize[1]*(1-(entry[0]+0.6)/len(entries))]
        entrytext.font.size = 20
        entrytext.font.bold = False
        entrytext.font.typeface = 'Helvetica'
        entrytext.color = Color.White
        #add color square
        marker_loc = [entrytext.position[0]-0.2*boxsize[0],
                      entrytext.position[1]]
        marker = frame.add_square(marker_loc, markersize, CoordSys.Frame)
        #Select color
        if entry[1] == 'mp_hybrid':
            marker.color = Color.Custom20
            marker.fill_color = Color.Custom20
        if entry[1] == 'mp_fieldline':
            marker.color = Color.Custom34
            marker.fill_color = Color.Custom34
        if entry[1] == 'mp_flowline':
            marker.color = Color.Custom11
            marker.fill_color = Color.Custom11
        if entry[1].find('mp_shue') != -1:
            marker.color = Color.Custom2
            marker.fill_color = Color.Custom2
        if entry[1] == 'mp_test':
            marker.color = Color.Custom7
            marker.fill_color = Color.Custom7
        if entry[1] == 'cps_zone':
            marker.color = Color.Custom8
            marker.fill_color = Color.Custom8

def set_orientation_axis(frame, *, position=[91,7]):
    """Sets position and size of orientation axis
    Inputs
        frame
        position- [x, y] list object
    """
    plt = frame.plot()
    plt.axes.orientation_axis.size = 8
    plt.axes.orientation_axis.position = position
    plt.axes.orientation_axis.color = Color.White

def add_fieldlines(frame, filename, showleg=False, mode='not_supermag'):
    """adds streamlines
    Inputs
        frame- frame to add to
        TBD more options for where to seed rakes
    """
    plt = frame.plot()
    ds = frame.dataset
    plt.vector.u_variable = ds.variable('B_x *')
    plt.vector.v_variable = ds.variable('B_y *')
    plt.vector.w_variable = ds.variable('B_z *')
    plt.show_streamtraces = True
    btilt = float(ds.zone(0).aux_data['BTHETATILT'])
    if mode=='supermag':
        #read in the station locations
        #with open('supermag.dat','r') as s:
        import pandas as pd
        stations_df = pd.read_csv('supermag.dat', delimiter='\t',
                                  skiprows=[0,1,2,4])
        latlons = stations_df[['MAGLAT','MAGLON']].values
    else:
        localtime = get_time(filename)
        tshift = (localtime.hour+localtime.minute/60) * (360/24)
        lons = (np.linspace(0,360,36,endpoint=False) + tshift)%360
        lats = np.zeros(len(lons))+80
        latlons = [l for l in zip(lats,lons)]
    for lat,lon in latlons:
        plt.streamtraces.add(mag2cart(lat,lon,btilt),Streamtrace.VolumeLine)
    """
    plt.streamtraces.add_rake([20,0,40],[20,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,0,40],[10,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-10,0,20],[-10,0,-20],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-20,0,10],[-20,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-20,0,10],[-30,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-30,0,10],[-20,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-30,0,10],[-50,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-50,0,10],[-30,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-80,0,10],[-80,0,-10],Streamtrace.VolumeLine)
    '''
    plt.streamtraces.add_rake([10,0,30],[-40,0,30],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,0,-30],[-40,0,-30],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,30,0],[-40,30,0],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,-30,0],[-40,-30,0],Streamtrace.VolumeLine)
    '''
    """
    plt.streamtraces.obey_source_zone_blanking = False
    plt.streamtraces.color = plt.contour(3)
    plt.contour(3).variable_index = frame.dataset.variable('Status').index
    plt.contour(3).colormap_name='Large Rainbow'
    plt.contour(3).colormap_filter.reversed=True
    plt.contour(3).levels.reset_levels([-1,0,1,2,3])
    plt.contour(3).legend.show = showleg
    plt.contour(3).legend.position[1] = 98
    plt.contour(3).legend.position[0] = 98
    plt.contour(3).legend.box.box_type = TextBox.Filled
    plt.contour(3).legend.box.fill_color = Color.Custom2
    plt.streamtraces.line_thickness = 0.2

def add_jy_slice(frame, jyindex, showleg):
    """adds iso contour for earth at r=1Re
    Inputs
        frame
        jyindex
    """
    frame.plot().show_slices = True
    frame.plot().slice(1).show = False
    yslice = frame.plot().slice(0)
    yslice.show=True
    yslice.orientation = SliceSurface.YPlanes
    yslice.origin[1] = -.10
    yslice.contour.flood_contour_group_index = 1
    yslice.effects.use_translucency = True
    yslice.effects.surface_translucency = 40
    jycontour = frame.plot().contour(1)
    jycontour.variable_index=jyindex
    jycontour.colormap_name = 'green-pink'
    jycontour.legend.vertical = True
    jycontour.legend.position[1] = 98
    jycontour.legend.position[0] = 98
    jycontour.legend.box.box_type = TextBox.Filled
    jycontour.legend.box.fill_color = Color.Custom2
    jycontour.legend.show = showleg
    jycontour.levels.reset_levels(np.linspace(-0.003,0.003,11))
    jycontour.labels.step = 2
    jycontour.colormap_filter.distribution=ColorMapDistribution.Continuous
    jycontour.colormap_filter.continuous_max = 0.003
    jycontour.colormap_filter.continuous_min = -0.003

def add_jz_slice(frame, jzindex, showleg):
    """adds iso contour for earth at r=1Re
    Inputs
        frame
        jyindex
    """
    frame.plot().show_slices = True
    zslice = frame.plot().slice(1)
    zslice.show=True
    zslice.orientation = SliceSurface.ZPlanes
    zslice.origin[1] = -.10
    zslice.contour.flood_contour_group_index = 2
    zslice.effects.use_translucency = True
    zslice.effects.surface_translucency = 40
    jzcontour = frame.plot().contour(2)
    jzcontour.variable_index=jzindex
    jzcontour.colormap_name = 'green-pink'
    jzcontour.legend.vertical = True
    jzcontour.legend.position[1] = 72
    jzcontour.legend.position[0] = 98
    jzcontour.legend.box.box_type = TextBox.Filled
    jzcontour.legend.box.fill_color = Color.Custom2
    jzcontour.legend.show = showleg
    jzcontour.levels.reset_levels(np.linspace(-0.003,0.003,11))
    jzcontour.labels.step = 2
    jzcontour.colormap_filter.distribution=ColorMapDistribution.Continuous
    jzcontour.colormap_filter.continuous_max = 0.003
    jzcontour.colormap_filter.continuous_min = -0.003

def add_Bstar_slice(frame, bindex, showleg):
    """adds iso contour for earth at r=1Re
    Inputs
        frame
        jyindex
    """
    print(frame.dataset.variable(bindex).name)
    frame.plot().show_slices = True
    frame.plot().slice(0).show = False
    bslice = frame.plot().slice(1)
    bslice.show=True
    bslice.orientation = SliceSurface.YPlanes
    bslice.origin[1] = 0
    bslice.contour.flood_contour_group_index = 2
    bslice.effects.use_translucency = True
    bslice.effects.surface_translucency = 40
    bcontour = frame.plot().contour(2)
    bcontour.variable_index=bindex
    bcontour.colormap_name = 'blue-6'
    bcontour.legend.vertical = True
    bcontour.legend.position[1] = 72
    bcontour.legend.position[0] = 98
    bcontour.legend.box.box_type = TextBox.Filled
    bcontour.legend.box.fill_color = Color.Custom2
    bcontour.legend.show = showleg
    bcontour.levels.reset_levels(np.linspace(0,10,11))
    bcontour.labels.step = 2
    bcontour.colormap_filter.distribution=ColorMapDistribution.Continuous
    bcontour.colormap_filter.continuous_max = 10
    bcontour.colormap_filter.continuous_min = 0

def add_earth_iso(frame, rindex):
    """adds iso contour for earth at r=1Re
    Inputs
        frame
        rindex
    """
    frame.plot().show_isosurfaces = True
    iso = frame.plot().isosurface(0)
    iso.definition_contour_group_index = 5
    iso.contour.flood_contour_group_index = 5
    frame.plot().contour(5).variable_index = rindex
    iso.isosurface_values[0] = 1
    iso.show = True
    frame.plot().contour(5).colormap_name = 'Sequential - Green/Blue'
    frame.plot().contour(5).colormap_filter.reversed = True
    frame.plot().contour(5).legend.show = False

def twodigit(num):
    """Function makes two digit str from a number
    Inputs
        num
    Ouputs
        num_str
    """
    return '{:.0f}{:.0f}'.format(np.floor(num/10),num%10)

def add_timestamp(frame, filename, position):
    """Adds timestampt to the frame
    Inputs
        frame- frame object
        filename- used to determine timestamp
        position- x, y tuple
    """
    #get text
    #ticks = get_time(filename)
    #time = ticks.UTC[0]+dt.timedelta(minutes=45)
    dateinfo='-'.join(filename.split('/')[-1].split(
                                       'e')[1].split('.')[0].split('-')[0:-1])
    time = dt.datetime.strptime(dateinfo,'%Y%m%d-%H%M%S')
    #time = time+dt.timedelta(minutes=45)
    #time = swmf_read_time()
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    second = time.second
    time_text = ('{:.0f}-'.format(year)+
                '{}-'.format(twodigit(month))+
                '{} '.format(twodigit(day))+
                '{}:'.format(twodigit(hour))+
                '{}:{}UTC'.format(twodigit(minute), twodigit(second)))
    #add timestamp
    timebox= frame.add_text(time_text)
    timebox.position = position
    timebox.font.size = 20
    timebox.font.bold = False
    timebox.font.typeface = 'Helvetica'
    timebox.color = Color.White

def add_energy_contour(frame, powermax, contour_key, mapnumber, showleg, *,
                          colormap='Doppler modified (1)'):
    """Function sets contour settings for energy input
    Inputs
        frame- object to set contour on
        powermax- saturation/limit for contour colorbar
        colormap- which colormap to use
    """
    colorbar = np.linspace(-1*powermax, powermax, 11)
    contourvar = frame.dataset.variable(contour_key).index
    contour = frame.plot().contour(mapnumber)
    contour.variable_index = contourvar
    contour.colormap_name = colormap
    contour.legend.vertical = True
    contour.legend.position[1] = 98
    contour.legend.position[0] = 98
    contour.legend.box.box_type = TextBox.Filled
    contour.legend.box.fill_color = Color.Custom2
    contour.legend.show = showleg
    contour.levels.reset_levels(colorbar)
    contour.labels.step = 2
    contour.colormap_filter.distribution=ColorMapDistribution.Continuous
    contour.colormap_filter.continuous_max = colorbar[-1]
    contour.colormap_filter.continuous_min = colorbar[0]
    contour.colormap_filter.reversed = True

def set_camera(frame, *, setting='iso_day'):
    """Function sets camera angle based on setting
    Inputs
        frame- frame on which to change camera
        setting- iso_day, iso_tail, xy, xz, yz
    """
    plot = frame.plot()
    view = plot.view
    if setting == 'iso_day':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0, 240, 64)
        view.position = (1960, 1140, 1100)
        #view.magnification = 4.7
        view.magnification = 6.7
        oa_position = [91,7]
    elif setting == 'iso_tail':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,137,64)
        view.position = (-500,509,333)
        view.magnification = 7.470
        oa_position = [95,7]
    elif setting == 'other_iso':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,17,116)
        view.position = (-680,-2172,-1120)
        view.magnification = 5.56
        oa_position = [91,7]
    elif setting == 'inside_from_tail':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,83,84)
        view.position = (-2514.8,-310.4,264.5)
        view.magnification = 15.30
        oa_position = [95,7]
    elif setting == 'zoomed_out':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,156,80)
        view.position = (-342,694,122)
        view.magnification = 2.603
        oa_position = [95,7]
    elif setting == 'hood_open_north':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (-1.75,92,44)
        view.position = (-1770,62.5,1828)
        view.magnification = 15.73
        oa_position = [12,85]
    else:
        print('Camera setting {} not developed!'.format(setting))
    #light source settings
    plot.light_source.specular_shininess=0
    plot.light_source.specular_intensity=0
    plot.light_source.background_light=100
    plot.light_source.intensity=100
    plot.light_source.direction=(0,0,0)
    #orientation axis
    set_orientation_axis(frame, position=oa_position)

def set_3Daxes(frame, *,
                  xmax=20, xmin=-25, ymax=20, ymin=-20, zmax=20, zmin=-20,
                  do_blanking=True):
    """Function sets axes in 3D and blanks data outside of axes range
    Inputs
        frame- frame object to manage axes on
        xyz max/min- axis ranges available overwrite
        do_blanking- blanking to include blanking
    """
    axes = frame.plot().axes
    axes.axis_mode = AxisMode.Independent
    axes.x_axis.show = True
    axes.x_axis.max = xmax
    axes.x_axis.min = xmin
    axes.x_axis.scale_factor = 1
    axes.x_axis.title.position = 30
    axes.x_axis.title.color = Color.Custom2
    axes.x_axis.tick_labels.color = Color.Custom2
    axes.x_axis.line.color = Color.Custom2
    axes.x_axis.grid_lines.color = Color.Custom2
    axes.y_axis.show = True
    axes.y_axis.max = ymax
    axes.y_axis.min = ymin
    axes.y_axis.scale_factor = 1
    axes.y_axis.title.position = 30
    axes.y_axis.title.color = Color.Custom2
    axes.y_axis.tick_labels.color = Color.Custom2
    axes.y_axis.line.color = Color.Custom2
    axes.y_axis.grid_lines.color = Color.Custom2
    axes.z_axis.show = True
    axes.z_axis.max = zmax
    axes.z_axis.min = zmin
    axes.z_axis.scale_factor = 1
    axes.z_axis.title.offset = -8
    axes.z_axis.title.color = Color.Custom2
    axes.z_axis.tick_labels.color = Color.Custom2
    axes.z_axis.line.color = Color.Custom2
    axes.z_axis.grid_lines.color = Color.Custom2
    axes.grid_area.filled = False
    if do_blanking:
        #blanking outside 3D axes
        frame.plot().value_blanking.active = True
        #x
        xblank = frame.plot().value_blanking.constraint(1)
        xblank.active = True
        xblank.variable = frame.dataset.variable('X *')
        xblank.comparison_operator = RelOp.LessThan
        xblank.comparison_value = xmin
        xblank = frame.plot().value_blanking.constraint(2)
        xblank.active = True
        xblank.variable = frame.dataset.variable('X *')
        xblank.comparison_operator = RelOp.GreaterThan
        xblank.comparison_value = xmax
        #y
        yblank = frame.plot().value_blanking.constraint(3)
        yblank.active = True
        yblank.variable = frame.dataset.variable('Y *')
        yblank.comparison_operator = RelOp.LessThan
        yblank.comparison_value = ymin
        yblank = frame.plot().value_blanking.constraint(4)
        yblank.active = True
        yblank.variable = frame.dataset.variable('Y *')
        yblank.comparison_operator = RelOp.GreaterThan
        yblank.comparison_value = ymax
        #z
        zblank = frame.plot().value_blanking.constraint(5)
        zblank.active = True
        zblank.variable = frame.dataset.variable('Z *')
        zblank.comparison_operator = RelOp.LessThan
        zblank.comparison_value = zmin
        zblank = frame.plot().value_blanking.constraint(6)
        zblank.active = True
        zblank.variable = frame.dataset.variable('Z *')
        zblank.comparison_operator = RelOp.GreaterThan
        zblank.comparison_value = zmax

def variable_blank(frame, variable_str, value, *,
                   slot=5, operator=RelOp.GreaterThan):
    """Function sets up blanking based on given criteria
    Inputs
        frame- frame object to manage axes on
        variable_str, value- which variable to use for condition
        slot- only 8 available slots for tecplot, 0-7
        operator- see tecplot.constant for options
    """
    frame.plot().value_blanking.active = True
    blank = frame.plot().value_blanking.constraint(slot)
    blank.active = True
    blank.variable = frame.dataset.variable(variable_str)
    blank.comparison_operator = operator
    blank.comparison_value = value

def manage_zones(frame, nslice, translucency, cont_num, zone_hidekeys,
                 energyfracs, fracnames, mode, *,
                 approved_zones=None):
    """Function shows/hides zones, sets shading and translucency
    Inputs
        frame- frame object to manage zones on
        nslice- used to show only outer surface, assumes cylindrical
        approved_zones- default shows all in predetermined list
    Outputs
        show_list- list of zones shown after operations
    """
    plt = frame.plot()
    show_list = []
    hide_keys = zone_hidekeys
    shadings = {'mp_iso_betastar':Color.Cyan,
                'mp_iso_betastarinnerbound':Color.Custom11,
                'plasmasheet':Color.Custom9,
                'ms_nlobe':Color.Yellow,
                'ms_slobe':Color.Yellow,
                'ms_rc':Color.Custom32,
                'ms_ps':Color.Purple,
                'ms_closed':Color.Custom23,
                'ms_qDp':Color.Custom19,
                'shue97':Color.Custom8,
                'shue98':Color.Custom7,
                'ext_bs':Color.Custom10}
    #hide all other zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if 'inner' in zone.name:
                cont_num=2
            if zone.name == 'global_field':
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = None
            elif any([zone.name.find(key)!=-1 for key in hide_keys]):
                plt.fieldmap(map_index).show = False
            else:
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = (
                                                   SurfacesToPlot.IKPlanes)
                plt.fieldmap(map_index).show = True
                plt.fieldmap(map_index).surfaces.i_range = (-1,-1,1)
                plt.fieldmap(map_index).surfaces.k_range = (0,-1, nslice-1)
                plt.fieldmap(map_index).contour.flood_contour_group_index=(
                                                                    cont_num)
                plt.fieldmap(map_index).shade.color=shadings.get(zone.name,
                                                             Color.Custom2)
                show_list.append(zone.name)
    #Transluceny and shade settings
    '''
    transluc = dict()
    for name in enumerate(fracnames):
        transluc.update({name[1]:int(100-energyfracs[name[0]]*100)})
    '''
    transluc = {'mp_iso_betastar':translucency,
                'mp_iso_betastarinnerbound':1,
                'plasmasheet':translucency,
                'ms_nlobe':translucency,
                'ms_slobe':translucency,
                'ms_rc':translucency,
                'ms_ps':translucency,
                'ms_closed':translucency,
                'ms_qDp':translucency,
                'shue97':translucency,
                'shue98':translucency,
                'ext_bs':translucency}
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            frame.plot(PlotType.Cartesian3D).use_translucency=True
            plt.fieldmap(map_index).effects.use_translucency=True
            plt.fieldmap(map_index).effects.surface_translucency=(
                                      transluc.get(zone.name,translucency))
            '''old shade settings
            if zone.name.find('hybrid') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom20
            if zone.name.find('fieldline') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom34
            if zone.name.find('flowline') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom11
            if zone.name.find('shue') != -1:
                if zone.name.find('97') != -1:
                    plt.fieldmap(map_index).shade.color = Color.Custom9
                else:
                    plt.fieldmap(map_index).shade.color = Color.Custom2
            if zone.name.find('test') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom7
            if zone.name.find('cps') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom8
            '''
    return show_list

def set_satellites(satnames, frame):
    """Function sets view settings for 3D satellite data
    Inputs
        satnames- list of satellite names
    """
    dataset = frame.dataset
    plot = frame.plot()
    plot.show_mesh = True
    plot.show_scatter = True
    #Get the name of magnetosphere variable, if it exists
    for zone in dataset.zones('*innerBound'):
        mp_surface_var = zone.name.split('innerbound')[0]
    #Get corresponding satellite fieldmap variable indices
    satindices,loc_satindices = [], []
    for sat in satnames:
        satindices.append(dataset.zone(sat).index)
        loc_satindices.append(int(dataset.zone('loc_'+sat).index))
    '''
    for name in satnames:
        satindices.append(int(dataset.zone(name).index))
        if len([zn for zn in dataset.zones('loc_'+name)]) > 0:
            loc_satzone = dataset.zone('loc_'+name)
        else:
            #create local sat zone with sat current position
            loc_satzone =  dataset.add_ordered_zone('loc_'+name, [1,1,1])
            #get the current position of the satellite based on aux data
            eventstring =dataset.zone('global_field').aux_data['TIMEEVENT']
            startstring=dataset.zone('global_field').aux_data[
                                                          'TIMEEVENTSTART']
            eventdt = dt.datetime.strptime(eventstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            startdt = dt.datetime.strptime(startstring,
                                                    '%Y/%m/%d %H:%M:%S.%f')
            deltadt = eventdt-startdt
            tvals = dataset.zone(name).values('t').as_numpy_array()
            xvals = dataset.zone(name).values('X *').as_numpy_array()
            yvals = dataset.zone(name).values('Y *').as_numpy_array()
            zvals = dataset.zone(name).values('Z *').as_numpy_array()
            svals=dataset.zone(name).values(mp_surface_var).as_numpy_array()
            xpos = xvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            ypos = yvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            zpos = zvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            status = svals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            loc_satzone.values('X *')[0] = xpos
            loc_satzone.values('Y *')[0] = ypos
            loc_satzone.values('Z *')[0] = zpos
            loc_satzone.values(mp_surface_var)[0] = state
        #add new zone index
        loc_satindices.append(int(loc_satzone.index))
    '''
    #Turn off mesh  and scatter for all maps that arent satellites
    for index in range(0, dataset.num_zones):
        if not any([ind == index for ind in satindices]):
            plot.fieldmap(index).mesh.show = False
        if not any([ind == index for ind in loc_satindices]):
            plot.fieldmap(index).scatter.show = False
    #setup colors that dont look terrible
    colorwheel = [Color.Custom3, Color.Custom5, Color.Custom6,
                  Color.Custom7, Color.Custom8, Color.Custom11,
                  Color.Custom40, Color.Custom34, Color.Custom42,
                  Color.Custom50, Color.Custom51, Color.Custom19,
                  Color.Custom27, Color.Custom35]
    #setup Status variable contour map
    plot.contour(2).variable_index = int(
                                    dataset.variable(mp_surface_var).index)
    plot.contour(2).colormap_name = 'Diverging - Orange/Purple'
    plot.contour(2).levels.reset_levels([-1,0,1,2,3])
    plot.contour(2).legend.position[1] = 46
    plot.contour(2).legend.position[0] = 98
    plot.contour(2).legend.box.box_type = TextBox.Filled
    #Set color, linestyle, marker and sizes for satellite
    for sat in enumerate(satindices):
        index = loc_satindices[sat[0]]
        #getstate
        status = dataset.zone(index).values('Status')[0]
        if status == 0:
            inside = False
        else:
            inside = True
        #mesh
        plot.fieldmap(sat[1]).mesh.show = True
        #plot.fieldmap(sat[1]).mesh.color = colorwheel[sat[0]]
        plot.fieldmap(sat[1]).mesh.color = Color.White
        plot.fieldmap(sat[1]).mesh.line_thickness = 1
        plot.fieldmap(sat[1]).mesh.line_pattern = LinePattern.Solid
        #scatter
        plot.fieldmap(index).show = True
        plot.fieldmap(index).scatter.show = True
        if inside:
            plot.fieldmap(index).scatter.symbol().shape=GeomShape.Octahedron
            plot.fieldmap(index).scatter.color = Color.Yellow
            plot.fieldmap(index).scatter.size = 2
        else:
            plot.fieldmap(index).scatter.symbol().shape = GeomShape.Cube
            plot.fieldmap(index).scatter.color = Color.Purple
            plot.fieldmap(index).scatter.size = 1.5



def display_single_iso(frame, filename, *, mode='iso_day', **kwargs):
    """Function adjusts viewsettings for a single panel 3D image
    Inputs
        frame- object for the tecplot frame
        filename
        mode
    **kwargs
        contour_key- string key for which contour variable to plot
        energyrange- limits for contour saturation on energy contour
        save_img- default True
        pngpath- path for saving .png file
        save_plt- default True
        pltpath- path for saving .png file
        outputname- default is 'output.png'
        show_contour, show_fieldline, show_slice, do_blanking
        tile- boolean for tile mode
        mpslice- number of x slices in mp surface
        cpsslice- number of x slices in cps surface
    """
    ###Always included
    path = os.getcwd()+'/energetics.map'
    tp.macro.execute_command('$!LOADCOLORMAP "'+path+'"')
    #frame.background_color = Color.Custom46
    #frame.background_color = Color.Black
    frame.background_color = Color.Custom17
    add_earth_iso(frame, rindex=frame.dataset.variable('r *').index)
    ###DEFAULTS for genaric mode###
    default = {'transluc': 1,           #zone settings
               'energyfracs':[.4,.4,.4,.4,.4],
               'fracnames':['ms_nlobe','ms_slobe','ms_rc','ms_ps','ms_qDp'],
               'zone_hidekeys':['sphere','box','lcb','shue','future'],
               'plot_satellites': False,
               'satzones': [],
               'do_blanking': True,
               'xtail': -45,
               'mpslice':60,
               'cpsslice':20,
               'contour_key': 'K_net *', #contour settings
               'energyrange': 3e9,
               'show_contour': True,
               'contourmap': 0,
               'show_legend': True,
               'save_img': True,         #io settings
               'save_plt': False,
               'mhddir': './',
               'pltpath': './',
               'pngpath': './',
               'outputname': 'output',
               'pngwidth':1600,
               'IDstr': '0',
               'show_slice': True,       #slice settings
               'slicetype': 'jy',
               'show_fieldline': False,
               'show_flegend': False,
               'show_slegend': True,
               'add_clock': False,       #overlay settings
               'show_timestamp': True,
               'timestamp_pos': [4,5],
               'show_shade_legend': False,
               'shade_legend_pos': [5,70],
               'shade_markersize': 3}
    ###############################
    ###Mode specific settings###
    if mode == 'inside_from_tail':
        default['xtail'] = -15
        default['show_slice'] = False
        default['transluc'] = 40
    elif mode=='hood_open_north':
        default['show_slice']=False
        default['transluc'] = 50
        variable_blank(frame, 'Z *', 5, operator=RelOp.GreaterThan)
    elif mode == 'iso_tail' or mode=='zoomed_out':
        default['transluc'] = 60
    elif mode == 'other_iso' or mode=='zoomed_out':
        default['add_clock'] = True
        default['transluc'] = 30
    ###############################
    ###Overwrite w/ kwargs###
    for inkey in kwargs:
        default[inkey]=kwargs[inkey]
    ###############################
    #Produce image
    zones_shown= manage_zones(frame,default['mpslice'],default['transluc'],
                            default['contourmap'],default['zone_hidekeys'],
                            default['energyfracs'],default['fracnames'],mode)
    set_3Daxes(frame, xmin=default['xtail'], do_blanking=False)
    set_camera(frame, setting=mode)
    if default['show_contour']:
        add_energy_contour(frame,default['energyrange'],
                           default['contour_key'], default['contourmap'],
                           default['show_legend'])
        if mode == 'hood_open_north':
            add_energy_contour(frame, 0.01,'J_par *',2,True,
                               colormap='cmocean - balance')
    frame.plot().show_contour = default['show_contour']
    if default['show_slice']:
        if default['slicetype']=='jy':
            add_jy_slice(frame, frame.dataset.variable('J_y *').index,
                         default['show_slegend'])
        elif default['slicetype']=='jz':
            add_jz_slice(frame, frame.dataset.variable('J_z *').index,
                         showleg=default['show_slegend'])
        elif slicetype=='betastar':
            add_Bstar_slice(frame,frame.dataset.variable('beta_star').index,
                            showleg=default['show_slegend'])
    if default['show_fieldline']:
        add_fieldlines(frame, filename, showleg=default['show_flegend'])
    if default['show_shade_legend']:
        add_shade_legend(frame, zones_shown, default['shade_legend_pos'],
                         default['shade_markersize'])
    if default['plot_satellites']:
        if satzones == []:
            print('No satellite zones to plot')
        else:
            set_satellites(default['satzones'], frame)
    if default['show_timestamp']:
        add_timestamp(frame, filename, default['timestamp_pos'])
    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
    if default['add_clock']:
        #get clock angle from probing data at x=xmax
        clock = float(frame.dataset.zone('global_field').aux_data[
                                                          'imf_clock_deg'])
        coordsys = frame.dataset.zone('global_field').aux_data[
                                                            'COORDSYSTEM']
        bmag= float(frame.dataset.zone('global_field').aux_data['imf_mag'])
        pdyn= float(frame.dataset.zone('global_field').aux_data['sw_pdyn'])
        add_IMF_clock(frame, clock, coordsys, bmag, pdyn, (0,0), 30,
                      default['IDstr'])
    frame.plot().contour(4).legend.show = False
    frame.plot().contour(5).legend.show = False
    frame.plot().contour(6).legend.show = False
    frame.plot().contour(7).legend.show = False
    if default['save_img']:
        #multiframe image (default)
        tp.export.save_png(os.getcwd()+'/'+default['pngpath']+'/'+
                           default['outputname']+'.png',
                                                 width=default['pngwidth'])
    if default['save_plt']:
        tp.data.save_tecplot_plt(default['pltpath']+
                                              default['outputname']+'.plt',
                                 include_data_share_linkage=True,
                                 include_autogen_face_neighbors=True)

def display_2D_contours(frame, **kwargs):
    """Function does view settings for 2D contour plot
    Inputs
        frame
        kwargs:
    """
    #Initialization
    ds = frame.dataset
    filename = kwargs.get('filename','var_1_e20130430-040200-000.h5')
    betastar_index = ds.variable('beta_star').index
    mp_index = ds.variable('mp').index
    closed_index = ds.variable('closed').index
    plot = frame.plot()
    #Axis settings
    #plot.axes.axis_mode=AxisMode.Independent
    #plot.axes.preserve_scale=True
    plot.axes.x_axis.min=-60
    plot.axes.x_axis.max=30
    plot.axes.x_axis.reverse=True
    plot.axes.y_axis.min=-40
    plot.axes.y_axis.max=40
    #Contour settings
    plot.contour(0).variable_index= betastar_index
    plot.show_contour=True
    plot.contour(0).levels.reset_levels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,
                                             0.8,0.9,1,1.1,1.2,1.3,1.4])
    plot.contour(0).colormap_name='div3-green-brown-div'
    #Blanking 1- magnetopause boundary trace
    blank = plot.value_blanking
    blank.active=True
    blank.constraint(0).active=True
    blank.constraint(0).variable_index=mp_index
    blank.cell_mode=ValueBlankCellMode.TrimCells
    blank.constraint(0).show_line=True
    blank.constraint(0).color=Color.Cyan
    blank.constraint(0).comparison_value=0.7
    #Blanking 2- "closed" field line region
    blank.constraint(1).active=True
    blank.constraint(1).variable_index=closed_index
    blank.constraint(1).show_line=True
    blank.constraint(1).color=Color.White
    blank.constraint(1).comparison_operator=RelOp.LessThan
    blank.constraint(1).comparison_value=kwargs.get('closed_val',0)
    blank.constraint(1).line_pattern=LinePattern.Dashed
    #Copy zone so outline shows up on top of contours
    ds.copy_zones([1])
    ds.zone(-1).name='copy_initial_triangulation'
    plot.fieldmaps(1).effects.value_blanking=False
    #Put 1Re circle and 2.5Re boundary
    tp.macro.execute_command('''$!AttachGeom 
  GeomType = Circle
  Color = Custom2
  IsFilled = Yes
  FillColor = Custom2
  RawData
2.5''')
    tp.macro.execute_command('''$!AttachGeom 
  GeomType = Circle
  Color = Custom6
  IsFilled = Yes
  FillColor = Custom6
  RawData
1''')
    #Adjust contour legend
    plot.contour(0).legend.vertical=False
    plot.contour(0).legend.box.box_type=TextBox.Filled
    plot.contour(0).legend.box.fill_color=Color.Custom2
    plot.contour(0).legend.position = (85,98)
    #Frame background color
    plot.frame.background_color = Color.Custom1
    plot.axes.y_axis.tick_labels.color=Color.White
    plot.axes.y_axis.title.color=Color.White
    plot.axes.y_axis.line.color=Color.White
    plot.axes.x_axis.tick_labels.color=Color.White
    plot.axes.x_axis.title.color=Color.White
    plot.axes.x_axis.line.color=Color.White
    add_timestamp(frame, filename, kwargs.get('timestamp_pos',(15,15)))
    #Save output file
    tp.export.save_png(os.getcwd()+'/'+kwargs.get('pngpath','./')+'/'+
                           kwargs.get('outputname','2Dcontour')+'.png',
                           width=kwargs.get('pngwidth',1600))

# Use main functionality to reset view setting in connected mode
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"
if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()
    FRAME = tp.active_frame()
    INDEX = 30
    VAR = 30
    COLORBAR = np.linspace(-4,4,12)

    display_magnetopause(FRAME, INDEX, VAR, COLORBAR, False)
