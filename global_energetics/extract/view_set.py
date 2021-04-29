#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import os
import tecplot as tp
from tecplot.constant import *
import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from progress.bar import Bar
from global_energetics.makevideo import get_time
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import abs_to_timestamp

def add_IMF_clock(frame, clockangle, coordsys, position, size):
    """Adds a clock with current IMF orientation
    Inputs
        frame- tecplot frame to add clock to
        clockangle- IMF angle, angle from +Y towards +Z
        cor
        outputpath- where to save image (will be deleted)
        position, size- image settings
    """
    #Create a clock plot in python
    plt.rc('xtick', labelsize=30, color='white')
    plt.rc('ytick', labelsize=30, color='white')

    # force square figure and square axes looks better for polar, IMO
    arrowsize = 4
    # make a square figure
    fig = plt.figure(figsize=(arrowsize, arrowsize*1.2), facecolor='gray')
    ax = fig.add_axes([0.2, 0.1, 0.6, 0.6], polar=True, facecolor='gray',
                      frame_on=False)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids([0,90,180,270], ['+Z','+Y','-Z','-Y'])
    ax.set_rticks([])

    r = np.arange(0, 3.0, 0.01)
    ax.set_rmax(2.0)
    # arrow at 45 degree
    arrfwd = plt.arrow(clockangle/180.*np.pi, 0, 0, 1, alpha=0.5,
                       width=0.15, edgecolor='black', facecolor='cyan',
                       lw=6, zorder=5)
    arrback = plt.arrow((clockangle+180)/180.*np.pi, 0, 0, 1, alpha=0.5,
                        width = 0.15, edgecolor='black', facecolor='cyan',
                        lw=6, zorder=5, head_width=0)
    ax.set_title('\nIMF ({})\n'.format(coordsys),fontsize=40,color='white')
    #Save plot
    figname = (os.getcwd()+'/temp_imfclock'+
                str(np.random.rand()).split('.')[-1]+'.png')
    fig.savefig(figname, facecolor='gray',
                                                  edgecolor='gray')
    #Load plot onto current frame
    img = frame.add_image(figname, position, size)
    #Delete plot image file
    os.system('rm '+figname)

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

def add_fieldlines(frame):
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
    plt.streamtraces.add_rake([20,0,40],[20,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,0,40],[10,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-10,0,20],[-10,0,-20],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-20,0,10],[-20,0,-10],Streamtrace.VolumeLine)
    '''
    plt.streamtraces.add_rake([10,0,30],[-40,0,30],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,0,-30],[-40,0,-30],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,30,0],[-40,30,0],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,-30,0],[-40,-30,0],Streamtrace.VolumeLine)
    '''
    plt.streamtraces.color = Color.Custom41
    plt.streamtraces.line_thickness = 0.2

def add_jy_slice(frame, jyindex, showleg):
    """adds iso contour for earth at r=1Re
    Inputs
        frame
        jyindex
    """
    frame.plot().show_slices = True
    yslice = frame.plot().slice(0)
    yslice.show=True
    yslice.orientation = SliceSurface.YPlanes
    yslice.origin[1] = -.10
    yslice.contour.flood_contour_group_index = 1
    yslice.effects.use_translucency = True
    yslice.effects.surface_translucency = 40
    jycontour = frame.plot().contour(1)
    jycontour.variable_index=jyindex
    jycontour.colormap_name = 'orange-green-blue-gray'
    jycontour.legend.vertical = True
    jycontour.legend.position[1] = 72
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
    jzcontour.colormap_name = 'orange-green-blue-gray'
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
    ticks = get_time(filename)
    time = ticks.UTC[0]
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
    timebox.font.size = 28
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
        view.magnification = 4.7
        oa_position = [91,7]
    elif setting == 'iso_tail':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,137,64)
        view.position = (-490,519,328)
        view.magnification = 6.470
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
                  xmax=20, xmin=-65, ymax=35, ymin=-35, zmax=35, zmin=-35,
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
        #x
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

def manage_zones(frame, nslice, translucency, cont_num, zone_hidekeys, *,
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
    #hide all other zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
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
                show_list.append(zone.name)
    #Transluceny and shade settings
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            frame.plot(PlotType.Cartesian3D).use_translucency=True
            plt.fieldmap(map_index).effects.use_translucency=True
            plt.fieldmap(map_index).effects.surface_translucency=(
                                                              translucency)
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
    #Get corresponding satellite fieldmap variable indices
    satindices, loc_satindices = [], []
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
            svals = dataset.zone(name).values('Status').as_numpy_array()
            xpos = xvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            ypos = yvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            zpos = zvals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            status = svals[np.where(abs(tvals-deltadt.seconds) < 3)][0]
            loc_satzone.values('X *')[0] = xpos
            loc_satzone.values('Y *')[0] = ypos
            loc_satzone.values('Z *')[0] = zpos
            loc_satzone.values('Status')[0] = status
        #add new zone index
        loc_satindices.append(int(loc_satzone.index))
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
    plot.contour(2).variable_index = int(dataset.variable('Status').index)
    plot.contour(2).colormap_name = 'Diverging - Orange/Purple'
    plot.contour(2).levels.reset_levels([-1,0,1,2,3])
    plot.contour(2).legend.position[1] = 46
    plot.contour(2).legend.position[0] = 98
    plot.contour(2).legend.box.box_type = TextBox.Filled
    #Set color, linestyle, marker and sizes for satellite
    for sat in enumerate(satindices):
        index = loc_satindices[sat[0]]
        #getstatus
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
            plot.fieldmap(index).scatter.symbol().shape = GeomShape.Octahedron
            plot.fieldmap(index).scatter.color = Color.Yellow
            plot.fieldmap(index).scatter.size = 2
        else:
            plot.fieldmap(index).scatter.symbol().shape = GeomShape.Cube
            plot.fieldmap(index).scatter.color = Color.Purple
            plot.fieldmap(index).scatter.size = 1.5



def display_single_iso(frame, contour_key, filename, *, energyrange=3e9,
                       save_img=True, pngpath='./', save_plt=False,
                       pltpath='./', outputname='output', mhddir='./',
                       show_contour=True, show_slice=True, transluc=1,
                       show_fieldline=False, do_blanking=True, tile=False,
                       show_timestamp=True, mode='iso_day', satzones=[],
                       plot_satellites=False, energy_contourmap=0,
                       mpslice=60, cpsslice=20, zone_rename=None,
                       show_legend=True, add_clock=False,
                       zone_hidekeys=['sphere','box','lcb', 'shue']):
    """Function adjusts viewsettings for a single panel isometric 3D image
    Inputs
        frame- object for the tecplot frame
        contour_key- string key for which contour variable to plot
        filename
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
        zone_rename- optional rename of zone
    """
    ###Always included
    #Add colormaps
    path = os.getcwd()+'/energetics.map'
    tp.macro.execute_command('$!LOADCOLORMAP "'+path+'"')
    #set background color
    frame.background_color = Color.Custom17
    #zones
    zones_shown = manage_zones(frame, mpslice, transluc, energy_contourmap,
                               zone_hidekeys)
    if mode == 'inside_from_tail':
        xtail = -15
    else:
        xtail = -45
    set_3Daxes(frame, xmin=xtail)
    set_camera(frame, setting=mode)
    add_energy_contour(frame, energyrange, contour_key, energy_contourmap,
                       show_legend)
    add_earth_iso(frame, rindex=frame.dataset.variable('r *').index)
    #Optional items
    if show_slice:
        add_jy_slice(frame, frame.dataset.variable('J_y *').index,
                     show_legend)
        #add_jz_slice(frame, jzindex=frame.dataset.variable('J_z *').index,
                     #showleg=show_legend)
    if show_fieldline:
        add_fieldlines(frame)
    if tile:
        #hide global zone
        frame.plot().fieldmap(0).show=False
        #Increase axis labels
        frame.plot().axes.x_axis.tick_labels.font.size = 5
        frame.plot().axes.y_axis.tick_labels.font.size = 5
        frame.plot().axes.z_axis.tick_labels.font.size = 5
        frame.plot().axes.orientation_axis.size = 15
        frame.plot().axes.orientation_axis.position = [50,75]
        #tile
        proc = 'Multi Frame Manager'
        cmd = 'MAKEFRAMES3D ARRANGE=TOP SIZE=50'
        tp.macro.execute_extended_command(command_processor_id=proc,
                                          command=cmd)
        #hide orientation axis for small frames
        for fr in tp.frames('Frame *'):
            fr.plot().axes.orientation_axis.show=False
        #Change where text will be generated
        timestamp_pos = (4,15)
        shade_legend_pos = [5,50]
        shade_markersize = 1.5
        frame.plot().fieldmap(0).show=True
        #shift main image
        frame.plot().view.translate(x=20, y=5)
        frame.plot().view.magnification = 4.7

    else:
        timestamp_pos = (4,15)
        shade_legend_pos = [5,70]
        shade_markersize = 3
    if show_contour:
        frame.plot().show_contour = show_contour
    else:
        pass
        #add_shade_legend(frame, zones_shown, shade_legend_pos,
        #                 shade_markersize)
    if plot_satellites:
        if satzones == []:
            print('No satellite zones to plot')
        else:
            set_satellites(satzones, frame)
    if show_timestamp:
        add_timestamp(frame, filename, timestamp_pos)
    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
    if save_img:
        #multiframe image (default)
        tp.export.save_png(pngpath+outputname+'.png', width=3200)
        #each frame in a separate directory
        for fr in tp.frames():
            #make sure the directory for each frame image is there
            if not os.path.exists(pngpath+fr.name):
                os.system('mkdir '+pngpath+fr.name)
            tp.export.save_png(pngpath+fr.name+'/'+outputname+'.png',
                               region=fr, width=3200)
    if add_clock:
        #get clock angle from probing data at x=xmax
        clock = float(frame.dataset.zone('global_field').aux_data[
                                                          'imf_clock_deg'])
        coordsys = frame.dataset.zone('global_field').aux_data[
                                                            'COORDSYSTEM']
        add_IMF_clock(frame, clock, coordsys, (0,0), 30)
    if save_plt:
        tp.data.save_tecplot_plt(pltpath+outputname+'.plt',
                                 include_data_share_linkage=True,
                                 include_autogen_face_neighbors=True)


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
