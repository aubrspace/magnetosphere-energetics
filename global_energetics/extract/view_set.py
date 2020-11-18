#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import tecplot as tp
from tecplot.constant import *
import numpy as np
from global_energetics.makevideo import get_time
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import abs_to_timestamp


def twodigit(num):
    """Function makes two digit str from a number
    Inputs
        num
    Ouputs
        num_str
    """
    return '{:.0f}{:.0f}'.format(np.floor(num/10),num%10)

def display_boundary(frame, contour_key, filename, *, magnetopause=True,
                     plasmasheet=True, colorbar_range=0.25,
                     fullview=True, save_img=True, pngpath='./',
                     outputname='output', show_contour=True, nslice=40):
    """Function to center a boundary object and adjust colorbar
        settings
    Inputs
        frame- object for the tecplot frame
        filename
        contour_key- string key for which contour variable to plot
        colorbar- levels for colorbar
        fullview- True for global view of mangetopause, false for zoomed
        save_img- default True
        pngpath- path for saving .png file
        outputname- default is 'output.png'
    """
    plt = frame.plot()
    field_data = frame.dataset
    colorbar = np.linspace(-1*colorbar_range, colorbar_range,
                           int(4*(colorbar_range*10)+1))
    #create list of zones to be displayed based on inputs
    zone_list = ['global_field']
    #zone_list = []
    if magnetopause:
        zone_list.append('mp_zone')
    if plasmasheet:
        zone_list.append('cps_zone')
    #hide all other zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if not any([zone.name.find(item)!=-1 for item in zone_list]):
                plt.fieldmap(map_index).show = False
            elif zone.name != 'global_field':
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = (
                                            SurfacesToPlot.IKPlanes)
                plt.fieldmap(map_index).surfaces.i_range = (-1,-1,1)
                plt.fieldmap(map_index).surfaces.k_range = (0,-1,nslice-1)

    #mesh, contour and basic zoomed out position for consistency
    plt.show_mesh = False
    plt.show_contour = show_contour
    view = plt.view
    view.center()
    plt.fieldmap(0).show = False

    #mp and cps surface settings
    if magnetopause and plasmasheet:
        for map_index in plt.fieldmaps().fieldmap_indices:
            for zone in plt.fieldmap(map_index).zones:
                if zone.name.find('mp_zone') != -1:
                    plt.fieldmap(map_index).effects.use_translucency=True
                    frame.plot(PlotType.Cartesian3D).use_translucency=True
                    plt.fieldmap(map_index).effects.surface_translucency=60
                    plt.fieldmap(map_index).shade.color = Color.Custom34
                else:
                    plt.fieldmap(map_index).effects.use_translucency=True
                    frame.plot(PlotType.Cartesian3D).use_translucency=True
                    plt.fieldmap(map_index).effects.surface_translucency=60
                    plt.fieldmap(map_index).shade.color = Color.Custom15
                    plt.fieldmap(map_index).contour.show = False

    else:
        for map_index in plt.fieldmaps().fieldmap_indices:
            for zone in plt.fieldmap(map_index).zones:
                if zone.name.find('mp_zone') != -1:
                    plt.fieldmap(map_index).effects.use_translucency=True
                    frame.plot(PlotType.Cartesian3D).use_translucency=True
                    plt.fieldmap(map_index).effects.surface_translucency=30
                    plt.fieldmap(map_index).shade.color = Color.Custom34
    if fullview:
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)

    if show_contour:
        contourvar = field_data.variable(contour_key).index
        contour = plt.contour(0)
        contour.variable_index = contourvar
        contour.colormap_name = 'cmocean - balance'
        contour.legend.vertical = True
        contour.legend.position[1] = 98
        contour.legend.position[0] = 98
        contour.legend.box.box_type = TextBox.Filled
        contour.levels.reset_levels(colorbar)
        contour.labels.step = 2
        contour.colormap_filter.distribution=ColorMapDistribution.Continuous
        contour.colormap_filter.continuous_max = colorbar[-1]
        contour.colormap_filter.continuous_min = colorbar[0]

    #create iso-surface of r=1 for the earth
    plt.show_isosurfaces = True
    iso = plt.isosurface(0)
    iso.definition_contour_group_index = 5
    iso.contour.flood_contour_group_index = 5
    plt.contour(5).variable_index = 14
    iso.isosurface_values[0] = 1
    plt.contour(5).colormap_name = 'Sequential - Green/Blue'
    plt.contour(5).colormap_filter.reversed = True
    plt.contour(5).legend.show = False

    #add scale then turn on fielddata map
    plt.axes.axis_mode = AxisMode.Independent
    plt.axes.x_axis.show = True
    plt.axes.x_axis.max = 15
    plt.axes.x_axis.min = -40
    plt.axes.x_axis.scale_factor = 1
    plt.axes.x_axis.title.position = 30
    plt.axes.x_axis.title.color = Color.White
    plt.axes.x_axis.tick_labels.color = Color.White
    plt.axes.y_axis.show = True
    plt.axes.y_axis.max = 40
    plt.axes.y_axis.min = -40
    plt.axes.y_axis.scale_factor = 1
    plt.axes.y_axis.title.position = 30
    plt.axes.y_axis.title.color = Color.White
    plt.axes.y_axis.tick_labels.color = Color.White
    plt.axes.z_axis.show = True
    plt.axes.z_axis.max = 40
    plt.axes.z_axis.min = -40
    plt.axes.z_axis.scale_factor = 1
    plt.axes.z_axis.title.offset = -8
    plt.axes.z_axis.title.color = Color.White
    plt.axes.z_axis.tick_labels.color = Color.White
    plt.axes.grid_area.fill_color = Color.Custom1
    frame.background_color = Color.Black
    plt.fieldmap(0).show = True

    #Set camera angle
    view.alpha, view.theta, view.psi = (0,137,64)
    view.position = (-490,519,328)
    view.magnification = 5.470

    #add slice at Y=0
    plt.show_slices = True
    yslice = plt.slice(0)
    yslice.orientation = SliceSurface.YPlanes
    yslice.origin[1] = -.10
    yslice.contour.flood_contour_group_index = 1
    yslice.effects.use_translucency = True
    jycontour = plt.contour(1)
    jycontour.variable_index=12
    jycontour.colormap_name = 'Diverging - Brown/Green'
    jycontour.legend.vertical = True
    jycontour.legend.position[1] = 72
    jycontour.legend.position[0] = 98
    jycontour.legend.box.box_type = TextBox.Filled
    jycontour.levels.reset_levels(np.linspace(-0.005,0.005,11))
    jycontour.labels.step = 2
    jycontour.colormap_filter.distribution=ColorMapDistribution.Continuous
    jycontour.colormap_filter.continuous_max = 0.003
    jycontour.colormap_filter.continuous_min = -0.003
    jycontour.colormap_filter.reversed = True

    #add B field lines seeded in XZ plane
    plt.show_streamtraces = True
    plt.streamtraces.add_rake([20,0,40],[20,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([10,0,40],[10,0,-40],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-10,0,20],[-10,0,-20],Streamtrace.VolumeLine)
    plt.streamtraces.add_rake([-20,0,10],[-20,0,-10],Streamtrace.VolumeLine)
    plt.streamtraces.color = Color.Custom41
    plt.streamtraces.line_thickness = 0.2

    #blanking outside 3D axes
    plt.value_blanking.active = True
    xblank = plt.value_blanking.constraint(1)
    xblank.active = True
    xblank.variable = frame.dataset.variable('X *')
    xblank.comparison_operator = RelOp.LessThan
    xblank.comparison_value = -40
    zblank = plt.value_blanking.constraint(2)
    zblank.active = True
    zblank.variable = frame.dataset.variable('Z *')
    zblank.comparison_operator = RelOp.LessThan
    zblank.comparison_value = -40

    #add timestamp
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
    timebox = frame.add_text(time_text)
    timebox.position = (20,16)
    timebox.font.size = 28
    timebox.font.bold = False
    timebox.font.typeface = 'Helvetica'
    timebox.color = Color.White

    #move orientation axis out of the way
    plt.axes.orientation_axis.size = 8
    plt.axes.orientation_axis.position = [91, 7]
    plt.axes.orientation_axis.color = Color.White


    if save_img:
        tp.export.save_png(pngpath+outputname+'.png', width=3200)


def bargraph_setup(frame, color, barid, axis_title, axis_range, *,
                   var_index=0, newaxis=True):
    """Function to display bargraph of variable quantity in upper left
    Inputs
        frame- tecplot frame object
        var_index- index for displayed variable (2 for integrated qty)
        color
        barid- for multiple bars, setup to the right of the previous
        axis_title
        axis_range- expects two element numpy array object
        newaxis- True if plotting on new axis
    """
    #Position frame
    frame.position = [1.75+0.25*barid, 0.5]
    frame.width = 1.5
    frame.height = 4
    frame.show_border = False
    frame.transparent = True
    frame.plot_type = PlotType.XYLine
    #Adjust what is shown in frame
    plt = frame.plot(PlotType.XYLine)
    plt.linemap(0).show = False
    plt.linemap(var_index).show = True
    plt.show_symbols = False
    plt.show_bars = True
    plt.linemap(var_index).bars.show = True
    plt.linemap(var_index).bars.size = 8
    plt.linemap(var_index).bars.line_color = color
    plt.linemap(var_index).bars.fill_color = color
    plt.view.translate(x=-10,y=0)
    plt.axes.x_axis(0).show = False
    plt.axes.y_axis(0).title.title_mode = AxisTitleMode.UseText
    plt.axes.y_axis(0).title.text= axis_title
    plt.axes.y_axis(0).min = axis_range.min()
    plt.axes.y_axis(0).max = axis_range.max()
    #Adjust axis settings if shown
    if newaxis:
        plt.axes.y_axis(0).show = True
        plt.axes.y_axis(0).line.offset = -20
        plt.axes.y_axis(0).title.offset = 20
        frame.transparent = False
        if barid > 2:
            plt.axes.y_axis(0).line.alignment = AxisAlignment.WithGridMax
            plt.axes.y_axis(0).line.offset = -30
            plt.axes.y_axis(0).title.offset = 20
            plt.axes.y_axis(0).tick_labels.offset = 5
    else:
        plt.axes.y_axis(0).show = False

def integral_display(searchkey, *, left_aligned=True, show_influx=True,
                     show_netflux=True, show_outflux=True,
                     save_img=True, pngpath='./', outputname='output.png',
                     show_contour=True):
    '''Function to adjust settings for colorbars to be displayed
    Inputs
        searchkey- string used to identify appropriate integral frames
        left_aligned- default position is upper left corner, boosts barid
                      if false for multiple quanties on one plot
        show_influx
        show_netflux
        show_outflux
        pngpath- path for saving .png file
        outputname- default is 'output.png'
        save_img- default True
    '''
    #Set frame objects for selected zone
    framelist = []
    for frame in tp.frames(searchkey+'*'):
        if frame.name.find('K_in') != -1:
            framelist.append(frame)
        if frame.name.find('K_net') != -1:
            framelist.append(frame)
        if frame.name.find('K_out') != -1:
            framelist.append(frame)

    #set bar id
    if left_aligned:
        bar_id = 0
    else:
        bar_id = 3

    #Create color pallete
    pallete = [Color.Red, Color.Black, Color.Blue, Color.Orange,
               Color.Green, Color.Purple]

    #Set axis range for variables
    if (framelist[0].name.find('mp') != -1 or
        framelist[0].name.find('magnetopause') !=-1):
        axis_range = np.array([-14000, 14000])
    if (framelist[0].name.find('cps') != -1 or
        framelist[0].name.find('plasmasheet') !=-1):
        axis_range = np.array([-100, 100])

    #Setup each frame in framelist
    for frame in reversed(framelist):
        frame.activate()
        show_axis = (bar_id==0 or bar_id==5)
        bargraph_setup(frame, pallete[bar_id], bar_id,
                       frame.name, axis_range, newaxis=show_axis)
        bar_id +=1
        frame.move_to_top()

    if save_img:
        save_to_png(outputname=outputname, pngpath=pngpath)

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
