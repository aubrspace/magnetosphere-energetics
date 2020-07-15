#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import tecplot as tp
from tecplot.constant import *
import numpy as np

def display_boundary(frame, contourvar, *, magnetopause=True,
                     plasmasheet=True, colorbar_range=2.5,
                     fullview=True, save_img=True, pngpath='./',
                     outputname='output.png', show_contour=True):
    """Function to center a boundary object and adjust colorbar
        settings
    Inputs
        frame- object for the tecplot frame
        zoneid- index for zone of interest
        contourvar- variable to be used for the contour
        colorbar- levels for colorbar
        fullview- True for global view of mangetopause, false for zoomed
        save_img- default True
        pngpath- path for saving .png file
        outputname- default is 'output.png'
    """
    plt = frame.plot()
    field_data = frame.dataset
    colorbar = np.linspace(-1*colorbar_range, colorbar_range,
                           int(4*colorbar_range+1))
    #create list of zones to be displayed based on inputs
    zone_list = ['global_field']
    if magnetopause:
        zone_list.append('mp_zone')
    if plasmasheet:
        zone_list.append('cps_zone')
    #hide all other zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if not any([zone.name.find(item)!=-1 for item in zone_list]):
                plt.fieldmap(map_index).show = False
                plt.fieldmap(map_index).show = True
            elif zone.name != 'global_field':
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = (
                                            SurfacesToPlot.BoundaryFaces)
    plt.show_mesh = False
    plt.show_contour = show_contour
    view = plt.view
    view.center()

    if magnetopause and plasmasheet:
        for map_index in plt.fieldmaps().fieldmap_indices:
            for zone in plt.fieldmap(map_index).zones:
                if zone.name.find('mp_zone') != -1:
                    plt.fieldmap(map_index).effects.use_translucency=True
                    frame.plot(PlotType.Cartesian3D).use_translucency=True
                    plt.fieldmap(map_index).effects.surface_translucency = 60
                    plt.fieldmap(map_index).shade.color = Color.Custom34
                else:
                    plt.fieldmap(map_index).effects.use_translucency=True
                    frame.plot(PlotType.Cartesian3D).use_translucency=True
                    plt.fieldmap(map_index).effects.surface_translucency = 10
                    plt.fieldmap(map_index).shade.color = Color.Custom15
    if fullview:
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        contour = plt.contour(0)
        contour.variable_index = contourvar
        contour.colormap_name = 'cmocean - balance'
        contour.legend.vertical = False
        contour.legend.position[1] = 20
        contour.legend.position[0] = 75
        contour.levels.reset_levels(colorbar)
        contour.labels.step = 2

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

    if save_img:
        #write .plt and .lay files
        #tp.data.save_tecplot_plt(pltpath+outputname+'.plt')
        #tp.save_layout(laypath+outputname+'.lay')
        tp.export.save_png(pngpath+outputname+'.png')

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
    frame.position = [1.25+0.25*barid, 0]
    frame.width = 2
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
        if barid > 2:
            plt.axes.y_axis(0).line.alignment = AxisAlignment.WithGridMax
            plt.axes.y_axis(0).line.offset = -30
            plt.axes.y_axis(0).title.offset = 20
            plt.axes.y_axis(0).tick_labels.offset = 5
    else:
        plt.axes.y_axis(0).show = False

def integral_display(searchkey, *, left_aligned=True, show_influx=True,
                     show_netflux=True, show_outflux=True):
    '''Function to adjust settings for colorbars to be displayed
    Inputs
        searchkey- string used to identify appropriate integral frames
        left_aligned- default position is upper left corner, boosts barid
                      if false for multiple quanties on one plot
        show_influx
        show_netflux
        show_outflux
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
