#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import tecplot as tp
from tecplot.constant import *
import numpy as np
from progress.bar import Bar
from global_energetics.makevideo import get_time
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import abs_to_timestamp

def manage_camera(frame, *, setting='iso_day'):
    """Function sets camera angle based on setting
    Inputs
        frame- frame on which to change camera
        setting- iso_day, iso_tail, xy, xz, yz
    """
    view = frame.plot().view
    if setting == 'iso_day':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,-120,64)
        view.position = (-490,519,328)
        view.magnification = 5.470
    elif setting == 'iso_tail':
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        view.alpha, view.theta, view.psi = (0,137,64)
        view.position = (-490,519,328)
        view.magnification = 5.470
    else:
        print('Camera setting {} not developed!'.format(setting))

def manage_3Daxes(frame, *,
                  xmax=15, xmin=-40, ymax=40, ymin=-40, zmax=40, zmin=-40,
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
    axes.x_axis.title.color = Color.White
    axes.x_axis.tick_labels.color = Color.White
    axes.y_axis.show = True
    axes.y_axis.max = ymax
    axes.y_axis.min = ymin
    axes.y_axis.scale_factor = 1
    axes.y_axis.title.position = 30
    axes.y_axis.title.color = Color.White
    axes.y_axis.tick_labels.color = Color.White
    axes.z_axis.show = True
    axes.z_axis.max = zmax
    axes.z_axis.min = zmin
    axes.z_axis.scale_factor = 1
    axes.z_axis.title.offset = -8
    axes.z_axis.title.color = Color.White
    axes.z_axis.tick_labels.color = Color.White
    axes.grid_area.fill_color = Color.Custom1
    frame.background_color = Color.Black
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

def manage_zones(frame, nslice, *, approved_zones=None):
    """Function shows/hides zones, sets shading and translucency
    Inputs
        frame- frame object to manage zones on
        nslice- used to show only outer surface, assumes cylindrical
        approved_zones- default shows all in predetermined list
    """
    plt = frame.plot()
    #Generate predetermined approved zone names
    zone_list = ['global_field', 'mp_shue', 'mp_test', 'mp_flowline',
                 'mp_fieldline', 'mp_hybrid', 'cps_zone']
    if approved_zones != None:
        for zone in approved_zones:
            zone_list.append(zone)
    #hide all other zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if not any([zone.name.find(item)!=-1 for item in zone_list]):
                plt.fieldmap(map_index).show = False
            elif zone.name == 'global_field':
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = None
            else:
                #some assertion that checks that zone is cylindrical
                plt.fieldmap(map_index).surfaces.surfaces_to_plot = (
                                                   SurfacesToPlot.IKPlanes)
                plt.fieldmap(map_index).show = True
                plt.fieldmap(map_index).surfaces.i_range = (-1,-1,1)
                plt.fieldmap(map_index).surfaces.k_range = (0,-1, nslice-1)
    #Transluceny and shade settings
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            frame.plot(PlotType.Cartesian3D).use_translucency=True
            plt.fieldmap(map_index).effects.use_translucency=True
            plt.fieldmap(map_index).effects.surface_translucency=40
            if zone.name.find('mp_hybrid') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom20
            if zone.name.find('mp_fieldline') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom34
            if zone.name.find('mp_flowline') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom11
            if zone.name.find('mp_shue') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom2
            if zone.name.find('mp_test') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom7
            if zone.name.find('cps_zone') != -1:
                plt.fieldmap(map_index).shade.color = Color.Custom8

def twodigit(num):
    """Function makes two digit str from a number
    Inputs
        num
    Ouputs
        num_str
    """
    return '{:.0f}{:.0f}'.format(np.floor(num/10),num%10)

def display_single_iso(frame, contour_key, filename, *, energyrange=0.1,
                       save_img=True, pngpath='./', save_plt=True,
                       pltpath='./', outputname='output',
                       show_contour=True, show_slice=True,
                       show_fieldline=True, do_blanking=True, mpslice=60,
                       cpsslice=20, zone_rename=None):
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
        mpslice- number of x slices in mp surface
        cpsslice- number of x slices in cps surface
        zone_rename- optional rename of zone
    """
    manage_zones(frame, mpslice)
    manage_3Daxes(frame)
    manage_camera(frame)

def display_boundary(frame, contour_key, filename, *, magnetopause=True,
                     plasmasheet=True, colorbar_range=0.25,
                     fullview=True, save_img=True, pngpath='./',
                     save_plt=True, pltpath='./',
                     outputname='output', show_contour=True,
                     show_slice=True, show_fieldline=True, do_blanking=True,
                     mpslice=60, cpsslice=20, zone_rename=None):
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
        save_plt- default True
        pltpath- path for saving .png file
        outputname- default is 'output.png'
        show_contour, show_fieldline, do_blanking
        mpslice- number of x slices in mp surface
        cpsslice- number of x slices in cps surface
        zone_rename- optional rename of zone
    """
    plt = frame.plot()
    field_data = frame.dataset
    colorbar = np.linspace(-1*colorbar_range, colorbar_range,
                           int(4*(colorbar_range*10)+1))
    with tp.session.suspend():
    #create list of zones to    be displayed based on inputs
        zone_list = ['global_field']
        if magnetopause:
            zone_list.append('mp_shue')
            zone_list.append('mp_test')
            zone_list.append('mp_flowline')
            zone_list.append('mp_fieldline')
            zone_list.append('mp_hybrid')
        if plasmasheet:
            zone_list.append('cps_zone')
        if zone_rename != None:
            zone_list.append(zone_rename)
        #hide all other zones
        for map_index in plt.fieldmaps().fieldmap_indices:
            for zone in plt.fieldmap(map_index).zones:
                if not any([zone.name.find(item)!=-1 for item in
                                                                zone_list]):
                    plt.fieldmap(map_index).show = False
                elif zone.name != 'global_field':
                    plt.fieldmap(map_index).surfaces.surfaces_to_plot = (
                                                SurfacesToPlot.IKPlanes)
                    plt.fieldmap(map_index).surfaces.i_range = (-1,-1,1)
                    if zone.name == 'mp_zone':
                        plt.fieldmap(map_index).surfaces.k_range = (0,-1,
                                                                mpslice-1)
                    if zone.name == 'cps_zone':
                        plt.fieldmap(map_index).surfaces.k_range = (0,-1,
                                                                cpsslice-1)

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

        print('Creating earth isosurface')
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

        print('Adjusting 3D axes')
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

        print('Setting Camera')
        #Set camera angle
        view.alpha, view.theta, view.psi = (0,137,64)
        view.position = (-490,519,328)
        view.magnification = 5.470

        print('2D Slice Settings')
        if show_slice:
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

        print('Fieldline Settings')
        if show_fieldline:
            #add B field lines seeded in XZ plane
            plt.show_streamtraces = True
            plt.streamtraces.add_rake([20,0,40],[20,0,-40],Streamtrace.VolumeLine)
            plt.streamtraces.add_rake([10,0,40],[10,0,-40],Streamtrace.VolumeLine)
            plt.streamtraces.add_rake([-10,0,20],[-10,0,-20],Streamtrace.VolumeLine)
            plt.streamtraces.add_rake([-20,0,10],[-20,0,-10],Streamtrace.VolumeLine)
            plt.streamtraces.color = Color.Custom41
        plt.streamtraces.line_thickness = 0.2

        print('Blanking')
        if do_blanking:
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

        print('Timestamp')
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

        print('Orientation box')
        #move orientation axis out of the way
        plt.axes.orientation_axis.size = 8
        plt.axes.orientation_axis.position = [91, 7]
        plt.axes.orientation_axis.color = Color.White

    print('Saving output files')
    if save_img:
        tp.export.save_png(pngpath+outputname+'.png', width=3200)
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
