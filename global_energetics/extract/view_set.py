#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import tecplot as tp
from tecplot.constant import *
import numpy as np

def display_boundary(frame, zoneid, contourvar, colorbar, *,
                     fullview=True):
    """Function to center a boundary object and adjust colorbar
        settings
    Inputs
        frame- object for the tecplot frame
        zoneid- index for zone of interest
        contourvar- variable to be used for the contour
        colorbar- levels for colorbar
        fullview- True for global view of mangetopause, false for zoomed
    """
    plt = frame.plot()
    #hide all non essential zones
    for map_index in plt.fieldmaps().fieldmap_indices:
        for zone in plt.fieldmap(map_index).zones:
            if (zone.name.find('global_field') == -1 and
                zone.name.find('cps_zone') == -1 and
                zone.name.find('mp_zone') == -1):
                plt.fieldmap(map_index).show = False
        plt.fieldmap(zoneid).show = True
        plt.fieldmap(zoneid).surfaces.surfaces_to_plot = (
                                            SurfacesToPlot.BoundaryFaces)
        plt.show_mesh = True
        plt.show_contour = True
        view = plt.view
        view.center()
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

def integral_display(frame, zoneid, *, influx=True, netflux=True,
                     outflux=True):
    '''Function to adjust settings for colorbars to be displayed
    '''
    #adjust main frame settings
    display_magnetopause(main, mp_index, Knet_index, colorbar, False)
    for frames in tp.frames('MP K_in*'):
        influx = frames
    for frames in tp.frames('MP K_net*'):
        netflux = frames
    for frames in tp.frames('MP K_out*'):
        outflux = frames
    outflux.move_to_top()
    netflux.move_to_top()
    influx.move_to_top()
    outflux.activate()
    outflux_df, _ = dump_to_pandas([1],[4],'outflux.csv')
    netflux.activate()
    netflux_df, _ = dump_to_pandas([1],[4],'netflux.csv')
    influx.activate()
    influx_df, _ = dump_to_pandas([1],[4],'influx.csv')

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
