#!/usr/bin/env python3
"""Controls view settings in tecplot for primary output
"""
import tecplot as tp
from tecplot.constant import *
import numpy as np

def display_magnetopause(frame, zoneid, contourvar, colorbar, multiframe):
    """Function to center the magnetopause object and adjust colorbar
        settings
    Inputs
        frame- object for the tecplot frame
        zoneid- index for zone of interest
        contourvar- variable to be used for the contour
        colorbar- levels for colorbar
        multiframe- boolean, false for single frame
    """
    plt = frame.plot()
    for zone in range(0,zoneid-1):
            plt.fieldmap(zone).show=False
    if not multiframe:
        plt.fieldmap(zoneid).show = True
        plt.fieldmap(zoneid).surfaces.surfaces_to_plot = (
                                            SurfacesToPlot.BoundaryFaces)
        plt.show_mesh = True
        plt.show_contour = True
        view = plt.view
        view.center()
        view.zoom(xmin=-40,xmax=-20,ymin=-90,ymax=10)
        contour = plt.contour(0)
        contour.variable_index = contourvar
        contour.colormap_name = 'cmocean - balance'
        contour.legend.vertical = False
        contour.legend.position[1] = 20
        contour.legend.position[0] = 75
        contour.levels.reset_levels(colorbar)
        contour.labels.step = 2

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
