#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import numpy as np
from numpy import pi
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetopause
from global_energetics.extract import plasmasheet
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import view_set

if __name__ == "__main__":
    print('\nProcessing {pltfile}\n'.format(pltfile=sys.argv[1]))
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    #datafile = '3d__mhd_2_e20140219-123000-000.plt'
    #pass in arguments
    datafile = sys.argv[1]
    PNGPATH = sys.argv[2]
    OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-cps_m'

    #python objects
    field_data=tp.data.load_tecplot(datafile)
    field_data.zone(0).name = 'global_field'

    magnetopause.get_magnetopause(field_data, datafile, save_img=False)
    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              field_data.variable('K_in *').index)

    #Create stream zones along meridional plane for visualization
    phi = np.append(np.linspace(-pi,np.deg2rad(-160),int(50/2)),
                    np.linspace(pi,np.deg2rad(160),int(50/2)))
    stream_tools.calc_plasmasheet(field_data, np.deg2rad(55), phi, -20,
                                  100, pi/90)
    '''
    #load already calculated surface
    tp.load_layout('freshview.lay')
    '''

    #Display meridional streamlines for visualization with x scale slice
    plt = tp.active_frame().plot()
    plt.show_mesh = True
    plt.show_contour = True
    plt.fieldmap(1).mesh.show = False
    plt.view.psi = 90
    plt.view.theta = 180
    plt.view.center()
    plt.show_slices = True
    plt.slice(0).contour.flood_contour_group_index = 1
    plt.contour(1).variable_index = 0
    plt.slice(0).orientation = SliceSurface.YPlanes
    plt.view.zoom(xmin=-60,xmax=-50,ymin=-50,ymax=10)
    plt.view.translate(y=-30, x=30)
    x_color_bar = np.linspace(0,-40,11)
    plt.contour(1).levels.reset_levels(x_color_bar)
    plt.slice(0).origin=(plt.slice(0).origin[0],
                         -5,
                         plt.slice(0).origin[1])

    #save image of streamlines
    tp.export.save_png(PNGPATH+OUTPUTNAME+'.png')

    #plasmasheet.get_plasmasheet(field_data, datafile)

    #view_set.integral_display('mp')
    #view_set.integral_display('cps', left_aligned=False)

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
