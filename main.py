#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import numpy as np
from numpy import pi
import datetime as dt
import spacepy
import spacepy.time as spacetime
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import magnetopause
from global_energetics.extract import plasmasheet
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set

if __name__ == "__main__":
    print('\nProcessing {pltfile}\n'.format(pltfile=sys.argv[1]))
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    tp.new_layout()
    #pass in arguments
    datafile = sys.argv[1].split('/')[-1]
    nameout = datafile.split('e')[1].split('-000.')[0]+'-mp'
    print('nameout:'+nameout)
    PNGPATH = sys.argv[2]
    OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-a'

    #python objects
    #tp.load_layout('example5/zoomed.lay')
    #tp.load_layout('example4/backside.lay')
    #field_data = tp.active_frame().dataset
    field_data=tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'

    '''
    #calculate lat and lon
    tp.data.operate.execute_equation(
                    '{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
    tp.data.operate.execute_equation(
                    '{lat [deg]} = 180/pi*asin({Z [R]} / {r [R]})')
    tp.data.operate.execute_equation(
                    '{lon [deg]} = if({X [R]}>0,'+
                                     '180/pi*atan({Y [R]} / {X [R]}),'+
                                  'if({Y [R]}>0,'+
                                     '180/pi*atan({Y [R]}/{X [R]})+180,'+
                                     '180/pi*atan({Y [R]}/{X [R]})-180))')

    #Create ring of field lines at raw north pole and MAG(assume GSE)
    lon_set = np.linspace(-180, 180, 20)
    lat = 89
    simtime = spacetime.Ticktock(dt.datetime(2014,2,18,7,30),'UTC')
    for lon in lon_set:
        stream_tools.create_stream_zone(field_data, 1, lat,lon,'raw_pole_',
                                        'inner_mag')
        x,y,z = stream_tools.mag2gsm(1,lat,lon,simtime)
        stream_tools.create_stream_zone(field_data, x,y,z,'mag_pole_',
                                        'inner_mag', cart_given=True)
    '''


    #Caclulate surfaces
    magnetopause.get_magnetopause(field_data, datafile, nfill=10,
                                  integrate_volume=False,
                                  integrate_surface=True)
    plasmasheet.get_plasmasheet(field_data, datafile)

    #adjust view settings
    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              'K_out *', datafile, magnetopause=True,
                              pngpath=PNGPATH, show_contour=False,
                              outputname=nameout)

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
                         -15,
                         plt.slice(0).origin[1])
    '''

    #save image of streamlines
    #tp.export.save_png(PNGPATH+OUTPUTNAME+'.png')
    #timestamp

    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
