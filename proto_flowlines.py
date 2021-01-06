#/usr/bin/env python
"""prototype for visualizing upstream flowlines for dev mp surface alt
"""
import sys
import os
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import spacepy
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import global_energetics
from global_energetics.extract import view_set
from global_energetics.extract import stream_tools
from global_energetics.extract.stream_tools import create_stream_zone
from global_energetics.extract.stream_tools import check_streamline_closed

def calc_flow_mp(field_data, ntheta, rmax, *, fieldkey='global_field',
                                             flowkey_x='U_x*', rmin=2,
                                             itr_max=20, tolerance=0.1,
                                             rcheck=10):
    """Function to create zones tha makeup mp as defined by upstream flow
    Inputs
        field_data- Tecplot Dataset with flowfield information
        ntheta- number of azimuthal angles for determining radial distance
        rmax- upsteam maximum radial distance on plane x=xmax
        fieldkey- string ID for field data zone from dataset
        flowkey_x- string ID for x velocity variable, will assume for y,z
        rmin- default set to 0
        itr_max, tolderance- parameters for bisection algorithm
    Outputs
       None, creates zones in tecplot
    """
    #get xmax
    xmax = field_data.zone('global_field').values('X *').max()
    xmax = -10

    #set U as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = field_data.variable('U_x*')
    plot.vector.v_variable = field_data.variable('U_y*')
    plot.vector.w_variable = field_data.variable('U_z*')

    #get theta points
    theta = np.linspace(0,360*(1-1/ntheta), ntheta)

    for a in theta:
        #Create initial max/min
        create_stream_zone(field_data, xmax, rmax*cos(deg2rad(a)),
                           rmax*sin(deg2rad(a)), 'maxflow-{}'.format(a),
                           cart_given=True)
        create_stream_zone(field_data, xmax, rmin*cos(deg2rad(a)),
                           rmin*sin(deg2rad(a)), 'minflow-{}'.format(a),
                           cart_given=True)
        #Check that last closed is bounded, delete min/max
        max_closed = check_streamline_closed(field_data, 'maxflow*',
                                             rcheck, None)
        min_closed = check_streamline_closed(field_data, 'minflow*',
                                             rcheck, None)
        field_data.delete_zones(field_data.zone('minflow*'),
                               field_data.zone('maxflow*'))
        if max_closed and min_closed:
            print("WARNING:flowlines closed at {}R_e from YZ".format(rmax))
            create_stream_zone(field_data, xmax, rmax*cos(deg2rad(a)),
                           rmax*sin(deg2rad(a)), 'maxflow-{}'.format(a),
                           cart_given=True)
        elif not max_closed and not min_closed:
            print("WARNING:flowlines good at {}R_e from YZ".format(rmin))
            create_stream_zone(field_data, xmax, rmin*cos(deg2rad(a)),
                           rmin*sin(deg2rad(a)), 'minflow-{}'.format(a),
                           cart_given=True)
        else:
            rmid = (rmax+rmin)/2
            itr = 0
            notfound = True
            rout = rmax
            rin = rmin
            while(notfound and itr < itr_max):
                #create mid
                create_stream_zone(field_data, xmax, rmid*cos(deg2rad(a)),
                           rmid*sin(deg2rad(a)), 'midflow-{}'.format(a),
                           cart_given=True)
                #check midclosed
                mid_closed = check_streamline_closed(field_data,'midflow*',
                                                     rcheck, None)
                if mid_closed:
                    rin = rmid
                else:
                    rout = rmid
                if abs(rout-rin) < tolerance and not mid_closed:
                    notfound = False
                    field_data.zone('midflow*').name='flow-{:.1f}'.format(a)
                else:
                    rmid = (rin+rout)/2
                    field_data.delete_zones(field_data.zone('midflow*'))
                itr += 1

def get_flow_seed_grid(xmax, rmax, *, shape='rings', ndim1=5, ndim2=36,
                                center=True):
    """Function creates grid of points for flowline streaming at x=xmax pln
    Inputs
        xmax- plane where points will be seeded
        rmax- maximum radial (relative to X plane) distance of points
        ndim1- first dimension number of points, default is # of rings
        ndim2- second dim number of points, default is # point in a ring
        center- boolean to include y,z=(0,0)
    Outputs
        np.array(grid)- numpy array of 2D list of [[y,z],...]
    """
    grid = []
    if shape == 'rings':
        #center point
        if center:
            rtemp = 0
            thtemp = 0
            grid.append([rtemp*cos(deg2rad(thtemp)),
                         rtemp*sin(deg2rad(thtemp))])
        #loop through rings
        for r in range(1,ndim1+1):
            for th in range(0,ndim2):
                rtemp = r/ndim1*rmax
                thtemp = th/36*360
                grid.append([rtemp*cos(deg2rad(thtemp)),
                             rtemp*sin(deg2rad(thtemp))])
    else:
        print('NON-RINGMODE NOT DEVELOPED!!!')
        grid= [[0,0]]
    return np.array(grid)

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
    OUTPUTNAME = datafile.split('e')[1].split('-000.')[0]+'-a'

    #python objects
    field_data=tp.data.load_tecplot(sys.argv[1])
    field_data.zone(0).name = 'global_field'

    #set frame object and radial coordinates
    main_frame = tp.active_frame()
    main_frame.name = 'main'
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

    #adjust view settings
    view_set.display_boundary([frame for frame in tp.frames('main')][0],
                              'Rho *', datafile, plasmasheet=False,
                              magnetopause=False, save_img=False,
                              show_contour=False, show_slice=False,
                              show_fieldline=False, do_blanking=False)


    #call function
    calc_flow_mp(field_data, 36, 20)

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(np.int(ltime/60),
                                           np.mod(ltime,60)))
