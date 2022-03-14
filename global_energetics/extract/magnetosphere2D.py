#/usr/bin/env python
"""script for analyzing 2D swmf files
"""
import sys
import os
import glob
import time
import numpy as np
import pandas as pd
import tecplot as tp
from tecplot.data.extract import triangulate
from tecplot.data.operate import interpolate_linear
from tecplot.data.query import probe_at_position as probe
from tecplot.constant import *
#interpackage
from global_energetics.makevideo import get_time
from global_energetics.preplot import load_hdf5_data, IDL_to_hdf5
from global_energetics.extract.stream_tools import standardize_vars
from global_energetics.extract.stream_tools import get_global_variables
from global_energetics.extract.stream_tools import streamfind_bisection
from global_energetics.extract.view_set import display_2D_contours

def set_yaxis(mode='Z'):
    """function changes the y axis variable between Z and Y
    inputs
        mode (str)- select between Z and Y
    """
    if mode=='Z':
        tp.active_frame().plot().axes.y_axis.variable = (
                tp.active_frame().dataset.variable('Z *'))
    elif mode=='Y':
        tp.active_frame().plot().axes.y_axis.variable = (
                tp.active_frame().dataset.variable('Y *'))

def save_tofile(infile,timestamp,*,outputdir='localdbug/2Dcuts/',xloc=-10,
                filetype='hdf',**points):
    """function saves data to an HDF file
    Inputs
        infile (str)- input filename, used to generate according outputfile
        timestamp (datetime)- datime object
        xloc (float)- used to change header names to retain info
        points:
            dict(list of values)- typically single valued list
    """
    df = pd.DataFrame(points)
    df['time']=timestamp
    df = df.add_suffix('X_'+str(xloc))
    #output filename
    outfile = infile.split('/')[-1].split('e')[-1].split('.out')[0]
    if 'hdf' in filetype:
        df.to_hdf(outputdir+outfile+'.h5', key='mp_points')
    if 'ascii' in filetype:
        df.to_csv(outputdir+outfile+'.dat',sep=' ',index=False)

def get_local_newell(zone,xloc,**kwargs):
    """function calculates newell function given upstream solar wind
        conditions
    inputs
        zone (Zone)- tecplot zone with the solar wind solution
        xloc (float)- where to pull the SW values
        kwargs
    return
        newell (float)- value Wb/s
    """
    probe_result = probe(xloc,0,zones=[zone])
    sw = dict(zip(zone.dataset.variable_names,probe_result.data))
    vsw = 1000*np.sqrt(sw['U_x [km/s]']**2+
                       sw['U_y [km/s]']**2+sw['U_z [km/s]'])
    clock = np.arctan2(sw['B_y [nT]'],sw['B_z [nT]'])
    Bt = np.sqrt(sw['B_y [nT]']**2+sw['B_z [nT]']**2)*1e-3
    Cmp = 1000 #Followed
    #https://supermag.jhuapl.edu/info/data.php?page=swdata
    #term comes from Cai and Clauer[2013].
    #Note that SuperMAG lists the term as 100,
    #however from the paper: "From our work,
    #                       α is estimated to be on order of 10^3"
    newell = Cmp * vsw**(4/3) * Bt**(2/3) * abs(np.sin(clock/2))**(8/3)

def get_night_mp_points(zone,xloc,**kwargs):
    """Function finds location of magnetopause in plane at xlocation
    inputs
        zone (Zone)- zone to search for data
        xloc (float)- xvalue
        kwargs:
            mpvar (str)- key for finding magnetopause variable
            plane (str)- default 'XZ'
            tol (float)- default 0.2
    returns
        rmax,rmin (float,float)- locations of magnetopause
    """
    #Copy data to numpy arrays
    mp = ds.zone(zone).values(kwargs.get('mpvar','mpXZ')).as_numpy_array()
    X = ds.zone(zone).values('X *').as_numpy_array()
    if kwargs.get('plane','XZ')=='XZ':
        R = ds.zone(zone).values('Z *').as_numpy_array()
    elif kwargs.get('plane','XZ')=='XY':
        R = ds.zone(zone).values('Y *').as_numpy_array()
    rvals = R[(abs(X-xloc)<kwargs.get('tol',1)) & (mp==1)]
    if len(rvals)==0:
        return 'None','None'
    else:
        return rvals.max(), rvals.min()

def get_nose(zone,**kwargs):
    """Function finds standoff distance
    inputs
        zone (Zone)- zone to search for data
        kwargs:
    """
    #Copy data to numpy arrays
    mp = ds.zone(zone).values(kwargs.get('mpvar','mpXZ')).as_numpy_array()
    X = ds.zone(zone).values('X *').as_numpy_array()
    nose = X[(mp==1)].max()
    #Save the field strength and nose value on the dayside for later
    B = zone.values('B_z *').as_numpy_array()
    ds.aux_data['daysideB'] = B[(X==nose)&(mp==1)].max()
    ds.aux_data['nose'] = nose
    return X[(mp==1)].max()

def get_XY_magnetopause(ds,**kwargs):
    """Function finds magnetopause in XZ plane given 2D data
    Inputs
        ds (Dataset)- tecplot dataset object
        kwargs:
            dayside_closedB (float)- value for determining closed field
    """
    #Triangulate data from unstructured to FE 2D zone
    set_yaxis(mode='Y')
    zone = triangulate(ds.zone(kwargs.get('XY_zone_index',1)))
    zone.name = 'XYTriangulation'
    #Calculate standard variables if not already there:
    if 'beta_star' not in ds.variable_names:
        get_global_variables(ds, '2DMagnetopause', is3D=False)
    eq = tp.data.operate.execute_equation
    if 'daysideB' in ds.aux_data:
        eq('{closedXY}=IF({B_z [nT]}>'+ds.aux_data['daysideB']+
                       '&&{X [R]}>0&&{X [R]}<'+ds.aux_data['nose']+',1,0)',
                       zones=[1])
    else:
        eq('{closedXY}==IF({B_z [nT]}>50&&{X [R]}>0&&{X [R]}<10,1,0)')
    eq('{mpXY} = IF({X [R]}>-20&&{X [R]}<20&&({beta_star}<0.7 ||'+
                                                    '{closedXY}==1),1,0)')

def get_XZ_magnetopause(ds,**kwargs):
    """Function finds magnetopause in XZ plane given 2D data
    Inputs
        ds (Dataset)- tecplot dataset object
        kwargs:
    """
    #Triangulate data from unstructured to FE 2D zone
    set_yaxis()
    zone = triangulate(ds.zone(kwargs.get('XZ_zone_index',0)))
    zone.name = 'XZTriangulation'
    #Calculate standard variables:
    get_global_variables(ds, '2DMagnetopause', is3D=False)
    #Find last "closed" fieldline in XZ, turn  into an area zone
    day_streamzone = streamfind_bisection(ds,'daysideXZ',
                                          None,10, 30, 3, 100, 0.1)
    day_closed_zone = triangulate(ds.zone(day_streamzone))
    day_closed_zone.name='day_streamtrace_triangulation'
    night_streamzone = streamfind_bisection(ds,'inner_magXZ',
                                            None,10, -30, -3, 100, 0.1)
    night_closed_zone = triangulate(ds.zone(night_streamzone))
    night_closed_zone.name='night_streamtrace_triangulation'
    eq = tp.data.operate.execute_equation
    eq(equation='{closed} = 1', zones=[day_closed_zone])
    eq(equation='{closed} = 1', zones=[night_closed_zone])
    interpolate_linear(destination_zone=zone,fill_value=0,
                         source_zones=[day_closed_zone, night_closed_zone],
                                   variables=[ds.variable('closed').index])
    #Create magnetopause state variable
    eq('{mpXZ} = IF({X [R]}>-20&&{X [R]}<20&&({beta_star}<0.7 ||'+
                                                    '{closed}==1),1,0)')

if __name__ == "__main__":
    inputfiles = []
    start_time = time.time()
    for arg in sys.argv:
        #if arg.endswith('.out'):
        #    inputfiles.append(arg)
        if '-c' in sys.argv:
            tp.session.connect()
    datapath = 'localdbug/2Dcuts/'
    ypath = 'y=0_var_1_e20140218-082700-000_20140220-022700-000/'
    zpath = 'z=0_var_2_e20140218-082700-000_20140220-022700-000/'
    inputfiles = glob.glob(datapath+ypath+'*.out')
    with tp.session.suspend():
        for infile in inputfiles[0:1]:
            #Get matching file
            matchfile = (datapath+zpath+'z=0_var_2'+
                                             infile.split('y=0_var_1')[-1])
            #Load files in
            tp.new_layout()
            zfile = IDL_to_hdf5(infile)
            yfile = IDL_to_hdf5(matchfile)
            XYvars=['/x','/y','/Bx','/By','/Bz','/jx','/jy','/jz',
                     '/P','/Rho','/Ux','/Uy','/Uz','/b1x','/b1y','/b1z']
            XYZvars=['/x','/y','/z','/Bx','/By','/Bz','/jx','/jy','/jz',
                     '/P','/Rho','/Ux','/Uy','/Uz','/b1x','/b1y','/b1z']
            load_hdf5_data(os.getcwd()+'/'+zfile)
            load_hdf5_data(os.getcwd()+'/'+yfile,in_variables=XYvars,
                           variable_name_list=XYZvars)
            ds = tp.active_frame().dataset
            XZ,XY = ds.zone(0), ds.zone(1)
            standardize_vars()

            #get_timestamp
            timestamp = get_time(infile)

            ##XZ plane
            get_XZ_magnetopause(ds, XZ_zone_index=0)
            #Get mp points
            zmax,zmin = get_night_mp_points(ds.zone('XZTriangulation'),-10)
            nose = get_nose(tp.active_frame().dataset.zone(2))

            ##XY plane
            get_XY_magnetopause(ds, XY_zone_index=1)
            ymax,ymin = get_night_mp_points(ds.zone('XYTriangulation'),-10,
                                            plane='XY',mpvar='mpXY')
            newell = get_local_newell(ds.zone('XYTriangulation'),xloc=20)

            #savedata
            save_tofile(infile,timestamp,xloc=-10,nose=[nose],
                        ymax=[ymax],ymin=[ymin],zmax=[zmax],zmin=[zmin])
            #Display
            #display_2D_contours(tp.active_frame(),
            #    outputname='localdbug/2Dcuts/test/'+hdffile.split('.h5')[0],
            #                    filename = hdffile)

            #Clean up
            os.remove(os.getcwd()+'/'+zfile)
            os.remove(os.getcwd()+'/'+yfile)
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
