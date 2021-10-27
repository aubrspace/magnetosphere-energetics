#/usr/bin/env python
"""script for analyzing 2D swmf files
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import tecplot as tp
from tecplot.constant import *
#interpackage
from global_energetics.preplot import load_hdf5_data, IDL_to_hdf5
from global_energetics.extract.stream_tools import standardize_vars
from global_energetics.extract.stream_tools import get_global_variables
from global_energetics.extract.stream_tools import streamfind_bisection
from global_energetics.extract.view_set import display_2D_contours


if __name__ == "__main__":
    inputfiles = []
    for arg in sys.argv:
        #if arg.endswith('.out'):
        #    inputfiles.append(arg)
        if '-c' in sys.argv:
            tp.session.connect()
    inputfiles = glob.glob('/Users/ngpdl/pngs/'+
        'y=0_var_1_e20130430-040200-000_20130501-220200-000/*.out')
    with tp.session.suspend():
        for infile in inputfiles:
            #Load files in
            tp.new_layout()
            hdffile = IDL_to_hdf5(infile)
            load_hdf5_data(os.getcwd()+'/'+hdffile)
            standardize_vars()
            ds = tp.active_frame().dataset
            #Triangulate data from unstructured to FE 2D zone
            zone = tp.data.extract.triangulate(ds.zone(0))
            #Calculate standard variables:
            get_global_variables(ds, '2DMagnetopause', is3D=False)
            #Find last "closed" fieldline in XZ, turn  into an area zone
            day_streamzone = streamfind_bisection(ds, 'daysideXZ', None,
                                                  10, 30, 3, 100, 0.1)
            day_closed_zone=tp.data.extract.triangulate(
                                                   ds.zone(day_streamzone))
            night_streamzone = streamfind_bisection(ds,'inner_magXZ', None,
                                                    10, -30, -3, 100, 0.1)
            night_closed_zone = tp.data.extract.triangulate(
                                                 ds.zone(night_streamzone))
            eq = tp.data.operate.execute_equation
            eq(equation='{closed} = 1', zones=[day_closed_zone])
            eq(equation='{closed} = 1', zones=[night_closed_zone])
            tp.data.operate.interpolate_linear(
                         source_zones=[day_closed_zone, night_closed_zone],
                                                     destination_zone=zone,
                                   variables=[ds.variable('closed').index],
                                                              fill_value=0)
            #Create magnetopause state variable
            eq('{mp} = IF({X [R]}>-20&&{X [R]}<11.2&&({beta_star}<0.7 ||'+
                                                    '{closed}==1),1,0)')
            display_2D_contours(tp.active_frame(),
                           outputname='2dtestrun/'+hdffile.split('.h5')[0],
                                filename = hdffile)
