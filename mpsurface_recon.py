#!/usr/bin/env python3
"""Script for turning 3D magnetopause data into a smooth connected zone
"""

#import os
#import cv2
import sys
import numpy as np
from numpy import pi
import time
import matplotlib as mpl
import matplotlib.pyplot as plot
import scipy
from scipy import interpolate as interp
import pandas as pd
import tecplot as tp

START = time.time()

def yz_slicer(zone,x_min, x_max, n_slice, n_theta, show):
    """Function loops through x position to create 2D closed curves in YZ
    Inputs
        zone- pandas DataFrame containing stream line 3D data
        x_min
        x_max
        n_slice- must be >= 2
        n_theta
        show- True for plotting
    Outputs
        mesh- mesh of X,Y,Z points in a pandas DataFrame object
    """
    dx = (x_max-x_min)/(2*(n_slice-1))
    mesh = pd.DataFrame(columns = ['X', 'Y', 'Z', 'alpha', 'r'])
    print(mesh)
    k = 0
    for x in np.linspace(x_min, x_max-dx, n_slice):
        #Gather data within one x-slice
        zone_temp = zone[(zone['X [R]'] < x+dx) & (zone['X [R]'] > x-dx)]

        if show:
            #create plot and dump raw data
            fig, ax = plot.subplots()
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='r',
                       label='raw')

        #add radial column and determine r_min
        radius = pd.DataFrame(
                    np.sqrt(zone_temp['Z [R]']**2+zone_temp['Y [R]']**2),
                              columns = ['r'])
        zone_temp = zone_temp.combine(radius, np.minimum, fill_value=1000)
        r_min = 0.5 * zone_temp['r'].max()

        #Sort values by YZ angle and remove r<r_min points
        zone_temp = zone_temp[zone_temp['r']>r_min]
        angle = pd.DataFrame(np.arctan2(zone_temp['Z [R]'],
                                        zone_temp['Y [R]']),
                             columns=['alpha'])
        zone_temp = zone_temp.combine(angle, np.minimum, fill_value=1000)
        zone_temp = zone_temp.sort_values(by=['alpha'])

        #Bin temp zone into alpha bins and take maximum r value
        n_angle = n_theta * 4
        da = 2*pi/n_angle
        for a in np.linspace(-1*pi+da, pi-da, n_angle-2):
            #if the section has values
            if not zone_temp[(zone_temp['alpha'] < a+da) &
                             (zone_temp['alpha'] > a-da)].empty:
                #identify the index of row with maximum r within section
                max_r_index = zone_temp[
                            (zone_temp['alpha'] < a+da) &
                            (zone_temp['alpha'] > a-da)].idxmax(0)['r']
                #cut out alpha slice except for max_r_index row
                zone_temp = zone_temp[
                                      (zone_temp['alpha'] > a+da) |
                                      (zone_temp['alpha'] < a-da) |
                                      (zone_temp.index == max_r_index)]
        print('\n\n\nX=: {:.2f}'.format(x))
        print("___ %s seconds ___" % (time.time() - START))
        print('\n\n\n')

        #Created closed interpolation of data
        tck, u = interp.splprep([zone_temp['Y [R]'],
                                 zone_temp['Z [R]']],
                                 s=20, per=True)
        y_curve, z_curve = interp.splev(np.linspace(0,1,1000), tck)
        #setup angle bins for mesh loading
        x_load, y_load, z_load = [], [], []
        n_angle = n_theta
        da = 2*pi/n_angle/10
        for a in np.linspace(-1*pi, pi, n_angle):
            #extract point from curve in angle range
            condition = ((np.arctan2(z_curve,y_curve)>a-da) &
                         (np.arctan2(z_curve,y_curve)<a+da))
            y_load = np.extract(condition, y_curve)[0]
            z_load = np.extract(condition, z_curve)[0]
            x_load = x
            mesh = mesh.append(pd.DataFrame([[x_load, y_load, z_load]],
                                        columns = ['X','Y','Z']),
                                ignore_index=True)

        #duplicate first value to make overlapping surface
        x_index, y_index, z_index = 0,1,2
        mesh.iloc[-1,x_index] = mesh.iloc[-50,x_index]
        mesh.iloc[-1,y_index] = mesh.iloc[-50,y_index]
        mesh.iloc[-1,z_index] = mesh.iloc[-50,z_index]

        if show:
            #plot interpolated data and save figure
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='green',
                       label='remaining')
            ax.plot(y_curve, z_curve, label='interpolated')
            ax.scatter(y_load, z_load, label ='mesh_final')
            ax.set_xlabel('Y [Re]')
            ax.set_ylabel('Z [Re]')
            ax.set_xlim([-30,30])
            ax.set_ylim([-30,30])
            ax.set_title('X= {:.2f} +/- {:.2f}'.format(x,dx))
            ax.legend(loc='upper left')
            if k<0.9:
                filename = 'slice_log/img-0{:.0f}.png'.format(10*k)
            else:
                filename = 'slice_log/img-{:.0f}.png'.format(10*k)
            plot.savefig(filename)
            k = k+.1

    if show:
        #write out log file with parameters listed
        with open('slice_log/slice.log.txt','w+') as log:
            log.write('SLICE FUNCTION PARAMETERS:\n'+
                    '\tFILE: mp_points.csv\n'+
                    '\tXRANGE: {:.2f}-{:.2f}\n'.format(x_min, x_max)+
                    '\tNSLICE: {:.2f}\n'.format(n_slice)+
                    '\tNTHETA: {:.2f}\n'.format(n_theta)+
                    '\tRMIN: {:.2f}\n'.format(r_min)+
                    '\tINTERP S: 20\n')

    return mesh

def show_video(image_folder):
    """Function to save images into a video for visualization
    Inputs
        image_folder
    """
    from makevideo import vid_compile
    framerate = 2
    title = 'video'
    vid_compile(image_folder, framerate, title)

#main program
'''Run program with -v to capture 2D images of sliced data and compile into video
'''
if __name__ == "__main__":
    if '-v' in sys.argv:
        SHOW_VIDEO = True
    else:
        SHOW_VIDEO = False
    #Read in data values and sort by X position
    ZONE = pd.read_csv('mp_points.csv')
    ZONE = ZONE.drop(columns=['Unnamed: 3'])
    ZONE = ZONE.sort_values(by=['X [R]'])
    X_MAX = ZONE['X [R]'].max()
    #X_MIN = ZONE['X [R]'].min()
    X_MIN = -20

    #Slice and construct XYZ data
    MESH = yz_slicer(ZONE, X_MIN, X_MAX, 50, 50, SHOW_VIDEO)
    MESH = MESH.drop(columns=['r','alpha'])
    #MESH.to_hdf('slice_mesh.h5', format='table', key='MESH', mode='w')
    MESH.to_csv('slice_mesh.csv', index=False)

    #Plot slices with finalized points
    if SHOW_VIDEO:
        show_video('slice_log')

    print("___ %s seconds ___" % (time.time() - START))
