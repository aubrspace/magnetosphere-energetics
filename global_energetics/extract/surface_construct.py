#!/usr/bin/env python3
"""Script for turning 3D magnetopause data into a smooth connected zone
"""

#import os
#import cv2
import sys
import numpy as np
from numpy import pi, sqrt
import time
import matplotlib as mpl
import matplotlib.pyplot as plot
import scipy
from scipy import interpolate as interp
import pandas as pd
import tecplot as tp

START = time.time()

def show_video(image_folder):
    """Function to save images into a video for visualization
    Inputs
        image_folder
    """
    from global_energetics.makevideo import vid_compile
    framerate = 2
    title = 'video'
    vid_compile(image_folder, framerate, title)

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
    mesh = pd.DataFrame(columns = ['X [R]', 'Y [R]', 'Z [R]'])
    k = 0
    for x in np.linspace(x_min, x_max-dx, n_slice):
        #Gather data within one x-slice
        zone_temp = zone[(zone['X [R]'] < x+dx) & (zone['X [R]'] > x-dx)]
        ymean = np.mean(zone['Y [R]'])
        zmean = np.mean(zone['Z [R]'])

        if show:
            #create plot and dump raw data
            fig, ax = plot.subplots()
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='r',
                       label='raw')

        #add radial column and determine r_min
        radius = pd.DataFrame(
                    np.sqrt((zone_temp['Z [R]']-zmean)**2+(
                             zone_temp['Y [R]']-ymean)**2),
                              columns = ['r'])
        zone_temp = zone_temp.combine(radius, np.minimum, fill_value=1000)
        r_min = 0.2 * zone_temp['r'].max()

        #Sort values by YZ angle and remove r<r_min points
        zone_temp = zone_temp[zone_temp['r']>r_min]
        if show:
            #create plot with dropped data
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='y',
                       label='after 1st cut', marker='o')
        #re-establish center point
        ymean = np.mean(zone['Y [R]'])
        zmean = np.mean(zone['Z [R]'])
        #define angle relative to center of field of points
        angle = pd.DataFrame(np.arctan2(zone_temp['Z [R]']-zmean,
                                        zone_temp['Y [R]']-ymean),
                             columns=['alpha'])
        zone_temp = zone_temp.combine(angle, np.minimum, fill_value=1000)
        zone_temp = zone_temp.sort_values(by=['alpha'])

        #Bin temp zone into alpha bins and take maximum r value
        n_angle = n_theta*10
        da = 2*pi/n_angle
        for a in np.linspace(-1*pi, pi, n_angle):
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
        print('X=: {:.2f}'.format(x))

        #Created closed interpolation of data
        y_points = np.r_[zone_temp['Y [R]'].values,
                         zone_temp['Y [R]'].values[0]]
        z_points = np.r_[zone_temp['Z [R]'].values,
                         zone_temp['Z [R]'].values[0]]
        tck, u = interp.splprep([y_points, z_points], s=20, per=True)
        y_curve, z_curve = interp.splev(np.linspace(0,1,1000), tck)
        translated_zcurve = [point-zmean for point in z_curve]
        translated_ycurve = [point-ymean for point in y_curve]
        dat_angles = np.sort(np.arctan2(translated_zcurve,
                                        translated_ycurve))
        print(dat_angles)
        y_load = y_curve[0]
        z_load = z_curve[0]
        #setup angle bins for mesh loading
        spoke, spoketxt = [], []
        n_angle = n_theta
        da = 2*pi/n_angle/5
        for a in np.linspace(-1*pi, pi, n_angle):
            #extract point from curve in angle range
            condition = ((np.arctan2(translated_zcurve,
                                     translated_ycurve)>a-da) &
                         (np.arctan2(z_curve, y_curve)<a+da))
            if condition.any():
                x_load = x
                y_load = np.extract(condition, y_curve)[0]
                z_load = np.extract(condition, z_curve)[0]
            else:
                print("WARNING: No extraction at X={:.2f}, alpha={:.2f}".format(x,a))
                r_previous = sqrt((y_load+ymean)**2+(z_load+zmean)**2)
                y_load = np.interp(r_previous*np.sin(a)+ymean,
                                   y_curve, z_curve, period=2*pi)
                z_load = np.interp(y_load, y_curve, z_curve,
                                   period=2*pi)
                x_load = x
                print('y_load:',y_load)
                print('putting in dummy point at',
                        'X= {:.2f}'.format(x),
                        'Y= {:.2f}'.format(y_load),
                        'Z= {:.2f}'.format(z_load))
            mesh = mesh.append(pd.DataFrame([[x_load, y_load, z_load]],
                               columns = ['X [R]','Y [R]','Z [R]']),
                               ignore_index=True)
            spoke.append(a)
            spoketxt.append('{:.2f}'.format(a))

        #duplicate first value to make overlapping surface
        x_index, y_index, z_index = 0,1,2
        mesh.iloc[-1,x_index] = mesh.iloc[-n_angle,x_index]
        mesh.iloc[-1,y_index] = mesh.iloc[-n_angle,y_index]
        mesh.iloc[-1,z_index] = mesh.iloc[-n_angle,z_index]

        if show:
            #plot interpolated data and save figure
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='green',
                       label='remaining')
            ax.plot(y_curve, z_curve, label='interpolated')
            ax.scatter(y_load, z_load, label ='mesh_final')
            ax.scatter(20*np.cos(spoke), 20*np.sin(spoke), label='spokes')
            for a in enumerate(spoke[::4]):
                ax.annotate(spoketxt[4*a[0]], (20*np.cos(a[1]),
                                    20*np.sin(a[1])))
            ax.set_xlabel('Y [Re]')
            ax.set_ylabel('Z [Re]')
            ax.set_xlim([-20,20])
            ax.set_ylim([-20,20])
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
        show_video('slice_log')

    print("___ %s seconds ___" % (time.time() - START))
    print('\n\n\n')
    return mesh


#main program
'''Run program with -v to capture 2D images of sliced data and compile into video
'''
if __name__ == "__main__":
    if '-v' in sys.argv:
        SHOW_VIDEO = True
    else:
        SHOW_VIDEO = False
    #Read in data values and sort by X position
    ZONE = pd.read_csv('stream_points.csv')
    ZONE = ZONE.drop(columns=['Unnamed: 3'])
    ZONE = ZONE.sort_values(by=['X [R]'])
    X_MAX = ZONE['X [R]'].max()
    #X_MIN = ZONE['X [R]'].min()
    X_MIN = -30

    #Slice and construct XYZ data
    MESH = yz_slicer(ZONE, X_MIN, X_MAX, 50, 50, SHOW_VIDEO)
    #MESH.to_hdf('slice_mesh.h5', format='table', key='MESH', mode='w')
    #MESH.to_csv('slice_mesh.csv', index=False)

    #Plot slices with finalized points
    if SHOW_VIDEO:
        show_video('slice_log')

    print("___ %s seconds ___" % (time.time() - START))
