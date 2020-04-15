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

def yz_slicer(zone,x_min, x_max, n_slice, n_theta):
    """Function loops through x position to create 2D closed curves in YZ
    Inputs
        x_min
        x_max
        n_slice- must be >= 2
        n_theta
    Outputs
        mesh- mesh of X,Y,Z points in a pandas DataFrame object
    """
    dx = (x_max-x_min)/(2*(n_slice-1))/10
    mesh = pd.DataFrame(columns = ['X', 'Y', 'Z'])
    for x in np.linspace(x_min, x_max-dx, n_slice-1):
        #Gather data within one x-slice
        zone_temp = zone[(zone['X [R]'] < x+dx) & (zone['X [R]'] > x-dx)]

        #create plot and dump raw data
        fig, ax = plot.subplots()
        ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='r',
                   label='raw')

        #Sort values by YZ angle and remove r<r_min points
        r_min = 3
        radius = pd.DataFrame(
                      np.sqrt(zone_temp['Z [R]']**2+zone_temp['Y [R]']**2),
                              columns = ['r'])
        zone_temp = zone_temp.combine(radius, np.minimum, fill_value=1000)
        zone_temp = zone_temp[zone_temp['r']>r_min]
        angle = pd.DataFrame(np.arctan2(zone_temp['Z [R]'],
                                        zone_temp['Y [R]']),
                             columns=['alpha'])
        zone_temp = zone_temp.combine(angle, np.minimum, fill_value=1000)
        zone_temp = zone_temp.sort_values(by=['alpha'])

        #Bin temp zone into alpha bins and take minimum r value
        n_angle = 100
        da = 2*pi/n_angle
        for a in np.linspace(-1*pi+da, pi-da, n_angle-2):
            #if the section has values
            if not zone_temp[(zone_temp['alpha'] < a+da) &
                             (zone_temp['alpha'] > a-da)].empty:
                #identify the index of row with minimum r within section
                min_r_index = zone_temp[
                            (zone_temp['alpha'] < a+da) &
                            (zone_temp['alpha'] > a-da)].idxmin(0)['r']
                #cut out alpha slice except for min_r_index row
                zone_temp = zone_temp[
                                      (zone_temp['alpha'] > a+da) |
                                      (zone_temp['alpha'] < a-da) |
                                      (zone_temp.index == min_r_index)]
        print('\n\n\n',x)
        print("___ %s seconds ___" % (time.time() - START))
        print('\n\n\n')

        #Created closed interpolation of data
        tck, u = interp.splprep([zone_temp['Y [R]'],
                                 zone_temp['Z [R]']],
                                 s=20, per=True)
        y_new, z_new = interp.splev(np.linspace(0,1,1000), tck)

        #plot interpolated data and save figure
        ax.plot(y_new, z_new, label='interpolated')
        ax.set_xlabel('Y [Re]')
        ax.set_ylabel('Z [Re]')
        ax.set_title('X= {:.2f} +/- {:.2f}'.format(x,dx))
        ax.legend()
        plot.savefig('slice_log/x_{:.2f}.png'.format(x))

def show_video(image_folder, filename):
    """Function to save images into a video for visualization
    Inputs
        image_folder
        filename
    """
    #image_folder = 'slice_log'
    #video_name = 'video.avi'

    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    #frame = cv2.imread(os.path.join(image_folder, images[0]))
    #height, width, layers = frame.shape

    #video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    #for image in images:
    #    video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    #video.release()
    pass

#main program
if __name__ == "__main__":
    if '-v' in sys.argv:
        SHOW_VIDEO = True
    else:
        SHOW_VIDEO = False
    #Read in data values and sort by X position
    ZONE = pd.read_csv('mp_points.csv')
    ZONE = ZONE.drop(columns=['Unnamed: 3'])
    ZONE = ZONE.sort_values(by=['X [R]'])

    #Slice and construct XYZ data
    yz_slicer(ZONE,-20, 10, 50, 50)

    #Plot slices with finalized points

    if SHOW_VIDEO:
        show_video('slice_log', 'x_')

    print("___ %s seconds ___" % (time.time() - START))
