#!/usr/bin/env python3
"""Script for turning 3D magnetopause data into a smooth connected zone
"""

#import os
#import cv2
import sys
import numpy as np
from numpy import pi, sqrt, linspace, arctan2, rad2deg
import time
import matplotlib as mpl
import matplotlib.pyplot as plot
import scipy
from scipy import interpolate as interp
import pandas as pd
from progress.bar import Bar
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

def interpolation(zone_x1val, zone_x2val, *, svar=20, closed=True,
                  maxpt=10000):
    """Function takes two coordinate data and interpolates using interp
    module
    Inputs
        zone_x1val- first (horizontal) coordinate valuesfor interpolation
        zone_x2val- second (vertical) values for interpolation
        svar- s variable for interpolation accuracy
        periodic- boolean for closed or open interpolation
        maxpt- number of points in the interpolated curve
    Returns
        x1_curve- first coordinate interpolated curve
        x2_curve- second coordinate interpolated curve
    """
    x1_points = np.r_[zone_x1val, zone_x1val[0]]
    x2_points = np.r_[zone_x2val, zone_x2val[0]]
    tck, u = interp.splprep([zone_x1val, zone_x2val], s=svar, per=closed)
    x1_curve, x2_curve = interp.splev(np.linspace(0,1,maxpt), tck)
    return x1_curve, x2_curve

def interp_linear(zone_x1val, zone_x2val, x1_solveon):
    """Function makes simple linear interpolation of values
    Inputs
        zone_x1val- first (horizontal) coordinate valuesfor interpolation
        zone_x2val- second (vertical) values for interpolation
        x1_solveon- set of x1 points to return with values
    Returns
        x1_curve- first coordinate interpolated curve
        x2_curve- second coordinate interpolated curve
    """
    x2_curve = []
    for x in x1_solveon:
        x2_curve.append(np.interp(x, zone_x1val, zone_x2val))
    return x1_solveon, x2_curve

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
    for x in linspace(x_min, x_max-dx, n_slice):
        #Gather data within one x-slice
        zone_temp = zone[(zone['X [R]'] < x+dx) & (zone['X [R]'] > x-dx)]

        if show:
            #create plot and dump raw data
            fig, ax = plot.subplots()
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='r',
                       label='raw')

        #Establish local radius, remove r<rmin and restablish
        for remove in [True, False]:
            #find central point
            ymean = np.mean(zone['Y [R]'])
            zmean = np.mean(zone['Z [R]'])
            #add radial column and determine r_min
            radius = pd.DataFrame(np.sqrt((zone_temp['Z [R]']-zmean)**2+(
                                           zone_temp['Y [R]']-ymean)**2),
                                           columns = ['r'])
            zone_temp = zone_temp.combine(radius, np.minimum,
                                          fill_value=1000)
            r_min = 0.2 * zone_temp['r'].max()
            if remove:
                zone_temp = zone_temp[zone_temp['r']>r_min]
                zone_temp = zone_temp.drop(columns=['r'])

        if show:
            #create plot with dropped data
            ax.scatter(zone_temp['Y [R]'], zone_temp['Z [R]'], c='y',
                       label='after 1st cut', marker='o')
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
        y_curve, z_curve = interpolation(zone_temp['Y [R]'].values,
                                         zone_temp['Z [R]'].values)
        translated_zcurve =[point-np.average(z_curve) for point in z_curve]
        translated_ycurve =[point-np.average(y_curve) for point in y_curve]
        dat_angles = np.sort(np.arctan2(translated_zcurve,
                                        translated_ycurve))
        y_load = y_curve[0]
        z_load = z_curve[0]
        #setup angle bins for mesh loading
        spoke, spoketxt = [], []
        n_angle = n_theta
        da = 2*pi/n_angle/5
        stop_stopping=False
        for a in np.linspace(-1*pi, pi, n_angle):
            #extract point from curve in angle range
            condition = ((np.arctan2(translated_zcurve,
                                     translated_ycurve)>a-da) &
                         (np.arctan2(translated_zcurve,
                                     translated_ycurve)<a+da))
            if condition.any():
                x_load = x
                y_load = np.extract(condition, y_curve)[0]
                z_load = np.extract(condition, z_curve)[0]
            else:
                r_previous = sqrt((y_load+ymean)**2+(z_load+zmean)**2)
                y_load = np.interp(r_previous*np.sin(a)+ymean,
                                   y_curve, z_curve, period=2*pi)
                z_load = np.interp(y_load, y_curve, z_curve,
                                   period=2*pi)
                x_load = x
                if show:
                    print("WARNING: No extraction at X={:.2f}, "+
                          "alpha={:.2f}".format(x,a))
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
            ax.scatter(mesh['Y [R]'].values, mesh['Z [R]'].values,
                       label ='mesh_final', marker='o')
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


def ah_slicer(zone, x_min, x_max, nX, n_slice, show):
    """Function loops through YZ anglular bins to create 2D closed curve
        in XZ
    Inputs
        zone- pandas DataFrame object with xyz datat points
        x_min
        x_max
        nX- number of point in the x direction for final mesh
        n_slice
        show- boolean for plotting
    """
    #bin in x direction and initialize newzone
    dx = (x_max-x_min)/(2*(nX-1))
    mesh = pd.DataFrame(columns=['X [R]', 'Y [R]', 'Z [R]', 'alpha'])
    newzone = pd.DataFrame(columns=['X [R]', 'Y [R]', 'Z [R]',
                                    'y_rel', 'z_rel',
                                    'alpha', 'h', 'h_rel',
                                    'y0', 'z0'])
    y0, z0 = [], []
    xlist, dx = linspace(x_min, x_max, nX, retstep=True)
    bar = Bar('getting x-wise data statistics', max=len(xlist))
    for x in xlist:
        #at each x bin calculate y0,z0 based on average
        zone_xbin = zone[(zone['X [R]']<x+dx/2) & (zone['X [R]']>x-dx/2)]
        zone_xbin = zone_xbin.reset_index(drop=True)
        y0.append(np.mean(zone_xbin['Y [R]']))
        z0.append(np.mean(zone_xbin['Z [R]']))
        #append rel_yz, alpha and h for each point in xbin
        y_rel = pd.DataFrame(zone_xbin['Y [R]']-y0[-1]).rename(
                                                columns={'Y [R]':'y_rel'})
        z_rel = pd.DataFrame(zone_xbin['Z [R]']-z0[-1]).rename(
                                                columns={'Z [R]':'z_rel'})
        alpha = pd.DataFrame(arctan2(z_rel.values,y_rel.values),
                             columns=['alpha'])
        h = pd.DataFrame(sqrt(zone_xbin['Z [R]'].values**2+
                              zone_xbin['Y [R]'].values**2),
                         columns=['h'])
        h_rel = pd.DataFrame(sqrt(z_rel.values**2+y_rel.values**2),
                         columns=['h_rel'])
        #combine all newly calculated columns
        zone_xbin = zone_xbin.reindex(columns=['X [R]', 'Y [R]', 'Z [R]',
                                               'y_rel', 'z_rel', 'alpha',
                                               'h','h_rel'],fill_value=999)
        for df in [y_rel,z_rel,alpha,h,h_rel]:
            zone_xbin = zone_xbin.combine(df,np.minimum,fill_value=1000)
        #append y0 and z0 for this xbin
        zone_xbin = zone_xbin.reindex(columns=['X [R]', 'Y [R]', 'Z [R]',
                                               'y_rel', 'z_rel',
                                               'alpha', 'h', 'h_rel',
                                               'y0'],fill_value=y0[-1])
        zone_xbin = zone_xbin.reindex(columns=['X [R]', 'Y [R]', 'Z [R]',
                                               'y_rel', 'z_rel', 'alpha',
                                               'h', 'h_rel', 'y0',
                                               'z0'],fill_value=z0[-1])
        #append xbin set to total dataset
        newzone = newzone.append(zone_xbin, ignore_index=True)
        bar.next()
    bar.finish()

    #for each alpha slice 
    k = 0
    alist, da = linspace(-pi, pi, n_slice, retstep=True)
    bar = Bar('identifying cylindrical mesh points', max=len(alist))
    for a in alist:
        zone_abin = newzone[(newzone['alpha'] < a+da/2) &
                            (newzone['alpha'] > a-da/2)]
        if show:
            #create plot and dump raw data
            fig, ax = plot.subplots()
            ax.scatter(zone_abin['X [R]'], (zone_abin['y0'].values**2+
                                            zone_abin['z0'].values**2),
                       label='centerline')
            ax.scatter(zone_abin['X [R]'], zone_abin['h'], c='b',
                       label='h_rel')

        #for each xbin, remove all but the max h_relative value
        xfinelist, dxfine = linspace(x_min, x_max, 2*nX, retstep=True)
        for x in xfinelist:
            #if the section has values
            if not zone_abin[(zone_abin['X [R]'] < x+dxfine/2) &
                             (zone_abin['X [R]'] > x-dxfine/2)].empty:
                #identify the index of row with maximum r within section
                max_h_rel_index = zone_abin[
                            (zone_abin['X [R]'] < x+dxfine/2) &
                            (zone_abin['X [R]'] > x-dxfine/2)].idxmax(0)[
                                                                   'h_rel']
                #cut out alpha slice except for max_r_index row
                zone_abin = zone_abin[
                                      (zone_abin['X [R]'] > x+dxfine/2) |
                                      (zone_abin['X [R]'] < x-dxfine/2) |
                                      (zone_abin.index == max_h_rel_index)]
            elif show:
                print("No points at a:{}, x:{}".format(rad2deg(a),x))

        #Load points into mesh
        j=0
        x_curve = linspace(x_min, x_max, nX)
        h_curve = []
        for x in x_curve:
            h_load = np.interp(x, zone_abin['X [R]'].values,
                                  zone_abin['h_rel'].values)
            h_curve.append(h_load)
            x_load = x
            y_load = y0[j] + h_load*np.cos(a)
            z_load = z0[j] + h_load*np.sin(a)
            mesh = mesh.append(pd.DataFrame([[x_load, y_load, z_load, a]],
                          columns = ['X [R]', 'Y [R]', 'Z [R]', 'alpha']),
                          ignore_index=True)
            j+=1
        '''
        #Created open interpolation of data using scipy spline interp
        x_curve, h_curve = interpolation(zone_abin['X [R]'].values,
                                         zone_abin['h_rel'].values,
                                         closed=False)
        #Select points from interpolated curve using basic linear np.interp
        j=0
        for x in linspace(x_min, x_max-dx, nX):
            h_load = np.interp(x, x_curve, h_curve)
            x_load = x
            y_load = y0[j] + h_load*np.cos(a)
            z_load = z0[j] + h_load*np.sin(a)
            mesh = mesh.append(pd.DataFrame([[x_load, y_load, z_load, a]],
                          columns = ['X [R]', 'Y [R]', 'Z [R]', 'alpha']),
                          ignore_index=True)
            j+=1
        '''

        #show interpolated curve
        if show:
            ax.scatter(x_curve, h_curve, c='orange',
                       label='interpolated')

        #show portion to plot points used for interpolation
        if show:
            ax.scatter(zone_abin['X [R]'], zone_abin['h'], c='r',
                       label='remaining')

        if show:
            #plot interpolated data and save figure
            ax.set_xlabel('X [Re]')
            ax.set_ylabel('h [Re]')
            ax.set_xlim([-30,15])
            ax.set_ylim([0,40])
            plot.gca().invert_xaxis()
            ax.set_title('alpha= {:.2f} +/- {:.2f}'.format(rad2deg(a),
                                                           rad2deg(da))+
                          '\nn= {:.2f}'.format(len(zone_abin)))
            ax.legend(loc='upper left')
            if k<0.9:
                filename = 'slice_log2/img-0{:.0f}.png'.format(10*k)
            else:
                filename = 'slice_log2/img-{:.0f}.png'.format(10*k)
            plot.savefig(filename)
            k = k+.1
        bar.next()
    bar.finish()
    #Sort mesh by X then alpha, then drop alpha column
    mesh = mesh.sort_values(by=['X [R]', 'alpha']).drop(columns='alpha')
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
    ZONE = pd.read_csv('cps_stream_points.csv')
    ZONE = ZONE.drop(columns=['Unnamed: 3'])
    ZONE = ZONE.sort_values(by=['X [R]'])
    ZONE = ZONE.reset_index(drop=True )
    X_MAX = ZONE['X [R]'].max()
    #X_MIN = ZONE['X [R]'].min()
    X_MIN = ZONE['X [R]'].min()

    #Slice and construct XYZ data
    MESH = ah_slicer(ZONE, X_MIN, -3, 20, 36, SHOW_VIDEO)
    #MESH.to_hdf('slice_mesh.h5', format='table', key='MESH', mode='w')
    #MESH.to_csv('slice_mesh.csv', index=False)

    #Plot slices with finalized points
    #if SHOW_VIDEO:
        #show_video('slice_log')

    print("___ %s seconds ___" % (time.time() - START))
