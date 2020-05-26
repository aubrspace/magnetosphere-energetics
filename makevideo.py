#/usr/bin/env python
"""Converts img-00.pdf files to img-00.png and then makes a video
"""

import glob
import os
import sys
import numpy as np

def get_time(filename):
    """Function gets time in seconds from file name assuming naming
        convention: YEARMODAY-HRMINSECdone.png
    Input
        filename
    Output
        abstime
    """
    date = filename.split('/')[-1].split('-')[0]
    year = int(''.join(list(date)[-8:-4]))
    month = int(''.join(list(date)[-4:-2]))
    day = int(''.join(list(date)[-2:]))
    time = int(filename.split('/')[-1].split('-')[1].split('d')[0])
    hour = np.floor(time/10000)
    minute = np.mod(time/100, 100)
    second = np.mod(time, 100)
    abstime = second + 60*(minute+ 60*(hour +
                           24*(day + 30*(month+12*year))))
    return abstime

def convert_pdf(folder, res):
    """Function converts .pdf files to .png files for compiling into video.
       IMPORTANT- have .pdf files in a XXX-AB.pdf format where AB
       starts at 00 and denotes the order of the videos
    Inputs
        folder
        res- horizontal resolution for .png file
    """
    for image in glob.glob(folder+'/*.pdf'):
        frame = image.split('-')[1].split('d')[0]
        filename = './img-'+frame+'.png'
        convert_cmd = 'convert -density '+str(res)+' '+ image +' '+ filename
        os.system(convert_cmd)
        print(filename, 'has been converted')

def set_frames(folder):
    """function preps files with date time format see get_time
    Input
        folder
    Output
        framedir
    """
    #create sorted list of image files
    framelist = sorted(glob.glob(folder+'/*.png'), key=get_time)
    os.system('mkdir '+folder+'/frames/')
    n=0
    for image in framelist:
        if n<0.099:
            filename = 'img-00{:.0f}.png'.format(100*n)
        elif n<0.999:
            filename = 'img-0{:.0f}.png'.format(100*n)
        else:
            filename = 'img-{:.0f}.png'.format(100*n)
        cp_cmd = 'cp '+image+' '+folder+'/frames/'+filename
        os.system(cp_cmd)
        print('n: {:.2f}, filename: {:s}'.format(n,filename))
        n = n+0.01
    return folder+'/frames'


def vid_compile(folder, framerate, title):
    """function that compiles video from sequence of .png files
    Inputs:
        folder
        framerate
        title
    """
    os.system('rm '+folder+'/'+title+'.avi')
    make_vid_cmd = 'ffmpeg -framerate '+str(framerate)+' -i '+folder+'/img-%03d.png '+folder+'/'+title+'.avi'
    os.system(make_vid_cmd)
    print('\nopening video: '+folder+'/'+title+'.avi\n')
    os.system('open '+folder+'/'+title+'.avi')

#Main program
if __name__ == '__main__':
    #Video settings
    RES = 400
    FRAMERATE = 2
    FOLDER = sys.argv[1]
    #determine file frame order
    FRAME_LOC = set_frames(FOLDER)
    #Convert all img .pdf to img.png
    #convert_pdf(FOLDER, RES)
    #Create video from .png
    vid_compile(FRAME_LOC, FRAMERATE, 'video')
