#/usr/bin/env python
"""Converts img-00.pdf files to img-00.png and then makes a video
"""

import glob
import os
import sys
import time as sleeptime
import numpy as np
import datetime as dt
#import spacepy.time as spacetime

def get_time(filename):
    """Function gets time from file name and returns spacepy Ticktock obj
    Input
        filename
    Output
        time- spacepy Ticktock object
    """
    date_string = filename.split('/')[-1].split('e')[-1].split('.')[0]
    time_dt = dt.datetime.strptime(date_string,'%Y%m%d-%H%M%S-%f')
    '''
    #import spacepy.time as spacetime
    date_string = filename.split('/')[-1].split('-')[0]
    year = int(''.join(list(date_string)[-8:-4]))
    month = int(''.join(list(date_string)[-4:-2]))
    day = int(''.join(list(date_string)[-2:]))
    time_string = int(filename.split('/')[-1].split('-')[1].split('-')[0])
    hour = int(np.floor(time_string/10000))
    minute = int(np.mod(time_string/100, 100))
    second = int(np.mod(time_string, 100))
    time_dt = dt.datetime(year, month, day, hour, minute, second)
    #time = spacetime.Ticktock(time_dt,'UTC')
    '''
    return time_dt

def time_sort(filename):
    """Function returns absolute time in seconds for use in sorting
    Inputs
        filename
    Outputs
        total_seconds
    """
    time = get_time(filename)
    relative_time = time-dt.datetime(1800, 1, 1)
    return (relative_time.days*86400 + relative_time.seconds)

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
    framelist = sorted(glob.glob(folder+'/*.png'), key=time_sort)
    os.makedirs(folder+'/frames/', exist_ok=True)
    n=0.01
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


def vid_compile(infolder, outfolder, framerate, title):
    """function that compiles video from sequence of .png files
    Inputs:
        folder
        framerate
        title
    """
    #########################################################
    #Notes on ffmpeg command:
    #   vcodec libx264 is h.264 format video
    #   pix_fmt fixes pixel format so that quicktime works
    #########################################################
    if glob.glob(outfolder+'/'+title+'.mp4') != []:
        os.remove(outfolder+'/'+title+'.mp4')
    print(glob.glob(infolder+'/img-???.png'))
    if glob.glob(infolder+'/img-???.png') != []:
        make_vid_cmd = 'ffmpeg -r '+str(framerate)+' -i '+infolder+'/img-%03d.png -vcodec libx264 -pix_fmt yuv420p '+outfolder+'/'+title+'.mp4'
    elif glob.glob(folder+'/img-??.png') != []:
        make_vid_cmd = 'ffmpeg -r '+str(framerate)+' -i '+infolder+'/img-%02d.png -vcodec libx264 -pix_fmt yuv420p '+outfolder+'/'+title+'.mp4'
    elif glob.glob(folder+'/img-?.png') != []:
        make_vid_cmd = 'ffmpeg -r '+str(framerate)+' -i '+infolder+'/img-%01d.png -vcodec libx264 -pix_fmt yuv420p '+outfolder+'/'+title+'.mp4'
    os.system(make_vid_cmd)

#Main program
if __name__ == '__main__':
    #Video settings
    RES = 400
    FRAMERATE = 12
    FOLDER = sys.argv[1]

    #determine if already in img-??.png form
    if '-q' in sys.argv:
        FRAME_LOC = FOLDER
    else:
        FRAME_LOC = set_frames(FOLDER)
    #convert_pdf(FOLDER, RES)

    #Create video from .png
    vid_compile(FRAME_LOC, FRAME_LOC, FRAMERATE, 'video')
