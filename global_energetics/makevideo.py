#/usr/bin/env python
"""Converts img-00.pdf files to img-00.png and then makes a video
"""

import glob
import os, warnings
import sys
import time as sleeptime
import numpy as np
import datetime as dt
#import spacepy.time as spacetime

def get_time(infile):
    """Function gets time from file name and returns spacepy Ticktock obj
    Input
        infile
    Output
        time- spacepy Ticktock object
    """
    try:#looking for typically BATSRUS 3D output
        date_string = infile.split('/')[-1].split('e')[-1].split('.')[0]
        time_dt = dt.datetime.strptime(date_string,'%Y%m%d-%H%M%S-%f')
    except ValueError:
        try:#looking for typical IE output
            date_string=infile.split('/')[-1].split('it')[-1].split('.')[0]
            time_dt = dt.datetime.strptime(date_string,'%y%m%d_%H%M%S_%f')
        except ValueError:
            try:#looking for typical UA output
                date_string=infile.split('_t')[-1].split('.')[0]
                time_dt = dt.datetime.strptime(date_string,'%y%m%d_%H%M%S')
            except ValueError:
                warnings.warn("Tried reading "+infile+
                          " as GM3d or IE output and failed",UserWarning)
                time_dt = None
    finally:
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

def add_timestamps(infolder):
    """function adds timestamp labels in post in case you forgot (:
    Inputs
        infolder
    Returns
        copyfolder
    """
    from PIL import Image, ImageDraw, ImageFont
    copyfolder = os.path.join(infolder,'copy_wstamps')
    os.makedirs(copyfolder, exist_ok=True)
    for i,infile in enumerate(sorted(glob.glob(infolder+'/*.png'),
        key=time_sort)):
        print(infile.split('/')[-1])
        #Create the stamp
        timestamp = get_time(infile)+dt.timedelta(minutes=45)
        if i==0: tstart=timestamp
        simtime = timestamp-tstart
        stamp1 = str(timestamp)
        stamp2 = 'tsim: '+str(simtime)

        #Setup the image
        image = Image.open(infile)
        I1 = ImageDraw.Draw(image)
        goodfont = ImageFont.truetype('FreeMono.ttf', 45)

        #Attach and save
        #I1.text((28,36), stamp1, font=goodfont, fill=(34,255,32))#TopLeftLimeGreen
        #I1.text((28,97), stamp2, font=goodfont, fill=(34,255,32))#TopLeftLimeGreen
        I1.text((828,1236), stamp1, font=goodfont, fill=(124,246,223))#BotRightCyan
        I1.text((828,1297), stamp2, font=goodfont, fill=(124,246,223))#BotRightCyan
        image.save(os.path.join(copyfolder,infile.split('/')[-1]))
    return copyfolder


#Main program
if __name__ == '__main__':
    #Video settings
    RES = 400
    FRAMERATE = 36
    FOLDER = sys.argv[1]
    if '-stamp' in sys.argv:
        FOLDER = add_timestamps(FOLDER)

    #determine if already in img-??.png form
    if '-q' in sys.argv:
        FRAME_LOC = FOLDER
    else:
        FRAME_LOC = set_frames(FOLDER)
    #convert_pdf(FOLDER, RES)

    #Create video from .png
    vid_compile(FRAME_LOC, FRAME_LOC, FRAMERATE, 'video')
