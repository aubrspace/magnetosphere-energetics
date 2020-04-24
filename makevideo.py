#/usr/bin/env python
"""Converts img-00.pdf files to img-00.png and then makes a video
"""

import glob
import os

def convert_pdf(folder, res):
    """Function converts .pdf files to .png files for compiling into video.
       IMPORTANT- have .pdf files in a XXX-AB.pdf format where AB
       starts at 00 and denotes the order of the videos
    Inputs
        folder
        res- horizontal resolution for .png file
    """
    for image in glob.glob(folder+'/*.pdf'):
        frame = image.split('-')[1].split('.')[0]
        filename = './img-'+frame+'.png'
        convert_cmd = 'convert -density '+str(res)+' '+ image +' '+ filename
        os.system(convert_cmd)
        print(filename, 'has been converted')

def vid_compile(folder, framerate, title):
    """function that compiles video from sequence of .png files
    Inputs:
        folder
        framerate
        title
    """
    os.system('rm '+folder+'/'+title+'.avi')
    make_vid_cmd = 'ffmpeg -framerate '+str(framerate)+' -i '+folder+'/img-%02d.png '+folder+'/'+title+'.avi'
    os.system(make_vid_cmd)
    print('\nopening video: '+folder+'/'+title+'.avi\n')
    os.system('open '+folder+'/'+title+'.avi')

#Main program
if __name__ == '__main__':
    #Video settings
    RES = 400
    FRAMERATE = 2
    FOLDER = 'slice_log'
    #Convert all img .pdf to img.png
    convert_pdf(FOLDER, RES)
    #Create video from .png
    vid_compile(FOLDER, FRAMERATE, 'video')
