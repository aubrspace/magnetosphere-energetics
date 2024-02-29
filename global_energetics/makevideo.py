#/usr/bin/env python
"""Parses swmf output filenames and creates videos out of images
"""
import glob
import os, warnings
import sys
import numpy as np
import datetime as dt
import time as sleeptime

def get_time(infile,**kwargs):
    """Function gets time from file name and returns spacepy Ticktock obj
    Input
        infile
        kwargs:
            timesep(str)- default 'e', could be 't' and 'n'
    Output
        time- spacepy Ticktock object
    """
    try:#looking for typically BATSRUS 3D output
        if '_t' in infile and '_n' in infile:
            date_string=infile.split('/')[-1].split('_t')[-1].split('_')[0]
            if len(date_string)==12:
                time_dt = dt.datetime.strptime(date_string,'%Y%m%d%H%M%S')
            elif len(date_string)==8:
                start_date = kwargs.get('start_date',dt.datetime(1998,1,1,0,0))
                time_dt = start_date+dt.timedelta(hours=24*int(date_string[0:2])+int(date_string[2:4]),
                                                  minutes=int(date_string[4:6]),
                                                  seconds=int(date_string[6:8]))
            time_dt = time_dt.replace(microsecond=0)
        else:
            date_string = infile.split('/')[-1].split('e')[-1].split('.')[0]
            if len(date_string)==19:
                time_dt = dt.datetime.strptime(date_string,'%Y%m%d-%H%M%S-%f')
                time_dt = time_dt.replace(microsecond=0)
            elif len(date_string)==15:
                time_dt = dt.datetime.strptime(date_string,'%Y%m%d-%H%M%S')
    except ValueError:
        try:#looking for typical IE output
            date_string=infile.split('/')[-1].split('it')[-1].split('.')[0]
            time_dt = dt.datetime.strptime(date_string,'%y%m%d_%H%M%S_%f')
            time_dt = time_dt.replace(microsecond=0)
        except ValueError:
            try:#looking for typical UA output
                date_string=infile.split('_t')[-1].split('.')[0]
                time_dt = dt.datetime.strptime(date_string,'%y%m%d_%H%M%S')
                time_dt = time_dt.replace(microsecond=0)
            except ValueError:
                warnings.warn("Tried reading "+infile+
                          " as GM3d or IE output and failed",UserWarning)
                '''
                #Last ditch effort
                #My dumb .h5 output, this should change
                date_string = infile.split('_')[-1].split('.')[0]
                dsplat = [int(s) for s in date_string.split('-')]
                time_dt = dt.datetime(*dsplat)
                time_dt = time_dt.replace(microsecond=0)
                '''
                #Last ditch effort to sort by any numbers in filename
                numbervalue=int(''.join([l for l in infile.split('/')[-1]
                                                        if l.isnumeric()]))
                time_dt = (dt.datetime(1800,1,1)+
                           dt.timedelta(minutes=numbervalue))
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
    for n, image in enumerate(framelist):
        filename = 'img-{:04d}'.format(n)+'.png'
        cp_cmd = 'cp '+image+' '+folder+'/frames/'+filename
        os.popen(cp_cmd)
        print('n: {:d}, filename: {:s}'.format(n,filename))
    return folder+'/frames'


def compile_video(infolder, outfolder, framerate, title):
    """function that compiles video from sequence of .png files
    Inputs:
        infolder, outfolder (str)- path to input/output
        framerate (int)- how many frames per second
        title (str)- name of output video
    """
    #########################################################
    #Notes on ffmpeg command:
    #   vcodec libx264 is h.264 format video
    #   pix_fmt fixes pixel format so that quicktime works
    #########################################################
    if glob.glob(outfolder+'/'+title+'.mp4') != []:
        os.remove(outfolder+'/'+title+'.mp4')
    framelist = glob.glob(infolder+'/*img-????.png')
    print(framelist)
    if framelist!=[]:
        fname=framelist[0].split('/')[-1].split('img-')[0]
        make_vid_cmd =(
        'ffmpeg -r '+str(framerate)+' -i '+infolder+'/'+fname+'img-%04d.png '+
        '-vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p '
        +outfolder+'/'+title+'.mp4')
    os.system(make_vid_cmd)

def add_timestamps(infolder,*, tshift=0):
    """function adds timestamp labels in post in case you forgot (:
    Inputs
        infolder (str)- path to images
    Returns
        copyfolder (str)- path to directory with now stamped images
    """
    from PIL import Image, ImageDraw, ImageFont
    copyfolder = os.path.join(infolder,'copy_wstamps')
    os.makedirs(copyfolder, exist_ok=True)
    for i,infile in enumerate(sorted(glob.glob(infolder+'/*.png'),
        key=time_sort)):
        print(infile.split('/')[-1])
        #Create the stamp
        timestamp = get_time(infile)+dt.timedelta(minutes=tshift)
        if i==0: tstart=timestamp
        simtime = timestamp-tstart
        stamp1 = str(timestamp)
        stamp2 = 'tsim: '+str(simtime)

        #Setup the image
        image = Image.open(infile)
        I1 = ImageDraw.Draw(image)
        font = ImageFont.truetype('fonts/roboto/Roboto-Black.ttf', 45)

        #Attach and save
        location1 = (28,936) # image size depenent
        location2 = (location1[0],location1[1]+50)
        color = (34,255,32) # RGB
        I1.text(location1, stamp1, font=font, fill=color)
        I1.text(location2, stamp2, font=font, fill=color)
        image.save(os.path.join(copyfolder,infile.split('/')[-1]))
    return copyfolder


#Main program
if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print("""
    Parses swmf output filenames and creates videos out of images
    Usage: python [pathtoscript]/makevideo.py [-flags] PATHTOFILES

    Options:
        -h  --help      prints this message then exit
        -f  --framerate sets the framerate of the compiled video
        -s  --stamp     adds timestamp to all frames
        -t  --tshift    adds a timeshift by a number of minutes
                        NOTE: only shifts forward in time and only
                               relevent if --stamp flag also given
        -q  --quick     assumes files are already ordered by frame number
                        and skips the sorting process


    Example:
        python global_energetics/makevideo.py output/png/
            this will create a video of all .png files found at ouput/png/

        python global_energetics/makevideo.py -s -t 45 output/png/
            this will create a video of all .png files found at ouput/png/
              and stamp each frame with the time determined by the filename
              +45minutes

        """)
        exit()
    ###########################################################
    # Read in arguments and flags
    PATHTOIMAGES = sys.argv[-1]
    if not os.path.exists(PATHTOIMAGES):
        print('Path not found please try again')
        exit()
    if 'makevideo.py'in PATHTOIMAGES or len(sys.argv)==1:
        print('No path to images given!! use -h or --help for more info')
        exit()
    # Video settings
    if '-f' in sys.argv or '--framerate' in sys.argv:
        try:
            FRAMERATE = sys.argv[sys.argv.index('-f')+1]
        except ValueError:
            FRAMERATE = sys.argv[sys.argv.index('--framerate')+1]
    else:
        FRAMERATE = 8
    if '-s' in sys.argv or '--stamp' in sys.argv:
        if '-t' in sys.argv or '--tshift' in sys.argv:
            try:
                timeshift = sys.argv[sys.argv.index('-t')+1]
            except ValueError:
                timeshift = sys.argv[sys.argv.index('--tshift')+1]
            PATHTOIMAGES = add_timestamps(PATHTOIMAGES, tshift=timeshift)
        else:
            PATHTOIMAGES = add_timestamps(PATHTOIMAGES)
    # Determine if already in img-??.png form
    if '-q' in sys.argv or '--quick':
        FRAME_LOC = PATHTOIMAGES
    else:
        pass
    FRAME_LOC = set_frames(PATHTOIMAGES)

    #Create video from .png
    compile_video(FRAME_LOC, FRAME_LOC, FRAMERATE, 'video')
