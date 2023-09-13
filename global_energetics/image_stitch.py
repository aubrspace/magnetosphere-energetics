#/usr/bin/env python
import glob
import numpy as np
from PIL import Image

def append_right(base_image,added_image,*,buf_factor=1.01):
    """Pastes a figure to the right of another figure
    Inputs
        base_image
        added_image
    Returns
        result (Image)
    """
    # Get the two image dimensions
    base_width, base_height = base_image.size
    added_width, added_height = added_image.size

    # Set the height based on the tallest
    result_height = max(base_height,added_height)

    # Find the total width and buffer amount
    result_width = int((base_width+added_width)*buf_factor)
    buffer_width = int((base_width+added_width)*(buf_factor-1))

    # Create new image
    result = Image.new('RGB', (result_width,result_height))
    result.paste(im=base_image, box=(0,0))
    result.paste(im=added_image, box=(base_width+buffer_width,0))

    return result

if __name__ == '__main__':
    ## Orbits
    clusterfile = '/home/aubr/Desktop/sat_orbit_summary/cluster.png'
    mmsfile = '/home/aubr/Desktop/sat_orbit_summary/mms.png'
    themisfile = '/home/aubr/Desktop/sat_orbit_summary/themis.png'
    '''
    cluster = Image.open(clusterfile)
    mms = Image.open(mmsfile)
    themis = Image.open(themisfile)
    twoimage = append_right(cluster,mms)
    threeimage = append_right(twoimage,themis)
    threeimage.save('/home/aubr/Code/swmf-energetics/jgr2023/figures/'+
                     'unfiled/sat_collage.png')
    '''
    ## FFJ
    ffj0file = '/home/aubr/Desktop/FFJ_snaps/t0.png'
    ffj1file = '/home/aubr/Desktop/FFJ_snaps/t1.png'
    ffj2file = '/home/aubr/Desktop/FFJ_snaps/t2.png'
    ffj3file = '/home/aubr/Desktop/FFJ_snaps/t3.png'
    ffj0 = Image.open(ffj0file)
    ffj1 = Image.open(ffj1file)
    ffj2 = Image.open(ffj2file)
    ffj3 = Image.open(ffj3file)
    twoimage = append_right(ffj0,ffj1)
    threeimage = append_right(twoimage,ffj2)
    fourimage = append_right(threeimage,ffj3)
    fourimage.save('/home/aubr/Code/swmf-energetics/jgr2023/figures/'+
                     'unfiled/ffj_collage.png')
    # Set the path to files and make a file list
    # Based on the number of images in the group find the total size
    # Create an empty background
    # Set in the images in the proper locations
    # Print the timestamp for each image above it
    # Save the image
