#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
import sys
import os
import time
import glob
import numpy as np
import datetime as dt
import pandas as pd
import tecplot as tp

def get_innermag_zone(deltadt, datapath, *, module='RCM2'):
    """Function to find the correct IM datafile and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of the event
        datapath- str path to ionosphere data
        module- what module is being read, default (and only option): RCM2
    """
    if module == 'RCM2':
        deltadtstring = str(deltadt)
        hr = deltadtstring.split(':')[0]
        if int(hr)<10:
            hr = '0'+hr
        minute = deltadtstring.split(':')[1]
        sec = deltadtstring.split(':')[2].split('.')[0]
        datafile = '2d__max_t00'+hr+minute+sec+'.plt'
        if os.path.exists(datapath+datafile):
            im_data = tp.data.load_tecplot(datapath+datafile)
            im_zone = im_data.zone('RCM*')
            return im_zone
        else:
            print('no IM data found!')
            return None
    else:
        print('only RCM2 files accepted at this time!')
        return None

if __name__ == "__main__":
    pass
