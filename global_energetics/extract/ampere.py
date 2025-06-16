#!/usr/bin/env python3
"""Extracting ampere data using their API
    https://ampere.jhuapl.edu/download/?page=webServiceTab
"""
import os,sys
import glob
import numpy as np
import datetime as dt
import requests

if __name__ == "__main__":
    curlstr = "https://ampere.jhuapl.edu/services/data-grd.php?logon=abrenner&start=2012-01-03T12:00&end=2012-01-03T14:00&pole=north"
    params = {'logon':'abrenner',
              'start':'2024-05-10T10:00',
              'end':'2024-05-12T23:59',
              'pole':'north'}
    r = requests.get('https://ampere.jhuapl.edu/services/data-grd.php',
                     params=params)
    #TODO figure out how to read from web directly to cdf file
    #   Check how swmfpy does it
    from IPython import embed; embed()
