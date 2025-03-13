#!/usr/bin/env python3
"""module processes observation data from AMPERE
"""
import os
import sys
import glob
import time
import numpy as np
from numpy import abs, pi, cos, sin, sqrt, rad2deg, matmul, deg2rad
import datetime as dt
import pandas as pd
from global_energetics.analysis.proc_indices import csv_to_pandas

def read_currents(infile: str) -> pd.DataFrame:
    with open(infile,'r') as f:
        header = f.readline()
        header_clean = header.replace('\n','').replace(
                                                    ", ",",").replace(' ','_')
        header = header_clean.split(',')
    df = csv_to_pandas(infile,
                    header=None,names=header,
                    tdict={'year':'year','month':'month','day':'day',
                           'hour':'hour','minute':'minute','second':'second'})
    return df
