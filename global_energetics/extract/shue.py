#!/usr/bin/env python3
"""Functions to represent 2D curves according to Shue et al 1997 and 1998
"""
import numpy as np
from numpy import cos, deg2rad, tanh, log

def r_shue(r0, alpha, theta):
    return r0*(2/(1+cos(deg2rad(theta))))**alpha

def r0_alpha_1997(Bz, Pdyn):
    """Function returns r0 and alpha parameters based on 1997 model
    Inputs
        Bz- IMF Bz component in nT
        Pdyn- IMF dynamic pressure in nPa
    Outputs
        r0, alpha- parameters for emperical model functional form
    """
    if Bz >= 0:
        r0 = (11.4 + 0.013*Bz)*(Pdyn)**(-1/6.6)
    else:
        r0 = (11.4 + 0.14*Bz)*(Pdyn)**(-1/6.6)
    alpha = (0.58 - 0.010*Bz)*(1 + 0.010*Pdyn)
    return r0, alpha

def r0_alpha_1998(Bz, Pdyn):
    """Function returns r0 and alpha parameters based on 1998 model
    Inputs
        Bz- IMF Bz component in nT
        Pdyn- IMF dynamic pressure in nPa
    Outputs
        r0, alpha- parameters for emperical model functional form
    """
    r0 = (10.22 + 1.29*tanh(0.184*(Bz+8.14)))*(Pdyn)**(-1/6.6)
    alpha = (0.58 - 0.007*Bz)*(1 + 0.024*log(Pdyn))
    return r0, alpha
