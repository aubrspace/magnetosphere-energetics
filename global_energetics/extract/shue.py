#!/usr/bin/env python3
"""Functions to represent 2D curves according to Shue et al 1997 and 1998
"""
import numpy as np
from numpy import cos, deg2rad, tanh, log, sqrt

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

def r0_bow_shock_Jerab2005(N,V,Ma,B,gamma=5/3):
    """Function returns bow shock nose r0
    Inputs
    Returns
        r0
    """
    #First find bow shock nose
    C = 91.55
    D = 0.937*(0.846+0.042*B)
    R0 = C/(N*V**2)**(1/6) * (1 + D*((gamma-1)*Ma**2+2)/
                                    ((gamma+1)*Ma**2-1))
    return R0

def r_bow_shock_Jerab2005(R0,X,Y,Z):
    #Next find Rave
    a11 = 0.45#X**2
    a22 = 1#Y**2
    a33 = 0.8#Z**2
    a12 = 0.18#XY
    a14 = 46.6#X
    a24 = -2.2#Y
    a34 = -0.6#Z
    a44 = -618#const
    #In this case X==R so we only need the above for the fits
    # so we're solving a11*X**2 + a14*X + a44 = 0 for X
    # using the quadratic formula:
    #                        
    #                   sqrt(a14**2 - 4*a11*a44)
    #       X = -a14 +- -----------------------
    #                             2*a11   
    Rave = -a14+sqrt(a14**2 - 4*a11*a44)/(2*a11)

