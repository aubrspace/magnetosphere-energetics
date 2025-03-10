"""Tools that **DONT REQUIRE TECPLOT or PARAVIEW** that each can leverage
"""

from numpy import sign
from numba import jit
import numpy as np
from numpy import deg2rad,sin,cos,arcsin,sqrt

@jit#NOTE this restricts the types and methods available so take care
def check_bin(x,theta_1,phi_1,inbin,state):
    """Function checks 4 quadrants of bin to determine contestation (daynight)
    Inputs
        x (arr[float])
        theta_1 (arr[float])
        phi_1 (arr[float])
        inbin (arr[float])
        state (arr[float])
    Return
        qs (list[list[bools]]) - quadrants of the original bin
        contested (bool) - if the quads agree about daynight
    """
    thHigh = theta_1[inbin].max()
    thLow = theta_1[inbin].min()
    phHigh = phi_1[inbin].max()
    phLow = phi_1[inbin].min()
    thMid = (thHigh+thLow)/2
    phMid = (phHigh+phLow)/2
    q1 = ((state==1)&
          (theta_1<thHigh)&(theta_1>thMid)&
          (phi_1<phMid)&(phi_1>phLow))
    q2 = ((state==1)&
          (theta_1<thMid)&(theta_1>thLow)&
          (phi_1<phMid)&(phi_1>phLow))
    q3 = ((state==1)&
          (theta_1<thMid)&(theta_1>thLow)&
          (phi_1<phHigh)&(phi_1>phMid))
    q4 = ((state==1)&
          (theta_1<thHigh)&(theta_1>thMid)&
          (phi_1<phHigh)&(phi_1>phMid))
    quadbins = []
    for q in [q1,q2,q3,q4]:
        if np.any(q):
            quadbins.append(q)
    signs = np.array([sign(x[q].mean()) for q in quadbins])
    contested = np.zeros(1,dtype=np.bool_)
    if signs.size == 0:
        contested[0] = 0
    elif (signs.all()) or (signs.max()==0):
        contested[0] = 0
    else:
        contested[0] = 1
    return quadbins, contested[0]
