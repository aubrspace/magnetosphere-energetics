# An eclectic  mix of functions ranging from critical to mildly useful
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
import numpy as np
from numpy import sin,cos,pi
#### import the simple module from paraview
from paraview.simple import *
### interpackage
#from global_energetics.extract.shared_tools import check_bin

def bfield_project(Input,r1,r2,**kwargs):
    # Calculate Latitude
    colat = Calculator(registrationName='colat',Input=Input)
    colat.Function = "acos(coordsZ/sqrt(mag(coords)))"
    colat.ResultArrayName = 'colat'
    # Expose XYZ coords using Calculator filter so PythonFilter can see
    xyz = Calculator(registrationName='xyz',Input=colat)
    xyz.Function = "coordsX*iHat+coordsY*jHat+coordsZ*kHat"
    xyz.ResultArrayName = 'XYZ'
    # Calculate Longitude
    lon = PythonCalculator(registrationName='lon',Input=xyz)
    lon.Expression = ("np.arctan2(inputs[0].PointData['XYZ'][:,1],"+
                                 "inputs[0].PointData['XYZ'][:,0])")
    lon.ArrayName = 'lon'
    # Calculate Projected latitude along dipole field to new radius
    colat2 = Calculator(registrationName='colat2',Input=lon)
    colat2.Function = f"asin(sqrt({r2}/{r1}*sin(colat)^2))"
    colat2.ResultArrayName = 'colat2'
    # Stretch sphere into position at r2
    stretch = Calculator(registrationName='stretch',Input=colat2)
    stretch.Function = (f"{r2}*sin(colat2)*cos(lon)*iHat+"+
                        f"{r2}*sin(colat2)*sin(lon)*jHat+"+
                        f"sign(coordsZ)*{r2}*cos(colat2)*kHat")
    stretch.ResultArrayName = 'stretch'
    stretch.CoordinateResults = 1
    return stretch

def reversed_mapping(pipeline,statekey,**kwargs):
    rmap = get_reverse_map_filter(pipeline)
    return rmap

def get_reverse_map_filter(pipeline,**kwargs):
    #TODO, put this in pv speak 
    #   assert FindSource('theta_1_deg') != None
    #   assert FindSource('phi_1_deg') != None
    #   assert FindSource('dvol_R^3') != None
    rmap =ProgrammableFilter(registrationName='reverse_map',Input=pipeline)
    rmap.Script = """
    # Get input
    data = inputs[0]
    theta_1 = data.PointData['theta_1_deg']
    phi_1 = data.PointData['phi_1_deg']
    volume = data.PointData['dvol_R^3']
    x = data.PointData['x']*volume #NOTE volume weighted
    # Make a new set of variables
    if 'daynight' in data.PointData.keys():
        daynight = data.PointData['daynight']
    else:
        daynight = np.zeros(len(x))
        # Create an initial set of coarse bins
        theta_bins = np.linspace(0,90,10)
        phi_bins = np.linspace(0,360,37)
        k=0
        # Iterate through each bin
        for i,thHigh in enumerate(theta_bins[1::]):
            for j,phHigh in enumerate(phi_bins[1::]):
                thLow = theta_bins[i-1]
                phLow = phi_bins[j-1]
                inbins = ((state==1)&
                        (theta_1<thHigh)&
                        (theta_1>thLow)&
                        (phi_1<phHigh)&
                        (phi_1>phLow))
                if any(inbins):
                    # Subdivide bin until 4 subquadrants agree
                    finished_bins = []
                    contested_bins = [inbins]
                    i=0
                    while len(contested_bins)>0:
                        i+=1
                        old_contested_bins = contested_bins
                        contested_bins = []
                        for b in old_contested_bins:
                            qs, contested = check_bin(x,theta_1,phi_1,b,state)
                            if not contested:
                                finished_bins.append(b)
                            else:
                                for q in qs:
                                    contested_bins.append(q)
                        if i>1 and kwargs.get('verbose',False):
                            print(i)
                        if i>5:
                            for q in qs:
                                finished_bins.append(q)
                            contested_bins = []
                    # Now actually set the values using the finished_bin list
                    for inbin in finished_bins:
                        dayside,nightside,split = 0,0,False
                        k+=1
                        #if kwargs.get('debug',False):
                        #    mapID[inbins] = k
                        if x[inbin].mean()>0:
                            dayside = 1
                        if x[inbin].mean()<0:
                            nightside = 1
                        if dayside*nightside>0:
                            split = True
                        if not split:
                            if dayside:
                                daynight[inbin] = 1
                            elif nightside:
                                daynight[inbin] = -1
                        else:
                            daynight[inbins] = -999
                        if kwargs.get('verbose',False):
                            print(k,thLow,thHigh,
                            phLow,phHigh,
                            x[inbins].min(),x[inbins].max(),
                            '\tday:',dayside,'\tnight:',nightside)
        # Set the values in Tecplot from our numpy array
        #TODO: finish the last part here!!!
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(daynight,'daynight')
    """
    return rmap
