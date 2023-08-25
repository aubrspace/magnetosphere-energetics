import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
import numpy as np
#### import the simple module from paraview
from paraview.simple import *

def tec2para(instr):
    badchars = ['{','}','[',']']
    replacements = {' [':'_','**':'^','1e':'10^','pi':'3.14159',#generic
          '/Re':'_Re', 'amu/cm':'amu_cm','km/s':'km_s','/m':'_m',#specific
                    'm^':'m','e^':'e'}#very specific, for units only
            #'/':'_',
    coords = {'X_R':'x','Y_R':'y','Z_R':'z'}
    outstr = instr
    for was_,is_ in replacements.items():
        outstr = is_.join(outstr.split(was_))
    for was_,is_ in coords.items():
        outstr = is_.join(outstr.split(was_))
        #NOTE this order to find "{var [unit]}" cases (space before [unit])
        #TODO see .replace
    for bc in badchars:
        outstr = ''.join(outstr.split(bc))
    #print('WAS: ',instr,'\tIS: ',outstr)
    return outstr

def eqeval(eqset,pipeline,**kwargs):
    for lhs_tec,rhs_tec in eqset.items():
        lhs = tec2para(lhs_tec)
        rhs = tec2para(rhs_tec)
        var = Calculator(registrationName=lhs, Input=pipeline)
        var.Function = rhs
        var.ResultArrayName = lhs
        pipeline = var
    return pipeline

def get_sphere_filter(pipeline,**kwargs):
    """Function calculates a sphere variable, NOTE:will still need to
        process variable into iso surface then cleanup iso surface!
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            betastar_max (float)- default 0.7
            status_closed (float)- default 3
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    #Must have the following conditions met first
    assert FindSource('r_R') != None
    radius = kwargs.get('radius',3)
    r_state =ProgrammableFilter(registrationName='r_state',Input=pipeline)
    r_state.Script = """
    # Get input
    data = inputs[0]
    r = data.PointData['r_R']

    #Compute sphere as logical for discrete points
    r_state = (abs(r-"""+str(radius)+""")<0.2).astype(int)

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(r_state,'r_state')
    """
    pipeline = r_state
    return pipeline
