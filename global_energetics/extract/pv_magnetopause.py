# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

import os
import time
import glob
import numpy as np
#### import the simple module from paraview
from paraview.simple import *

def equations(**kwargs):
    """Defines equations that will be used for global variables
    Inputs- none
    Return
        equations dict{dict{str(eqName):str(eqText)}}- nested dicts
    """
    equations = {}
    #Testing function for verifying matching interfaces
    equations['interface_testing'] = {'{test}':'1'}
    #Useful spatial variables
    equations['basic3d'] = {
                       '{r [R]}':'sqrt({X [R]}**2+{Y [R]}**2+{Z [R]}**2)',
                       '{Cell Size [Re]}':'{Cell Volume}**(1/3)',
                       '{h}':'sqrt({Y [R]}**2+{Z [R]}**2)'}
    #2D versions of spatial variables
    equations['basic2d_XY'] = {'{r [R]}':'sqrt({X [R]}**2 + {Y [R]}**2)'}
    equations['basic2d_XZ'] = {'{r [R]}':'sqrt({X [R]}**2 + {Z [R]}**2)'}
    #Dipolar coordinate variables
    if 'aux' in kwargs:
        aux=kwargs.get('aux')
        equations['dipole_coord'] = {
         '{mXhat_x}':'sin(('+aux['BTHETATILT']+'+90)*pi/180)',
         '{mXhat_y}':'0',
         '{mXhat_z}':'-1*cos(('+aux['BTHETATILT']+'+90)*pi/180)',
         '{mZhat_x}':'sin('+aux['BTHETATILT']+'*pi/180)',
         '{mZhat_y}':'0',
         '{mZhat_z}':'-1*cos('+aux['BTHETATILT']+'*pi/180)',
         '{lambda}':'asin('+
                        '(({mZhat_x}*{X [R]}+{mZhat_z}*{Z [R]})/{r [R]})-'+
                   'trunc(({mZhat_x}*{X [R]}+{mZhat_z}*{Z [R]})/{r [R]})'+
                        ')',
         '{Lshell}':'{r [R]}/cos({lambda})**2',
         '{theta [deg]}':'-180/pi*{lambda}'}
    ######################################################################
    #Physical quantities including:
    #   Dynamic Pressure
    #   Sonic speed
    #   Plasma Beta
    #   Plasma Beta* using total pressure
    #   B magnitude
    equations['basic_physics'] = {
     '{Dp [nPa]}':'{Rho [amu/cm^3]}*1e6*1.6605e-27*'+
              '({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*1e6*1e9',
     '{Cs [km/s]}':'sqrt(5/3*{P [nPa]}/{Rho [amu/cm^3]}/6.022)*10**3',
     '{beta}':'({P [nPa]})/({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                '*(2*4*pi*1e-7)*1e9',
     '{beta_star}':'({P [nPa]}+{Dp [nPa]})/'+
                          '({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'+
                '*(2*4*pi*1e-7)*1e9',
     '{Bmag [nT]}':'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)'}
    ######################################################################
    #Fieldlinemaping
    equations['fieldmapping'] = {
        '{Xd [R]}':'{mXhat_x}*({X [R]}*{mXhat_x}+{Z [R]}*{mXhat_z})',
        '{Zd [R]}':'{mZhat_z}*({X [R]}*{mZhat_x}+{Z [R]}*{mZhat_z})',
        '{phi}':'atan2({Y [R]}, {Xd [R]})',
        '{req}':'2.7/(cos({lambda})**2)',
        '{lambda2}':'sqrt(acos(1/{req}))',
        '{X_r1project}':'1*cos({phi})*sin(pi/2-{lambda2})',
        '{Y_r1project}':'1*sin({phi})*sin(pi/2-{lambda2})',
        '{Z_r1project}':'1*cos(pi/2-{lambda2})'}
    ######################################################################
    #Virial only intermediate terms, includes:
    #   Density times velocity between now and next timestep
    #   Advection term
    equations['virial_intermediate'] = {
        '{rhoUx_cc}':'{Rho [amu/cm^3]}*{U_x [km/s]}',
        '{rhoUy_cc}':'{Rho [amu/cm^3]}*{U_y [km/s]}',
        '{rhoUz_cc}':'{Rho [amu/cm^3]}*{U_z [km/s]}',
        '{rhoU_r [Js/Re^3]}':'{Rho [amu/cm^3]}*1.6605e6*6.371**4*('+
                                                 '{U_x [km/s]}*{X [R]}+'+
                                                 '{U_y [km/s]}*{Y [R]}+'+
                                                 '{U_z [km/s]}*{Z [R]})'}
    ######################################################################
    #Dipole field (requires coordsys and UT information!!!)
    if 'aux' in kwargs:
        aux=kwargs.get('aux')
        Bdx_eq,Bdy_eq,Bdz_eq = get_dipole_field(aux)
        equations['dipole'] = {
                Bdx_eq.split('=')[0]:Bdx_eq.split('=')[-1],
                Bdy_eq.split('=')[0]:Bdy_eq.split('=')[-1],
                Bdz_eq.split('=')[0]:Bdz_eq.split('=')[-1],
               '{Bdmag [nT]}':'sqrt({Bdx}**2+{Bdy}**2+{Bdz}**2)'}
        g = aux['GAMMA']
    else:
        g = '1.6667'
    ######################################################################
    #Volumetric energy terms, includes:
    #   Total Magnetic Energy per volume
    #   Thermal Pressure in Energy units
    #   Kinetic energy per volume
    #   Dipole magnetic Energy
    #+Constructions:
    #   Hydrodynamic Energy Density
    #   Total Energy Density
    equations['volume_energy'] = {
               '{uB [J/Re^3]}':'{Bmag [nT]}**2'+
                                   '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
               '{Pth [J/Re^3]}':'{P [nPa]}*6371**3',
               '{KE [J/Re^3]}':'{Dp [nPa]}/2*6371**3',
               '{uHydro [J/Re^3]}':'({P [nPa]}*1.5+{Dp [nPa]}/2)*6371**3',
               '{uB_dipole [J/Re^3]}':'{Bdmag [nT]}**2'+
                                   '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
               '{u_db [J/Re^3]}':'(({B_x [nT]}-{Bdx})**2+'+
                                '({B_y [nT]}-{Bdy})**2+'+
                                '({B_z [nT]}-{Bdz})**2)'+
                                   '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
               '{Utot [J/Re^3]}':'{uHydro [J/Re^3]}+{uB [J/Re^3]}'}

    ######################################################################
    #Virial Volumetric energy terms, includes:
    #   Disturbance Magnetic Energy per volume
    #   Special construction of hydrodynamic energy density for virial
    equations['virial_volume_energy'] = {
               '{Virial Ub [J/Re^3]}':'(({B_x [nT]}-{Bdx})**2+'+
                                       '({B_y [nT]}-{Bdy})**2+'+
                                       '({B_z [nT]}-{Bdz})**2)'+
                                   '/(2*4*pi*1e-7)*(1e-9)**2*1e9*6371**3',
               '{Virial 2x Uk [J/Re^3]}':'2*{KE [J/Re^3]}+{Pth [J/Re^3]}'}
    ######################################################################
    #Biot Savart terms, includes:
    # delta B in nT
    equations['biot_savart'] = {
               '{dB_x [nT]}':'-({Y [R]}*{J_z [uA/m^2]}-'+
                               '{Z [R]}*{J_y [uA/m^2]})*637.1/{r [R]}**3',
               '{dB_y [nT]}':'-({Z [R]}*{J_x [uA/m^2]}-'+
                               '{X [R]}*{J_z [uA/m^2]})*637.1/{r [R]}**3',
               '{dB_z [nT]}':'-({X [R]}*{J_y [uA/m^2]}-'+
                               '{Y [R]}*{J_x [uA/m^2]})*637.1/{r [R]}**3',
               '{dB [nT]}':'{dB_x [nT]}*{mZhat_x}+{dB_z [nT]}*{mZhat_z}'}
    ######################################################################
    #Energy Flux terms including:
    #   Magnetic field unit vectors
    #   Field Aligned Current Magntitude
    #   Poynting Flux
    #   Total pressure Flux (plasma energy flux)
    #   Total Energy Flux
    equations['energy_flux'] = {
        '{unitbx}':'{B_x [nT]}/'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)',
        '{unitby}':'{B_y [nT]}/'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)',
        '{unitbz}':'{B_z [nT]}/'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)',
        '{J_par [uA/m^2]}':'{unitbx}*{J_x [uA/m^2]} + '+
                              '{unitby}*{J_y [uA/m^2]} + '+
                              '{unitbz}*{J_z [uA/m^2]}',
        '{ExB_x [W/Re^2]}':'{Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_x [km/s]})-{B_x [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
        '{ExB_y [W/Re^2]}':'{Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_y [km/s]})-{B_y [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
        '{ExB_z [W/Re^2]}':'{Bmag [nT]}**2/(4*pi*1e-7)*1e-9*6371**2*('+
                              '{U_z [km/s]})-{B_z [nT]}*'+
                                            '({B_x [nT]}*{U_x [km/s]}+'+
                                             '{B_y [nT]}*{U_y [km/s]}+'+
                                             '{B_z [nT]}*{U_z [km/s]})'+
                                           '/(4*pi*1e-7)*1e-9*6371**2',
        '{P0_x [W/Re^2]}':'({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_x [km/s]}',
        '{P0_y [W/Re^2]}':'({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_y [km/s]}',
        '{P0_z [W/Re^2]}':'({P [nPa]}*(2.5)+{Dp [nPa]}/2)*6371**2'+
                          '*{U_z [km/s]}',
        '{K_x [W/Re^2]}':'{P0_x [W/Re^2]}+{ExB_x [W/Re^2]}',
        '{K_y [W/Re^2]}':'{P0_y [W/Re^2]}+{ExB_y [W/Re^2]}',
        '{K_z [W/Re^2]}':'{P0_z [W/Re^2]}+{ExB_z [W/Re^2]}'}
    ######################################################################
    #Reconnection variables: 
    #   -u x B (electric field in mhd limit)
    #   E (unit change)
    #   current density magnitude
    #   /eta magnetic field diffusivity E/J
    #   /eta (unit change)
    #   magnetic reynolds number (advection/magnetic diffusion)
    equations['reconnect'] = {
        '{minus_uxB_x}':'-({U_y [km/s]}*{B_z [nT]}-'+
                                               '{U_z [km/s]}*{B_y [nT]})',
        '{minus_uxB_y}':'-({U_z [km/s]}*{B_x [nT]}-'+
                                               '{U_x [km/s]}*{B_z [nT]})',
        '{minus_uxB_z}':'-({U_x [km/s]}*{B_y [nT]}-'+
                                               '{U_y [km/s]}*{B_x [nT]})',
        '{E [uV/m]}':'sqrt({minus_uxB_x}**2+'+
                                     '{minus_uxB_y}**2+{minus_uxB_z}**2)',
        '{J [uA/m^2]}':'sqrt({J_x [uA/m^2]}**2+'+
                                   '{J_y [uA/m^2]}**2+{J_z [uA/m^2]}**2)',
        '{eta [m/S]}':'IF({J [uA/m^2]}>0.002,'+
                                      '{E [uV/m]}/({J [uA/m^2]}+1e-9),0)',
        '{eta [Re/S]}':'{eta [m/S]}/(6371*1000)',
        '{Reynolds_m_cell}':'4*pi*1e-4*'+
                 'sqrt({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*'+
                                   '{Cell Size [Re]}/({eta [Re/S]}+1e-9)'}
    ######################################################################
    #Tracking IM GM overwrites
    equations['trackIM'] = {
        '{trackEth_acc [J/Re^3]}':'{dp_acc [nPa]}*6371**3',
        '{trackDp_acc [nPa]}':'{drho_acc [amu/cm^3]}*1e6*1.6605e-27*'+
              '({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2)*1e6*1e9',
        '{trackKE_acc [J/Re^3]}':'{trackDp_acc [nPa]}*6371**3',
        '{trackWth [W/Re^3]}':'IF({dtime_acc [s]}>0,'+
                             '{trackEth_acc [J/Re^3]}/{dtime_acc [s]},0)',
        '{trackWKE [W/Re^3]}':'IF({dtime_acc [s]}>0,'+
                             '{trackKE_acc [J/Re^3]}/{dtime_acc [s]},0)'}
    ######################################################################
    #Entropy and 1D things
    equations['entropy'] = {
        '{s [Re^4/s^2kg^2/3]}':'{P [nPa]}/{Rho [amu/cm^3]}**('+g+')*'+
                                    '1.67**('+g+')/6.371**4*100'}
    ######################################################################
    #Some extra's not normally included:
    equations['parallel'] = {
        '{KEpar [J/Re^3]}':'{Rho [amu/cm^3]}/2 *'+
                                    '(({U_x [km/s]}*{unitbx})**2+'+
                                    '({U_y [km/s]}*{unitby})**2+'+
                                    '({U_z [km/s]}*{unitbz})**2) *'+
                                    '1e6*1.6605e-27*1e6*1e9*6371**3',
        '{KEperp [J/Re^3]}':'{Rho [amu/cm^3]}/2 *'+
                   '(({U_y [km/s]}*{unitbz} - {U_z [km/s]}*{unitby})**2+'+
                    '({U_z [km/s]}*{unitbx} - {U_x [km/s]}*{unitbz})**2+'+
                    '({U_x [km/s]}*{unitby} - {U_y [km/s]}*{unitbx})**2)'+
                                       '*1e6*1.6605e-27*1e6*1e9*6371**3'}
    ######################################################################
    #Terms with derivatives (experimental) !Can take a long time!
    #   vorticity (grad x u)
    equations['development'] = {
        '{W [km/s/Re]}=sqrt((ddy({U_z [km/s]})-ddz({U_y [km/s]}))**2+'+
                              '(ddz({U_x [km/s]})-ddx({U_z [km/s]}))**2+'+
                              '(ddx({U_y [km/s]})-ddy({U_x [km/s]}))**2)'}
    ######################################################################
    return equations

def tec2para(instr):
    badchars = ['{','}','[',']']
    replacements = {' [':'_','**':'^','1e':'10^','pi':'3.14159',#generic
          '/Re':'_Re', 'amu/cm':'amu_cm','km/s':'km_s','/m':'_m',#specific
                    'm^':'m','e^':'e'}#very specific, for units only
            #'/':'_',
    outstr = instr
    for was_,is_ in replacements.items():
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

def read_tecplot(infile):
    """Function reads tecplot binary file
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
    Returns
        sourcedata (pvpython object)- python object attached to theVTKobject
                                      for the input data
    """
    # create a new 'VisItTecplotBinaryReader'
    sourcedata = VisItTecplotBinaryReader(FileName=[infile],
                                   registrationName=infile.split('/')[-1])
    # Call Mesh
    sourcedata.MeshStatus
    # Call Point arrays- we want to load everything
    status = sourcedata.GetProperty('PointArrayInfo')
    #listed as ['thing','1','thing2','0'] where '0' and '1' are 
    #   unloaded and loaded respectively
    arraylist = [s for s in status if s!='0' and s!='1']
    sourcedata.PointArrayStatus = arraylist
    return sourcedata

def fix_names(pipeline,**kwargs):
    names = ProgrammableFilter(registrationName='names', Input=pipeline)
    names.Script = """
        #Get upstream data
        data = inputs[0]
        #These are the variables names that cause issues
        rho = data.PointData["Rho_amu_cm^3"]
        jx = data.PointData["J_x_`mA_m^2"]
        jy = data.PointData["J_y_`mA_m^2"]
        jz = data.PointData["J_z_`mA_m^2"]
        #Copy input to output so we don't lose any data
        output.ShallowCopy(inputs[0].VTKObject)#maintaining other variables
        #Now append the copies of the variables with better names
        output.PointData.append(rho,'Rho_amu_cm3')
        output.PointData.append(jx,'J_x_uA_m2')
        output.PointData.append(jy,'J_y_uA_m2')
        output.PointData.append(jz,'J_z_uA_m2')
    """
    pipeline = names
    return pipeline

def get_vectors(pipeline,**kwargs):
    """Function sets up calculator filters to turn components into vector
        objects (this will allow field tracing and other things)
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            vector_comps (dict)- default empty, will try to detect some
    Return
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    ###Get Vectors from field variable components
    vector_comps = kwargs.get('vector_comps',{})
    #Dig for the variable names so all variables can be vectorized
    points = pipeline.PointData
    var_names = points.keys()
    #info = pipeline.GetPointDataInformation()
    #n_arr = info.GetNumberOfArrays()
    #var_names = ['']*n_arr
    #for i in range(0,n_arr):
    #    var_names[i] = info.GetArray(i).Name
    deconlist=dict([(v.split('_')[0],'_'+'_'.join(v.split('_')[2::]))
                          for v in var_names if('_x' in v or '_y' in v or
                                                             '_z' in v)])
    for (base,tail) in deconlist.items():
        vector = Calculator(registrationName=base,Input=pipeline)
        vector.Function = [base+'_x'+tail+'*iHat+'+
                           base+'_y'+tail+'*jHat+'+
                           base+'_z'+tail+'*kHat'][0]
        vector.ResultArrayName = base+tail
        pipeline=vector
    return pipeline

def get_pressure_gradient(pipeline,**kwargs):
    """Function calculates a pressure gradient variable
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            point_data_name (str)- default \'P_nPa\'
            new_name (str)- default "GradP_nPa_Re"
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    gradP = ProgrammableFilter(registrationName='gradP', Input=pipeline)
    P_name = kwargs.get('point_data_name','\'P_nPa\'')
    gradP_name = kwargs.get('new_name','GradP_nPa_Re')
    gradP.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]

    #Compute gradient
    gradP = algs.gradient(data.PointData"""+str(P_name)+"""])
    data.PointData.append(gradP,"""+str(gradP_name)+""")

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(gradP,"""+str(gradP_name)+""")
    """
    pipeline = gradP
    return pipeline

def get_ffj_filter1(pipeline,**kwargs):
    """Function to calculate the 'fourfieldjunction' to indicate a rxn site
    Inputs
        pipeline (filter/source)- upstream that calculator will process
        kwargs:
            status_opts (list)- default is 0-sw, 1-n, 2-s, 3-closed
    Returns
        pipeline (filter)- last filter applied keeping a straight pipeline
    """
    #Must have the following conditions met first
    ffj =ProgrammableFilter(registrationName='ffj',Input=pipeline)
    ffj.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]
    status = data.PointData['Status']
    m1 = (status==0).astype(int)
    m2 = (status==1).astype(int)
    m3 = (status==2).astype(int)
    m4 = (status==3).astype(int)
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(m1,'m1')
    output.PointData.append(m2,'m2')
    output.PointData.append(m3,'m3')
    output.PointData.append(m4,'m4')
    """
    pipeline = ffj
    return pipeline

def get_ffj_filter2(pipeline,**kwargs):
    #Must have the following conditions met first
    ffj =ProgrammableFilter(registrationName='ffj',Input=pipeline)
    ffj.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]

    m1cc = data.CellData['m1']
    m2cc = data.CellData['m2']
    m3cc = data.CellData['m3']
    m4cc = data.CellData['m4']

    ffj = ((m1cc>0)&
           (m2cc>0)&
           (m3cc>0)&
           (m4cc>0)).astype(int)
    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.CellData.append(ffj,'ffj')
    """
    pipeline = ffj
    return pipeline

def get_magnetopause_filter(pipeline,**kwargs):
    """Function calculates a magnetopause variable, NOTE:will still need to
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
    assert FindSource('beta_star') != None
    betastar_max = kwargs.get('betastar_max',0.7)
    closed_value = kwargs.get('status_closed',3)
    mp_state =ProgrammableFilter(registrationName='mp_state',Input=pipeline)
    mp_state.Script = """
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    # Get input
    data = inputs[0]
    beta_star = data.PointData['beta_star']
    status = data.PointData['Status']

    #Compute magnetopause as logical combination
    mp_state = ((beta_star<"""+str(betastar_max)+""")|
                (status=="""+str(closed_value)+""")).astype(int)

    #Assign to output
    output.ShallowCopy(inputs[0].VTKObject)#So rest of inputs flow
    output.PointData.append(mp_state,'mp_state')
    """
    pipeline = mp_state
    return pipeline

def setup_outline(source,**kwargs):
    """Once variables are set make an outline view of the domain
    Inputs
        source (filter/source)- what will be displayed
        kwargs:
            variablename (str)- what color to represent
    """
    # get active view
    renderView = GetActiveViewOrCreate('RenderView')
    # show data in view
    sourceDisplay = Show(source, renderView,
                        'UnstructuredGridRepresentation')

    # change representation type
    sourceDisplay.SetRepresentationType('Outline')
    # get color transfer function/color map
    variableLUT =GetColorTransferFunction(kwargs.get('variable','betastar'))
    # Hide the scalar bar for this color map if not used.
    HideScalarBarIfNotNeeded(variableLUT, renderView)

def create_iso_surface(inputsource, variable, name, **kwargs):
    """Function creates iso surface from variable
    Inputs
        inputsource (filter/source)- what data is used as input
        variable (str)- name of variable for contour
        name (str)- registration name for new object (filter)
        kwargs:
            iso_value (float)- default 1
            contourtyps (str)- default 'POINTS'
            mergemethod (str)- default 'Uniform Binning'
            trim_regions (bool)- default True, will keep largest connected
    Returns
        outputsource (filter)- filter applied so things can easily attach
    """
    # Create iso surface
    iso1 = Contour(registrationName=name, Input=inputsource)
    iso1.ContourBy = ['POINTS', variable]
    iso1.ComputeNormals = 0#NOTE if comptuted now, seem to cause trouble
    iso1.Isosurfaces = [kwargs.get('iso_value',1)]
    iso1.PointMergeMethod = kwargs.get('mergemethod','Uniform Binning')
    outputsource = iso1

    #Trim any small floating regions
    if kwargs.get('trim_regions',True):
        assert FindSource('MergeBlocks1')!=None
        # Keep only the largest connected region
        RenameSource(name+'_hits', outputsource)
        iso2 = Connectivity(registrationName=name, Input=outputsource)
        iso2.ExtractionMode = 'Extract Largest Region'
        outputsource = iso2

    #Generate normals now that the surface is fully constructed
    if kwargs.get('calc_normals',True):
        RenameSource(name+'_beforeNormals', outputsource)
        iso3 = GenerateSurfaceNormals(registrationName=name,
                                      Input=outputsource)
        iso3.ComputeCellNormals = 1
        iso3.NonManifoldTraversal = 0
        outputsource = iso3

    return outputsource

def point2cell(inputsource, fluxes):
    point2cell = PointDatatoCellData(registrationName='surface_p2cc',
                                     Input=inputsource)
    point2cell.ProcessAllArrays = 0
    convert_list = [f[0] for f in fluxes]
    convert_list.append('Normals')
    point2cell.PointDataArraytoprocess = convert_list
    return point2cell

def get_surface_flux(source,variable,name,**kwargs):
    #First find out if our variable lives on points or cell centers
    #NOTE if on both lists (bad practice) we'll use the cell centered one
    cc = variable in source.CellData.keys()
    if not cc:
        assert variable in source.PointData.keys(), "Bad variable name!"
        vartype = 'Point Data'
    else:
        vartype = 'Cell Data'
    #Create calculator filter that is flux
    flux = Calculator(registrationName=name,Input=source)
    flux.AttributeType = vartype
    flux.Function = 'dot('+variable+',Normals)'
    flux.ResultArrayName = name
    # create a new 'Integrate Variables'
    result=IntegrateVariables(registrationName=name+'_integrated',Input=flux)
    return result

def setup_table(**kwargs):
    """Function sets up a table (spreadsheet view) so data can be exported
    Inputs
        kwargs
            layout_name
            view_name
    Returns
        tableLayout
        tableView
    """
    # create new layout object 'Layout #2'
    tableLayout=CreateLayout(name=kwargs.get('layout_name','tableLayout'))
    # set active view
    SetActiveView(None)

    # Create a new 'SpreadSheet View'
    tableView = CreateView('SpreadSheetView')
    tableView.ColumnToSort = ''
    tableView.BlockSize = 1024

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=tableView, layout=tableLayout, hint=0)
    return tableLayout, tableView

def save_table_data(source, view, path, table_name):
    """Function exports tabular data from a given source to output file
    Inputs
        source
        view
        path
        table_name
    Returns
        None
    """
    # show data in view
    display = Show(source, view, 'SpreadSheetRepresentation')
    # export view
    ExportView(path+table_name+'.csv', view=view)

def export_datacube(pipeline,**kwargs):
    gradP.Script = """
    # Get input
    data = inputs[0]
    p = data.PointData['P_nPa']
    bs = data.PointData['beta_star']

    # Get data statistic info
    extents = data.GetExtent()
    bounds = data.GetBounds()
    # format: [nx0,nxlast, ny0, nylast, ...]
    shape_xyz = [extents[1]+1,
                 extents[3]+1,
                 extents[5]+1]
    # Reshape coordinates based on range/extent
    X = numpy.linspace(bounds[0],bounds[1],shape_xyz[0])
    Y = numpy.linspace(bounds[2],bounds[3],shape_xyz[1])
    Z = numpy.linspace(bounds[4],bounds[5],shape_xyz[2])
    P = numpy.reshape(p,shape_xyz)
    BS = numpy.reshape(bs,shape_xyz)

    # Set output file
    outpath = '/home/aubr/Code/swmf-energetics/localdbug/fte/'
    outname = 'test_downstream_cube.npz'

    # Save data
    numpy.savez(outpath+outname,x=X,y=Y,z=Z,
                                p=P,betastar=BS,
                                dims=shape_xyz)
                    """


def display_visuals(field,mp,renderView,**kwargs):
    """Function standin for separate file governing visual representations
    Inputs
        field (source/filter)- data w all variables for streams, slices etc.
        mp (source/filter)- finalized magnetopause data
        renderView (View)- where to show things
        kwargs:
            mpContourBy
            contourMin,contourMax
            cmap
            doSlice
            sliceContourBy
            sliceContourLog
            doFieldLines
    returns
        TBD
    """
    # show outline of field
    setup_outline(field)
    # show iso surface
    mpDisplay = Show(mp, renderView, 'GeometryRepresentation')

    if 'mpContourBy' in kwargs:
        # set scalar coloring
        ColorBy(mpDisplay, ('POINTS', kwargs.get('mpContourBy')))
        # get color & opacity transfer functions'
        mpLUT = GetColorTransferFunction(kwargs.get('mpContourBy'))
        mpPWF = GetOpacityTransferFunction(kwargs.get('mpContourBy'))
        # Set limits, default both equal
        mpLUT.RescaleTransferFunction(kwargs.get('contourMin',-10),
                                      kwargs.get('contourMax',10))
        mpPWF.RescaleTransferFunction(kwargs.get('contourMin',-10),
                                      kwargs.get('contourMax',10))
        # Apply a preset using its name. Note this may not work as expected
        #   when presets have duplicate names.
        mpLUT.ApplyPreset(kwargs.get('cmap','Cool to Warm (Extended)'), True)
        # Show contour legend
        mpDisplay.SetScalarBarVisibility(renderView,True)

    else:
        # change solid color
        ColorBy(mpDisplay, None)
        mpDisplay.AmbientColor = [0.0, 1.0, 1.0]
        mpDisplay.DiffuseColor = [0.0, 1.0, 1.0]

    # Properties modified on mpDisplay.DataAxesGrid
    mpDisplay.DataAxesGrid.GridAxesVisibility = 1
    # Properties modified on slice1Display
    mpDisplay.Opacity = 1


    if kwargs.get('doFFJ',True):
        isoFFJ = Contour(registrationName='FFJ', Input=field)
        isoFFJ.ContourBy = ['POINTS', variable]
        #isoFFJ.ContourBy = ['CELLS', 'ffj']
        isoFFJ.ComputeNormals = 0
        isoFFJ.Isosurfaces = [1]
        isoFFJdisplay = Show(isoFFJ, renderView, 'GeometryRepresentation')
        #ColorBy(isoFFJdisplay, ('CELLS', 'ffj'))
        ColorBy(isoFFJdisplay, ('POINTS', 'ffj'))
        #ColorBy(isoFFJdisplay, None)
        #isoFFJ.AmbientColor = [0.0, 1.0, 1.0]
        #isoFFJ.DiffuseColor = [0.0, 1.0, 1.0]

    if kwargs.get('doSlice',False):
        ###Slice
        # create a new 'Slice'
        slice1 = Slice(registrationName='Slice1', Input=field)
        if 'Plane' in kwargs.get('slicetype','yPlane'):
            slice1.SliceType = 'Plane'
            slice1.HyperTreeGridSlicer = 'Plane'
            slice1.SliceOffsetValues = [0.0]

            if 'y' in kwargs.get('slicetype','yPlane'):
                # init the 'Plane' selected for 'SliceType'
                slice1.SliceType.Origin = [0.0, 0.0, 0.0]
                slice1.SliceType.Normal = [0.0, 1.0, 0.0]

                # init the 'Plane' selected for 'HyperTreeGridSlicer'
                slice1.HyperTreeGridSlicer.Origin = [-94.25, 0.0, 0.0]
            else:
                #TODO
                pass
        # show data in view
        slice1Display = Show(slice1, renderView, 'GeometryRepresentation')
        # Properties modified on slice1Display
        slice1Display.Opacity = 0.6
        # set scalar coloring
        ColorBy(slice1Display, ('POINTS', 'B_z_nT'))
        # get color transfer function/color map for 'Rho_amu_cm3'
        bzLUT = GetColorTransferFunction('B_z_nT')
        bzLUT.RescaleTransferFunction(-10.0, 10.0)
        bzLUT.ApplyPreset('Gray and Red', True)
        bzLUT.InvertTransferFunction()

        # get opacity transfer function/opacity map for 'Rho_amu_cm3'
        bzPWF = GetOpacityTransferFunction('B_z_nT')


        # convert to log space
        #bzLUT.MapControlPointsToLogSpace()

        # Properties modified on rho_amu_cm3LUT
        #rho_amu_cm3LUT.UseLogScale = 1

        # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
        #bzLUT.ApplyPreset('Inferno (matplotlib)', True)
        # show color bar/color legend
        slice1Display.SetScalarBarVisibility(renderView, True)

    '''
    # camera placement for renderView
    renderView.CameraPosition = [123.8821359932328, 162.9578433260544, 24.207094682916125]
    renderView.CameraFocalPoint = [-56.49901724813269, -32.56582457808803, -12.159020395552512]
    renderView.CameraViewUp = [-0.06681529228180919, -0.12253235709025494, 0.9902128751855344]
    renderView.CameraParallelScale = 218.09415971089186
    '''
    # Zoomed dayside +Y side magnetosphere
    renderView.CameraPosition = [33.10, 56.51, 14.49]
    renderView.CameraFocalPoint = [-33.19, -15.11, -3.46]
    renderView.CameraViewUp = [-0.10, -0.15, 0.98]
    renderView.CameraParallelScale = 66.62

    '''
    renderView1.CameraPosition = [29.48, 55.98, 10.02]
    renderView1.CameraFocalPoint = [-54.66, -35.22, -6.93]
    renderView1.CameraViewUp = [-0.06, -0.12, 0.99]
    renderView1.CameraParallelScale = 218.09
    '''

def setup_pipeline(infile,**kwargs):
    """Function takes single data file and builds pipeline to find and
        visualize magnetopause
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
        kwargs:
            get_gradP (bool)- default false, will add additional variable
    Returns
        source (pvpython object)- python object attached to the VTKobject
                                  for the input data
        pipelinehead (pypython filter)- top level filter which starts the
                                        pipeline processing
        field (source/filter)- where the dataset has finished creating new
                               variables
        mp (source/filter)- final version of magnetopause
    """
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Read input file
    sourcedata = read_tecplot(infile)

    # apply 'Merge Blocks' so 'Connectivity' can be used
    mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1',
                               Input=sourcedata)

    ##Set the head of the pipeline, we will want to return this!
    pipelinehead = mergeBlocks1
    pipeline = mergeBlocks1


    ###Rename some tricky variables
    pipeline = fix_names(pipeline)
    ###Build functions up to betastar
    alleq = equations()
    pipeline = eqeval(alleq['basic_physics'],pipeline)
    ###Energy flux variables
    pipeline = eqeval(alleq['energy_flux'],pipeline)
    ###Get Vectors from field variable components
    pipeline = get_vectors(pipeline)

    ###Programmable filters
    # Pressure gradient, optional variable
    if kwargs.get('doGradP',False):
        pipeline = get_pressure_gradient(pipeline)
    if kwargs.get('ffj',True):
        ffj1 = get_ffj_filter1(pipeline)
        ffj2 = PointDatatoCellData(registrationName='ffj_interp1',
                                   Input=ffj1)
        ffj2.ProcessAllArrays = 1
        ffj3 = get_ffj_filter2(ffj2)
        ffj4 = CellDatatoPointData(registrationName='ffj_interp2',
                                   Input=ffj3)
        pipeline = ffj4

    # Magnetopause
    pipeline = get_magnetopause_filter(pipeline)

    ###Now that last variable is done set 'field' for visualizer and a View
    field = pipeline

    ###Contour (iso-surface) of the magnetopause
    mp = create_iso_surface(pipeline, 'mp_state', 'mp')

    return sourcedata, pipelinehead, field, mp

if __name__ == "__main__":
#if True:
    start_time = time.time()
    ######################################################################
    # USER INPUTS
    ######################################################################
    path='/home/aubr/Code/swmf-energetics/localdbug/fte/30min/'
    outpath = 'output5_fte_pv/'
    ######################################################################

    #Make the paths if they don't already exist
    os.makedirs(path, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    filelist = glob.glob(path+'*.plt')
    for infile in filelist[0:1]:
        print('processing '+infile.split('/')[-1]+'...')
        oldsource,pipelinehead,field,mp=setup_pipeline(infile)
        ###Surface flux on magnetopause
        get_surface_flux(mp, 'B_nT','Bnormal_net')
        mp_Bnorm = FindSource('Bnormal_net')
        #decide which values to calculate (will need to make cell data)
        #fluxes = [('K_W_Re2','k_flux'),('P0_W_Re2','h_flux'),
        #          ('ExB_W_Re2','p_flux')]
        #mp_cc = point2cell(mp,fluxes)#mp object with cell centered data
        #mp_K_flux = get_surface_flux(mp, 'K_W_Re2','k_flux')
        #mp_S_flux = get_surface_flux(mp_cc, 'ExB_W_Re2','s_net_flux')
        renderView1 = GetActiveViewOrCreate('RenderView')
        #TODO find how to limit integration variables and group all together
        #tableLayout, tableView = setup_table()
        #save_table_data(mp_S_flux, tableView, outpath,'s_net_flux')
        SetActiveView(renderView1)
        display_visuals(field,mp,renderView1,doSlice=True,
                        mpContourBy='B_x_nT',contourMin=-40,contourMax=40)

        # Create a new 'Render View'
        layout = GetLayout()
        layout.SplitVertical(0, 0.5)
        renderView2 = CreateView('RenderView')
        # assign view to a particular cell in the layout
        AssignViewToLayout(view=renderView2, layout=layout, hint=2)
        display_visuals(field,mp_Bnorm,renderView2,doSlice=True,
                        mpContourBy='Bnormal_net',
                        contourMin=-20,contourMax=20,
                        cmap='Cool to Warm')

        # Render and save screenshot
        RenderAllViews()
        # layout/tab size in pixels
        # get layout
        layout1 = GetLayout()
        layout1.SetSize(2162, 1079)
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
    '''
    for infile in filelist[1::]:
        print('processing '+infile.split('/')[-1]+'...')
        #Read in new file unattached to current pipeline
        SetActiveSource(None)
        newsource = read_tecplot(infile)

        #Attach pipeline to the new source file and delete the old
        pipelinehead.Input = newsource
        Delete(oldsource)

        # Render and save screenshot
        RenderAllViews()
        # layout/tab size in pixels
        # get layout
        layout1 = GetLayout()
        layout1.SetSize(2162, 1079)
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',
                       renderView)

        # Set the current source to be replaced on next loop
        oldsource = newsource
    '''
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
