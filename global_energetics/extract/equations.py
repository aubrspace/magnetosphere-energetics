#!/usr/bin/env python3
"""Equations for use in tecplot, see conversion tools for use in paraview
"""
from numpy import sin,cos,pi,deg2rad

def rotation(angle, axis):
    """Function returns rotation matrix given axis and angle
    Inputs
        angle
        axis
    Outputs
        matrix
    """
    if axis == 'x' or axis == 'X':
        matrix = [[1,           0,          0],
                  [0,  cos(angle), sin(angle)],
                  [0, -sin(angle), cos(angle)]]
    elif axis == 'y' or axis == 'Y':
        matrix = [[ cos(angle), 0, sin(angle)],
                  [0,           1,          0],
                  [-sin(angle), 0, cos(angle)]]
    elif axis == 'z' or axis == 'Z':
        matrix = [[ cos(angle), sin(angle), 0],
                  [-sin(angle), cos(angle), 0],
                  [0,           0,          1]]
    return matrix

def get_dipole_field(auxdata, *, B0=31000):
    """Function calculates dipole field in given coordinate system based on
        current time in UTC
    Inputs
        auxdata- tecplot object containing key data pairs
        B0- surface magnetic field strength
    """
    #Determine dipole vector in given coord system
    theta_tilt = float(auxdata['BTHETATILT'])
    axis = [sin(deg2rad(theta_tilt)), 0, -1*cos(deg2rad(theta_tilt))]
    #Create dipole matrix
    ######################################
    #   (3x^2-r^2)      3xy         3xz
    #B =    3yx     (3y^2-r^2)      3yz    * vec{m}
    #       3zx         3zy     (3z^2-r^2)
    ######################################
    M11 = '(3*{X [R]}**2-{r [R]}**2)'
    M12 = '3*{X [R]}*{Y [R]}'
    M13 = '3*{X [R]}*{Z [R]}'
    M21 = M12
    M22 = '(3*{Y [R]}**2-{r [R]}**2)'
    M23 = '3*{Y [R]}*{Z [R]}'
    M31 = M13
    M32 = M23
    M33 = '(3*{Z [R]}**2-{r [R]}**2)'
    #Multiply dipole matrix by dipole vector
    d_x='{Bdx}='+str(B0)+'/{r [R]}**5*('+(M11+'*'+str(axis[0])+'+'+
                                            M12+'*'+str(axis[1])+'+'+
                                            M13+'*'+str(axis[2]))+')'
    d_y='{Bdy}='+str(B0)+'/{r [R]}**5*('+(M21+'*'+str(axis[0])+'+'+
                                            M22+'*'+str(axis[1])+'+'+
                                            M23+'*'+str(axis[2]))+')'
    d_z='{Bdz}='+str(B0)+'/{r [R]}**5*('+(M31+'*'+str(axis[0])+'+'+
                                            M32+'*'+str(axis[1])+'+'+
                                            M33+'*'+str(axis[2]))+')'
    #Return equation strings to be evaluated
    return d_x, d_y, d_z

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
                       #'{Cell Size [Re]}':'{dvol [R]^3}**(1/3)',
                       '{h}':'sqrt({Y [R]}**2+{Z [R]}**2)'}
    #2D versions of spatial variables
    equations['basic2d_XY'] = {'{r [R]}':'sqrt({X [R]}**2 + {Y [R]}**2)'}
    equations['basic2d_XZ'] = {'{r [R]}':'sqrt({X [R]}**2 + {Z [R]}**2)'}
    #Dipolar coordinate variables
    if 'aux' in kwargs and kwargs.get('is3D',True):
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
         '{theta_SM}':'-180/pi*{lambda}',
         '{Xd [R]}':'{mXhat_x}*({X [R]}*{mXhat_x}+{Z [R]}*{mXhat_z})',
         '{Zd [R]}':'{mZhat_z}*({X [R]}*{mZhat_x}+{Z [R]}*{mZhat_z})',
         '{phi_SM}':'atan2({Y [R]}, {Xd [R]})',
         '{U_xd [km/s]}':'{mXhat_x}*'+
                        '({U_x [km/s]}*{mXhat_x}+{U_z [km/s]}*{mXhat_z})',
         '{U_zd [km/s]}':'{mZhat_z}*'+
                        '({U_x [km/s]}*{mZhat_x}+{U_z [km/s]}*{mZhat_z})',
         '{U_r}':'({U_xd [km/s]}*{Xd [R]}+'+
                  '{U_y [km/s]}*{Y [R]}+'+
                  '{U_zd [km/s]}*{Zd [R]})/{r [R]}',
         '{U_txd}':'{U_xd [km/s]}-{U_r}*{Xd [R]}/{r [R]}',
         '{U_ty}':'{U_y [km/s]}-{U_r}*{Y [R]}/{r [R]}',
         '{U_tzd}':'{U_zd [km/s]}-{U_r}*{Zd [R]}/{r [R]}'}
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
     '{Bmag [nT]}':'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2)',
     '{Br [Wb/Re^2]}':'({B_x [nT]}*{X [R]}+{B_y [nT]}*{Y [R]}+'+
                       '{B_z [nT]}*{Z [R]})/{r [R]}*6.371**2*1e3'}
    ######################################################################
    #Fieldlinemaping
    equations['fieldmapping'] = {
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
    if kwargs.get('aux')!=None:
        aux=kwargs.get('aux')
        Bdx_eq,Bdy_eq,Bdz_eq = get_dipole_field(aux)
        equations['dipole'] = {
                Bdx_eq.split('=')[0]:Bdx_eq.split('=')[-1],
                Bdy_eq.split('=')[0]:Bdy_eq.split('=')[-1],
                Bdz_eq.split('=')[0]:Bdz_eq.split('=')[-1],
               '{Bdmag [nT]}':'sqrt({Bdx}**2+{Bdy}**2+{Bdz}**2)'}
        g = aux['GAMMA']
    else:
        g = '1.66667'
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
    #Volumetric mass terms, includes:
    #   Total Mass per volume
    equations['volume_mass'] = {
               '{M [kg/Re^3]}':'{Rho [amu/cm^3]}*1.6605*10e-12*6371**3'}
    ######################################################################
    #Daynightmapping terms, includes:
    # Specific mappings for lobes
    #   daymapped_nlobe
    #   nightmapped_nlobe
    #   daymapped_slobe
    #   nightmapped_slobe
    # Generic mappings for closed and general magnetosphere
    #   daymapped
    #   nightmapped
    if False:
    #if 'aux' in kwargs:
        aux=kwargs.get('aux')
        phi1 = '{phi_centroid_1}'
        phi2 = '{phi_centroid_2}'
        phi1max = aux['phi_max_north']
        phi1min = aux['phi_min_north']
        phi2max = aux['phi_max_south']
        phi2min = aux['phi_min_south']
    else:
        phi1 = '{phi_1 [deg]}'
        phi2 = '{phi_2 [deg]}'
        phi1max = '270'
        phi1min = '90'
        phi2max = '270'
        phi2min = '90'
    equations['daynightmapping'] = {
        '{daymapped_nlobe}':'IF('+phi1+'>'+phi1max+'||'+
                              '('+phi1+'<'+phi1min+'&&'+phi1+'>0),1,0)',
        '{nightmapped_nlobe}':'IF('+phi1+'<'+phi1max+'&&'+
                                 ''+phi1+'>'+phi1min+',1,0)',
        '{daymapped_slobe}':'IF('+phi2+'>'+phi2max+'||'+
                              '('+phi2+'<'+phi2min+'&&'+phi2+'>0),1,0)',
        '{nightmapped_slobe}':'IF('+phi2+'<'+phi2max+'&&'+
                                 ''+phi2+'>'+phi2min+',1,0)',
        '{daymapped}':'IF(('+phi1+'>='+phi1max+'||'+
                        '('+phi1+'<='+phi1min+'&&'+phi1+'>=0))||'+
                         '('+phi2+'>='+phi2max+'||'+
                         '('+phi2+'<='+phi2min+'&&'+phi2+'>=0)),1,0)',
        '{nightmapped}':'IF(('+phi1+'<'+phi1max+'&&'+phi1+'>'+phi1min+')&&'+
                          '('+phi2+'<'+phi2max+'&&'+phi2+'>'+phi2min+'),1,0)',
            }
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
        '{unitbx}':'{B_x [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
        '{unitby}':'{B_y [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
        '{unitbz}':'{B_z [nT]}/MAX(1e-15,'+
                        'sqrt({B_x [nT]}**2+{B_y [nT]}**2+{B_z [nT]}**2))',
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
    #Wave Energy terms:
    equations['wave_energy'] = {
        '{UdotB}':'({U_x [km/s]}*{B_x [nT]}+'+
                   '{U_y [km/s]}*{B_y [nT]}+'+
                   '{U_z [km/s]}*{B_z [nT]})/{Bmag [nT]}',
        #'{Upar_x}':'({UdotB}/{Bmag [nT]}**2 * {B_x [nT]})',
        #'{Upar_y}':'({UdotB}/{Bmag [nT]}**2 * {B_y [nT]})',
        #'{Upar_z}':'({UdotB}/{Bmag [nT]}**2 * {B_z [nT]})',
        #'{U_perp [km/s]}':'sqrt(({U_x [km/s]} - {Upar_x})**2+'+
        #                       '({U_y [km/s]} - {Upar_y})**2+'+
        #                       '({U_z [km/s]} - {Upar_z})**2)',
        '{U_perp [km/s]}':'sqrt({U_x [km/s]}**2+{U_y [km/s]}**2+'+
                               '{U_z [km/s]}**2-{UdotB}**2)',
        '{gradU_perp [km/s/Re]}':'-('+
            'ddx({U_perp [km/s]})*{B_x [nT]}+'+
            'ddy({U_perp [km/s]})*{B_y [nT]}+'+
            'ddz({U_perp [km/s]})*{B_z [nT]})/{Bmag [nT]}',
        '{deltaU}':'{gradU_perp [km/s/Re]}*{Cell Size}',
        '{Valf [km/s]}':'({Bmag [nT]}*1e-9/sqrt('+
             '4*3.14159*10**-7*{Rho [amu/cm^3]}*1.66054*10**-27*10**6))*1e-3',
        '{Valf_x [km/s]}':'({B_x [nT]}*1e-9/sqrt('+
             '4*3.14159*10**-7*{Rho [amu/cm^3]}*1.66054*10**-27*10**6))*1e-3',
        '{Valf_y [km/s]}':'({B_y [nT]}*1e-9/sqrt('+
             '4*3.14159*10**-7*{Rho [amu/cm^3]}*1.66054*10**-27*10**6))*1e-3',
        '{Valf_z [km/s]}':'({B_z [nT]}*1e-9/sqrt('+
             '4*3.14159*10**-7*{Rho [amu/cm^3]}*1.66054*10**-27*10**6))*1e-3',
        #'{deltaU}':'{gradU_perp [km/s/Re]}*{Valf [km/s]}*10/6371',
        '{sawS_x [W/Re^2]}':'1/2*{Rho [amu/cm^3]}*{deltaU}**2*sign({deltaU})'+
                          '*{Valf_x [km/s]}*(6371**2*1.66054*10**-27*10**21)',
        '{sawS_y [W/Re^2]}':'1/2*{Rho [amu/cm^3]}*{deltaU}**2*sign({deltaU})'+
                          '*{Valf_y [km/s]}*(6371**2*1.66054*10**-27*10**21)',
        '{sawS_z [W/Re^2]}':'1/2*{Rho [amu/cm^3]}*{deltaU}**2*sign({deltaU})'+
                          '*{Valf_z [km/s]}*(6371**2*1.66054*10**-27*10**21)'}
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
    equations['ffj_setup'] = {
        '{m1}':'if({Status}==0,1,0)',
        '{m2}':'if({Status}==1,1,0)',
        '{m3}':'if({Status}==2,1,0)',
        '{m4}':'if({Status}==3,1,0)'}
    equations['ffj'] = {
            '{m1_cc}':'{m1}',
            '{m2_cc}':'{m2}',
            '{m3_cc}':'{m3}',
            '{m4_cc}':'{m4}',
            '{ffj}':'if({m1_cc}>0&&{m2_cc}>0&&{m3_cc}>0&&{m4_cc}>0,1,0)'}
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
    equations['local_curv']={'{mag_curl_bhat}':'sqrt('+
                                         '(ddy({unitbz})-ddz({unitby}))**2+'+
                                         '(ddz({unitbx})-ddx({unitbz}))**2+'+
                                         '(ddx({unitby})-ddy({unitbx}))**2)'}
    ######################################################################
    return equations

