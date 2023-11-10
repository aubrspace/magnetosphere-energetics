#!/usr/bin/env python3
"""Extraction routine for ionosphere surface
"""
#import logging as log
import os
import sys
import time
import glob
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import datetime as dt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
#interpackage modules
from global_energetics.makevideo import get_time, time_sort
from global_energetics.write_disp import write_to_hdf, display_progress
from global_energetics.extract.tec_tools import (integrate_tecplot,mag2gsm,
                                                    create_stream_zone,
                                                    calc_terminator_zone,
                                                    get_global_variables)
from global_energetics.extract.equations import (get_dipole_field)
from global_energetics.extract import line_tools
from global_energetics.extract import surface_tools

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def read_idl_ascii(infile,**kwargs):
    """Function reads .idl file that is ascii formatted
    Inputs
        infile
        kwargs:
    Returns
        north,south (DataFrame)- data for north and south hemispheres
        aux (dict)- dictionary of other information
    """
    headers = ['VALUES','TIME','SIMULATION','DIPOLE']
    aux = {}
    variables = []
    i=0
    with open(infile,'r') as f:
        for line in f:
            if 'TITLE' in line:
                title = f.readline()
                i+=1
            elif 'VARIABLE' in line:
                value = f.readline()
                i+=1
                isblank = all([c==''for c in value.split('\n')[0].split(' ')])
                while not isblank:
                    name= '_'.join([c.split('\n')[0] for c in value.split(' ')
                                                    if c!='\n' and c!=''][1::])
                    variables.append(name)
                    value = f.readline()
                    i+=1
                    isblank=all([c==''for c in value.split('\n')[0].split(' ')])
            elif any([h in line for h in headers]):
                value = f.readline()
                i+=1
                isblank = all([c==''for c in value.split('\n')[0].split(' ')])
                while not isblank:
                    qty = [c for c in value.split(' ') if c!='\n' and c!=''][0]
                    name= '_'.join([c.split('\n')[0] for c in value.split(' ')
                                                    if c!='\n' and c!=''][1::])
                    if qty.isnumeric(): qty = int(qty)
                    elif isfloat(qty): qty = float(qty)
                    aux[name] = qty
                    value = f.readline()
                    i+=1
                    isblank=all([c==''for c in value.split('\n')[0].split(' ')])
            elif 'BEGIN NORTH' in line:
                i_ns, northstart = i, line
            elif 'BEGIN SOUTH' in line:
                i_ss, southstart = i, line
            i+=1
    north = pd.read_csv(infile,sep='\s+',skiprows=i_ns+1,names=variables,
                        nrows=i_ss)
    south = pd.read_csv(infile,sep='\s+',skiprows=i_ss+1,names=variables)
    return north,south,aux

def calc_shell_variables(ds, **kwargs):
    """Calculates helpful variables such as cell area (labeled as volume)
    """
    tp.macro.execute_extended_command('CFDAnalyzer3',
                                      'CALCULATE FUNCTION = '+
                                      'CELLVOLUME VALUELOCATION = '+
                                      'CELLCENTERED')
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    #Generate cellcentered versions of postitional variables
    for var in ['X [R]','Y [R]','Z [R]','JouleHeat [mW/m^2]']:
        if var in zone.dataset.variable_names:
            newvar = var.split(' ')[0].lower()+'_cc'
            eq('{'+newvar+'}={'+var+'}', value_location=CC,
                                        zones=[zone.index])

def get_ionosphere_zone(eventdt, datapath):
    """Function to find the correct ionosphere datafile and append the data
        to the current tecplot session
    Inputs
        eventdt- datetime object of event time
        datapath- str path to ionosphere data
    """
    eventstring = str(eventdt)
    #parse event string (will update later)
    yr = eventstring.split('-')[0][-2::]
    mn = eventstring.split('-')[1]
    dy = eventstring.split('-')[2].split(' ')[0]
    hr = eventstring.split(' ')[1].split(':')[0]
    minute = eventstring.split(':')[1]
    datafile = 'it'+yr+mn+dy+'_'+hr+minute+'00_000.tec'
    #load file matching description
    if os.path.exists(datapath+datafile):
        ie_data = tp.data.load_tecplot(datapath+datafile)
        north_iezone = ie_data.zone('IonN *')
        south_iezone = ie_data.zone('IonS *')
        return north_iezone, south_iezone
    else:
        print('no ionosphere data found!')
        return None, None

'''
def get_ionosphere(field_data, *, show=False,
                   comp1=True, comp2=True, comp3=True,
                   local_integrate=False):
    """Function that finds, plots and calculates energetics on the
        ionosphere surface.
    Inputs
        field_data- tecplot DataSet object with 3D field data
        ndatafile, sdatafile- ionosphere data, assumes .csv files
        show- boolean for display
        comp1, comp2- boolean for which comp to include
        local_integrate- boolean for integration in python or pass to tec
    """
    #get date and time info from datafiles name

    #Read .csv files into dataframe objects

    #Call calc_components for each hemisphere

    if show:
        #display component values
        pass

    if local_integrate:
        #integrate in python using pandas
        comp1var, comp2var, comp3var = [],[],[]
        compvars = [comp1var, comp2var, comp3var]
        compstrings = ['comp1', 'comp2', 'comp3']
        for comp in enumerate([comp1, comp2, comp3]):
            if comp[1]:
                compvar[comp[0]] = (northdf[compstrings[comp[0]]].sum()+
                                   southdf[compstrings[comp[0]]].sum())[0]
        [comp1var, comp2var, comp3var] = compvars
    else:
        #pass data to tecplot
        load_ionosphere_tecplot(northdf, southdf)
'''

def save_tofile(infile,timestamp,filetype='hdf',outputdir='localdbug/ie',
                hdfkey='ie',**values):
    """Function saves data to file
    Inputs
        infile (str)- input filename
        timestamp (datettime)
        filetype (str)- only hdf5 supported
        hdfkey (str)
        values:
            dict(list of values)- typically single valued list
    """
    df = pd.DataFrame(values)
    df.index = [timestamp]
    #output info
    outfile = '/'+infile.split('/')[-1].split('it')[-1].split('.')[0]
    if 'hdf' in filetype:
        df.to_hdf(outputdir+outfile+'.h5', key='ie')
    if 'ascii' in filetype:
        df.to_csv(outputdir+outfile+'.dat',sep=' ',index=False)

def rotate_xyz(zones,angle,**kwargs):
    """Function rotates IE xyz variables according to rotation matrix
    Inputs
        zone
        angle
        kwargs:
            sm2gsm
    Returns
    """
    eq =  tp.data.operate.execute_equation
    # Get rotation matrix
    mXhat_x = str(sin((-angle+90)*pi/180))
    mXhat_y = str(0)
    mXhat_z = str(-1*cos((-angle+90)*pi/180))
    mZhat_x = str(sin(-angle*pi/180))
    mZhat_y = str(0)
    mZhat_z = str(-1*cos(-angle*pi/180))
    # Save old values
    eq('{Xd [R]} = {X [R]}',zones=zones)
    eq('{Zd [R]} = {Z [R]}',zones=zones)
    # Update xyz
    eq('{X [R]} = '+mXhat_x+'*({Xd [R]}*'+mXhat_x+'+{Zd [R]}*'+mXhat_z+')',
                                                                 zones=zones)
    eq('{Z [R]} = '+mZhat_z+'*({Xd [R]}*'+mZhat_x+'+{Zd [R]}*'+mZhat_z+')',
                                                                 zones=zones)
    # Update dirctions of other vectors
    for base in ['E','J','U']:
        if zones[0].dataset.variable(base+'x*') is not None:
            Xname = zones[0].dataset.variable(base+'x*').name
            Zname = zones[0].dataset.variable(base+'z*').name
            eq('{'+base+'xd} = {'+Xname+'}',zones=zones)
            eq('{'+base+'zd} = {'+Zname+'}',zones=zones)
            eq('{'+Xname+'} = '+mXhat_x+'*({'+base+'xd}*'+mXhat_x+
                              '+{'+base+'zd}*'+mXhat_z+')',zones=zones)
            eq('{'+Zname+'} = '+mZhat_z+'*({'+base+'xd}*'+mZhat_x+
                              '+{'+base+'zd}*'+mZhat_z+')',zones=zones)
    # Calculate the dipole field at the new locations
    Bdx_eq,Bdy_eq,Bdz_eq = get_dipole_field({'BTHETATILT':str(angle)})
    eq('{r [R]} = sqrt({X [R]}**2+{Y [R]}**2+{Z [R]}**2)',zones=zones)
    eq(Bdx_eq,zones=zones)
    eq(Bdy_eq,zones=zones)
    eq(Bdz_eq,zones=zones)

def trace_status(points,zone,hemi,**kwargs):
    """Function traces all the points in a zone to detrmine if each point is
        open or closed
    Inputs
        zone
        kwargs:
    Returns
    """
    eq =  tp.data.operate.execute_equation
    # set vector settings
    tp.active_frame().plot().vector.u_variable = zone.dataset.variable('B_x *')
    tp.active_frame().plot().vector.v_variable = zone.dataset.variable('B_y *')
    tp.active_frame().plot().vector.w_variable = zone.dataset.variable('B_z *')
    field_line = tp.active_frame().plot().streamtraces
    stat_var = kwargs.get('stat_var','Status')
    if '_cc' in stat_var:
        x = zone.values('x_cc2').as_numpy_array()
        y = zone.values('y_cc2').as_numpy_array()
        z = zone.values('z_cc2').as_numpy_array()
    else:
        x = zone.values('X *').as_numpy_array()
        y = zone.values('Y *').as_numpy_array()
        z = zone.values('Z *').as_numpy_array()
    #field_line.min_step_size=0.25
    #field_line.max_steps=3000
    # Use value blanking
    # trace lines
    if hemi=='North':
        direction=StreamDir.Reverse
    if hemi=='South':
        direction=StreamDir.Forward
    for i,p in enumerate([p for p in points]):
        if True:
            # Find the location in the target_zone
            index = int(np.where((x==p[0])&(y==p[1])&(z==p[2]))[0][0])
            field_line.add(seed_point=p, direction=direction,
                           stream_type=Streamtrace.VolumeLine)
            field_line.extract()
            field_line.delete_all()
            if hemi=='North':
                if zone.dataset.zone(-1).values('r *')[0:1]<2.75:
                    zone.values(stat_var)[index] = 3
                else:
                    zone.values(stat_var)[index] = 2
            elif hemi=='South':
                if zone.dataset.zone(-1).values('r *')[-2:-1]<2.75:
                    zone.values(stat_var)[index] = 3
                else:
                    zone.values(stat_var)[index] = 1
        else:
            zone.values(stat_var)[index] = 3
        delete_zones = zone.dataset.zone(-1)
        zone.dataset.delete_zones([delete_zones])
        #if '_cc' in stat_var:
        #    from IPython import embed; embed()
        #    time.sleep(3)
        if i%100==0:
            print(i)
    '''
    #TODO implement this to double check all islands
    for i,value in enumerate(zone.values(stat_var).as_numpy_array()):
        if i<181:
            behind_loop = len(zone.values(stat_var))-181+i
        else:
            behind_loop = i-181
        if i>(len(zone.values(stat_var))-181):
            forward_loop = i+181-len(zone.values(stat_var))
        else:
            forward_loop = i+181
        if i==0:
            previous = len(zone.values(stat_var))
        else:
            previous = i-1
        if i==len(zone.values(stat_var)):
            next = 0
        else:
            next = i+1
        check_array = [value-zone.values(stat_var)[behind_loop],
                       value-zone.values(stat_var)[forward_loop],
                       value-zone.values(stat_var)[previous],
                       value-zone.values(stat_var)[next]]
        if check_array.count(0) == 0:
            pass
    '''
    # Delete traced zones
    #if 'cc' not in stat_var:
    #    delete_zones = [z for z in zone.dataset.zones('Streamtrace*')]
    #    zone.dataset.delete_zones(delete_zones)
    # Turn off value blanking
    #closed_blank.active=False
    #blank.active = False

def nearest_neighbor_map(iezone,source,hemi):
    """
    """
    # pull ie coordinates
    iex = iezone.values('X *').as_numpy_array()
    iey = iezone.values('Y *').as_numpy_array()
    iez = iezone.values('Z *').as_numpy_array()
    ietheta = iezone.values('Theta *').as_numpy_array()
    iephi = iezone.values('Psi *').as_numpy_array()
    # pull souce coodinates and status
    source_status = source.values('Status').as_numpy_array()
    if hemi=='North':
        source_theta = source.values('theta_1 *').as_numpy_array()
        source_phi = source.values('phi_1 *').as_numpy_array()
        # Take only the top
        source_phi = source_phi[source_theta>0]
        source_status = source_status[source_theta>0]
        source_theta = source_theta[source_theta>0]
        #ietheta_match = -ietheta+90
        source_thmatch = -source_theta+90
    else:
        source_theta = source.values('theta_2 *').as_numpy_array()
        source_phi = source.values('phi_2 *').as_numpy_array()
        # Take only the bottom
        source_phi = source_phi[(source_theta<0)&(source_theta)>-90]
        source_status = source_status[(source_theta<0)&(source_theta)>-90]
        source_theta = source_theta[(source_theta<0)&(source_theta)>-90]
        #ietheta_match = ietheta-90
        source_match = -source_theta+90
    for i,(th_in,phi_in) in enumerate(zip(ietheta[0:10000],
                                          iephi[0:10000])):
        th1 = source_thmatch*pi/180
        phi1 = source_phi*pi/180
        th2 = th_in*pi/180
        phi2 = phi_in*pi/180
        d = np.sqrt(2*(1-sin(th1)*sin(th2)*cos(phi1-phi2)+cos(th1)*cos(th2)))
        status_value = source_status[np.where(d==d.min())[0][0]]
        if hemi=='North' and status_value<3:
            iezone.values('Status')[i] = 2
            print(i, 2)
        elif hemi=='South' and status_value<3:
            iezone.values('Status')[i] = 1
            print(i, 1)
        else:
            iezone.values('Status')[i] = 3
            print(i, 3)
    #from IPython import embed; embed()

def sphere2cart_map(spzone,**kwargs):
    eq = tp.data.operate.execute_equation
    # Calculate Theta from theta_1/2
    eq('{Theta [deg]}=IF({Zd [R]}>0,-{theta_1 [deg]}+90,-{theta_2 [deg]}+90)',
                                                               zones=[spzone])
    eq('{Psi [deg]}=IF({Zd [R]}>0,{phi_1 [deg]},{phi_2 [deg]})',
                                                               zones=[spzone])
    # Calculate XYZ @ r=1 given Theta and Psi (IE convention)
    eq('{X [R]} = 1*sin({Theta [deg]}*pi/180)*cos({Psi [deg]}*pi/180)',
                                                               zones=[spzone])
    eq('{Y [R]} = 1*sin({Theta [deg]}*pi/180)*sin({Psi [deg]}*pi/180)',
                                                               zones=[spzone])
    eq('{Z [R]} = 1*cos({Theta [deg]}*pi/180)', zones=[spzone])

def blank_and_trace(zone,hemi,**kwargs):
    """Function blanks out the closed field points and traces the rest to
    verify the connectivity
    Inputs
    Returns
    """
    eq = tp.data.operate.execute_equation
    # Blank according to hemisphere
    blank = tp.active_frame().plot().value_blanking
    blank.active = True
    blank.cell_mode = ValueBlankCellMode.PrimaryValue
    # Blank all Z_sm for one hemisphere
    zblank = blank.constraint(1)
    zblank.variable = zone.dataset.variable('Z *')
    if hemi=='North':
        zblank.comparison_operator = RelOp.LessThan
    elif hemi=='South':
        zblank.comparison_operator = RelOp.GreaterThan
    zblank.comparison_value = 0
    zblank.active = kwargs.get('blankZ',True)
    # Blank Status==3
    closedblank = blank.constraint(2)
    closedblank.variable = zone.dataset.variable('Status')
    closedblank.comparison_operator = RelOp.EqualTo
    closedblank.comparison_value = 3
    closedblank.active=True
    blank.cell_mode = ValueBlankCellMode.AllCorners
    # Extract the blanked region
    [capZone] = tp.data.extract.extract_blanked_zones(zone)
    eq('{originalStatus} = {Status}',zones=[zone,capZone])
    zblank.active=False
    closedblank.active=False
    blank.active = False
    # Trace remaining points and update Status as necessary
    points = list(zip(capZone.values('X *').as_numpy_array(),
                      capZone.values('Y *').as_numpy_array(),
                      capZone.values('Z *').as_numpy_array()))
    zone.dataset.delete_zones([capZone])
    trace_status(points, zone, hemi)

def blank_and_interpolate(source,target,hemi):
    eq = tp.data.operate.execute_equation
    # Blank section of the sphere
    blank = tp.active_frame().plot().value_blanking
    blank.active = True
    blank.cell_mode = ValueBlankCellMode.PrimaryValue
    # Blank all Z_sm for one hemisphere
    zblank = blank.constraint(1)
    zblank.variable = source.dataset.variable('Z *')
    if hemi=='North':
        zblank.comparison_operator = RelOp.LessThan
    elif hemi=='South':
        zblank.comparison_operator = RelOp.GreaterThan
    zblank.comparison_value = 0
    zblank.active=True
    # Interpolate from source to target
    #from IPython import embed; embed()
    #tp.data.operate.interpolate_inverse_distance(
    tp.data.operate.interpolate_kriging(
                                target,
                                zero_value=0.05,
                                source_zones=[source],
                                variables=[source.dataset.variable('Status')])
    # Boost the Status signal post interpolation for firm discrete states
    if hemi=='North':
        #eq('{Status} = if({Status}<2.9,2,3)',zones=[target])
        pass
    elif hemi=='South':
        #eq('{Status} = if({Status}<2.9,1,3)',zones=[target])
        pass
    eq('{Status_old} = {Status}',zones=[target])
    eq('{Status_cc}={Status}',value_location=ValueLocation.CellCentered,
                                   zones=[target])
    eq('{Status_old_cc}={Status}',value_location=ValueLocation.CellCentered,
                                  zones=[target])
    # Turn off blanking
    blank.active = False
    zblank.active = False

def match_variable_names(zones):
    """Function ports over the variable names from IE so they match GM
    Inputs
        zones
    Returns
        None
    """
    eq = tp.data.operate.execute_equation
    eq('{Rho [amu/cm^3]} = 56',zones=zones)
    if 'Ux [km/s]' in zones[0].dataset.variable_names:
        eq('{U_x [km/s]} = {Ux [km/s]}',zones=zones)
        eq('{U_y [km/s]} = {Uy [km/s]}',zones=zones)
        eq('{U_z [km/s]} = {Uz [km/s]}',zones=zones)
    if 'Bdx' in zones[0].dataset.variable_names:
        eq('{B_x [nT]} = {Bdx}',zones=zones)
        eq('{B_y [nT]} = {Bdy}',zones=zones)
        eq('{B_z [nT]} = {Bdz}',zones=zones)
    eq('{P [nPa]} = 0.5',zones=zones)
    eq('{J_x [uA/m^2]} = {Jx [`mA/m^2]}',zones=zones)
    eq('{J_y [uA/m^2]} = {Jy [`mA/m^2]}',zones=zones)
    eq('{J_z [uA/m^2]} = {Jz [`mA/m^2]}',zones=zones)
    eq('{theta_1 [deg]} = -{Theta [deg]}+90',zones=zones)
    eq('{theta_2 [deg]} = {Theta [deg]}-90',zones=zones)
    eq('{phi_1 [deg]} = {Psi [deg]}',zones=zones)
    eq('{phi_2 [deg]} = {Psi [deg]}',zones=zones)
    if 'RT 1/B [1/T]' in zones[0].dataset.variable_names:
        eq('{Status} = if({RT 1/B [1/T]}<-1,2,3)',zones=zones[0])
        eq('{Status} = if({RT 1/B [1/T]}<-1,1,3)',zones=zones[1])

def check_edges(zone,hemi,**kwargs):
    """
    """
    eq, CC = tp.data.operate.execute_equation, ValueLocation.CellCentered
    stat_var = kwargs.get('stat_var','Status_cc')
    status = zone.values(stat_var).as_numpy_array()
    if hemi=='North':
        check_indices = np.where((status>2.01)&(status<2.99))[0]
    elif hemi=='South':
        check_indices = np.where((status>1.01)&(status<2.99))[0]
    if 'cc' in stat_var:
        eq('{x_cc2}={X [R]}',value_location=CC,zones=[zone])
        eq('{y_cc2}={Y [R]}',value_location=CC,zones=[zone])
        eq('{z_cc2}={Z [R]}',value_location=CC,zones=[zone])
        x = zone.values('x_cc2').as_numpy_array()[check_indices]
        y = zone.values('y_cc2').as_numpy_array()[check_indices]
        z = zone.values('z_cc2').as_numpy_array()[check_indices]
    else:
        x = zone.values('X *').as_numpy_array()[check_indices]
        y = zone.values('Y *').as_numpy_array()[check_indices]
        z = zone.values('Z *').as_numpy_array()[check_indices]
    points = list(zip(x,y,z))
    trace_status(points,zone,hemi,stat_var=kwargs.get('stat_var','Status_cc'))
    # Boost the Status signal post interpolation for firm discrete states
    if hemi=='North':
        eq('{'+stat_var+'old2} = {'+stat_var+'}',zones=[zone])
        eq('{'+stat_var+'} = if({'+stat_var+'}<2.9,2,3)',zones=[zone])
    elif hemi=='South':
        eq('{'+stat_var+'_old2} = {'+stat_var+'}',zones=[zone])
        eq('{'+stat_var+'} = if({'+stat_var+'}<2.9,1,3)',zones=[zone])

def get_ionosphere(dataset,**kwargs):
    """Function routes to various extraction and analysis options for IE data
        based on kwargs given
    Inputs
        dataset (tecplot dataset)
        kwargs:
            hasGM (bool) - default False
    Returns
    """
    zoneNorth = dataset.zone(kwargs.get('ieZoneHead','ionosphere_')+'north')
    zoneSouth = dataset.zone(kwargs.get('ieZoneHead','ionosphere_')+'south')
    data_to_write={}
    if kwargs.get('hasGM',False) and kwargs.get('mergeGM',True):
        zoneGM = dataset.zone(kwargs.get('zoneGM','global_field'))
        zoneSphere = dataset.zone(kwargs.get('zoneSphere','perfectsphere*'))
        aux = zoneGM.aux_data
        if 'eventtime' in kwargs:
            eventtime = kwargs.get('eventtime')
        # TODO check that XYZ are given in IE data
        # Map sphere onto Z aligned IE data
        blank_and_trace(zoneSphere,'North')
        blank_and_trace(zoneSphere,'South')
        sphere2cart_map(zoneSphere)
        blank_and_interpolate(zoneSphere,zoneNorth,'North')
        blank_and_interpolate(zoneSphere,zoneSouth,'South')
        # Rotate IE_xyz(SM) to GM_xyz(GSM)
        rotate_xyz([zoneNorth,zoneSouth,zoneSphere],float(aux['BTHETATILT']))
        match_variable_names([zoneNorth,zoneSouth])
        #rotate_xyz([zoneNorth,zoneSphere],float(aux['BTHETATILT']))
        #match_variable_names([zoneNorth])
        check_edges(zoneNorth,'North')
        check_edges(zoneSouth,'South')
        #TODO check how this edge check compares to simply calling all >1% in
        #TODO Visually investigate the pinch point in north and island in south
        #       Then, pick out some worst cases from the first cut video
        #       Run this new method for those cases and do a side by side
        get_global_variables(dataset,'',aux=aux)
    else:
        if 'eventtime' in kwargs:
            eventtime = kwargs.get('eventtime')
        match_variable_names([zoneNorth,zoneSouth])
        get_global_variables(dataset,'',only_dipole=True,
                             aux={'BTHETATILT':'0','GAMMA':'1.6666667'})
    #Analyze data
    if kwargs.get('integrate_line',False):
        # Create a dipole terminator zone
        north_term,_ = calc_terminator_zone('terminator',zoneNorth,
                                                hemi='North',
                                                #stat_var='Status_cc',
                                                sp_rmax=1,ionosphere=True)
        _,south_term = calc_terminator_zone('terminator',zoneSouth,
                                                hemi='South',
                                                #stat_var='Status_cc',
                                                sp_rmax=1,ionosphere=True)
    if kwargs.get('integrate_surface',False):
        for zone in [zoneNorth,zoneSouth]:
            #integrate power on created surface
            print('\nWorking on: '+zone.name+' surface')
            surf_results = surface_tools.surface_analysis(zone,surfGeom=True,
                                                          **kwargs)
            surf_results['Time [UTC]'] = eventtime
            data_to_write.update({zone.name+'_surface':surf_results})
        data_to_write = surface_tools.post_proc(data_to_write,
                         do_interfacing=kwargs.get('do_interfacing',False))
    if kwargs.get('integrate_line',False):
        for zone in [north_term,south_term]:
            #integrate fluxes across the 1D curve or line
            print('\nWorking on: '+zone.name+' line')
            line_results = line_tools.line_analysis(zone,**kwargs)
            line_results['Time [UTC]'] = eventtime
            data_to_write.update({zone.name:line_results})
    if kwargs.get('write_data',True):
        datestring = ('{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(
                                            eventtime.year,eventtime.month,
                                            eventtime.day,eventtime.hour,
                                            eventtime.minute,eventtime.second))
        write_to_hdf(kwargs.get('outputpath')+'/energeticsdata/IE/ie_output_'+
                        datestring+'.h5', data_to_write)
    if kwargs.get('disp_result',True):
        display_progress('NoMesh',
                    kwargs.get('outputpath')+'/energeticsdata/IE/ie_output_'+
                         datestring+'.h5',
                         data_to_write.keys())
    return data_to_write


# Must list .plt that script is applied for proper execution
# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()
    #datapath = ('/Users/ngpdl/Code/swmf-energetics/localdbug/ie/')
    datapath = ('/nfs/solsticedisk/tuija/ccmc/2019-05-13/run/IE/ionosphere/')
    filelist = sorted(glob.glob(datapath+'it*.idl'), key=time_sort)
    for file in filelist[0:1]:
        #get timestamp
        timestamp = get_time(file)
        #read data
        north,south,aux = read_idl_ascii(file)
        #Integrate
        #save data
        #save_tofile(file,timestamp,
        #            nJouleHeat_W=[njoul],sJouleHeat_W=[sjoul],
        #            nEFlux_W=[nenergy],sEFlux_W=[senergy])
