#!/usr/bin/env python3
"""SWMF Energetics with Tecplot
"""
import logging as log
import os
import sys
from array import array
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *

log.basicConfig(level=log.INFO)

def create_stream_zone(r_start, theta_start, phi_start, zone_name):
    """Function to create a streamline, created in 2 directions from starting point
    Inputs
        r_start [R]- starting position for streamline
        theta_start [rad]
        phi_start [rad]
        zone_name
    """
    # Get starting position in cartesian coordinates
    [x_start, y_start, z_start] = sph_to_cart(r_start, theta_start, phi_start)
    # Create streamline
    tp.active_frame().plot().show_streamtraces = True
    field_line = tp.active_frame().plot().streamtraces
    field_line.add(seed_point=[x_start, y_start, z_start],
                   stream_type=Streamtrace.VolumeLine,
                   direction=StreamDir.Both)
    # Create zone
    field_line.extract()
    SWMF_DATA.zone(-1).name = zone_name + '{}'.format(phi_start)
    # Delete streamlines
    field_line.delete_all()


def check_streamline_closed(zone_name, r_eq, r_cap):
    """Function to check if a streamline is open or closed
    Inputs
        zone_name
        r_eq [R]- equitorial radial position used to seed field line
        r_cap [R]- radius of cap that determines if line is closed
    Outputs
        isclosed- boolean, True for closed
        max(r_end_n, r_cap)- furthest out point at pole, for making smooth surface on the caps
        max(r_end_s, r_cap)
    """
    # Get starting and endpoints of streamzone
    r_values = SWMF_DATA.zone(zone_name+'*').values('r *').as_numpy_array()
    r_end_n, r_end_s = r_values[0], r_values[-1]
    #check if closed
    if (r_end_n > r_eq) or (r_end_s > r_eq):
        isclosed = False
    else:
        isclosed = True
    return isclosed, max(r_end_n, r_cap), max(r_end_s, r_cap)


def create_polar_caps(cap_distance, cap_diameter, cap_area_resolution):
    """Function to create 3D polar cap in the magnetopause
    Inputs
        cap_distance [Re]- radial distance to center of the cap
        cap_diameter [Re]- curvilinear diameter of the cap
    """
    # Obtain latitude bounds and number of elements given conditions
    theta_total = cap_diameter/abs(cap_distance)
    num_elements = np.ceil(abs(cap_distance)*sqrt(theta_total*2*pi/cap_area_resolution))
    #create cap data point mesh
    phi_lin = np.linspace(0, 2*pi, int(num_elements))
    theta_lin = np.linspace(-0.5*theta_total, 0.5*theta_total, int(num_elements))
    theta, phi_cap = np.meshgrid(theta_lin, phi_lin, indexing='ij')
    cap_x, cap_y, cap_z = sph_to_cart(cap_distance, theta, phi_cap)
    with tp.session.suspend():
        #Create northern polar cap
        log.info('creating polar cap zones')
        north_cap_data = tp.active_frame().create_dataset('NorthCap', ['x', 'y', 'z'])
        north_cap_zone = north_cap_data.add_ordered_zone('North Cap', [num_elements, num_elements])
        north_cap_zone.values('x')[:] = cap_x.ravel()
        north_cap_zone.values('y')[:] = cap_y.ravel()
        north_cap_zone.values('z')[:] = cap_z.ravel()
        #Mirror to southern polar cap
        tp.macro.execute_command('''$!CreateMirrorZones
                                SourceZones =  {'North Cap'}
                                MirrorVars =  [3]''')
        tp.execute_command("""$!RenameDataSetZone
                            Zone = {'Mirror: North Cap'}
                            Name = 'South Cap'""")

def sph_to_cart(radius, theta, phi):
    """Function converts spherical coordinates to cartesian coordinates
    Inputs
        radius- radial position
        theta
        phi
    Outputs
        [x_pos, y_pos, z_pos]- list of x y z_pos coordinates
    """
    x_pos = (radius * sin(theta) * cos(phi))
    y_pos = (radius * sin(theta) * sin(phi))
    z_pos = (radius * cos(theta))
    return [x_pos, y_pos, z_pos]

def find_tail_disk_point(rho, psi_disc, x_pos):
    """Function find spherical coordinates of a point on a disk at a constant x position in the tail
    Inputs
        rho- radial position relative to the center of the disk
        psi_disc- angle relative to the axis pointing out from the center of the disk
        x_pos- x position of the disk
    Outputs
        [radius, theta, phi_disc]- spherical coordinates of the point relative to the global origin
    """
    y_pos = rho*sin(psi_disc)
    z_pos = rho*cos(psi_disc)
    radius = sqrt(x_pos**2+rho**2)
    theta = pi/2 - np.arctan(z_pos/abs(x_pos))
    phi_disc = pi + np.arctan(y_pos/abs(x_pos))
    return [radius, theta, phi_disc]


def calc_dayside_mp(phi, r_max, r_min):
    """"Function to create zones that will makeup dayside magnetopause
    Inputs
        phi- set of phi angle points
        r_max- maxium radial distance for equitorial search
        r_min- min
    Outputs
        r_north- set of cuttoff points for northern hemisphere
        r_south- southern hemisphere
    """
    #Initialize objects that will be modified in creation loop
    r_eq_mid = np.zeros(int(len(phi)))
    r_north = np.zeros(int(len(phi)))
    r_south = np.zeros(int(len(phi)))
    itr = 0
    r_eq_max, r_eq_min = r_max, r_min

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = SWMF_DATA.variable('B_x*')
    plot.vector.v_variable = SWMF_DATA.variable('B_y*')
    plot.vector.w_variable = SWMF_DATA.variable('B_z*')


    #Create Dayside Magnetopause field lines
    for i in range(int(len(phi))):
        #Create initial max min and mid field lines
        create_stream_zone(r_min, pi/2, phi[i], 'min_field_line')
        create_stream_zone(r_max, pi/2, phi[i], 'max_field_line')
        #Check that last closed is bounded
        min_closed, _, __ = check_streamline_closed('min_field_line', r_min, R_CAP)
        max_closed, _, __ = check_streamline_closed('max_field_line', r_max, R_CAP)
        SWMF_DATA.delete_zones(SWMF_DATA.zone('min_field*'), SWMF_DATA.zone('max_field*'))
        print('phi: {:.1f}, iters: {}, err: {}'.format(np.rad2deg(phi[i]), itr, r_eq_max-r_eq_min))
        if max_closed and min_closed:
            print('WARNING: field line closed at max of {}R_e'.format(r_max))
            create_stream_zone(r_max, pi/2, phi[i], 'field_phi_')
        elif not max_closed and not min_closed:
            print('WARNING: first field line open at {}R_e'.format(r_min))
            create_stream_zone(r_min, pi/2, phi[i], 'field_phi_')
        else:
            #if i is 0:
            #   r_eq_mid[i] = (R_MAX+R_MIN)/2
            #else: #inherit last mid r position for faster convergence
            #   r_eq_mid[i] = r_eq_mid[i-1]
            r_eq_mid[i] = (r_max+r_min)/2
            itr = 0
            notfound = True
            r_eq_min, r_eq_max = r_min, r_max
            while(notfound and itr < ITR_MAX):
                #This is a bisection root finding algorithm with init guess at previous phi solution
                create_stream_zone(r_eq_mid[i], pi/2, phi[i], 'temp_field_phi_')
                mid_closed, r_north[i], r_south[i] = check_streamline_closed('temp_field_phi_',
                                                                             r_eq_mid[i], R_CAP)
                if mid_closed:
                    r_eq_min = r_eq_mid[i]
                else:
                    r_eq_max = r_eq_mid[i]
                if abs(r_eq_min - r_eq_max) < TOL and mid_closed:
                    notfound = False
                    SWMF_DATA.zone('temp*').name = 'field_phi_{:.1f}'.format(np.rad2deg(phi[i]))
                else:
                    r_eq_mid[i] = (r_eq_max+r_eq_min)/2
                    SWMF_DATA.delete_zones(SWMF_DATA.zone('temp_field*'))
                itr += 1
    return r_north, r_south




def calc_tail_mp(psi, x_disc, rho_max, rho_step):
    """Function to create the zones that will become the tail magnetopause
    Inputs
        psi- set of disc azimuthal angles
        x_disc- x position of the tail disc
        rho_max- outer radial bounds of the tail disc
        rho_step- radial distance increment for marching algorithm
    """
    #Initialize objects that will be modified in creation loop
    rho_tail = rho_max
    r_north = np.zeros(int(len(psi)))
    r_south = np.zeros(int(len(psi)))

    #set B as the vector field
    plot = tp.active_frame().plot()
    plot.vector.u_variable = SWMF_DATA.variable('B_x*')
    plot.vector.v_variable = SWMF_DATA.variable('B_y*')
    plot.vector.w_variable = SWMF_DATA.variable('B_z*')

    #Create Tail Magnetopause field lines
    for i in range(int(len(psi))):
        r_tail, theta_tail, phi_tail = find_tail_disk_point(rho_max, psi[i], x_disc)
        print('r: {}, theta: {:.1f}, phi: {:.1f}'.format(r_tail, np.rad2deg(theta_tail),
                                                         np.rad2deg(phi_tail)))
        create_stream_zone(r_tail, theta_tail, phi_tail, 'temp_tail_line_')
        #check if closed
        tail_closed, r_north[i], r_south[i] = check_streamline_closed('temp_tail_line_', rho_max,
                                                                      R_CAP)
        print('psi: {:.1f}, rho: {:.2f}'.format(np.rad2deg(psi[i]), rho_tail))
        if tail_closed:
            print('WARNING: field line closed at RHO_MAX={}R_e'.format(rho_max))
            SWMF_DATA.zone('temp_tail_line*').name = 'tail_field_{:.1f}'.format(np.rad2deg(psi[i]))
        else:
            #This is a basic marching algorithm from outside in starting at RHO_MAX
            rho_tail = rho_max
            notfound = True
            while notfound and rho_tail > rho_step:
                SWMF_DATA.delete_zones(SWMF_DATA.zone('temp_tail_line*'))
                rho_tail = rho_tail - rho_step
                r_tail, theta_tail, phi_tail = find_tail_disk_point(rho_tail, psi[i], x_disc)
                create_stream_zone(r_tail, theta_tail, phi_tail, 'temp_tail_line_')
                tail_closed, r_north[i], r_south[i] = check_streamline_closed('temp_tail_line_',
                                                                              rho_tail, R_CAP)
                if tail_closed:
                    SWMF_DATA.zone('temp*').name = 'tail_field_{:.1f}'.format(np.rad2deg(psi[i]))
                    notfound = False
                if rho_tail <= rho_step:
                    print('WARNING: placemnt not possible at psi={:.1f}'.format(np.rad2deg(psi[i])))



def stitch_zones(zone_name, spar):
    """Function that creates single ordered zone out of streamzones
    Inputs
        zone_name
        spar- only 1 per sparse number of points loaded into zone
    """
    #Create magnetopause surface by stitching together all but 1st zone
    #Get shape of the total mp zone
    zone_step, mpdatashape = [], [0, 0, 1]
    for i in range(1, SWMF_DATA.num_zones):
        mpdatashape[0] = max(mpdatashape[0], SWMF_DATA.zone(i).dimensions[0])
        mpdatashape[1] += SWMF_DATA.zone(i).dimensions[1]
        zone_step.append(SWMF_DATA.zone(i).dimensions[0])
    #"Ordered Zone" automatically determines connectivity based on i, k, j ordered points
    mp_zone = SWMF_DATA.add_ordered_zone(zone_name, mpdatashape)
    print('created mp zone with dimension: {}'.format(mp_zone.dimensions))
    #Fill the created zone by iterating over all zones excpt zone 0 and last zone
    for i in range(1, SWMF_DATA.num_zones-1):
        start = mpdatashape[0]*(i-1)
        end = start + len(SWMF_DATA.zone(i).values('X *')[:])
        mp_zone.values('X *')[start:end] = SWMF_DATA.zone(i).values('X *')[:]
        mp_zone.values('Y *')[start:end] = SWMF_DATA.zone(i).values('Y *')[:]
        mp_zone.values('Z *')[start:end] = SWMF_DATA.zone(i).values('Z *')[:]







# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
    if '-c' in sys.argv:
        tp.session.connect()

    tp.new_layout()

    #Load .plt file, come back to this later for batching
    log.info('loading .plt and reformatting')
    SWMF_DATA = tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')
    SWMF_DATA.zone(0).name = 'global_field'
    print(SWMF_DATA)

    #Create R from cartesian coordinates
    tp.data.operate.execute_equation('{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')

    #Set the parameters for streamline seeding
    #DaySide
    N_AZIMUTH_DAY = 20
    AZIMUTH_RANGE = [np.deg2rad(-122), np.deg2rad(122)] #need to come back
    PHI = np.linspace(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1], N_AZIMUTH_DAY)
    R_MAX = 30
    R_MIN = 3.5

    #Tail
    N_AZIMUTH_TAIL = 20
    PSI = np.linspace(-pi*(1-pi/N_AZIMUTH_TAIL), pi, N_AZIMUTH_TAIL)
    RHO_MAX = 50
    RHO_STEP = 0.5
    X_TAIL_CAP = -30

    #Other
    R_CAP = 3.5
    ITR_MAX = 100
    TOL = 0.1
    SPARSENESS = 100

    #Create Dayside Magnetopause field lines
    with tp.session.suspend():
        #Create Dayside Magnetopause field lines
        calc_dayside_mp(PHI, R_MAX, R_MIN)

        #Create Tail magnetopause field lines
        calc_tail_mp(PSI, X_TAIL_CAP, RHO_MAX, RHO_STEP)

        #Stitch into single ordered zone
        stitch_zones('MagnetoPause', SPARSENESS)

        #Interpolate field data and calculate normal energy flux on magnetopause zone

        #Interpolate data

#
#        -------------------------------------------------------------
#                 Function to check if a streamzone is open or closed
#                 Inputs |1| -> Streamzone number / ID
#                        |2| -> rPoint
#
#
# ============================================================================
# Create the dayside magnetopause zone
#
#
#
# ==================================================================================================
# Interpolate field data and calculate normal energy flux on magnetopause zone
#
# Interpolate data
#
# Create MP surface normal vector
#
# Calculate energy flux for all zones
#     Electric field (no resistivity)
#
#     Poynting flux
#
#     Total Energy flux
#
# Calculate orthogonal projection of energy flux through magnetopause
#     Component normal flux
#
#     Signed magnitude normal flux
