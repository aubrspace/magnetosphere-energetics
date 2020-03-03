#!/usr/bin/env python3
"""SWMF Energetics with Tecplot
"""
import logging as log
import os
import sys
import numpy as np
from numpy import abs, pi, cos, sin
import tecplot as tp
from tecplot.constant import *

log.basicConfig(level=log.INFO)

# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
	if '-c' in sys.argv:
		tp.session.connect()

	#Load .plt file, come back to this later for batching
	tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')

	#Create R from cartesian coordinates
	tp.data.operate.execute_equation('{r [R]} = sqrt(V1**2 + V2**2 + V3**2)')

	#Create polar cap zones
	north_cap_x, north_cap_y, north_cap_z = create_polar_cap(3.5,0.6,0.01)
	south_cap_x, south_cap_y, south_cap_z = create_polar_cap(-3.5,0.6,0.01)


	with tp.session.suspend():
		log.info('creating polar cap zones')
		north_cap_data = tp.active_frame().create_dataset('Polar Caps',['X [Re]','Y [Re]', 'Z [Re]'])
		north_cap_zone = north_cap_data.add_ordered_zone()
		north_cap_zone.values('X [Re]')[:] = north_cap_x.ravel()
		north_cap_zone.values('Y [Re]')[:] = north_cap_y.ravel()
		north_cap_zone.values('Z [Re]')[:] = north_cap_z.ravel()


def create_stream_zone(r_equator, theta_start, phi_start, cap):
	"""Function to create a streamline, created in 2 directions from starting point
	Inputs
		r_equator [Re]- starting position for streamline
		theta_start [rad]
		phi_start [rad]
		cap- 3D tulip of p
	Outputs
		streamline
	"""
	# Get starting position in cartesian coordinates
	[x_start,y_start,z_start] = sph_to_cart(r_equator, theta_start, phi_start)
	# Create the streamlines 
	streamline = plot.streamtraces
	streamline.add([x_start,y_start,z_start],
				   stream_type=Streamtrace.VolumeLine,
				   direction=StreamDir.Both)
	return streamline



def check_streamline_closed(streamzone_ID, r):
	"""Function to check if a streamline is open or closed
	Inputs
		streamline
		r_equator
	"""
	# Get starting and endpoints of streamzone
	pass	



def create_polar_cap(cap_distance, cap_diameter, cap_area_resolution):
	"""Function to create 3D polar cap in the magnetopause
	Inputs
		cap_distance [Re]- radial distance to center of the cap
		cap_diameter [Re]- curvilinear diameter of the cap
	Outputs
		cap- 3D tuple of x,y,z coordinates of cap in space
	"""
	# Obtain latitude bounds and number of elements given conditions
	theta_total = cap_diameter/cap_distance
	num_elements = np.ceil(cap_distance*np.sqrt(theta_total*2*pi/cap_area_resolution))
	#create temp version that can be dynammically modified
	phi_lin = np.linspace(0,2*pi,num_elements)
	theta_lin = np.linspace(-0.5*theta_total, 0.5*theta_total,num_elements)
	theta, phi = np.meshgrid(theta_lin, phi_lin, indexing='ij')
	cap_temp_x, cap_temp_y, cap_temp_z = sph_to_cart(cap_distance, theta, phi)
	#wrap list in tuple so that tecplot can use it
	cap_x = tuple(cap_temp_x)
	cap_y = tuple(cap_temp_y)
	cap_z = tuple(cap_temp_z)
	return cap_x, cap_y, cap_z



def sph_to_cart(r, theta, phi):
	"""Function converts spherical coordinates to cartesian coordinates
	Inputs
		r- radial position
		theta
		phi
	Outputs
		[x,y,z]- list of x y z coordinates
	"""
	x = (r * sin(theta) * cos(phi))
	y = (r * sin(theta) * sin(phi))
	z = (r * cos(theta))
	return [x,y,z]

#		 -------------------------------------------------------------
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
# ===================================================================================================
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

