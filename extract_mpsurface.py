#!/usr/bin/env python3
"""SWMF Energetics with Tecplot
"""
import logging as log
import os
import sys
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *

log.basicConfig(level=log.INFO)

def create_stream_zone(r_equator, phi_start):
	"""Function to create a streamline, created in 2 directions from starting point
	Inputs
		r_equator [R]- starting position for streamline
		phi_start [rad]
	"""
	# Get starting position in cartesian coordinates
	[x_start,y_start,z_start] = sph_to_cart(r_equator, pi/2, phi_start)
	# Create streamline
	tp.active_frame().plot().show_streamtraces=True
	field_line = tp.active_frame().plot().streamtraces
	field_line.add(seed_point=[x_start,y_start,z_start],
				   stream_type=Streamtrace.VolumeLine,
				   direction=StreamDir.Both)
	# Create zone
	stream_zone = field_line.extract()
	# Delete streamlines
	field_line.delete_all()


def check_streamline_closed(zone_name, r_eq, r_cap):
	"""Function to check if a streamline is open or closed
	Inputs
		zone_name- name of the zone used to identify zone
		r_eq [R]- equitorial radial position used to seed field line
		r_cap [R]- radius of cap that determines if line is closed
	"""
	# Get starting and endpoints of streamzone
	r_values = swmf_data.zone(zone_name).values('r *').as_numpy_array()
	r_north, r_south = r_values[0], r_values[-1]
	#check if closed
	print('r north, r south ', ' ', r_north, r_south)
	if (r_north>r_eq) or (r_south>r_eq):
		isclosed = False
	else:
		isclosed = True
	return isclosed, max(r_north,r_cap), max(r_south,r_cap)



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
	phi_lin = np.linspace(0,2*pi,int(num_elements))
	theta_lin = np.linspace(-0.5*theta_total, 0.5*theta_total,int(num_elements))
	theta, phi = np.meshgrid(theta_lin, phi_lin, indexing='ij')
	cap_x, cap_y, cap_z = sph_to_cart(cap_distance, theta, phi)
	with tp.session.suspend():
		#Create northern polar cap
		log.info('creating polar cap zones')
		north_cap_data = tp.active_frame().create_dataset('NorthCap',['x','y','z'])
		north_cap_zone = north_cap_data.add_ordered_zone('North Cap',[num_elements,num_elements])
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
	plot = tp.active_frame().plot()


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

# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

if __name__ == "__main__":
	if '-c' in sys.argv:
		tp.session.connect()
	
	tp.new_layout()

	#Load .plt file, come back to this later for batching
	log.info('loading .plt and reformatting')
	swmf_data = tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')
	swmf_data.zone(0).name = 'global_field'
	print(swmf_data)

	#Create R from cartesian coordinates
	tp.data.operate.execute_equation('{r [R]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')

	#Set the parameters for streamline seeding 
	n_azimuth = 60
	phi = np.linspace(0,2*pi,n_azimuth)

	#set B as the vector field
	plot = tp.active_frame().plot()
	plot.vector.u_variable = swmf_data.variable('B_x*')
	plot.vector.v_variable = swmf_data.variable('B_y*')
	plot.vector.w_variable = swmf_data.variable('B_z*')

	create_stream_zone(3.5,0)
	swmf_data.zone('Stream*').name = 'min_field_line'
	min_closed, r_north, r_south = check_streamline_closed('min_field_line',3.5,3.5)	
	print(min_closed)
	create_stream_zone(10,0)
	swmf_data.zone('Stream*').name = 'max_field_line'
	min_closed, r_north, r_south = check_streamline_closed('max_field_line',10,3.5)	
	print(min_closed)
	#for i in range(n_azimuth-1)
		#create min
		#check min
		#create max
		#check max
		#if((max_open && min_open) || (!max_open && !min_open))
			#printerror
		#else
			#while(notfound && iter<iter_max)
				#create mid
				#check mid
				#if(mid_open)
					#r_eq_max = r_eq_mid
				#else
					#r_eq_min = r_eq_mid
				#if(abs(r_eq_min - r_eq_max) < threashold)
					#notfound = FALSE
				#iter = iter+1

		#createAndCheck(max)
		

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

