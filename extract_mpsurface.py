#!/usr/bin/env python3
"""SWMF Energetics with Tecplot
"""
import logging as log
import os
import sys
import numpy as np
from numpy import abs, pi, cos, sin, sqrt
from array import array
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
	[x_start,y_start,z_start] = sph_to_cart(r_start, theta_start, phi_start)
	# Create streamline
	tp.active_frame().plot().show_streamtraces=True
	field_line = tp.active_frame().plot().streamtraces
	field_line.add(seed_point=[x_start,y_start,z_start],
				   stream_type=Streamtrace.VolumeLine,
				   direction=StreamDir.Both)
	# Create zone
	stream_zone = field_line.extract()
	swmf_data.zone('Stream*').name = zone_name + '{}'.format(phi_start)
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
		max(r_end_n,r_cap)- furthest out point at pole, for making smooth surface on the caps
		max(r_end_s,r_cap)
	"""
	# Get starting and endpoints of streamzone
	r_values = swmf_data.zone(zone_name+'*').values('r *').as_numpy_array()
	r_end_n, r_end_s = r_values[0], r_values[-1]
	#check if closed
	if (r_end_n>r_eq) or (r_end_s>r_eq):
		isclosed = False
	else:
		isclosed = True
	return isclosed, max(r_end_n,r_cap), max(r_end_s,r_cap)



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

def find_tail_disk_point(rho, psi, x):
	"""Function finds the spherical coordinates of a point on a disk at a constant x position in the tail
	Inputs
		rho- radial position relative to the center of the disk
		psi- angle relative to the axis pointing out from the center of the disk
		x- x position of the disk
	Outputs
		[r, theta, phi]- spherical coordinates of the point relative to the global origin
	"""
	y = rho*sin(psi)
	z = rho*cos(psi)
	r = sqrt(x**2+rho**2)
	theta = pi/2 - np.arctan(z/abs(x))
	phi = pi + np.arctan(y/abs(x))
	return [r,theta,phi]

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
	#DaySide
	n_azimuth_day = 3
	azimuth_range = [np.deg2rad(-122), np.deg2rad(122)] #need to come back to dynamically determine where to switch modes
	phi = np.linspace(azimuth_range[0],azimuth_range[1],n_azimuth_day)
	R_MAX = 30
	R_MIN = 3.5

	#Tail
	n_azimuth_tail = 5
	psi = np.linspace(-pi*(1-pi/n_azimuth_tail),pi,n_azimuth_tail)
	RHO_MAX = 50
	RHO_STEP = 0.5
	X_TAIL_CAP = -30

	#Other
	R_CAP = 3.5
	itr_max = 100
	tol = 0.1

	#Initialize objects that will be modified in creation loop
	r_eq_mid = np.zeros(n_azimuth_day)
	r_north = np.zeros(n_azimuth_day+n_azimuth_tail)
	r_south = np.zeros(n_azimuth_day+n_azimuth_tail)
	itr = 0
	r_eq_max, r_eq_min = R_MAX, R_MIN

	#set B as the vector field
	plot = tp.active_frame().plot()
	plot.vector.u_variable = swmf_data.variable('B_x*')
	plot.vector.v_variable = swmf_data.variable('B_y*')
	plot.vector.w_variable = swmf_data.variable('B_z*')


#	create_stream_zone(3.5,0,'min_field_line')
#	min_closed, _, _ = check_streamline_closed('min_field_line',3.5,3.5)	
#	print(min_closed)
#	create_stream_zone(10,0,'max_field_line')
#	max_closed, r_north[0], r_south[0] = check_streamline_closed('max_field_line',10,3.5)	
#	print(max_closed)

	#Create Dayside Magnetopause field lines
	with tp.session.suspend():
		for i in range(n_azimuth_day):
			#Create initial max min and mid field lines
			log.info('Creating dayside magnetopause boundary')
			create_stream_zone(R_MIN,pi/2,phi[i],'min_field_line')
			create_stream_zone(R_MAX,pi/2,phi[i],'max_field_line')
			#Check that last closed is bounded
			min_closed, _, __ = check_streamline_closed('min_field_line',R_MIN,R_CAP)
			max_closed, _, __ = check_streamline_closed('max_field_line',R_MAX,R_CAP)
			swmf_data.delete_zones(swmf_data.zone('min_field*'),swmf_data.zone('max_field*'))
			print('phi: {:.1f}, iterations: {}, error: {}'.format(np.rad2deg(phi[i]),itr,r_eq_max-r_eq_min))
			if max_closed and min_closed:
				print('WARNING: field line closed at max of {}R_e'.format(R_MAX))
				create_stream_zone(R_MAX,pi/2,phi[i],'field_phi_')
			elif not max_closed and not min_closed:
				print('WARNING: first field line open at {}R_e'.format(R_MIN))
				create_stream_zone(R_MIN,pi/2,phi[i],'field_phi_')
			else:
				#if i is 0:
				#	r_eq_mid[i] = (R_MAX+R_MIN)/2
				#else: #inherit last mid r position for faster convergence
				#	r_eq_mid[i] = r_eq_mid[i-1]
				r_eq_mid[i] = (R_MAX+R_MIN)/2
				itr = 0
				notfound = True
				r_eq_min, r_eq_max = R_MIN, R_MAX
				while(notfound and itr<itr_max):
					#This is a bisection root finding algorithm with initial guess at the previous phi solution
					create_stream_zone(r_eq_mid[i],pi/2,phi[i],'temp_field_phi_')
					mid_closed, r_north[i], r_south[i] = check_streamline_closed('temp_field_phi_',r_eq_mid[i],R_CAP)
					if mid_closed:
						r_eq_min = r_eq_mid[i]
					else:
						r_eq_max = r_eq_mid[i]
					if abs(r_eq_min - r_eq_max) < tol and mid_closed:
						notfound = False
						swmf_data.zone('temp_field_phi_*').name = 'field_phi_{:.1f}'.format(np.rad2deg(phi[i]))
					else:
						r_eq_mid[i] = (r_eq_max+r_eq_min)/2
						swmf_data.delete_zones(swmf_data.zone('temp_field*'))
					itr += 1
			
		rho_tail = RHO_MAX
		#Create Tail Magnetopause field lines
		for i in range(n_azimuth_tail):
			log.info('Creating tail magnetopause boundary')
			r_tail, theta_tail, phi_tail = find_tail_disk_point(RHO_MAX,psi[i],X_TAIL_CAP)
			print('r: {}, theta: {:.1f}, phi: {:.1f}'.format(r_tail,np.rad2deg(theta_tail),np.rad2deg(phi_tail)))
			create_stream_zone(r_tail,theta_tail,phi_tail,'temp_tail_line_')
			#check if closed
			tail_closed, r_north[i+n_azimuth_day], r_south[i+n_azimuth_day] = check_streamline_closed('temp_tail_line_',RHO_MAX,R_CAP)
			print('psi: {:.1f}, rho: {:.2f}'.format(np.rad2deg(psi[i]),rho_tail))
			if tail_closed:
				print('WARNING: field line closed at RHO_MAX={}R_e'.format(RHO_MAX))
				swmf_data.zone('temp_tail_line*').name = 'tail_field_{:.1f}'.format(np.rad2deg(psi[i]))
			else:
				#This is a basic marching algorithm from outside in starting at RHO_MAX
				rho_tail = RHO_MAX
				notfound = True
				while notfound and rho_tail>RHO_STEP:
					swmf_data.delete_zones(swmf_data.zone('temp_tail_line*'))
					rho_tail = rho_tail - RHO_STEP
					r_tail, theta_tail, phi_tail = find_tail_disk_point(rho_tail,psi[i],X_TAIL_CAP)
					create_stream_zone(r_tail,theta_tail,phi_tail,'temp_tail_line_')
					tail_closed, r_north[i+n_azimuth_day], r_south[i+n_azimuth_day] = check_streamline_closed('temp_tail_line_',rho_tail,R_CAP)
					if tail_closed:
						swmf_data.zone('temp_tail_line*').name = 'tail_field_{:.1f}'.format(np.rad2deg(psi[i]))
						notfound = False
					if rho_tail <= RHO_STEP:
						print('WARNING: placement not possible at psi={:.1f}'.format(np.rad2deg(psi[i])))
						
		#Create magnetopause surface by stitching together created zones
		#Get shape of the total mp zone
		n_points, zone_step = 0,[]
		for i in range(1,swmf_data.num_zones):
			zone_step.append(len(swmf_data.zone(i).values('X *')[::100]))
			n_points += zone_step[-1]
		mpdatashape = [n_points,n_points,1]
		print(mpdatashape, zone_step)
		#"Ordered Zone" automatically determines connectivity based on i,k,j ordered points
		mp_zone = swmf_data.add_ordered_zone('MagnetoPause', mpdatashape)
		print('created mp zone with dimension: {}'.format(mp_zone.dimensions))
		#Fill the created zone by iterating over all zones
		fill_start = 0
		for i in range(1,swmf_data.num_zones):
			mp_zone.values('X *')[fill_start:(fill_start+zone_step[i-1])] = swmf_data.zone(i).values('X *')[::100]
			mp_zone.values('Y *')[fill_start:(fill_start+zone_step[i-1])] = swmf_data.zone(i).values('Y *')[::100]
			mp_zone.values('Z *')[fill_start:(fill_start+zone_step[i-1])] = swmf_data.zone(i).values('Z *')[::100]
			fill_start += zone_step[i-1]
			#mp_zone = swmf_data.add_ordered_zone('MagnetoPause',shape)
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

