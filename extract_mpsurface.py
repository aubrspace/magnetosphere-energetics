#Import necessary libraries
import logging as log
import os
import sys
import numpy as np
from np import abs, pi, cos, sin

log.basicConfig(level=log.INFO)

import tecplot as tp
from tp.constant import *

# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

import sys
if '-c' in sys.argv:
    tp.session.connect()

#Load .plt file, come back to this later for batching
tp.data.load_tecplot('3d__mhd_2_e20140219-123000-000.plt')

#Create R from cartesian coordinates
tp.data.operate.execute_equation('{r [R]} = sqrt(V1**2 + V2**2 + V3**2)')

    def createStreamline(r,theta,phi)
    # Function to create a streamline, created in 2 directions from starting point
    # Inputs
    #    r [Re]- starting position for streamline
    #    theta [rad]
    #    phi [rad]
        # Get starting position in cartesian coordinates
        x_start= (r * sin(theta) * cos(phi))
	y_start= (r * sin(theta) * sin(phi))
	z_start= (r * cos(theta))
        # Create the streamlines 
        streamline = plot.streamtraces
        streamline.add([x_start,y_start,z_start],
            stream_type = Streamtrace.VolumeLine,
            direction=StreamDir.Both)
        # Create zone from streamline
        stream_zone = plot.streamtraces.extract()
        # Delete streamlines
        delete_all()

		# -------------------------------------------------------------
                # Function to check if a streamzone is open or closed
                # Inputs |1| -> Streamzone number / ID
                #        |2| -> rPoint


# ============================================================================
# Create the dayside magnetopause zone



# ===================================================================================================
# Interpolate field data and calculate normal energy flux on magnetopause zone

# Interpolate data

# Create MP surface normal vector

# Calculate energy flux for all zones
    # Electric field (no resistivity)

    # Poynting flux

    # Total Energy flux

# Calculate orthogonal projection of energy flux through magnetopause
    # Component normal flux

    # Signed magnitude normal flux
