This is swmf-energetics, a project to explore energy transfer using SWMF
output data

Contact: Austin Brenner aubr@umich.edu University of Michigan
Created: 2020
Last edited: April 2021

__package structure__
global_energetics
=================
makevideo.py-           simple module for turning saved figures to video
preplot.py-             finds/runs python preplot (only tested for Mac)
supermag-data-          soft link only (NOT FOR EXTERNAL USE)
write_disp.py-          i/o functions, typically fr pandas.DataFrame->hdf5

analysis
----------------
analyze_energetics.py-  module for plotting both 3D and timeseries results
proc_indices.py-        module for plotting timeseries indices of "classic"
                        outputs like cpcp, epsilon, newell coupling etc
proc_spatial.py-        out of date, TBD statistical study of 3D surf geom
proc_temporal.py-       manipulations of 3D integrated qtys for timeseries

extract
-------
innermag.py-            innermagnetosphere handling, separate IM files
ionosphere.py-          ionosphere, separate IE files
plasmasheet.py-         similar to magnetopause but for plasmasheet (TBD)
magnetopause.py-        primary module for identifying and processing 3dtcp
satellites.py-          processes satellite trajectory files
shue.py-                functions for shue(97/98) emperical models
stream_tools.py-        functions for interfacing with tecplot
surface_construct.py-   outdated, was for constructing surface from
                        streamline data
surface_tools.py-       performs 3D surface integral analysis
swmf_access.py-         specific data processing for swmf files (TBD)
view_set.py-            tecplot view settings
volume_tools.py-        performs 3D volume integral analysis

