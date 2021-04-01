This is swmf-energetics, a project to explore energy transfer using SWMF
output data

Contact: Austin Brenner aubr@umich.edu University of Michigan
Created: 2020
Last edited: March 2021

__package structure__
global_energetics
=================
makevideo.py- simple module for turning saved figures to video
preplot.py- finds/runs python preplot (if exists, only tested for Mac)
supermag-data- soft link only (NOT FOR EXTERNAL USE)
write_disp.py- output functions, typically from pandas.DataFrame->hdf5

classic_analysis
----------------
proc_indices.py- module for plotting timeseries indices and "classic"
outputs like cpcp, epsilon, newell coupling etc
extract
-------
innermag.py-            innermagnetosphere handling, separate IM files
ionosphere.py-          ionosphere, separate IE files
plasmasheet.py-         similar to magnetopause but for plasmasheet
magnetopause.py-        primary module for identifying and processing 3dtcp
satellites.py-          processes satellite trajectory files
shue.py-                functions for shue(97/98) emperical models
stream_tools.py-        functions for interfacing with tecplot
surface_construct.py-   outdated, was for constructing surface from
                        streamline data
surface_tools.py-       performs 3D surface integral analysis
swmf_access.py-         specific data processing for swmf files
view_set.py-            tecplot view settings
volume_tools.py-        performs 3D volume integral analysis

mpydynamics_analysis
--------------------
proc_spatial.py-        analyzing magnetopause spatial 3D characteristics
proc_temporal.py-       analyzing spatially averaged/integrated (boiled
                        down to one number somehow) quantities over time
    
