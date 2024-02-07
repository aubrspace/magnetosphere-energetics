In this tutorial we will be usind the paraview python scripts to:  
    -> fix tecplot .tec binary output so it will load in paraview  
    -> load a file  
    -> call the setup_pipeline core function from pv_magnetopause.py

You will need
=============
-> Working installation of paraview with built in python  
-> Linked global_energetics package to paraview  
-> 3D output file fixed for paraview (see util/pltfixer.py)

Option 1- interactive
=====================
1.Open python-paraview in interactive mode
------------------------------------------
*In the directory swmf-energetics* launch pvpython
```
    pvpython
```
2.Load in the paraview simple module
------------------------------------
```
    from paraview.simple import *
```
3.Load in magnetopause and display functions
--------------------------------------------
```
    from pv_magnetopause import setup_pipeline
    from pv_visuals import display_visuals
```
4.Call setup_pipeline function
------------------------------
There are lots of options available, see help(setup_pipeline) for more details
```
    oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline('fixed_3Dfile.plt')
```
5.Render and set visuals
------------------------
```
    renderView1 = GetActiveViewOrCreate('RenderView')
    SetActiveView(renderView1)
    display_visuals(field,mp,renderView1,
                    mpContourBy='B_x_nT',
                     contourMin=-5,
                     contourMax=5)
```
6.Save a screenshot
-------------------
```
    layout = GetLayout()
    layout.SetSize(1280, 720)
    SaveScreenshot('./example_image.png',layout,
                    ImageResolution=[1280,720])
```

Option 2- via script
====================
similar to before copy this now into "myscript.py"
```
    from paraview.simple import *
    from pv_magnetopause import setup_pipeline
    from pv_visuals import display_visuals
    
    infile = 'some/path/fixed_3Dfile.plt'
    oldsource,pipelinehead,field,mp,fluxResults=setup_pipeline(infile)
    renderView1 = GetActiveViewOrCreate('RenderView')
    SetActiveView(renderView1)
    display_visuals(field,mp,renderView1,
                    mpContourBy='B_x_nT',
                    contourMin=-5,
                    contourMax=5)
    layout = GetLayout()
    layout.SetSize(1280, 720)
    SaveScreenshot('./example_image.png',layout,
                ImageResolution=[1280,720])
```
1.Call pvbatch mode with your script
------------------------------------
*In the directory swmf-energetics/* call the script using pvbatch
```
    pvbatch myscript.py
```
In this way you can also call any of the scripts found in
swmf-energetics/runscripts/
