In this tutorial we will be usind the paraview python scripts to:  
    -> fix tecplot .tec binary output so it will load in paraview  
    -> load a file  
    -> call the setup_pipeline core function from pv_magnetopause.py

You will need
=============
-> Working installation of paraview with built in python  
-> Global_energetics package visible to python within paraview
-> 3D output file fixed for paraview (https://drive.google.com/drive/folders/1E0Aw7sI5Wwop4E0e-nZ_wk8RdgLZHdy3?usp=sharing)

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
from global_energetics.extract.pv_magnetosphere import setup_pipeline, generate_surfaces
from global_energetics.extract.pv_visuals import display_visuals
```
4.Call setup_pipeline function
------------------------------
There are lots of options available, see help(setup_pipeline) for more details
```
pipeline_dict = setup_pipeline("fixed_3Dfile.plt")
field = pipeline_dict['field']

surfaces_dict = generate_surfaces(field)
mp = surfaces_dict['mp']
```
5.Render and set visuals
------------------------
```
renderView1 = GetActiveViewOrCreate('RenderView')
SetActiveView(renderView1)
display_visuals(field,mp,renderView1,
                mpContourBy=('CELLS','B_nT','X'),
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
from global_energetics.extract.pv_magnetosphere import setup_pipeline, generate_surfaces
from global_energetics.extract.pv_visuals import display_visuals
 
infile = 'some/path/fixed_3Dfile.plt'

pipeline_dict = setup_pipeline(infile)
field = pipeline_dict['field']

surfaces_dict = generate_surfaces(field)
mp = surfaces_dict['mp']

renderView1 = GetActiveViewOrCreate('RenderView')
SetActiveView(renderView1)
display_visuals(field,mp,renderView1,
                mpContourBy=('CELLS','B_nT','X'),
                contourMin=-5,
                contourMax=5)
layout = GetLayout()
layout.SetSize(1280, 720)
SaveScreenshot('./example_image.png',layout,
            ImageResolution=[1280,720])
```
1.Call pvbatch mode with your script
------------------------------------
*In the directory magnetosphere-energetics/* call the script using pvbatch
```
pvbatch myscript.py
```
In this way you can also call any of the scripts found in
magnetosphere-energetics/runscripts/

*NOTE* may need to use os and sys packages to ensure that paraview can
find the custom python packages, or follow instructions in modify_paraview.md
to soft link them in yourself.

```
import os,sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
```

