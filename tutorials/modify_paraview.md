A quick guide on how to modify paraview to include python modules from
global_energetics and others

Disclaimer: This is very adhoc and not garunteed to work for your specific
paraview build nor particular python packages. I have used it on

__Check your paraview version__
Using GUI interface
===================
Launch paraview and click on **Help**->**about**->**client information**

There you will see the following useful information:

**paraview version**:  eg. 5.11.1
**Python Library Path**: eg. /usr/local/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/lib/python3.9

<!--TODO add info about python version compatibility-->

Copy **Python Library Path** and navigate to that location in your terminal
```
cd /usr/local/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/lib/python3.9
```

Then **softlink** your desired custom python modules there. For example:

```
ln -s ~/swmf-energetics/global_energetics/extract/equations.py
```

<!--TODO- put a warning about which custom module should NOT be linked-->

As a reminder you can find the location of any python module you have available to your environment using the **__path__** method. For example:

```
ipython
import geopack
geopack.__path__
```

will return the path to where the module geopack.py is located.
