import os,sys
import time
import glob
import numpy as np
import datetime as dt
#### paraview
import paraview
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
#### Custom packages #####
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import global_energetics
from global_energetics.makevideo import time_sort, get_time
from global_energetics.extract.pv_magnetopause import setup_pipeline
from global_energetics.extract.pv_input_tools import read_tecplot

global FILTER
FILTER = paraview.vtk.vtkAlgorithm # Generic "filter" object

def get_magnetosheath(Input:FILTER,**kwargs:dict) -> FILTER:
    magnetosheath = ProgrammableFilter(registrationName='magnetosheath',
                                       Input=Input)
    magnetosheath.Script = update_magnetosheath(**kwargs)
    return magnetosheath

def update_magnetosheath(**kwargs:dict) -> str:
    return f"""
    import numpy as np
    x0 = {kwargs.get('x0',15)}
    y0 = {kwargs.get('y0',0)}
    z0 = {kwargs.get('z0',0)}
    threshold = {kwargs.get('threshold',2.4)}

    # Pull out point data to play with
    points = inputs[0].PointData
    mp = points['mp_state']
    Status = points['Status']
    x = points['x']
    y = points['y']
    z = points['z']
    s = points['s']

    # Get upstream solar wind entropy
    s0 = s[(abs(x-x0)<1) & (abs(y-y0)<1) & (abs(z-z0)<1)].mean()

    # Define magnetosheath as: s/s0 > Const. AND not magnetosphere
    magnetosheath = (((s/s0)>threshold) & (~ mp)).astype(int)

    # Categorize regions NOTE- eventually move this out to its own filter
    category = np.zeros(len(x))
    category[Status<0] = -1
    category[mp==1] = 2
    category[magnetosheath==1] = 1

    output.ShallowCopy(inputs[0].VTKObject)
    output.PointData.append(magnetosheath,'msh')
    output.PointData.append(category,'category')
    """

def update_datacube(**kwargs:dict) -> str:
    return f"""
    import numpy as np

    datacube = dict()
    outfile = "{OUTPATH}{kwargs.get('outname','test_datacube2.npz')}"
    print(outfile)

    xBounds = {kwargs.get('xBounds',[-10,15])}
    yBounds = {kwargs.get('xBounds',[-25,25])}
    zBounds = {kwargs.get('xBounds',[-25,25])}

    # Get input
    data = inputs[0]
    x = data.PointData['x']
    y = data.PointData['y']
    z = data.PointData['z']
    inbounds = ((x>xBounds[0])&(x<xBounds[1])&
                (y>yBounds[0])&(y<yBounds[1])&
                (z>zBounds[0])&(z<zBounds[1]))
    for variable in {kwargs.get('variables',['x','y','z',
                                             'Rho_amu_cm3','P_nPa',
                                             'U_x_km_s','U_y_km_s','U_z_km_s',
                                             'B_x_nT','B_y_nT','B_z_nT',
                                          'J_x_uA_m2','J_y_uA_m2','J_z_uA_m2',
                                             'category'])}:
        datacube[variable] = np.array(data.PointData[variable][inbounds])

    np.savez_compressed(outfile,**datacube)
    print("Saved ",outfile)
    """

def save_datacube(field:FILTER,**kwargs:dict) -> None:
    # Initialize
    datacube = {}
    outfile = f"{OUTPATH}{kwargs.get('outname','test_datacube2.npz')}"

    xBounds = kwargs.get('xBounds',[-10,15])
    yBounds = kwargs.get('xBounds',[-25,25])
    zBounds = kwargs.get('xBounds',[-25,25])

    # Fetch data
    data = servermanager.Fetch(field)
    data = dsa.WrapDataObject(data)
    points = data.PointData

    # Set bounds
    x = points['x']
    y = points['y']
    z = points['z']
    inbounds = ((x>xBounds[0])&(x<xBounds[1])&
                (y>yBounds[0])&(y<yBounds[1])&
                (z>zBounds[0])&(z<zBounds[1]))

    # Construct the output dict
    for variable in ['x','y','z','Rho_amu_cm3','P_nPa',
                     'U_x_km_s','U_y_km_s','U_z_km_s',
                     'B_x_nT','B_y_nT','B_z_nT',
                     'J_x_uA_m2','J_y_uA_m2','J_z_uA_m2',
                     'category']:
        datacube[variable] = np.array(points[variable][inbounds])

    # Save
    np.savez_compressed(outfile,**datacube)
    return

def main() -> None:
    filelist = sorted(glob.glob(f"{INPATH}/*paraview*.plt"),
                      key=time_sort)
    #LoadState('localdbug/mothersday/datacube_state.pvsm')
    LoadState(os.path.join(os.getcwd(),'cosmetic/datacube_state.pvsm'))
    renderView = GetActiveView()
    print(f"Processing ({len(filelist)})...")
    for infile in filelist:
        print(f"\t{infile.split('/')[-1]}")
        localtime = get_time(infile)
        # Update time
        timestamp = FindSource('time')
        timestamp.Text = str(localtime)
        # Locate the start of the pipeline and the old data
        pipehead = FindSource('MergeBlocks1')
        oldData = pipehead.Input
        # Read in new data and feed into the pipe, delete old data
        newData = read_tecplot(infile)
        pipehead.Input = newData
        Delete(oldData)
        del oldData

        outfile = infile.split('/')[-1].split('e')[-1].replace('.plt','')
        # Save a new datacube
        save_datacube(FindSource('magnetosheath'),outname=outfile)

        # Save screenshot
        layouts = GetLayouts()
        for i,layout in enumerate(layouts.values()):
            SaveScreenshot(OUTPATH+outfile+f'_{i}.png',layout)

        '''INITIALIZATION - only need this once
        _,__,field,____,_____ = setup_pipeline(infile,doEntropy=True,
                                               tail_x=-60,
                                               path=OUTPATH)
        field = get_magnetosheath(field)
        save_datacube(field)
        #datacube = ProgrammableFilter(registrationName='datacube',Input=field)
        #datacube.Script = update_datacube()
        '''

#if __name__ == "__main__":
if True:
    start_time = time.time()

    global INPATH,OUTPATH

    #INPATH  = os.path.join(os.getcwd(),"gannon-storm/data/large/GM/IO2/")
    #OUTPATH = os.path.join(os.getcwd(),"localdbug/mothersday/")
    INPATH = os.path.join(os.getcwd(),"run_mothersday_ne/GM/IO2/")
    OUTPATH = os.path.join(os.getcwd(),"outputs_mothersday_ne/datacubes/")

    main()

    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
