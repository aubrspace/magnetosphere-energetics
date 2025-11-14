import paraview
import os
import time
import glob
import numpy as np
import datetime as dt
#### import the simple module from paraview
from paraview.simple import *
### Interpackage
from global_energetics.extract import equations
from global_energetics.extract import pv_tools
from global_energetics.extract import pv_input_tools
from global_energetics.extract import pv_surface_tools
from global_energetics.extract import pv_volume_tools
from global_energetics.extract import pv_tabular_tools
from global_energetics.extract import pv_visuals
from global_energetics.extract import pv_fte
from global_energetics.extract import pv_mapping

def merge_times(sourcelist:list[object],*,
               sourcenames:list=['past','present','future'],
                                               **kwargs:dict) -> object:
    """ Merges multiple sources together
    """
    merged = ProgrammableFilter(registrationName='merged',Input=sourcelist)
    merged.Script = update_merge(**kwargs)
    return merged

def update_merge(*,
        keeplist:list=['Rho_amu_cm3','U_km_s','B_nT','P_nPa','J_uA_m2',
                       'Status','dvol_R3','r_R',
                       'phi_1_deg','phi_2_deg','theta_1_deg','theta_2_deg',
                       'Utot_J_Re3','uB_J_Re3','uHydro_J_Re3',
                       'KE_J_Re3','Pth_J_Re3','uB_dipole_J_Re3','u_db_J_Re3',
                       'mp','closed','lobes','inner','sheath'],
                                                        **kwargs:dict) -> str:
    return f"""
past    = inputs[0]
present = inputs[1]
future  = inputs[2]

# pass XYZ to the merge
output.PointData.append(present.PointData['x'],'x')
output.PointData.append(present.PointData['y'],'y')
output.PointData.append(present.PointData['z'],'z')

# tag variables and pass only some things to the output
keep_list = {keeplist}
for variable in keep_list:
    if (variable in past.PointData.keys() and
        variable in present.PointData.keys() and
        variable in future.PointData.keys()):
        output.PointData.append(past.PointData[variable],'PAST'+variable)
        output.PointData.append(present.PointData[variable],variable)
        output.PointData.append(future.PointData[variable],'FUTURE'+variable)
    """

def generate_volumes(pipeline:object,**kwargs:dict) -> object:
    """ Calls volume_tools extract_volume function to generate 3D sub-volumes
    """
    volume_dict = {}
    for surface in kwargs.get('surfaces',['mp']):
        volume_kwargs = {}
        variable = surface
        volume_name = surface+'Vol'
        skipVolume = False
        if 'inner' in surface:
            skipVolume = True
        if not skipVolume:
            volume_dict[surface] = pv_volume_tools.extract_volume(pipeline,
                                                                  variable,
                                                              volume_name,
                                                              **volume_kwargs)
    return volume_dict

def generate_surfaces(pipeline:object,**kwargs:dict) -> object:
    """ Calls surface_tools iso_surface function to generate 2D surfaces in 3D
    Inputs
        pipeline (filter/source)- upstream should include state variables!!!
        kwargs:
            inner_r (float) - default 3.0, used to set the inner contour
    Returns
        surfaces_dict (dict{str:filter})- dict of filters generated 1 per surf
    """
    surfaces_dict = {}
    for surface in kwargs.get('surfaces',['mp']):
        iso_kwargs = {}
        variable = surface
        surface_name = surface
        if 'lobes' in surface:
            # Dont trim, since we expect 2 separate non-touching lobes
            iso_kwargs = {'trim_regions':False}
        if 'inner' in surface:
            assert 'r_R' in pipeline.PointData, (
                                  'GENERATESURFACE inner: No radius variable!')
            iso_kwargs = {'iso_value':kwargs.get('inner_r',3.0)}
            variable = 'r_R'
        surfaces_dict[surface] = pv_surface_tools.create_iso_surface(pipeline,
                                                                     variable,
                                                                 surface_name,
                                                                  **iso_kwargs)
    return surfaces_dict

def load_state_variables(pipeline:object,**kwargs:dict) -> object:
    script = ''
    for surface in kwargs.get('surfaces',['mp']):
        if 'mp' in surface:
            state = pv_surface_tools.create_magnetopause_state(pipeline,
                                               **{k:kwargs[k] for k in
                                                  ['verbose_pipeline',
                                                   'betastar_max','inner_r',
                                                   'status_closed','tail_x']
                                                  if k in kwargs})
        if 'closed' in surface:
            state = pv_surface_tools.create_closed_state(pipeline,
                                             **{k:kwargs[k] for k in
                                                ['verbose_pipeline','inner_r',
                                                 'status_closed','tail_x']
                                                 if k in kwargs})
        if 'lobes' in surface:
            state = pv_surface_tools.create_lobes_state(pipeline,
                                               **{k:kwargs[k] for k in
                                                  ['verbose_pipeline',
                                                   'betastar_max','inner_r',
                                                   'status_open','tail_x']
                                                  if k in kwargs})
        if 'inner' in surface:
            #NOTE for this one the definition is simple so the logic is diff
            pass
            #state = pv_surface_tools.create_inner_state(pipeline,
            #                                   **{k:kwargs[k] for k in
            #                                      ['verbose_pipeline',
            #                                       'inner_r']
            #                                      if k in kwargs})
        if 'sheath' in surface:
            state = pv_surface_tools.create_sheath_state(pipeline,
                                               **{k:kwargs[k] for k in
                                                  ['verbose_pipeline',
                                                   'betastar_max',
                                                   'status_closed','tail_x',
                                                   'x0','y0','z0','s_ratio']
                                                  if k in kwargs})
        if kwargs.get('verbose_pipeline',False):
            pipeline = state
        else:
            script += state

    if kwargs.get('verbose_pipeline',False):
        return pipeline
    else:
        return script

def setup_pipeline(infile:str,**kwargs:dict):
    """Function takes single data file and builds pipeline to find and
        visualize magnetopause
    Inputs
        infile (str)- full path to tecplot binary (.plt) BATSRUS output
        kwargs:
            aux (dict)- optional to include dictionary with other data
            get_gradP (bool)- default false, will add additional variable
    Returns
        source (pvpython object)- python object attached to the VTKobject
                                  for the input data
        pipelinehead (pypython filter)- top level filter which starts the
                                        pipeline processing
        field (source/filter)- where the dataset has finished creating new
                               variables
        mp (source/filter)- final version of magnetopause
    """
    print(f"INITIALIZING PIPELINE FROM {infile.split('/')[-1]} ...")
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    # Read input file
    sourcedata = pv_input_tools.read_tecplot(infile)

    # apply 'Merge Blocks' so 'Connectivity' can be used
    mergeBlocks1 = MergeBlocks(
                        registrationName=infile.split('/')[-1].split('.')[0],
                               Input=sourcedata)

    ##Set the head of the pipeline, we will want to return this!
    pipelinehead = mergeBlocks1
    pipeline = mergeBlocks1

    ### Coordinate conversions
    if 'convert' in kwargs:
        if kwargs.get('convert')=='eci':
            pipeline = pv_tools.gsm_to_eci(pipeline,kwargs.get('ut',0))
        elif kwargs.get('convert')=='gsm':
            #NOTE for now assuming gse
            pipeline = pv_tools.gse_to_gsm(pipeline,kwargs.get('ut',0))
    #if kwargs.get('repair_status',False): NOTE depreciated!!
    #    pipeline = pv_input_tools.status_repair(pipeline,**kwargs)
    ########################################################################
    # Gather a few prog filter scripts together to save memory ...
    #
    script = 'output_staging = {}'
    ###Check if unitless variables are present
    if kwargs.get('dimensionless',False):
        if kwargs.get('verbose_pipeline',False):
            pipeline = pv_input_tools.todimensional(pipeline,**kwargs)
        else:
            script += pv_input_tools.todimensional(pipeline,**kwargs)
    else:
        ###Rename some tricky variables
        if kwargs.get('verbos_pipeline',False):
            pipeline = pv_input_tools.fix_names(pipeline,**kwargs)
        else:
            script +=  pv_input_tools.fix_names(pipeline,**kwargs)
    ###Build functions up to betastar
    alleq = equations.equations(**kwargs)
    evaluation_set,evaluation_save = {},{}
    if not kwargs.get('verbose_pipeline',False):
        doSave = False
    evaluation_set = pv_tools.eq_add(alleq['basic3d'],evaluation_set,
                                     evaluation_save,doSave=True)#NOTE
    evaluation_set = pv_tools.eq_add(alleq['basic_physics'],evaluation_set,
                                     evaluation_save,doSave=doSave)
    if 'aux' in kwargs:
        evaluation_set = pv_tools.eq_add(alleq['dipole_coord'],evaluation_set,
                                         evaluation_save,doSave=doSave)
        evaluation_set = pv_tools.eq_add(alleq['dipole'],evaluation_set,
                                         evaluation_save,doSave=doSave)
    if kwargs.get('doEntropy',False):
        evaluation_set = pv_tools.eq_add(alleq['entropy'],evaluation_set,
                                         evaluation_save,doSave=doSave)
    # Energy flux variables
    if kwargs.get('doEnergyFlux',False):
        evaluation_set = pv_tools.eq_add(alleq['energy_flux'],evaluation_set,
                                         evaluation_save,doSave=doSave)
    # Volume energy variables
    if kwargs.get('doVolumeEnergy',False):
        assert 'aux' in kwargs, "No AUX data, cannont calculate Volume Energy"
        evaluation_set = pv_tools.eq_add(alleq['volume_energy'],evaluation_set,
                                         evaluation_save,doSave=True)#NOTE
    if kwargs.get('verbose_pipeline',False):
        pipeline = pv_tools.all_evaluate(evaluation_set,evaluation_save,
                                         pipeline,verbose_pipeline=True)
    else:
        script += pv_tools.all_evaluate(evaluation_set,evaluation_save,
                                        pipeline,verbose_pipeline=False,
                                        vectorize_variables=True)
    ###Fix tracing
    if (pipeline.PointData['Status'].GetRange()[0] == -3 and
        'theta_1_deg' in pipeline.PointData.keys()):
        if kwargs.get('verbose_pipeline',False):
            pipeline = pv_input_tools.fix_status(pipeline,**kwargs)
        else:
            script += pv_input_tools.fix_status(pipeline,**kwargs)
    ###Daynight mapping
    if kwargs.get('do_daynight',False):
        #TODO finish this:
        #       Add shared_tools & pv_mapping to the paraview build
        #       Test
        #       double check the preconditions (what "state_var" is needed)
        #eq('{trace_limits}=if({Status}==3 && '+
        #                    '{r [R]}>'+str(kwargs.get('inner_r',3)-1)+',1,0)')
        if kwargs.get('verbose_pipeline',False):
            pipeline = pv_mapping.reversed_mapping(pipeline,**kwargs)
        else:
            script += pv_mapping.reversed_mapping(pipeline,**kwargs)
    ###Get Vectors from field variable components
    if kwargs.get('doVectors',False):
        if kwargs.get('verbose_pipeline',False):
            pipeline = pv_tools.get_vectors(pipeline,**kwargs)
        else:
            print("TODO- implement vectors w/out verbose!!!")
            #script += pv_tools.get_vectors(pipeline,**kwargs)

    ###Programmable filters
    # Pressure gradient, optional variable
    if kwargs.get('doGradP',False):
        if kwargs.get('verbose_pipeline',False):
            pipeline = pv_tools.get_pressure_gradient(pipeline)
        else:
            script += pv_tools.get_pressure_gradient(pipeline,**kwargs)

    ###Regional state variables
    if kwargs.get('verbose_pipeline',False):
        pipeline = load_state_variables(pipeline,**kwargs)
    else:
        script += load_state_variables(pipeline,**kwargs)

    ###Write out the mega-script
    if not kwargs.get('verbose_pipeline',False):
        pipeline = ProgrammableFilter(registrationName='equations+',
                                      Input=pipeline)
        pipeline.Script = script+"""
for key,arr in output_staging.items():
    output.PointData.append(arr,key)
        """
    #
    # End prog filter things
    ########################################################################
    ###Generate surfaces using the contour filter
    #surfaces_dict = generate_surfaces(pipeline,**kwargs)

    ###Generate volumes as threshold filters
    #volumes_dict = generate_volumes(pipeline,**kwargs)

    # Stand-alone, larger filters that are less default
    if kwargs.get('ffj',False):
        ffj1 = pv_tools.get_ffj_filter1(pipeline)
        ffj2 = PointDatatoCellData(registrationName='ffj_interp1',
                                   Input=ffj1)
        ffj2.ProcessAllArrays = 1
        ffj3 = pv_tools.get_ffj_filter2(ffj2)
        #pipeline=ffj3
        ffj4 = CellDatatoPointData(registrationName='ffj_interp2',
                                   Input=ffj3)
        pipeline = ffj4

    if kwargs.get('doFTE',False):
        # FTE
        pipeline = pv_fte.load_fte(pipeline)


    ###Read satellite trajectories
    if kwargs.get('doSat',False):
        satfiles = kwargs.get('satfiles')
        for satin in satfiles:
            name = satin.split('.csv')[0]
            csv = CSVReader(registrationName=name+'_in',
                            FileName=kwargs.get('path')+satin)
            points = TableToPoints(registrationName=name,
                                   Input=csv)
            points.XColumn = 'x'
            points.YColumn = 'y'
            points.ZColumn = 'z'
            renderView = GetActiveViewOrCreate('RenderView')
            pointsDisplay=Show(points,renderView,'GeometryRepresentation')
            colors = {
                    'cl1':[0.9,0.9,0.9],
                    'cl2':[0.9,0.9,0.9],
                    'cl3':[0.9,0.9,0.9],
                    'cl4':[0.9,0.9,0.9],
                    'thA':[0.9,0.9,0.9],
                    'thB':[0.9,0.9,0.9],
                    'thC':[0.9,0.9,0.9],
                    'thD':[0.9,0.9,0.9],
                    'thE':[0.9,0.9,0.9],
                    'geo':[0.9,0.9,0.9],
                    'mms1':[0.9,0.9,0.9],
                    'mms2':[0.9,0.9,0.9],
                    'mms3':[0.9,0.9,0.9],
                    'mms4':[0.9,0.9,0.9]
                    }
    ###Now that last variable is done set 'field' for visualizer and a View
    field = pipeline

    if kwargs.get('fte',False):
        ###Contour (iso-surface) of the magnetopause
        fte = pv_surface_tools.create_iso_surface(field, 'fte', 'fte')

    ###Field line seeding or Field line projected flux volumes
    fluxResults = None
    if(kwargs.get('doFieldlines',False)or kwargs.get('doFluxVol',False)):
        from magnetometer import read_station_paraview
        print(kwargs.get('station_file'))
        station_MAG, success = read_station_paraview(
                                    kwargs.get('localtime'),
                                    n=kwargs.get('n',379),
                        file_in=kwargs.get('station_file','stations.csv'),
                                    path=kwargs.get('path'))
        if success and 'localtime' in kwargs and 'tilt' in kwargs:
            stations = pv_tools.magPoints2Gsm(station_MAG,
                                              kwargs.get('localtime'),
                                              kwargs.get('tilt'))
            renderView = GetActiveViewOrCreate('RenderView')
            stationsDisplay = Show(stations, renderView,
                                   'GeometryRepresentation')
            stationsDisplay.AmbientColor = [1.0, 1.0, 0.0]
            stationsDisplay.DiffuseColor = [1.0, 1.0, 0.0]
            #Blank inside the earth
            clip1 = Clip(registrationName='Clip1', Input=pipeline)
            clip1.ClipType = 'Sphere'
            clip1.Invert = 0
            clip1.ClipType.Center = [0.0, 0.0, 0.0]
            clip1.ClipType.Radius = 0.99
            '''
            #Blank outside the magnetosphere (as in 0.7 beta*)
            clip2 = Clip(registrationName='Clip2', Input=clip1)
            clip2.ClipType = 'Scalar'
            clip2.Scalars = ['POINTS', 'mp_state']
            clip2.Value = 1
            clip2.Invert = 0
            #clip2=clip1
            '''
            if kwargs.get('doFieldlines',False):
                pv_visuals.add_fieldlines(clip1)
            if kwargs.get('doFluxVol',False):
                clip1.ClipType.Radius = 2.5
                obj, fluxResults = pv_volume_tools.add_fluxVolume(clip2,
                                                                  **kwargs)
    ### Collect returnable items
    return {'source':sourcedata,'head':pipelinehead,'field':field}

if __name__ == "__main__":
    import os
    import time
    import glob
    import numpy as np
#if True:
    #from pv_input_tools import time_sort, read_aux, read_tecplot
    start_time = time.time()
    ######################################################################
    # USER INPUTS
    ######################################################################
    #path='/Users/ngpdl/Code/swmf-energetics/localdbug/fte/'
    #path='/home/aubr/Code/swmf-energetics/localdbug/fte/'
    path='/nfs/solsticedisk/tuija/amr_fte/secondtry/GM/IO2/'
    outpath = 'output7_fte_pv/'
    #outpath = '/Users/ngpdl/Code/swmf-energetics/localdbug/fte/output5_fte_pv/'
    ######################################################################

    #Make the paths if they don't already exist
    os.makedirs(path, exist_ok=True)
    #os.makedirs(outpath, exist_ok=True)

    filelist = sorted(glob.glob(path+'*paraview*.plt'),
                      key=time_sort)
    #filelist = ['/home/aubr/Code/swmf-energetics/localdbug/fte/3d__paraview_1_e20140610-010000-000.plt']
    #filelist = ['/Users/ngpdl/Code/swmf-energetics/localdbug/febstorm/3d__paraview_1_e20140219-024500-000.plt']
    for infile in filelist[-1::]:
        print('processing '+infile.split('/')[-1]+'...')
        aux = read_aux(infile.replace('.plt','.aux'))
        oldsource,pipelinehead,field,mp=setup_pipeline(infile,aux=aux,
                                                       doEnergy=False)
        ###Surface flux on magnetopause
        get_surface_flux(mp, 'B_nT','Bnormal_net')
        mp_Bnorm = FindSource('Bnormal_net')
        #decide which values to calculate (will need to make cell data)
        #fluxes = [('K_W_Re2','k_flux'),('P0_W_Re2','h_flux'),
        #          ('ExB_W_Re2','p_flux')]
        #mp_cc = point2cell(mp,fluxes)#mp object with cell centered data
        #mp_K_flux = get_surface_flux(mp, 'K_W_Re2','k_flux')
        #mp_S_flux = get_surface_flux(mp_cc, 'ExB_W_Re2','s_net_flux')
        renderView1 = GetActiveViewOrCreate('RenderView')
        #TODO find how to limit integration variables and group all together
        #tableLayout, tableView = setup_table()
        #save_table_data(mp_S_flux, tableView, outpath,'s_net_flux')
        SetActiveView(renderView1)
        pv_visuals.display_visuals(field,mp,renderView1,
                        mpContourBy='B_x_nT',contourMin=-10,contourMax=10,
                        **kwargs)

        # Create a new 'Render View'
        layout = GetLayout()
        layout.SplitVertical(0, 0.5)
        renderView2 = CreateView('RenderView')
        # assign view to a particular cell in the layout
        AssignViewToLayout(view=renderView2, layout=layout, hint=2)
        display_visuals(field,mp_Bnorm,renderView2,doSlice=True,
                        mpContourBy='Bnormal_net',
                        contourMin=-10,contourMax=10,
                        cmap='Cool to Warm')

        # Render and save screenshot
        RenderAllViews()
        # layout/tab size in pixels
        layout.SetSize(2162, 1079)
        SaveScreenshot(outpath+
                       infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                       SaveAllViews=1,ImageResolution=[2162,1079])
    for infile in filelist[0:-1]:
        print('processing '+infile.split('/')[-1]+'...')
        outfile=outpath+infile.split('/')[-1].split('.plt')[0]+'.png'
        if os.path.exists(outfile):
            print(outfile+' already exists, skipping')
        else:
            #Read in new file unattached to current pipeline
            SetActiveSource(None)
            newsource = read_tecplot(infile)

            #Attach pipeline to the new source file and delete the old
            pipelinehead.Input = newsource
            Delete(oldsource)

            # Render and save screenshot
            RenderAllViews()
            # layout/tab size in pixels
            layout.SetSize(2162, 1079)
            SaveScreenshot(outpath+
                        infile.split('/')[-1].split('.plt')[0]+'.png',layout,
                        SaveAllViews=1,ImageResolution=[2162,1079])

            # Set the current source to be replaced on next loop
            oldsource = newsource
    #timestamp
    ltime = time.time()-start_time
    print('DONE')
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
