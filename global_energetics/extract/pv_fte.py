from paraview.simple import *

def load_fte(pipeline,**kwargs):
    filtername = 'fte_state'
    # Usually we have Point Data so if reading Cell Data need to filter
    if kwargs.get('isCellData',True):
        xyzCell = PointDatatoCellData(registrationName='xyzCell',
                                      Input=pipeline)
        xyzCell.ProcessAllArrays = 0
        xyzCell.PointDataArraytoprocess = ['x', 'y', 'z']
        xyzCell.PassPointData = 1
        filtername = 'fteCell'
        pipeline = xyzCell

    # Progammable filter will read things in *hopefully* fast enough
    if 'file' not in kwargs:
        kwargs['file'] = '/home/aubr/Downloads/FTE020500.npz'
    fteFilter=ProgrammableFilter(registrationName=filtername,Input=pipeline)
    fteFilter.Script = update_fte(**kwargs)
    pipeline = fteFilter

    # Now change that Cell Data back to Point Data
    if kwargs.get('isCellData',True):
        ftePoint = CellDatatoPointData(registrationName='fte_state',
                                       Input=fteFilter)
        ftePoint.ProcessAllArrays = 0
        ftePoint.CellDataArraytoprocess = ['fte']
        ftePoint.PassCellData = 0
        pipeline = ftePoint

    return pipeline

def update_fte(**kwargs):
    return"""
        data = inputs[0]
        #Read .npz file
        fte = numpy.load('"""+kwargs.get('file')+"""')

        #Reconfigure XYZ data
        X = data.CellData['x']
        Y = data.CellData['y']
        Z = data.CellData['z']

        #Create new variable
        fte_indata = numpy.zeros(len(X))

        source_arr = numpy.zeros((len(X),3))
        source_arr[:,0] = X
        source_arr[:,1] = Y
        source_arr[:,2] = Z
        #Isolate just the points within the bounds of the fte_extent
        town = numpy.where((X<=fte['ftex'].max())&
                           (X>=fte['ftex'].min())&
                           (Y<=fte['ftey'].max())&
                           (Y>=fte['ftey'].min())&
                           (Z<=fte['ftez'].max())&
                           (Z>=fte['ftez'].min()))

        village = town[0][::]
        #Change all the matching points to have value=1
        Xunique = numpy.unique(X[village])
        Yunique = numpy.unique(Y[village])
        for Xi in Xunique:
            if Xi in fte['ftex']:
                for Yi in Yunique:
                    if Yi in fte['ftey']:
                        fte_sub=fte['ftez'][numpy.where((fte['ftex']==Xi)&
                                                       (fte['ftey']==Yi))[0]]
                        if len(fte_sub)>0:
                            extrusion = numpy.where((X[village]==Xi)&
                                                    (Y[village]==Yi))[0]
                            Z_extrusion = Z[village[extrusion]]
                            street = numpy.where((Z_extrusion<fte_sub.max())&
                                                 (Z_extrusion>fte_sub.min()))
                            for address in street[0]:
                                if Z_extrusion[address] in fte_sub:
                                    fte_indata[village[extrusion[address]]]=1

        #Copy input to output
        output.ShallowCopy(inputs[0].VTKObject)

        #Add our new array
        output.CellData.append(fte_indata,'fte')
    """
