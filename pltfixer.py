#/usr/bin/env python
import sys
import glob
import tecplot as tp

if __name__ == '__main__':
    header = 'files_for_paraview/'
    if '-c' in sys.argv:
        tp.session.connect()
    for infile in glob.glob('febstorm/*.plt')[0:1]:
        outfile = 'paraview'.join(infile.split('/')[-1].split('var'))
        print('fixing '+outfile+'....')
        ds = tp.data.load_tecplot(infile)
        ds.variable('X *').name = 'x'
        ds.variable('Y *').name = 'y'
        ds.variable('Z *').name = 'z'
        aux = ds.zone(0).aux_data.as_dict()
        with open(header+outfile.split('.plt')[0]+'.aux','w') as f:
            for key,value in aux.items():
                f.write('%s:%s\n' % (key,value))
        ds.zone(0).aux_data.clear()
        tp.data.save_tecplot_plt(header+outfile,dataset=ds)
        tp.new_layout()
    print('DONE')
