#/usr/bin/env python
"""script for calculating integrated quantities from mp and cps
"""
import sys
import os
import time
import logging
import glob
import numpy as np
from numpy import pi
import pandas as pd
import datetime as dt
import spacepy
import tecplot as tp
import tecplot
from tecplot.constant import *
from tecplot.exception import *
#import global_energetics
from global_energetics.extract import magnetosphere
from global_energetics.extract import plasmasheet
from global_energetics.extract import satellites
from global_energetics.extract import stream_tools
from global_energetics.extract import surface_tools
from global_energetics.extract import volume_tools
from global_energetics.extract import view_set
from global_energetics.write_disp import write_to_hdf

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
    #pass in arguments
    #Nice condition
    #starlink = ('localdbug/starlink/3d__var_1_e20220203-114000-000.plt',
    #            'localdbug/starlink/3d__var_1_e20220203-115000-000.plt')
    #starlink = ('starlink/3d__var_1_e20220204-223000-000.plt',
    #            'starlink/3d__var_1_e20220204-224000-000.plt')
    #starlink = ('ccmc_2022-02-02/3d__var_1_e20220202-223000-000.plt',
    #            'ccmc_2022-02-02/3d__var_1_e20220202-224000-000.plt')
    starlink = ('ccmc_2022-02-02/3d__var_1_e20220203-051400-036.plt',
                'ccmc_2022-02-02/3d__var_1_e20220203-051500-026.plt')
    starlink2 = ('ccmc_2022-02-02/3d__var_1_e20220203-051500-026.plt',
                 'ccmc_2022-02-02/3d__var_1_e20220203-051600-024.plt')
    starlink3 = ('ccmc_2022-02-02/3d__var_1_e20220203-051600-024.plt',
                 'ccmc_2022-02-02/3d__var_1_e20220203-051700-004.plt')
    starlink4 = ('ccmc_2022-02-02/3d__var_1_e20220203-115400-000.plt',
                 'ccmc_2022-02-02/3d__var_1_e20220203-115500-004.plt')
    #Current fails
    #starlink = ('starlink/3d__var_1_e20220202-050300-000.plt',
    #            'starlink/3d__var_1_e20220202-050400-000.plt')
    #Future fails
    #starlink = ('starlink/3d__var_1_e20220202-050200-000.plt',
    #            'starlink/3d__var_1_e20220202-050300-000.plt')
    #Some other fail
    #starlink = ('ccmc_2022-02-02/3d__var_1_e20220202-061500-011.plt',
    #            'ccmc_2022-02-02/3d__var_1_e20220202-061600-036.plt')
    febstorm = ('febstorm/3d__var_1_e20140218-060300-037.plt',
                'febstorm/3d__var_1_e20140218-060400-033.plt')
    feb_asym = ('febstorm/3d__var_1_e20140219-130000-000.plt',
                'febstorm/3d__var_1_e20140219-130100-010.plt')
    trackim = ('localdbug/trackim/3d__var_1_e20140219-020000-000.plt',
               'localdbug/trackim/3d__var_1_e20140219-020100-000.plt')
    paleo=('/home/aubr/Code/paleo/3d__var_4_e20100320-030000-000_40125_kya.plt')
    ccmc  = ('output/CCMC/3d__var_1_e20130713-204700-037.plt',
             'output/CCMC/3d__var_1_e20130713-204700-037.plt')
    ccmc2  = (
            'ccmc_2019-08-30/3d__var_1_e20190830-165000-001.plt',
            'ccmc_2019-08-30/3d__var_1_e20190830-165100-032.plt')
    ccmc3  = (
            'ccmc_2019-05-13/3d__var_1_e20190513-225800-010.plt',
            'ccmc_2019-05-13/3d__var_1_e20190513-225900-036.plt')
    ccmc4  = (
            'ccmc_2019-05-13/3d__var_1_e20190514-025600-028.plt',
            'ccmc_2019-05-13/3d__var_1_e20190514-025700-023.plt')
    ccmc5 = ('ccmc_2019-05-13/3d__var_1_e20190513-175100-015.plt',
             'ccmc_2019-05-13/3d__var_1_e20190515-092200-017.plt',
             'ccmc_2019-05-13/3d__var_1_e20190515-095200-019.plt',
             'ccmc_2019-05-13/3d__var_1_e20190515-102200-022.plt')
    ccmc6  = (
            'ccmc_2019-05-13/3d__var_1_e20190514-071500-000.plt',
            'ccmc_2019-05-13/3d__var_1_e20190514-072300-017.plt',
            'ccmc_2019-05-13/3d__var_1_e20190515-004700-018.plt')


    '''
    #load from file
    tp.load_layout('/Users/ngpdl/Desktop/volume_diff_sandbox/visual_starter/blank_visuals.lay')
    field_data = tp.active_frame().dataset
    '''

    #for inputs in starlink:
    #inputs = starlink
    mp_surf = pd.DataFrame()
    mp_vol = pd.DataFrame()
    lobe_surf = pd.DataFrame()
    lobe_vol = pd.DataFrame()
    closed_surf = pd.DataFrame()
    closed_vol = pd.DataFrame()
    #for inputs in [starlink,starlink2,starlink3,starlink4]:
    #if False:
    #for inputs in [starlink4]:
    for inputs in [starlink,starlink2,starlink3]:
        tp.new_layout()
        mhddatafile = inputs[0]
        OUTPUTNAME = mhddatafile.split('e')[-1].split('.')[0]
        #python objects
        field_data = tp.data.load_tecplot(inputs)
        field_data.zone(0).name = 'global_field'
        if len(field_data.zone_names)>1:
            field_data.zone(1).name = 'future'
        main = tp.active_frame()
        main.name = 'main'

        #Perform data extraction
        with tp.session.suspend():
            #Caclulate surfaces
            _,results = magnetosphere.get_magnetosphere(field_data,
                                                        save_mesh=False,
                                    verbose=True,
                                    do_cms=False,
                                    analysis_type='energy',
                                    modes=['iso_betastar',
                                           'closed',
                                           'nlobe','slobe'],
                                    do_interfacing=True,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    integrate_line=False,
                                    outputpath='babyrun/',
                                    #surface_unevaluated_type='energy',
                                    #add_eqset=['energy_flux','volume_energy'],
                                    #customTerms={'Utot [J/Re^3]':'Utot [J]'})
                                    customTerms={'test':'TestArea [Re^2]'})
            '''
            mesh, data = magnetosphere.get_magnetosphere(field_data,
                                      write_data=True,
                                      disp_result=False,
                                      do_cms=False,
                                      analysis_type='energymassmag',
                                      modes=['sphere','terminator'],
                                      #modes=['iso_betastar','closed','nlobe','slobe','rc'],
                                      sp_rmax=2.65,
                                      do_interfacing=True,
                                      integrate_line=True,
                                      integrate_surface=True,
                                      integrate_volume=False,
                                      verbose=False,
                                      extract_flowline=False,
                                      outputpath='babyrun/')
                                      #customTerms={'test':'TestArea [Re^2]'})
                                      #analysis_type='energymassmag',
            '''
            '''
            #MODE 2 "full" magnetosphere stuff
            magnetosphere.get_magnetosphere(field_data,save_mesh=False,
                                    do_cms=True,
                                    analysis_type='energymassmag',
                                    modes=['iso_betastar','closed',
                                           'nlobe','slobe','rc'],
                                    customTerms={'test':'TestArea [Re^2]'},
                                    do_interfacing=True,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    integrate_line=False,
                                    outputpath='babyrun/')
                                    #tshift=45,
            '''
            """
            mp_surf=pd.concat([mp_surf,results['mp_iso_betastar_surface']],
                              ignore_index=True)
            mp_vol=pd.concat([mp_vol,results['mp_iso_betastar_volume']],
                             ignore_index=True)
            lobe_surf=pd.concat([lobe_surf,results['ms_lobes_surface']],
                                ignore_index=True)
            lobe_vol=pd.concat([lobe_vol,results['ms_lobes_volume']],
                                ignore_index=True)
            closed_surf=pd.concat([closed_surf,
                                   results['ms_closed_surface']],
                                ignore_index=True)
            closed_vol=pd.concat([closed_vol,results['ms_closed_volume']],
                                ignore_index=True)
            """
    for file in glob.glob('babyrun/energeticsdata/*.h5'):
        results = pd.HDFStore(file)
        mp_surf=pd.concat([mp_surf,results['mp_iso_betastar_surface']],
                              ignore_index=True)
        mp_vol=pd.concat([mp_vol,results['mp_iso_betastar_volume']],
                             ignore_index=True)
        lobe_surf=pd.concat([lobe_surf,results['ms_lobes_surface']],
                                ignore_index=True)
        lobe_vol=pd.concat([lobe_vol,results['ms_lobes_volume']],
                                ignore_index=True)
        closed_surf=pd.concat([closed_surf,
                                   results['ms_closed_surface']],
                                ignore_index=True)
        closed_vol=pd.concat([closed_vol,results['ms_closed_volume']],
                                ignore_index=True)
    if True:
        #K1,5 from mp
        K1 = mp_surf['K_netK1 [W]']
        K5 = mp_surf['K_netK5 [W]']
        #K2,3,4 from lobes
        K2al = lobe_surf['K_netK2a [W]']
        K2bl = lobe_surf['K_netK2b [W]']
        K3 = lobe_surf['K_netK3 [W]']
        K4 = lobe_surf['K_netK4 [W]']
        #K2,6,7 from closed
        K2ac = closed_surf['K_netK2a [W]']
        K2bc = closed_surf['K_netK2b [W]']
        K6 = closed_surf['K_netK6 [W]']
        K7 = closed_surf['K_netK7 [W]']

        #T1,5 from mp
        T1 = mp_vol['UtotK1 [J]']
        T5 = mp_vol['UtotK5 [J]']
        Tmp = mp_vol['Utot [J]']
        #T2,3,4 from lobes
        T2al = lobe_vol['UtotK2a [J]']
        T2bl = lobe_vol['UtotK2b [J]']
        T3 = lobe_vol['UtotK3 [J]']
        T4 = lobe_vol['UtotK4 [J]']
        Tl = lobe_vol['Utot [J]']
        #T2,6,7 from closed
        T2ac = closed_vol['UtotK2a [J]']
        T2bc = closed_vol['UtotK2b [J]']
        T6 = closed_vol['UtotK6 [J]']
        T7 = closed_vol['UtotK7 [J]']
        Tc = closed_vol['Utot [J]']

        #Adjust M terms to central difference
        #T1,5 from mp
        dTdt1 = -1*surface_tools.central_diff(T1,60)
        dTdt5 = -1*surface_tools.central_diff(T5,60)
        dTdtmp = -1*surface_tools.central_diff(Tmp,60)
        #T2,3,4 from lobes
        dTdt2al = -1*surface_tools.central_diff(T2al,60)
        dTdt2bl = -1*surface_tools.central_diff(T2bl,60)
        dTdt3 = -1*surface_tools.central_diff(T3,60)
        dTdt4 = -1*surface_tools.central_diff(T4,60)
        dTdtl = -1*surface_tools.central_diff(Tl,60)
        #T2,6,7 from closed
        dTdt2ac = -1*surface_tools.central_diff(T2ac,60)
        dTdt2bc = -1*surface_tools.central_diff(T2bc,60)
        dTdt6 = -1*surface_tools.central_diff(T6,60)
        dTdt7 = -1*surface_tools.central_diff(T7,60)
        dTdtc = -1*surface_tools.central_diff(Tc,60)

        '''
        #Combine into dEdt_sum
        #   Lobes- (KM1,2,3,4)
        #dEdt_suml = K1+K2al+K2bl+K3+K4+M1+M2al+M2bl+M3+M4
        dEdt_suml = lobe_surf['K_net [W]']+lobe_vol['Utot_net [W]']
        #   Closed- (KM5,2,6,7)
        #dEdt_sumc = K5+K2ac+K2bc+K6+K7+M5+M2ac+M2bc+M6+M7
        dEdt_sumc = closed_surf['K_net [W]']+closed_vol['Utot_net [W]']
        #   Total- (KM,1,5,3,4,6,7)
        #dEdt_sumt = K1+K5+K3+K4+K6+K7+M1+M5+M3+M4+M6+M7
        dEdt_sumt = mp_surf['K_net [W]']+mp_vol['Utot_net [W]']
        #Send Utot_lobes to cdiff
        dEdt_cdiffl=-1*surface_tools.central_diff(lobe_vol['Utot [J]'],60)
        dEdt_cdiffc=-1*surface_tools.central_diff(closed_vol['Utot [J]'],60)
        dEdt_cdifft = -1*surface_tools.central_diff(mp_vol['Utot [J]'],60)
        '''

        #Display errors
        error_2as = (K2al+K2ac)/K2ac*100
        error_2bs = (K2bl+K2bc)/K2bc*100
        error_2a = (dTdt2al+dTdt2ac)/dTdt2ac*100
        error_2b = (dTdt2bl+dTdt2bc)/dTdt2bc*100
        error_volume = (mp_vol['Volume [Re^3]']-lobe_vol['Volume [Re^3]']
                        - closed_vol['Volume [Re^3]'])
        '''
        #Display error with K_net and Utot_net
        error_dEdtl = dEdt_suml - dEdt_cdiffl
        error_dEdtc = dEdt_sumc - dEdt_cdiffc
        error_dEdtt = dEdt_sumt - dEdt_cdifft

        print('\nLobes\ndEdt_sum {:<.3}\t'.format(dEdt_suml[1])+
              'dEdt_cdiff {:<.3}\t'.format(dEdt_cdiffl[1])+
              'error {:<.3}'.format(error_dEdtl[1]))
        print('\nClosed\ndEdt_sum {:<.3}\t'.format(dEdt_sumc[1])+
              'dEdt_cdiff {:<.3}\t'.format(dEdt_cdiffc[1])+
              'error {:<.3}'.format(error_dEdtc[1]))
        print('\nTot\ndEdt_sum {:<.3}\t'.format(dEdt_sumt[1])+
              'dEdt_cdiff {:<.3}\t'.format(dEdt_cdifft[1])+
              'error {:<.3}'.format(error_dEdtt[1]))
        print('\nno issues!')
        '''
    from IPython import embed; embed()

    #with tp.session.suspend():
    if False:#manually switch on or off
        #adjust view settings
        #proc = 'Multi Frame Manager'
        #cmd = 'MAKEFRAMES3D ARRANGE=TILE SIZE=50'
        #tp.macro.execute_extended_command(command_processor_id=proc,
        #                                  command=cmd)
        mode = ['iso_day', 'other_iso', 'iso_tail', 'hood_open_north']
        zone_hidekeys = ['sphere', 'box','shue','future','innerbound',
                         'lcb','nlobe','slobe','closed','rc']
        timestamp=True
        for n, frame in enumerate(tp.frames()):
            #frame[1].activate()
            if n==0:
                legend = False
                timestamp = True
                doslice = False#
                slicelegend = False
                fieldlegend = True
                fieldline=False
            if n==1:
                legend = True
                timestamp = False
                doslice = True
                slicelegend = False
                fieldlegend = False
                fieldline=True
            if n==2:
                legend = False
                timestamp = False
                doslice = True
                slicelegend = True
                fieldlegend = False
                fieldline=False
            if n==3:
                legend = True
                save = True
                timestamp = False
                doslice = False
                slicelegend = False
                fieldlegend = False
                fieldline=True
                zone_hidekeys = ['sphere', 'box','shue','future','lcb']
            view_set.display_single_iso(frame, mhddatafile,
                                        mode=mode[n],
                                        show_contour=False,
                                        show_fieldline=fieldline,
                                        show_legend=legend,
                                        show_slegend=slicelegend,
                                        show_flegend=fieldlegend,
                                        show_slice=doslice,
                                        timestamp_pos=[4,5],
                                        zone_hidekeys=zone_hidekeys,
                                        show_timestamp=timestamp)
            view_set.add_fieldlines(tp.active_frame(),mhddatafile,showleg=True,
                                    mode='allstations',
                                    station_file=
                         'ccmc_2019-08-30/magnetometers_e20190830-161300.mag')
    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
