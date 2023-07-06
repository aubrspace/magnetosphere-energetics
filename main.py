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
#import spacepy
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
from global_energetics import makevideo

if __name__ == "__main__":
    start_time = time.time()
    if '-c' in sys.argv:
        tp.session.connect()
        tp.new_layout()

    else:
        pass
        #os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2018r2/bin:/usr/local/tecplot/360ex_2018r2/bin/sys:/usr/local/tecplot/360ex_2018r2/bin/sys-util'
        #os.environ["LD_LIBRARY_PATH"]='/usr/local/tecplot/360ex_2022r2/bin/llvm:/usr/local/tecplot/360ex_2022r2/bin/osmesa:/usr/local/tecplot/360ex_2022r2/bin:/usr/local/tecplot/360ex_2022r2/bin/sys-util:/usr/local/tecplot/360ex_2022r2/bin/Qt'
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
    starlink5 = ('ccmc_2022-02-02/3d__var_1_e20220203-115300-032.plt',
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
    febtest = ('febstorm/3d__var_1_e20140218-230000-000.plt')
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
    all_main_phase = glob.glob('ccmc_2022-02-02/3d*')
    all_times = sorted(glob.glob('ccmc_2022-02-02/3d__var*.plt'),
                                key=makevideo.time_sort)[0::]
    starlink_impact = (dt.datetime(2022,2,2,23,58)+
                       dt.timedelta(hours=3,minutes=55))
    starlink_endMain1 = dt.datetime(2022,2,3,11,54)

    oggridfile = 'ccmc_2022-02-02/3d__volume_e20220202.plt'

    mp_surf = pd.DataFrame()
    mp_vol = pd.DataFrame()
    lobe_surf = pd.DataFrame()
    lobe_vol = pd.DataFrame()
    closed_surf = pd.DataFrame()
    closed_vol = pd.DataFrame()
    #if False:

    #for inputs in [febtest]:
    #for inputs in [starlink,starlink2,starlink3,starlink4]:
    i=0
    for k,f in enumerate(all_times):
        filetime = makevideo.get_time(f)
        #if filetime>starlink_impact and filetime<starlink_endMain1:
        if filetime>dt.datetime(2022,2,3,9,0) and filetime<dt.datetime(2022,2,3,10,0):
            print('('+str(i)+') ',filetime)
            i+=1
            tp.new_layout()
            #mhddatafile = inputs[0]
            mhddatafile = f
            OUTPUTNAME = mhddatafile.split('e')[-1].split('.')[0]
            #python objects
            #field_data = tp.data.load_tecplot(inputs)
            field_data = tp.data.load_tecplot(mhddatafile)
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
                                    debug=True,
                                    do_cms=False,
                                    do_central_diff=False,
                                    analysis_type='energy_mass',
                                    modes=['iso_betastar','closed',
                                           'nlobe','slobe'],
                                    inner_r=4,
                                    blankvalue=4,
                                    #modes=['sphere'],
                                    #sp_rmax=10,
                                    #sp_rmin=4,
                                    #xc=10,yc=40,zc=40,
                                    #keep_zones='all',
                                    do_interfacing=True,
                                    integrate_surface=True,
                                    integrate_volume=True,
                                    integrate_line=False,
                                    truegridfile=oggridfile,
                                    outputpath='dynamic_test/r4ReInnerBound/',
                                    #surface_unevaluated_type='energy',
                                    #add_eqset=['energy_flux','volume_energy'],
                                    #customTerms={'test':'TestArea [Re^2]'}
                                    )
            if i>30:
                break
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
    if False:
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

        #M1,2,5,total from mp
        M1 = mp_vol['UtotM1 [W]']
        M2 = mp_vol['UtotM2 [W]']
        M5 = mp_vol['UtotM5 [W]']
        M = mp_vol['UtotM [W]']
        #M1a,1b,2b,il from lobes
        M1a = lobe_vol['UtotM1a [W]']
        M1b = lobe_vol['UtotM1b [W]']
        M2b = lobe_vol['UtotM2b [W]']
        M2d = lobe_vol['UtotM2d [W]']
        Mil = lobe_vol['UtotMil [W]']
        #M5a,5b,2a,ic from closed
        M5a = closed_vol['UtotM5a [W]']
        M5b = closed_vol['UtotM5b [W]']
        M2a = closed_vol['UtotM2a [W]']
        M2c = closed_vol['UtotM2c [W]']
        Mic = closed_vol['UtotMic [W]']
        for m in [M1,M2,M5,M,M1a,M1b,M2a,M2b,M2c,M2d,M5a,M5b,Mic,Mil]:
            m = (m[0]+m[1])/2

        #Volume totals
        Tmp = mp_vol['Utot [J]']
        Tl = lobe_vol['Utot [J]']
        Tc = closed_vol['Utot [J]']

        #Central difference
        dTdtmp = -1*surface_tools.central_diff(Tmp,60)
        dTdtl = -1*surface_tools.central_diff(Tl,60)
        dTdtc = -1*surface_tools.central_diff(Tc,60)

        #Combine into dEdt_sum
        #   Lobes- (KM1,2,3,4)
        # NOTE M2 is from lobes -> closed and is equiv to:
        #           -M2a+M2b-M2c+M2d
        #   use -M2 or the reverse of above for closed balance
        #dEdt_suml = K1+K2al+K2bl+K3+K4+M1a+M1b+M2a+M2b
        dEdt_suml = M1a+M1b-M2a+M2b-M2c+M2d
        #   Closed- (KM5,2,6,7)
        #dEdt_sumc = K5+K2ac+K2bc+K6+K7+M5a+M5b+M2a+M2b
        dEdt_sumc = M5a+M5b+M2a-M2b+M2c-M2d
        #   Total- (KM,1,5,3,4,6,7)
        #dEdt_sumt1 = K1+K5+K3+K4+K6+K7+M1a+M1b+M5a+M5b
        #dEdt_sumt2 = K1+K5+K3+K4+K6+K7+M1+M5
        dEdt_sumt = M

        #Display errors
        error_volume = (mp_vol['Volume [Re^3]']-lobe_vol['Volume [Re^3]']
                        - closed_vol['Volume [Re^3]'])
        from IPython import embed; embed()
    if False:
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

    if '-c' in sys.argv:
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{X = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Y = 0}')
        tp.macro.execute_command('$!GlobalThreeD RotateOrigin{Z = 0}')
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
