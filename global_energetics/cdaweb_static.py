# This is a static set of python dictionaries for easy navigation w CDAWeb
test_dict = {'key1':'pair1'}

spacecraft_IDs = {'arase':[''],
                'cluster':['1','2','3','4'],
                   'goes':['16','17'],
                    'mms':['1','2','3','4'],
                   'rbsp':['A','B'],
                 'themis':['A','B','C','D','E']}

pos_instrument_dict = {'arase':'ERG_ORB_L2',
                     'cluster':'CL_SP_AUX',
                        'goes':'GOES*_EPHEMERIS_SSC',
                         'mms':'MMS*_MEC_SRVY_L2_EPHT89D',
                        'rbsp':'RBSP*_REL04_ECT-HOPE-MOM-L3',
                      'themis':'TH*_OR_SSC'}

bfield_instrument_dict = {'arase':'ERG_MGF_L2_8SEC',
                        'cluster':'C*_CP_FGM_SPIN',
                           'goes':'DN_MAGN-L2-HIRES_G*',
                            'mms':'MMS*_FGM_SRVY_L2',
                           'rbsp':'RBSP-*_MAGNETOMETER_4SEC-GSM_EMFISIS-L3',
                         'themis':'TH*_L2_FGM'}

plasma_instrument_dict = {'arase':['ERG_LEPE_L2_OMNIFLUX',
                                   'ERG_MEPE_L2_OMNIFLUX',
                                   'ERG_HEP_L2_OMNIFLUX',
                                   'ERG_XEP_L2_OMNIFLUX'],
                        'cluster':['C*_PP_CIS'],
                           'rbsp':['RBSP*_REL04_ECT-MAGEIS-L3',
                                   'RBSP*_REL03_ECT-REPT-SCI-L2'],
                            'mms':['MMS*_FPI_FAST_L2_DIS-MOMS'],
                          #'themis':['TH*_L2_MOM']}
                          'themis':['TH*_L2_GMOM']}

pos_variable_keys_dict = {'arase':['pos_gsm'],
                        'cluster':['sc_dr*_xyz_gse__CL_SP_AUX'],
                           'goes':['XYZ_GSM'],
                            'mms':['mms*_mec_r_gsm'],
                           'rbsp':['Position'],
                         'themis':['XYZ_GSM']}

bfield_variable_keys_dict = {'arase':'mag_8sec_gsm',
                           'cluster':'B_vec_xyz_gse__C*_CP_FGM_SPIN',
                              'goes':'b_gsm',
                               'mms':'mms*_fgm_b_gsm_srvy_l2_clean',
                              'rbsp':'Mag',
                            'themis':'th*_fgs_gsm'}

plasma_variable_keys_dict = {'cluster':{'C*_PP_CIS':
                                              ['N_p__C*_PP_CIS',# n/cc
                                               'V_p_xyz_gse__C*_PP_CIS', #km/s
                                               'T_p_par__C*_PP_CIS', #MK
                                               'T_p_perp__C*_PP_CIS'] #MK
                                        },
                             'arase':{'ERG_LEPE_L2_OMNIFLUX':['FEDO'],
                                      'ERG_MEPE_L2_OMNIFLUX':['FEDO'],
                                      'ERG_HEP_L2_OMNIFLUX':['FEDO_L',# lowE
                                                             'FEDO_H'],# highE
                                      'ERG_XEP_L2_OMNIFLUX':['FEDO_SSD']
                                      },
                              'rbsp':{'RBSP*_REL04_ECT-MAGEIS-L3':
                                                ['Position',
                                                 'MLT','L','L_star',
                                                 'FEDU_CORR_plasmagram',
                                                 'FPDU_plasmagram'],
                                     'RBSP*_REL03_ECT-REPT-SCI-L2':
                                                ['Position',
                                                 'MLT','L','L_star',
                                                 'FESA','FEDU','FPSA','FPDU']
                                     },
                               'mms':{'MMS*_FPI_FAST_L2_DIS-MOMS':
                                               ['mms*_dis_numberdensity_fast',
                                                'mms*_dis_bulkv_gse_fast',
                                                'mms*_dis_temppara_fast',
                                                'mms*_dis_tempperp_fast',
                                             'mms*_dis_energyspect_omni_fast']
                                     },
                             #'themis':{'TH*_L2_MOM':
                             #                  ['th*_peem_density',
                             #                   'th*_peim_density',
                             #                   'th*_peim_velocity_gsm',
                             #                   'th*_peim_ptot']
                             'themis':{'TH*_L2_GMOM':
                                                ['th*_pteff_en_eflux',
                                                 'th*_ptiff_en_eflux']
                                     }
                                 }

pos_time_key_dict = {'arase':'epoch',
                   'cluster':'Epoch__CL_SP_AUX',
                      'goes':'Epoch',
                       'mms':'Epoch',
                      'rbsp':'Epoch_Ion',
                    'themis':'Epoch'}

bfield_time_key_dict = {'arase':'epoch_8sec',
                      'cluster':'Epoch__C*_CP_FGM_SPIN',
                         'rbsp':'Epoch',
                         'goes':'Epoch',
                          'mms':'Epoch',
                       'themis':'th*_fgs_epoch'}

plasma_time_key_dict = {'cluster':'Epoch__C*_PP_CIS',
                          'arase':'epoch',
                           'rbsp':'Epoch',
                            'mms':'Epoch',
                         'themis':['th*_pteff_epoch',
                                   'th*_ptiff_epoch']}

needs_rotation = {'arase':False,
                'cluster':True,
                   'goes':False,
                    'mms':False,
                   'rbsp':False,
                 'themis':False}

unit_conversion_dict = {'arase':1,
                      'cluster':1/6371,
                         'goes':1,
                          'mms':1/6371,
                         'rbsp':1/6371,
                       'themis':1}

