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
                         'mms':'GE_OR_DEF',
                        'rbsp':'RBSP*_REL04_ECT-HOPE-MOM-L3',
                      'themis':'TH*_OR_SSC'}

bfield_instrument_dict = {'arase':'ERG_MGF_L2_8SEC',
                        'cluster':'C*_CP_FGM_SPIN',
                           'goes':'',#TODO
                            'mms':'',#TODO
                           'rbsp':'',#TODO
                         'themis':''}#TODO

plasma_instrument_dict = {'arase':['ERG_LEPE_L2_OMNIFLUX',
                                   'ERG_MEPE_L2_OMNIFLUX',
                                   'ERG_HEP_L2_OMNIFLUX',
                                   'ERG_XEP_L2_OMNIFLUX'],
                        'cluster':['C*_PP_CIS'],
                          }#TODO

pos_variable_keys_dict = {'arase':['pos_gsm'],
                        'cluster':['sc_dr*_xyz_gse__CL_SP_AUX'],
                           'goes':['XYZ_GSM'],
                            'mms':['GSM_POS'],
                           'rbsp':['Position'],
                         'themis':['XYZ_GSM']}

bfield_variable_keys_dict = {'arase':'mag_8sec_gsm',
                           'cluster':'B_vec_xyz_gse__C*_CP_FGM_SPIN',
                              'goes':'',#TODO
                               'mms':'',#TODO
                              'rbsp':'',#TODO
                            'themis':''}#TODO

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
                                 #TODO
                                 }

pos_time_key_dict = {'arase':'epoch',
                   'cluster':'Epoch__CL_SP_AUX',
                      'goes':'Epoch',
                       'mms':'Epoch',
                      'rbsp':'Epoch_Ion',
                    'themis':'Epoch'}

bfield_time_key_dict = {'arase':'epoch_8sec',
                      'cluster':'Epoch__C*_CP_FGM_SPIN'}

plasma_time_key_dict = {'cluster':'Epoch__C*_PP_CIS',
                        'arase':'epoch'}

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
                         'rbsp':1,
                       'themis':1}

