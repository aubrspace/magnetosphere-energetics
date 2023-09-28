#!/usr/bin/env python3
#TODO: Try estimating the altered (kinetic?) energy from the inner boundary
#       condition, evaluate for all the points are at the edge of 2.5,
#       basically the inner most points. Then need to account for each's small
#       volume and do 1/2 rho u^2 * volume???
#
#       If this process works, perhaps we keep the analysis all the way in
#
#       Could also determine the fixed inner cell roster a single time and take
#       the integral that way on the predetermined cells
#           > by cell id number?
#           > or location?
#           > def save the routine to recalculate if necessary (AMR)
import sys,os
from matplotlib import pyplot as plt
import datetime as dt
import pandas as pd
from global_energetics.analysis.workingtitle import central_diff
from global_energetics.analysis.plot_tools import (general_plot_settings,
                                                   pyplotsetup)

def load_data(case_dict):
    ## Analysis Data
    dataset = {}
    for key,file in case_dict.items():
        store = pd.HDFStore(file)
        dataset[key] = {}
        for storekey in store.keys():
            dataset[key][storekey] = store[storekey]
        store.close()
    return dataset

def combine_data(in_dataset,combo_list=['test'],newname='combined'):
    dataset = in_dataset.copy()
    for sectionkey in dataset.keys():
        data = dataset[sectionkey]
        if all([k in data.keys() for k in combo_list]):
            data[newname] = pd.DataFrame()
            for combo_target in combo_list:
                olddf = data.pop(combo_target)
                for dfkey,values in olddf.items():
                    data[newname][dfkey] = values
    return dataset

if __name__ == "__main__":
    ## Input path, then create output dir's
    inBase = sys.argv[-1]
    outPath = os.path.join(inBase,'figures')
    for path in [outPath]:
        os.makedirs(path,exist_ok=True)

    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    ## Manually input case dictionary
    case_dict = {'base':inBase+'ideal_test_sp10-3.h5',
                 'conserve':inBase+'ideal_conserve_sp10-3.h5',
                 'ie1':inBase+'ideal_IE1_sp10-3.h5',
                 'noRCM1':inBase+'ideal_noRCM1_sp10-3.h5',
                 'GMonly':inBase+'GMonly_sp10-3.h5',
                 'GMtail':inBase+'GMonly_tail10-3.h5',
                 'refined':inBase+'ideal_refined_sp10-3.h5'}

    ## Load data and do minor adjustments
    dataset = load_data(case_dict)
    #split off interior surface
    base_interior = dataset['base']['/sphere10_inner_surface']
    conserve_interior = dataset['conserve']['/sphere10_inner_surface']
    ie1_interior = dataset['ie1']['/sphere10_inner_surface']
    noRCM1_interior = dataset['noRCM1']['/sphere10_inner_surface']
    GMonly_interior = dataset['GMonly']['/sphere10_inner_surface']
    GMtail_interior = dataset['GMtail']['/sphere10_inner_surface']
    refined_interior = dataset['refined']['/sphere10_inner_surface']
    #combine exterior with volume results
    dataset = combine_data(dataset,combo_list=['/sphere10_surface',
                                               '/sphere10_volume'],
                           newname='sphere10')

    ## Shorthand
    # Dataframes
    base_sphere = dataset['base']['sphere10']
    conserve_sphere = dataset['conserve']['sphere10']
    ie1_sphere = dataset['ie1']['sphere10']
    noRCM1_sphere = dataset['noRCM1']['sphere10']
    GMonly_sphere = dataset['GMonly']['sphere10']
    GMtail_sphere = dataset['GMtail']['sphere10']
    refined_sphere = dataset['refined']['sphere10']

    # Times
    interv = conserve_sphere.index[10::]
    interv3 = noRCM1_sphere.index[10::]
    interv98 = GMonly_sphere.index[10::]

    reltimes = interv-interv[0]
    reltimes3 = interv3-interv3[0]
    reltimes98 = interv98-interv98[0]

    times = [float(n) for n in reltimes.to_numpy()]
    times3 = [float(n) for n in reltimes3.to_numpy()]
    times98 = [float(n) for n in reltimes98.to_numpy()]

    dt = [t.seconds for t in interv[1::]-interv[0:-1]]
    dt3 = [t.seconds for t in interv3[1::]-interv3[0:-1]]
    dt98 = [t.seconds for t in interv98[1::]-interv98[0:-1]]

    dt.append(dt[-1])
    dt3.append(dt3[-1])
    dt98.append(dt98[-1])

    # Fluxes
    # Unmodified
    baseKs1 = base_sphere.loc[interv,'K_net [W]']
    baseKs3 = base_interior.loc[interv,'K_net [W]']
    # ConservationCriteria
    conserveKs1 = conserve_sphere.loc[interv,'K_net [W]']
    conserveKs3 = conserve_interior.loc[interv,'K_net [W]']
    # IonosphereChange 1
    ie1Ks1 = ie1_sphere.loc[interv,'K_net [W]']
    ie1Ks3 = ie1_interior.loc[interv,'K_net [W]']
    # NoRCM 1
    noRCM1Ks1 = noRCM1_sphere.loc[interv3,'K_net [W]']
    noRCM1Ks3 = noRCM1_interior.loc[interv3,'K_net [W]']
    # GMonly
    GMonlyKs1 = GMonly_sphere.loc[interv98,'K_net [W]']
    GMonlyKs3 = GMonly_interior.loc[interv98,'K_net [W]']
    # GMtail
    GMtailKs1 = GMtail_sphere.loc[interv98,'K_net [W]']
    GMtailKs3 = GMtail_interior.loc[interv98,'K_net [W]']
    # Refined to 1/16th at inner boundary
    refinedKs1 = refined_sphere.loc[interv,'K_net [W]']
    refinedKs3 = refined_interior.loc[interv,'K_net [W]']

    # Energies
    # Unmodified
    baseU = base_sphere.loc[interv,'Utot [J]']
    baseK_sp = -1*central_diff(baseU)
    # ConservationCriteria
    conserveU = conserve_sphere.loc[interv,'Utot [J]']
    conserveK_sp = -1*central_diff(conserveU)
    # IonosphereChange 1
    ie1U = ie1_sphere.loc[interv,'Utot [J]']
    ie1K_sp = -1*central_diff(ie1U)
    # noRCM 1
    noRCM1U = noRCM1_sphere.loc[interv3,'Utot [J]']
    noRCM1K_sp = -1*central_diff(noRCM1U)
    # GMonly
    GMonlyU = GMonly_sphere.loc[interv98,'Utot [J]']
    GMonlyK_sp = -1*central_diff(GMonlyU)
    # GMtail
    GMtailU = GMtail_sphere.loc[interv98,'Utot [J]']
    GMtailK_sp = -1*central_diff(GMtailU)
    # Refined to 1/16th at inner boundary
    refinedU = refined_sphere.loc[interv,'Utot [J]']
    refinedK_sp = -1*central_diff(refinedU)

    # Errors
    # Unmodified
    baseError = (baseKs1+baseKs3-baseK_sp)
    baseRelErr = baseError/baseK_sp
    # ConservationCriteria
    conserveError = (conserveKs1+conserveKs3-conserveK_sp)
    conserveRelErr = conserveError/conserveK_sp
    # IonosphereChange 1
    ie1Error = (ie1Ks1+ie1Ks3-ie1K_sp)
    ie1RelErr = ie1Error/ie1K_sp
    # noRCM 1
    noRCM1Error = (noRCM1Ks1+noRCM1Ks3-noRCM1K_sp)
    noRCM1RelErr = noRCM1Error/noRCM1K_sp
    # GMonly
    GMonlyError = (GMonlyKs1+GMonlyKs3-GMonlyK_sp)
    GMonlyRelErr = GMonlyError/GMonlyK_sp
    # GMtail
    GMtailError = (GMtailKs1+GMtailKs3-GMtailK_sp)
    GMtailRelErr = GMtailError/GMtailK_sp
    # Refined to 1/16th at inner boundary
    refinedError = (refinedKs1+refinedKs3-refinedK_sp)
    refinedRelErr = refinedError/refinedK_sp
    from IPython import embed; embed()

    #######################################################################
    ## plots
    if True:
        ##########################
        #setup figure
        figure1,(toppanel,botpanel) = plt.subplots(2,1,figsize=[16,16])
        #Plot
        toppanel.plot(times, baseK_sp/1e12, label='baseTrue')
        toppanel.plot(times, (baseKs1+baseKs3)/1e12, label='base')
        #toppanel.plot(times98, GMonlyK_sp/1e12, label='GMonlyTrue')
        #toppanel.plot(times98, (GMonlyKs1+GMonlyKs3)/1e12, label='GMonly')
        #toppanel.plot(times98, GMtailK_sp/1e12, label='GMtailTrue')
        #toppanel.plot(times98, (GMtailKs1+GMonlyKs3)/1e12, label='GMtail')
        toppanel.plot(times, refinedK_sp/1e12, label='refinedTrue')
        toppanel.plot(times, (refinedKs1+refinedKs3)/1e12, label='refined')
        #toppanel.plot(times3, noRCM1K_sp/1e12, label='noRCM1True')
        #toppanel.plot(times3, (noRCM1Ks1+noRCM1Ks3)/1e12, label='noRCM1')
        #toppanel.plot(times, conserveK_sp/1e12, label='conserveTrue')
        #toppanel.plot(times, (conserveKs1+conserveKs3)/1e12, label='conserve')

        botpanel.fill_between(times, conserveK_sp/1e12, label='conserveTrue',
                              fc='grey')
        botpanel.plot(times, (conserveKs1)/1e12, label='conserveK1')
        botpanel.plot(times, (conserveKs3)/1e12, label='conserveK3')
        botpanel.plot(times, (conserveKs1+conserveKs3)/1e12, label='conserve')

        #Decorations
        general_plot_settings(toppanel,do_xlabel=False,legend=True,
                              #ylim=[-10,10],
                              ylabel=r'Energy Flux $\left[ TW\right]$',
                              timedelta=True)
        general_plot_settings(botpanel,do_xlabel=False,legend=True,
                              #ylim=[-10,10],
                              ylabel=r'Energy Flux $\left[ TW\right]$',
                              timedelta=True)
        #save
        figure1.tight_layout(pad=1)
        figurename = outPath+'/demo.png'
        figure1.savefig(figurename)
        plt.close(figure1)
        print('\033[92m Created\033[00m',figurename)
        ##########################
        ##########################
        #setup figure
        figure2,(toppanel,botpanel) = plt.subplots(2,1,figsize=[16,16])
        #Plot
        toppanel.fill_between(times, (baseU-baseU[0])/1e15,fc='grey',
                              label='baseEnergy')
        toppanel.plot(times, -1*(baseK_sp).cumsum()*dt/1e15,label='baseCheck')
        toppanel.plot(times, -1*(baseKs1+baseKs3).cumsum()*dt/1e15,
                                                              label='baseSUM')

        '''
        botpanel.fill_between(times3, (noRCM1U-noRCM1U[0])/1e15,fc='grey',
                              label='noRCM1Energy')
        botpanel.plot(times3, -1*(noRCM1K_sp).cumsum()*dt3/1e15,
                                                          label='noRCM1Check')
        botpanel.plot(times3, -1*(noRCM1Ks1+noRCM1Ks3).cumsum()*dt3/1e15,
                                                            label='noRCM1SUM')
        botpanel.plot(times3, -1*(noRCM1Ks1).cumsum()*dt3/1e15,
                                                            label='noRCM1K1')
        botpanel.plot(times3, -1*(noRCM1Ks3).cumsum()*dt3/1e15,
                                                            label='noRCM1K3')
        '''
        botpanel.fill_between(times, (conserveU-conserveU[0])/1e15,fc='grey',
                              label='conserveEnergy')
        botpanel.plot(times, -1*(conserveK_sp).cumsum()*dt/1e15,
                                                        label='conserveCheck')
        botpanel.plot(times, -1*(conserveKs1+conserveKs3).cumsum()*dt/1e15,
                                                        label='conserveSUM')
        botpanel.plot(times, -1*(conserveKs1).cumsum()*dt/1e15,
                                                        label='conserveK1')
        botpanel.plot(times, -1*(conserveKs3).cumsum()*dt/1e15,
                                                        label='conserveK3')
        #Decorations
        general_plot_settings(toppanel,do_xlabel=False,legend=True,
                              #ylim=[-10,10],
                              ylabel=r'Energy $\left[ PJ\right]$',
                              timedelta=True)
        general_plot_settings(botpanel,do_xlabel=False,legend=True,
                              #ylim=[-10,10],
                              ylabel=r'Energy $\left[ PJ\right]$',
                              timedelta=True)
        #save
        figure2.tight_layout(pad=1)
        figurename = outPath+'/demo2.png'
        figure2.savefig(figurename)
        plt.close(figure2)
        print('\033[92m Created\033[00m',figurename)
        ##########################
    #######################################################################
