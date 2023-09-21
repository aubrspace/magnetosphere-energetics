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
import pandas as pd
from global_energetics.analysis.workingtitle import central_diff

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

    ## Manually input case dictionary
    case_dict = {'base':inBase+'ideal_test_sp10-3.h5',
                 'conserve':inBase+'ideal_conserve_sp10-3.h5',
                 'ie1':inBase+'ideal_IE1_sp10-3.h5'}

    ## Load data and do minor adjustments
    dataset = load_data(case_dict)
    #split off interior surface
    base_interior = dataset['base']['/sphere10_inner_surface']
    conserve_interior = dataset['conserve']['/sphere10_inner_surface']
    ie1_interior = dataset['ie1']['/sphere10_inner_surface']
    #combine exterior with volume results
    dataset = combine_data(dataset,combo_list=['/sphere10_surface',
                                               '/sphere10_volume'],
                           newname='sphere10')

    ## Shorthand
    # Dataframes
    base_sphere = dataset['base']['sphere10']
    conserve_sphere = dataset['conserve']['sphere10']
    ie1_sphere = dataset['ie1']['sphere10']

    # Times
    times = conserve_sphere.index
    interv = times

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
    from IPython import embed; embed()
