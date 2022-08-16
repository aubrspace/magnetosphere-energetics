#!/usr/bin/env python3
"""Functions for handling and processing time varying magnetopause volume
    data in order to reorganize or otherwise manipulate prior to plotting
"""
import pandas as pd

def get_Lvalue(key_in,lparse):
    """interprets string to determine what Lvalue to assign
    Inputs
        key_in (str)- string input NOTE the key strings are expected to
                      have the form: qty+lparse+valuetag+units
        lparse (str)- info on how to find the right columns
    Returns
        Lvalue (float)
    """
    Linfo, daynight = key_in.split(' ')[0].split(lparse)[-1].split('_')
    if daynight=='night': sign = -1
    else: sign = 1
    valuestr = Linfo.split(lparse)[-1]
    if ('<' in valuestr):
        Lvalue = int(valuestr.split('<')[-1].split('>')[-1])-0.5
    if ('>' in valuestr):
        Lvalue = int(valuestr.split('<')[-1].split('>')[-1])+0.5
    elif '-' in valuestr:
        Lvalue = (float(valuestr.split('-')[0].split('l')[-1])+
                  float(valuestr.split('-')[1].split('l')[-1]))/2
    return Lvalue*sign

def get_derivatives(df, keys, values, *, lparse='_l'):
    """Function adds some lshell derivatives to the list of variables
    Inputs
        df (DataFrame)- dataframe object of input data
        keys (list)- which keys to take derivative of
        values (list)- Lshell values used to calculate finite differnce
    Returns
        df
    """
    for i,(key,L) in enumerate(zip(keys,values)):
        #Parse overcomplicated key
        meat,units = key.split(' ')
        energy,spatial = meat.split(lparse)
        ltag,daynight = spatial.split('_')
        #Find next lshell
        matchlist =[k for k in keys if(energy+'_l' in k)and(daynight in k)]
        if key==matchlist[-1]:
            nextLkey = key
        else:
            nextLkey = matchlist[matchlist.index(key)+1]#Assume nice order!
        #calculate the Lshell difference
        nextL = get_Lvalue(nextLkey,lparse)
        dL = nextL - L
        #calculate the value diff & make new key with dEdL value 
        df.loc[:,'d'+meat+'dL '+units] = (df[nextLkey]-df[key])/dL
        print('d'+meat+'dL '+units)
        #dEdL = (df.copy()[nextLkey]-df.copy()[key])/dL
        #df['d'+meat+'dL '+units] = dEdL
        #TODO only doing one of the base energy variables
    #from IPython import embed; embed()
    return df

def parse_lshells(df, lparse):
    """Function interprets all keys from dataframe to gt lshell keys,values
    Inputs
        df (DataFrame)- dataframe object of input data
        lparse (str)- info on how to find the right columns
    Returns
        lkeys (list)- subset of keys that are related to Lshell splitting
        Lvalues (list)- list of each Lvalue associated with lkeys
    """
    #figure out which columns are the lshell variable ones
    lkeys = [k for k in df.keys() if lparse in k]
    #use lparse to deduce the lshell level, assign a meaninful value
    qtys = set([qty.split(lparse)[0] for qty in lkeys])
    Lvalues = len(lkeys) * [0]
    for i, key in enumerate(lkeys):
        Lvalues[i] = get_Lvalue(key,lparse)
    Lvalues = set(Lvalues)
    return lkeys, Lvalues

def reformat_lshell(df_in, lparse, **kwargs):
    """Function restructures lshell splits so that coarse contour maps can
        be created
    Inputs
        df_in (DataFrame)- dataframe object of input data
        lparse (str)- info on how to find the right columns
        kwargs:
    Returns
        df_out (DataFrame)- new dataframe, NOTE structure is altered and
                            only lshell data + time is maintained!
    """
    lkeys, Lvalues = parse_lshells(df_in,lparse)
    '''
    #figure out which columns are the lshell variable ones
    lkeys = [k for k in df_in.keys() if lparse in k]
    #use lparse to deduce the lshell level, assign a meaninful value
    qtys = set([qty.split(lparse)[0] for qty in lkeys])
    Lvalues = len(lkeys) * [0]
    for i, key in enumerate(lkeys):
        Lvalues[i] = get_Lvalue(key,lparse)
    Lvalues = set(Lvalues)
    '''
    if kwargs.get('derivatives',True):
        df_in = get_derivatives(df_in,lkeys,Lvalues)
        lkeys, Lvalues = parse_lshells(df_in,lparse)
    #from IPython import embed; embed()
    #create new dataframe by duplicating with the L assignments
    df_out = pd.DataFrame()
    for Lvalue in Lvalues:
        og_keyset = [k for k in lkeys if get_Lvalue(k,lparse)==Lvalue]
        new_df = df_in[og_keyset].copy()
        for key in new_df.keys():
            simplified_key = key.split(lparse)[0]+' '+key.split(' ')[-1]
            new_df[simplified_key] = new_df[key]
        new_df.drop(columns=og_keyset,inplace=True)
        new_df['L'] = Lvalue
        df_out = df_out.append(new_df)
    return df_out
