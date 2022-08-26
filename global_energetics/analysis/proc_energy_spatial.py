#!/usr/bin/env python3
"""Functions for handling and processing time varying magnetopause volume
    data in order to reorganize or otherwise manipulate prior to plotting
"""
import pandas as pd
import time

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
    if 'night' in daynight: sign = -1
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

def get_l_derivatives(df, keys, *, lparse='_l',
                      base_vars=['Utot2','u_db','uHydro','Utot']):
    """Function adds some lshell derivatives to the list of variables
    Inputs
        df (DataFrame)- dataframe object of input data
        keys (list)- which keys to take derivative of
    Returns
        df
    """
    for i,key in enumerate([k for k in keys if
                                       any([b in k for b in base_vars])]):
        #Parse overcomplicated key
        meat,units = key.split(' ')
        energy,spatial = meat.split(lparse)
        ltag,daynight = spatial.split('_')
        #Find next lshell
        matchlist =[k for k in keys if(energy+'_l' in k)and(daynight in k)]
        if key==matchlist[-1]:
            dL = 0.1
        else:
            nextLkey = matchlist[matchlist.index(key)+1]#Assume nice order!
            #calculate the Lshell difference
            L = get_Lvalue(key,lparse)
            nextL = get_Lvalue(nextLkey,lparse)
            dL = nextL - L
        #calculate the value diff & make new key with dEdL value 
        #df.loc[:,'d'+meat+'dL '+units] = (df[nextLkey]-df[key])/dL
        df.loc[:,'d'+meat+'dL '+units] = (df[nextLkey]-df[key])/dL/df[key]
        #df['d'+meat+'dL '+units].fillna(method='bfill',inplace=True)
        #df['d'+meat+'dL '+units].fillna(value=0,inplace=True)
    return df

def get_t_derivatives(df,*,qtys=['Utot2','u_db','uHydro','Utot'],**kwargs):
    """Function uses index to infer time stamps and calculates forward
        finite difference temporal derivative
    Inputs
        df (DataFrame)- data to manipulate
        qtys (list[str])- which keys to cacluate derivatives on
        kwargs:
            None
    Return
        df (altered)
    """
    times = ((df.index-df.index[0]).days*1440+
             (df.index-df.index[0]).seconds/60)/60
    df['t'] = times.values
    #Get (dEdt)/(dE/dL) perp to gradient levels of dL/dt
    for qty in ['Utot2','u_db','uHydro','Utot']:
        #df['d'+qty+'dt'] = df[qty+' [J]'].diff()/(df['t'].diff())#perHr
        df['d'+qty+'dt'] = df[qty+' [J]'].diff()/(df['t'].diff())/df[qty+' [J]']
        df['dLdt_'+qty]= df['d'+qty+'dt']/df['d'+qty+' [J]']
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
    if kwargs.get('derivatives',True):
        df_in = get_l_derivatives(df_in,lkeys)
        lkeys, Lvalues = parse_lshells(df_in,lparse)
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
    df_out = get_t_derivatives(df_out)
    return df_out
