#/usr/bin/env python
"""accesses satellite data from nasa CDA and creates <sat>.dat for SWMF input
"""
import os,sys
sys.path.append(os.getcwd().split('magnetosphere-energetics')[0]+
                                      'magnetosphere-energetics/')
import datetime as dt
import numpy as np
from numpy import sin,cos
from matplotlib import pyplot as plt
from matplotlib import patches
# Custom
from global_energetics.analysis.proc_indices import read_indices
from global_energetics.cdaweb_interface import(get_satellite_positions,
                                               get_satellite_bfield,
                                               get_satellite_plasma)
from global_energetics.wind_to_swmfInput import (collect_themis,collect_mms,
                                             collect_cluster,collect_geotail,
                                             collect_goes,collect_arase,
                                                 collect_rbsp)

def dual_half_circle(center:[float,float],
                     radius:float,
                      angle:int=90,
                         ax:plt.Axes=None,
                     colors:[str,str]=('black','white'),
                   **kwargs:dict) -> [patches.Wedge,patches.Wedge]:
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, ec=colors[0],
                       fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, ec=colors[0],
                       fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]

def write_SWMF_satfile(df,outname,outpath,**kwargs):
    """ function writes data from DataFrame to swmf satfile
    Inputs
    Returns
    """
    with open(os.path.join(outpath,outname),'w') as f:
        if 'note' in kwargs:
            f.write(kwargs.get('note')+'\n')
        f.write('Year Mo Dy Hr Mn Sc Msc X Y Z\n')
        f.write('\n')
        f.write('#COORD\n')
        f.write(kwargs.get('coord','GSM\n'))
        f.write('\n')
        f.write('#START\n')
        for t,(x,y,z) in zip(df.index,df[['x_gsm','y_gsm','z_gsm']].values):
            line =[]
            line.append(f'{t.year}')
            line.append(f'{t.month}')
            line.append(f'{t.day}')
            line.append(f'{t.hour:02n}')
            line.append(f'{t.minute:02n}')
            line.append(f'{t.second:02n}')
            line.append(f'{t.microsecond*1e3:03n}')
            line.append(f'{x:.2f}')
            line.append(f'{y:.2f}')
            line.append(f'{z:.2f}')
            f.write(''.join([s.rjust(7) for s in line])[3::]+'\n')
        print(f'\t\033[92m Created\033[00m {os.path.join(outpath,outname)}')

#Main program
def main() -> None:
    imf = read_indices(IMFPATH,start=START,end=END,read_supermag=False)
    solarwind = imf['swmf_sw']
    # Scrape data from CDAWeb
    cluster_pos = get_satellite_positions('cluster',START,END)
    #cluster_b   = get_satellite_bfield('cluster',START,END)
    rbsp_pos    = get_satellite_positions('rbsp',START,END)
    goes_pos    = get_satellite_positions('goes',START,END)
    mms_pos     = get_satellite_positions('mms',START,END,
                                          spacecraft_list=['1'])
    themis_pos  = get_satellite_positions('themis',START,END,
                                          spacecraft_list=['A','D','E'])
    arase_pos    = get_satellite_positions('arase',START,END)
    #TODO
    #   test / iterate through with cluster for plasma data
    #   (hopefully now robust) update static dictionaries and test for:
    #       goes
    #       mms
    #       rbsp
    #       themis
    #   Then make a reader/npz converter for CIMI satellite output
    #   (finally)
    #       actually start poking through data
    #       look for:
    #           energization in particle data
    #           evidence of pressure pulse in magnetic field measurments
    #           evidence of pressure pulse in particle data
    #           evidence of reduced anisotropy in agreement with FLC scattering
    '''
    arase_b      = get_satellite_bfield('arase',START,END)
    arase_plasma = get_satellite_plasma('arase',START,END)
    arase_plasma.pop('ERG_HEP_L2_OMNIFLUX_Epoch_H')# NOTE <- lazy

    np.savez_compressed(f'{OUTPATH}/arase_position.npz',**arase_pos,
                        allow_pickle=False)
    print(f'\t\033[92m Created\033[00m {OUTPATH}/arase_position.npz')
    np.savez_compressed(f'{OUTPATH}/arase_b.npz',**arase_b,
                        allow_pickle=False)
    print(f'\t\033[92m Created\033[00m {OUTPATH}/arase_b.npz')
    np.savez_compressed(f'{OUTPATH}/arase_plasma.npz',**arase_plasma,
                        allow_pickle=False)
    print(f'\t\033[92m Created\033[00m {OUTPATH}/arase_plasma.npz')
    '''
    '''
    arase_pos, arase_b, arase_plasma = collect_arase(START,END,
                                            skip_bfield=True,skip_plasma=True,
                                            writeData=True)
    #rbsp_pos, rbsp_b, rbsp_plasma = collect_rbsp(START,END,
    #                                        skip_bfield=True,skip_plasma=True,
    #                                        writeData=True)
    goes_pos, goes_b, goes_plasma = collect_goes(START,END,
                                            skip_bfield=False,skip_plasma=True,
                                            probes=['16','17'],
                                            writeData=True)
    cluster_pos, cluster_b,cluster_plasma = collect_cluster(START, END,
                                            skip_bfield=False,
                                            skip_plasma=False,
                                            probes=['1','2','3','4'],
                                            writeData=True)
    mms_pos, mms_b,mms_plasma = collect_mms(START, END,
                                            skip_bfield=False,
                                            skip_plasma=False,
                                            probes=['1'],
                                            writeData=True)
    themis_pos, themis_b,themis_plasma = collect_themis(START, END,
                                            skip_bfield=True,
                                            skip_plasma=True,
                                            probes=['A','D','E'],
                                            writeData=True)
    '''

    # Quick Plot
    if PLOT_DATA:
        quicklook,(equatorial,meridional) =plt.subplots(1,2,figsize=[20,10])
        dual_half_circle((0,0),1,ax=equatorial)
        dual_half_circle((0,0),1,ax=meridional)
        # Get the Shue magnetopause at its most compressed
        sw_min = solarwind.iloc[solarwind['r_shue98'].argmin()]
        sw_max = solarwind.iloc[solarwind['r_shue98'].argmax()]
        zenith = np.linspace(160,0,100)*np.pi/180
        r_shue_min = sw_min['r_shue98']*(2/(1+cos(zenith)))**sw_min['alpha']
        X_shue_min = r_shue_min*cos(zenith)
        Y_shue_min = r_shue_min*sin(zenith)
        r_shue_max = sw_max['r_shue98']*(2/(1+cos(zenith)))**sw_max['alpha']
        X_shue_max = r_shue_max*cos(zenith)
        Y_shue_max = r_shue_max*sin(zenith)
        Y_low = np.interp(X_shue_max,X_shue_min,Y_shue_min)
        equatorial.fill_between(X_shue_max,Y_low,Y_shue_max,fc='grey',
                                alpha=0.6)
        equatorial.fill_between(X_shue_max,-Y_low,-Y_shue_max,fc='grey',
                                alpha=0.6)
        meridional.fill_between(X_shue_max,Y_low,Y_shue_max,fc='grey',
                                alpha=0.6)
        meridional.fill_between(X_shue_max,-Y_low,-Y_shue_max,fc='grey',
                                alpha=0.6)
        for sat in arase_pos.keys():
            equatorial.scatter(arase_pos[sat]['x_gsm'],arase_pos[sat]['y_gsm'],
                            label=sat)
            meridional.scatter(arase_pos[sat]['x_gsm'],arase_pos[sat]['z_gsm'],
                            label=sat)
        for sat in rbsp_pos.keys():
            equatorial.scatter(rbsp_pos[sat]['x_gsm'],rbsp_pos[sat]['y_gsm'],
                            label=sat)
            meridional.scatter(rbsp_pos[sat]['x_gsm'],rbsp_pos[sat]['z_gsm'],
                            label=sat)
        for sat in cluster_pos.keys():
            equatorial.scatter(cluster_pos[sat]['x_gsm'],
                               cluster_pos[sat]['y_gsm'],
                        label=sat)
            meridional.scatter(cluster_pos[sat]['x_gsm'],
                               cluster_pos[sat]['z_gsm'],
                        label=sat)
        for sat in themis_pos.keys():
            equatorial.scatter(themis_pos[sat]['x_gsm'],
                               themis_pos[sat]['y_gsm'],
                        label=sat)
            meridional.scatter(themis_pos[sat]['x_gsm'],
                               themis_pos[sat]['z_gsm'],
                        label=sat)
        for sat in mms_pos.keys():
            equatorial.scatter(mms_pos[sat]['x_gsm'],mms_pos[sat]['y_gsm'],
                            label=sat)
            meridional.scatter(mms_pos[sat]['x_gsm'],mms_pos[sat]['z_gsm'],
                            label=sat)
        for sat in goes_pos.keys():
            equatorial.scatter(goes_pos[sat]['x_gsm'],goes_pos[sat]['y_gsm'],
                            label=sat)
            meridional.scatter(goes_pos[sat]['x_gsm'],goes_pos[sat]['z_gsm'],
                            label=sat)
        equatorial.set_xlim(20,-20)
        equatorial.set_ylim(20,-20)
        equatorial.set_xlabel('X GSM [R]')
        equatorial.set_ylabel('Y GSM [R]')
        equatorial.legend()
        meridional.set_xlim(20,-20)
        meridional.set_ylim(-20,20)
        meridional.set_xlabel('X GSM [R]')
        meridional.set_ylabel('Z GSM [R]')
        meridional.legend()
        plt.show()

    '''
    print('Writing Arase Satfiles ...')
    for sat in arase_pos.keys():
        write_SWMF_satfile(arase_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing RBSP Satfiles ...')
    for sat in rbsp_pos.keys():
        write_SWMF_satfile(rbsp_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing Cluster Satfiles ...')
    for sat in cluster_pos.keys():
        write_SWMF_satfile(cluster_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing MMS Satfiles ...')
    for sat in mms_pos.keys():
        write_SWMF_satfile(mms_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing THEMIS Satfiles ...')
    for sat in themis_pos.keys():
        write_SWMF_satfile(themis_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing GOES Satfiles ...')
    for sat in goes_pos.keys():
        write_SWMF_satfile(goes_pos[sat],sat+'.dat',OUTPATH,
                           note=f'{sat} Created {str(dt.datetime.now())}')
    '''

if __name__ == '__main__':
    global START,END,OUTPATH,PLOT_DATA
    #############################USER INPUTS HERE##########################
    START     = dt.datetime(2019,5,13,19,0)
    END       = dt.datetime(2019,5,15,10,30)
    OUTPATH   = './inputs/satellites'
    PLOT_DATA = False
    IMFPATH   = './inputs/simulations/'
    #######################################################################

    main()
