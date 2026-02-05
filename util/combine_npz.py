import numpy as np
from glob import glob
from tqdm import tqdm

def main() -> None:
    # Glob files
    filelist = glob(f"{INPATH}/{KEY}")
    # Load data
    combined_data = dict(np.load(filelist[0]))
    for infile in tqdm(filelist[1::]):
        data = dict(np.load(infile))
        for key in[k for k in combined_data.keys()if 'allow_pickle' not in k]:
            if combined_data[key].size == 1:
                combined_data[key] = np.concatenate(([combined_data[key]],
                                                     [data[key]]))
            else:
                combined_data[key] = np.concatenate((combined_data[key],
                                                     [data[key]]))
    # Sort by time
    iorder = np.argsort(combined_data['time'])
    for key in [k for k in combined_data.keys() if 'allow_pickle' not in k]:
        combined_data[key] = combined_data[key][iorder]

    # Save combined file in the same place
    np.savez_compressed(f'{INPATH}/energetics.npz',**combined_data)
    print(f"\033[92m Saved\033[00m {INPATH}/energetics.npz")

if __name__=='__main__':

    global INPATH,KEY

    #INPATH = 'outputs_may2019'
    INPATH = 'data/analysis'
    KEY    = 'energetics_*.npz'

    main()
