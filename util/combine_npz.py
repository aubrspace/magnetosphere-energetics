import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

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
    # Example text
    example_text = """
examples:
    python util/combine_npz.py -i run_quiet/output3d/ -f output_*.npz
    """

    # Built in argument parser argument (I think this is built on sys)
    parser = argparse.ArgumentParser(epilog=example_text,
                         formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add parser options
    parser.add_argument('-i','--input',default='./',help='path to input data')
    parser.add_argument('-f','--file',default='energetics_*.npz',
                        help='file template used to glob the files')


    args = parser.parse_args()

    global INPATH,KEY

    INPATH = args.input
    KEY    = args.file

    main()
