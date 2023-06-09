import os
import argparse
import numpy as np
from glob import glob1
import astropy.io.fits as fits

def data_stats(data_dir, img_type=['gamma1', 'gamma2', 'kappa']):
    img_names = sorted(glob1(os.path.join(data_dir, img_type), '*.fits'))
    img_list = []
    iter_num = 0
    for img_name in img_names:
        with fits.open(os.path.join(data_dir, img_type, img_name), memmap=False) as f:
            img_list.append(f[0].data)
            iter_num += 1
        if iter_num % 500 == 0:
            print(f'Processing {iter_num}/{len(img_names)} images')
    img_list = np.array(img_list)
    print('Converted to array.')
    if np.any(np.isnan(img_list)):
        print('There are nan values in the dataset:')
        nan_idx = np.argwhere(np.isnan(img_list))[:,0]
        nan_idx = np.unique(nan_idx)
        return nan_idx
    else:
        return "No nan present."

def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('name', type=str, help='type of data')
    return parser.parse_args()


data_dir = '/share/lirui/Wenhan/WL/data_new'

if __name__ == '__main__':
    args = get_args()
    nan_idx = data_stats(data_dir, img_type=args.name)
    print('Nan img indexes are:', nan_idx)
    print('Note: index starts from 0, so img num might be index + 1')