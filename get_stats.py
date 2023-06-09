# %%
import os
import numpy as np
from glob import glob1
import astropy.io.fits as fits

# get global stats
def data_stats(data_dir, img_type=['gamma1', 'gamma2', 'kappa']):
    img_names = sorted(glob1(os.path.join(data_dir, img_type), '*.fits'))
    img_list = []
    iter_num = 0
    for img_name in img_names:
        with fits.open(os.path.join(data_dir, img_type, img_name), memmap=False) as f:
            img_list.append(f[0].data)
            iter_num += 1
        if iter_num % 2000 == 0:
            print(f'Processing {iter_num}/{len(img_names)} images')
    img_list = np.array(img_list)
    print('Converted to array.')
    return np.mean(img_list), np.std(img_list)

# data_dir = '/Users/danny/Desktop/WL/data_new'
data_dir = '/share/lirui/Wenhan/WL/data_new'

mean, std = data_stats(data_dir, img_type='gamma1')
print(f'gamma 1: mean = {mean}, std = {std}')

mean, std = data_stats(data_dir, img_type='gamma2')
print(f'gamma 2: mean = {mean}, std = {std}')

# %%
