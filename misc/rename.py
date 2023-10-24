# %%
import os
from glob import glob1

rename_dir = '/Users/danny/Desktop/data_new/gamma2'

file_list = sorted(glob1(rename_dir, '*'))

for fname in file_list:
    old_name = os.path.join(rename_dir, fname)
    if len(fname) == 15:
        new_name = fname[:-8] + '00' + fname[-8:]
        new_name = os.path.join(rename_dir, new_name)
        os.rename(old_name, new_name)
    elif len(fname) == 16:
        new_name = fname[:-9] + '0' + fname[-9:]
        new_name = os.path.join(rename_dir, new_name)
        os.rename(old_name, new_name)

# %%
rename_dir = '/Users/danny/Desktop/data_new/kappa'

file_list = sorted(glob1(rename_dir, '*'))

for fname in file_list:
    old_name = os.path.join(rename_dir, fname)
    if len(fname) == 12:
        new_name = 'map_' + '00' + fname[-8:]
        new_name = os.path.join(rename_dir, new_name)
        os.rename(old_name, new_name)
    elif len(fname) == 13:
        new_name = 'map_' + '0' + fname[-9:]
        new_name = os.path.join(rename_dir, new_name)
        os.rename(old_name, new_name)

# %%
