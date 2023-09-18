import os
from glob import glob
import numpy as np
from astropy.table import Table

def glob_selected_z(directory, z):
    z_folder = os.path.join(directory, str(z), '*.fits')
    return np.array(sorted(glob(z_folder)))

def glob_data(directory, zlist):
    if len(zlist) == 1:
        fnames = glob_selected_z(directory, zlist[0])
        return np.array([[fname] for fname in fnames]).transpose()
    else:
        for z in zlist:
            fnames = glob_selected_z(directory, z)
            if z == zlist[0]:
                arr = fnames
            else:
                arr = np.vstack([arr, fnames])
        return arr

# zlist = np.arange(1, 38)
zlist = [37]
t = Table()
t['gamma1'] = glob_data('/share/lirui/Wenhan/WL/data_1024_2d/gamma1', zlist=zlist).transpose()
t['gamma2'] = glob_data('/share/lirui/Wenhan/WL/data_1024_2d/gamma2', zlist=zlist).transpose()
t['kappa'] = glob_data('/share/lirui/Wenhan/WL/data_1024_2d/kappa', zlist=zlist).transpose()
# t['halo'] = glob_data('/share/lirui/Wenhan/WL/data_1024_2d/halomap', zlist=zlist).transpose()
# t['density'] = glob_data('/share/lirui/Wenhan/WL/data_1024_2d/density', zlist=zlist).transpose()

t[0:5000].write('/share/lirui/Wenhan/WL/data_1024_2d/train.ecsv', overwrite=True)
t[5000:6000].write('/share/lirui/Wenhan/WL/data_1024_2d/validation.ecsv', overwrite=True)
t[6032:6144].write('/share/lirui/Wenhan/WL/data_1024_2d/test.ecsv', overwrite=True)   # the first 32 test images are corrupted by mistake