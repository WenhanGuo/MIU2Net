import os
from glob import glob
import numpy as np
from astropy.table import Table

def glob_selected_z(directory, z):
    z_folder = os.path.join(directory, str(z), '*.fits')
    return np.array(sorted(glob(z_folder)))

def glob_data(directory, zlist):
    for z in zlist:
        fnames = glob_selected_z(directory, z)
        if z == zlist[0]:
            arr = fnames
        else:
            arr = np.vstack([arr, fnames])
    return arr

zlist = np.arange(1, 38)
t = Table()
t['gamma1'] = glob_data('/ksmap/map/gamma1', zlist=zlist).transpose()
t['gamma2'] = glob_data('/ksmap/map/gamma2', zlist=zlist).transpose()
t['kappa'] = glob_data('/ksmap/map/kappa', zlist=zlist).transpose()
t['halo'] = glob_data('/ksmap/ks/halomap', zlist=zlist).transpose()
t['density'] = glob_data('/ksmap/ks/density', zlist=zlist).transpose()

t[0:5000].write('/ksmap/train.ecsv', overwrite=True)
t[5000:6000].write('/ksmap/validation.ecsv', overwrite=True)
t[6032:6144].write('/ksmap/test.ecsv', overwrite=True)   # the first 32 test images are corrupted by mistake