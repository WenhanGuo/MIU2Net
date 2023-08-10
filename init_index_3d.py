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
t['gamma1'] = sorted(glob('/share/lirui/Wenhan/WL/gamma1_cube/*.fits'))[0:2000]
t['gamma2'] = sorted(glob('/share/lirui/Wenhan/WL/gamma2_cube/*.fits'))[0:2000]
# t['kappa'] = glob_data('/ksmap/map/kappa', zlist=zlist)
t['halo'] = sorted(glob('/share/lirui/Wenhan/WL/halomap/*.fits'))[0:2000]
t['density'] = sorted(glob('/share/lirui/Wenhan/WL/density/*.mat'))[0:2000]

t[0:1800].write('/share/lirui/Wenhan/WL/train.ecsv', overwrite=True)
t[1800:2000].write('/share/lirui/Wenhan/WL/validation.ecsv', overwrite=True)
t[6000:6144].write('/share/lirui/Wenhan/WL/test.ecsv', overwrite=True)