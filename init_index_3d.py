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

zlist = [8, 15, 22, 29, 37]
t = Table()
t['gamma1'] = glob_data('../data_3d/gamma1', zlist=zlist).transpose()
t['gamma2'] = glob_data('../data_3d/gamma2', zlist=zlist).transpose()
t['kappa'] = glob_data('../data_3d/kappa', zlist=zlist).transpose()

t[0:2500].write('../data_3d/train.ecsv', overwrite=True)
t[2500:3000].write('../data_3d/validation.ecsv', overwrite=True)
t[3000:3072].write('../data_3d/test.ecsv', overwrite=True)