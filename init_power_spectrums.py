# %%
# only works on MacOS where sparse2d is installed
# generate noise and signal power spectra for Wiener filtering
import numpy as np
from my_cosmostat.astro.wl.mass_mapping import massmap2d
from my_cosmostat.misc.im_isospec import im_isospec
from astropy.io import fits
import pandas as pd
import os

# Create the covariance matrix, assumed to be diagonal
CovMat = np.ones((512, 512)) * (0.1951**2)   # std = 0.1951 for 50 galaxies per arcmin^2

# Create the mass mapping structure and initialise it
M = massmap2d(name='mass')
M.init_massmap(nx=512, ny=512)

p_noise = M.get_noise_powspec(CovMat=CovMat, nsimu=1000)
fits.writeto('noise_power_spectrum.fits', data=p_noise, overwrite=False)

# %%
catalog = '/Users/danny/Desktop/WL/data_512/train.csv'
kappa_names = pd.read_csv(catalog)['kappa'][:5000]

for i in range(len(kappa_names)):
    base_name = kappa_names[i]
    path = os.path.join('/Users/danny/Desktop/WL/data_512/kappa', base_name)
    with fits.open(path, memmap=False) as f:
        kappa = np.float32(f[0].data)
    p = im_isospec(kappa)   # get 1d power spectrum
    if i == 0:
        Np = p.shape[0]
        TabP = np.zeros([len(kappa_names), Np], dtype=float)
    else:
        TabP[i, :] = p
    if i % 1000 == 0:
        print(f'{i}th kappa loaded')
p_signal = np.mean(TabP, axis=0)
fits.writeto('signal_power_spectrum.fits', data=p_signal, overwrite=False)

# %%
