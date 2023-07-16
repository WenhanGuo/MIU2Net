# %%
import torch
import os
import numpy as np
from scipy.fft import fft2, ifft2
from astropy.io import fits
from glob import glob
import astropy.io.fits as fits
import matplotlib.pyplot as plt


def shear_rec(shear1, shear2):
    N_grid = shear1.shape[0]
    theta = np.linspace(-N_grid+1, N_grid-1, 2*N_grid-1)
    theta_x, theta_y = np.meshgrid(theta, theta)
    D_starkernel = -1. / (theta_x + 1j*theta_y) ** 2
    D_starkernel[N_grid-1, N_grid-1] = 0
    y = np.real(ifft2(fft2(D_starkernel, (3*N_grid-2, 3*N_grid-2)) * fft2(shear1 + 1j*shear2, (3*N_grid-2, 3*N_grid-2)))) / np.pi
    y = y[N_grid-1:2*N_grid-1, N_grid-1:2*N_grid-1]
    return y

def calc_and_save(shear1, shear2, out_name):
    kappa = shear_rec(-shear1, shear2)
    fits.writeto(out_name, data=np.float32(kappa), overwrite=True)

def add_gaussian_noise(img, std):
    img = torch.tensor(np.float32(img))
    noise = torch.randn(img.size()) * std
    return np.array(img + noise)

# %%
gamma1_fnames = sorted(glob('/Users/danny/Desktop/WL/data_new/gamma1/*[2][4][0-9][0-9][0-9].fits'))
gamma2_fnames = sorted(glob('/Users/danny/Desktop/WL/data_new/gamma2/*[2][4][0-9][0-9][0-9].fits'))
out_dir = '/Users/danny/Desktop/WL/data_new/kappa_ks'
std = 0.1951

for i in range(len(gamma1_fnames)):
    gamma1 = fits.open(gamma1_fnames[i])[0].data
    n_gamma1 = add_gaussian_noise(gamma1, std=std)
    gamma2 = fits.open(gamma2_fnames[i])[0].data
    n_gamma2 = add_gaussian_noise(gamma2, std=std)

    kappa_name = os.path.join(out_dir, 'ksmap_' + str(np.char.zfill(str(i+24000), 5)) + '.fits')
    calc_and_save(n_gamma1, n_gamma2, out_name=kappa_name)

# %%
