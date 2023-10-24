# %%
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table
from astropy.cosmology import WMAP9
import astropy.units as u
import astropy.cosmology.units as cu
import os

z_cat = pd.read_csv('/share/lirui/Wenhan/WL/kappa_map/scripts/redshift_info.txt', sep=' ')
z_lens = np.array(z_cat['z_lens'])
z_src = np.array(z_cat['z_source'])

def z_to_D(z_list):
    z_list = z_list * cu.redshift
    d_list = z_list.to(u.Mpc, cu.with_redshift(WMAP9)).value
    return np.array(d_list)

d_lens = z_to_D(z_lens)
d_src = z_to_D(z_src)

def halo_weights(d_lens, src_distance=3363.07062107):
    """
    d_lens: array of lens slices distances in Mpc
    src_distance: source slice distance in Mpc
    source z = 1 corresponds to 3363.07062107 Mpc
    """
    D_d = d_lens
    D_ds = src_distance - d_lens
    D_s = src_distance
    return (D_d * D_ds) / D_s

w_halo = halo_weights(d_lens=d_lens, src_distance=d_src[-1])
w_halo = w_halo / np.sum(w_halo)
print('halo weights =', w_halo)
w_halo = np.reshape(w_halo, newshape=(1, 1, 37))

# %%
catalog = Table.read('/ksmap/test.ecsv')
halo_cat = catalog['halo']

for idx in range(len(halo_cat)):
    img_paths = halo_cat[idx]   # 37 image names for 1 cube
    cube = None
    for img_path in img_paths:
        with fits.open(img_path, memmap=False) as f:
            img = np.expand_dims(f[0].data, axis=-1)
            cube = np.concatenate([cube, img], axis=-1) if cube is not None else img
    cube = np.float32(cube)   # force apply float32 to resolve endian conflict
    cube_w = cube * w_halo   # apply halo weights to 37 slices
    cumu_img = np.sum(cube_w, axis=-1)   # sum all 37 slices to get cumulative halomap
    cumu_img = np.float32(cumu_img)

    img_name = os.path.basename(img_path)
    cumu_name = img_name.replace('_37.fits', '.fits')
    fits.writeto('/share/lirui/Wenhan/WL/data_1024_2d/halomap/37/'+cumu_name, data=cumu_img, overwrite=False)
    print(idx)

# %%
