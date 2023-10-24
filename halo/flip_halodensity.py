import os
from glob import glob
import numpy as np
from astropy.io import fits

density_paths = sorted(glob("/ksmap/ks/density/*/*.fits"))
halo_paths = sorted(glob("/ksmap/ks/halomap/*/*.fits"))

for img_path in density_paths:
    img = fits.open(img_path)[0].data
    basename = os.path.basename(img_path)
    dirname = os.path.pardir
    fits.writeto(img_path, data=np.flipud(img), overwrite=True)

for img_path in halo_paths:
    img = fits.open(img_path)[0].data
    basename = os.path.basename(img_path)
    dirname = os.path.pardir
    fits.writeto(img_path, data=np.flipud(np.rot90(img, k=1)), overwrite=True)