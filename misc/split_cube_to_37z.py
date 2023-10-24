import os
from glob import glob
import numpy as np
from astropy.io import fits


def split_cube(path, out_dir):
    base_name = os.path.basename(path)
    with fits.open(path, memmap=False) as f:
        cube = np.float32(f[0].data)
        for i, img in enumerate(cube):
            idx = str(i+1)
            new_base_name = base_name[:-5]+'_'+idx+'.fits'   # change this for .mat (:-4)
            new_path = os.path.join(out_dir, idx, new_base_name)
            fits.writeto(new_path, data=img, overwrite=True)


if __name__ == '__main__':
    cube_paths = sorted(glob("/ksmap/map/gamma2_cube/*.fits"))
    for i in range(1, 38):
        os.mkdir('/ksmap/map/gamma2/'+str(i))

    for cube_path in cube_paths:
        split_cube(cube_path, out_dir='/ksmap/map/gamma2')

# -----------------------------------------------

# import h5py

# def split_cube(path, out_dir):
#     base_name = os.path.basename(path)
#     with h5py.File(path, 'r') as f:
#         variables = {}
#         for k, v in f.items():
#             variables[k] = np.array(v)
#     cube = np.float32(variables['Sigma_2D'])
#     for i, img in enumerate(cube):
#         idx = str(i+1)
#         new_base_name = base_name[:-4]+'_'+idx+'.fits'   # change this for .mat (:-4)
#         new_path = os.path.join(out_dir, idx, new_base_name)
#         fits.writeto(new_path, data=img, overwrite=True)


# if __name__ == '__main__':
#     cube_paths = sorted(glob("/ksmap/ks/density_cube/*.mat"))
#     for i in range(1, 38):
#         os.mkdir('/ksmap/ks/density/'+str(i))

#     for cube_path in cube_paths:
#         split_cube(cube_path, out_dir='/ksmap/ks/density')
