import os
import multiprocessing
import numpy as np
from astropy.io import fits
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
import glob


def shape_noise(n_galaxy):
    sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
    theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
    variance = (sigma_e**2 / 2) / (theta_G**2 * n_galaxy)
    std = np.sqrt(variance)
    print('shape noise std =', std)
    return std


def save_cube(true, ml, ks, wiener, sparse, save_dir, save_name):
    true1 = np.expand_dims(true, axis=0)
    ml1 = np.expand_dims(ml, axis=0)
    ks1 = np.expand_dims(ks, axis=0)
    wiener1 = np.expand_dims(wiener, axis=0)
    sparse1 = np.expand_dims(sparse, axis=0)
    c = np.concatenate([true1, ml1, ks1, wiener1, sparse1])
    fits.writeto(os.path.join(save_dir, save_name), data=np.float32(c), overwrite=True)
    print(f'saved cube {save_name}')


def make_cube(args, fname):
    with fits.open(os.path.join(args.dir, fname)) as f:
        cube = f[0].data
        shear1 = cube[0]
        shear2 = cube[1]
        ks = cube[2]
        wiener = cube[3]
        pred = cube[4]
        true = cube[5]
        res = cube[6]

    # initialize CosmoStat shear class
    D = shear_data()
    D.g1 = - shear1
    D.g2 = shear2
    noise_std = shape_noise(n_galaxy=args.n_galaxy)
    CovMat = np.ones((512, 512)) * (noise_std**2)
    D.Ncov = CovMat
    D.nx, D.ny = 512, 512

    # create the mass mapping structure and initialize it
    M = massmap2d(name='mass')
    M.init_massmap(nx=512, ny=512)

    # sparse reconstruction with a 5 sigma detection
    sparse, ti = M.sparse_recons(D, UseNoiseRea=False, Nsigma=5, ThresCoarse=False, Inpaint=False, Nrea=None)

    save_cube(true, pred, ks, wiener, sparse, save_dir=args.save_dir, save_name=fname)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/kappa_map/result/prediction', type=str)
    parser.add_argument("--save-dir", default='/share/lirui/Wenhan/WL/kappa_map/result/master_cubes', type=str)
    parser.add_argument("--cpu", default=32, type=int, help='number of cpu cores to use for multiprocessing')
    parser.add_argument("--n-galaxy", default=50, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    fnames = glob.glob1(args.dir, '*.fits')
    with multiprocessing.Pool(processes=args.cpu) as pool:
        arguments = [(args, fname) for fname in fnames]
        pool.starmap(func=make_cube, iterable=arguments, chunksize=2)