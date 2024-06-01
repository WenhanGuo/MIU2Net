import torch
from torchvision.transforms.functional import resize
import os
import multiprocessing
import numpy as np
from astropy.io import fits
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
import glob
try:
    import pysparse
except ImportError:
    print(
        "Warning in make_master_cubes_multiproc.py: do not find pysap bindings ==> use slow python code. "
    )

def shape_noise(n_galaxy):
    sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
    theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
    variance = (sigma_e**2 / 2) / (theta_G**2 * n_galaxy)
    std = np.sqrt(variance)
    print('shape noise std =', std)
    return std

def downsample(image, size):
    image = torch.tensor(np.float32(image))
    image = resize(image.unsqueeze(0), size=size, antialias=True)
    return image.numpy().astype(np.float32)[0]

def save_cube(true, ml, ks, wiener, sparse, mcalens, save_dir, save_name):
    true1 = np.expand_dims(true, axis=0)
    ml1 = np.expand_dims(ml, axis=0)
    ks1 = np.expand_dims(ks, axis=0)
    wiener1 = np.expand_dims(wiener, axis=0)
    sparse1 = np.expand_dims(sparse, axis=0)
    mcalens1 = np.expand_dims(mcalens, axis=0)
    c = np.concatenate([true1, ml1, ks1, wiener1, sparse1, mcalens1])
    fits.writeto(os.path.join(save_dir, save_name), data=np.float32(c), overwrite=True)
    print(f'saved cube {save_name}')


def make_cube(args, fname):
    basename = os.path.basename(fname)
    noisy_shear_name = basename[5:-13] + 'noisy_shear.fits'
    noisy_shear_path = os.path.join(args.shear_dir, noisy_shear_name)
    with fits.open(noisy_shear_path) as f:
        cube = f[0].data
        shear1 = cube[0]
        shear2 = cube[1]
    with fits.open(os.path.join(args.dir, fname)) as f:
        cube = f[0].data
        pred = cube[args.pred_id]
        true = cube[args.true_id]

    # initialize CosmoStat shear class
    D = shear_data()
    # D.g1 = np.float32(-shear1)   # negative sign is important
    D.g1 = np.float32(shear1)   # after fixing my_dataset
    D.g2 = np.float32(shear2)
    noise_std = shape_noise(n_galaxy=args.n_galaxy)
    CovMat = np.ones((args.crop, args.crop)) * (noise_std**2)
    D.Ncov = downsample(CovMat, size=args.resize)
    D.nx, D.ny = args.resize, args.resize

    # create the mass mapping structure and initialize it
    M = massmap2d(name='mass')
    psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
    psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
    M.init_massmap(nx=args.resize, ny=args.resize, pass_class=[psWT_gen1, psWT_gen2])

    p_signal = fits.open('./pspec/signal_power_spectrum.fits')[0].data
    if args.n_galaxy == 1059:
        p_noise = fits.open('./pspec/noise_power_spectrum_g1059.fits')[0].data
    if args.n_galaxy == 50:
        p_noise = fits.open('./pspec/noise_power_spectrum_g50_256.fits')[0].data
        # p_noise = fits.open('./pspec/noise_power_spectrum_g50.fits')[0].data
    elif args.n_galaxy == 30:
        p_noise = fits.open('./pspec/noise_power_spectrum_g30.fits')[0].data
    elif args.n_galaxy == 20:
        p_noise = fits.open('./pspec/noise_power_spectrum_g20.fits')[0].data
    elif args.n_galaxy == 5:
        p_noise = fits.open('./pspec/noise_power_spectrum_g5.fits')[0].data

    # ks reconstruction
    ks =  M.g2k(D.g1, D.g2, pass_class=[psWT_gen1, psWT_gen2])

    # wiener filtering
    wiener, reti = M.wiener(D.g1, D.g2, 
                            PowSpecSignal=p_signal, 
                            PowSpecNoise=p_noise, 
                            pass_class=[psWT_gen1, psWT_gen2])

    # sparse reconstruction with a 5 sigma detection
    sparse, ti = M.sparse_recons(InshearData=D, 
                               UseNoiseRea=False, 
                               niter=12, 
                               Nsigma=5, 
                               ThresCoarse=False, 
                               Inpaint=False, 
                               pass_class=[psWT_gen1, psWT_gen2])

    # mcalens reconstruction
    mcalens, k1i5, k2r5, k2i = M.sparse_wiener_filtering(InshearData=D, 
                                                    PowSpecSignal=p_signal,
                                                    niter=12, 
                                                    Nsigma=5, 
                                                    Inpaint=False, 
                                                    Bmode=False, 
                                                    ktr=None, 
                                                    pass_class=[psWT_gen1, psWT_gen2])

    save_cube(true, pred, ks, wiener, sparse, mcalens, save_dir=args.save_dir, save_name=fname)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/kappa_map/result/prediction', type=str)
    parser.add_argument("--shear-dir", default='/share/lirui/Wenhan/WL/kappa_map/result/noisy_shear', type=str)
    parser.add_argument("--save-dir", default='/share/lirui/Wenhan/WL/kappa_map/result/master_cubes', type=str)
    parser.add_argument("--cpu", default=20, type=int, help='number of cpu cores to use for multiprocessing')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float)
    parser.add_argument("--pred-id", default=3, type=int, help='id number of prediction frame, starting from 0')
    parser.add_argument("--true-id", default=2, type=int, help='id number of true frame, starting from 0')
    parser.add_argument("--crop", default=512, type=int, help='kappa size to crop (stored shear size)')
    parser.add_argument("--resize", default=256, type=int, help='kappa size to downsample')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    fnames = glob.glob1(args.dir, '*.fits')
    with multiprocessing.Pool(processes=args.cpu) as pool:
        arguments = [(args, fname) for fname in fnames]
        pool.starmap(func=make_cube, iterable=arguments)