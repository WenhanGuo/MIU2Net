import torch
from torch.utils.data import Dataset
import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import resize

import os
import numpy as np
import pandas as pd

import astropy.io.fits as fits
from astropy.table import Table
from astropy.constants import G
from astropy.cosmology import WMAP9
import astropy.units as u
import astropy.cosmology.units as cu



# Dataset class for our own data structure
class ImageDataset(Dataset):
    def __init__(self, args, catalog, transforms):
        """
        catalog: name of .csv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transforms: equivariant transformations to input data (gamma) and target (kappa) prior to training
        gaus_noise: adding gaussian noise to input data (gamma) prior to training 
        """
        self.img_names = Table.read(catalog)
        self.n_galaxy = args.n_galaxy
        self.transforms = transforms
        self.gaus_blur = args.gaus_blur
        self.resize = args.resize
        self.save_noisy_shear = args.save_noisy_shear
        self.save_noisy_shear_dir = args.save_noisy_shear_dir
        self.wiener_res = args.wiener_res
    
    def __len__(self):
        return len(self.img_names)
    
    def read_img(self, idx, img_type=['gamma1', 'gamma2', 'kappa']):
        # img_name = self.img_names[img_type][idx][2:-2]   # bypass strange indexing of '["img_name"]' in py3.6 env
        img_name = self.img_names[img_type][idx][0]   # get single image name
        with fits.open(img_name, memmap=False) as f:
            img = f[0].data   # read image into numpy array
        return np.float32(img), img_name   # force apply float32 to resolve endian conflict
    
    def __getitem__(self, idx):
        # read in images
        gamma1, g1name = self.read_img(idx, img_type='gamma1')
        gamma2, _ = self.read_img(idx, img_type='gamma2')
        kappa, _ = self.read_img(idx, img_type='kappa')

        # reformat data shapes
        gamma = np.array([gamma1, gamma2])
        gamma = np.moveaxis(gamma, 0, 2)
        kappa = np.expand_dims(kappa, 2)
        
        # add gaussian noise to shear only
        tt = T.ToTensor()
        gn = T.AddGaussianNoise(n_galaxy=self.n_galaxy)
        image = gn(tt(gamma))
        target = tt(kappa)

        # apply transforms
        image, target = self.transforms(image, target)
        if self.wiener_res == True:
            # target = true - wiener, assuming args.wiener == 'add' and image[2] is wiener
            target = target - image[2]

        # save noisy shear data
        if self.save_noisy_shear == True:
            save_name = os.path.basename(g1name)[:-14] + 'noisy_shear.fits'
            save_path = os.path.join(self.save_noisy_shear_dir, save_name)
            fits.writeto(save_path, np.float32(image), overwrite=True)

        if self.gaus_blur == True:
            gb = GaussianBlur(kernel_size=5, sigma=2.0)
            image = gb(image)
        
        image = resize(image, size=self.resize)
        target = resize(target, size=self.resize)
        
        # gamma shape = torch.Size([2, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3, 512, 512]); last channel is ks map
        return image, target



class ImageDataset_kappa3d(Dataset):
    def __init__(self, args, catalog, z_cat, shear_zslices, kappa_zslices, transforms, gaus_blur=None):
        """
        catalog: name of .ecsv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transforms: equivariant transformations to input data (gamma) and target (kappa) prior to training
        gaus_noise: adding gaussian noise to input data (gamma) prior to training 
        """
        self.catalog = Table.read(catalog)
        z_cat = pd.read_csv(z_cat, sep=' ')
        self.z_list = np.array(z_cat['z_lens'])
        self.n_galaxy = args.n_galaxy
        self.shear_zslices = shear_zslices
        self.kappa_zslices = kappa_zslices
        self.transforms = transforms
        self.gaus_blur = gaus_blur
        self.save_noisy_shear = args.save_noisy_shear
        self.save_noisy_shear_dir = args.save_noisy_shear_dir
        self.wiener_res = args.wiener_res
    
    def __len__(self):
        return len(self.catalog)

    def read_data(self, idx, img_type=['gamma1', 'gamma2', 'kappa', 'halo', 'density']):
        if img_type == 'kappa':
            img_names = self.catalog[img_type][idx][self.kappa_zslices]
        elif img_type in ['gamma1', 'gamma2']:
            img_names = self.catalog[img_type][idx][self.shear_zslices]
        
        cube = None
        for img_name in img_names:
            with fits.open(img_name, memmap=False) as f:
                img = np.expand_dims(f[0].data, axis=-1)
                cube = np.concatenate([cube, img], axis=-1) if cube is not None else img
        return np.float32(cube), img_name   # force apply float32 to resolve endian conflict
    

    def __getitem__(self, idx):
        # read in images
        gamma1, g1name = self.read_data(idx, img_type='gamma1')
        gamma2, _ = self.read_data(idx, img_type='gamma2')
        kappa, _ = self.read_data(idx, img_type='kappa')

        # assemble image cube and target cube
        tt = T.ToTensor()
        gn = T.AddGaussianNoise(n_galaxy=self.n_galaxy)
        gamma1, gamma2 = gn(tt(gamma1)), gn(tt(gamma2))
        image = torch.concat([gamma1, gamma2], dim=0)
        target = tt(kappa)

        # save noisy shear data
        if self.save_noisy_shear == True:
            save_name = os.path.basename(g1name)[:-14] + 'noisy_shear.fits'
            save_path = os.path.join(self.save_noisy_shear_dir, save_name)
            noisy_shear = F.center_crop(img=image, output_size=512)
            fits.writeto(save_path, np.float32(noisy_shear), overwrite=True)
        
        # apply transforms
        image, target = self.transforms(image, target)
        if self.gaus_blur:
            target = self.gaus_blur(target)
        
        if self.wiener_res == True:
            # target = true - wiener, assuming args.wiener == 'add' and image[2] is wiener
            target = target - image[2]
        
        # gamma shape = torch.Size([2x, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3x, 512, 512]); last channel is ks map
        return image, target



'''
class ImageDataset_density(Dataset):
    def __init__(self, catalog, z_cat, n_galaxy, lens_zslices, src_zslices, transforms, gaus_blur=None):
        """
        catalog: name of .ecsv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transforms: equivariant transformations to input data (gamma) and target (kappa) prior to training
        gaus_noise: adding gaussian noise to input data (gamma) prior to training 
        """
        self.catalog = Table.read(catalog)
        z_cat = pd.read_csv(z_cat, sep=' ')
        self.z_list = np.array(z_cat['z_lens'])
        self.n_galaxy = n_galaxy
        self.lens_zslices = lens_zslices
        self.src_zslices = src_zslices
        self.transforms = transforms
        self.gaus_blur = gaus_blur
        self.Sigma_mean = np.float32(self.mean_density())
    
    def __len__(self):
        return len(self.catalog)

    def read_data(self, idx, img_type=['gamma1', 'gamma2', 'kappa', 'halo', 'density']):
        if img_type in ['gamma1', 'gamma2', 'kappa']:
            img_names = self.catalog[img_type][idx][self.src_zslices]
        elif img_type in ['halo', 'density']:
            img_names = self.catalog[img_type][idx][self.lens_zslices]
        # img_names = self.catalog[img_type][idx][self.lens_zslices]
        
        cube = None
        for img_name in img_names:
            with fits.open(img_name, memmap=False) as f:
                img = np.expand_dims(f[0].data, axis=-1)
                cube = np.concatenate([cube, img], axis=-1) if cube is not None else img
        return np.float32(cube)   # force apply float32 to resolve endian conflict
    
    def mean_density(self):
        zs = self.z_list[self.lens_zslices]   # list of z values for the given slice indexes
        d_list = (zs*cu.redshift).to(u.Mpc, cu.with_redshift(WMAP9))   # convert z to distance (Mpc)
        d_list = np.array(d_list.value)
        arcmin2kpc = (1/60/180*np.pi) * d_list * 1000   # for a given distance, how many kpc is a arcmin

        H = WMAP9.H(z=zs)   # Hubble constant for given list of z values
        rho_crit = (3 * H**2) / (8 * np.pi * G)   # critical volume densities
        rho_crit = rho_crit.to(u.Msun / u.kpc**3)   # volume density, unit = M☉/kpc^3
        # integrate volume density along the depth of a slice (64 Mpc)
        Sigma_crit = rho_crit * 64000 * u.kpc   # surface density, unit = M☉/kpc^2

        return Sigma_crit.value * arcmin2kpc**2   # mean surface density for the slice

    def __getitem__(self, idx):
        tt = T.ToTensor()
        gn = T.AddGaussianNoise(n_galaxy=self.n_galaxy)
        # read in images
        gamma1 = self.read_data(idx, img_type='gamma1')
        print('gamma1 shape before tt =', gamma1.shape)
        gamma2 = self.read_data(idx, img_type='gamma2')
        gamma1, gamma2 = gn(tt(gamma1)), gn(tt(gamma2))
        print('gamma1 shape after tt =', gamma1.shape)
        # kappa = self.read_data(idx, img_type='kappa')
        # kappa = tt(kappa)

        halo = self.read_data(idx, img_type='halo')
        density = self.read_data(idx, img_type='density')
        halo = torch.log10(tt(halo))
        density = tt((density + self.Sigma_mean) / self.Sigma_mean)
        # print('halo max, min =', torch.max(halo), torch.min(halo))
        # print('density max, min =', torch.max(density), torch.min(density))

        # assemble image cube and target cube
        # image = halo
        image = torch.concat([gamma1, gamma2, halo], dim=0)
        print('image shape =', image.shape)
        # image = torch.concat([kappa, halo], dim=0)
        target = density
        print('target shape =', target.shape)

        # apply transforms
        image, target = self.transforms(image, target)
        if self.gaus_blur:
            target = self.gaus_blur(target)
        
        # gamma shape = torch.Size([2, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3, 512, 512]); last channel is ks map
        return image, target
'''