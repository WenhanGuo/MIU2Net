import torch
from torch.utils.data import Dataset
import transforms as T
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
                cube = np.stack([cube, f[0].data], axis=2) if cube is not None else f[0].data
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