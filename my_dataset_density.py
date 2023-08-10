import torch
from torch.utils.data import Dataset
import transforms as T
import numpy as np
import astropy.io.fits as fits
import h5py
from astropy.table import Table


# Dataset class for our own data structure
class ImageDataset(Dataset):
    def __init__(self, catalog, n_galaxy, lens_zslices, src_zslices, transforms, gaus_blur=None):
        """
        catalog: name of .csv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transforms: equivariant transformations to input data (gamma) and target (kappa) prior to training
        gaus_noise: adding gaussian noise to input data (gamma) prior to training 
        """
        self.catalog = Table.read(catalog)
        self.n_galaxy = n_galaxy
        self.lens_zslices = lens_zslices
        self.src_zslices = src_zslices
        self.transforms = transforms
        self.gaus_blur = gaus_blur
        self.counter = 0
    
    def __len__(self):
        return len(self.catalog)

    def read_lens(self, idx, img_type=['gamma1', 'gamma2', 'kappa']):
        cube_name = self.catalog[img_type][idx]
        with fits.open(cube_name, memmap=False) as f:
            cube = f[0].data[self.lens_zslices]
        return np.float32(cube)   # force apply float32 to resolve endian conflict

    def read_halo(self, idx):
        cube_name = self.catalog['halo'][idx]
        with fits.open(cube_name, memmap=False) as f:
            cube = f[0].data[self.src_zslices]
        return np.float32(cube)   # downgrade to float32

    def read_density(self, idx):
        mat_name = self.catalog['density'][idx]
        with h5py.File(mat_name, 'r') as f:
            variables = {}
            for k, v in f.items():
                variables[k] = np.array(v)
        cube = variables['Sigma_2D'][self.src_zslices]
        return np.float32(cube)   # downgrade to float32

    def __getitem__(self, idx):
        # print(self.counter)
        self.counter += 1

        # read in images
        gamma1 = self.read_lens(idx, img_type='gamma1')
        gamma2 = self.read_lens(idx, img_type='gamma2')
        tt = T.ToTensor()
        gn = T.AddGaussianNoise(n_galaxy=self.n_galaxy)
        gamma1, gamma2 = gn(tt(gamma1)), gn(tt(gamma2))
        # kappa = self.read_lens(idx, img_type='kappa')

        halo = self.read_halo(idx)
        density = self.read_density(idx)
        halo = torch.log10(tt(halo))
        density = tt(density)
        density = torch.log10(density + abs(torch.min(density)) + 1)
        # print('halo max, min =', torch.max(halo), torch.min(halo))
        # print('density max, min =', torch.max(density), torch.min(density))

        # assemble image cube and target cube
        image = torch.concat([gamma1, gamma2, halo], dim=0)
        target = density

        # apply transforms
        image, target = self.transforms(image, target)
        if self.gaus_blur:
            target = self.gaus_blur(target)
        
        # gamma shape = torch.Size([2, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3, 512, 512]); last channel is ks map
        return image, target