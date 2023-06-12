from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import astropy.io.fits as fits


# Dataset class for our own data structure
class ImageDataset(Dataset):
    def __init__(self, catalog, data_dir, transforms=None, gaus_noise=None):
        """
        catalog: name of .csv file containing image names to be read
        data_dir: path to data directory containing gamma1, gamma2, kappa folders
        transforms: equivariant transformations to input data (gamma) and target (kappa) prior to training
        gaus_noise: adding gaussian noise to input data (gamma) prior to training 
        """
        self.img_names = pd.read_csv(catalog)
        self.data_dir = data_dir
        self.transforms = transforms
        self.gaus_noise = gaus_noise
    
    def __len__(self):
        return len(self.img_names)
    
    def read_img(self, idx, img_type=['gamma1', 'gamma2', 'kappa']):
        img_name = self.img_names[img_type][idx]   # get single image name
        img_path = os.path.join(self.data_dir, img_type, img_name)
        with fits.open(img_path, memmap=False) as f:
            img = f[0].data   # read image into numpy array
        return np.float32(img)   # force apply float32 to resolve endian conflict
    
    def __getitem__(self, idx):
        # read in images
        gamma1 = self.read_img(idx, img_type='gamma1')
        gamma2 = self.read_img(idx, img_type='gamma2')
        kappa = self.read_img(idx, img_type='kappa')
        # reformat data shapes
        gamma = np.array([gamma1, gamma2])
        gamma = np.moveaxis(gamma, 0, 2)
        kappa = np.expand_dims(kappa, 2)
        # apply transforms
        if self.transforms:
            gamma, kappa = self.transforms(gamma, kappa)
        if self.gaus_noise:
            gamma = self.gaus_noise(gamma)
        
        # gamma shape: torch.Size([2, 512, 512]); kappa shape: torch.Size([1, 512, 512])
        return gamma, kappa