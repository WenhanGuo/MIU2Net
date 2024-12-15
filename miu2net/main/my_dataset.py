from torch.utils.data import Dataset
import os
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


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
        self.transforms = transforms
        self.reduced_shear = args.reduced_shear
        self.mask_frac = args.mask_frac
        self.resize = args.resize
        self.save_noisy_shear = args.save_noisy_shear
        self.save_noisy_shear_dir = args.save_noisy_shear_dir
        self.wiener_res = args.wiener_res
        self.cosmo2 = args.cosmo2

    def __len__(self):
        return len(self.img_names)

    def read_img(self, idx, img_type=['gamma1', 'gamma2', 'kappa']):
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
        gamma = np.array([-gamma1, gamma2])  # negative sign is important
        gamma = np.moveaxis(gamma, 0, 2)
        kappa = np.expand_dims(kappa, 2)

        # apply transforms
        image, target = self.transforms(gamma, kappa)

        # if self.wiener_res == True:
        #     # target = true - wiener, assuming args.wiener == 'add' and image[2] is wiener
        #     target = target - image[2]

        # save noisy shear data
        if self.save_noisy_shear == True:
            if self.cosmo2:
                save_name = os.path.basename(g1name)[:-11] + 'noisy_shear.fits'  # for cosmology 2
            else:
                save_name = os.path.basename(g1name)[:-14] + 'noisy_shear.fits'
            save_path = os.path.join(self.save_noisy_shear_dir, save_name)
            fits.writeto(save_path, np.float32(image), overwrite=True)

        # gamma shape = torch.Size([2, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3, 512, 512]); last channel is ks map
        return image, target


'''
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