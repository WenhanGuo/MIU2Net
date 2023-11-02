import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import random
import numpy as np
from typing import Sequence
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
from astropy.io import fits
try:
    import pysparse
except ImportError:
    print(
        "Warning in transforms.py: do not find pysap bindings ==> use slow python code. "
    )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        if target:
            return F.to_tensor(image), F.to_tensor(target)
        else:
            return F.to_tensor(image)


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class DiscreteRotation(object):
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, image, target):
        angle = random.choice(self.angles)
        image = F.rotate(image, angle)
        target = F.rotate(target, angle)
        return image, target


class ContinuousRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle)
        target = F.rotate(target, angle)
        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


# add gaussian noise to shear
class AddGaussianNoise(object):
    def __init__(self, n_galaxy, mean=0.):
        """
        calculate the gaussian noise standard deviation to be added to shear.
        please refer to https://articles.adsabs.harvard.edu/pdf/2004MNRAS.350..893H Eq.12. 面积替换为方形的
        noise_std^2 = {sigma_e^2 / 2} / {θ_G^2 * n_galaxy}
        """
        self.n_galaxy = n_galaxy
        self.mean = mean
        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * self.n_galaxy)
        self.std = np.sqrt(variance)
        # print('shear noise std =', self.std)

    def __call__(self, image):
        # image = image + np.random.normal(loc=self.mean, scale=self.std, size=image.shape)
        image = image + torch.randn(image.size()) * self.std + self.mean
        return image
        # for 50 galaxies per arcmin^2, std = 0.1951; 
        # for 20 galaxies per arcmin^2, std = 0.3085


class KS_rec(object):
    """
    reconstruct kappa map from shear using Kaiser-Squires deconvolution.
    """
    def __init__(self, args):
        self.activate = args.ks
        self.M = massmap2d(name='mass_ks')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=1024, ny=1024, pass_class=[self.psWT_gen1, self.psWT_gen2])
        print('KS initialized')

    def shear_rec(self, shear1, shear2):
        ks =  self.M.g2k(shear1, shear2, pass_class=[self.psWT_gen1, self.psWT_gen2])
        return ks

    def __call__(self, image, target):
        if self.activate == 'off':
            return image, target
        
        elif self.activate == 'add':
            # perdict kappa using KS and add it as a 3rd channel to gamma
            # if ks add: image shape = torch.Size([3, 512, 512]); last channel is ks map
            ks_kappa = self.shear_rec(-image[0], image[1])   # negative sign is important
            ks_kappa = torch.FloatTensor(ks_kappa)
            image = torch.concat((image, ks_kappa.unsqueeze(0)), dim=0)
            return image, target
            
        elif self.activate == 'only':
            # perdict kappa using KS and remove shear information
            # if ks only: image shape = torch.Size([1, 512, 512])
            ks_kappa = self.shear_rec(-image[0], image[1])   # negative sign is important
            ks_kappa = torch.FloatTensor(ks_kappa)
            image = ks_kappa.unsqueeze(0)
            # ks_kappa = np.float32(ks_kappa)
            # image = np.expand_dims(ks_kappa, axis=0)
            return image, target


class Wiener(object):
    """
    reconstruct kappa map from shear using Wiener filtering.
    """
    def __init__(self, args):
        self.activate = args.wiener
        self.p_signal = fits.open('./signal_power_spectrum.fits')[0].data
        if args.n_galaxy == 50:
            self.p_noise = fits.open('./noise_power_spectrum_g50.fits')[0].data
        elif args.n_galaxy == 20:
            self.p_noise = fits.open('./noise_power_spectrum_g20.fits')[0].data
        # Create the cosmostat mass mapping structure and initialize it
        self.M = massmap2d(name='mass_wiener')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=512, ny=512, pass_class=[self.psWT_gen1, self.psWT_gen2])
        print('wiener initialized')

    def wiener(self, shear1, shear2):
        retr, reti = self.M.wiener(shear1, shear2, 
                                   PowSpecSignal=self.p_signal, 
                                   PowSpecNoise=self.p_noise, 
                                   pass_class=[self.psWT_gen1, self.psWT_gen2])
        return retr, reti

    def __call__(self, image, target):
        if self.activate == 'off':
            return image, target
        
        elif self.activate == 'add':
            wf_kappa, _ = self.wiener(-image[0], image[1])   # negative sign is important
            wf_kappa = np.float32(wf_kappa)
            image = np.concatenate([image, np.expand_dims(wf_kappa, axis=0)], axis=0)
            return image, target
        
        elif self.activate == 'only':
            wf_kappa, _ = self.wiener(-image[0], image[1])   # negative sign is important
            wf_kappa = np.float32(wf_kappa)
            # image = wf_kappa.unsqueeze(0)
            image = np.expand_dims(wf_kappa, axis=0)
            return image, target


class sparse(object):
    """
    reconstruct kappa map from shear using sparse reconstruction.
    """
    def __init__(self, args):
        self.activate = args.sparse
        self.M = massmap2d(name='mass_sparse')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=512, ny=512, pass_class=[self.psWT_gen1, self.psWT_gen2])
        self.D = shear_data()

        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * args.n_galaxy)
        std = np.sqrt(variance)

        # Create the covariance matrix, assumed to be diagonal
        CovMat = np.ones((512, 512)) * (std**2)
        self.D.Ncov = CovMat

        print('sparse initialized')

    def sparse(self, shear1, shear2):
        self.D.g1 = shear1
        self.D.g2 = shear2
        # Do a sparse reconstruction with a 5 sigma detection
        ksr5, ti = self.M.sparse_recons(InshearData=self.D, 
                                        UseNoiseRea=False, 
                                        niter=12, 
                                        Nsigma=5, 
                                        ThresCoarse=False, 
                                        Inpaint=False, 
                                        pass_class=[self.psWT_gen1, self.psWT_gen2])
        return ksr5, ti

    def __call__(self, image, target):
        if self.activate == 'off':
            return image, target
        
        elif self.activate == 'add':
            sp_kappa, _ = self.sparse(np.float32(-image[0]), np.float32(image[1]))   # negative sign is important
            sp_kappa = np.float32(sp_kappa)
            image = np.concatenate([image, np.expand_dims(sp_kappa, axis=0)], axis=0)
            return image, target


class MCALens(object):
    """
    reconstruct kappa map from shear using MCALens reconstruction.
    """
    def __init__(self, args):
        self.activate = args.mcalens
        self.p_signal = fits.open('./signal_power_spectrum.fits')[0].data
        self.M = massmap2d(name='mass_mcalens')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=512, ny=512, pass_class=[self.psWT_gen1, self.psWT_gen2])
        self.D = shear_data()

        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * args.n_galaxy)
        std = np.sqrt(variance)

        # Create the covariance matrix, assumed to be diagonal
        CovMat = np.ones((512, 512)) * (std**2)
        self.D.Ncov = CovMat

        print('MCALens initialized')

    def mcalens(self, shear1, shear2):
        self.D.g1 = shear1
        self.D.g2 = shear2
        # MCAlens reconstruction with a 5 sigma detection
        k1r5, k1i5, k2r5, k2i = self.M.sparse_wiener_filtering(InshearData=self.D, 
                                                               PowSpecSignal=self.p_signal,
                                                               niter=12, 
                                                               Nsigma=5, 
                                                               Inpaint=False, 
                                                               Bmode=False, 
                                                               ktr=None, 
                                                               pass_class=[self.psWT_gen1, self.psWT_gen2])
        return k1r5, k1i5, k2r5, k2i

    def __call__(self, image, target):
        if self.activate == 'off':
            return image, target
        
        elif self.activate == 'add':
            mca_kappa, _, _, _ = self.mcalens(np.float32(-image[0]), np.float32(image[1]))   # negative sign is important
            # mca_kappa = torch.FloatTensor(mca_kappa)
            mca_kappa = np.float32(mca_kappa)
            # image = torch.concat((image, mca_kappa.unsqueeze(0)), dim=0)
            image = np.concatenate([image, np.expand_dims(mca_kappa, axis=0)], axis=0)
            return image, target