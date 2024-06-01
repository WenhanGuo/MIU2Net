import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import random
import numpy as np
from typing import Sequence
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
from astropy.io import fits
from astropy.modeling.models import Disk2D
from astropy.table import QTable
from photutils.datasets import make_model_sources_image
try:
    import pysparse
except ImportError:
    print(
        "Warning in transforms.py: do not find pysap bindings ==> use slow python code. "
    )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), F.to_tensor(target)


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


class CenterCrop(object):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        image = F.resize(image, size=self.size, antialias=True)
        target = F.resize(target, size=self.size, antialias=True)
        return image, target


# add gaussian noise to shear
class AddGaussianNoise(object):
    def __init__(self, args):
        """
        calculate the gaussian noise standard deviation to be added to shear.
        please refer to https://articles.adsabs.harvard.edu/pdf/2004MNRAS.350..893H Eq.12. 面积替换为方形的
        noise_std^2 = {sigma_e^2 / 2} / {θ_G^2 * n_galaxy}
        """
        self.n_galaxy = args.n_galaxy
        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 3.5*60/(1024/args.crop)/args.crop   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * self.n_galaxy)
        self.std = np.sqrt(variance)
        self.noise_seed = args.noise_seed
        # for 50 galaxies per arcmin^2, std = 0.1951; 
        # for 20 galaxies per arcmin^2, std = 0.3085

    def __call__(self, image, target):
        if self.noise_seed == 0:
            noise = torch.randn(image.shape) * self.std
        else:
            assert type(self.noise_seed) == int
            # set random seed by the top left pixel of gamma1
            seed = int(image[0][0][0] * 1e9)
            # construct list of [1, 2, ..., n_realizations] for seed modification
            mod_seed = [*range(1, self.noise_seed+1, 1)]
            # round seed to nearest 10, then add a random choice of modification with equal prob for each
            seed = round(seed, ndigits=-1) + random.choice(mod_seed)
            g = torch.Generator()  # pseudo random generator
            noise = torch.randn(image.shape, generator=g.manual_seed(seed)) * self.std
        return image + noise, target


# add bright star masks to shear
class AddStarMask(object):
    def __init__(self, args, fill_value=0):
        self.size = args.resize
        self.n_sources = int(1100 * args.mask_frac)
        self.model = Disk2D()
        self.rng = np.random.default_rng(seed=None)
        self.fill_value = fill_value

    def __call__(self, image, target):
        sources = QTable()
        sources['amplitude'] = np.ones(self.n_sources)
        sources['x_0'] = self.rng.uniform(low=0, high=self.size, size=self.n_sources)
        sources['y_0'] = self.rng.uniform(low=0, high=self.size, size=self.n_sources)
        sources['R_0'] = self.rng.power(a=0.3, size=self.n_sources) * 13
        data = make_model_sources_image(shape=(self.size, self.size), 
                                        model=self.model, 
                                        source_table=sources)
        mask = torch.tensor((data < 1)).repeat(image.shape[0], 1, 1)
        image[mask == 0] = self.fill_value
        return image, target


# use reduced shear (g) instead of shear (gamma)
class ReducedShear(object):
    def __init__(self, args):
        self.mode = args.reduced_shear
    
    def __call__(self, image, target):
        if self.mode == True:
            image = image / (1 - target)  # gamma = gamma / (1 - kappa)
            return image, target
        return image, target


class KS_rec(object):
    """
    reconstruct kappa map from shear using Kaiser-Squires deconvolution.
    """
    def __init__(self, args):
        self.mode = args.ks
        self.M = massmap2d(name='mass_ks')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=1024, ny=1024, pass_class=[self.psWT_gen1, self.psWT_gen2])

    def shear_rec(self, shear1, shear2):
        ks =  self.M.g2k(shear1, shear2, pass_class=[self.psWT_gen1, self.psWT_gen2])
        return ks

    def __call__(self, image, target):
        if self.mode == 'off':
            return image, target
        
        elif self.mode == 'add':
            # perdict kappa using KS and add it as a 3rd channel to gamma
            # if ks add: image shape = torch.Size([3, 512, 512]); last channel is ks map
            ks_kappa = self.shear_rec(image[0], image[1])   # negative sign is important
            ks_kappa = torch.FloatTensor(ks_kappa)
            image = torch.concat((image, ks_kappa.unsqueeze(0)), dim=0)
            return image, target
            
        elif self.mode == 'only':
            # perdict kappa using KS and remove shear information
            # if ks only: image shape = torch.Size([1, 512, 512])
            ks_kappa = self.shear_rec(image[0], image[1])   # negative sign is important
            ks_kappa = torch.FloatTensor(ks_kappa)
            image = ks_kappa.unsqueeze(0)
            return image, target


class Wiener(object):
    """
    reconstruct kappa map from shear using Wiener filtering.
    """
    def __init__(self, args):
        self.mode = args.wiener
        self.p_signal = fits.open('./pspec/signal_power_spectrum.fits')[0].data
        if args.n_galaxy == 50:
            self.p_noise = fits.open('./pspec/noise_power_spectrum_g50.fits')[0].data
        elif args.n_galaxy == 20:
            self.p_noise = fits.open('./pspec/noise_power_spectrum_g20.fits')[0].data
        # Create the cosmostat mass mapping structure and initialize it
        self.M = massmap2d(name='mass_wiener')
        self.psWT_gen1 = pysparse.MRStarlet(bord=1, gen2=False, nb_procs=1, verbose=0)
        self.psWT_gen2 = pysparse.MRStarlet(bord=1, gen2=True, nb_procs=1, verbose=0)
        self.M.init_massmap(nx=512, ny=512, pass_class=[self.psWT_gen1, self.psWT_gen2])

    def wiener(self, shear1, shear2):
        retr, reti = self.M.wiener(shear1, shear2, 
                                   PowSpecSignal=self.p_signal, 
                                   PowSpecNoise=self.p_noise, 
                                   pass_class=[self.psWT_gen1, self.psWT_gen2])
        return retr, reti

    def __call__(self, image, target):
        if self.mode == 'off':
            return image, target
        
        elif self.mode == 'add':
            wf_kappa, _ = self.wiener(image[0], image[1])   # negative sign is important
            wf_kappa = torch.FloatTensor(wf_kappa)
            image = torch.concat((image, wf_kappa.unsqueeze(0)), dim=0)
            return image, target
        
        elif self.mode == 'only':        
            wf_kappa, _ = self.wiener(image[0], image[1])   # negative sign is important
            wf_kappa = torch.FloatTensor(wf_kappa)
            image = wf_kappa.unsqueeze(0)
            return image, target


class sparse(object):
    """
    reconstruct kappa map from shear using sparse reconstruction.
    """
    def __init__(self, args):
        self.mode = args.sparse
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
        if self.mode == 'off':
            return image, target
        
        elif self.mode == 'add':
            sp_kappa, _ = self.sparse(np.float32(image[0]), np.float32(image[1]))   # negative sign is important
            sp_kappa = np.float32(sp_kappa)
            image = np.concatenate([image, np.expand_dims(sp_kappa, axis=0)], axis=0)
            return image, target


class MCALens(object):
    """
    reconstruct kappa map from shear using MCALens reconstruction.
    """
    def __init__(self, args):
        self.mode = args.mcalens
        self.p_signal = fits.open('./pspec/signal_power_spectrum.fits')[0].data
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
        if self.mode == 'off':
            return image, target
        
        elif self.mode == 'add':
            mca_kappa, _, _, _ = self.mcalens(np.float32(image[0]), np.float32(image[1]))   # negative sign is important
            # mca_kappa = torch.FloatTensor(mca_kappa)
            mca_kappa = np.float32(mca_kappa)
            # image = torch.concat((image, mca_kappa.unsqueeze(0)), dim=0)
            image = np.concatenate([image, np.expand_dims(mca_kappa, axis=0)], axis=0)
            return image, target