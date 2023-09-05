import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import random
import numpy as np
from typing import Sequence


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


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

    def __call__(self, image):
        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * self.n_galaxy)
        std = np.sqrt(variance)
        image = image + torch.randn(image.size()) * std + self.mean
        return image
        # for 50 galaxies per pix, std = 0.1951; 
        # for 20 galaxies per pix, std = 0.3085


class KS_rec(object):
    """
    reconstruct kappa map from shear using Kaiser-Squires deconvolution.
    """
    def __init__(self, activate=False):
        self.activate = activate
    
    def shear_rec(self, shear1, shear2):
        N_grid = shear1.shape[0]
        theta = torch.linspace(-N_grid+1, N_grid-1, 2*N_grid-1, device=shear1.device)
        theta_x, theta_y = torch.meshgrid([theta, theta], indexing='ij')
        D_starkernel = -1. / (theta_x + 1j*theta_y) ** 2
        D_starkernel[N_grid-1, N_grid-1] = 0
        y = torch.fft.ifftn(torch.fft.fftn(D_starkernel, s=(3*N_grid-2, 3*N_grid-2)) * torch.fft.fftn(shear1 + 1j*shear2, s=(3*N_grid-2, 3*N_grid-2)))
        y = y.real / torch.tensor([np.pi], device=shear1.device)
        y = y[N_grid-1:2*N_grid-1, N_grid-1:2*N_grid-1]
        return y

    def __call__(self, image, target):
        if self.activate == True:
            # perdict kappa using KS and add it as a 3rd channel to gamma
            ks_kappa = self.shear_rec(-image[0], image[1])   # the negative sign is important
            image = torch.concat((image, ks_kappa.unsqueeze(0)), dim=0)
            return image, target
        # gamma shape = torch.Size([2, 512, 512]); kappa shape = torch.Size([1, 512, 512])
        # if ks: gamma shape = torch.Size([3, 512, 512]); last channel is ks map
        else:
            # do nothing
            return image, target