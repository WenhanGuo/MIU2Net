import torch
from torchvision.transforms import functional as F
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
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


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

    def __call__(self, tensor):
        sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
        theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
        variance = (sigma_e**2 / 2) / (theta_G**2 * self.n_galaxy)
        std = np.sqrt(variance)
        return tensor + torch.randn(tensor.size()) * std + self.mean
        # for 50 galaxies per pix, std = 0.1951; 
        # for 20 galaxies per pix, std = 0.3085