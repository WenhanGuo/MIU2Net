# MIU<sup>2</sup>Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Weak Lensing Mass Inversion

MIU<sup>2</sup>Net stands for Mass Inversion U<sup>2</sup>Net. It uses deep learning to convert weak lensing shear ($\gamma$) maps to convergence ($\kappa$) maps, which trace the projected dark matter distribution. 

We develop MIU<sup>2</sup>Net as a deep learning framework for weak lensing mass inversion. MIU<sup>2</sup>Net includes observations effects like shape noise, reduced shear, and data masks in the training.


## Installation

The main MIU<sup>2</sup>Net package depends on the following packages:
- pytorch
- numpy
- scipy
- astropy

Prior to installing MIU<sup>2</sup>Net, make sure to [install PyTorch](https://pytorch.org/) according to your OS and compute platform. We recommend installing `pytorch 1.12.0` and `torchvision 0.13.0` to avoid unexpected dependency errors. The main MIU<sup>2</sup>Net package includes the full training and testing code for deep learning.

To reconstruct convergence maps using traditional (non- deep learning) methods, we have modified the [cosmostat](https://github.com/CosmoStat/cosmostat) package developed at the CosmoStat Lab in CEA Paris-Saclay, so that we can use traditional $\kappa$ map reconstructions alongside deep learning for comparison, or even use traditional methods during network training. Avaiilable reconstruction methods from cosmostat include:
- Kaiser-Squires (KS) deconvolution
- Wiener Filtering (WF)
- sparse reconstruction ([(Lanusse et al. 2016)](https://arxiv.org/abs/1603.01599), [Glimpse](https://github.com/CosmoStat/Glimpse/tree/v1.0)) 
- MCALens [Starck et al.](https://arxiv.org/abs/2102.04127)

To use these methods within MIU<sup>2</sup>Net, you should install [Sparse2D](https://github.com/CosmoStat/Sparse2D) developed by the CosmoStat Lab. This is not required by the deep learning framework. 


## MIU<sup>2</sup>Net train/test

To train a MIU<sup>2</sup>Net model:

```console
cd ./miu2net/main
python train.py xxxxxxxxxx
```

Testing MIU<sup>2</sup>Net model:

```console
cd ./miu2net/main
python pred.py xxxxxxxxxx
```