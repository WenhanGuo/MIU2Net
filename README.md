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
- sparse reconstruction ([Lanusse et al. 2016](https://arxiv.org/abs/1603.01599), [Glimpse](https://github.com/CosmoStat/Glimpse/tree/v1.0)) 
- MCALens ([Starck et al.](https://arxiv.org/abs/2102.04127))

To use these methods within MIU<sup>2</sup>Net, you should install [Sparse2D](https://github.com/CosmoStat/Sparse2D) developed by the CosmoStat Lab. This is not required by the deep learning framework. 


## Training MIU<sup>2</sup>Net

To train a MIU<sup>2</sup>Net model with noise corresponding to 20 galaxies per square arcmin, reduced shear, and 0-20% randomized masked pixels:
```console
cd ./miu2net/main
python -u train.py --gpu-ids 6,7 --cpu 32 -b 128 -g 20 -e 2000 --mixed-precision --load pretrain_k2d_e472_c5r2_huber_reduced_m20_g20 --reduced-shear --mask-frac 0.2 --rand-mask-frac --freq-loss freq1d --beta 3.0 | tee k2d_trainlog.txt
```

## Testing MIU<sup>2</sup>Net

rmg20 (noise galaxy = 20, mask frac = 20%, reduced shear):
```console
python pred.py k2d_e1893_c5r2_huber_freq1d_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --reduced-shear --mask-frac 0.2
python make_master_cubes_multiproc.py -g 20 --cpu 32
```

rg20 (noise galaxy = 20, no masks, reduced shear):
```console
python pred.py k2d_e1893_c5r2_huber_freq1d_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --reduced-shear
python make_master_cubes_multiproc.py -g 20 --cpu 32
```

cosmology 2, rmg20
```console
python pred.py k2d_e1893_c5r2_huber_freq1d_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --reduced-shear --mask-frac 0.2 --dir /share/lirui/Wenhan/WL/cosmology2 --cosmo2
python make_master_cubes_multiproc.py -g 20 --cpu 32 --cosmo2
```

cosmology 2, rg20
```console
python pred.py k2d_e1893_c5r2_huber_freq1d_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --reduced-shear --dir /share/lirui/Wenhan/WL/cosmology2 --cosmo2
python make_master_cubes_multiproc.py -g 20 --cpu 32 --cosmo2
```

testing for power spectrum, rg20
```console
python pred_pspec.py k2d_e1893_c5r2_huber_freq1d_reduced_mrand20_g20 -g 20 --num-avg 500 --num-it 30 --cpu 32 --reduced-shear --noise-seed 0
```


## Training DeepMass

```console
python -u train_deepmass.py --gpu-ids 4 --cpu 32 -b 16 -g 20 --reduced-shear --mask-frac 0.2 --rand-mask-frac --wiener only | tee deepmass_trainlog.txt
```

## Testing DeepMass
rmg20
```console
python pred_deepmass.py deepmass_e349_c5r2_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --mask-frac 0.2 --wiener only
```

rg20
```console
python pred_deepmass.py deepmass_e349_c5r2_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --wiener only
```

cosmology 2, rmg20
```console
python pred_deepmass.py deepmass_e349_c5r2_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --dir /share/lirui/Wenhan/WL/cosmology2 --mask-frac 0.2 --wiener only --cosmo2
```

cosmology 2, rg20
```console
python pred_deepmass.py deepmass_e349_c5r2_reduced_mrand20_g20 -g 20 --num 500 --cpu 16 --dir /share/lirui/Wenhan/WL/cosmology2 --wiener only --cosmo2
```