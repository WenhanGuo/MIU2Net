# %%
import numpy as np
from astropy.io import fits
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
from my_cosmostat.misc.im_isospec import im_isospec
import matplotlib.pyplot as plt

fname = '/Users/danny/Desktop/ks_wf_sp_g20.fits'
# open image
with fits.open(fname) as f:
    cube = f[0].data
    shear1 = cube[0]
    shear2 = cube[1]
    ks = cube[2]
    wiener = cube[3]
    sparse = cube[4]
    pred = cube[5]
    true = cube[6]
    res = cube[7]

# number of galaxies per arcmin^2, used to calculate shape noise
n_galaxy = 20

# %%
# calculate shape noise due to n_galaxy
    # for 50 gal / arcmin^2, std = 0.1951
    # for 20 gal / arcmin^2, std = 0.3085
def shape_noise(n_galaxy):
    sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
    theta_G = 0.205   # pixel side length in arcmin (gaussian smoothing window)
    variance = (sigma_e**2 / 2) / (theta_G**2 * n_galaxy)
    std = np.sqrt(variance)
    print('shape noise std =', std)
    return std
noise_std = shape_noise(n_galaxy=n_galaxy)

# initialize CosmoStat shear class
D = shear_data()
D.g1 = - shear1
D.g2 = shear2
p_signal = im_isospec(true)
CovMat = np.ones((512, 512)) * (noise_std**2)
D.Ncov = CovMat
D.nx, D.ny = 512, 512

# create the mass mapping structure and initialize it
M = massmap2d(name='mass')
M.init_massmap(nx=512, ny=512)
p_noise = M.get_noise_powspec(CovMat=CovMat, nsimu=10)

# %%
# sparse reconstruction with a 5 sigma detection
sparse, ti = M.sparse_recons(D, UseNoiseRea=False, Nsigma=5, ThresCoarse=False, Inpaint=False, Nrea=None)

# %%
# MCALens reconstruction
mcalens, k1i5, k2r5, k2i = M.sparse_wiener_filtering(D, p_signal, Nsigma=5, Inpaint=False, Bmode=True, ktr=None)

# %%
# visualize 6 plots: true, ML, KS, Wiener, sparse, MCALens
def draw6(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 3, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

def cbar(cax):
    plt.tight_layout()
    fig.subplots_adjust(right=0.87)
    plt.colorbar(cax=cax, orientation="vertical")

fig = plt.figure(figsize=(12,8))
draw6(1, true, 'True', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(2, pred, 'Hybrid-ML', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(3, ks, 'KS Deconvolution', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(4, wiener, 'Wiener Filtering', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(5, sparse, 'Sparse Recovery', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(6, mcalens, 'MCALens', scale=[true.min(), true.max()/3], cmap='viridis')

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
# save 6 plots into a master cube for further statistics
def save_cube(true, ml, ks, wiener, sparse, mcalens):
    true1 = np.expand_dims(true, axis=0)
    ml1 = np.expand_dims(ml, axis=0)
    ks1 = np.expand_dims(ks, axis=0)
    wiener1 = np.expand_dims(wiener, axis=0)
    sparse1 = np.expand_dims(sparse, axis=0)
    mcalens1 = np.expand_dims(mcalens, axis=0)
    c = np.concatenate([true1, ml1, ks1, wiener1, sparse1, mcalens1])
    fits.writeto('/Users/danny/Desktop/master_cube.fits', data=c, overwrite=False)

save_cube(true, pred, ks, wiener, sparse, mcalens)

# %%
