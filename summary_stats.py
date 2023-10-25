# %%
import numpy as np
import glob
from scipy.spatial import KDTree
from scipy.stats import binned_statistic
from skimage.feature import peak_local_max
from sklearn.metrics import mean_squared_error as MSE
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc_file_defaults()

def read_prediction(fname):
    with fits.open(fname, memmap=False) as f:
        cube = f[0].data
        true = cube[0]
        ml = cube[1]
        ks = cube[2]
        wiener = cube[3]
        sparse = cube[4]
    return true, ml, ks, wiener, sparse

def read_folder(fnames):
    N = len(fnames)
    true_cube = []
    ml_cube = []
    ks_cube = []
    wiener_cube = []
    sparse_cube = []

    for i in range(N):
        fname = fnames[i]
        true, ml, ks, wiener, sparse = read_prediction(fname)
        true_cube.append(true)
        ml_cube.append(ml)
        ks_cube.append(ks)
        wiener_cube.append(wiener)
        sparse_cube.append(sparse)
    
    true_cube = np.array(true_cube)
    ml_cube = np.array(ml_cube)
    ks_cube = np.array(ks_cube)
    wiener_cube = np.array(wiener_cube)
    sparse_cube = np.array(sparse_cube)

    return true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube


fnames = glob.glob('/Users/danny/Desktop/master_cubes/*.fits')[:50]
true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube = read_folder(fnames)

# %% [markdown]
# ## Peak Count

# %%
# peak count statistics
def peak_count(img, gaussian_blur_std=None, peak_thres=None):
    if gaussian_blur_std == None:
        coords = peak_local_max(img, min_distance=1, threshold_abs=peak_thres)
        pcounts = img[tuple(zip(*coords))]
        noise = np.std(img)
    else:
        k = Gaussian2DKernel(gaussian_blur_std)
        gb_img = convolve(img, kernel=k)
        coords = peak_local_max(gb_img, min_distance=1, threshold_abs=peak_thres)
        pcounts = gb_img[tuple(zip(*coords))]
        noise = np.std(gb_img)
    snr = pcounts / noise
    return pcounts, snr


def peak_hist(l, bins=20, ran=None):
    hist, bin_edges = np.histogram(l, bins=bins, range=ran)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return hist, bin_centers


def peak_count_all(cube, gaussian_blur_std=None, peak_thres=None):
    hist_peak_cube, hist_snr_cube = [], []
    for idx in range(len(cube)):
        pcounts, snr = peak_count(cube[idx], gaussian_blur_std=gaussian_blur_std, peak_thres=peak_thres)
        hist_peak, bin_cen_peak = peak_hist(l=pcounts, bins=20, ran=(0, 0.1))
        hist_snr, bin_cen_snr = peak_hist(l=snr, bins=20, ran=(1, 6))
        hist_peak_cube.append(hist_peak)
        hist_snr_cube.append(hist_snr)
    hist_peak_cube = np.array(hist_peak_cube)
    hist_snr_cube = np.array(hist_snr_cube)
    return (hist_peak_cube, bin_cen_peak), (hist_snr_cube, bin_cen_snr)


def avg_peak_count(cube, gaussian_blur_std=None, peak_thres=None):
    (pcube, pbin), (snrcube, snrbin) = peak_count_all(cube=cube, gaussian_blur_std=gaussian_blur_std, peak_thres=peak_thres)
    avgp = np.mean(pcube, axis=0)
    avgsnr = np.mean(snrcube, axis=0)
    return (avgp, pbin), (avgsnr, snrbin)


# gb_std = 4.8762
gb_std = 3
peak_thres = 0

(true_avgp, true_pbin), (true_avgsnr, true_snrbin) = avg_peak_count(cube=true_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ml_avgp, ml_pbin), (ml_avgsnr, ml_snrbin) = avg_peak_count(cube=ml_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ks_avgp, ks_pbin), (ks_avgsnr, ks_snrbin) = avg_peak_count(cube=ks_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(wiener_avgp, wiener_pbin), (wiener_avgsnr, wiener_snrbin) = avg_peak_count(cube=wiener_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(sparse_avgp, sparse_pbin), (sparse_avgsnr, sparse_snrbin) = avg_peak_count(cube=sparse_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)

# %%
sns.set()
fig = plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"peak count (gaus blur std = {gb_std})")
plt.plot(true_pbin, true_avgp, label='true', c='k', ls='--')
plt.plot(ml_pbin, ml_avgp, label='ml')
plt.plot(ks_pbin, ks_avgp, label='ks')
plt.plot(wiener_pbin, wiener_avgp, label='wiener')
plt.plot(sparse_pbin, sparse_avgp, label='sparse')
plt.xlabel(r"$\kappa$")
plt.ylabel("number of peaks")
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.title(f"peak count (gaus blur std = {gb_std})")
plt.plot(true_snrbin, true_avgsnr, label='true', c='k', ls='--')
plt.plot(ml_snrbin, ml_avgsnr, label='ml')
plt.plot(ks_snrbin, ks_avgsnr, label='ks')
plt.plot(wiener_snrbin, wiener_avgsnr, label='wiener')
plt.plot(sparse_snrbin, sparse_avgsnr, label='sparse')
plt.xlabel(r"SNR $\nu = \kappa \, / \, \sigma_i$")
plt.ylabel("number of peaks")
plt.xlim(1, 6)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# %%
def plot_peak_counts(true, ml, ks, wiener, sparse, gaussian_blur_std=None, peak_thres=None):
    gb_std = gaussian_blur_std
    true_pcounts, true_snr = peak_count(img=true, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    ml_pcounts, ml_snr = peak_count(img=ml, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    ks_pcounts, ks_snr = peak_count(img=ks, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    wiener_pcounts, wiener_snr = peak_count(img=wiener, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    sparse_pcounts, sparse_snr = peak_count(img=sparse, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    
    sns.set()
    fig = plt.figure(figsize=(7, 5))
    plt.title(f"peak count (gaus blur std = {gaussian_blur_std})")
    plt.hist(true_pcounts, bins=20, range=(0, 0.1), label='true', histtype='step', color='k', ls='--', lw=1.5)
    plt.hist(ml_pcounts, bins=20, range=(0, 0.1), label='ml', histtype='step', lw=1.5)
    plt.hist(ks_pcounts, bins=20, range=(0, 0.1), label='ks', histtype='step', lw=1.5)
    plt.hist(wiener_pcounts, bins=20, range=(0, 0.1), label='wiener', histtype='step', lw=1.5)
    plt.hist(sparse_pcounts, bins=20, range=(0, 0.1), label='sparse', histtype='step', lw=1.5)
    plt.xlabel(r"$\kappa$")
    plt.ylabel("number of peaks")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(7, 5))
    plt.title(f"peak count (gaus blur std = {gaussian_blur_std})")
    plt.hist(true_snr, bins=20, range=(1, 10), label='true', histtype='step', color='k', ls='--', lw=1.5)
    plt.hist(ml_snr, bins=20, range=(1, 10), label='ml', histtype='step', lw=1.5)
    plt.hist(ks_snr, bins=20, range=(1, 10), label='ks', histtype='step', lw=1.5)
    plt.hist(wiener_snr, bins=20, range=(1, 10), label='wiener', histtype='step', lw=1.5)
    plt.hist(sparse_snr, bins=20, range=(1, 10), label='sparse', histtype='step', lw=1.5)
    plt.xlabel(r"SNR $\nu = \kappa \, / \, \sigma_i$")
    plt.ylabel("number of peaks")
    plt.xlim(0, 10)
    plt.legend()
    plt.show()

# %%
