import numpy as np
from scipy.stats import binned_statistic
from skimage.feature import peak_local_max
from sklearn.metrics import mean_squared_error as MSE
from astropy.io import fits
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt

# reading images
# ----------------------------------
def read_prediction(fname):
    with fits.open(fname, memmap=False) as f:
        cube = f[0].data
        true = cube[0]
        ml = cube[1]
        ks = cube[2]
        wiener = cube[3]
        sparse = cube[4]
        mcalens = cube[5]
    return true, ml, ks, wiener, sparse, mcalens

def read_folder(fnames):
    N = len(fnames)
    true_cube = []
    ml_cube = []
    ks_cube = []
    wiener_cube = []
    sparse_cube = []
    mcalens_cube = []

    for i in range(N):
        fname = fnames[i]
        true, ml, ks, wiener, sparse, mcalens = read_prediction(fname)
        true_cube.append(true)
        ml_cube.append(ml)
        ks_cube.append(ks)
        wiener_cube.append(wiener)
        sparse_cube.append(sparse)
        mcalens_cube.append(mcalens)
    
    true_cube = np.array(true_cube)
    ml_cube = np.array(ml_cube)
    ks_cube = np.array(ks_cube)
    wiener_cube = np.array(wiener_cube)
    sparse_cube = np.array(sparse_cube)
    mcalens_cube = np.array(mcalens_cube)

    return true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube, mcalens_cube


# peak count statistics
# ----------------------------------
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

# average peak count over many images
def avg_peak_count(cube, gaussian_blur_std=None, peak_thres=None):
    (pcube, pbin), (snrcube, snrbin) = peak_count_all(cube=cube, gaussian_blur_std=gaussian_blur_std, peak_thres=peak_thres)
    avgp = np.mean(pcube, axis=0)
    avgsnr = np.mean(snrcube, axis=0)
    return (avgp, pbin), (avgsnr, snrbin)


# power spectrum
# ----------------------------------
# calculate 2D power spectrum
def pspec(img):
    def my_rfft_to_fft():
        fft_abs = np.abs(np.fft.rfftn(img))
        fftstar_abs = fft_abs.copy()[:, -2:0:-1]
        fftstar_abs[1::, :] = fftstar_abs[:0:-1, :]
        return np.concatenate((fft_abs, fftstar_abs), axis=1)
    fft = np.fft.fftshift(my_rfft_to_fft())
    ps2D = np.power(fft, 2.)
    return ps2D

# calculate radial average (1D ps) from 2D power spectrum
def radial_pspec(ps2D, binsize=1.0, logspacing=False):
    def make_radial_arrays(shape):
        y_center = np.floor(shape[0] / 2.).astype(int)
        x_center = np.floor(shape[1] / 2.).astype(int)
        y = np.arange(-y_center, shape[0] - y_center)
        x = np.arange(-x_center, shape[1] - x_center)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        return yy, xx
    def make_radial_freq_arrays(shape):
        yfreqs = np.fft.fftshift(np.fft.fftfreq(shape[0]))
        xfreqs = np.fft.fftshift(np.fft.fftfreq(shape[1]))
        yy_freq, xx_freq = np.meshgrid(yfreqs, xfreqs, indexing='ij')
        return yy_freq[::-1], xx_freq[::-1]

    yy, xx = make_radial_arrays(ps2D.shape)
    dists = np.sqrt(yy**2 + xx**2)
    nbins = int(np.round(dists.max() / binsize) + 1)
    yy_freq, xx_freq = make_radial_freq_arrays(ps2D.shape)
    freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)
    zero_freq_val = freqs_dist[np.nonzero(freqs_dist)].min() / 2.
    freqs_dist[freqs_dist == 0] = zero_freq_val

    max_bin = 0.5
    min_bin = 1.0 / min(ps2D.shape)
    if logspacing:
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), nbins + 1)
    else:
        bins = np.linspace(min_bin, max_bin, nbins + 1)

    dist_arr = freqs_dist
    finite_mask = np.isfinite(ps2D)
    ps1D, bin_edge, cts = binned_statistic(dist_arr[finite_mask].ravel(),
                                           ps2D[finite_mask].ravel(),
                                           bins=bins,
                                           statistic=np.nanmean)
    bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.

    return bin_cents, ps1D

# calculate 1D power spectrum from image
def P(img, logspacing=False):
    ps2D = pspec(img=img)
    freqs, ps1D = radial_pspec(ps2D=ps2D, binsize=1.0, logspacing=logspacing)
    freqs = freqs / u.pix
    return freqs, ps1D

# average 1D power spectrum over many images
def avg_P(cube, logspacing=False):
    ps_cube = []
    for idx in range(len(cube)):
        freqs, ps = P(cube[idx], logspacing=logspacing)
        ps_cube.append(ps)
        avg_ps = np.mean(ps_cube, axis=0)
    return freqs, avg_ps

def pix_to_arcmin(pixel_value):
    # Convert a value in pixel units to the given angular unit.
    pixel_scale = 1.75*60/512 * u.arcmin / u.pix
    return pixel_value * pixel_scale

def spatial_freq_unit_conversion(pixel_value):
    # Same as pix_to_arcmin, but handles the inverse units.
    return 1 / pix_to_arcmin(1 / pixel_value)


# binned MSE
# ----------------------------------
def rel_mse(y_true, y_pred, y_standard, thresholds, mode, binsize=0.05):
    assert mode in ['min_thres', 'bin_thres', 'max_thres']
    pred_to_standard = []
    for t in thresholds:
        if mode == 'min_thres':
            mask = y_true > t
        elif mode == 'bin_thres':
            mask = (y_true > t) & (y_true < t + binsize)
        elif mode == 'max_thres':
            mask = y_true < t
        pred_mse = MSE(y_true=y_true * mask, y_pred=y_pred * mask)
        standard_mse = MSE(y_true=y_true * mask, y_pred=y_standard * mask)
        pred_to_standard.append(pred_mse / standard_mse)
    return pred_to_standard

def avg_rel_mse(y_true_cube, y_pred_cube, y_standard_cube, thresholds, mode, binsize=0.05):
    pred_to_standard_cube = []
    for idx in range(len(y_true_cube)):
        pred_to_standard = rel_mse(y_true_cube[idx], y_pred_cube[idx], y_standard_cube[idx], thresholds, mode, binsize)
        pred_to_standard_cube.append(pred_to_standard)
    avg_ratio = np.mean(pred_to_standard_cube, axis=0)
    return avg_ratio


# visualization functions
# ----------------------------------
def draw4(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 2, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

def draw6(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 3, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

def cbar(fig, cax):
    plt.tight_layout()
    fig.subplots_adjust(right=0.87)
    plt.colorbar(cax=cax, orientation="vertical")

def plot_pspec(xvals, ps1D, logy, label, c=None):
    xvals = spatial_freq_unit_conversion(xvals).value
    plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.plot(xvals, ps1D, c=c, label=label)