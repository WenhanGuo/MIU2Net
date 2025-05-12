import numpy as np
from scipy.stats import binned_statistic
from skimage.feature import peak_local_max
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
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
    def make_radial_freq_arrays(shape):
        yfreqs = np.fft.fftshift(np.fft.fftfreq(shape[0]))
        xfreqs = np.fft.fftshift(np.fft.fftfreq(shape[1]))
        yy_freq, xx_freq = np.meshgrid(yfreqs, xfreqs, indexing='ij')
        return yy_freq[::-1], xx_freq[::-1]

    nbins = int(40 / binsize)  # for no logspacing; this corresponds to each bin's span approx r = 1 pixel in fft domain
    yy_freq, xx_freq = make_radial_freq_arrays(ps2D.shape)
    freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)
    zero_freq_val = freqs_dist[np.nonzero(freqs_dist)].min() / 2.
    freqs_dist[np.isclose(freqs_dist, 0)] = zero_freq_val

    max_bin = 0.1
    min_bin = 1.0 / min(ps2D.shape)
    # min_bin = 1e-9
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
def P(img, binsize=1.0, logspacing=False):
    ps2D = pspec(img=img)
    freqs, ps1D = radial_pspec(ps2D=ps2D, binsize=binsize, logspacing=logspacing)
    freqs = freqs / u.pix
    return freqs, ps1D

# average 1D power spectrum over many images
def avg_P(cube, binsize=1.0, logspacing=False):
    ps_cube = []
    for idx in range(len(cube)):
        freqs, ps = P(cube[idx], binsize=binsize, logspacing=logspacing)
        ps_cube.append(ps)
        avg_ps = np.mean(ps_cube, axis=0)
        std_ps = np.std(ps_cube, axis=0)
    return freqs, avg_ps, std_ps

def jackknife_P(cube, binsize=1.0, logspacing=False):
    ps_cube = []
    for idx in range(len(cube)):
        freqs, ps = P(cube[idx], binsize=binsize, logspacing=logspacing)
        ps_cube.append(ps)
    N = len(ps_cube)  # number of samples
    ps_cube = np.array(ps_cube)                     # shape: (N, n_bins)
    avg_ps = np.mean(ps_cube, axis=0)

    # compute total sum across all samples for each bin
    total_sum = np.sum(ps_cube, axis=0)             # shape: (n_bins)
    # generate jackknife samples by leaving one sample out each time
    jk_samples = (total_sum - ps_cube) / (N - 1)    # shape: (N, n_bins)
    # compute mean of jackknife samples
    jk_mean = np.mean(jk_samples, axis=0)           # shape: (n_bins)
    # compute jackknife variance
    variance = (N - 1) / N * np.sum((jk_samples - jk_mean) ** 2, axis=0)  # Shape: (n_bins,)
    std_ps = np.sqrt(variance)                         # shape: (n_bins)
    return freqs, avg_ps, std_ps

def pix_to_arcmin(pixel_value, size=256):
    # Convert a value in pixel units to the given angular unit.
    pixel_scale = 1.75*60/size * u.arcmin / u.pix
    return pixel_value * pixel_scale

def spatial_freq_unit_conversion(pixel_value, size):
    # Same as pix_to_arcmin, but handles the inverse units.
    return 1 / pix_to_arcmin(1 / pixel_value, size=size)


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
        pred_mse = mse(y_true=y_true * mask, y_pred=y_pred * mask)
        standard_mse = mse(y_true=y_true * mask, y_pred=y_standard * mask)
        pred_to_standard.append(pred_mse / standard_mse)
    return pred_to_standard

def avg_rel_mse(y_true_cube, y_pred_cube, y_standard_cube, thresholds, mode, binsize=0.05):
    pred_to_standard_cube = []
    for idx in range(len(y_true_cube)):
        pred_to_standard = rel_mse(y_true_cube[idx], y_pred_cube[idx], y_standard_cube[idx], thresholds, mode, binsize)
        pred_to_standard_cube.append(pred_to_standard)
    avg_ratio = np.mean(pred_to_standard_cube, axis=0)
    return avg_ratio


# structural similarity
# ----------------------------------
def mean_ssim(true_cube, pred_cube):
    nimg = true_cube.shape[0]
    mssim = np.empty(nimg, dtype=np.float32)
    for n in range(nimg):
        result = ssim(true_cube[n], pred_cube[n], data_range=pred_cube[n].max() - pred_cube[n].min())
        mssim[n] = result
    mssim = mssim.mean()
    return mssim


# gaussian blurred relative MSE err
# ----------------------------------
def gb_mse(true_cube, pred_cube, gaussian_blur_std):
    k = Gaussian2DKernel(gaussian_blur_std)
    res_cube = pred_cube - true_cube
    nimg = true_cube.shape[0]
    mmse = np.empty(nimg, dtype=np.float32)
    for n in range(nimg):
        true = true_cube[n]
        gb_res = convolve(res_cube[n], kernel=k)
        result = np.sqrt((gb_res ** 2).sum() / (true ** 2).sum())
        mmse[n] = result
    return mmse.mean(), mmse.std()

def gb_mse_without_mean(true_cube, pred_cube, gaussian_blur_std):
    k = Gaussian2DKernel(gaussian_blur_std)
    nimg = true_cube.shape[0]
    mmse = np.empty(nimg, dtype=np.float32)
    for n in range(nimg):
        true = true_cube[n] - true_cube[n].mean()
        pred = pred_cube[n] - pred_cube[n].mean()
        gb_res = convolve(pred, kernel=k) - convolve(true, kernel=k)
        result = np.sqrt((gb_res ** 2).sum() / (true ** 2).sum())
        mmse[n] = result
    return mmse.mean(), mmse.std()

def mse_at_all_scales(true_cube, pred_cube, gaussian_blur_std=[0, 2, 4, 6, 10]):
    mmse_list, errorbar_list = [], []
    for std in gaussian_blur_std:
        mmse, errorbar = gb_mse_without_mean(true_cube, pred_cube, std)
        mmse_list.append(mmse)
        errorbar_list.append(errorbar)
    return mmse_list, errorbar_list


# pixel-wise comparison
# ----------------------------------
# def pix_comp(true_cube, pred_cube):

#     for n in range(nimg):
#         true = true_cube[n]
#         gb_res = convolve(res_cube[n], kernel=k)
#         result = np.sqrt((gb_res ** 2).sum() / (true ** 2).sum())
#         mmse[n] = result
#     return mmse.mean(), mmse.std()


# visualization functions
# ----------------------------------
def draw4(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet):
    plt.subplot(2, 2, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}')
    plt.xticks([])
    plt.yticks([])

def draw6(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet):
    plt.subplot(2, 3, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}')
    plt.xticks([])
    plt.yticks([])

def cbar(fig, cax):
    plt.tight_layout()
    fig.subplots_adjust(right=0.87)
    plt.colorbar(cax=cax, orientation="vertical")

def plot_pspec(xvals, ps1D, logy, label, ls='-', yerr=False, size=256, errfmt='shade', c=None, lw=None):
    xvals = spatial_freq_unit_conversion(xvals, size=size).value
    plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.plot(xvals, ps1D, ls=ls, c=c, label=label, lw=lw)
    if type(yerr) == np.ndarray:
        if errfmt == 'shade':
            plt.fill_between(xvals, ps1D-yerr, ps1D+yerr, alpha=0.15, color=c, lw=lw)
        elif errfmt == 'bar':
            plt.errorbar(xvals, ps1D, yerr=yerr, ls=ls, c=c, lw=lw)
        elif errfmt == None:
            pass