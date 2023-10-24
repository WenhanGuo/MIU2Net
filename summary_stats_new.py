# %%
import numpy as np
import scipy.ndimage as ndimage
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

fname = '/Users/danny/Desktop/ks_wf_sp_g20.fits'
# Open image
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

def draw4(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 2, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

def cbar(cax):
    plt.tight_layout()
    fig.subplots_adjust(right=0.87)
    plt.colorbar(cax=cax, orientation="vertical")

gausblur_pred = ndimage.gaussian_filter(pred, sigma=2, radius=2, order=0)
gausblur_true = ndimage.gaussian_filter(true, sigma=2, radius=2, order=0)

fig = plt.figure(figsize=(8,7))
draw4(1, pred, 'Prediction', scale=[true.min(), true.max()/2])
draw4(2, true, 'True', scale=[true.min(), true.max()/2])
draw4(3, gausblur_pred, 'Gaus Blur Prediction', scale=[true.min(), true.max()/2])
draw4(4, gausblur_true, 'Gaus Blur True', scale=[true.min(), true.max()/2])

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
fig = plt.figure(figsize=(8,7))
draw4(1, true, 'True', scale=[true.min(), true.max()/3], cmap='viridis')
draw4(2, pred, 'Hybrid-ML Prediction', scale=[true.min(), true.max()/3], cmap='viridis')
draw4(3, ks, 'KS Deconvolution', scale=[true.min(), true.max()/3], cmap='viridis')
draw4(4, wiener, 'Wiener Filtering', scale=[true.min(), true.max()/3], cmap='viridis')

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
pcoord = peak_local_max(gausblur_pred, min_distance=3, num_peaks=200, threshold_rel=0.1)
tcoord = peak_local_max(gausblur_true, min_distance=5, num_peaks=100, threshold_rel=0.1)

kdtree = KDTree(pcoord)
distances, idxes = kdtree.query(tcoord)

matched_pcoord, sub_tcoord = [], []
for j in range(len(distances)):
    if distances[j] <= 3:
        matched_pcoord.append(pcoord[idxes[j]])
        sub_tcoord.append(tcoord[j])
matched_pcoord, sub_tcoord = np.array(matched_pcoord), np.array(sub_tcoord)

fig = plt.figure(figsize=(10,9))
draw4(1, gausblur_pred, f'Blurred Prediction ({len(pcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(pcoord[:, 1], pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw4(2, gausblur_true, f'Blurred True ({len(tcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(tcoord[:, 1], tcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)

draw4(3, gausblur_pred, f'Blurred Prediction ({len(matched_pcoord)} matched peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(matched_pcoord[:, 1], matched_pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw4(4, gausblur_true, f'Blurred True ({len(sub_tcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(sub_tcoord[:, 1], sub_tcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
true_peak_mask = np.zeros_like(true, dtype=bool)
true_peak_mask[tuple(sub_tcoord.T)] = True
masked_true = np.ma.MaskedArray(true, true_peak_mask)
plt.imshow(masked_true)

# %%
pred_peak_counts = gausblur_pred[tuple(zip(*matched_pcoord))]
true_peak_counts = gausblur_true[tuple(zip(*sub_tcoord))]
abs_err = abs(pred_peak_counts - true_peak_counts)
rel_err = abs(pred_peak_counts - true_peak_counts) / abs(true_peak_counts)

_ = plt.hist(pred_peak_counts, 20, density=True, alpha=0.7, label='pred peak values')
_ = plt.hist(true_peak_counts, 20, density=True, alpha=0.7, label='true peak values')
plt.legend()

print('avg abs err per peak =', sum(abs_err)/len(sub_tcoord))
print('avg rel err per peak =', sum(rel_err)/len(sub_tcoord))

# %%
fig, ax = plt.subplots()
plt.scatter(pred.flatten(), true.flatten(), s=1, alpha=0.2)
plt.xlim(min(true.flatten())-0.05, max(true.flatten())+0.05)
plt.ylim(min(true.flatten())-0.05, max(true.flatten())+0.05)
plt.xlabel('predicted pixel intensity')
plt.ylabel('true pixel intensity')
ax.set_aspect('equal', adjustable='box')

# %%
fpred = sorted(pred.flatten())
ftrue = sorted(true.flatten())
stack = np.stack((fpred, ftrue), axis=1)

fig, ax = plt.subplots()
plt.plot(stack[:, 0], stack[:, 1])
plt.xlim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.xlabel('predicted pixel intensity')
plt.ylabel('true pixel intensity')
ax.set_aspect('equal', adjustable='box')
plt.grid(True)

# %%
# ------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error as MSE
from my_cosmostat.astro.wl.mass_mapping import massmap2d, shear_data
from my_cosmostat.misc.im_isospec import im_isospec

D = shear_data()
D.g1 = - shear1
D.g2 = shear2
p_signal = im_isospec(true)
CovMat = np.ones((512, 512)) * (0.1951**2)   # std = 0.1951 for 50 galaxies per arcmin^2
D.Ncov = CovMat
D.nx, D.ny = 512, 512

# Create the mass mapping structure and initialise it
M = massmap2d(name='mass')
M.init_massmap(nx=512, ny=512)
p_noise = M.get_noise_powspec(CovMat=CovMat, nsimu=10)

retr, reti = M.wiener(D.g1, D.g2, PowSpecSignal=p_signal, PowSpecNoise=p_noise)

# %%
# Do a sparse reconstruction with a 5 sigma detection
sparse,ti = M.sparse_recons(D, UseNoiseRea=False,Nsigma=5, ThresCoarse=False, Inpaint=False,Nrea=None)

# %%
mcalens,k1i5,k2r5,k2i = M.sparse_wiener_filtering(D, p_signal, Nsigma=5, Inpaint=False, Bmode=True, ktr=None)

# %%
print('pred_mse =', MSE(y_true=true, y_pred=pred))
print('ks_mse =', MSE(y_true=true, y_pred=ks))
print('wiener_mse =', MSE(y_true=true, y_pred=wiener))
print('retr_mse =', MSE(y_true=true, y_pred=retr))
print('sparse_mse =', MSE(y_true=true, y_pred=sparse))
print('MCALens_mse =', MSE(y_true=true, y_pred=mcalens))

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

def plot_all(thresholds, true, pred, ks, wiener, retr, sparse, mcalens, mode, title, xlabel):
    plt.subplots(figsize=(10,8))
    plt.title(title, fontsize=18)
    plt.plot(thresholds, rel_mse(true, pred, ks, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{KS}}$')
    plt.plot(thresholds, rel_mse(true, pred, wiener, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{wiener}}$')
    plt.plot(thresholds, rel_mse(true, pred, sparse, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{sparse5}}$')
    plt.plot(thresholds, rel_mse(true, pred, mcalens, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{MCALens}}$')
    plt.xlabel(xlabel)
    plt.axhline(y = 1.0, color='r', linestyle='--', alpha=0.7)
    plt.axvspan(true.min(), sorted(true.flatten())[259522], alpha=0.15)
    plt.text(x=-0.02, y=-0.05, s='99% of pixels', fontsize=16)
    plt.axvspan(sorted(true.flatten())[259522], 0.4, color='r', alpha=0.05)
    plt.text(x=0.2, y=-0.05, s='top 1% (peaks)', fontsize=16)
    plt.legend(loc='upper left')

thresholds = np.arange(start=-0.035, stop=0.4, step=0.005)
plot_all(thresholds, true, pred, ks, wiener, retr, sparse, mcalens, mode='bin_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')
plot_all(thresholds, true, pred, ks, wiener, retr, sparse, mcalens, mode='min_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')
plot_all(thresholds, true, pred, ks, wiener, retr, sparse, mcalens, mode='max_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')

# %%
def draw6(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 3, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

fig = plt.figure(figsize=(12,8))
draw6(1, true, 'True', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(2, pred, 'Hybrid-ML', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(3, ks, 'KS Deconvolution', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(4, wiener, 'Wiener Filtering', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(5, sparse, 'Sparse Recovery', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(6, mcalens, 'MCALens', scale=[true.min(), true.max()/3], cmap='viridis')

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
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
def draw_pix2pix(pred, true, label):
    fpred = sorted(pred.flatten())
    ftrue = sorted(true.flatten())
    stack = np.stack((ftrue, fpred), axis=1)
    plt.plot(stack[:, 0], stack[:, 1], label=label)

plt.figure(figsize=(6, 6))
draw_pix2pix(pred, true, label='ML')
draw_pix2pix(ks, true, label='KS')
draw_pix2pix(wiener, true, label='Wiener')
draw_pix2pix(sparse, true, label='sparse5')
draw_pix2pix(mcalens, true, label='MCALens')
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

plt.title('flattened pix to pix comparison', fontsize=16)
plt.xlim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylabel(r'predicted $\hat{\kappa}$ pixel intensity')
plt.xlabel(r'true $\kappa$ pixel intensity')
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()

# %%
# ------------------------------------------------------------------------------
import astropy.units as u
from scipy.stats import binned_statistic

cube = fits.open('/Users/danny/Desktop/master_cube.fits')[0].data
true = cube[0]
ml = cube[1]
ks = cube[2]
wiener = cube[3]
sparse = cube[4]
mcalens = cube[5]

# %%
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

# %%
def pix_to_arcmin(pixel_value):
    '''
    Convert a value in pixel units to the given angular unit.
    '''
    pixel_scale = 1.75*60/512 * u.arcmin / u.pix
    return pixel_value * pixel_scale


def spatial_freq_unit_conversion(pixel_value):
    '''
    Same as pix_to_arcmin, but handles the inverse units.
    Feed in as the inverse of the value, and then inverse again so that
    the unit conversions will work.
    '''
    return 1 / pix_to_arcmin(1 / pixel_value)


def plot_pspec(xvals, ps1D, logy, label, c=None):
    xvals = spatial_freq_unit_conversion(xvals).value
    plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.plot(xvals, ps1D, c=c, label=label)

# %%
# plot 1D power spectrums for truth & all recovery methods
f_true, p_true = P(img=true, logspacing=False)
f_ml, p_ml = P(img=ml, logspacing=False)
f_ks, p_ks = P(img=ks, logspacing=False)
f_wiener, p_wiener = P(img=wiener, logspacing=False)
f_sparse, p_sparse = P(img=sparse, logspacing=False)
f_mcalens, p_mcalens = P(img=mcalens, logspacing=False)

plt.figure(figsize=(8, 6))
plt.title('1D Power Spectrum')
plot_pspec(xvals=f_true, ps1D=p_true, logy=True, label='true', c='k')
plot_pspec(xvals=f_ml, ps1D=p_ml, logy=True, label='ml')
plot_pspec(xvals=f_ks, ps1D=p_ks, logy=True, label='ks')
plot_pspec(xvals=f_wiener, ps1D=p_wiener, logy=True, label='wiener')
plot_pspec(xvals=f_sparse, ps1D=p_sparse, logy=True, label='sparse')
plot_pspec(xvals=f_mcalens, ps1D=p_mcalens, logy=True, label='mcalens')
plt.xlabel("Spatial Frequency (1 / arcmin)")
plt.ylabel(r"$P(\kappa)$")
plt.grid(True)
plt.legend()

# %%
# plot powerspec(true - pred) / powerspec(true)
f_res_ml, p_res_ml = P(img=true-ml, logspacing=False)
f_res_ks, p_res_ks = P(img=true-ks, logspacing=False)
f_res_wiener, p_res_wiener = P(img=true-wiener, logspacing=False)
f_res_sparse, p_res_sparse = P(img=true-sparse, logspacing=False)
f_res_mcalens, p_res_mcalens = P(img=true-mcalens, logspacing=False)

plt.figure(figsize=(8, 6))
plt.title(r"Normalized Power Spectrum $P_\Delta$ of Residual $\hat{\kappa} - \kappa_{\rm{truth}}$")
plot_pspec(xvals=f_res_ml, ps1D=p_res_ml/p_true, logy=False, label='ml')
plot_pspec(xvals=f_res_ks, ps1D=p_res_ks/p_true, logy=False, label='ks')
plot_pspec(xvals=f_res_wiener, ps1D=p_res_wiener/p_true, logy=False, label='wiener')
plot_pspec(xvals=f_res_sparse, ps1D=p_res_sparse/p_true, logy=False, label='sparse')
plot_pspec(xvals=f_res_mcalens, ps1D=p_res_mcalens/p_true, logy=False, label='mcalens')
plt.xlabel("Spatial Frequency (1 / arcmin)")
plt.ylabel(r"$P_\Delta \, / \, P_{\rm{true}}$")
plt.xlim(xmin=10**-1.7)
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()

# %%
