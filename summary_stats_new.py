# %%
import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# fname = '/Users/danny/Desktop/ks_wf_add2.fits'
fname = '/Users/danny/Desktop/ks_wf_add2_2.fits'
# Open image
with fits.open(fname) as f:
    cube = f[0].data
    shear1 = cube[0]
    shear2 = cube[1]
    ks = cube[2]
    wiener = cube[3]
    pred = cube[4]
    true = cube[5]
    res = cube[6]

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
ksr5,ti = M.sparse_recons(D, UseNoiseRea=False,Nsigma=5, ThresCoarse=False, Inpaint=False,Nrea=None)

# %%
k1r5,k1i5,k2r5,k2i = M.sparse_wiener_filtering(D, p_signal, Nsigma=5, Inpaint=False, Bmode=True, ktr=None)

# %%
print('pred_mse =', MSE(y_true=true, y_pred=pred))
print('ks_mse =', MSE(y_true=true, y_pred=ks))
print('wiener_mse =', MSE(y_true=true, y_pred=wiener))
print('retr_mse =', MSE(y_true=true, y_pred=retr))
print('sparse5_mse =', MSE(y_true=true, y_pred=ksr5))
print('MCALens_mse =', MSE(y_true=true, y_pred=k1r5))

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

def plot_all(thresholds, true, pred, ks, wiener, retr, ksr5, k1r5, mode, title, xlabel):
    plt.subplots(figsize=(10,8))
    plt.title(title, fontsize=18)
    plt.plot(thresholds, rel_mse(true, pred, ks, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{KS}}$')
    plt.plot(thresholds, rel_mse(true, pred, wiener, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{wiener}}$')
    # plt.plot(thresholds, rel_mse(true, pred, retr, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{retr}}$')
    plt.plot(thresholds, rel_mse(true, pred, ksr5, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{sparse5}}$')
    plt.plot(thresholds, rel_mse(true, pred, k1r5, thresholds, mode=mode), label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{MCALens}}$')
    plt.xlabel(xlabel)
    plt.axhline(y = 1.0, color='r', linestyle='--', alpha=0.7)
    plt.axvspan(true.min(), sorted(true.flatten())[259522], alpha=0.15)
    plt.text(x=-0.02, y=-0.05, s='99% of pixels', fontsize=16)
    plt.axvspan(sorted(true.flatten())[259522], 0.4, color='r', alpha=0.05)
    plt.text(x=0.2, y=-0.05, s='top 1% (peaks)', fontsize=16)
    plt.legend(loc='upper left')

thresholds = np.arange(start=-0.035, stop=0.4, step=0.005)
plot_all(thresholds, true, pred, ks, wiener, retr, ksr5, k1r5, mode='min_thres', title='MSE ratio (min threshold)', xlabel=r'$\kappa_{\rm{truth}} > X$')
plot_all(thresholds, true, pred, ks, wiener, retr, ksr5, k1r5, mode='max_thres', title='MSE ratio (max threshold)', xlabel=r'$\kappa_{\rm{truth}} < X$')
plot_all(thresholds, true, pred, ks, wiener, retr, ksr5, k1r5, mode='bin_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')

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
draw6(5, ksr5, 'Sparse Recovery', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(6, k1r5, 'MCALens', scale=[true.min(), true.max()/3], cmap='viridis')

cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
p_signal = im_isospec(true)
p_pred = im_isospec(pred - true)
p_ks = im_isospec(ks - true)
p_wiener = im_isospec(wiener - true)
p_sparse = im_isospec(ksr5 - true)
p_mcalens = im_isospec(k1r5 - true)

plt.plot(p_pred / p_signal, label='pred')
plt.plot(p_wiener / p_signal, label='wiener')
plt.plot(p_sparse / p_signal, label='sparse')
plt.plot(p_mcalens / p_signal, label='mcalens')
plt.ylim(0, 1.2)
plt.xlim(1, 50)
plt.ylabel(r'$P_\Delta \, / \, P_{\rm{truth}}$')
plt.xlabel('arbitrary')
plt.legend()

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
draw_pix2pix(ksr5, true, label='sparse5')
draw_pix2pix(k1r5, true, label='MCALens')
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

plt.title('flattened pix to pix comparison', fontsize=16)
plt.xlim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylabel(r'predicted $\hat{\kappa}$ pixel intensity')
plt.xlabel(r'true $\kappa$ pixel intensity')
ax.set_aspect('equal', adjustable='box')
plt.legend()

# %%
