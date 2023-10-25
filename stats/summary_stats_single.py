# %%
import numpy as np
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

# load cube containing results from all prediction methods
cube = fits.open('/Users/danny/Desktop/WL/kappa_map/result/master_cube_ks_wf.fits')[0].data
true = cube[0]
ml = cube[1]
ks = cube[2]
wiener = cube[3]
sparse = cube[4]
mcalens = cube[5]

# gaussian blur all methods with std = 1 arcmin
k = Gaussian2DKernel(1)
gb_true = convolve(true, k)
gb_ml = convolve(ml, k)
gb_ks = convolve(ks, k)
gb_wiener = convolve(wiener, k)
gb_sparse = convolve(sparse, k)
gb_mcalens = convolve(mcalens, k)

# %% [markdown]
# ### Visualize all methods and their gaussian blurs

# %%
from summary_stats_func import draw4, draw6, cbar

fig = plt.figure(figsize=(12,8))
draw6(1, true, 'True', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(2, ml, 'Hybrid-ML', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(3, ks, 'KS Deconvolution', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(4, wiener, 'Wiener Filtering', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(5, sparse, 'Sparse Recovery', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(6, mcalens, 'MCALens', scale=[true.min(), true.max()/3], cmap='viridis')
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))
plt.show()

fig = plt.figure(figsize=(12,8))
plt.title("Gaussian Blurred")
draw6(1, gb_true, 'GB True', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(2, gb_ml, 'GB Hybrid-ML', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(3, gb_ks, 'GB KS Deconvolution', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(4, gb_wiener, 'GB Wiener Filtering', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(5, gb_sparse, 'GB Sparse Recovery', scale=[true.min(), true.max()/3], cmap='viridis')
draw6(6, gb_mcalens, 'GB MCALens', scale=[true.min(), true.max()/3], cmap='viridis')
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))
plt.show()

# %% [markdown]
# ### Matched Peaks Relative Err

# %%
# visualize peak matching
pcoord = peak_local_max(gb_ml, min_distance=3, num_peaks=200, threshold_rel=0.1)
tcoord = peak_local_max(gb_true, min_distance=3, num_peaks=100, threshold_rel=0.1)
kdtree = KDTree(pcoord)
distances, idxes = kdtree.query(tcoord)

matched_pcoord, sub_tcoord = [], []
for j in range(len(distances)):
    if distances[j] <= 3:
        matched_pcoord.append(pcoord[idxes[j]])
        sub_tcoord.append(tcoord[j])
matched_pcoord, sub_tcoord = np.array(matched_pcoord), np.array(sub_tcoord)

fig = plt.figure(figsize=(10,9))
draw4(1, gb_ml, f'Blurred Prediction ({len(pcoord)} peaks)', scale=[ml.min(), ml.max()], cmap='viridis')
plt.scatter(pcoord[:, 1], pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw4(2, gb_true, f'Blurred True ({len(tcoord)} peaks)', scale=[ml.min(), ml.max()], cmap='viridis')
plt.scatter(tcoord[:, 1], tcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw4(3, gb_ml, f'Blurred Prediction ({len(matched_pcoord)} matched peaks)', scale=[ml.min(), ml.max()], cmap='viridis')
plt.scatter(matched_pcoord[:, 1], matched_pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw4(4, gb_true, f'Blurred True ({len(sub_tcoord)} peaks)', scale=[ml.min(), ml.max()], cmap='viridis')
plt.scatter(sub_tcoord[:, 1], sub_tcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
def find_and_match_peak(target, pred):
    pred_coords = peak_local_max(pred, min_distance=3, num_peaks=200, threshold_rel=0.1)
    target_coords = peak_local_max(target, min_distance=3, num_peaks=100, threshold_rel=0.1)
    kdtree = KDTree(pred_coords)
    distances, idxes = kdtree.query(target_coords)
    matched_pcoord, sub_tcoord = [], []
    for j in range(len(distances)):
        if distances[j] <= 3:
            matched_pcoord.append(pred_coords[idxes[j]])
            sub_tcoord.append(target_coords[j])
    matched_pcoord, sub_tcoord = np.array(matched_pcoord), np.array(sub_tcoord)
    target_pcounts = target[tuple(zip(*sub_tcoord))]
    pred_pcounts = pred[tuple(zip(*matched_pcoord))]
    return (sub_tcoord, target_pcounts), (matched_pcoord, pred_pcounts)

def matched_peak_err(target_pcounts, pred_pcounts, label):
    abs_err = abs(pred_pcounts - target_pcounts)
    rel_err = abs(pred_pcounts - target_pcounts) / abs(target_pcounts)
    print(f'{label} matched peak err w/ {len(target_pcounts)} matched peaks')
    print('-------------------')
    print(f'avg abs err per peak = {sum(abs_err)/len(sub_tcoord):4f}')
    print(f'avg rel err per peak = {(sum(rel_err)/len(sub_tcoord)) * 100:.2f} %')
    print('\n')

(true_pcoords, true_pcounts), (ml_pcoords, ml_pcounts) = find_and_match_peak(target=gb_true, pred=gb_ml)
matched_peak_err(true_pcounts, ml_pcounts, label='ml')
(true_pcoords, true_pcounts), (ks_pcoords, ks_pcounts) = find_and_match_peak(target=gb_true, pred=gb_ks)
matched_peak_err(true_pcounts, ks_pcounts, label='ks')
(true_pcoords, true_pcounts), (wiener_pcoords, wiener_pcounts) = find_and_match_peak(target=gb_true, pred=gb_wiener)
matched_peak_err(true_pcounts, wiener_pcounts, label='wiener')
(true_pcoords, true_pcounts), (sparse_pcoords, sparse_pcounts) = find_and_match_peak(target=gb_true, pred=gb_sparse)
matched_peak_err(true_pcounts, sparse_pcounts, label='sparse')
(true_pcoords, true_pcounts), (mcalens_pcoords, mcalens_pcounts) = find_and_match_peak(target=gb_true, pred=gb_mcalens)
matched_peak_err(true_pcounts, mcalens_pcounts, label='mcalens')

_ = plt.hist(true_pcounts, 20, density=True, color='k', histtype='step', label='true peak values')
_ = plt.hist(ml_pcounts, 20, density=True, histtype='step', label='ml peak values')
_ = plt.hist(ks_pcounts, 20, density=True, histtype='step', label='ks peak values')
_ = plt.hist(wiener_pcounts, 20, density=True, histtype='step', label='wiener peak values')
_ = plt.hist(sparse_pcounts, 20, density=True, histtype='step', label='sparse peak values')
plt.legend()

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

def plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=None, peak_thres=None):
    gb_std = gaussian_blur_std
    true_pcounts, true_snr = peak_count(img=true, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    ml_pcounts, ml_snr = peak_count(img=ml, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    ks_pcounts, ks_snr = peak_count(img=ks, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    wiener_pcounts, wiener_snr = peak_count(img=wiener, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    sparse_pcounts, sparse_snr = peak_count(img=sparse, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    mcalens_pcounts, mcalens_snr = peak_count(img=mcalens, gaussian_blur_std=gb_std, peak_thres=peak_thres)
    
    sns.set()
    fig = plt.figure(figsize=(7, 5))
    plt.title(f"peak count (gaus blur std = {gaussian_blur_std})")
    plt.hist(true_pcounts, bins=20, range=(0, 0.1), label='true', histtype='step', color='k', ls='--', lw=1.5)
    plt.hist(ml_pcounts, bins=20, range=(0, 0.1), label='ml', histtype='step', lw=1.5)
    plt.hist(ks_pcounts, bins=20, range=(0, 0.1), label='ks', histtype='step', lw=1.5)
    plt.hist(wiener_pcounts, bins=20, range=(0, 0.1), label='wiener', histtype='step', lw=1.5)
    plt.hist(sparse_pcounts, bins=20, range=(0, 0.1), label='sparse', histtype='step', lw=1.5)
    # plt.hist(mcalens_pcounts, bins=20, range=(0, 0.1), label='mcalens', histtype='step', lw=1.5)
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
    # plt.hist(mcalens_snr, bins=20, range=(1, 10), label='mcalens', histtype='step', lw=1.5)
    plt.xlabel(r"SNR $\nu = \kappa \, / \, \sigma_i$")
    plt.ylabel("number of peaks")
    plt.xlim(0, 10)
    plt.legend()
    plt.show()

# %%
# plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=3, peak_thres=0)
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=4.8762, peak_thres=0)

# %%
fig = plt.figure(figsize=(12,10))
plt.suptitle("Peak Count")
plt.subplot(3, 2, 1)
plt.title("gaus blur std = 0.1 (no blur)")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=0.1)
plt.subplot(3, 2, 2)
plt.title("gaus blur std = 0.5")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=0.5)
plt.subplot(3, 2, 3)
plt.title("gaus blur std = 1")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=1)
plt.subplot(3, 2, 4)
plt.title("gaus blur std = 3")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=3)
plt.subplot(3, 2, 5)
plt.title("gaus blur std = 4.8762 (1 arcmin)")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=4.8762)
plt.subplot(3, 2, 6)
plt.title("gaus blur std = 10")
plot_peak_counts(true, ml, ks, wiener, sparse, mcalens, gaussian_blur_std=10)
plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(7,5))
# plt.title("Kernel Density Plot")
# sns.kdeplot(true_pcounts, c='k', ls='--', label='true')
# sns.kdeplot(ml_pcounts, label='ml')
# sns.kdeplot(ks_pcounts, label='ks')
# sns.kdeplot(wiener_pcounts, label='wiener')
# sns.kdeplot(sparse_pcounts, label='sparse')
# # sns.kdeplot(mcalens_pcounts, label='mcalens')
# # plt.xlim(-0.05, 0.5)
# plt.xlabel(r"$\kappa$")
# plt.legend()
# plt.show()

# %% [markdown]
# ## Binned MSE Err

# %%
from summary_stats_func import plot_all_mse
print(f'ML_mse = {MSE(y_true=true, y_pred=ml):4f}')
print(f'KS_mse = {MSE(y_true=true, y_pred=ks):4f}')
print(f'Wiener_mse = {MSE(y_true=true, y_pred=wiener):4f}')
print(f'sparse_mse = {MSE(y_true=true, y_pred=sparse):4f}')
print(f'MCALens_mse = {MSE(y_true=true, y_pred=mcalens):4f}')

thresholds = np.arange(start=-0.035, stop=0.4, step=0.005)
plot_all_mse(thresholds, true, ml, ks, wiener, sparse, mcalens, mode='bin_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')
plot_all_mse(thresholds, true, ml, ks, wiener, sparse, mcalens, mode='min_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')
plot_all_mse(thresholds, true, ml, ks, wiener, sparse, mcalens, mode='max_thres', title='MSE ratio (bin threshold)', xlabel=r'$X < \kappa_{\rm{truth}} < X + 0.05$')

# %% [markdown]
# ### Pixel to Pixel Comparison

# %%
def draw_pix2pix(pred, true, label):
    fpred = sorted(pred.flatten())
    ftrue = sorted(true.flatten())
    stack = np.stack((ftrue, fpred), axis=1)
    plt.plot(stack[:, 0], stack[:, 1], label=label)

fig, ax = plt.subplots(figsize=(6, 6))
draw_pix2pix(ml, true, label='ML')
draw_pix2pix(ks, true, label='KS')
draw_pix2pix(wiener, true, label='Wiener')
draw_pix2pix(sparse, true, label='sparse5')
draw_pix2pix(mcalens, true, label='MCALens')
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

ftrue = sorted(true.flatten())
plt.title('flattened pix to pix comparison', fontsize=16)
plt.xlim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylim(min(ftrue)-0.05, max(ftrue)+0.05)
plt.ylabel(r'predicted $\hat{\kappa}$ pixel intensity')
plt.xlabel(r'true $\kappa$ pixel intensity')
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()

# %% [markdown]
# ### Power Spectrum

# %%
from summary_stats_func import P, plot_pspec
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
