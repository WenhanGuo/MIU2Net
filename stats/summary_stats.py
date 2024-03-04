# %%
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import astropy.io.fits as fits
matplotlib.rc_file_defaults()

# %%
from summary_stats_func import read_folder
fnames = glob.glob('/Users/danny/Desktop/WL/kappa_map/result/master_cubes_bare_g20/*.fits')[:500]
true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube, mcalens_cube = read_folder(fnames)
# ml_cube = ml_cube + wiener_cube

# %% [markdown]
# ## Peak Count

# %%
from summary_stats_func import avg_peak_count
# gb_std = 4.8762
gb_std = 0.1
peak_thres = 0
(true_avgp, true_pbin), (true_avgsnr, true_snrbin) = avg_peak_count(cube=true_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ml_avgp, ml_pbin), (ml_avgsnr, ml_snrbin) = avg_peak_count(cube=ml_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ks_avgp, ks_pbin), (ks_avgsnr, ks_snrbin) = avg_peak_count(cube=ks_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(wiener_avgp, wiener_pbin), (wiener_avgsnr, wiener_snrbin) = avg_peak_count(cube=wiener_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(sparse_avgp, sparse_pbin), (sparse_avgsnr, sparse_snrbin) = avg_peak_count(cube=sparse_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(mcalens_avgp, mcalens_pbin), (mcalens_avgsnr, mcalens_snrbin) = avg_peak_count(cube=mcalens_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)

# %%
# normalized peak count
from summary_stats_func import avg_peak_count
# gb_std = 4.8762
gb_std = 0.1
peak_thres = 0
(true_avgp, true_pbin), (true_avgsnr, true_snrbin) = avg_peak_count(cube=(true_cube)/np.std(true_cube), gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ml_avgp, ml_pbin), (ml_avgsnr, ml_snrbin) = avg_peak_count(cube=(ml_cube)/np.std(ml_cube), gaussian_blur_std=gb_std, peak_thres=peak_thres)
# (ks_avgp, ks_pbin), (ks_avgsnr, ks_snrbin) = avg_peak_count(cube=ks_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(wiener_avgp, wiener_pbin), (wiener_avgsnr, wiener_snrbin) = avg_peak_count(cube=(wiener_cube)/np.std(wiener_cube), gaussian_blur_std=gb_std, peak_thres=peak_thres)
# %%
sns.set()
fig = plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"peak count (gaus blur std = {gb_std})")
plt.plot(true_pbin, true_avgp, label='true', c='k', ls='--')
plt.plot(ml_pbin, ml_avgp*4.8, label='ml')
plt.plot(ks_pbin, ks_avgp, label='ks')
plt.plot(wiener_pbin, wiener_avgp, label='wiener')
plt.plot(sparse_pbin, sparse_avgp, label='sparse')
plt.plot(mcalens_pbin, mcalens_avgp, label='mcalens')
plt.xlabel(r"$\kappa$")
plt.ylabel("number of peaks")
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.title(f"peak count (gaus blur std = {gb_std})")
plt.plot(true_snrbin, true_avgsnr, label='true', c='k', ls='--')
plt.plot(ml_snrbin, ml_avgsnr*8, label='ml')
plt.plot(ks_snrbin, ks_avgsnr, label='ks')
plt.plot(wiener_snrbin, wiener_avgsnr, label='wiener')
plt.plot(sparse_snrbin, sparse_avgsnr, label='sparse')
plt.plot(mcalens_snrbin, mcalens_avgsnr, label='mcalens')
plt.xlabel(r"SNR $\nu = \kappa \, / \, \sigma_i$")
plt.ylabel("number of peaks")
plt.xlim(1, 6)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Power Spectrum

# %%
from summary_stats_func import avg_P, plot_pspec

# plot 1D power spectrums for truth & all recovery methods
f_true, p_true, ci_true = avg_P(cube=true_cube, binsize=1.5, logspacing=False)
f_ml, p_ml, ci_ml = avg_P(cube=ml_cube, binsize=1.5, logspacing=False)
f_ks, p_ks, ci_ks = avg_P(cube=ks_cube, binsize=1.5, logspacing=False)
f_wiener, p_wiener, ci_wiener = avg_P(cube=wiener_cube, binsize=1.5, logspacing=False)
f_sparse, p_sparse, ci_sparse = avg_P(cube=sparse_cube, binsize=1.5, logspacing=False)
f_mcalens, p_mcalens, ci_mcalens = avg_P(cube=mcalens_cube, binsize=1.5, logspacing=False)
# %%
plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
plt.subplot(gs[0])
plt.title('1D Power Spectrum')
plot_pspec(xvals=f_true, ps1D=p_true, yerr=ci_true, logy=True, label='true', c='k', errfmt='bar')
plot_pspec(xvals=f_ml, ps1D=p_ml, yerr=ci_ml, logy=True, label='ml', c='tab:blue')
# plot_pspec(xvals=f_ks, ps1D=p_ks, yerr=ci_ks, logy=True, label='ks')
plot_pspec(xvals=f_wiener, ps1D=p_wiener, yerr=ci_wiener, logy=True, label='wiener', c='tab:orange')
# plot_pspec(xvals=f_sparse, ps1D=p_sparse, yerr=ci_sparse, logy=True, label='sparse')
plot_pspec(xvals=f_mcalens, ps1D=p_mcalens, yerr=ci_mcalens, logy=True, label='mcalens', c='tab:brown')
plt.ylabel(r"$P(\kappa)$")
plt.xlim(right=2e-1)
plt.ylim(1e2, 3e4)
plt.grid(True)
plt.legend()

# power spectrum ratio (residual)
plt.subplot(gs[1])
plot_pspec(xvals=f_true, ps1D=p_ml/p_true, logy=False, label='ml', c='tab:blue')
plot_pspec(xvals=f_true, ps1D=p_wiener/p_true, logy=False, label='wiener', c='tab:orange')
plot_pspec(xvals=f_true, ps1D=p_mcalens/p_true, logy=False, label='mcalens', c='tab:brown')
plt.axhline(y=1.0, color='tab:red', linestyle='--', alpha=0.7)
plt.xlim(right=2e-1)
plt.ylim(0.9, 1.1)
plt.axhspan(0.97, 1.03, alpha=0.2)
plt.xlabel("Spatial Frequency (1 / arcmin)")
plt.ylabel(r"$P(\rm{pred}) \, / \, P(\rm{true})$")
plt.tight_layout()

# %%
# plot powerspec(true - pred) / powerspec(true)
f_res_ml, p_res_ml, _ = avg_P(cube=true_cube-ml_cube, binsize=1.5, logspacing=False)
# f_res_ks, p_res_ks, _ = avg_P(cube=true_cube-ks_cube, binsize=1.5, logspacing=False)
f_res_wiener, p_res_wiener, _ = avg_P(cube=true_cube-wiener_cube, binsize=1.5, logspacing=False)
f_res_sparse, p_res_sparse, _ = avg_P(cube=true_cube-sparse_cube, binsize=1.5, logspacing=False)
# f_res_mcalens, p_res_mcalens, _ = avg_P(cube=true_cube-mcalens_cube, binsize=1.5, logspacing=False)

plt.figure(figsize=(7, 5))
plt.title(r"Normalized Power Spectrum $P_\Delta$ of Residual $\hat{\kappa} - \kappa_{\rm{truth}}$")
plot_pspec(xvals=f_res_ml, ps1D=p_res_ml/p_true, logy=False, label='ml')
# plot_pspec(xvals=f_res_ks, ps1D=p_res_ks/p_true, logy=False, label='ks')
plot_pspec(xvals=f_res_wiener, ps1D=p_res_wiener/p_true, logy=False, label='wiener')
plot_pspec(xvals=f_res_sparse, ps1D=p_res_sparse/p_true, logy=False, label='sparse')
# plot_pspec(xvals=f_res_mcalens, ps1D=p_res_mcalens/p_true, logy=False, label='mcalens')
plt.xlabel("Spatial Frequency (1 / arcmin)")
plt.ylabel(r"$P_\Delta \, / \, P_{\rm{true}}$")
plt.xlim(xmin=10**-1.7)
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()

# %% [markdown]
# ## Binned MSE

# %%
from summary_stats_func import avg_rel_mse

thresholds = np.arange(start=0, stop=0.3, step=0.002)
mse_ks = avg_rel_mse(true_cube, ml_cube, ks_cube, thresholds, mode='bin_thres', binsize=0.05)
mse_wiener = avg_rel_mse(true_cube, ml_cube, wiener_cube, thresholds, mode='bin_thres', binsize=0.05)
mse_sparse = avg_rel_mse(true_cube, ml_cube, sparse_cube, thresholds, mode='bin_thres', binsize=0.05)
mse_mcalens = avg_rel_mse(true_cube, ml_cube, mcalens_cube, thresholds, mode='bin_thres', binsize=0.05)

# %%
sns.set()
plt.subplots(figsize=(7, 5))
plt.title("MSE ratio (bin threshold)")
plt.plot(thresholds, mse_ks, label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{KS}}$')
plt.plot(thresholds, mse_wiener, label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{wiener}}$')
plt.plot(thresholds, mse_sparse, label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{sparse}}$')
plt.plot(thresholds, mse_mcalens, label=r'MSE$_{\rm{ML}}$ / MSE$_{\rm{mcalens}}$')
plt.axhline(y = 1.0, color='r', linestyle='--', alpha=0.7)
plt.xlabel(r'$X < \kappa_{\rm{truth}} < X + 0.05$')
plt.legend()

# %%
from summary_stats_func import mean_ssim
print(mean_ssim(true_cube, pred_cube=ml_cube))
print(mean_ssim(true_cube, pred_cube=ks_cube))
print(mean_ssim(true_cube, pred_cube=wiener_cube))
print(mean_ssim(true_cube, pred_cube=sparse_cube))
print(mean_ssim(true_cube, pred_cube=mcalens_cube))

# %%
from summary_stats_func import mse_at_all_scales
fwhm_arr = np.array([0.01, 0.05, 0.1, 0.2, 0.32, 0.5, 0.7, 1])   # in arcmin
std_arr = fwhm_arr / 0.2051 * 2 * np.sqrt(2 * np.log(2))   # pixel scale = 0.2051 arcmin/pix
ml_mse, ml_errbar = mse_at_all_scales(true_cube, pred_cube=ml_cube, gaussian_blur_std=std_arr)
ks_mse, ks_errbar = mse_at_all_scales(true_cube, pred_cube=ks_cube, gaussian_blur_std=std_arr)
wiener_mse, wiener_errbar = mse_at_all_scales(true_cube, pred_cube=wiener_cube, gaussian_blur_std=std_arr)
sparse_mse, sparse_errbar = mse_at_all_scales(true_cube, pred_cube=sparse_cube, gaussian_blur_std=std_arr)
mcalens_mse, mcalens_errbar = mse_at_all_scales(true_cube, pred_cube=mcalens_cube, gaussian_blur_std=std_arr)

# %%
matplotlib.rc_file_defaults()
plt.figure(figsize=(7, 5))
plt.errorbar(fwhm_arr, ml_mse, yerr=ml_errbar, capsize=2, fmt="-", marker='.', lw=1, label='ml')
plt.errorbar(fwhm_arr, ks_mse, yerr=ks_errbar, capsize=2, fmt="-", marker='.', lw=1, label='ks')
plt.errorbar(fwhm_arr, wiener_mse, yerr=wiener_errbar, capsize=2, fmt="-", marker='.', lw=1, label='wiener')
plt.errorbar(fwhm_arr, sparse_mse, yerr=sparse_errbar, capsize=2, fmt="-", marker='.', lw=1, label='sparse')
plt.errorbar(fwhm_arr, mcalens_mse, yerr=mcalens_errbar, capsize=2, fmt="-", marker='.', lw=1, label='mcalens')
plt.xlabel("gaussian blur FWHM (arcmin)")
plt.ylabel(r"$\sqrt{\sum{(\rm{G}_\sigma(\hat{\kappa} - \kappa))^2} \, / \, \sum{\kappa^2}}$")
plt.ylim(0, 1.2)
plt.legend()

# %%
def draw_pix2pix(pred, true, label):
    fpred = sorted(pred.flatten())
    ftrue = sorted(true.flatten())
    # stack = np.stack((ftrue, fpred), axis=1)
    # plt.scatter(stack[:, 0], stack[:, 1], label=label)
    plt.plot(ftrue, fpred, label=label)

fig, ax = plt.subplots(figsize=(6, 6))
# draw_pix2pix(ml_cube[2], true_cube[2], label='ML')
draw_pix2pix(ks_cube[2], true_cube[2], label='KS')
draw_pix2pix(wiener_cube[2], true_cube[2], label='Wiener')
draw_pix2pix(sparse_cube[2], true_cube[2], label='sparse5')
draw_pix2pix(mcalens_cube[2], true_cube[2], label='MCALens')
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

for i in range(7):
    fpred = sorted(ml_cube[i].flatten())
    ftrue = sorted(true_cube[i].flatten())
    plt.plot(ftrue, fpred, c='b')

plt.title('flattened pix to pix comparison', fontsize=16)
plt.xlim(1e-5, 1e-1)
plt.ylim(1e-5, 1e-1)
plt.ylabel(r'predicted $\hat{\kappa}$ pixel intensity')
plt.xlabel(r'true $\kappa$ pixel intensity')
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.legend()
# %%
from astropy.visualization import hist

true = true_cube[1]
ml = ml_cube[1]
ks = ks_cube[1]
wiener = wiener_cube[1]
sparse = sparse_cube[1]

ml_res = ml - true
ks_res = ks - true
wiener_res = wiener - true
sparse_res = sparse - true
plt.title('PDF for pred - true')
hist(ml_res.flatten(), bins=200, range=(-0.1, 0.1), histtype='step', label='ml')
hist(wiener_res.flatten(), bins=200, range=(-0.1, 0.1), histtype='step', label='wiener')
hist(sparse_res.flatten(), bins=200, range=(-0.1, 0.1), histtype='step', label='sparse')
plt.legend()

# %%
plt.title('PDF for flattened maps')

# t = (true.flatten() - np.mean(true)) / np.std(true)
# m = (ml.flatten() - np.mean(ml)) / np.std(ml)
# k = (ks.flatten() - np.mean(ks)) / np.std(ks)
# w = (wiener.flatten() - np.mean(wiener)) / np.std(wiener)
# s = (sparse.flatten() - np.mean(sparse)) / np.std(sparse)

t = true.flatten()
m = ml.flatten()
k = ks.flatten()
w = wiener.flatten()
s = sparse.flatten()
hist(t.flatten(), bins=200, range=(-0.05, 0.1), histtype='step', label='true', color='k')
hist(m.flatten(), bins=200, range=(-0.05, 0.1), histtype='step', label='ml')
hist(k.flatten(), bins=200, range=(-0.05, 0.1), histtype='step', label='ks')
hist(w.flatten(), bins=200, range=(-0.05, 0.1), histtype='step', label='wiener')
hist(s.flatten(), bins=200, range=(-0.05, 0.1), histtype='step', label='sparse')
plt.legend()

# sns.kdeplot(data=t.flatten(), bw_adjust=0.05, label='true', color='k')
# sns.kdeplot(data=m.flatten(), bw_adjust=0.05, label='ml')
# sns.kdeplot(data=k.flatten(), bw_adjust=0.05, label='ks')
# sns.kdeplot(data=w.flatten(), bw_adjust=0.05, label='wiener')
# sns.kdeplot(data=s.flatten(), bw_adjust=0.05, label='sparse')
# plt.legend()
# plt.xlim(-0.1,0.1)

# %%
sns.set_theme(style="dark")
fig, ax = plt.subplots(figsize=(6, 6))
# sns.scatterplot(x=true.flatten(), y=ml_res.flatten(), s=5, color=".15", label='ML')
# sns.histplot(x=true.flatten(), y=ml_res.flatten(), bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=true.flatten()[:26000], y=ml_res.flatten()[:26000], levels=5, color="k", linewidths=1, label="ML")

# sns.scatterplot(x=true.flatten(), y=wiener_res.flatten(), s=5, label='Wiener')
# sns.histplot(x=true.flatten(), y=wiener_res.flatten(), bins=50, pthresh=.1, cmap="rocket")
sns.kdeplot(x=true.flatten()[:26000], y=wiener_res.flatten()[:26000], levels=5, color="b", linewidths=1, label="wiener")

plt.xlim(-0.06, 0.06)
plt.ylim(-0.06, 0.06)
plt.grid(True)
plt.legend()


# %%
sns.set_theme(style="dark")
fpred = np.log10((ml_cube[1]-true_cube[1].min()).flatten()[:26000])
ftrue = np.log10((true_cube[1]-true_cube[1].min()).flatten()[:26000])

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=ftrue, y=fpred, s=5, color=".15", label='ML')
sns.histplot(x=ftrue, y=fpred, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=ftrue, y=fpred, levels=5, color="w", linewidths=1)
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

plt.xlim(-3, 0)
plt.ylim(-3, 0)
plt.ylabel(r'log10 predicted $\hat{\kappa}$')
plt.xlabel(r'log10 true $\kappa$')
# ax.set_aspect('equal', adjustable='box')
plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
# %%
fpred = np.log10((wiener_cube[1]-true_cube[1].min()).flatten()[:26000])
ftrue = np.log10((true_cube[1]-true_cube[1].min()).flatten()[:26000])

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=ftrue, y=fpred, s=5, color=".15", label='Wiener')
sns.histplot(x=ftrue, y=fpred, bins=50, pthresh=.1, cmap="rocket")
sns.kdeplot(x=ftrue, y=fpred, levels=5, color="w", linewidths=1)
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

plt.xlim(-3, 0)
plt.ylim(-3, 0)
plt.ylabel(r'log10 predicted $\hat{\kappa}$')
plt.xlabel(r'log10 true $\kappa$')
# ax.set_aspect('equal', adjustable='box')
plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
# %%
fpred = np.log10((sparse_cube[1]-true_cube[1].min()).flatten()[:26000])
ftrue = np.log10((true_cube[1]-true_cube[1].min()).flatten()[:26000])

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=ftrue, y=fpred, s=5, color=".15", label='sparse')
sns.histplot(x=ftrue, y=fpred, bins=50, pthresh=.1, cmap="viridis")
sns.kdeplot(x=ftrue, y=fpred, levels=5, color="w", linewidths=1)
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)

plt.xlim(-3, 0)
plt.ylim(-3, 0)
plt.ylabel(r'log10 predicted $\hat{\kappa}$')
plt.xlabel(r'log10 true $\kappa$')
# ax.set_aspect('equal', adjustable='box')
plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
plt.legend()

# %%
# draw 2D Power Spectrum
from summary_stats_func import pspec, draw6, cbar
ps2D_true = pspec(img=true_cube[1])
ps2D_ml = pspec(img=ml_cube[1])
ps2D_ks = pspec(img=ks_cube[1])
ps2D_wiener = pspec(img=wiener_cube[1])
ps2D_sparse = pspec(img=sparse_cube[1])
ps2D_mcalens = pspec(img=mcalens_cube[1])

fig = plt.figure(figsize=(12,8))
d = dict(scale=(np.min(np.log10(ps2D_true)), np.max(np.log10(ps2D_true))), cmap='viridis')
draw6(1, np.log10(ps2D_true), 'True', **d)
draw6(2, np.log10(ps2D_ml), 'ML', **d)
draw6(3, np.log10(ps2D_ks), 'KS Deconvolution', **d)
draw6(4, np.log10(ps2D_wiener), 'Wiener Filtering', **d)
draw6(5, np.log10(ps2D_sparse), 'Sparse Recovery', **d)
draw6(6, np.log10(ps2D_mcalens), 'MCALens', **d)
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))
plt.show()

# %%
from summary_stats_func import draw6, cbar
# load cube containing results from all prediction methods
true = true_cube[0]
ml = ml_cube[1]
ks = ks_cube[2]
wiener = wiener_cube[3]
sparse = sparse_cube[4]
mcalens = mcalens_cube[5]

fig = plt.figure(figsize=(12,8))
d = dict(scale=[true.min(), true.max()/3], cmap='viridis')
draw6(1, true, 'True', **d)
draw6(2, ml+wiener, 'ml + wiener', **d)
draw6(3, ml, 'ml pred', **d)
draw6(4, wiener, 'Wiener Filtering', **d)
draw6(5, true-ml-wiener, 'true-ml-wiener', **d)
draw6(6, true-wiener, 'true-wiener', **d)
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))
plt.show()
# %%
# mass-sheet degeneracy
t = np.mean(true_cube, axis=(1,2))
m = np.mean(ml_cube, axis=(1,2))
w = np.mean(wiener_cube, axis=(1,2))

fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(t, m, s=3, label='ml')
sns.kdeplot(x=t.flatten(), y=m.flatten(), levels=5, color="tab:blue", linewidths=1, label="ML")
plt.scatter(t, w, s=3, label='wiener')
plt.title(r'mean $\kappa$ comparison', fontsize=16)
plt.axline((0, 0), slope=1, linestyle='--', c='r', alpha=0.7)
plt.ylabel(r'predicted $\hat{\kappa}$ mean')
plt.xlabel(r'true $\kappa$ mean')
plt.xlim(-0.01, 0.012)
plt.ylim(-0.01, 0.012)
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()
# %%
_ = plt.hist(true_cube.flatten(), bins=200, density=True)
_ = plt.hist(ml_cube.flatten(), bins=200, density=True)
_ = plt.hist(wiener_cube.flatten(), bins=200, density=True)
plt.xlim(-0.1, 0.1)

# %%
_ = plt.hist((true_cube.flatten()-np.mean(true_cube))/np.std(true_cube), bins=200, density=True)
_ = plt.hist((ml_cube.flatten()-np.mean(ml_cube))/np.std(ml_cube), bins=200, density=True)
_ = plt.hist((wiener_cube.flatten()-np.mean(wiener_cube))/np.std(wiener_cube), bins=200, density=True)
plt.xlim(-5, 7)
# %%
# Plot 1D distribution (Korea)
t = true_cube[0:200].flatten()
m = ml_cube[0:200].flatten()
w = wiener_cube[0:200].flatten()

sns.kdeplot(data=t, bw_adjust=0.05, label='true', color='k')
sns.kdeplot(data=m, bw_adjust=0.05, label='ml', color='b')
sns.kdeplot(data=w, bw_adjust=0.05, label='wiener', color='orange')
# plt.axvline(x=np.mean(t), color='k', linestyle='-', alpha=0.7)
# plt.axvline(x=np.mean(m), color='r', linestyle='--', alpha=0.7)
# plt.axvline(x=np.mean(w), color='orange', linestyle='--', alpha=0.7)
plt.xlim(-0.05, 0.1)
plt.legend()

# %%
# Plot normalized 1D distribution (Korea)
t = true_cube[0:200].flatten()
m = ml_cube[0:200].flatten()
w = wiener_cube[0:200].flatten()

# sns.kdeplot(data=(t-np.mean(t))/np.std(t), bw_adjust=0.05, label='true', color='k')
# sns.kdeplot(data=(m-np.mean(m))/np.std(m), bw_adjust=0.05, label='ml', color='b')
# sns.kdeplot(data=(w-np.mean(w))/np.std(w), bw_adjust=0.05, label='wiener', color='orange')
sns.kdeplot(data=(t)/np.std(t), bw_adjust=0.05, label='true', color='k')
sns.kdeplot(data=(m)/np.std(m), bw_adjust=0.05, label='ml', color='b')
sns.kdeplot(data=(w)/np.std(w), bw_adjust=0.05, label='wiener', color='orange')
plt.xlim(-5, 7)
plt.legend()

# %%
