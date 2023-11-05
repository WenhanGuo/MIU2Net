# %%
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc_file_defaults()

# %%
from summary_stats_func import read_folder
fnames = glob.glob('/Users/danny/Desktop/master_cubes/*.fits')[:100]
true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube, mcalens_cube = read_folder(fnames)

# %% [markdown]
# ## Peak Count

# %%
from summary_stats_func import avg_peak_count
# gb_std = 4.8762
gb_std = 1
peak_thres = 0
(true_avgp, true_pbin), (true_avgsnr, true_snrbin) = avg_peak_count(cube=true_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ml_avgp, ml_pbin), (ml_avgsnr, ml_snrbin) = avg_peak_count(cube=ml_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(ks_avgp, ks_pbin), (ks_avgsnr, ks_snrbin) = avg_peak_count(cube=ks_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(wiener_avgp, wiener_pbin), (wiener_avgsnr, wiener_snrbin) = avg_peak_count(cube=wiener_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(sparse_avgp, sparse_pbin), (sparse_avgsnr, sparse_snrbin) = avg_peak_count(cube=sparse_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)
(mcalens_avgp, mcalens_pbin), (mcalens_avgsnr, mcalens_snrbin) = avg_peak_count(cube=mcalens_cube, gaussian_blur_std=gb_std, peak_thres=peak_thres)

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
plt.plot(mcalens_pbin, mcalens_avgp, label='mcalens')
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
f_true, p_true = avg_P(cube=true_cube, logspacing=False)
f_ml, p_ml = avg_P(cube=ml_cube, logspacing=False)
f_ks, p_ks = avg_P(cube=ks_cube, logspacing=False)
f_wiener, p_wiener = avg_P(cube=wiener_cube, logspacing=False)
f_sparse, p_sparse = avg_P(cube=sparse_cube, logspacing=False)
f_mcalens, p_mcalens = avg_P(cube=mcalens_cube, logspacing=False)

plt.figure(figsize=(7, 5))
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
f_res_ml, p_res_ml = avg_P(cube=true_cube-ml_cube, logspacing=False)
f_res_ks, p_res_ks = avg_P(cube=true_cube-ks_cube, logspacing=False)
f_res_wiener, p_res_wiener = avg_P(cube=true_cube-wiener_cube, logspacing=False)
f_res_sparse, p_res_sparse = avg_P(cube=true_cube-sparse_cube, logspacing=False)
f_res_mcalens, p_res_mcalens = avg_P(cube=true_cube-mcalens_cube, logspacing=False)

plt.figure(figsize=(7, 5))
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
