# %%
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
import seaborn as sns
import scienceplots
plt.style.use(['science', 'nature'])
plt.rcParams.update({'figure.dpi': '300'})
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# %%
from summary_stats_func import read_folder
fnames = glob.glob('/Users/danny/Desktop/WL/kappa_map/result/final_k2d_fr16beta3_rmg20/*.fits')[:500]
true_cube, ml_cube, ks_cube, wiener_cube, sparse_cube, mcalens_cube = read_folder(fnames)

# %% [markdown]
# ## Qualitative Comparison
# %%
from summary_stats_func import draw6, cbar
# load cube containing results from all prediction methods
true = true_cube[0]
ml = ml_cube[0]
ks = ks_cube[0]
wiener = wiener_cube[0]
sparse = sparse_cube[0]
mcalens = mcalens_cube[0]

# 4 panels (true, KS, Wiener, ML)
fig = plt.figure(figsize=(3.5, 3.5))
d = dict(vmin=true.min(), vmax=true.max()/1.3, cmap='Spectral_r')
ax = plt.subplot(2, 2, 1)
im = ax.imshow(true, **d)
ax.set_title('Truth')
ax.add_patch(patches.Rectangle((79, 214), 20, 20, ls='--', lw=1, edgecolor='#9e0142', facecolor='none'))
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 2)
ax.imshow(ks, **d)
ax.set_title('KS')
ax.add_patch(patches.Rectangle((79, 214), 20, 20, ls='--', lw=1, edgecolor='#9e0142', facecolor='none'))
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 3)
ax.imshow(wiener, **d)
ax.set_title('WF')
ax.add_patch(patches.Rectangle((79, 214), 20, 20, ls='--', lw=1, edgecolor='#9e0142', facecolor='none'))
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 4)
ax.imshow(ml, **d)
ax.set_title(r'MIU$^2$Net')
ax.add_patch(patches.Rectangle((79, 214), 20, 20, ls='--', lw=1, edgecolor='#9e0142', facecolor='none'))
ax.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
fig.subplots_adjust(right=0.85)
plt.colorbar(im, cax=plt.axes([0.88, 0.08, 0.04, 0.8]), orientation="vertical")
plt.show()

# %%
from astropy.stats import sigma_clipped_stats
from photutils.datasets import make_4gaussians_image
from photutils.morphology import data_properties

def profile_stats(data):
    cat = data_properties(data)
    columns = ['label','xcentroid','ycentroid','fwhm','gini','eccentricity','orientation','kron_flux']
    tbl = cat.to_table(columns=columns)
    xcen, ycen, fwhm = tbl['xcentroid'][0], tbl['ycentroid'][0], tbl['fwhm'][0].value
    return fwhm / 256 * 1.75 * 60  # convert pix to arcmin

# Single peak profile visualization
def profile3d(subplot, data, title, zmin, zmax, true_xcen=9, true_ycen=9):
    xx, yy = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    d = dict(rstride=1, cstride=1, cmap='Spectral_r', linewidth=0)
    ax = fig.add_subplot(*subplot, projection='3d')
    im = ax.plot_surface(xx, yy, data, vmin=zmin, vmax=zmax, **d)
    ax.set_zlim(zmin, zmax)
    ax.set_title(rf'{title}', pad=-0.1)
    ax.tick_params(labelbottom=False, labelleft=False)

    dmax = data.max()
    ceil = np.argmax(data)
    ceilx, ceily = ceil // data.shape[0], ceil % data.shape[0]
    ax.plot([ceilx-1, ceilx+1], [ceily-1, ceily+1], [dmax, dmax], c='k', lw=0.6, zorder=10)
    ax.plot([ceilx+1, ceilx-1], [ceily-1, ceily+1], [dmax, dmax], c='k', lw=0.6, zorder=10)
    ax.text(ceilx-5, ceily-5, dmax+0.04, f'({ceilx-true_xcen}, {ceily-true_ycen}, {dmax:.3f})')
    fwhm = profile_stats(data)
    ax.text(25, 3, 0.22, rf'$\rm FWHM = {fwhm:.2f} \, arcmin$', transform=ax.transAxes)
    return im


fig = plt.figure(figsize=(3.5, 3.5))
subt = true[215:235, 80:100]  # single peak
subk = ks[215:235, 80:100]
subw = wiener[215:235, 80:100]
subm = ml[215:235, 80:100]
# subt = true[35:65, 160:190]  # double peak
# subk = ks[35:65, 160:190]
# subw = wiener[35:65, 160:190]
# subm = ml[35:65, 160:190]
# subt = true[10:40, 80:110]  # triple peak
# subk = ks[10:40, 80:110]
# subw = wiener[10:40, 80:110]
# subm = ml[10:40, 80:110]
tfwhm = profile_stats(subt)

im = profile3d((2, 2, 1), subt, 'Truth', subt.min(), subt.max())
_ = profile3d((2, 2, 2), subk, 'KS', subt.min(), subt.max())
_ = profile3d((2, 2, 3), subw, 'WF', subt.min(), subt.max())
_ = profile3d((2, 2, 4), subm, r'MIU$^2$Net', subt.min(), subt.max())

plt.tight_layout()
fig.subplots_adjust(right=0.85)
plt.colorbar(im, cax=plt.axes([0.88, 0.08, 0.04, 0.8]), orientation="vertical")
plt.show()

# %% [markdown]
# ## Dynamic range
# %%
fig = plt.figure(figsize=(4, 2))
ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=1, rowspan=1)
ax3 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1, rowspan=2)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
              linestyle="none", color='k', mec='k', mew=0.4, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.scatter(np.min(true_cube, axis=(1,2)), np.min(ml_cube, axis=(1,2)), s=0.2, alpha=0.6)
ax2.scatter(np.min(true_cube, axis=(1,2)), np.min(ks_cube, axis=(1,2)), s=0.2, alpha=0.6, c='#00B945')
ax1.scatter(np.min(true_cube, axis=(1,2)), np.min(wiener_cube, axis=(1,2)), s=0.2, alpha=0.6, c='#FF9500')
ax1.axline((0, 0), slope=1, linestyle='--', c='k', linewidth=0.5, alpha=0.8)
ax1.set_xlim(-0.038, -0.027)
ax2.set_xlim(-0.038, -0.027)
ax1.set_ylim(-0.064, -0.01)
ax2.set_ylim(-0.3, -0.52)
ax2.set_xticks([-0.0375, -0.0325, -0.0275])
ax2.set_xlabel(r'$\min(\pmb\kappa)$')
ax1.set_ylabel(r'$\min(\pmb{\hat\kappa})$')
ax1.yaxis.set_label_coords(-0.27, -0.03)

ax3.set_xlim(0, 0.95)
ax3.set_ylim(0, 0.95)
ax3.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
d = dict(scatter_kws=dict(s=0.2, alpha=0.6), line_kws=dict(lw=0.5), truncate=False)
sns.regplot(x=np.max(true_cube, axis=(1,2)), y=np.max(ml_cube, axis=(1,2)), label=r'MIU$^2$Net', **d)
sns.regplot(x=np.max(true_cube, axis=(1,2)), y=np.max(ks_cube, axis=(1,2)), label='KS', **d)
sns.regplot(x=np.max(true_cube, axis=(1,2)), y=np.max(wiener_cube, axis=(1,2)), label='WF', **d)
ax3.axline((0, 0), slope=1, linestyle='--', c='k', linewidth=0.5, alpha=0.8)
ax3.set_xlabel(r'$\max(\pmb\kappa)$')
ax3.set_ylabel(r'$\max(\pmb{\hat\kappa})$')
ax3.legend()

plt.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()

# %%
# 6 panels (+ sparse and MCALens)
fig = plt.figure(figsize=(6,4))
d = dict(scale=[true.min(), true.max()/2], cmap='Spectral_r')
draw6(1, true, 'True', **d)
draw6(2, ml, 'ML', **d)
draw6(3, ks, 'KS', **d)
draw6(4, wiener, 'WF', **d)
draw6(5, sparse, 'sparse', **d)
draw6(6, mcalens, 'MCALens', **d)
cbar(fig, cax=plt.axes([0.88, 0.08, 0.04, 0.8]))
plt.show()

# %% [markdown]
# ## Mass-Sheet Degeneracy
# %%
t = np.mean(true_cube, axis=(1,2))
m = np.mean(ml_cube, axis=(1,2))
w = np.mean(wiener_cube, axis=(1,2))
k = np.mean(ks_cube, axis=(1,2))

fig = plt.figure(figsize=(5, 2.4))
d = dict(scatter_kws=dict(s=0.5), line_kws=dict(lw=0.8), truncate=False)

ax = plt.subplot(1, 2, 1)
ax.set_xlim(-0.01, 0.012)
ax.set_ylim(-0.01, 0.012)
ax.set_yticks([-0.01, -0.005, 0, 0.005, 0.01])
sns.regplot(x=t, y=m, label=r'MIU$^2$Net', ci=95, **d)
# sns.kdeplot(x=t.flatten(), y=m.flatten(), levels=5, linewidths=0.7, color='#0C5DA5')
ax.scatter(t, k, s=0.5, label='KS', marker='.', alpha=0.5, c='#00B945')
ax.scatter(t, w, s=0.5, label='WF', marker='.', alpha=0.5, c='#FF9500')
ax.axline((0, 0), slope=1, linestyle='--', c='k', linewidth=0.8, alpha=0.8)
ax.set_ylabel(r'$\langle \pmb{\hat{\kappa}} \rangle$')
ax.set_xlabel(r'$\langle \pmb{\kappa} \rangle$')
plt.gca().set_aspect('equal')
plt.legend()

ax = plt.subplot(1, 2, 2)
ax.set_xlim(-0.01, 0.012)
ax.set_ylim(-0.01, 0.012)
ax.set_yticks([-0.01, -0.005, 0, 0.005, 0.01])
# sns.regplot(x=t, y=m*1.6526+0.00013, label=r'MIU$^2$Net (corrected)', ci=95, **d)
sns.regplot(x=t, y=m*1.7710+0.00018, label=r'MIU$^2$Net (corrected)', ci=95, **d)
# sns.kdeplot(x=t.flatten(), y=m.flatten()/0.6022, levels=5, linewidths=0.7, color='#0C5DA5')
ax.scatter(t, k, s=0.5, label='KS', marker='.', alpha=0.5, c='#00B945')
ax.scatter(t, w, s=0.5, label='WF', marker='.', alpha=0.5, c='#FF9500')
ax.axline((0, 0), slope=1, linestyle='--', c='k', linewidth=0.8, alpha=0.8)
ax.set_ylabel(r'$\mu_{\rm ms} \langle \pmb{\hat{\kappa}} \rangle + b_{\rm ms}$')
ax.set_xlabel(r'$\langle \pmb{\kappa} \rangle$')
plt.gca().set_aspect('equal')
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
# f_sparse, p_sparse, ci_sparse = avg_P(cube=sparse_cube, binsize=1.5, logspacing=False)
# f_mcalens, p_mcalens, ci_mcalens = avg_P(cube=mcalens_cube, binsize=1.5, logspacing=False)

# %%
# calculate frequency cutoff (r_max = 16)
s = 256
x, y = np.meshgrid(np.arange(-s/2, s/2), np.arange(-s/2, s/2), indexing='ij')
d = np.sqrt(x**2 + y**2)
yfreqs = xfreqs = np.fft.fftshift(np.fft.fftfreq(s))
yy_freq, xx_freq = np.meshgrid(yfreqs, xfreqs, indexing='ij')
freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)
mask = d < 15.5  # 15.5 because it's bin center not bin edge
plt.imshow(freqs_dist * mask)

max_freq = np.max(freqs_dist * mask)  # max_freq = 0.05975 pix when r_max = 16
max_k = 1 / (1/max_freq * 1.75*60/256)  # max spatial frequency in 1/arcmin
# %%
plt.figure(figsize=(3.5, 3.8))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax = plt.subplot(gs[0])
plot_pspec(xvals=f_true, ps1D=p_true, yerr=ci_true, logy=True, label=r'Truth $\langle P(\pmb\kappa) \rangle$', ls='--', c='k', errfmt='bar', lw=0.7)
plot_pspec(xvals=f_ml, ps1D=p_ml, yerr=ci_ml, logy=True, label=r'MIU$^2$Net', c='#0C5DA5', lw=0.7)
plot_pspec(xvals=f_ks, ps1D=p_ks, yerr=ci_ks, logy=True, label='KS', c='#00B945', lw=0.7)
plot_pspec(xvals=f_wiener, ps1D=p_wiener, yerr=ci_wiener, logy=True, label='WF', c='#FF9500', lw=0.7)
# plot_pspec(xvals=f_sparse, ps1D=p_sparse, yerr=ci_sparse, logy=True, label='sparse')
# plot_pspec(xvals=f_mcalens, ps1D=p_mcalens, yerr=ci_mcalens, logy=True, label='mcalens', c='tab:brown')
plt.axvline(x=max_k, color='k', linestyle=':', lw=0.7, alpha=0.3)
plt.axvspan(max_k, 2e-1, color='k', alpha=0.08)
ax.set_ylabel(r"$\langle P(\pmb{\hat\kappa}) \rangle$")
ax.set_xlim(1e-2, 2e-1)
ax.set_ylim(1e2, 3e4)

secax = ax.secondary_xaxis(location='top', functions=(lambda x: x*60*180/np.pi, lambda x: x/60/180*np.pi))
secax.set_xlabel(r'Multipole Number $l$')
secax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
secax.set_xticks([100, 200, 500])
secax.set_xticklabels([r'$1 \times 10^{2}$', r'$2 \times 10^{2}$', r'$5 \times 10^{2}$'])
# plt.grid(True)
plt.legend()

# power spectrum ratio (residual)
plt.subplot(gs[1])
plot_pspec(xvals=f_true, ps1D=p_ml/p_true-1, logy=False, label=r'MIU$^2$Net', errfmt=None, c='#0C5DA5', lw=0.7)
plot_pspec(xvals=f_true, ps1D=p_wiener/p_true-1, logy=False, label='WF', errfmt=None, c='#FF9500', lw=0.7)
plt.axvline(x=max_k, color='k', linestyle=':', lw=0.7, alpha=0.3)
plt.axvspan(max_k, 2e-1, color='k', alpha=0.08)
plt.axhline(y=0.0, color='k', linestyle='--', lw=0.7, alpha=0.7)
plt.axhspan(-0.04, 0.04, alpha=0.2)
plt.xlim(1e-2, 2e-1)
plt.ylim(-0.15, 0.15)
# plt.ylim(-0.4, 0.4)
plt.xlabel(r"Spatial Frequency $(\rm{arcmin}^{-1})$")
plt.ylabel(r"$\langle P(\pmb{\hat\kappa}) \rangle / \langle P(\pmb\kappa) \rangle - 1$")
plt.tight_layout()

# %% [markdown]
# ## 1D Distribution
# %%
# Plot 1D distribution
t = true_cube[0:200].flatten()
m = ml_cube[0:200].flatten()
k = ks_cube[0:200].flatten()
w = wiener_cube[0:200].flatten()

plt.figure(figsize=(6, 1.8))
plt.subplot(1, 3, 1)
sns.kdeplot(data=t/np.std(t), bw_adjust=0.2, label='Truth', color='k')
sns.kdeplot(data=m/np.std(t), bw_adjust=0.2, label=r'MIU$^2$Net', color='#0C5DA5')
sns.kdeplot(data=k/np.std(t), bw_adjust=0.2, label='KS', color='#00B945')
sns.kdeplot(data=w/np.std(t), bw_adjust=0.2, label='WF', color='#FF9500')
plt.xlabel(r'$\kappa$')
# plt.xlim(-0.05, 0.1)
plt.xlim(-3, 6)
plt.xticks([-2, 0, 2, 4, 6])
plt.legend()

# Plot 1D distribution normalized by std
plt.subplot(1, 3, 2)
sns.kdeplot(data=(t-np.mean(t))/np.std(t), bw_adjust=0.2, label='Truth', color='k')
sns.kdeplot(data=(m-np.mean(m))/np.std(m), bw_adjust=0.2, label=r'MIU$^2$Net', color='#0C5DA5')
sns.kdeplot(data=(k-np.mean(k))/np.std(k), bw_adjust=0.2, label='KS', color='#00B945')
sns.kdeplot(data=(w-np.mean(w))/np.std(w), bw_adjust=0.2, label='WF', color='#FF9500')
plt.xlabel(r'$(\kappa - \langle \kappa \rangle ) / \sigma_{\kappa}$')
plt.xlim(-4, 7)
plt.xticks([-2.5, 0.0, 2.5, 5.0])
plt.legend()

plt.subplot(1, 3, 3)
sns.kdeplot(data=(t-np.mean(t))/np.std(t), bw_adjust=0.2, label='Truth', color='k')
sns.kdeplot(data=(m-np.mean(m))/np.std(m), bw_adjust=0.2, label=r'MIU$^2$Net', color='#0C5DA5')
sns.kdeplot(data=(k-np.mean(k))/np.std(k), bw_adjust=0.2, label='KS', color='#00B945')
sns.kdeplot(data=(w-np.mean(w))/np.std(w), bw_adjust=0.2, label='WF', color='#FF9500')
plt.xlabel(r'$(\kappa - \langle \kappa \rangle ) / \sigma_{\kappa}$')
plt.xlim(2.5, 20)
plt.ylim(1e-7, 1e-1)
plt.xticks([4, 8, 12, 16, 20])
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Peak Count

# %%
from summary_stats_func import avg_peak_count
(true_avgp, true_pbin), (true_avgsnr, true_snrbin) = avg_peak_count(cube=true_cube, gaussian_blur_std=None, peak_thres=0)
(ml_avgp, ml_pbin), (ml_avgsnr, ml_snrbin) = avg_peak_count(cube=ml_cube, gaussian_blur_std=None, peak_thres=0)
(ks_avgp, ks_pbin), (ks_avgsnr, ks_snrbin) = avg_peak_count(cube=ks_cube, gaussian_blur_std=None, peak_thres=0)
(wiener_avgp, wiener_pbin), (wiener_avgsnr, wiener_snrbin) = avg_peak_count(cube=wiener_cube, gaussian_blur_std=None, peak_thres=0)
# (sparse_avgp, sparse_pbin), (sparse_avgsnr, sparse_snrbin) = avg_peak_count(cube=sparse_cube, gaussian_blur_std=None, peak_thres=0)
# (mcalens_avgp, mcalens_pbin), (mcalens_avgsnr, mcalens_snrbin) = avg_peak_count(cube=mcalens_cube, gaussian_blur_std=None, peak_thres=0)
# %%
fig = plt.figure(figsize=(4, 2))
plt.subplot(1, 2, 1)
plt.plot(true_pbin, true_avgp, label='Truth', c='k', ls='--')
plt.plot(wiener_pbin, wiener_avgp, label='WF', c='#FF9500')
plt.plot(ks_pbin, ks_avgp, label='KS', c='#00B945')
plt.plot(ml_pbin, ml_avgp, label=r'MIU$^2$Net', c='#0C5DA5', ls=':')
# plt.plot(sparse_pbin, sparse_avgp, label='sparse')
# plt.plot(mcalens_pbin, mcalens_avgp, label='mcalens')
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$n_{\rm peaks}$")
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(true_snrbin, true_avgsnr, label='Truth', c='k', ls='--')
plt.plot(wiener_snrbin, wiener_avgsnr, label='WF', c='#FF9500')
plt.plot(ks_snrbin, ks_avgsnr, label='KS', c='#00B945')
plt.plot(ml_snrbin, ml_avgsnr, label=r'MIU$^2$Net', c='#0C5DA5', ls=':')
# plt.plot(ml_snrbin, ml_avgsnr*7.4578, label=r'MIU$^2$Net (corrected)', c='#0C5DA5')
plt.plot(ml_snrbin, ml_avgsnr*8.5978, label=r'MIU$^2$Net (corrected)', c='#0C5DA5')
# plt.plot(sparse_snrbin, sparse_avgsnr, label='sparse')
# plt.plot(mcalens_snrbin, mcalens_avgsnr, label='mcalens')
plt.xlabel(r"SNR $\nu = \kappa \, / \, \sigma$")
plt.ylabel(r"$n_{\rm peaks}$")
plt.xlim(1, 6)
plt.xticks([1, 2, 3, 4, 5, 6])
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## MSE at different blur scales
# %%
from summary_stats_func import mse_at_all_scales
fwhm_arr = np.array([0.01, 0.05, 0.1, 0.2, 0.32, 0.5, 0.7, 1])   # in arcmin
std_arr = fwhm_arr / 0.2051 * 2 * np.sqrt(2 * np.log(2))   # pixel scale = 0.2051 arcmin/pix
ml_mse, ml_errbar = mse_at_all_scales(true_cube, pred_cube=ml_cube, gaussian_blur_std=std_arr)
ks_mse, ks_errbar = mse_at_all_scales(true_cube, pred_cube=ks_cube, gaussian_blur_std=std_arr)
wiener_mse, wiener_errbar = mse_at_all_scales(true_cube, pred_cube=wiener_cube, gaussian_blur_std=std_arr)
# sparse_mse, sparse_errbar = mse_at_all_scales(true_cube, pred_cube=sparse_cube, gaussian_blur_std=std_arr)
# mcalens_mse, mcalens_errbar = mse_at_all_scales(true_cube, pred_cube=mcalens_cube, gaussian_blur_std=std_arr)
# %%
plt.figure(figsize=(3, 2.5))
plt.errorbar(fwhm_arr, ml_mse, yerr=ml_errbar, capsize=1, capthick=0.5, fmt="-", marker='.', lw=0.5, label=r'MIU$^2$Net')
plt.errorbar(fwhm_arr, ks_mse, yerr=ks_errbar, capsize=1, capthick=0.5, fmt="-", marker='.', lw=0.5, label='KS')
plt.errorbar(fwhm_arr, wiener_mse, yerr=wiener_errbar, capsize=1, capthick=0.5, fmt="-", marker='.', lw=0.5, label='WF')
# plt.errorbar(fwhm_arr, sparse_mse, yerr=sparse_errbar, capsize=1, capthick=0.5, fmt="-", marker='.', lw=0.5, label='sparse')
# plt.errorbar(fwhm_arr, mcalens_mse, yerr=mcalens_errbar, capsize=1, capthick=0.5, fmt="-", marker='.', lw=0.5, label='mcalens')
plt.xlabel(r"$\sigma (\rm arcmin)$ note in paper: it's gaussian blur FWHM")
plt.ylabel(r"$\sqrt{\sum{(\rm{G}_\sigma(\pmb{\hat\kappa} - \pmb\kappa))^2} / \sum{\pmb\kappa^2}}$")
plt.ylim(0, 1.5)
plt.legend()

# %%
from astropy.io import fits
from astropy.modeling.models import Disk2D
from astropy.table import QTable
from photutils.datasets import make_model_sources_image
from skimage.transform import resize

gamma1 = fits.open('/Users/danny/Desktop/中科院/MNRAS paper/ml wl demo fits/cos0_Set1_rotate1_area1_37_gamma1.fits')[0].data[0:512, 0:512]
kappa = fits.open('/Users/danny/Desktop/中科院/MNRAS paper/ml wl demo fits/cos0_Set1_rotate1_area1_37_kappa.fits')[0].data[0:512, 0:512]

s = 512
n_galaxy = 20
sigma_e = 0.4   # rms amplitude of the intrinsic ellipticity distribution
theta_G = 3.5*60/(1024/s)/s   # pixel side length in arcmin (gaussian smoothing window)
variance = (sigma_e**2 / 2) / (theta_G**2 * n_galaxy)
std = np.sqrt(variance)
noise = np.random.randn(s, s) * std

s = 256
mask_frac = 0.2
n_sources = int(1100 * mask_frac)  # args.mask_frac = 0.2
model = Disk2D()
rng = np.random.default_rng(seed=None)
sources = QTable()
sources['amplitude'] = np.ones(n_sources)
sources['x_0'] = rng.uniform(low=0, high=s, size=n_sources)
sources['y_0'] = rng.uniform(low=0, high=s, size=n_sources)
sources['R_0'] = rng.power(a=0.3, size=n_sources) * 13
data = make_model_sources_image(shape=(s, s), 
                                model=model, 
                                source_table=sources)
mask = data < 1
fill_value = 0

# 4 panels ()
fig = plt.figure(figsize=(3.5, 3.5))
d = dict(vmin=gamma1.min(), vmax=gamma1.max(), cmap='YlGnBu_r')
ax = plt.subplot(2, 2, 1)
im = ax.imshow(resize(gamma1, (256, 256)), **d)
ax.set_title(r'Shear $\pmb{\gamma}_1$')
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 2)
g1 = gamma1/(1-kappa)
ax.imshow(resize(g1, (256, 256)), **d)
ax.set_title(r'Reduced Shear $\pmb{g}_1$')
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 3)
ax.imshow(mask, cmap='Greys_r')
ax.set_title('Data Mask')
ax.tick_params(labelbottom=False, labelleft=False)

ax = plt.subplot(2, 2, 4)
masked = resize(g1, (256, 256))
masked += resize(noise, (256, 256))
masked[mask == 0] = fill_value
ax.imshow(masked, **d)
ax.set_title(r'MIU$^2$Net Input')
ax.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
fig.subplots_adjust(right=0.85)
plt.colorbar(im, cax=plt.axes([0.88, 0.08, 0.04, 0.8]), orientation="vertical")
plt.show()
# %%
