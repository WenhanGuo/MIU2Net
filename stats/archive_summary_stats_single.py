# %%
import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import KDTree

fname = '/Users/danny/Desktop/WL/kappa_map/result/prediction_epoch70_aug5_native/map_24156.fits'
# Open image
with fits.open(fname) as f:
    pred = f[0].data
    true = f[1].data
    res = f[2].data

def draw(plot_id, data, title, scale=[-0.02, 0.05], cmap=plt.cm.jet, fontsize=18):
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
draw(1, pred, 'Prediction', scale=[true.min(), true.max()/2])
draw(2, true, 'True', scale=[true.min(), true.max()/2])
draw(3, gausblur_pred, 'Gaus Blur Prediction', scale=[true.min(), true.max()/2])
draw(4, gausblur_true, 'Gaus Blur True', scale=[true.min(), true.max()/2])

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
draw(1, gausblur_pred, f'Blurred Prediction ({len(pcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(pcoord[:, 1], pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw(2, gausblur_true, f'Blurred True ({len(tcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(tcoord[:, 1], tcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)

draw(3, gausblur_pred, f'Blurred Prediction ({len(matched_pcoord)} matched peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
plt.scatter(matched_pcoord[:, 1], matched_pcoord[:, 0], s=30, color='yellow', marker='+', linewidth=0.8)
draw(4, gausblur_true, f'Blurred True ({len(sub_tcoord)} peaks)', scale=[pred.min(), pred.max()], cmap='viridis')
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
fname = '/Users/danny/Desktop/WL/kappa_map/result/fig_faug15_ng1058.8_norotate/map_24009.fits'
# Open image
with fits.open(fname) as f:
    pred = f[0].data
    true = f[1].data
fig, ax = plt.subplots()
plt.scatter(pred.flatten(), true.flatten(), s=1, alpha=0.2)
plt.xlim(min(true.flatten())-0.05, max(true.flatten())+0.05)
plt.ylim(min(true.flatten())-0.05, max(true.flatten())+0.05)
plt.xlabel('predicted pixel intensity')
plt.xlabel('true pixel intensity')
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
