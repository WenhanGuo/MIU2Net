# %%
import numpy as np
import scipy.ndimage as ndimage
from glob import glob
import astropy.io.fits as fits
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import KDTree


def read_fits(name):
    with fits.open(name, memmap=False) as f:
        pred = f[0].data
        # pred = ndimage.gaussian_filter(pred, sigma=1, order=0)
        true = f[1].data
        # true = ndimage.gaussian_filter(true, sigma=1, order=0)
        res = f[2].data
    return pred, true, res

fnames = sorted(glob('../result/prediction_epoch70_aug5_native/*fits'))
# fnames = sorted(glob('../result/prediction_epoch51_aug3_native/*fits'))
N = len(fnames)
avg_err = np.zeros((512,512))
pred_cube, true_cube = [], []

for i in range(N):
    name = fnames[i]
    pred, true, res = read_fits(name)
    pred_cube.append(pred)
    true_cube.append(true)
pred_cube, true_cube = np.array(pred_cube), np.array(true_cube)

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

# %%
# Whole-field mean relative error
pred_cube = np.delete(pred_cube, [50], axis=0)
true_cube = np.delete(true_cube, [50], axis=0)
# pred_cube = np.delete(pred_cube, [222, 226, 254, 392], axis=0)
# true_cube = np.delete(true_cube, [222, 226, 254, 392], axis=0)
N = len(pred_cube)
err_cube = 2 * ((pred_cube) - (true_cube)) / (abs(pred_cube) + abs(true_cube))
avg_err = np.mean(err_cube, axis=0)

fig, axes = plt.subplots(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(avg_err, vmin=-0.25, vmax=0.25)
plt.title('Mean Relative Error (signed)')
plt.colorbar()

# %%
# Pixel intensity histogram
_ = plt.hist(pred_cube.flatten(), 200, range=(-0.05,0.1), weights=np.ones(len(pred_cube.flatten()))/len(pred_cube.flatten()), alpha=0.7, label='pred distribution')
_ = plt.hist(true_cube.flatten(), 200, range=(-0.05,0.1), weights=np.ones(len(pred_cube.flatten()))/len(pred_cube.flatten()), alpha=0.7, label='true distribution')
plt.title('Histogram for 576-1 images')
plt.xlabel('pixel intensity')
plt.ylabel('probability density')
plt.legend()

# %%
# Pixel intensity histogram (cumulative)
_ = plt.hist(pred_cube.flatten(), 200, range=(-0.05,0.1), density=True, cumulative=True, alpha=0.7, label='pred distribution')
n, bins, _ = plt.hist(true_cube.flatten(), 200, range=(-0.05,0.1), density=True, cumulative=True, alpha=0.7, label='true distribution')
plt.title('Cumulative Histogram for 576-1 images')
plt.xlabel('pixel intensity')
plt.ylabel('probability density')
plt.legend()

one_third = bins[sum(~(n > 0.333))]
two_thirds = bins[sum(~(n > 0.667))]
print("1/3 threshold =", one_third,"\n2/3 threshold =", two_thirds)

# %%
# High, mid, low density kappa relative errors
err_map = np.zeros((3, 512, 512))
def threshold_img(img, threshold_range):
    mask = (img >= threshold_range[0]) == (img < threshold_range[1])
    return np.ma.MaskedArray(img, ~mask)

def rel_err(pred, targ):
    return 2 * (pred - targ) / (abs(pred) + abs(targ))

for idx in range(N):
    pred, true = pred_cube[idx], true_cube[idx]
    high_true = threshold_img(true, threshold_range=(two_thirds, np.inf))
    mid_true = threshold_img(true, threshold_range=(one_third, two_thirds))
    low_true = threshold_img(true, threshold_range=(-np.inf, one_third))

    # cannot write += because masks will misfunction
    err_map[0] = err_map[0] + rel_err(pred, high_true) / N * 3
    err_map[1] = err_map[1] + rel_err(pred, mid_true) / N * 3
    err_map[2] = err_map[2] + rel_err(pred, low_true) / N * 3

fig = plt.figure(figsize=(8,7))
draw(1, err_map[0], 'high kappa relative err', scale=[-1.8,1.2])
draw(2, err_map[1], 'mid kappa relative err', scale=[-1.8,1.2])
draw(3, err_map[2], 'low kappa relative err', scale=[-1.8,1.2])
cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

tot_err = [np.sum(err_map[i].flatten()) for i in range(len(err_map))]
print('avg image err: high, mid, low =', np.array(tot_err))
print('avg pixel err: high, mid, low =', np.array(tot_err)/512/512)

# %%
# Single peak profile analysis
subpred = pred_cube[0][195:225, 245:275]   # high peak
subtrue = true_cube[0][195:225, 245:275]
# subpred = pred_cube[0][160:200, 120:160]   # mid peak
# subtrue = true_cube[0][160:200, 120:160]
xx, yy = np.mgrid[0:subpred.shape[0], 0:subpred.shape[1]]
fig, [ax1,ax2] = plt.subplots(1,2,subplot_kw={"projection": "3d"}, figsize=(16,12))
ax1.set_zlim(subtrue.min(),subtrue.max())
ax1.plot_surface(xx, yy, subpred, rstride=1, cstride=1, cmap='viridis', linewidth=0)
ax2.plot_surface(xx, yy, subtrue, rstride=1, cstride=1, cmap='viridis', linewidth=0)
ax1.set_title('Prediction')
ax2.set_title('True')

# %%
# Peak analysis
pred_peak_values, true_peak_values = [], []

for idx in range(N):
    pred, true = pred_cube[idx], true_cube[idx]
    pcoord = peak_local_max(pred, min_distance=3, num_peaks=200, threshold_rel=0.1)
    tcoord = peak_local_max(true, min_distance=5, num_peaks=100, threshold_rel=0.1)

    kdtree = KDTree(pcoord)
    distances, tree_idxes = kdtree.query(tcoord)

    matched_pcoord, sub_tcoord = [], []
    for j in range(len(distances)):
        if distances[j] <= 3:
            matched_pcoord.append(pcoord[tree_idxes[j]])
            sub_tcoord.append(tcoord[j])
    matched_pcoord, sub_tcoord = np.array(matched_pcoord), np.array(sub_tcoord)

    img_peak_pval = pred[tuple(zip(*matched_pcoord))]
    img_peak_tval = true[tuple(zip(*sub_tcoord))]
    pred_peak_values.append(img_peak_pval.flatten())
    true_peak_values.append(img_peak_tval.flatten())
pred_peak_values = np.hstack(pred_peak_values)
true_peak_values = np.hstack(true_peak_values)

abs_err = abs(pred_peak_values - true_peak_values)
rel_err = abs(pred_peak_values - true_peak_values) / abs(true_peak_values)

# %%
npeaks = len(pred_peak_values)
_ = plt.hist(pred_peak_values, 200, range=(0.0, 1.0), weights=np.ones(npeaks)/npeaks, alpha=0.7, label='pred peak values')
_ = plt.hist(true_peak_values, 200, range=(0.0, 1.0), weights=np.ones(npeaks)/npeaks, alpha=0.7, label='true peak values')
plt.title(f'Histogram for {npeaks} peaks')
plt.xlabel('pixel intensity')
plt.ylabel('probability density')
plt.legend()

print(f'avg abs err per peak = {sum(abs_err)/npeaks:.3f}')
print(f'avg rel err per peak = {sum(rel_err)/npeaks:.3f}')
print(f'avg number of peaks per image = {npeaks/N:.2f}')

# %%
