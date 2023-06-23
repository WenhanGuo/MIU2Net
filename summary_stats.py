# %%
import numpy as np
import scipy.ndimage as ndimage
from glob import glob
import astropy.io.fits as fits
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.detection import find_peaks
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
N = len(fnames)
avg_err = np.zeros((512,512))
pred_cube, true_cube = [], []

for i in range(N):
    name = fnames[i]
    pred, true, res = read_fits(name)
    pred_cube.append(pred)
    true_cube.append(true)
pred_cube, true_cube = np.array(pred_cube), np.array(true_cube)

# %%
pred_cube = np.delete(pred_cube, [50], axis=0)
true_cube = np.delete(true_cube, [50], axis=0)
# pred_cube = np.delete(pred_cube, [222, 226, 254, 392], axis=0)
# true_cube = np.delete(true_cube, [222, 226, 254, 392], axis=0)
N = len(pred_cube)
err_cube = (pred_cube - true_cube)
avg_err = np.mean(err_cube, axis=0)

fig, axes = plt.subplots(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(avg_err, vmin=-0.005, vmax=0.001)
plt.title('Mean Absolute Error (signed)')
plt.colorbar()

# %%
_ = plt.hist(pred_cube.flatten(), 200, range=(-0.05,0.1), density=True, alpha=0.7, label='pred distribution')
_ = plt.hist(true_cube.flatten(), 200, range=(-0.05,0.1), density=True, alpha=0.7, label='true distribution')
plt.title('Histogram for 576-4 images')
plt.xlabel('pixel intensity')
plt.ylabel('probability density')
plt.legend()

# %%
_ = plt.hist(pred_cube.flatten(), 200, range=(-0.05,0.1), density=True, cumulative=True, alpha=0.7, label='pred distribution')
n, bins, _ = plt.hist(true_cube.flatten(), 200, range=(-0.05,0.1), density=True, cumulative=True, alpha=0.7, label='true distribution')
plt.title('Cumulative Histogram for 576-4 images')
plt.xlabel('pixel intensity')
plt.ylabel('probability density')
plt.legend()

one_third = bins[sum(~(n > 0.333))]
two_thirds = bins[sum(~(n > 0.667))]
print("1/3 threshold =", one_third,"\n2/3 threshold =", two_thirds)

# %%
err_map = np.zeros((3, 512, 512))
def threshold_img(img, threshold_range):
    mask = (img >= threshold_range[0]) == (img < threshold_range[1])
    return np.ma.MaskedArray(img, ~mask)

def rel_err(pred, targ):
    return (pred - targ)

for idx in range(N):
    pred, true = pred_cube[idx], true_cube[idx]
    high_true = threshold_img(true, threshold_range=(two_thirds, np.inf))
    mid_true = threshold_img(true, threshold_range=(one_third, two_thirds))
    low_true = threshold_img(true, threshold_range=(-np.inf, one_third))

    # cannot write += because masks will misfunction
    err_map[0] = err_map[0] + rel_err(pred, high_true) / N * 3
    err_map[1] = err_map[1] + rel_err(pred, mid_true) / N * 3
    err_map[2] = err_map[2] + rel_err(pred, low_true) / N * 3

tot_err = [np.sum(err_map[i].flatten()) for i in range(len(err_map))]
print(np.array(tot_err)/512/512)

# %%
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

vmin = err_map.mean()
vmax = err_map.mean()

fig = plt.figure(figsize=(8,7))
draw(1, err_map[0], 'high kappa err', scale=[-0.03, 0.02])
draw(2, err_map[1], 'mid kappa err', scale=[-0.03, 0.02])
draw(3, err_map[2], 'low kappa err', scale=[-0.03, 0.02])
cbar(cax=plt.axes([0.88, 0.08, 0.04, 0.8]))

# %%
pred_peak_values, true_peak_values = [], []
avg_abs_err, avg_rel_err = 0, 0

for idx in range(N):
    pred, true = pred_cube[idx], true_cube[idx]
    pcoord = peak_local_max(pred, min_distance=5, num_peaks=100, threshold_rel=0.1)
    tcoord = peak_local_max(true, min_distance=10, num_peaks=50, threshold_rel=0.1)

    kdtree = KDTree(pcoord)
    d, i = kdtree.query(tcoord)
    matched_coord = pcoord[i]
    n_unique = len(np.unique(matched_coord, axis=0))

    img_peak_pval = pred[tuple(zip(*matched_coord))]
    img_peak_tval = true[tuple(zip(*tcoord))]
    pred_peak_values.append(img_peak_pval)
    true_peak_values.append(img_peak_tval)

    abs_err = abs(img_peak_pval - img_peak_tval)
    rel_err = abs(img_peak_pval - img_peak_tval) / abs(img_peak_tval)
    avg_abs_err += abs_err / N
    avg_rel_err += rel_err / N

pred_peak_values = np.array(pred_peak_values)
true_peak_values = np.array(true_peak_values)

_ = plt.hist(pred_peak_values.flatten(), 200, density=True, alpha=0.7, label='pred peak values')
_ = plt.hist(true_peak_values.flatten(), 200, density=True, alpha=0.7, label='true peak values')
plt.legend()

print('avg abs err per peak =', sum(abs_err)/len(tcoord))
print('avg rel err per peak =', sum(rel_err)/len(tcoord))

# %%
