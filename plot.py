import scipy.ndimage as ndimage
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from glob import glob

def read_fits(name):
    with pyfits.open(name, memmap=False) as f:
        pred = f[0].data * 100
        true = f[1].data * 100
        res = f[2].data * 100
    return pred, true, res

def draw(plot_id, data, title, scale=[-2, 5], cmap=plt.cm.jet, fontsize=18):
    plt.subplot(2, 2, plot_id)
    plt.imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.title(f'{title}', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])


fnames = sorted(glob('../result/prediction/*fits'))

for i in range(len(fnames)):
    name = fnames[i]
    pred, true, res = read_fits(name)
    res_smooth = ndimage.gaussian_filter(res, sigma=5, order=0)

    fig = plt.figure(figsize=(8,7))
    draw(1, pred, 'Prediction')
    draw(2, true, 'True')
    draw(3, res, 'Residual')
    draw(4, res_smooth, 'Res_smoothed')

    plt.tight_layout()
    fig.subplots_adjust(right=0.87)
    cax = plt.axes([0.88, 0.08, 0.04, 0.8])
    plt.colorbar(cax=cax, orientation="vertical")

    name = name.split('/')[-1]
    save_dir = '../result/fig/' + name + '.png'
    plt.savefig(save_dir)
    print(f"saved figure {i+1}: {name}")
    plt.close()
