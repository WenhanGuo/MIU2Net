import torch
import argparse
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='visualize adding gaussian noise')
    parser.add_argument('path', type=str, help='full path to fits image')
    parser.add_argument('--mean', type=float, default=0., help='mean of gaussian noise to be added')
    parser.add_argument('--std', type=float, default=0.1951, help='std of gaussian noise to be added')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    img = fits.open(args.path)[0].data
    img = torch.tensor(np.float32(img))
    img_noisy = img + torch.randn(img.size()) * args.std + args.mean
    print('img =', img)
    print('img max =', img.max())
    print('img min =', img.min())
    print('noisy img =', img_noisy)
    print('noisy img max =', img_noisy.max())
    print('noisy img min =', img_noisy.min())
    
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

    im1 = ax1.imshow(img, cmap='viridis')
    ax1.set_title('True shear')
    
    im2 = ax2.imshow(img_noisy, cmap='viridis')
    ax2.set_title(f'Noisy shear, sigma={args.std}')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    plt.show()