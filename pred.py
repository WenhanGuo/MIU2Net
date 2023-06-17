import torch
import transforms as T
from my_dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
import os
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as fits


def save_img(pred, true, res, fname):
    hdu = fits.PrimaryHDU(pred)
    hdu.writeto(fname, overwrite=True)
    fits.append(fname, true)
    fits.append(fname, res)
    return


def main(args):
    test_cat = pd.read_csv(os.path.join(args.dir, 'test.csv'))
    assert args.num <= len(test_cat)
    test_cat = test_cat[:args.num]

    # load test dataset and dataloader
    test_args = dict(data_dir=args.dir, 
                     transforms=T.Compose([
                         T.ToTensor()
                         ]), 
                     gaus_noise=T.AddGaussianNoise(n_galaxy=args.n_galaxy)
                     )
    test_data = ImageDataset(catalog=os.path.join(args.dir, 'test.csv'), **test_args)
    test_data = Subset(test_data, np.arange(args.num))
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

    # load model weights
    model = torch.load(f'./models/{args.name}.pth')
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    torch.cuda.empty_cache()

    model.eval()
    test_step = 0
    with torch.no_grad():
        for gamma, kappa in test_dataloader:
            gamma = gamma.to(device, memory_format=torch.channels_last)
            kappa = kappa.to(device, memory_format=torch.channels_last)
            y_pred = np.float32(model(gamma)[0][0].cpu())
            y_true = np.float32(kappa[0][0].cpu())
            res = y_true - y_pred
            map_name = test_cat['kappa'][test_step]
            map_path = os.path.join('./result/prediction', map_name)
            save_img(y_pred, y_true, res, map_path)
            test_step += 1


def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('name', type=str, help='name of weights file')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("--num", default=32, type=int, help='number of test images to run')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_new', type=str, help='data directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # define testing device (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    main(args)