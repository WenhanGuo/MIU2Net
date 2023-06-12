import torch
import transforms as T
from my_dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
import os
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as fits


# define training device (cpu/gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device)


# define global parameters
def set_parameters():
    global test_num, data_dir
    test_num = 64
    data_dir = '/share/lirui/Wenhan/WL/data_new'
    return

def save_img(pred, true, res, fname):
    hdu = fits.PrimaryHDU(pred)
    hdu.writeto(fname, overwrite=True)
    fits.append(fname, true)
    fits.append(fname, res)
    return

def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('name', type=str, help='name of weights file')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    set_parameters()
    test_cat = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_cat = test_cat[:test_num]

    # load test dataset and dataloader
    test_args = dict(data_dir=data_dir, 
                     transforms=T.Compose([
                         T.ToTensor()
                         ]), 
                     gaus_noise=T.AddGaussianNoise(n_galaxy=50)
                     )
    test_data = ImageDataset(catalog=os.path.join(data_dir, 'test.csv'), **test_args)
    test_data = Subset(test_data, np.arange(test_num))
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
