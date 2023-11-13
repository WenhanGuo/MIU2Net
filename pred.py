import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import transforms as T
from torchvision.transforms import GaussianBlur
from my_dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


def save_img(pred, true, res, fname):
    hdu = fits.PrimaryHDU(pred)
    hdu.writeto(fname, overwrite=True)
    fits.append(fname, true)
    fits.append(fname, res)
    return


def main(args):
    catalog_name = os.path.join(args.dir, 'test.ecsv')
    test_cat = Table.read(catalog_name)
    assert args.num <= len(test_cat)
    test_cat = test_cat[:args.num]

    if args.gaus_blur == True:
        shear_gb = GaussianBlur(kernel_size=5, sigma=2.0)
    else:
        shear_gb = None
    # load test dataset and dataloader
    test_data = ImageDataset(catalog=catalog_name, 
                             n_galaxy=args.n_galaxy, 
                             transforms=T.Compose([
                                 T.KS_rec(args), 
                                 T.RandomCrop(size=512), 
                                 T.Wiener(args), 
                                 T.sparse(args), 
                                 T.MCALens(args)
                                 ]), 
                             gaus_blur=shear_gb
                             )
    test_data = Subset(test_data, np.arange(args.num))
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=args.cpu)

    # load model weights
    model = torch.load(f'../models/{args.name}.pth')
    model.to(device=device, memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    model.eval()
    test_step = 0
    with torch.no_grad():
        for image, target in test_dataloader:
            image_g = image.to(device, memory_format=torch.channels_last)
            target_g = target.to(device, memory_format=torch.channels_last)
            y_pred = np.float32(model(image_g)[0].cpu())
            y_true = np.float32(target_g[0].cpu())
            res = y_true - y_pred
            map_name = test_cat['kappa'][test_step][0]
            base_name = os.path.basename(map_name)
            map_path = os.path.join('../result/prediction', 'pred_'+base_name)

            cube = np.concatenate([np.float32(image.cpu())[0], y_pred, y_true, res], axis=0)
            print('writeto cube shape =', cube.shape)
            fits.writeto(map_path, data=cube, overwrite=True)
            test_step += 1
            print(f'{test_step} imgs completed')

def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('name', type=str, help='name of weights file')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("--num", default=8, type=int, help='number of test images to run')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_1024_2d', type=str, help='data directory')
    parser.add_argument("--cpu", default=4, type=int, help='number of cpu cores to use')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--ks", default='off', type=str, choices=['off', 'add', 'only'], help='KS93 deconvolution (no KS, KS as an extra channel, no shear and KS only)')
    parser.add_argument("--wiener", default='off', type=str, choices=['off', 'add', 'only'], help='Wiener reconstruction')
    parser.add_argument("--sparse", default='off', type=str, choices=['off', 'add', 'only'], help='sparse reconstruction')
    parser.add_argument("--mcalens", default='off', type=str, choices=['off', 'add', 'only'], help='MCAlens reconstruction')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # define testing device (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    main(args)