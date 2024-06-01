import torch
import transforms as T
from torchvision.transforms import GaussianBlur
from my_dataset import ImageDataset_density
from torch.utils.data import DataLoader, Subset
import os
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
        target_gb = GaussianBlur(kernel_size=5, sigma=2.0)
    else:
        target_gb = None
    # load test dataset and dataloader
    test_data = ImageDataset_density(catalog=catalog_name, 
                                     z_cat=args.zcat, 
                                     n_galaxy=args.n_galaxy, 
                                     lens_zslices=[10], 
                                     src_zslices=[15], 
                                     transforms=T.Compose([
                                         T.KS_rec(activate=args.ks), 
                                         T.RandomCrop(size=512)
                                     ]), 
                                     gaus_blur=target_gb)
    test_data = Subset(test_data, np.arange(args.num))
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=2)

    # load model weights
    model = torch.load(f'../models/{args.name}.pth')
    model.to(device=device, memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    model.eval()
    test_step = 0
    with torch.no_grad():
        for image, target in test_dataloader:
            image = image.to(device, memory_format=torch.channels_last)
            target = target.to(device, memory_format=torch.channels_last)
            y_pred = np.float32(model(image)[0][0].cpu())
            y_true = np.float32(target[0][0].cpu())
            res = y_true - y_pred
            map_name = test_cat['density'][test_step][10]
            base_name = os.path.basename(map_name)
            map_path = os.path.join('../result/prediction', base_name)
            save_img(y_pred, y_true, res, map_path)
            test_step += 1
            print(f'{test_step} imgs completed')


def get_args():
    parser = argparse.ArgumentParser(description='Predict density from halo map and shear')
    parser.add_argument('name', type=str, help='name of weights file')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("--num", default=8, type=int, help='number of test images to run')
    parser.add_argument("--dir", default='/ksmap', type=str, help='data directory')
    parser.add_argument("--zcat", default='/share/lirui/Wenhan/WL/kappa_map/scripts/redshift_info.txt', type=str, help='path to zcat')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--ks", default=False, action='store_true', help='predict kappa using KS deconvolution and make this an extra channel')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # define testing device (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    main(args)