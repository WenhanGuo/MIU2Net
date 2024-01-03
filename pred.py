import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from model import UNet
import torch
import transforms as T
from my_dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from torchvision.transforms.functional import resize


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

    # load test dataset and dataloader
    test_data = ImageDataset(catalog=catalog_name, 
                             args=args, 
                             transforms=T.Compose([
                                 T.KS_rec(args), 
                                 T.CenterCrop(size=args.crop), 
                                 T.Wiener(args), 
                                 T.sparse(args), 
                                 T.MCALens(args)])
                             )
    test_data = Subset(test_data, np.arange(args.num))
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=args.cpu)

    # initialize UNet model
    in_channels = 2
    if args.ks == 'add':
        in_channels += 1
    if args.wiener == 'add':
        in_channels += 1
    if args.sparse == 'add':
        in_channels += 1
    if args.mcalens == 'add':
        in_channels += 1
    elif args.ks == 'only' or args.wiener == 'only':
        in_channels = 1
    print('in_channels =', in_channels)
    model = UNet(n_channels=in_channels)

    # load model weights
    print(f'initializing model using {args.load}.pth')
    state_dict = torch.load('../models/'+args.load+'.pth', map_location=device)
    remove_prefix = 'module.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)
    torch.cuda.empty_cache()

    model.eval()
    test_step = 0
    with torch.no_grad():
        for image, target in test_dataloader:
            image = image.to(device, memory_format=torch.channels_last)
            target = target.to(device, memory_format=torch.channels_last)
            outputs = model(image)
            if type(outputs) == list:
                y_pred = np.float32(outputs[0][0].cpu())
            else:
                y_pred = np.float32(outputs[0].cpu())
            if args.wiener_res == True:
                y_true = np.float32((target[0] + image[0][2]).cpu())
                y_targ = np.float32(target[0].cpu())
                res = y_targ - y_pred
                cube = np.concatenate([np.float32(image[0].cpu()), y_true, y_targ, y_pred, res], axis=0)
            else:
                y_true = np.float32(target[0].cpu())
                res = y_true - y_pred
                cube = np.concatenate([np.float32(image[0].cpu()), y_true, y_pred, res], axis=0)
            '''
            image.shape = torch.Size([1, 2, 512, 512])
            target.shape = torch.Size([1, 1, 512, 512])
            len(outputs) = 7
            outputs[0].shape = torch.Size([1, 1, 512, 512]) ==> reconstructed image
            outputs[1].shape = torch.Size([1, 1, 512, 512]) ==> Laplacian pyramid finest layer
            outputs[2].shape = torch.Size([1, 1, 256, 256])
            outputs[3].shape = torch.Size([1, 1, 128, 128])
            outputs[4].shape = torch.Size([1, 1, 56, 56])
            outputs[5].shape = torch.Size([1, 1, 32, 32])
            outputs[6].shape = torch.Size([1, 1, 16, 16]) ==> Laplacian pyramid coarest layer
            y_pred.shape = y_true.shape = res.shape = (1, 512, 512)
            cube.shape = (n, 512, 512), n = shear*2 + ks + wf + sp + mca + pred + true + res + laplacian_pyramid*6
            '''
            for y_side in outputs[1:]:
                upsampled = resize(y_side, size=cube.shape[-2:], interpolation='nearest')
                cube = np.concatenate([cube, np.float32(upsampled[0].cpu())], axis=0)
            print('writeto cube shape =', cube.shape)

            map_name = test_cat['kappa'][test_step][0]
            base_name = os.path.basename(map_name)
            map_path = os.path.join('../result/prediction', 'pred_'+base_name)
            fits.writeto(map_path, data=cube, overwrite=True)
            test_step += 1
            print(f'{test_step} imgs completed')

def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('load', type=str, help='name of weights file')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("--num", default=8, type=int, help='number of test images to run')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_1024_2d', type=str, help='data directory')
    parser.add_argument("--cpu", default=4, type=int, help='number of cpu cores to use')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--crop", default=512, type=int, help='crop 1024x1024 kappa to this size')
    parser.add_argument("--resize", default=256, type=int, help='downsample kappa to this size')
    parser.add_argument("--ks", default='off', type=str, choices=['off', 'add', 'only'], help='KS93 deconvolution (no KS, KS as an extra channel, no shear and KS only)')
    parser.add_argument("--wiener", default='off', type=str, choices=['off', 'add', 'only'], help='Wiener reconstruction')
    parser.add_argument("--sparse", default='off', type=str, choices=['off', 'add', 'only'], help='sparse reconstruction')
    parser.add_argument("--mcalens", default='off', type=str, choices=['off', 'add', 'only'], help='MCAlens reconstruction')
    parser.add_argument("--wiener-res", default=False, action='store_true', help='if the target is true - wiener')
    parser.add_argument("--save-noisy-shear", default=True, type=bool, help='write shear with added gaussian noise to disk')
    parser.add_argument("--save-noisy-shear-dir", default='/share/lirui/Wenhan/WL/kappa_map/result/noisy_shear', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.save_noisy_shear == False:
        print('WARNING: Not saving noisy shear.')

    # define testing device (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    main(args)