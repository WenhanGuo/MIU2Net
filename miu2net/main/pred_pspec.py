import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from model import u2net_full
import torch
import transforms as T
from my_dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from stats.summary_stats_func import P


def main(args):
    catalog_name = os.path.join(args.dir, 'validation.ecsv')
    test_cat = Table.read(catalog_name)
    assert args.num_avg <= len(test_cat)
    test_cat = test_cat[:args.num_avg]

    # load test dataset and dataloader
    test_data = ImageDataset(catalog=catalog_name, 
                             args=args, 
                             transforms=T.Compose([
                                 T.ToTensor(), 
                                 T.ReducedShear(args), 
                                 T.AddGaussianNoise(args), 
                                 T.KS_rec(args), 
                                 T.CenterCrop(size=args.crop), 
                                 T.Wiener(args), 
                                 T.sparse(args), 
                                 T.MCALens(args), 
                                 T.Resize(size=args.resize), 
                                 T.AddStarMask(args)])
                             )
    test_data = Subset(test_data, np.arange(args.num_avg))
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=args.cpu)

    # initialize U2Net model
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
    model = u2net_full(in_ch=in_channels)
    
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
    pspec_arr = np.zeros([args.num_it, 26])

    with torch.no_grad():
        for it in range(args.num_it):
            avg_pspec = np.zeros(26)  # average pspec per batch (500)
            for image, target in test_dataloader:
                image = image.to(device, memory_format=torch.channels_last)
                target = target.to(device, memory_format=torch.channels_last)
                outputs = model(image)
                if type(outputs) == list:
                    y_pred = np.float32(outputs[0][0].cpu())
                else:
                    y_pred = np.float32(outputs[0].cpu())
                # print('writeto cube shape =', cube.shape)

                freqs, ps = P(y_pred[0], binsize=1.5, logspacing=False)
                avg_pspec = avg_pspec + ps / args.num_avg
                test_step += 1
                # print(f'{test_step} imgs completed')
            pspec_arr[it] = avg_pspec
            print(f'{it+1} iterations completed')
        np.save('../pspec/pspec_arr.npy', pspec_arr)

def get_args():
    parser = argparse.ArgumentParser(description='Predict kappa from test shear')
    parser.add_argument('load', type=str, help='name of weights file')
    parser.add_argument("-g", "--n-galaxy", default=50, type=float, help='number of galaxies per arcmin (to determine noise level)')
    parser.add_argument("--noise-seed", default=1, type=int, help='how many noise realizations for each training image; 0 for new realization every time')
    parser.add_argument("--num-avg", default=500, type=int, help='number of test images to average per iteration')
    parser.add_argument("--num-it", default=10, type=int, help='number of iterations to run for std calculation')
    parser.add_argument("--dir", default='/share/lirui/Wenhan/WL/data_1024_2d', type=str, help='data directory')
    parser.add_argument("--cpu", default=4, type=int, help='number of cpu cores to use')
    parser.add_argument("--gaus-blur", default=False, action='store_true', help='whether to blur shear before feeding into ML')
    parser.add_argument("--crop", default=512, type=int, help='crop 1024x1024 kappa to this size')
    parser.add_argument("--resize", default=256, type=int, help='downsample kappa to this size')
    parser.add_argument("--reduced-shear", default=False, action='store_true', help='use reduced shear (g) instead of shear (gamma)')
    parser.add_argument("--mask-frac", default=0, type=float, help='randomly mask this fraction of pixels')
    parser.add_argument("--rand-mask-frac", default=False, action='store_true', help='if true, then randomize 0% - mask-frac% masked pixels; otherwise always generate mask-frac%')
    parser.add_argument("--ks", default='off', type=str, choices=['off', 'add', 'only'], help='KS93 deconvolution (no KS, KS as an extra channel, no shear and KS only)')
    parser.add_argument("--wiener", default='off', type=str, choices=['off', 'add', 'only'], help='Wiener reconstruction')
    parser.add_argument("--sparse", default='off', type=str, choices=['off', 'add', 'only'], help='sparse reconstruction')
    parser.add_argument("--mcalens", default='off', type=str, choices=['off', 'add', 'only'], help='MCAlens reconstruction')
    parser.add_argument("--wiener-res", default=False, action='store_true', help='if the target is true - wiener')
    parser.add_argument("--save-noisy-shear", default=True, type=bool, help='write shear with added gaussian noise to disk')
    parser.add_argument("--save-noisy-shear-dir", default='/share/lirui/Wenhan/WL/kappa_map/miu2net/result/noisy_shear', type=str)
    parser.add_argument("--cosmo2", default=False, action='store_true', help='if using cosmology2')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.save_noisy_shear == False:
        print('WARNING: Not saving noisy shear.')

    # define testing device (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    main(args)