from halo_nfw_model import HaloMap3D
import os
import multiprocessing
import pandas as pd
from glob import glob1
import astropy.io.fits as fits


def cat_to_map(halo_cat, z_list, map_type, out_dir):
    m = HaloMap3D(halo_cat, z_list)
    m.map_all(map_type=map_type)
    hdu = fits.PrimaryHDU(m.data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(out_dir, m.name+'_nfw.fits'), overwrite=True)

def multiprocess_wrapper(args, cat_name):
    halo_cat = pd.read_csv(os.path.join(args.cat_dir, cat_name), sep='\t')
    halo_cat.name = cat_name[:-4]
    cat_to_map(halo_cat, args.z_list, args.map_type, args.out_dir)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convert halo catalog to halo NFW profile FITS cubes')
    parser.add_argument("--cat-dir", default='/ksmap/ks/halocat', type=str, help='directory where all halo catalog .txt files are')
    parser.add_argument("--out-dir", default='/ksmap/ks/halomap', type=str, help='directory to which output FITS cubes should save')
    parser.add_argument("--zcat-path", default='./redshift_info.txt', type=str, help='full path to redshift_info.txt')
    parser.add_argument("--num", default=6144, type=int, help='number of catalog .txt files to convert to FITS cube')
    parser.add_argument("--cpu", default=64, type=int, help='number of cpu cores to use for multiprocessing')
    parser.add_argument("--map-type", default='Sigma', type=str, choices=['rho', 'Sigma'], help='map rho (density) or Sigma (projected surface density)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    halo_cats = sorted(glob1(args.cat_dir, '*area[0-9].txt'))[:args.num]
    redshift_cat = pd.read_csv(args.zcat_path, sep=' ')
    args.z_list = list(redshift_cat['z_lens'])

    with multiprocessing.Pool(processes=args.cpu) as pool:
        arguments = [(args, cat_name) for cat_name in halo_cats]
        pool.starmap(func=multiprocess_wrapper, iterable=arguments, chunksize=10)