import os
from glob import glob
import numpy as np
from astropy.io import fits
import pandas as pd


def make_cube(df, idx):
    img_paths = df.iloc[idx]
    cube = None
    for img_path in img_paths:
        with fits.open(img_path, memmap=False) as f:
            img = f[0].data
            img = np.expand_dims(img, axis=0)
            cube = np.concatenate([cube, img], axis=0) if cube is not None else img

    img_name = os.path.basename(img_path)
    cube_name = img_name.replace('_37_', '_')
    fits.writeto('/share/lirui/Wenhan/WL/gamma2_cube/'+cube_name, data=cube, overwrite=True)


if __name__ == '__main__':
    file_dict = {i: [] for i in range(1, 38)}
    img_paths = sorted(glob("/share/lirui/Wenhan/WL/gamma2/*.fits"))

    for img_path in img_paths:
        # Extract the number from the file name
        img_name = os.path.basename(img_path)
        num = int(img_name.split("_")[4])
        # Append the file name to the corresponding list in the dictionary
        file_dict[num].append(img_path)
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(file_dict)

    for i in range(len(df)):
        make_cube(df=df, idx=i)
