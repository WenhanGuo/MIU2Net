# %%
import h5py
import numpy as np

def load_mat_file_v73(file_path):
    with h5py.File(file_path, 'r') as f:
        variables = {}
        for k, v in f.items():
            variables[k] = np.array(v)
    return variables

file_path = '/Volumes/Elements SE/density/cos0_Set1_rotate1_area1_Sigma2D.mat'
data = load_mat_file_v73(file_path)


# %%
