# %%
from halo_nfw_model import HaloMap3D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

redshift_cat = pd.read_csv('/Users/danny/Desktop/WL/kappa_map/scripts/redshift_info.txt', sep=' ')
z_list = list(redshift_cat['z_lens'])

cat_name = '/Users/danny/Desktop/cos0_Set1_rotate1_area1.txt'
halo_cat = pd.read_csv(cat_name, sep='\t')
halo_cat.name = cat_name[:-4]
m = HaloMap3D(halo_cat, z_list)
xy = m.map_slice(z_idx=5, map_type='Sigma')
# m.map_all(map_type='Sigma')

# %%
plt.imshow(xy.T, interpolation='None', norm=colors.LogNorm())
plt.colorbar()
plt.title(r'Halo Surface Mass Density Map ($\mathrm{M}_{\odot}\mathrm{kpc}^{-2}$)'+f'\nz = {z_list[36]:.4}')

# %%
