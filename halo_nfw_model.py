import numpy as np
from astropy.constants import G
from astropy.cosmology import WMAP9
import astropy.units as u
import astropy.cosmology.units as cu


def distance_to_npixel(dis, z):
    d = z.to(u.Mpc, cu.with_redshift(WMAP9))
    theta = ((dis / d)*u.rad).to(u.arcmin)   # angle span for dis at redshift z plane
    pixel_scale = (3.5*u.deg / 1024).to(u.arcmin)   # angle span for a pixel at redshift z plane
    return theta / pixel_scale   # how many pixels does dis span

def npixel_to_distance(npix, z):
    d = z.to(u.Mpc, cu.with_redshift(WMAP9))
    pixel_scale = (3.5*u.deg / 1024).to(u.rad)
    return npix * d * pixel_scale.value


class OneHalo(object):

    def __init__(self, M, z):
        self.H = WMAP9.H(z)     # quantity
        self.M = M              # quantity; input needs to be in M☉
        self.z = z              # quantity; input needs to be in cu.redshift
        self.rho_crit = self.crit_density().value           # scalar; unit = M☉/kpc^3
        self.c_NFW = self.concentration().value             # scalar; unitless
        self.delta_s = self.overdensity()                   # scalar; unitless
        self.r_s = self.scale_radius().to(u.Mpc).value      # scalar; unit = Mpc
        self.r_vir = self.virial_radius().to(u.Mpc).value   # scalar; unit = Mpc
    
    # ρ_crit: critical density of the universe = 3H^2 / 8πG
    def crit_density(self):
        rho_crit = (3 * self.H**2) / (8 * np.pi * G)
        return rho_crit.to(u.Msun / u.kpc**3)

    # c_NFW: concentration parameter of the halo (Bullock 2001)
    # this relation has a large scatter (Jing 2000)
    def concentration(self):
        h = WMAP9.H0.value / 100
        return 8/(1+self.z) * (self.M / (1e14/h*u.Msun))**(-0.13)

    # δ_s: overdensity where δ_s * ρ_crit = ρ_s in NFW equation
    def overdensity(self):
        delta_vir = 200   # threshold overdensity for spherical collapse
        c = self.c_NFW
        return delta_vir/3 * c**3 / (np.log(1+c) - c/(1+c))

    # r_s: scale radius = 1/c * (MG/100H^2)^(1/3)
    def scale_radius(self):
        return 1/self.c_NFW * (self.M*G / (100*self.H**2))**(1/3)

    # r_vir: virial radius = c_NFW * r_s = (MG/100H^2)^(1/3)
    def virial_radius(self):
        return (self.M*G / (100*self.H**2))**(1/3)

    # get halo center coordinate in 1024*1024 img. input x1 and x2 should be quantities: kpc
    def center_coord(self, x1, x2, area_cen):
        x_cen = 511.5 + distance_to_npixel(dis=x1-area_cen[0], z=self.z).value
        y_cen = 511.5 + distance_to_npixel(dis=x2-area_cen[1], z=self.z).value
        self.coord = np.array((x_cen, y_cen))
        return self.coord

    # get NFW density for an input radius (in Mpc!), unit = M☉/kpc^3
    def call_NFW_density(self, r):
        y = r / self.r_s
        rho = np.select(condlist=[r <= self.r_vir, r > self.r_vir], choicelist=
                            [
                                self.delta_s * self.rho_crit / (y * (1+y)**2),
                                0
                            ])
        return rho
    
    # get line-of-sight surface mass density for an input projected radius (in Mpc!), unit = M☉/kpc^2
    def call_surface_mass_density(self, r_proj):
        y = r_proj / self.r_s

        def f(y):
            c = self.c_NFW
            c2 = c**2
            y2 = y**2
            rt = np.sqrt
            return np.select(condlist=[y < 1, y == 1, ((y > 1) & (y <= c)), y > c], choicelist=
                    [
                        -(rt(c2-y2)/(1-y2)/(1+c)) + 1/(1-y2)**(3/2) * np.arccosh((y2+c)/(y*(1+c))),
                        rt(c2-1)/(3*(1+c)) * (1+1/(1+c)),
                        -(rt(c2-y2)/(1-y2)/(1+c)) - 1/(y2-1)**(3/2) * np.arccos((y2+c)/(y*(1+c))), 
                        0
                    ])
        
        Sigma = 2 * self.delta_s * self.rho_crit * self.r_s * f(y)   # Σ(y) = 2 * ρ_s * r_s * f(y)

        return Sigma * 1000   # M☉/kpc^3 * Mpc = M☉/kpc^2 * 1000


class HaloMap3D(object):
    
    def __init__(self, halo_cat, z_list):
        self.name = str(halo_cat.name)
        self.nslice = 37
        self.z_list = z_list
        assert self.nslice == len(z_list)
        d_list = self.z_list.to(u.Mpc, cu.with_redshift(WMAP9)).value
        self.d_list = np.array(d_list)                              # arr of scalar; unit = Mpc
        self.pixel_scale = (3.5*u.deg / 1024).to(u.rad).value       # scalar; unit = rad
        self.pix2mpc_list = self.d_list * self.pixel_scale          # arr of scalar; unit = Mpc

        area = str(self.name[-5:])
        area_cen = np.select(
                condlist=[area=='area1', area=='area2', area=='area3', area=='area4'], 
                choicelist=[np.array([80000,  80000]),
                            np.array([80000,  240000]),
                            np.array([240000, 80000]),
                            np.array([240000, 240000])]
                            )
        self.area_cen = area_cen / 0.7 * u.kpc
        self.halos = self.init_halos_from_cat(halo_cat)

    def init_halos_from_cat(self, halo_cat):
        halos = []
        halo_cat = halo_cat.groupby('z_cen')
        keys = np.array(list(halo_cat.groups.keys()))
        key_id = 0

        for n in range(self.nslice):
            if self.z_list.value[n].round(5) not in keys.round(5):
                halos.append([])
            else:
                sub_cat = halo_cat.get_group(keys[key_id])
                key_id += 1
                halos_subgroup = []
                redshift = self.z_list[n]
                for idx, h_meta in sub_cat.iterrows():
                    mass = h_meta.mass / 0.7 * u.Msun
                    x1, x2 = h_meta.x1 / 0.7 * u.kpc, h_meta.x2 / 0.7 * u.kpc
                    h = OneHalo(M=mass, z=redshift)
                    h.center_coord(x1=x1, x2=x2, area_cen=self.area_cen)
                    halos_subgroup.append(h)
                halos.append(halos_subgroup)
        
        return halos

    def map_slice(self, z_idx, map_type='Sigma'):
        assert map_type in ['rho', 'Sigma']
        x_cen = np.arange(1024) + 0.5
        y_cen = np.arange(1024) + 0.5
        xx, yy = np.meshgrid(x_cen, y_cen, indexing='xy')
        xy = np.zeros((1024, 1024))

        for idx, halo in enumerate(self.halos[z_idx]):
            x_dis, y_dis = halo.coord[0]-xx, halo.coord[1]-yy
            pix_dis = np.linalg.norm(np.array([x_dis, y_dis]), axis=0)
            comoving_dis = pix_dis * self.pix2mpc_list[z_idx]
            if map_type == 'rho':
                xy += halo.call_NFW_density(r=comoving_dis)
            if map_type == 'Sigma':
                xy += halo.call_surface_mass_density(r_proj=comoving_dis)

        return xy
    
    def map_all(self, map_type='Sigma'):
        data = None
        for z_idx in range(self.nslice):
            xy = self.map_slice(z_idx=z_idx, map_type=map_type)
            xy = np.expand_dims(xy, axis=0)
            data = np.concatenate([data, xy], axis=0) if data is not None else xy
        self.data = data