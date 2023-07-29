import numpy as np
import pandas as pd
from astropy.constants import G
from astropy.cosmology import WMAP9
import astropy.units as u
import astropy.cosmology.units as cu


def distance_to_npixel(dis, z):
    """Helper function to convert distance at redshift z to the number of pixels it spans"""
    d = z.to(u.Mpc, cu.with_redshift(WMAP9))
    theta = ((dis / d)*u.rad).to(u.arcmin)   # angle span for dis at redshift z plane
    pixel_scale = (3.5*u.deg / 1024).to(u.arcmin)   # angle span for a pixel at redshift z plane
    return theta / pixel_scale   # how many pixels does dis span

def npixel_to_distance(npix, z):
    """Helper function to convert number of pixels at redshift z to physical distance in Mpc"""
    d = z.to(u.Mpc, cu.with_redshift(WMAP9))
    pixel_scale = (3.5*u.deg / 1024).to(u.rad)
    return npix * d * pixel_scale.value


class OneHalo(object):
    """
    A class for an NFW-like dark matter halo.

    Takes the virial mass (M_vir) and redshift (z) to initialize. 
    Once initialized, you can access properties such as
        critical density of the universe (ρ_crit), 
        concentration parameter of the halo (c_vir), 
        overdensity of halo (δ_s), 
        density parameter (normalization factor) of the halo (ρ_s), 
        scale radius (r_s), 
        virial radius (r_vir).
    Calling center_coord will add the 'coord' property to object.
    Calling call_NFW_density or call_BMO_density returns ρ(r).
    Calling call_NFW_surface_mass_density or call_BMO_surface_mass_density returns Σ(r).
    Density and surface mass density calls allow input radius to be in both scalar and ndarray.

    Attributes
    ----------
    H : astropy.units.quantity.Quantity
        quantity for Hubble parameter; redshift dependent

    M_vir : astropy.units.quantity.Quantity
        quantity in M☉ for the virial mass of the halo
    
    z : astropy.units.quantity.Quantity
        quantity in cu.redshift (unitless) for the redshift of the halo

    rho_crit : np.float64
        scalar for the critical density of the universe; implicit unit = M☉/kpc^3

    c_vir : np.float64
        scalar for the concentration parameter of the halo; unitless
        c_vir = r_vir / r_s

    delta_s : np.float64
        scalar for the overdensity of the halo; unitless
    
    rho_s : np.float64
        scalar for the density parameter of the halo; implicit unit = M☉/kpc^3
        ρ_s = δ_s * ρ_crit
    
    r_s : np.float64
        scalar for the scaled radius of the halo; implicit unit = Mpc
    
    r_vir : np.float64
        scalar for the virial radius of the halo; implicit unit = Mpc

    coord : ndarray; shape = (2,)
        xy coordinate of the halo center on the 1024 * 1024 grid

    Methods
    -------
    center_coord(x1, x2, area_cen)
        Adds the 'coord' property to object
    
    call_NFW_density(self, r: np.ndarray) -> np.ndarray
        Returns ρ_TJ(r), NFW density for input radius
        This NFW profile is truncated at virial radius by the Heaviside step function
        so it is effectively a TJ profile (Takada & Jain 2003a,b)

    call_BMO_density(self, r: np.ndarray, tau_V: float = 5., n: int = 2) -> np.ndarray
        Returns ρ_BMO(r), BMO density for input radius
        This NFW profile is truncated in a different form (Baltz, Marshall & Oguri 2009)
        so that shear and convergence profiles are differentiable at truncation radius
        which leads to convergence in the flexion profile
        At virial radius, density turns over to ρ_BMO(r) ∝ r^(-7), which well-approximates a cutoff
        The total mass of a BMO halo converges

    call_NFW_surface_mass_density(self, r: np.ndarray) -> np.ndarray
        Returns Σ_TJ(r), projected surface mass density for NFW profile
        Σ_TJ(r) becomes undefined when r > virial radius
        so it is truncated at virial radius by the Heaviside step function

    call_BMO_surface_mass_density(self, r: np.ndarray, tau_V: float = 5.) -> np.ndarray
        Returns Σ_BMO(r), projected surface mass density for BMO profile
        Σ_BMO(r) is defined everywhere except at r = scale radius
    """

    def __init__(self, M_vir: u.quantity.Quantity, z: u.quantity.Quantity) -> None:
        self.H = WMAP9.H(z)
        self.M_vir = M_vir
        self.z = z
        self.rho_crit = self.crit_density()
        self.c_vir = self.concentration()
        self.delta_s = self.overdensity()
        self.rho_s = self.delta_s * self.rho_crit
        self.r_s = self.scale_radius()
        self.r_vir = self.virial_radius()
    
    def crit_density(self) -> np.float64:
        """ρ_crit: critical density of the universe = 3H^2 / 8πG"""
        rho_crit = (3 * self.H**2) / (8 * np.pi * G)

        return rho_crit.to(u.Msun / u.kpc**3).value

    def concentration(self) -> np.float64:
        """c_vir: concentration parameter of the halo (Oguri 2011)"""
        h = WMAP9.H0.value / 100

        return (7.26 * (self.M_vir / (1e12/h*u.Msun))**(-0.086) * (1+self.z)**(-0.71)).value

    def overdensity(self) -> np.float64:
        """δ_s: overdensity where δ_s * ρ_crit = ρ_s"""
        delta_vir = 200   # threshold overdensity for spherical collapse
        c = self.c_vir

        return delta_vir/3 * c**3 / (np.log(1+c) - c/(1+c))

    def scale_radius(self) -> np.float64:
        """r_s: scale radius = 1/c * (MG/100H^2)^(1/3)"""
        r_s = 1/self.c_vir * (self.M_vir*G / (100*self.H**2))**(1/3)

        return r_s.to(u.Mpc).value

    def virial_radius(self) -> np.float64:
        """r_vir: virial radius = c_vir * r_s = (MG/100H^2)^(1/3)"""
        r_vir = (self.M_vir*G / (100*self.H**2))**(1/3)

        return r_vir.to(u.Mpc).value

    def center_coord(self, x1, x2, area_cen) -> None:
        """Adds the 'coord' property to object.

        Parameters
        ----------
        x1 : astropy.units.quantity.Quantity
            value from the 'x1' column in halo catalog; horizontal coordinate in box
            x1 should be a quantity in u.kpc as an input
        
        x2 : astropy.units.quantity.Quantity
            value from the 'x2' column in halo catalog; vertical coordinate in box
            x2 should be a quantity in u.kpc as an input

        area_cen : ndarray; shape = (2,)
            center coordinate of the halo-containing area in the simulation box
        """

        x_cen = 511.5 + distance_to_npixel(dis=x1-area_cen[0], z=self.z).value
        y_cen = 511.5 + distance_to_npixel(dis=x2-area_cen[1], z=self.z).value
        self.coord = np.array((x_cen, y_cen))

    def call_NFW_density(self, r: np.ndarray) -> np.ndarray:
        """get NFW density for an input radius (in Mpc!), ρ unit = M☉/kpc^3"""

        y = r / self.r_s
        rho = np.select(condlist=[r <= self.r_vir, r > self.r_vir], choicelist=
                            [
                                self.rho_s / (y * (1+y)**2),
                                0
                            ])
        return rho
    
    def call_BMO_density(self, r: np.ndarray, tau_V: float = 5., n: int = 2) -> np.ndarray:
        """get BMO density for an input radius (in Mpc!), ρ unit = M☉/kpc^3"""

        r_t = tau_V * self.r_vir
        x = r / self.r_s
        rho_NFW = self.rho_s / (x * (1+x)**2)
        truncate = (r_t**2 / (r**2 + r_t**2))**n
        return rho_NFW * truncate
    
    def call_NFW_surface_mass_density(self, r: np.ndarray) -> np.ndarray:
        """get NFW surface mass density for an input radius (in Mpc!), Σ unit = M☉/kpc^2"""

        y = r / self.r_s

        def f(y):
            c = self.c_vir
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
        
        Sigma = 2 * self.rho_s * self.r_s * f(y)   # Σ(y) = 2 * ρ_s * r_s * f(y)

        return Sigma * 1000   # M☉/kpc^3 * Mpc = M☉/kpc^2 * 1000
    
    def call_BMO_surface_mass_density(self, r: np.ndarray, tau_V: float = 5.) -> np.ndarray:
        """get BMO surface mass density for an input radius (in Mpc!), Σ unit = M☉/kpc^2
        This only corresponds to the n = 2 case BMO profile."""

        x = r / self.r_s
        tau = tau_V * self.c_vir

        def F(x):
            return np.select(condlist=[x < 1, x > 1], choicelist=
                             [
                                 1/np.sqrt(1-x**2) * np.arctanh(np.sqrt(1-x**2)),
                                 1/np.sqrt(x**2-1) * np.arctan(np.sqrt(x**2-1))
                             ])
        def L(x, tau):
            return np.log(x / (np.sqrt(tau**2+x**2) + tau))
        
        x2, tau2, tau4, t2px2 = x**2, tau**2, tau**4, tau**2+x**2
        Sigma = (4 * self.rho_s * self.r_s) * (tau4 / (4*(tau2+1)**3)) * (
            (2*(tau2+1) / (x2-1)) * (1-F(x))
            + 8 * F(x) + (tau4-1) / (tau2*t2px2) - (np.pi * (4*t2px2+tau2+1)) / (t2px2**1.5)
            + ((tau2*(tau4-1) + t2px2*(3*tau4-6*tau2-1)) / (tau**3*t2px2**1.5)) * L(x, tau)
        )

        return Sigma


class HaloMap3D(object):
    """
    A class for a collection of OneHalo objects.

    Takes a halo catalog and a redshift catalog to initialize halos on each redshift plane.
    
    Attributes
    ----------
    name : str
        name of halo catalog

    nslice : int
        number of redshift slices

    z_list : arr of np.float64
        array of z values for each redshift slice; len(z_list) = nslice

    d_list : arr of np.float64
        array of physical distances (scalars; implied unit = Mpc) for each redshift slice

    pixel_scale : np.float64
        how many radians does one side of pixel span (scalar; implied unit = rad)
    
    pix2mpc_list : arr of np.float64
        array of physical distances (scalars; implied unit = Mpc) a pixel side spans at each z

    area : str in ['area1', 'area2', 'area3', 'area4']
        the area that the halo catalog belongs to
    
    area_cen : arr of astropy.units.quantity.Quantity, shape = (2,)
        center coordinates of the corresponding area, unit = (kpc, kpc)
    
    halos : list of lists of OneHalo objects, len = 37
        list of halo objects for each z
        if a z slice has no halos, the list for that z will be []

    data_type : str
        type of data mapped by map_all ('rho' or 'Sigma')

    data : ndarray, shape = (37, 1024, 1024)
        3D data cube for 37 z slices, 1024 * 1024 pixels per slice
        if data_type == 'rho', data unit = M☉/kpc^3
        if data_type == 'Sigma', data unit = M☉/kpc^2

    Methods
    -------
    map_slice(z_idx: int, map_type: str = 'Sigma') -> ndarray
        Map all halos on the z_idx z slice. Note that z_idx ∈ [0, 36]
        if map_type == 'rho', maps BMO ρ(r) by calling OneHalo.call_BMO_density(r)
        if map_type == 'Sigma', maps BMO Σ(r) by calling OneHalo.call_BMO_surface_mass_density(r)
        Returns an ndarray of shape (1024, 1024), summing all halos at redshift z

    map_all(self, map_type: str = 'Sigma') -> None
        Map 3D data on all 37 z_idx z slices by calling map_slice 37 times
        if map_type == 'rho', maps BMO ρ(r)
        if map_type == 'Sigma', maps BMO Σ(r)
        Adds 'data' and 'data_type' properties to object
    """

    def __init__(self, halo_cat: pd.DataFrame, z_list: np.ndarray) -> None:
        self.name = str(halo_cat.name)
        self.nslice = 37
        self.z_list = z_list
        assert self.nslice == len(z_list)
        d_list = z_list.to(u.Mpc, cu.with_redshift(WMAP9)).value
        self.d_list = np.array(d_list)
        self.pixel_scale = (3.5*u.deg / 1024).to(u.rad).value
        self.pix2mpc_list = self.d_list * self.pixel_scale

        # determine which area does this halo_cat belong to
        self.area = area = str(self.name[-5:])
        assert area in ['area1', 'area2', 'area3', 'area4']
        area_cen = np.select(
                condlist=[area=='area1', area=='area2', area=='area3', area=='area4'], 
                choicelist=[np.array([80000,  80000]),
                            np.array([80000,  240000]),
                            np.array([240000, 80000]),
                            np.array([240000, 240000])]
                            )
        self.area_cen = area_cen / 0.7 * u.kpc
        
        # initiate all halo objects in each redshift slice
        self.halos = self.init_halos_from_cat(halo_cat)

    def init_halos_from_cat(self, halo_cat: pd.DataFrame) -> list:
        """Input halo catalog, initiate halo objects in each redshift slice.
        Outputs a list of lists of halo objects, len = 37"""

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
                    h = OneHalo(M_vir=mass, z=redshift)
                    h.center_coord(x1=x1, x2=x2, area_cen=self.area_cen)
                    halos_subgroup.append(h)
                halos.append(halos_subgroup)
        
        return halos

    def map_slice(self, z_idx: int, map_type: str = 'Sigma') -> np.ndarray:
        """Map the data on the z_idx z slice. Note that z_idx ∈ [0, 36]"""

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
                xy += halo.call_BMO_density(r=comoving_dis)
            elif map_type == 'Sigma':
                xy += halo.call_BMO_surface_mass_density(r=comoving_dis)

        return xy
    
    def map_all(self, map_type: str = 'Sigma') -> None:
        """Map 3D data on all 37 z_idx z slices."""

        self.data_type = map_type
        data = None
        for z_idx in range(self.nslice):
            xy = self.map_slice(z_idx=z_idx, map_type=map_type)
            xy = np.expand_dims(xy, axis=0)
            data = np.concatenate([data, xy], axis=0) if data is not None else xy
        
        self.data = data