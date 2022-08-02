#%% Loading requirements

import time ; t0 = time.perf_counter()
import numpy as np
import scipy.constants as const
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool

fermi_level = 9.5659
kB = 8.617333262e-5    # eV/K
T = 10    # Kelvin
fermi_min, fermi_max = fermi_level-0.8 , fermi_level+0.8
fermi_div = 1001

ahc_const = 10**8 * const.e**2/const.hbar
wdir = "./"
datagrids_file = 'TBM_datagrids_nk-41.npy'
num_cpus = 144

##%% Defining functions

def fd_dist(en, mu):
    en = en.astype('float128')
    return 1/( 1 + np.exp((en-mu)/(kB*T)) )

def calc_ahc_xy(mu):
    prod_sum = 0
    for n in range(nbands):
        # weight_fn = np.heaviside(mu - eigval_grid[n], 1)
        weight_fn = fd_dist(eigval_grid[n], mu)
        prod_sum += (weight_fn * berry_grid[n,2])
    res = simps(simps(simps(prod_sum, KZ), KY), KX)
    return res * ahc_const /(2*np.pi)**3

##%% AHC Data from Berry grid

datagrids = np.load(wdir+datagrids_file, allow_pickle=True).item()
nbands = datagrids['nbands']
nk1, nk2, nk3 = datagrids['grid_shape']
kgrid = datagrids['kgrid']
eigval_grid = datagrids['eigval_grid']
berry_grid = datagrids['berry_grid']
KX, KY, KZ = kgrid[:,0,0,0], kgrid[0,:,0,1], kgrid[0,0,:,2]

mu_list = np.linspace(fermi_min, fermi_max, fermi_div)

pool = Pool(num_cpus)
ahc_xy = list(pool.map(calc_ahc_xy, mu_list))
pool.close() ; pool.join()

ahc_xy = np.vstack((mu_list, ahc_xy)).T
np.savetxt(wdir+f'TBM_ahc-xy_nk-{nk1}.dat', ahc_xy)

# del datagrids, berry_grid, eigval_grid, mu_list
print(f"Finished obtaining AHC from Berry (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

# %% Plot AHC from Berry

# ahc_xy = np.loadtxt(wdir+'TBM_ahc-xy_nk-81.dat')
ahc_filt = ahc_xy[:,1]
# ahc_filt = gaussian_filter1d(ahc_xy[:,1].astype('float64'), 5)

from matplotlib import pyplot as plt
# %matplotlib auto
plt.figure()
plt.title("AHC from Berry (TBM)", fontsize=15)
plt.axis(xmin=np.min(ahc_xy[:,0]-fermi_level),xmax=np.max(ahc_xy[:,0])-fermi_level, ymin=-1000, ymax=1500)
# plt.yticks([-10,-5,0,5,10], [-10,-5,0,5,10])
plt.locator_params(axis='x', nbins=9)
plt.locator_params(axis='y', nbins=5)
plt.axvline(0, color='grey', lw=1)
plt.axhline(0, color='grey', lw=1)
plt.plot(ahc_xy[:,0]-fermi_level, -ahc_filt, lw=1, color='blue')
ax = plt.gca()
plt.xlabel("Fermi energy (eV)", fontsize=13)
plt.ylabel(r"$\sigma_{xy}$ (S/cm)", fontsize=13)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1, labelsize=12, bottom=1, top=1, left=1, right=1 )
plt.tight_layout()
# plt.savefig(wdir+"WB_ahc-xy.pdf")
plt.show()

#%% The end