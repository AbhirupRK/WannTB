#%% Loading requirements

import time ; t0 = time.perf_counter()
import numpy as np
from mpmath import mp, fp
import scipy.constants as const
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool

fermi_level = 9.5659    # eV
kB = 8.617333262e-5    # eV/K
T = 300    # Kelvin
fermi_min, fermi_max = fermi_level-0.25 , fermi_level+0.25
fermi_div = 101

peltier_const = 10**8 * const.k * const.e / const.hbar
wdir = "./"
datagrids_file = 'TBM_datagrids_nk-41.npy'
num_cpus = 108

##%% Defining functions

polylog2 = np.vectorize(lambda z: float(fp.polylog(2, z)))    # for 0 < z < 1 

def fd_dist(en, mu):
    en = en.astype('float128')
    return 1/( 1 + np.exp((en-mu)/(kB*T)) )

def peltier_kernel(en, mu):
    en = en.astype('float128')
    fd = 1/( 1 + np.exp((en-mu)/(kB*T)) )
    res = ( fd * (en-mu)/(kB*T) ) + np.log( 1 + np.exp(-(en-mu)/(kB*T)) )
    return res

def calc_peltier_xy(mu):
    prod_sum = 0
    for n in range(nbands):
        weight_fn = peltier_kernel(eigval_grid[n], mu)
        prod_sum += (weight_fn * berry_grid[n,2])
    res = simps(simps(simps(prod_sum, KZ), KY), KX)
    return res * peltier_const /(2*np.pi)**3

##%% Peltier Data from Berry grid

datagrids = np.load(wdir+datagrids_file, allow_pickle=True).item()
nbands = datagrids['nbands']
nk1, nk2, nk3 = datagrids['grid_shape']
kgrid = datagrids['kgrid']
eigval_grid = datagrids['eigval_grid']
berry_grid = datagrids['berry_grid']
KX, KY, KZ = kgrid[:,0,0,0], kgrid[0,:,0,1], kgrid[0,0,:,2]

mu_list = np.linspace(fermi_min, fermi_max, fermi_div)

pool = Pool(num_cpus)
peltier_xy = list(pool.map(calc_peltier_xy, mu_list))
pool.close() ; pool.join()

peltier_xy = np.vstack((mu_list, peltier_xy)).T
np.savetxt(wdir+f'TBM_peltier-xy.dat', peltier_xy)

# del datagrids, berry_grid, eigval_grid, mu_list
print(f"Finished obtaining Peltier from Berry (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Plot Peltier from Berry

# peltier_xy = np.loadtxt(wdir+'TBM_peltier-xy.dat')
peltier_filt = peltier_xy[:,1]
# peltier_filt = gaussian_filter1d(peltier_xy[:,1].astype('float64'), 5)

from matplotlib import pyplot as plt
# %matplotlib auto
plt.figure()
plt.title("Peltier Conductivity from Berry (TBM)", fontsize=15)
plt.axis(xmin=np.min(peltier_xy[:,0]-fermi_level),xmax=np.max(peltier_xy[:,0])-fermi_level)#, ymin=-0.0004, ymax=0.0005)
# plt.yticks([-10,-5,0,5,10], [-10,-5,0,5,10])
plt.locator_params(axis='x', nbins=9)
plt.locator_params(axis='y', nbins=5)
plt.axvline(0, color='grey', lw=1)
plt.axhline(0, color='grey', lw=1)
plt.plot(peltier_xy[:,0]-fermi_level, -peltier_filt, lw=1, color='blue')
ax = plt.gca()
plt.xlabel("Fermi energy (eV)", fontsize=13)
plt.ylabel(r"$\kappa_{xy}$ (W/K-cm)", fontsize=13)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1, labelsize=12, bottom=1, top=1, left=1, right=1 )
plt.tight_layout()
# plt.savefig(wdir+"WB_peltier-xy.pdf")
plt.show()

#%% The end