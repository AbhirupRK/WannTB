#%%
import time
t0 = time.perf_counter()

import os
import sys
import ray
import numpy as np
import scipy
import wannierberri as wberri
import pickle

wdir = "/home/abhirup/qe_files/Co3Sn2S2.wann/"
seedname = "wann"
fermi_level = 12.5383
num_occ = 39
kT = 0.00086

fermi_min, fermi_max = fermi_level-0.8 , fermi_level+0.8
fermi_div = 1001
nkpath = 100
nkgrid = 23
num_cpus = 36
generators = ['Inversion','C3z']

#%% Loaing TB system

if os.path.isfile(wdir+'WB_system.pickle'):
    system=pickle.load(open(wdir+'WB_system.pickle','rb'))
else:
    system=wberri.System_tb(wdir+'wann_tb.dat',berry=True)
    pickle.dump(system ,open(wdir+'WB_system.pickle','wb'))

print(f"Finished loading TB system (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Berry curvature computation (Path mode)

knodes=[[0.25, 0.5, -0.25],[0, 0, -0.5],[0, 0.375, -0.375],[0, 0.5, 0],[0, 0, 0]]
klabels=["W","T","U","L","G"]
path=wberri.Path(system, k_nodes=knodes, labels=klabels, nk=nkpath)
kpath=path.getKline()
np.savetxt("WB_kpath.dat", kpath)

# parallel=wberri.Parallel(method="ray",num_cpus=num_cpus)
# berry_path=wberri.tabulate(system, grid=path, quantities=['berry'], parallel=parallel)
if os.path.isfile(wdir+'WB_berry-path.pickle'):
    berry_path=pickle.load(open(wdir+'WB_berry-path.pickle','rb'))
else:
    parallel=wberri.Parallel(method="ray",num_cpus=num_cpus)
    berry_path=wberri.tabulate(system, grid=path, quantities=['berry'], parallel=parallel)
    pickle.dump(berry_path,open("WB_berry-path.pickle","wb"))

ray.shutdown()
del knodes, klabels, path
print(f"Finished computing Berry (Path mode) (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Plot Berry along K-path

n_list = np.arange(0, num_occ)
berry_path_z = berry_path.get_data(quantity='berry',iband=n_list,component="z")
berry_path_z_sum = berry_path_z.sum(axis=1)
berry_path_z_dat = np.vstack((kpath, berry_path_z_sum)).T
np.savetxt("WB_berry-z_kpath.dat", berry_path_z_dat)

from matplotlib import pyplot as plt
# %matplotlib auto

kticks=np.genfromtxt(wdir+seedname+"_band.labelinfo.dat", dtype='str', usecols = (0, 2))
plt.figure()
plt.title("Berry curvature (WannierBerri)", fontsize=15)
plt.axis(xmin=np.min(berry_path_z_dat[:,0]),xmax=np.max(berry_path_z_dat[:,0]))# , ymin=np.min(berry_path_z_dat[:,1]), ymax=np.max(berry_path_z_dat[:,1]))
plt.xticks(kticks[:,1].astype(float), ['$' + s + '$' for s in kticks[:,0]])
#plt.yticks([-10,-5,0,5,10], [-10,-5,0,5,10])
plt.locator_params(axis='y', nbins=5)
[plt.axvline(x, color='grey', lw=1) for x in kticks[:,1].astype(float)]
plt.axhline(0, color='grey', lw=1)
plt.plot(berry_path_z_dat[:,0], berry_path_z_dat[:,1], lw=1, color='blue')
ax = plt.gca()
plt.xlabel("k-path", fontsize=13)
plt.ylabel(r"$\Omega_z$ ($\AA$)", fontsize=13)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1, labelsize=12, bottom=1, top=1, left=1, right=1 )
plt.tight_layout()
plt.savefig(wdir+"WB_berry-z_kpath.png")
# plt.show()

del n_list, berry_path_z, berry_path_z_sum, berry_path_z_dat, kticks, ax
print(f"Finished plotting Berry (Path mode) (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%%Berry curvature computation (Grid mode)

# grid=wberri.Grid(system, NK=nkgrid)
# system.set_symmetry(generators)
# parallel=wberri.Parallel(method="ray",num_cpus=num_cpus)
# berry_grid = wberri.tabulate(system, grid=grid, quantities=['berry'], parallel=parallel)

if os.path.isfile(wdir+'WB_berry-grid.pickle'):
    berry_grid=pickle.load(open(wdir+'WB_berry-grid.pickle','rb'))
else:
    grid=wberri.Grid(system, NK=nkgrid)
    system.set_symmetry(generators)
    parallel=wberri.Parallel(method="ray",num_cpus=num_cpus)
    berry_grid = wberri.tabulate(system, grid=grid, quantities=['berry'], parallel=parallel)
    pickle.dump(berry_grid, open("WB_berry-grid.pickle","wb"))

ray.shutdown()
print(f"Finished computing Berry (Grid mode) (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% AHC from Berry

const = 10**8 * scipy.constants.e**2/scipy.constants.hbar

def fd_dist(en, mu):
    return 1/( 1 + np.exp((en-mu)/kT) )

mu_list = np.linspace(fermi_min, fermi_max, fermi_div)

sig_list = []
for mu in mu_list:
    sig_xy = 0
    for n in range(berry_grid.nband):
        bcz = berry_grid.get_data(quantity='berry', iband=n, component="z")
        E = berry_grid.get_data(iband=n, quantity='E')
        weight_fn = np.heaviside(mu - E, 1)
        # weight_fn = fd_dist(E, mu)
        bcz_int = (weight_fn*bcz)
        sig_xy += bcz_int.sum()
    sig_list.append(sig_xy)
sig_list = const * np.asarray(sig_list)/(np.prod(berry_grid.grid)*(2*np.pi)**3)

sigxy_dat = np.vstack((mu_list, sig_list)).T
np.savetxt(wdir+'WB_sig-xy.dat', sigxy_dat)

del mu_list, sig_list
print(f"Finished computing AHC from Berry (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Plot AHC from Berry

# from scipy.signal import savgol_filter
# sig_filt = savgol_filter(sigxy_dat[:,1], 21, 1)
sig_filt = sigxy_dat[:,1]

from matplotlib import pyplot as plt
# %matplotlib auto
plt.figure()
plt.title("AHC from Berry (WannierBerri)", fontsize=15)
plt.axis(xmin=np.min(sigxy_dat[:,0]-fermi_level),xmax=np.max(sigxy_dat[:,0])-fermi_level, ymin=-1000, ymax=1500)
# plt.yticks([-10,-5,0,5,10], [-10,-5,0,5,10])
plt.locator_params(axis='x', nbins=9)
plt.locator_params(axis='y', nbins=5)
plt.axvline(0, color='grey', lw=1)
plt.axhline(0, color='grey', lw=1)
plt.plot(sigxy_dat[:,0]-fermi_level, 2*sig_filt, lw=1, color='blue')
ax = plt.gca()
plt.xlabel("Fermi energy (eV)", fontsize=13)
plt.ylabel(r"$\sigma_{xy}$ (S/cm)", fontsize=13)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1, labelsize=12, bottom=1, top=1, left=1, right=1 )
plt.tight_layout()
# plt.savefig(wdir+"WB_ahc-xy.pdf")
plt.show()

del sig_filt, ax

#%% AHC Direct

# nk1, nk2, nk3 = nkgrid, nkgrid, nkgrid
# KX, KY, KZ = np.meshgrid(np.arange(0,1,1/nk1), np.arange(0,1,1/nk2), np.arange(0,1,1/nk3))
# kgrid = np.array([KX.flatten(), KY.flatten(), KZ.flatten()]).T
# grid = wberri.Path(system, k_list=kgrid)

grid=wberri.Grid(system, NK=nkgrid)
system.set_symmetry(generators)

parallel=wberri.Parallel(method="ray",num_cpus=num_cpus)
wberri.integrate(system, grid,
            Efermi=np.linspace(fermi_min, fermi_max, fermi_div),
            smearEf=10, # 10K
            quantities=["ahc"],
            parallel = parallel,
            adpt_num_iter=10,
            fout_name="WB")

# np.savetxt("WB_ahc.dat", ahc)
ray.shutdown()
print(f"Finished computing AHC directly (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% The end