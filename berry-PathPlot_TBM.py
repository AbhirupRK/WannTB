#%% Loading requirements
import time
t0 = time.perf_counter()

import os
import sys
import itertools
import numpy as np
from numpy import linalg as LA
from numpy.linalg import multi_dot as mdot
import tbmodels as tbm
from multiprocessing import Pool

wdir = "./"
seedname = "wann"
fermi_level = 12.5383

dk = 0.00001
num_cpus = 144

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

#%% Loading TB model

if os.path.isfile(wdir+'TBM_model.hdf5'):
    tbmodel = tbm.Model.from_hdf5_file('TBM_model.hdf5')
else:
    tbmodel = tbm.Model.from_wannier_files(hr_file=wdir+seedname+"_hr.dat")
    tbmodel.to_hdf5_file(wdir+'TBM_model.hdf5')

nbands = np.size(tbmodel.eigenval(k=[0,0,0]))
KX, KY, KZ = np.meshgrid(np.arange(0,1,1/nk1), np.arange(0,1,1/nk2), np.arange(0,1,1/nk3))
kgrid = np.array([KX.flatten(), KY.flatten(), KZ.flatten()]).T

del KX, KY, KZ
print(f"Finished loading TB model (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Defining functions

def calc_berryz_sum(tbmodel, kpt, bands):

    ham = tbmodel.hamilton(k=kpt)
    w, u = LA.eig(ham)
    inds = w.argsort()
    w, u = w[inds].real, u[:,inds]
    dHx = (tbmodel.hamilton(k=[kpt[0]+dk, kpt[1], kpt[2]])-ham)/dk
    dHy = (tbmodel.hamilton(k=[kpt[0], kpt[1]+dk, kpt[2]])-ham)/dk
    bc = 0
    for n in bands:
        for m in range(nbands):
            if m!=n:
                bc += 2j * ( mdot([u[:,n].conj(), dHx, u[:,m]])*mdot([u[:,m].conj(), dHy, u[:,n]]) ) / (w[n] - w[m])**2
    return bc.real

#%% Berry curvature along high symmetry path

bands = range(0, 39)
kpath = np.genfromtxt(wdir+seedname+"_band.kpt", skip_header=1, usecols = (0,1,2))
kticks=np.genfromtxt(wdir+seedname+"_band.labelinfo.dat", dtype='str', usecols = (0, 2))
xgrid = np.linspace(0, float(max(kticks[:,1])), np.size(kpath, axis=0))

pool = Pool(num_cpus)
arglist = list(itertools.product([tbmodel], kpath, [bands]))
berryz_kpath = list(pool.starmap(calc_berryz_sum, arglist))

berryz_kpath_dat = np.vstack((xgrid, np.asarray(berryz_kpath))).T
np.savetxt(wdir+'TBM_berry-z_kpath.dat', berryz_kpath_dat)

del bands, kpath, xgrid, arglist, berryz_kpath
print(f"Finished computing berry curvature along k-path (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Plot Berry along K-path

# berryz_kpath_dat = np.loadtxt(wdir+'TBM_berry-z_kpath.dat')
from matplotlib import pyplot as plt
# %matplotlib auto
plt.figure()
plt.title("Berry curvature (TBM)", fontsize=15)
plt.axis(xmin=np.min(berryz_kpath_dat[:,0]),xmax=np.max(berryz_kpath_dat[:,0]))# , ymin=np.min(berryz_kpath_dat[:,1]), ymax=np.max(berryz_kpath_dat[:,1]))
plt.xticks(kticks[:,1].astype(float), ['$' + s + '$' for s in kticks[:,0]])
plt.locator_params(axis='y', nbins=5)
[plt.axvline(x, color='grey', lw=1) for x in kticks[:,1].astype(float)]
plt.axhline(0, color='grey', lw=1)
plt.plot(berryz_kpath_dat[:,0], -berryz_kpath_dat[:,1], lw=1, color='blue')
ax = plt.gca()
plt.xlabel("k-path", fontsize=13)
plt.ylabel(r"$\Omega_z$ ($\AA^2$)", fontsize=13)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1, labelsize=12, bottom=1, top=1, left=1, right=1 )
plt.tight_layout()
# plt.savefig(wdir+"TBM_berry-z_kpath.png")
plt.show()

#%% The end