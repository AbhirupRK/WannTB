#%% Loading requirements

import time ; t0 = time.perf_counter()
import os
import sys
import itertools
import numpy as np
from numpy import linalg as LA
from numpy.linalg import multi_dot as mdot
import tbmodels as tbm
import ray

nkgrid = 11
nk1, nk2, nk3 = nkgrid, nkgrid, nkgrid
dk = 1.e-5
num_cpus = 144

wdir = "./"
seedname = "wann"
outfile = f'TBM_datagrids_nk-{nkgrid}.npy'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

#%% Loading TB model

if os.path.isfile(wdir+'TBM_model.hdf5'):
    tbmodel = tbm.Model.from_hdf5_file(wdir+'TBM_model.hdf5')
else:
    tbmodel = tbm.Model.from_wannier_files(hr_file=wdir+seedname+'_hr.dat')
    tbmodel.to_hdf5_file(wdir+'TBM_model.hdf5')

nbands = np.size(tbmodel.eigenval(k=[0,0,0]))
KX, KY, KZ = np.linspace(0,1,nk1), np.linspace(0,1,nk2), np.linspace(0,1,nk3)
kgrid = np.array(list(itertools.product(KX, KY, KZ)))

print(f"Finished loading TB model (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% Defining functions

@ray.remote
def eigenvals(tbmodel, kpt):
    return tbmodel.eigenval(k=kpt)

@ray.remote
def calc_velocity(tbmodel, kpt):
    eigval = tbmodel.eigenval(k=kpt)
    dEx = (tbmodel.eigenval(k=[kpt[0]+dk, kpt[1], kpt[2]])-eigval)/dk
    dEy = (tbmodel.eigenval(k=[kpt[0], kpt[1]+dk, kpt[2]])-eigval)/dk
    dEz = (tbmodel.eigenval(k=[kpt[0], kpt[1], kpt[2]+dk])-eigval)/dk
    return np.array([dEx, dEy, dEz]).T

@ray.remote
def calc_berry(tbmodel, kpt, n):

    ham = tbmodel.hamilton(k=kpt)
    w, u = LA.eig(ham)
    inds = w.argsort()
    w, u = w[inds].real, u[:,inds]
    dHx = (tbmodel.hamilton(k=[kpt[0]+dk, kpt[1], kpt[2]])-ham)/dk
    dHy = (tbmodel.hamilton(k=[kpt[0], kpt[1]+dk, kpt[2]])-ham)/dk
    dHz = (tbmodel.hamilton(k=[kpt[0], kpt[1], kpt[2]+dk])-ham)/dk
    bc = 0
    for m in range(nbands):
        if m!=n:
            bc += 2j * np.array([mdot([u[:,n].conj(), dHy, u[:,m]])*mdot([u[:,m].conj(), dHz, u[:,n]]), 
                        mdot([u[:,n].conj(), dHz, u[:,m]])*mdot([u[:,m].conj(), dHx, u[:,n]]), 
                        mdot([u[:,n].conj(), dHx, u[:,m]])*mdot([u[:,m].conj(), dHy, u[:,n]])]) / (w[n] - w[m])**2
    return bc.real

#%% Grid calculations

grid_shape = (nk1, nk2, nk3)

ray.init(num_cpus=num_cpus)
tbmodel_id=ray.put(tbmodel)

arglist = list(itertools.product([tbmodel_id], kgrid))
ids = list(itertools.starmap(eigenvals.remote, arglist))
eigval_grid = np.array(ray.get(ids))
print(f"Finished calculating eigenvalue grid (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

arglist = list(itertools.product([tbmodel_id], kgrid))
ids = list(itertools.starmap(calc_velocity.remote, arglist))
velocity_grid = np.array(ray.get(ids))
print(f"Finished calculating velocity grid (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

arglist = list(itertools.product([tbmodel_id], kgrid, range(nbands)))
ids = list(itertools.starmap(calc_berry.remote, arglist))
berry_grid = np.array(ray.get(ids)).reshape((-1, nbands, 3))
print(f"Finished calculating Berry grid (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

# del KX, KY, KZ, calc_berry, calc_velocity, eigenvals, tbmodel_id, arglist, ids
ray.shutdown()

#%% Shaping and writing

kgrid = kgrid.reshape(grid_shape + (3,))
eigval_grid = np.moveaxis(eigval_grid, 0, -1).reshape((nbands,)+grid_shape)
velocity_grid = np.moveaxis(velocity_grid, 0, -1).reshape((nbands, 3,)+grid_shape)
berry_grid = np.moveaxis(berry_grid, 0, -1).reshape((nbands, 3,)+grid_shape)

datagrids = { "nbands" : nbands, "grid_shape" : grid_shape, "kgrid" : kgrid,  "eigval_grid" : eigval_grid, "velocity_grid" : velocity_grid, "berry_grid" : berry_grid}
np.save(wdir+outfile, datagrids)
print(f"Finished writing data-grid (Total time elapsed : {round(time.perf_counter() - t0, 1)} seconds)")

#%% The end