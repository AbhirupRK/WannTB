#%% Loading requirements

import numpy as np

wdir = "./"
datagrids_file = 'TBM_datagrids_nk-81.2.npy'

#%%

datagrids = np.load(wdir+datagrids_file, allow_pickle=True).item()
nbands = datagrids['nbands']
nk = datagrids['grid_shape']
kgrid = datagrids['kgrid']
eigval_grid = datagrids['eigval_grid']
berry_grid = datagrids['berry_grid']
KX, KY, KZ = kgrid[:,0,0,0], kgrid[0,:,0,1], kgrid[0,0,:,2]

#%%

berry_z_xz = 0

for i in range(39):
    berry_z_xz += berry_grid[i,2,:,40,:]


a=np.argwhere(berry_z_xz<-20000)
#%%

# berry_vb_z_xz = berry_grid[39,2,:,25,:]



import matplotlib.pyplot as plt
%matplotlib qt

plt.figure()
# ax = plt.axes(projection ='3d')
# plt.imshow(berry_z_xz, cmap="bwr", interpolation='nearest', extent =[-0.5, 0.5, -0.5, 0.5],vmin=-20000,vmax=20000)
plt.pcolor(KX,KZ,berry_z_xz, cmap='bwr',vmin=-200000,vmax=200000)
# ax.plot_surface(KX,KZ.T,berry_grid[38,2,:,0,:],cmap="Reds")
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(False)
# ax.plot_surface(KX,KZ,berry_grid[39,2,:,0,:])
plt.colorbar()#plot, ax=axs, ticks=[-1, 0, 1], orientation='vertical', shrink=0.62, fraction=0.1)
plt.show()

#%%

X, Z = np.meshgrid(KX, KZ)

import matplotlib.pyplot as plt
%matplotlib qt

plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X,Z,-berry_z_xz,cmap="bwr",vmin=-5000,vmax=5000)

# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(False)
# ax.set_zlim3d(-250000, 5000)
ax.locator_params(axis='z', nbins=5)
ax.set_xlabel(r"$k_x$", fontsize=18)
ax.set_ylabel(r"$k_z$", fontsize=18)
ax.set_zlabel(r"$\Omega_z$ ($\AA^2$)", fontsize=18)
ax.tick_params(axis="x",direction="in", pad=0)
ax.tick_params(axis="y",direction="in", pad=0)
ax.tick_params(axis="z",direction="in", pad=-30)
ax.set_zticks([-1000,0,100000,200000])
ax.set_zticklabels([r"",r"",r"$-1 \times 10^5$",r"$-2 \times 10^5$"])
ax.tick_params(labelsize=14)
# ax.colorbar()
plt.tight_layout()
plt.show()

#%%

X, Z = np.meshgrid(KX, KZ)

import matplotlib.pyplot as plt
%matplotlib qt

plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(X,Z,eigval_grid[38,:,25,:])
ax.plot_surface(X,Z,eigval_grid[39,:,25,:])
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.grid(False)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

strains = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
fermi _levels = np.array([9.5762, %%%, 9.7111, 9.7796, 9.8488, 9.9188, 9.9897, 10.0614, 10.134])
wp_energies = np.array([62.9, %%%, 47.6, 40.9, 35.2, 28.7, 22.6, 16.7, 11.2])
stress_z = np.array([0.16, %%%, 20.8, 32.1, 43.9, 56.4, 69.5, 83.3, 97.8])
pressure = np.array([0.3, %%%, 11.3, 17.2, 23.5, 30.1, 37, 44.3, 51.9])
ahc = np.array([1290.202705, %%%, 1358.222199, 1393.498521, 1412.647781, 1456.564122, 1502.713673, 1554.076256, 1610.016283])


plt.figure()


plt.show()


#%%