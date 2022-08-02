#%%
import numpy as np
import tbmodels as tbm
from matplotlib import pyplot as plt
# %matplotlib auto

###############################################################################
# Changes to be made only in this block
###############################################################################

wdir = "./"
seedname = "wannier90"

fermi_level = 9.2606
ymin, ymax = -1, 1

###############################################################################

model = tbm.Model.from_wannier_files(hr_file=wdir+seedname+"_hr.dat")
kpts = np.genfromtxt(wdir+seedname+"_band.kpt", skip_header=1, usecols = (0,1,2))
kpath=np.genfromtxt(wdir+seedname + "_band.labelinfo.dat", dtype='str', usecols = (0, 2))

nbands = np.size(model.eigenval(k=[0,0,0]))      # even number
xgrid = np.linspace(0, float(kpath[-1,1]), np.size(kpts,axis=0))

bnds = np.zeros((1,nbands+1))
for i in range(0,np.size(kpts,axis=0)):
    E = np.around(model.eigenval(k=kpts[i]), decimals=5) - fermi_level
    bnds = np.append(bnds, np.reshape(np.concatenate((np.array([xgrid[i]]),E)), (1,np.size(E)+1)), axis=0)
bnds=np.delete(bnds, 0, axis=0)

#%%

plt.figure()
plt.title("Band structure (hr.dat)", fontsize=15)
plt.xticks(kpath[:,1].astype(float), ['$' + s + '$' for s in kpath[:,0]])
#plt.yticks([-10,-5,0,5,10], [-10,-5,0,5,10])
plt.locator_params(axis='y', nbins=5)
plt.vlines(plt.xticks()[0], -50, 50, colors='grey')
plt.axhline(0, color='grey')

for i in range(1,np.size(bnds,1)):
    plt.plot(bnds[:,0],bnds[:,i], lw=2)

plt.tight_layout(pad=0, rect=(0,0.04,0.99,1))
ax = plt.gca()
plt.xlabel("k-path", fontsize=15)
plt.ylabel("Energy (eV)", fontsize=15)
plt.setp(ax.spines.values(), linewidth=1.5)
plt.axis(xmin=np.min(bnds[:,0]),xmax=np.max(bnds[:,0]), ymin=ymin, ymax=ymax)
plt.tick_params(axis = 'both', direction = 'in', length = 5, width = 1.5, labelsize=15, bottom=1, top=1, left=1, right=1 )
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
plt.tight_layout()
plt.show()

#%%