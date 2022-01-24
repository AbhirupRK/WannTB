#%%
import numpy as np
import scipy.constants as const
from matplotlib import pyplot as plt
# %matplotlib auto


wdir = "./"
fermi_level = 12.5383    # eV
kB = 8.617333262e-5    # eV/K
T = 10    # Kelvin

ahc_xy = np.loadtxt(wdir+'TBM_ahc-xy_nk-81.dat')
athc_xy = np.loadtxt(wdir+'TBM_athc-xy_nk-81.dat')
lorenz = athc_xy[:,1]/(ahc_xy[:,1]*T)
                                                                                      

# ahc_const = 10**8 * const.e**2/const.hbar
# athc_const = 10**8 * const.k**2 * T / const.hbar
# athc_const/(ahc_const*T)*np.pi**2/3

#%%

plt.figure()
plt.title("Lorenz number (TBM)", fontsize=15)
# plt.axis(xmin=-0.2, xmax=0.2, ymin=0, ymax=5.e-8)
plt.plot(ahc_xy[:,0]-fermi_level, lorenz, color='blue')
plt.xlabel("Fermi energy (eV)", fontsize=13)
plt.ylabel(r"L (W $\Omega$ K$^{-2}$)", fontsize=13)
plt.show()

#%%