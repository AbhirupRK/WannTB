## Scripts to calculate Berry curvature-induced properties directly from the Wannier90 output

### Berry curvature on a grid
The main code "berry-DataGrids_TBM.py" provides the recipe for calculating Berry curvature on a three-dimensional grid right from the *_hr.dat file of the Wannier90 output. If utilized properly, this script can be used to calculate any properties from the Hamiltonian of the system.

### Post-processing
There are several other scipts that focues on calculating properties such as anomalous Hall effect, anomalous thermal Hall effect, Peltier effect, etc and some scripts to plot the Berry curvature by utilizing the above script.
