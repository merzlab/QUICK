User Manual
=================

Keywords for QUICK Input
^^^^^^^^^^^^^^^^^^^^^^^^

1. Hamiltonians

HF Hartree-Fock Hamiltonian to be used

DFT Density Functional Theory to be used.

NOTE: One Hamiltonian must be selected. There is no default.

2. Density Functional Theory

BLYP    Built in BLYP functional

B3LYP   Built in B3LYP functional

LIBXC=FUNCTIONAL1,FUNCTIONAL2 Use density functionals from LIBXC library. Where FUNCTIONAL1 
FUNCTIONAL2 are exchange and correlation functionals. 
See LIBXC webpage for functional names: <https://www.tddft.org/programs/libxc/functionals/>

3. SCF Convergence

SCF=Integer user defined maximum self-consistent field cycles = Integer. Default: 200

DENSERMS=FLOAT user defined density matrix maximum RMS for convergence. Default : 1.0D-8.

4. Atomic Charges

CHARGE=INT A net charge is to be placed on system.

MULLIKEN Write Mulliken charges to charge output file of MC-run.

5. Geometry Optimization

OPTIMIZE=Integer do a maximum of Integer cycles of optimization. Default: 3 x Number of atoms.

GRADIENT Calculates analytical gradients.


