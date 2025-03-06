<p align="right">
 <img src="https://github.com/merzlab/QUICK/workflows/Build%20and%20Test%20QUICK%20Serial%20Version/badge.svg?branch=master">
 <img src="https://github.com/merzlab/QUICK/workflows/Build%20and%20Test%20QUICK%20MPI%20Version/badge.svg?branch=master">
 <img src='https://readthedocs.org/projects/quick-docs/badge/?version=latest' alt='Documentation Status' />
</p>
<p align="left">
<img width="299" height="169" src="./tools/logo.png">
</p>

An open source, GPU enabled, *ab initio* and density functional
theory program developed by Götz lab at University of California San Diego and Merz
lab at Michigan State University.

Features
--------
* Hartree-Fock
* Density functional theory (LDA, GGA and Hybrid-GGA functionals)
* Grimme type dispersion corrections
* Restricted closed-shell and unrestricted open-shell wavefunctions
* Gradient and geometry optimization calculations (in-house and DL-FIND optimizers available) 
* Includes a wide range of popular Gaussian basis sets
* Mulliken charge analysis
* Exports Molden format for visualization of geometry and orbital data
* Supports QM/MM calculations with Amber22 and later
* Fortran API to use QUICK as QM energy and force engine
* MPI parallelization for CPU platforms
* Massively parallel GPU implementation via CUDA/HIP for Nvidia/AMD GPUs
* Multi-GPU support via MPI + CUDA/HIP, also across multiple compute nodes

Limitations
-----------
* Supports energy/gradient calculations with basis functions up to f
* GPU f function code is not highly optimized, requires large amount of RAM (may fail on consumer GPUs)
* No open shell gradients with f functions on GPUs
* Supports only Cartesian basis functions (no spherical harmonics)
* Effective core potentials (ECPs) are not supported
* DFT calculations are performed exclusively using the SG1 grid system 
* No meta-GGA functionals, no range-separated hybrid functionals

Installation
------------
Supported platforms: Linux (x86 and ARM), macOS (x86 and ARM)

* [Installation Guide](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#installation-guide)
   1. [Compatible Compilers and Hardware](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#compatible-compilers-and-hardware)
   2. [Installation](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#installation)
   3. [Testing](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#environment-variables-and-testing)
   4. [Uninstallation](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#uninstallation-and-cleaning)

Getting Started
---------------
* [Hands-on Tutorials](https://quick-docs.readthedocs.io/en/latest/hands-on-tutorials.html)
* [User Manual](https://quick-docs.readthedocs.io/en/latest/user-manual.html)

Known Issues
------------
A list of installation and runtime issues can be found [here](https://quick-docs.readthedocs.io/en/latest/known-issues.html#known-issues-of-current-version).

Citation
--------
Please cite QUICK-24.03 as follows.

Manathunga, M.; O'Hearn, K. A.; Shajan, A.; Smith, J.; Miao, Y.; He, X.; Ayers, K;
Brothers, E.; Götz, A. W.; Merz, K. M. QUICK-24.03.
University of California, San Diego, CA and
Michigan State University, East Lansing, MI, 2024.

If you perform density functional theory calculations please also cite:

Manathunga, M.; Miao, Y.; Mu, D.; Götz, A. W.; Merz, K. M.
Parallel Implementation of Density Functional Theory Methods in the Quantum Interaction Computational Kernel Program. 
[*J. Chem. Theory Comput.* 16, 4315-4326 (2020)](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00290).

and in addition for any XC functional except BLYP and B3LYP:

Lehtola, S.; Steigemann, C.; Oliveira, M. J. T.; Marques, M. A. L.
Recent developments in Libxc - A comprehensive library of functionals for density functional theory.
[*Software X* 7, 1 (2018)](http://dx.doi.org/10.1016/j.softx.2017.11.002)

If you use the GPU version please also cite:

Manathunga, M.; Aktulga, H. M.; Götz, A. W.; Merz, K. M.
Quantum Mechanics/Molecular Mechanics Simulations on NVIDIA and AMD Graphics Processing Units.
[*J. Chem. Inf. Model.* 63, 711-717 (2023)](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01505).

Miao, Y.; Merz, K. M.
Acceleration of High Angular Momentum Electron Repulsion Integrals and Integral Derivatives on Graphics Processing Units. 
[*J. Chem. Theory Comput.* 11, 1449–1462 (2015)](https://pubs.acs.org/doi/10.1021/ct500984t).

and for multi-GPU calculations please also cite:

Manathunga, M.; Jin, C; Cruzeiro, V. W. D.; Miao, Y.; Mu, D.; Arumugam, K.; Keipert, K.; Aktulga, H. M.; Merz, K. M.; Götz, A. W. 
Harnessing the Power of Multi-GPU Acceleration into the Quantum Interaction Computational Kernel Program.
[*J. Chem. Theory Comput.* 17, 3955–3966 (2021)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c00145).

If you use QUICK in QM/MM simulations please cite:

Manathunga, M.; Aktulga, H. M.; Götz, A. W.; Merz, K. M.
Quantum Mechanics/Molecular Mechanics Simulations on NVIDIA and AMD Graphics Processing Units.
[*J. Chem. Inf. Model.* 63, 711-717 (2023)](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01505).

Cruzeiro, V. W. D.; Manathunga, M.; Merz, K. M.; Götz, A. W.
Open-Source Multi-GPU-Accelerated QM/MM Simulations with AMBER and QUICK.
[*J. Chem. Inf. Model.* 61, 2109–2115 (2021)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00169).

If you perform geometry optimizations please cite:

Shajan, A.; Manathunga, M.; Götz, A.W.; Merz, K.M.
Geometry optimization: A comparison of different open-source geometry optimizers.
[*J. Chem. Theory Comput.* 19, 7533-7541 (2023)](https://doi.org/10.1021/acs.jctc.3c00188).

If you use the DL-FIND for the geometry optimizations please also cite:

Kästner, J.; Carr, J. M.; Keal, T. W.; Thiel, W.; Wander, A.; Sherwood, P.
DL-FIND: An Open-Source Geometry Optimizer for Atomistic Simulations. 
[*J. Phys. Chem. A* 113, 11856-11865 (2009)](https://pubs.acs.org/doi/10.1021/jp9028968).

License
-------
QUICK is licensed under Mozilla Public License 2.0. More information can be found [here](https://quick-docs.readthedocs.io/en/latest/license.html#mozilla-public-license-version-2-0).

Special Note to Users
---------------------
Although QUICK is tested on a range of hardware with different compiler versions, we cannot guarantee that it will work flawlessly in all your applications. But we are working hard to detect and fix any issues. If you experience any compile or runtime issues, please report to us through the issues section of this repository. We appreciate any feedback and contributions to the code.
