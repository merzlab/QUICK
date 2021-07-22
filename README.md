<p align="right">
<img src="https://github.com/Madu86/QUICK/workflows/Serial%20Build/badge.svg">
<img src="https://github.com/Madu86/QUICK/workflows/MPI%20Build/badge.svg">
<img src='https://readthedocs.org/projects/quick-docs/badge/?version=21.3.0' alt='Documentation Status' />
</p>
<p align="left">
<img width="299" height="107" src="./tools/logo.png">
</p>

An open source, GPU enabled, *ab initio* and density functional
theory program developed by Götz lab at University of California San Diego and Merz
lab at Michigan State University.

Features
--------
* Hartree-Fock energy calculations 
* Density functional theory calculations (LDA, GGA and Hybrid-GGA functionals available).
* Gradient and geometry optimization calculations 
* Mulliken charge analysis
* Supports QM/MM calculations with Amber21
* MPI parallelization for CPU platforms
* Massively parallel GPU implementation via CUDA for Nvidia GPUs
* Multi-GPU support via MPI + CUDA, also across multiple compute nodes

Limitations
-----------
* Supports only closed shell systems
* Supports energy/gradient calculations with basis functions up to d  
* Supports only cartesian basis functions (no spherical harmonics)
* DFT calculations are performed exclusively using SG1 grid system 

Installation
------------
Supported platforms: Linux

* [Installation Guide](https://quick-docs.readthedocs.io/en/21.3.0/installation-guide.html#installation-guide)
   1. [Compatible Compilers and Hardware](https://quick-docs.readthedocs.io/en/21.3.0/installation-guide.html#compatible-compilers-and-hardware)
   2. [Installation](https://quick-docs.readthedocs.io/en/21.3.0/installation-guide.html#installation)
   3. [Testing](https://quick-docs.readthedocs.io/en/21.3.0/installation-guide.html#environment-variables-and-testing)
   4. [Uninstallation](https://quick-docs.readthedocs.io/en/21.3.0/installation-guide.html#uninstallation-and-cleaning)

Getting Started
---------------
* [Hands-on Tutorials](https://quick-docs.readthedocs.io/en/21.3.0/hands-on-tutorials.html)
* [User Manual](https://quick-docs.readthedocs.io/en/21.3.0/user-manual.html)

Known Issues
------------
A list of installation and runtime issues can be found [here](https://quick-docs.readthedocs.io/en/21.3.0/known-issues.html#known-issues-of-current-version).

Citation
--------
Please cite QUICK-21.03 as follows.

Manathunga, M.; Jin, C.; Cruzeiro, V.W.D.; Smith, J.; Keipert, K.; Pekurovsky, D.; Mu, D.; Miao, Y.;He, X.; Ayers,K;
Brothers, E.; Götz, A.W.; Merz,K. M. QUICK-21.03 University of California San Diego, CA and Michigan State University, East Lansing, MI, 2021.

If you perform density functional theory calculations please also cite:

Manathunga, M.; Miao, Y.; Mu, D.; Götz, A. W.; Merz, K. M.
Parallel Implementation of Density Functional Theory Methods in the Quantum Interaction Computational Kernel Program. 
[*J. Chem. Theory Comput.* 16, 4315-4326 (2020)](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00290).

and in addition for any XC functional except B3LYP:

Lehtola, S.; Steigemann, C.; Oliveira, M. J. T.; Marques, M. A. L.
Recent developments in Libxc - A comprehensive library of functionals for density functional theory.
[*Software X* 7, 1 (2018)](http://dx.doi.org/10.1016/j.softx.2017.11.002)

If you use the GPU version please also cite:

Miao, Y.; Merz, K. M.
Acceleration of High Angular Momentum Electron Repulsion Integrals and Integral Derivatives on Graphics Processing Units. 
[*J. Chem. Theory Comput.* 11, 1449–1462 (2015)](https://pubs.acs.org/doi/10.1021/ct500984t).

License
-------
QUICK is licensed under Mozilla Public License 2.0. More information can be found [here](https://quick-docs.readthedocs.io/en/21.3.0/license.html#mozilla-public-license-version-2-0).

Special Note to Users
---------------------
QUICK is still in the experimental stage and we do not guarantee
it will work flawlessly in all your applications. But we are working hard to
detect and fix issues. If you experience any compile or runtime issues, please
report to us through issues section of this repository.
