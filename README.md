![Serial build status] (https://github.com/Madu86/QUICK/blob/master/.github/workflows/serial.yml/badge.svg)
![MPI build status] (https://github.com/Madu86/QUICK/blob/master/.github/workflows/mpi.yml/badge.svg)
![QUICK logo](./tools/logo.png)

An open source, GPU enabled, linear scaling *ab initio* and density functional
theory program developed by Goetz lab at University of California San Diego and Merz
lab at Michigan State University.

Features
--------
* Single point Hartree-Fock calculations (closed shell only) 
* Density functional theory calculations (LGA, GGA and Hybrid-GGA functionals available, closed shell only).
* Gradient and geometry optimization calculations (LBFGS solver available)
* Mulliken charge analysis

Installation
------------
Supported platforms: Linux and OSX

* [Installation Guide](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#installation-guide)
   1. [Compatible Compilers and Hardware](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#compatible-compilers-and-hardware)
   2. [Serial Installation](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#serial-installation)
   3. [MPI Installation](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#mpi-installation)
   4. [CUDA Installation](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#cuda-version-installation)

Getting Started
---------------
* [Testing QUICK Installation](https://quick-docs.readthedocs.io/en/20.6.0/installation-guide.html#environment-variables-and-testing)
* [Hands-on Tutorials](https://quick-docs.readthedocs.io/en/20.6.0/hands-on-tutorials.html)
* [User Manual](https://quick-docs.readthedocs.io/en/20.6.0/user-manual.html)

Known Issues
------------
A list of installation and runtime issues can be found [here](https://quick-docs.readthedocs.io/en/20.6.0/known-issues.html#known-issues-of-current-version).

Citation
--------
Please cite QUICK-20.06 as follows.

Manathunga, M.; Chi, J.; Cruzeiro, V.W.D.; Mu, D.; Keipert, C.; Miao,Y.; He, X.; Ayers,K; Brothers, E.;
Götz, A.W.; Merz,K. M. QUICK-20.06 University of California San Diego, CA and Michigan State University, East Lansing, MI, 2020.

License
-------
QUICK is licensed uder Mozilla Public License 2.0. More information can be found [here](https://quick-docs.readthedocs.io/en/20.6.0/license.html#mozilla-public-license-version-2-0).

Special Note to Users
---------------------
QUICK is still in the experimental stage and we do not guarantee
it will work flawlessly in all your applications. But we are working hard to
detect and fix issues. If you experience any compile or runtime issues, please
report to us through issues section of this repository.
