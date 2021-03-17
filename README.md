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

Limitations
-----------
* Supports only closed shell systems
* Supports energy/gradient calculations with basis functions upto d  

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

License
-------
QUICK is licensed under Mozilla Public License 2.0. More information can be found [here](https://quick-docs.readthedocs.io/en/21.3.0/license.html#mozilla-public-license-version-2-0).

Special Note to Users
---------------------
QUICK is still in the experimental stage and we do not guarantee
it will work flawlessly in all your applications. But we are working hard to
detect and fix issues. If you experience any compile or runtime issues, please
report to us through issues section of this repository.
