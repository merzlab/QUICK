module purge 
module load GCC/9.3.0 CUDA/11.0.207 intel/2020a
./configure --debug --serial --cuda --arch volta --prefix ./make_install intel
tree -P  '*cmake*'
-----
10/30/2020
module purge
module load GCC/9.3.0 CUDA/11.0.207 OpenMPI/4.0.3-CUDA
./configure  --serial --cuda --arch volta --prefix ./make_install gnu
-----
12/12/2020
Q3 max mem has been increased from 1.5GB to 15GB
-----
01/06/2021
copied from 0923_mysv, this is to investigate Q3 kernel
----
03/20/2021
synced with Merz/Master

module purge
module load GCC/8.3.0 CUDA/10.1.243
./configure  --serial --cuda --arch volta --prefix ./make_install gnu



