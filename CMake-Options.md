## CMake Build System Options

This page gives a summary of CMake options that can be used with QUICK. Note that like all CMake options, these options are sticky. Once passed to CMake, they will remain set unless you set them to a different value (with -D), unset them (with -U), or delete the build directory.

#### General options
- `-DCOMPILER=<GNU|INTEL|INTELLLVM|ONEAPI|PGI|CLANG|LLVM|CRAY|AUTO|MANUAL>`: Allows selection of the compiler toolchain to use. *-DCOMPILER=AUTO* enables default CMake behaviour. *NOTE:* the INTELLLVM|ONEAPI and PGI options should be used for the Intel oneAPI and NVIDIA HPC SDK (NVHPC) compilers, respectively. For CLANG, a mixed GNU/Clang build is performed (clang C/C++ compilers, GNU gfortran Fortran compiler). For MANUAL, the CMAKE_<LANGUAGE>_COMPILER CMake variables are used.
- `-DENABLEF=<TRUE|FALSE>`: Toggle compilation of time consuming F functions in the ERI code of the GPU versions. Default is FALSE. **NOTE**: The current version of the F function code takes very long to compile (hours) and requires a large amount of RAM. Work is planned to optimize this in future releases.
- `-DCMAKE_BUILD_TYPE=<Debug|Release>`: Controls whether to build debug or release versions. Default is Release.
- `-DOPTIMIZE=<TRUE|FALSE>`: Controls whether to enable compiler optimizations. On by default.
- `-DCMAKE_INSTALL_PREFIX=<path>`: Controls where QUICK will be installed. Default is /usr/local/bin/. 
- `-DQUICK_DEBUG=TRUE`: Compiles a debug version of QUICK with extra prints enabled.
- `-DQUICK_DEBUG_TIME=TRUE`: Compiles a debug version of QUICK that reports more information on timing.

#### External library control

- `-DFORCE_INTERNAL_LIBS=blas`: Forces use of the internal BLAS library even if a system one is available.
- `-DFORCE_DISABLE_LIBS=mkl`: Disable use of system MKL to replace BLAS and LAPACK.
- `-DBLAS_LIBRARIES*/*-DLAPACK_LIBRARIES=<path>`: path to BLAS/LAPACK library files
- `-DCMAKE_PREFIX_PATH=<path>`: Use the given path as a prefix where dependencies are installed. Libraries and headers will be searched for in <path>/lib and <path>/include.
- `-DMKL_HOME=<path>`: Look for Intel MKL in the given directory. The environment variable MKL_HOME is also searched. *NOTE:* When using this flag, the additional flag *-DTRUST_SYSTEM_LIBS=TRUE* must also be appended.
- `-DMKL_MULTI_THREADED=<TRUE|FALSE>`: Specify whether the Intel MKL library should be used as single or multi-threaded.
- `-DMAGMA=TRUE`: Enable matrix diagonalization using Magma library in HIP/HIP-MPI version.
- `-DMAGMA_PATH=<path>`: Look for Magma library in the given directory. 
- `-DHDF5=TRUE`: Sets up quick restart capabilities using HDF5 data file. Please compile with HDF5 for full restart capability.

#### Parallel versions
By default QUICK will only build the serial version. This can be changed with
these options:

- `-DMPI=TRUE`: Also build MPI versions of all programs.
- `-DCUDA=TRUE`: Also build CUDA versions of all programs. If both MPI and CUDA are active at the same time, a MPI+CUDA version will additionally be built.
- `-DHIP=TRUE`: Build HIP versions of all programs. If both MPI and HIP are active at the same time, a MPI+HIP version will additionally be built.
- `-DQUICK_USER_ARCH=<kepler|maxwell|pascal|volta|turing|ampere|adalovelace|hopper|blackwell|blackwell2|gfx906|gfx908|gfx90a|gfx942>`: Build CUDA/HIP codes only for the given architecture. If not provided, CUDA versions will be compiled for multiple architectures based on the CUDA toolkit version, while HIP versions will be compiled for gfx908. Note that the build report after the configure phase in the build system lists the GPU target architectures. See the listing there for additional details.
- `-DQUICK_VERBOSE_PTXAS=TRUE`:  Pass -v flag to ptxas to dump details about compiled functions in CUDA code.
- `-DHIP_TOOLKIT_ROOT_DIR=<path>`: Path to ROCm installation where hip, rocBLAS, rocSolver etc. directories are located. 
