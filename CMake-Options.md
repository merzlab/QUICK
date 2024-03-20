## CMake Build System Options

This page gives a summary of CMake options that can be used with QUICK.  All common options should be included here, though several others exist that are used with the Amber build system framework.

Note that like all CMake options, these options are sticky.  Once passed to CMake, they will remain set unless you set them to a different value (with -D), unset them (with -U), or delete the build directory.

#### General options
- `-DCOMPILER`: Allows selection of the toolchain to use.  `-DCOMPILER=AUTO` to use CMake default behavior.  Run CMake with no arguments to see the list of options.
- `-DENABLEF=<TRUE|FALSE>`: Enables the compilation of time consuming f functions in the ERI code of cuda version.
- `-DCMAKE_BUILD_TYPE=<Debug|Release>`: Controls whether to build with debugging symbols (`Debug`) or not (`Release`).
- `-DOPTIMIZE=<TRUE|FALSE>`: Controls whether to enable compiler optimizations.  On by default.
- `-DCMAKE_INSTALL_PREFIX=...`: Controls where QUICK will be installed when you run `make install`.
- `-DQUICK_DEBUG=TRUE`: Compiles a debug version of QUICK with extra prints enabled.
- `-DQUICK_DEBUG_TIME=TRUE`: Compiles a debug version of QUICK that reports more information on timing.

#### External library control
QUICK will use all external libraries that it can find (and which appear to work) on the system.
These options allow you do adjust this behavior.

- `-DFORCE_INTERNAL_LIBS=blas`: Forces use of the internal BLAS library even if a system one is available.
- `-DFORCE_DISABLE_LIBS=mkl`: Disable use of system MKL to replace BLAS and LAPACK 
- `-DCMAKE_PREFIX_PATH=<path>`: Use the given path as a prefix where dependencies are installed. Libraries and headers will be searched for in `<path>/lib` and `<path>/include`.
- `-DMKL_HOME=...`: Look for Intel MKL in the given directory.  The environment variable MKL_HOME is also searched.

#### Parallel versions
By default QUICK will only build the serial version.  This can be changed with these options:
- `-DMPI=TRUE`: Also build MPI versions of all programs.
- `-DCUDA=TRUE`: Also build CUDA versions of all programs.  If both MPI and CUDA are active at the same time, CUDA MPI versions will additionally be built.
- `-DQUICK_USER_ARCH=<kepler|maxwell|pascal|volta|turing|ampere|adalovelace>`: Build CUDA code only for the given architecture.  If not provided, quick will compile for all supported architectures in your CUDA version.
- `-DQUICK_VERBOSE_PTXAS=TRUE`: Pass -v flag to ptxas to dump details about compiled functions in CUDA code.
