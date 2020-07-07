MAKEIN = ../../make.in
include $(MAKEIN)

####################################################################
#  BLAS make include file.                                         #
#  March 2007                                                      #
####################################################################
#
SHELL = /bin/sh
#
#  The machine (platform) identifier to append to the library names
#
PLAT = _LINUX
#  
#  Modify the FORTRAN and OPTS definitions to refer to the
#  compiler and desired compiler options for your machine.  NOOPT
#  refers to the compiler options desired when NO OPTIMIZATION is
#  selected.  Define LOADER and LOADOPTS to refer to the loader and 
#  desired load options for your machine.
#
FORTRAN  = $(FC)
OPTS     = -O3 $(LIB_FLAGS)
DRVOPTS  = $(OPTS)
NOOPT    =
LOADER   = $(FC)
LOADOPTS =
#
#  The archiver and the flag(s) to use when building archive (library)
#  If you system has no ranlib, set RANLIB = echo.
#
ifeq "$(LIB_TYPE)" "SHARED"
  ARCH     = $(FC)
  ARCHFLAGS= -shared -o
  RANLIB   = echo
else
  ARCH     = ar
  ARCHFLAGS= cr
  RANLIB   = ranlib
endif
#
#  The location and name of the Reference BLAS library.
#
ifeq "$(LIB_TYPE)" "SHARED"
  BLASLIB      = libblas.so 
else
  BLASLIB      = libblas.a
endif
