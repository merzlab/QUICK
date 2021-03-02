# - Find XBlas, the extended-precision BLAS library.
#
# This module defines the following variables:
#     XBLAS_LIBRARY  - Library to link for xblas
#     XBLAS_FOUND - Whether xblas was found
#
# If xblas was found, this module also creates the following imported target:
#     xblas::xblas - Target to link xblas

# search for the library
find_library(XBLAS_LIBRARY 
	NAMES xblas xblas-amb
	DOC "Path to libxblas.a")

# check that library works
if(XBLAS_LIBRARY)

	try_link_library(XBLAS_C_WORKS
		LANGUAGE C
		FUNCTION BLAS_dgemm_x
		LIBRARIES ${XBLAS_LIBRARY})

	try_link_library(XBLAS_FORTRAN_WORKS
		LANGUAGE Fortran
		FUNCTION BLAS_dgemm_x
		LIBRARIES ${XBLAS_LIBRARY})
endif()

find_package_handle_standard_args(XBLAS
	REQUIRED_VARS XBLAS_LIBRARY XBLAS_C_WORKS XBLAS_FORTRAN_WORKS)

add_library(xblas::xblas UNKNOWN IMPORTED)
set_property(TARGET xblas::xblas PROPERTY IMPORTED_LOCATION ${XBLAS_LIBRARY})
