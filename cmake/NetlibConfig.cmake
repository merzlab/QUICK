# This script sets up a target based on the math libraries currently configured:
# netlib -- linear algebra libraries (blas and lapack), arpack, and whatever supporting math libraries are necessary.
# do NOT link any of the netlib libraries directly except through the netlib target.  That will not work when MKL is enabled.  
# Also, make ABSOLUTELY sure that fftw is AHEAD OF netlib in the link order of any targets that use them together.
# This is so that it overrides MKL's weird partial fftw interface.  If you do not do this, you may get undefined behavior.

set(NETLIB_LIBRARIES "")

# basic linear algebra libraries
if(mkl_ENABLED)
    list(APPEND NETLIB_LIBRARIES ${MKL_FORTRAN_LIBRARIES})
    
   	#this affects all Amber programs
    add_definitions(-DMKL)
else()
	list(APPEND NETLIB_LIBRARIES blas lapack)
endif()

# arpack
if(arpack_ENABLED)
	list(APPEND NETLIB_LIBRARIES arpack)
endif()

# if the system lacks a C99 complex library, use our own implementation
if(c9x-complex_ENABLED)
    list(APPEND NETLIB_LIBRARIES mc)
endif()

# link system math library if it exists
list(APPEND NETLIB_LIBRARIES C::Math)


# --------------------------------------------------------------------

import_libraries(netlib LIBRARIES ${NETLIB_LIBRARIES})
