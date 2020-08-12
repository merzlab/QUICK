# - Find ARPACK
# Finds the arpack library on your system, and check that it is able to be linked to with the current compiler.
# This module defines:
#
#  ARPACK_LIBRARY      -- the library needed to use Arpack
#  ARPACK_FOUND        -- if false, do not try to use Arpack.
#  ARPACK_HAS_ARSECOND -- true if the discovered Arpack library contains the arsecond_ function.  Only defined if Arpack was found.

include(FindPackageHandleStandardArgs)

find_library(ARPACK_LIBRARY arpack)

set(FIND_ARPACK_FAILURE_MESSAGE "The ARPACK library was not found.  Please set ARPACK_LIBRARY to point to it.")

if(EXISTS "${ARPACK_LIBRARY}")

	try_link_library(ARPACK_WORKS
		LANGUAGE Fortran
		FUNCTION dsaupd
		LIBRARIES ${ARPACK_LIBRARY})
	
	if(ARPACK_WORKS)
	
		# Test for arsecond_

		#Some arpacks (e.g. Ubuntu's package manager's one) don't have the arsecond_ function from wallclock.c
		#sff uses it, so we have to tell sff to build it

		try_link_library(ARPACK_HAS_ARSECOND
			LANGUAGE Fortran
			FUNCTION arsecond
			LIBRARIES ${ARPACK_LIBRARY})
		
	else()
		set(FIND_ARPACK_FAILURE_MESSAGE "The ARPACK library was found, but ${ARPACK_LIBRARY} is not linkable.  Perhaps it was built with an incompatible Fortran compiler? \
Please set ARPACK_LIBRARY to point to a working ARPACK library.")
	endif()
	
else()

	# make sure that this variable isn't hanging around from a previous time when Arpack was found
	unset(ARPACK_HAS_ARSECOND CACHE)
endif()

find_package_handle_standard_args(ARPACK ${FIND_ARPACK_FAILURE_MESSAGE} ARPACK_LIBRARY ARPACK_WORKS)