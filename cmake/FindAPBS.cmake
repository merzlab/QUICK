# - Find APBS
# Find the native APBS includes and library
# This module defines the following variables:
#
#  APBS_LIBRARIES, the libraries needed to use APBS.
#  APBS_FOUND, If false, do not try to use APBS.
#  APBS_INNER_INCLUDES, directories to include so that you can #include <apbs.h>
#  APBS_INCLUDES, directory to include so that you can #include <apbs/apbs.h>

include(FindPackageHandleStandardArgs)

find_library(APBS_API_LIB iapbs)
find_library(APBS_GENERIC_LIB apbs_generic)
find_library(APBS_ROUTINES_LIB apbs_routines)
find_library(APBS_PMGC_LIB apbs_pmgc)
find_library(APBS_MG_LIB apbs_mg)
find_library(APBS_MALOC_LIB maloc)

set(APBS_LIBRARIES ${APBS_API_LIB} ${APBS_ROUTINES_LIB} ${APBS_MG_LIB} ${APBS_PMGC_LIB} ${APBS_GENERIC_LIB} ${APBS_MALOC_LIB} )

# on Windows, maloc needs to link to ws2_32.dll
if("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
	# we can probably get away with not using find_library() here since ws2_32 is a standard universal Windows library
	list(APPEND APBS_LIBRARIES ws2_32)
endif()

find_path(APBS_INCLUDES apbs/apbs.h)

#some of apbs's headers #include <maloc/maloc.h>, so we have to supply the outer include directory as well
set(APBS_INNER_INCLUDES ${APBS_INCLUDES}/apbs ${APBS_INCLUDES})
	
if(NOT(APBS_GENERIC_LIB AND APBS_ROUTINES_LIB AND APBS_PMGC_LIB AND APBS_MG_LIB AND APBS_MALOC_LIB))

	set(FIND_APBS_FAILURE_MESSAGE "Could not find some or all of the five main APBS libraries. Please set APBS_GENERIC_LIB, APBS_ROUTINES_LIB, 
APBS_PMGC_LIB, APBS_MG_LIB, and APBS_MALOC_LIB to point to the correct libraries")

elseif(NOT APBS_API_LIB)

	set(FIND_APBS_FAILURE_MESSAGE "Could not find the APBS API library libiapbs.  Configure APBS with -DENABLE_iAPBS=TRUE for this library to be built.")
	
else()

	# check if APBS works
	try_link_library(APBS_WORKS
		LANGUAGE Fortran
		FUNCTION apbsdrv
		LIBRARIES ${APBS_LIBRARIES})

	if(APBS_WORKS)
		set(FIND_APBS_FAILURE_MESSAGE "Could not find the APBS headers.  Please set APBS_INCLUDES to point to the directory containing the directory containing apbs.h.")
	else()
		set(FIND_APBS_FAILURE_MESSAGE "Could not link with APBS.")
	endif()

endif()

find_package_handle_standard_args(APBS ${FIND_APBS_FAILURE_MESSAGE} APBS_API_LIB APBS_ROUTINES_LIB APBS_MG_LIB APBS_PMGC_LIB APBS_GENERIC_LIB APBS_MALOC_LIB APBS_WORKS APBS_INCLUDES)