# Find the (internal or external) KMMD includes and library
# This module defines the following variables:
#
#  KMMD_LIB, the library needed to use KMMD (Machine Learning MD).
#  KMMD_INCLUDES, directory to include so that you can #include <kmmd.h>
#  KMMD_FOUND,    If false, do not try to use KMMD.

include(FindPackageHandleStandardArgs)

message(STATUS "hints given to look for KMMD in DIR: ${KMMD_DIR}")

find_library(KMMD_LIB libkmmd.so HINTS ${KMMD_DIR})
message(STATUS "found KMMD?:   ${KMMD_LIB}")


#find_path(KMMD_INCLUDES kmmd.h HINTS ${KMMD_DIR})

message(STATUS "found INCLUDES?: " ${KMMD_INCLUDES})

	
if(NOT(KMMD_LIB))

	set(FIND_KMMD_FAILURE_MESSAGE "Did not find an external KMMD library, should be able to use bundled. If you have a custom version then set the path with -DKMMD_DIR")
else()

	# check if KMMD works
	#try_link_library(KMMD_WORKS
	#	LANGUAGE CXX
	#	FUNCTION hash_function 
	#	LIBRARIES ${KMMD_LIB})

	#if(KMMD_WORKS)
	#	set(FIND_KMMD_FAILURE_MESSAGE "Could not find the kmmd headers.  Please set KMMD_INCLUDES to point to the directory containing the directory containing kmmd.h.")
	#else()
	#	set(FIND_KMMD_FAILURE_MESSAGE "Could not link with KMMD.")
	#endif()

endif()

find_package_handle_standard_args(KMMD ${FIND_KMMD_FAILURE_MESSAGE} KMMD_LIB) # KMMD_INCLUDES)
