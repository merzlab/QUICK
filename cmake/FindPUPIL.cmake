# - Find PUPIL
# Find the native PUPIL libraries
# NOTE: the Java link library is required to link with PUPIL, so this module will find it as well
#
# This module defines the following variables:
#
#  PUPIL_LIBRARIES, the libraries needed to use PUPIL.
#  PUPIL_FOUND, If false, do not try to use PUPIL.
#
# If pupil was found, this module also creates the following imported target:
#  pupil::pupil - Target for pupil library

include(FindPackageHandleStandardArgs)

find_library(PUPIL_MAIN_LIB 
	NAMES PUPIL)
find_library(PUPIL_BLIND_LIB
	NAMES PUPILBlind)
find_library(PUPIL_TIME_LIB 
	NAMES PUPILTime)

find_package(JNI)

set(PUPIL_LIBRARIES ${PUPIL_MAIN_LIB} ${PUPIL_BLIND_LIB} ${PUPIL_TIME_LIB} ${JNI_LIBRARIES})

try_link_library(PUPIL_WORKS
	LANGUAGE Fortran
	FUNCTION getquantumforces
	LIBRARIES ${PUPIL_LIBRARIES})

find_package_handle_standard_args(PUPIL REQUIRED_VARS PUPIL_MAIN_LIB PUPIL_BLIND_LIB PUPIL_TIME_LIB JNI_FOUND PUPIL_WORKS)

if(PUPIL_FOUND)
	add_library(pupil::pupil_lib UNKNOWN IMPORTED)
    set_property(TARGET pupil::pupil_lib PROPERTY IMPORTED_LOCATION ${PUPIL_MAIN_LIB})

	add_library(pupil::blind_lib UNKNOWN IMPORTED)
    set_property(TARGET pupil::blind_lib PROPERTY IMPORTED_LOCATION ${PUPIL_BLIND_LIB})
    
    add_library(pupil::time_lib UNKNOWN IMPORTED)
    set_property(TARGET pupil::time_lib PROPERTY IMPORTED_LOCATION ${PUPIL_TIME_LIB})

    # add combined target
	add_library(pupil::pupil INTERFACE IMPORTED)
	set_property(TARGET pupil::pupil PROPERTY INTERFACE_LINK_LIBRARIES pupil::pupil_lib pupil::blind_lib pupil::time_lib ${JNI_LIBRARIES})
endif()
