# - Find the LIO library for GPU quantum mechanics calculations.
#
# This module defines the following variables:
#
#  LIO_LIBRARIES - the libraries needed to use lio
#  LIO_FOUND - whether lio was found on the current system
#
# The following variables are accepted as input:
#  ENV{LIOHOME} - Path to built LIO source directory.
#  LIOHOME - Path to built LIO source directory.  Takes preference over the environment version
#
# If lio was found, this module also creates the following imported target:
#  lio::lio - Target for lio library

set(LIO_PATHS "")

if(DEFINED LIOHOME)
	list(APPEND LIO_PATHS ${LIOHOME})
endif()

if(DEFINED ENV{LIOHOME})
	list(APPEND LIO_PATHS $ENV{LIOHOME})
endif()

find_library(LIO_G2G_LIBRARY 
	NAMES g2g 
	PATHS ${LIO_PATHS}
	PATH_SUFFIXES g2g 
	DOC "Path to libg2g.so")

find_library(LIO_AMBER_LIBRARY 
	NAMES lio-g2g 
	PATHS ${LIO_PATHS}
	PATH_SUFFIXES lioamber 
	DOC "Path to liblio-g2g.so")

set(LIO_LIBRARIES ${LIO_G2G_LIBRARY} ${LIO_AMBER_LIBRARY})

try_link_library(LIO_WORKS
	LANGUAGE Fortran
	FUNCTION init_lio_amber
	LIBRARIES ${LIO_LIBRARIES})

find_package_handle_standard_args(LIO
	REQUIRED_VARS LIO_G2G_LIBRARY LIO_AMBER_LIBRARY LIO_WORKS)

# create imported target
if(LIO_FOUND)
	add_library(lio::g2g_lib UNKNOWN IMPORTED)
    set_property(TARGET lio::g2g_lib PROPERTY IMPORTED_LOCATION ${LIO_G2G_LIBRARY})

    add_library(lio::amber_lib UNKNOWN IMPORTED)
    set_property(TARGET lio::amber_lib PROPERTY IMPORTED_LOCATION ${LIO_AMBER_LIBRARY})

    # add combined target
	add_library(lio::lio INTERFACE IMPORTED)
	set_property(TARGET lio::lio PROPERTY INTERFACE_LINK_LIBRARIES lio::amber_lib lio::g2g_lib)
endif()
