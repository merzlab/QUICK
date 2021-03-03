# - Find PLUMED
# 
# This module finds the PLUMED molecular dynamics library on your system.
#
# Variables: 
#  PLUMED_LIBRARIES - Libraries to link to use PLUMED as a shared library
#  PLUMED_INCLUDES - Path where PLUMED's headers can be found
#  PLUMED_FOUND - Whether PLUMED was found
#  PLUMED_VERSION - Version of plumed found, if it was found
#  
# If PLUMED was found, this module also creates the following imported target:
#  plumed::plumed - Target for plumed's libraries

# if the plumed executable is in the path, we can try to use it to find where plumed is installed
get_filename_component(PLUMED_EXECUTABLE_PATH plumed PROGRAM)

set(PLUMED_HINTS "" CACHE PATH "Directories where PLUMED might be installed" INTERNAL)

if(EXISTS "${PLUMED_EXECUTABLE_PATH}" AND "${PLUMED_HINTS}" STREQUAL "")
	execute_process(
		COMMAND ${PLUMED_EXECUTABLE_PATH} info --configuration
		RESULT_VARIABLE PLUMED_EXIT_CODE
		OUTPUT_VARIABLE PLUMED_CONFIG_OUTPUT)

	if(PLUMED_EXIT_CODE STREQUAL 0)
		if("${PLUMED_CONFIG_OUTPUT}" MATCHES "prefix=(.+)")
			set(PLUMED_HINTS "${CMAKE_MATCH_1}" CACHE PATH "Directories where PLUMED might be installed" INTERNAL FORCE)
		endif()
	endif()
endif()

# search for libraries
find_library(PLUMED_KERNEL_LIBRARY
	NAMES plumedKernel
	DOC "Path to PLUMED kernel library"
	HINTS ${PLUMED_HINTS})

find_library(PLUMED_LIBRARY
	NAMES plumed
	DOC "Path to PLUMED library"
	HINTS ${PLUMED_HINTS})

set(PLUMED_LIBRARIES ${PLUMED_LIBRARY} ${PLUMED_KERNEL_LIBRARY})

# search for headers
find_path(PLUMED_INCLUDES
	NAMES wrapper/Plumed.h
	PATH_SUFFIXES plumed
	DOC "Path where PLUMED's includes are.  Should contain wrapper/Plumed.h"
	HINTS ${PLUMED_HINTS})

# find version number
if(EXISTS ${PLUMED_INCLUDES}/config/version.h)
	file(READ ${PLUMED_INCLUDES}/config/version.h PLUMED_VERSION_HEADER_CONTENT)
	if("${PLUMED_VERSION_HEADER_CONTENT}" MATCHES "#define PLUMED_VERSION_LONG \"([0-9.]+)\"")
		set(PLUMED_VERSION ${CMAKE_MATCH_1})
	endif()
endif()

# check functionality
# plumed_symbol_table_reexport is the function called by the wrapper to get funciton pointers from the library
try_link_library(PLUMED_WORKS
	LANGUAGE C
	FUNCTION plumed_symbol_table_reexport
	LIBRARIES ${PLUMED_LIBRARIES})


find_package_handle_standard_args(PLUMED
	REQUIRED_VARS PLUMED_LIBRARY PLUMED_KERNEL_LIBRARY PLUMED_INCLUDES PLUMED_WORKS)

# create imported target
if(PLUMED_FOUND)
	add_library(plumed::plumed_lib UNKNOWN IMPORTED)
    set_property(TARGET plumed::plumed_lib PROPERTY IMPORTED_LOCATION ${PLUMED_LIBRARY})
    set_property(TARGET plumed::plumed_lib PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PLUMED_INCLUDES})

    add_library(plumed::kernel_lib UNKNOWN IMPORTED)
    set_property(TARGET plumed::kernel_lib PROPERTY IMPORTED_LOCATION ${PLUMED_KERNEL_LIBRARY})
    set_property(TARGET plumed::kernel_lib PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PLUMED_INCLUDES})

    # add combined target
	add_library(plumed::plumed INTERFACE IMPORTED)
	set_property(TARGET plumed::plumed PROPERTY INTERFACE_LINK_LIBRARIES plumed::plumed_lib plumed::kernel_lib)

endif()