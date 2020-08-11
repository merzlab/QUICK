# Make config file for MPI
# MUST be included after OpenMPConfig, if OpenMPConfig is included at all

# change config.h for libxc
if(MPI)
	find_package(MPI)
	if(MPI_FOUND)
        message("find MPI!!!")
		message("after find_package(MPI), MPI_C_INCLUDE_PATH is ${MPI_C_INCLUDE_PATH}")
    endif()
	
	add_definitions(-DMPIV)	

	foreach(LANG ${ENABLED_LANGUAGES})
		message("LANG is ${LANG}")
		message("MPI_${LANG}_INCLUDE_PATH is ${MPI_${LANG}_INCLUDE_PATH}")
		message("MPI_${LANG}_LIBRARIES is ${MPI_${LANG}_LIBRARIES}")
        if(NOT MPI_${LANG}_FOUND)
            message(FATAL_ERROR "You requested MPI, but the MPI ${LANG} library was not found.  \
Please install one and try again, or set MPI_${LANG}_INCLUDE_PATH and MPI_${LANG}_LIBRARIES to point to your MPI.")
        endif()
        message(STATUS "MPI ${LANG} Compiler: ${MPI_${LANG}_COMPILER}")
    endforeach()


endif()


