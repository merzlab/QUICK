# Make config file for MPI

if(MPI OR CUDAMPI)

	find_package(MPI)

	foreach(LANG ${ENABLED_LANGUAGES})
        if(NOT MPI_${LANG}_FOUND)
            message(FATAL_ERROR "You requested MPI, but the MPI ${LANG} library was not found.  \
Please install one and try again, or set MPI_${LANG}_INCLUDE_PATH and MPI_${LANG}_LIBRARIES to point to your MPI.")
        endif()
        message(STATUS "MPI ${LANG} Compiler: ${MPI_${LANG}_COMPILER}")
    endforeach()

	# Add MPI support to an object library
	macro(mpi_object_library TARGET LANGUAGE)
		# Note: In CMake 3.12, you can link a target to an object library directly
		# to apply its interface properties.  However, we don't have that feature yet.
		target_compile_options(${TARGET} PRIVATE ${MPI_${LANG}_COMPILE_OPTIONS})
		target_include_directories(${TARGET} PUBLIC ${MPI_${LANG}_INCLUDE_PATH})
	endmacro()

	# make a MPI version of the thing passed 
	# also allows switching out sources if needed
	# INSTALL - causes the new target to get installed in the MPI component to the default location (BINDIR etc)
	# usage: make_mpi_version(<target> <new name> LANGUAGES <language 1> [<language 2...>] [SWAP_SOURCES <source 1...> TO <replacement source 1...>] INSTALL)
	function(make_mpi_version TARGET NEW_NAME) 
	
		# parse arguments
		# --------------------------------------------------------------------	
		cmake_parse_arguments(MAKE_MPI "INSTALL" "" "LANGUAGES;SWAP_SOURCES;TO" ${ARGN})
	
		if("${MAKE_MPI_LANGUAGES}" STREQUAL "")
			message(FATAL_ERROR "Incorrect usage.  At least one LANGUAGE should be provided.")
		endif()
		
		if(NOT "${MAKE_MPI_UNPARSED_ARGUMENTS}" STREQUAL "")
			message(FATAL_ERROR "Incorrect usage.  Extra arguments provided.")
		endif()
	
		
		# figure out if it's an object library, and if so, use mpi_object_library()		
		get_property(TARGET_TYPE TARGET ${TARGET} PROPERTY TYPE)
		
		if("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
			set(IS_OBJECT_LIBRARY TRUE)
		else()
			set(IS_OBJECT_LIBRARY FALSE)
		endif()
		
		if("${ARGN}" STREQUAL "")
			message(FATAL_ERROR "make_mpi_version(): you must specify at least one LANGUAGE") 
		endif()
		
		# make a new one
		# --------------------------------------------------------------------
		if("${MAKE_MPI_SWAP_SOURCES}" STREQUAL "" AND "${MAKE_MPI_TO}" STREQUAL "")
			copy_target(${TARGET} ${NEW_NAME})
		else()
			copy_target(${TARGET} ${NEW_NAME} SWAP_SOURCES ${MAKE_MPI_SWAP_SOURCES} TO ${MAKE_MPI_TO})
		endif()
		
		# this ensures that the new version builds after all of the target's dependencies 
		# that have been manually added with add_dependencies() have been satisfied.
		# Yes it is a bit of an ugly hack, but since we can't copy dependencies, this is the next-best thing.
		add_dependencies(${NEW_NAME} ${TARGET})
		
		# apply MPI flags
		# --------------------------------------------------------------------
		foreach(LANG ${MAKE_MPI_LANGUAGES})
			# validate arguments
			if(NOT ("${LANG}" STREQUAL "C" OR "${LANG}" STREQUAL "CXX" OR "${LANG}" STREQUAL "Fortran"))
				message(FATAL_ERROR "make_mpi_version(): invalid argument: ${LANG} is not a LANGUAGE")
			endif()
			
			if(IS_OBJECT_LIBRARY)
				mpi_object_library(${NEW_NAME} ${LANG})
			else()
				# don't use target_link_libraries here to avoid "picking a side" with
				# plain vs keyword form of target_link_libraries
				set_property(TARGET ${NEW_NAME} APPEND PROPERTY LINK_LIBRARIES ${MPI_${LANG}})
				set_property(TARGET ${NEW_NAME} APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_${LANG}})
			endif()
			
		endforeach()
		
		# install if necessary
		# --------------------------------------------------------------------
		if(MAKE_MPI_INSTALL)
			if("${TARGET_TYPE}" STREQUAL "EXECUTABLE")
				install(TARGETS ${NEW_NAME} DESTINATION ${BINDIR} COMPONENT MPI)
			else()
				install_libraries(${NEW_NAME} COMPONENT MPI)
			endif()
		endif()
		
	endfunction(make_mpi_version)

endif()

