#CMake config file for MPI
# MUST be included after OpenMPConfig, if OpenMPConfig is included at all

# Creates these variables (in addition to what is produced by FindMPI):
# MPI_<lang>_COMPILE_OPTIONS - List of MPI compile options for a language
# MPI_<lang> - List that can be added to link libraries.  Links the MPI libraries for the specified language, and applies MPI compile flags for all languages present in the target to the correct source files.  

option(MPI "Build ${PROJECT_NAME} with MPI inter-machine parallelization support." FALSE)

include(ParallelizationConfig)

if(MPI)
	find_package(MPI)
	
	foreach(LANG ${ENABLED_LANGUAGES})
		if(NOT MPI_${LANG}_FOUND)
			message(FATAL_ERROR "You requested MPI, but the MPI ${LANG} library was not found.  \
Please install one and try again, or set MPI_${LANG}_INCLUDE_PATH and MPI_${LANG}_LIBRARIES to point to your MPI.")
		endif()
	
		message(STATUS "MPI ${LANG} Compiler: ${MPI_${LANG}_COMPILER}")
	endforeach()
	
	message(STATUS "If these are not the correct MPI wrappers, then set MPI_<language>_COMPILER to the correct wrapper and reconfigure.")
	
	# the MinGW port-hack of MS-MPI needs to be compiled with -fno-range-check
	if("${MPI_Fortran_LIBRARIES}" MATCHES "msmpi" AND "${CMAKE_Fortran_COMPILER_ID}" STREQUAL GNU)
		message(STATUS "MS-MPI range check workaround active")
		
		#create a non-cached variable with the contents of the cache variable plus extra flags
		set(MPI_Fortran_COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS} -fno-range-check -Wno-conversion")
	endif()
			
	# create imported targets and variables
	# --------------------------------------------------------------------

	# Amber MPI target that provides automatic MPI flags for all languages
	# using generator expressions.
	# This is is important when we have mixed language programs that use MPI --
	# it allows the correct flags to get passed to the correct compiler
	add_library(MPI_all_language_flags INTERFACE)

	foreach(LANG ${ENABLED_LANGUAGES})
		using_library_targets(MPI::MPI_${LANG})

		# get list form of MPI compile flags from target
		get_property(MPI_${LANG}_COMPILE_OPTIONS TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_COMPILE_OPTIONS)

		# add compile flags and include dirs for source files of this language
		target_compile_options(MPI_all_language_flags INTERFACE $<$<COMPILE_LANGUAGE:${LANG}>:${MPI_${LANG}_COMPILE_OPTIONS}>)
		target_include_directories(MPI_all_language_flags INTERFACE $<$<COMPILE_LANGUAGE:${LANG}>:${MPI_${LANG}_INCLUDE_PATH}>)

		# C++ MPI doesn't like having -DMPI defined, but it's used as the MPI flag
		# in most Amber C and Fortran code
		if(NOT ${LANG} STREQUAL CXX)
			target_compile_definitions(MPI_all_language_flags INTERFACE $<$<COMPILE_LANGUAGE:${LANG}>:MPI>)
		endif()

		# generate library list for MPI of this language
		set(MPI_${LANG}
			# when building, use our custom flags target
			$<BUILD_INTERFACE:MPI_all_language_flags>
			# Now link MPI only as a link library
			$<BUILD_INTERFACE:${MPI_${LANG}_LIBRARIES}>
			$<BUILD_INTERFACE:${MPI_${LANG}_LINK_FLAGS}>

			# in the install interface, just use the MPI target itself
			$<INSTALL_INTERFACE:MPI_${LANG}>)
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

