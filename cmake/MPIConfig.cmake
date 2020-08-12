#CMake config file for MPI
# MUST be included after OpenMPConfig, if OpenMPConfig is included at all
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
	
	foreach(LANG C CXX Fortran)
		
		#Trim leading spaces from the compile flags.  They cause problems with PROPERTY COMPILE_OPTIONS
		
		# this shadows the cache variable with a local variable	
		string(STRIP "${MPI_${LANG}_COMPILE_FLAGS}" MPI_${LANG}_COMPILE_FLAGS)
		string(STRIP "${MPI_${LANG}_LINK_FLAGS}" MPI_${LANG}_LINK_FLAGS)
		
		#convert space seperated flag variables into proper lists
		separate_arguments(MPI_${LANG}_COMPILE_OPTIONS UNIX_COMMAND "${MPI_${LANG}_COMPILE_FLAGS}")
		separate_arguments(MPI_${LANG}_LINK_OPTIONS UNIX_COMMAND "${MPI_${LANG}_LINK_FLAGS}")
		
		# --------------------------------------------------------------------
		# sometimes MPI will include flags of the form "-Xlinker -rpath -Xlinker /usr/foo".  Unfortunately,
		# This causes errors because it conflicts with CMake's automatic RPATH flags.
		# get rid of RPATH flags; CMake will set those itself
		string(REPLACE "-Xlinker;-rpath;" "" MPI_${LANG}_LINK_OPTIONS "${MPI_${LANG}_LINK_OPTIONS}")
		
		# for each two arguments, check if they are either "-Xlinker;<some directory>", "-Wl,-rpath,...", or "-Wl,<some directory>". If so, delete them.
		list(LENGTH MPI_${LANG}_LINK_OPTIONS NUM_LINK_OPT)
		
		set(INDEX 1)
		set(TEMP_REBUILT_LINK_OPTIONS "")
		
		while(INDEX LESS NUM_LINK_OPT)
			math(EXPR FIRST_INDEX "${INDEX} - 1")
		
			list(GET MPI_${LANG}_LINK_OPTIONS ${FIRST_INDEX} FIRST_ELEMENT)
			list(GET MPI_${LANG}_LINK_OPTIONS ${INDEX} SECOND_ELEMENT)
			
			set(IS_XLINKER_DIRECTORY FALSE)
			set(IS_WL_DIRECTORY FALSE)
			set(IS_WL_RPATH FALSE)
			
			# try and figure out if SECOND_ELEMENT looks like a folder path
			if("${FIRST_ELEMENT}" STREQUAL "-Xlinker" AND "${SECOND_ELEMENT}" MATCHES "^/.*")
				if(EXISTS "${SECOND_ELEMENT}")
					if(IS_DIRECTORY "${SECOND_ELEMENT}")
						set(IS_XLINKER_DIRECTORY TRUE)
					endif()
				else()
					# sometimes, we are passed nonexistant directories.
					# in this case, as long as they don't have an extension, remove them.
					
					get_filename_component(PATH_EXTENSION "${SECOND_ELEMENT}" EXT)
					
					if("${PATH_EXTENSION}" STREQUAL "")
						set(IS_XLINKER_DIRECTORY TRUE)
					endif()
				endif()
			elseif("${FIRST_ELEMENT}" MATCHES "-Wl,(/.*)")
				if(EXISTS "${CMAKE_MATCH_1}")
					if(IS_DIRECTORY "${CMAKE_MATCH_1}")
						set(IS_WL_DIRECTORY TRUE)
					endif()
				else()
					# sometimes, we are passed nonexistant directories.
					# in this case, as long as they don't have an extension, remove them.
					
					get_filename_component(PATH_EXTENSION "${CMAKE_MATCH_1}" EXT)
					
					if("${PATH_EXTENSION}" STREQUAL "")
						set(IS_WL_DIRECTORY TRUE)
					endif()
				endif()
			elseif("${FIRST_ELEMENT}" MATCHES "-Wl,-rpath.*")
				set(IS_WL_RPATH TRUE)
			endif()
			
			if(IS_XLINKER_DIRECTORY)
				math(EXPR INDEX "${INDEX} + 2")
				message(STATUS "Found and removed RPATH control flags from MPI flags")
			elseif(IS_WL_DIRECTORY)
				math(EXPR INDEX "${INDEX} + 1")
				message(STATUS "Found and removed RPATH control flags from MPI flags")
			elseif(IS_WL_RPATH)
				math(EXPR INDEX "${INDEX} + 1")
				message(STATUS "Found and removed RPATH control flags from MPI flags")
			else()
				list(APPEND TEMP_REBUILT_LINK_OPTIONS ${FIRST_ELEMENT})
				math(EXPR INDEX "${INDEX} + 1")
			endif()
		endwhile()
		
		set(MPI_${LANG}_LINK_OPTIONS ${TEMP_REBUILT_LINK_OPTIONS})
					
	endforeach()
			
	# create imported targets
	# --------------------------------------------------------------------
	foreach(LANG ${ENABLED_LANGUAGES})
		string(TOLOWER ${LANG} LANG_LOWERCASE)
		
		import_libraries(mpi_${LANG_LOWERCASE} LIBRARIES ${MPI_${LANG}_LINK_OPTIONS} ${MPI_${LANG}_LIBRARIES} INCLUDES ${MPI_${LANG}_INCLUDE_PATH})
		set_property(TARGET mpi_${LANG_LOWERCASE} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MPI_${LANG}_INCLUDE_PATH})
		
		if(MCPAR_WORKAROUND_ENABLED)
			# use generator expression
			set_property(TARGET mpi_${LANG_LOWERCASE} PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:${LANG}>:${MPI_${LANG}_COMPILE_OPTIONS}>)
		else()
			set_property(TARGET mpi_${LANG_LOWERCASE} PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_${LANG}_COMPILE_OPTIONS})
		endif()
		
		# C++ MPI doesn't like having "MPI" defined, but it's what Amber uses as the MPI switch in most programs (though not EMIL)
		if(NOT ${LANG} STREQUAL CXX)
			set_property(TARGET mpi_${LANG_LOWERCASE} PROPERTY INTERFACE_COMPILE_DEFINITIONS MPI)	
		endif()
			
	endforeach()
	
	
	# Add MPI support to an object library
	macro(mpi_object_library TARGET LANGUAGE)
		if(MCPAR_WORKAROUND_ENABLED)
			# use generator expression
			set_property(TARGET ${TARGET} APPEND PROPERTY COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:${LANGUAGE}>:${MPI_${LANGUAGE}_COMPILE_OPTIONS}>)
		else()
			set_property(TARGET ${TARGET} APPEND PROPERTY COMPILE_OPTIONS ${MPI_${LANGUAGE}_COMPILE_OPTIONS})
		endif()

		target_include_directories(${TARGET} PUBLIC ${MPI_${LANGUAGE}_INCLUDE_PATH})
		
		if(NOT ${LANGUAGE} STREQUAL CXX)
			target_compile_definitions(${TARGET} PRIVATE MPI)
		endif()
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
		
		# this ensures that the MPI version builds after all of the target's dependencies have been satisfied.
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
				string(TOLOWER ${LANG} LANG_LOWERCASE)
				target_link_libraries(${NEW_NAME} mpi_${LANG_LOWERCASE})
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

