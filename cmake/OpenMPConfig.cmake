#Cmake config file for OpenMP
option(OPENMP "Use OpenMP for shared-memory parallelization." FALSE)

include(ParallelizationConfig)

if(OPENMP)
	if(DRAGONEGG)
		message(FATAL_ERROR "OpenMP is not compatible with Dragonegg.  Disable one or the other to build.")
	endif()
	
	find_package(OpenMPFixed)
	
	# check that OpenMP was found for each enabled language
	foreach(LANG ${ENABLED_LANGUAGES})
		if(NOT OpenMP_${LANG}_FOUND)
			message(FATAL_ERROR "You requested OpenMP support, but your ${LANG} compiler doesn't seem to support OpenMP.  Please set OPENMP to FALSE, or switch to a compiler that supports it.")
		endif()
	endforeach()
	
	foreach(LANG ${ENABLED_LANGUAGES})
		# add libraries to library tracker
		using_external_libraries(${OpenMP_${LANG}_LIBRARIES})
	endforeach()
		
	# Add OpenMP support to an object library
	macro(openmp_object_library TARGET LANGUAGE)
		if(MCPAR_WORKAROUND_ENABLED)
			# use generator expression
			set_property(TARGET ${TARGET} PROPERTY COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:${LANGUAGE}>:${OpenMP_${LANGUAGE}_OPTIONS}>)
		else()
			set_property(TARGET ${TARGET} PROPERTY COMPILE_OPTIONS ${OpenMP_${LANGUAGE}_OPTIONS})
		endif()
		
		target_include_directories(${TARGET} PUBLIC ${OpenMP_${LANGUAGE}_INCLUDE_PATH})
	endmacro()
	
	# make an OpenMP version of the thing passed 
	# also allows switching out sources if needed
	# INSTALL - causes the new target to get installed in the OpenMP component to the default location (BINDIR etc)
	# usage: make_openmp_version(<target> <new name> LANGUAGES <language 1> [<language 2...>] [SWAP_SOURCES <source 1...> TO <replacement source 1...>] [INSTALL])
	function(make_openmp_version TARGET NEW_NAME) 
	
		# parse arguments
		# --------------------------------------------------------------------	
		cmake_parse_arguments(MAKE_OPENMP "INSTALL" "" "LANGUAGES;SWAP_SOURCES;TO" ${ARGN})
	
		if("${MAKE_OPENMP_LANGUAGES}" STREQUAL "")
			message(FATAL_ERROR "Incorrect usage.  At least one LANGUAGE should be provided.")
		endif()
		
		if(NOT "${MAKE_OPENMP_UNPARSED_ARGUMENTS}" STREQUAL "")
			message(FATAL_ERROR "Incorrect usage.  Extra arguments provided.")
		endif()
	
		
		# figure out if it's an object library, and if so, use openmp_object_library()		
		get_property(TARGET_TYPE TARGET ${TARGET} PROPERTY TYPE)
		
		if("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
			set(IS_OBJECT_LIBRARY TRUE)
		else()
			set(IS_OBJECT_LIBRARY FALSE)
		endif()
		
		if("${ARGN}" STREQUAL "")
			message(FATAL_ERROR "make_openmp_version(): you must specify at least one LANGUAGE") 
		endif()
		
		# make a new one
		# --------------------------------------------------------------------
		if("${MAKE_OPENMP_SWAP_SOURCES}" STREQUAL "" AND "${MAKE_OPENMP_TO}" STREQUAL "")
			copy_target(${TARGET} ${NEW_NAME})
		else()
			copy_target(${TARGET} ${NEW_NAME} SWAP_SOURCES ${MAKE_OPENMP_SWAP_SOURCES} TO ${MAKE_OPENMP_TO})
		endif()
		
		# this ensures that the new version builds after all of the target's dependencies have been satisfied.
		# Yes it is a bit of an ugly hack, but since we can't copy dependencies, this is the next-best thing.
		add_dependencies(${NEW_NAME} ${TARGET})
		
		# apply OpenMP flags
		# --------------------------------------------------------------------
		foreach(LANG ${MAKE_OPENMP_LANGUAGES})
			# validate arguments
			if(NOT ("${LANG}" STREQUAL "C" OR "${LANG}" STREQUAL "CXX" OR "${LANG}" STREQUAL "Fortran"))
				message(FATAL_ERROR "make_openmp_version(): invalid argument: ${LANG} is not a LANGUAGE")
			endif()
			
			if(IS_OBJECT_LIBRARY)
				openmp_object_library(${NEW_NAME} ${LANG})
			else()
				string(TOLOWER ${LANG} LANG_LOWERCASE)
				target_link_libraries(${NEW_NAME} openmp_${LANG_LOWERCASE})
			endif()
			
		endforeach()
		
		# install if necessary
		# --------------------------------------------------------------------
		if(MAKE_OPENMP_INSTALL)
			if("${TARGET_TYPE}" STREQUAL "EXECUTABLE")
				install(TARGETS ${NEW_NAME} DESTINATION ${BINDIR} COMPONENT OpenMP)
			else()
				install_libraries(${NEW_NAME} COMPONENT OpenMP)
			endif()
		endif()
		
	endfunction(make_openmp_version)
	
endif()

