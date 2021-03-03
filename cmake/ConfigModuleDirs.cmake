#Function which (a) sets the module output directory for a target, and (b) includes other module directories.
#Preserves the include directories already set on the target, adding them at the end of the list.

# It removes all previous include directories pointing to Amber's modules folder,
# so that if you use copy_target() you can set new module paths without leaving the old ones lingering
# at the end of the include path like a bad smell.

set(MODULE_DIR_NAME amber-modules)

function(config_module_dirs TARGETNAME TARGET_MODULE_DIR) #3rd optional argument: extra module include directories

	if(IS_ABSOLUTE "${TARGET_MODULE_DIR}")
		# legacy full path, pass through unmodified
		set(INCLUDE_DIRS ${TARGET_MODULE_DIR})
		set(MODULE_OUTPUT_DIR ${TARGET_MODULE_DIR})
	else()
		# add both the build dir path and the installed path
		set(INCLUDE_DIRS $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/${MODULE_DIR_NAME}/${TARGET_MODULE_DIR}> $<INSTALL_INTERFACE:include/${MODULE_DIR_NAME}/${TARGET_MODULE_DIR}>)
		set(MODULE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${MODULE_DIR_NAME}/${TARGET_MODULE_DIR})
	endif()

	# convert relative module paths being included to absolute
	foreach(MOD_DIR ${ARGN})
		if(IS_ABSOLUTE "${MOD_DIR}")
			# pass through unmodified
			list(APPEND INCLUDE_DIRS $<BUILD_INTERFACE:${MOD_DIR}>)
		else()
			# convert to absolute path for build
			list(APPEND INCLUDE_DIRS $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/${MODULE_DIR_NAME}/${MOD_DIR}>)
		endif()
	endforeach()

	# get old include directories
	get_property(PREV_INC_DIRS TARGET ${TARGETNAME} PROPERTY INCLUDE_DIRECTORIES)
	get_property(PREV_INT_INC_DIRS TARGET ${TARGETNAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
	
	# remove any other directories in the amber modules directory
	set(LEFTOVER_INC_DIRS "")
	set(LEFTOVER_INT_INC_DIRS "")
	
	foreach(DIR ${PREV_INC_DIRS})
		if(NOT "${DIR}" MATCHES "${MODULE_DIR_NAME}")
			list(APPEND LEFTOVER_INC_DIRS ${DIR})
		endif()
	endforeach()
	
	foreach(DIR ${PREV_INT_INC_DIRS})
		if(NOT "${DIR}" MATCHES "${MODULE_DIR_NAME}")
			list(APPEND LEFTOVER_INT_INC_DIRS ${DIR})
		endif()
	endforeach()

	# combine the new module dirs with the rest of the include dirs
	set_property(TARGET ${TARGETNAME} PROPERTY Fortran_MODULE_DIRECTORY ${MODULE_OUTPUT_DIR})
	set_property(TARGET ${TARGETNAME} PROPERTY INCLUDE_DIRECTORIES ${INCLUDE_DIRS} ${LEFTOVER_INC_DIRS})
	set_property(TARGET ${TARGETNAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${INCLUDE_DIRS} ${LEFTOVER_INT_INC_DIRS})
	
endfunction()