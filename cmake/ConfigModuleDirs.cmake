#Function which (a) sets the module output directory for a target, and (b) includes other module directories
#preserves the include directories already set on the target, adding them at the end of the list.

# It removes all previous include directories pointing to Amber's modules folder,
# so that if you use copy_target() you can set new module paths without leaving the old ones lingering
# at the end of the include path like a bad smell.

function(config_module_dirs TARGETNAME TARGET_MODULE_DIR) #3rd optional argument: extra module include directories

	#add all of the passed module directories
	set(INCLUDE_DIRS ${TARGET_MODULE_DIR} ${ARGN})
		
	# get old include directories
	get_property(PREV_INC_DIRS TARGET ${TARGETNAME} PROPERTY INCLUDE_DIRECTORIES)
	get_property(PREV_INT_INC_DIRS TARGET ${TARGETNAME} PROPERTY INCLUDE_DIRECTORIES)
	
	# filter directories in the amber modules directory
	set(LEFTOVER_INC_DIRS "")
	set(LEFTOVER_INT_INC_DIRS "")
	
	foreach(DIR ${PREV_INC_DIRS})
		if(NOT "${DIR}" MATCHES "amber-modules/")
			list(APPEND LEFTOVER_INC_DIRS ${DIR})
		endif()
	endforeach()
	
	foreach(DIR ${PREV_INT_INC_DIRS})
		if(NOT "${DIR}" MATCHES "amber-modules/")
			list(APPEND LEFTOVER_INT_INC_DIRS ${DIR})
		endif()
	endforeach()
	# prepend module dir to include dirs
	set_property(TARGET ${TARGETNAME} PROPERTY Fortran_MODULE_DIRECTORY ${TARGET_MODULE_DIR})
	set_property(TARGET ${TARGETNAME} PROPERTY INCLUDE_DIRECTORIES ${INCLUDE_DIRS} ${LEFTOVER_INC_DIRS})
	set_property(TARGET ${TARGETNAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${INCLUDE_DIRS} ${LEFTOVER_INT_INC_DIRS})
	
endfunction()
	

