# Script run at install-time to:
# * locate dependencies of executables and libraries
# * copy them to the install directory
# * fix the install_name of the depender to point to the dependency on the RPATH
# * add a new RPATH entry on the dependency for its new location and remove its old install name
# * repeat above for dependencies of the dependency, and all other dependencies

# arguments:
# PACKAGE_PREFIX -- root of a UNIX-structure package to operate on
# CMAKE_SHARED_LIBRARY_SUFFIX -- pass this variable in from your CMake script
# CMAKE_EXECUTABLE_SUFFIX -- pass this variable in from your CMake script
# EXTRA_RPATH_SEARCH_DIRS -- list of extra directories to search in when trying to resolve @rpath references 
# PREFIX_RELATIVE_PYTHONPATH -- pass this variable in from your CMake script to indicate 
#      where relative to PACKAGE_PREFIX your python packages are located.  Empty string means
#      no python packages

# notes:
# * assumes that ${PACKAGE_PREFIX}/lib is, and should be, the rpath for all internal and external libraries
# * does not handle @executable_path since Amber doesn't use it; only handles @rpath and @loader_path

# This script was inspired by Hai Nguyen's similar script at https://github.com/Amber-MD/ambertools-binary-build/blob/master/conda_tools/update_gfortran_libs_osx.py

include(GetPrerequisites)
include(${CMAKE_CURRENT_LIST_DIR}/../Shorthand.cmake)

# Returns true iff the given dependency library should be ignored and not copied to the prefix
function(should_ignore_dep_library LIB_PATH OUTPUT_VARIABLE)
	if("${LIB_PATH}" MATCHES "\\.framework")
		set(${OUTPUT_VARIABLE} 1 PARENT_SCOPE)
	elseif("${LIB_PATH}" MATCHES "libSystem")
		set(${OUTPUT_VARIABLE} 1 PARENT_SCOPE)
	else()
		set(${OUTPUT_VARIABLE} 0 PARENT_SCOPE)
	endif()
endfunction(should_ignore_dep_library)

# Makes sure that the library named by LIB_PATH has the given RPATH location
function(add_rpath LIB_PATH RPATH)

	message(STATUS ">>>> Adding RPATH of \"${RPATH}\" to ${LIB_PATH}")

	execute_process(COMMAND install_name_tool
		-add_rpath ${RPATH} ${LIB_PATH}
		ERROR_VARIABLE INT_ERROR_OUTPUT
		RESULT_VARIABLE INT_RESULT_CODE)
	
	# uhhh, I really hope the user has their language set to English...	
	if("${INT_ERROR_OUTPUT}" MATCHES "would duplicate path")
		# do nothing, it already exists which is OK
	elseif(NOT ${INT_RESULT_CODE} EQUAL 0)
		message("!! Failed to execute install_name_tool! Error message was: ${INT_ERROR_OUTPUT}")
	endif()
	
endfunction(add_rpath)

# Changes the install_name that the given library refers to a dependency by.
# Does nothing if OLD_INSTNAME is not valid.
function(change_dependency_instname LIB_PATH OLD_INSTNAME NEW_INSTNAME)

	message(STATUS ">>>> Changing dependency reference \"${OLD_INSTNAME}\" in ${LIB_PATH} to \"${NEW_INSTNAME}\"")

	execute_process(COMMAND install_name_tool
		-change ${OLD_INSTNAME} ${NEW_INSTNAME} ${LIB_PATH}
		ERROR_VARIABLE INT_ERROR_OUTPUT
		RESULT_VARIABLE INT_RESULT_CODE)
	
	if(NOT ${INT_RESULT_CODE} EQUAL 0)
		message("!! Failed to execute install_name_tool! Error message was: ${INT_ERROR_OUTPUT}")
	endif()
	
endfunction(change_dependency_instname)

# Sets the install name (the name that other libraries save at link time, and use at runtime to find the library) of the given library to INSTALL_NAME
function(set_install_name LIB_PATH INSTALL_NAME)

	message(STATUS ">> Setting install name of ${LIB_PATH} to \"${INSTALL_NAME}\"")

	execute_process(COMMAND install_name_tool
		-id ${INSTALL_NAME} ${LIB_PATH}
		ERROR_VARIABLE INT_ERROR_OUTPUT
		RESULT_VARIABLE INT_RESULT_CODE)
	
	if(NOT ${INT_RESULT_CODE} EQUAL 0)
		message(STATUS "!! Failed to execute install_name_tool! Error message was: ${INT_ERROR_OUTPUT}")
	endif()
	
endfunction(set_install_name)

# uses the file command to determine if a file is an executable or a shared library
function(is_executable_or_library OUTPUT_VARIABLE FILE_PATH)
	
	execute_process(COMMAND file ${FILE_PATH}
		ERROR_VARIABLE FILE_CMD_ERROR_OUTPUT
		OUTPUT_VARIABLE FILE_CMD_OUTPUT
		RESULT_VARIABLE FILE_CMD_RESULT_CODE)
	
	if(NOT ${FILE_CMD_RESULT_CODE} EQUAL 0)
		message(STATUS "!! Failed to execute file! Error message was: ${FILE_CMD_ERROR_OUTPUT}")
		set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
		return()
	endif()

	if("${FILE_PATH}" MATCHES "bin/amber\\.")
		#special case for "amber.*" entries, which are symlinks to executables outside the Amber install dir
		set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
	elseif("${FILE_PATH}" MATCHES "\\.dSYM")
		# ignore debugging symbol libraries
		set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)	
	elseif("${FILE_CMD_OUTPUT}" MATCHES "Mach-O universal binary" OR "${FILE_CMD_OUTPUT}" MATCHES "Mach-O .+ executable")
		#executables
		set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
		
	elseif("${FILE_CMD_OUTPUT}" MATCHES "Mach-O .+ dynamically linked shared library" OR "${FILE_CMD_OUTPUT}" MATCHES "Mach-O .+ bundle")
		#shared libraries
		set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
	
	else()
		#everything else
		set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
		
	endif()

endfunction(is_executable_or_library)

message("Bundling OSX dependencies for package rooted at: ${PACKAGE_PREFIX}")

file(GLOB PACKAGE_LIBRARIES "${PACKAGE_PREFIX}/lib/*${CMAKE_SHARED_LIBRARY_SUFFIX}")
file(GLOB PACKAGE_EXECUTABLES "${PACKAGE_PREFIX}/bin/*")

if(NOT "${PREFIX_RELATIVE_PYTHONPATH}" STREQUAL "")
	# note: on OS X, python extension modules use ".so", not ".dylib"
	file(GLOB_RECURSE PYTHON_EXTENSION_MODULES "${PACKAGE_PREFIX}${PREFIX_RELATIVE_PYTHONPATH}/*.so")
else()
	set(PYTHON_EXTENSION_MODULES "")
endif()

# items are taken from, and added to, this stack.
# All files in this list are already in the installation prefix, and already have correct RPATHs
set(ITEMS_TO_PROCESS ${PACKAGE_LIBRARIES} ${PACKAGE_EXECUTABLES} ${PYTHON_EXTENSION_MODULES})

# lists of completed items (can skip if we see a dependency on these)
# This always contains the path inside the prefix
set(PROCESSED_ITEMS_BY_NEW_PATH "")

# List of external libraries which have already been copied to the prefix (by their external paths)
set(COPIED_EXTERNAL_DEPENDENCIES "")

# List that matches each index in the above list with the new path of the library
set(COPIED_EXTERNAL_DEPS_NEW_PATHS "")

if(NOT DEFINED EXTRA_RPATH_SEARCH_DIRS)
	set(EXTRA_RPATH_SEARCH_DIRS "")
endif()

# always use the prefix lib folder as the first RPATH search dir
set(RPATH_SEARCH_DIRS "${PACKAGE_PREFIX}/lib" ${EXTRA_RPATH_SEARCH_DIRS})

while(1)

	list(LENGTH ITEMS_TO_PROCESS NUM_ITEMS_LEFT)
		
	if(${NUM_ITEMS_LEFT} EQUAL 0)
		break()
	endif()
	
	list(GET ITEMS_TO_PROCESS 0 CURRENT_ITEM)
	
	message(STATUS "Considering ${CURRENT_ITEM}")
	
	is_executable_or_library(IS_EXEC_OR_LIB "${CURRENT_ITEM}")
		
	if(IS_EXEC_OR_LIB)
	
		set(CURRENT_ITEM_PREREQUISITES "")
		get_prerequisites(${CURRENT_ITEM} CURRENT_ITEM_PREREQUISITES 0 0 "" ${PACKAGE_PREFIX}/lib ${PACKAGE_PREFIX}/lib)
		
		foreach(PREREQUISITE_LIB_REFERENCE ${CURRENT_ITEM_PREREQUISITES})
			
	
			should_ignore_dep_library(${PREREQUISITE_LIB_REFERENCE} SHOULD_IGNORE_PREREQUISITE)
			
			if(SHOULD_IGNORE_PREREQUISITE)
				message(STATUS ">> Ignoring dependency: ${PREREQUISITE_LIB_REFERENCE}")
			else()
				
				# resolve RPATH references
				if("${PREREQUISITE_LIB_REFERENCE}" MATCHES "^@rpath")
					
					string(REPLACE "@rpath" "" RPATH_SUFFIX_PATH "${PREREQUISITE_LIB_REFERENCE}")
					
					# find the first folder in our RPATH search dirs that contains the library
					set(PREREQUISITE_LIB "")
					foreach(SEARCH_DIR ${RPATH_SEARCH_DIRS})
						#message("Checking ${SEARCH_DIR}${RPATH_SUFFIX_PATH}")
						if(EXISTS "${SEARCH_DIR}${RPATH_SUFFIX_PATH}")
							set(PREREQUISITE_LIB ${SEARCH_DIR}${RPATH_SUFFIX_PATH})
						endif()
					endforeach()
				else()
					set(PREREQUISITE_LIB ${PREREQUISITE_LIB_REFERENCE})
				endif()
			
				
				if(NOT EXISTS "${PREREQUISITE_LIB}")
					message("!! Unable to resolve library dependency ${PREREQUISITE_LIB_REFERENCE} -- skipping")
				else()
					
					# check if we already know about this library, and copy it here if we don't
					list(FIND COPIED_EXTERNAL_DEPENDENCIES "${PREREQUISITE_LIB}" INDEX_IN_COPIED_DEPS)
					list(FIND PACKAGE_LIBRARIES "${PREREQUISITE_LIB}" INDEX_IN_PACKAGE_LIBRARIES)
					
					if(NOT INDEX_IN_COPIED_DEPS EQUAL -1)
					
						message(STATUS ">> Already copied dependency: ${PREREQUISITE_LIB}")
						list(GET COPIED_EXTERNAL_DEPS_NEW_PATHS ${INDEX_IN_COPIED_DEPS} PREREQ_LIB_REALPATH)
						
					elseif(NOT INDEX_IN_PACKAGE_LIBRARIES EQUAL -1)
					
						message(STATUS ">> Dependency is internal: ${PREREQUISITE_LIB}")
						set(PREREQ_LIB_REALPATH ${PREREQUISITE_LIB})
						
					else()
						# previously unseen library -- copy to the prefix and queue for processing
						message(STATUS ">> Copy library dependency: ${PREREQUISITE_LIB}")
						
						# resolve symlinks
						get_filename_component(PREREQ_LIB_REALPATH ${PREREQUISITE_LIB} REALPATH)
						file(COPY "${PREREQ_LIB_REALPATH}" DESTINATION ${PACKAGE_PREFIX}/lib FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ WORLD_READ)
						
						# find new filename
						get_filename_component(PREREQ_LIB_FILENAME "${PREREQ_LIB_REALPATH}" NAME)
						set(NEW_PREREQ_PATH "${PACKAGE_PREFIX}/lib/${PREREQ_LIB_FILENAME}")
						
						# add correct RPATH
						add_rpath(${NEW_PREREQ_PATH} "@loader_path/../lib")
						
						list(APPEND COPIED_EXTERNAL_DEPENDENCIES ${PREREQUISITE_LIB})
						list(APPEND COPIED_EXTERNAL_DEPS_NEW_PATHS ${NEW_PREREQ_PATH})
						list(APPEND ITEMS_TO_PROCESS ${NEW_PREREQ_PATH})
					endif()
					
					# now, update how CURRENT_ITEM refers to this prerequisite
					get_filename_component(PREREQUISITE_FILENAME "${PREREQ_LIB_REALPATH}" NAME)
					if(NOT "${PREREQUISITE_LIB_REFERENCE}" STREQUAL "@rpath/${PREREQUISITE_FILENAME}")
						change_dependency_instname(${CURRENT_ITEM} ${PREREQUISITE_LIB_REFERENCE} "@rpath/${PREREQUISITE_FILENAME}")
					endif()
				endif()
			endif()
		endforeach()
	
		if("${CURRENT_ITEM}" MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
		
			# if it's a library, set its install name to refer to it on the RPATH (so anything can link to it as long as it uses the $AMBERHOME/lib RPATH)
			get_filename_component(CURRENT_ITEM_FILENAME "${CURRENT_ITEM}" NAME)
			set_install_name(${CURRENT_ITEM} "@rpath/${CURRENT_ITEM_FILENAME}")
			
		endif()
	
		list(APPEND PROCESSED_ITEMS_BY_NEW_PATH ${CURRENT_ITEM})
	
	else(IS_EXEC_OR_LIB)
		message(STATUS ">> Not an executable or shared library, skipping: ${CURRENT_ITEM}")
			
	endif()

	list(REMOVE_AT ITEMS_TO_PROCESS 0)
	
endwhile()

message(STATUS "Dependency bundling done!")